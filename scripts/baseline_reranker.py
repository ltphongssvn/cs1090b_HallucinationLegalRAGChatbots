# scripts/baseline_reranker.py
"""MS4 cross-encoder reranker on top of fused (RRF) candidate set.

Architecture (query-shard, single rank per GPU)
-----------------------------------------------
Each rank handles a disjoint slice of queries [shard_start, shard_end). For
each query in its slice, the rank scores all top-K candidates from the input
results file using BAAI/bge-reranker-v2-m3, then re-orders them by reranker
score. No cross-rank coordination is needed at retrieval time — the merge
step is a simple ordered concatenation of per-rank output files.

Two scoring modes (selected via --score-mode):
  - concat (default): cluster text = first N chunks concatenated, single
                      (query, concat_text) pair scored. Legacy behavior.
  - maxp            : each chunk scored independently as a (query, chunk)
                      pair; cluster score = max over its chunk scores
                      (Dai & Callan 2019, BERT-MaxP). Industry-standard
                      long-document reranking method.

Why query sharding (not corpus sharding)
-----------------------------------------
Reranking is per-query work over a fixed candidate set (not corpus index
build). Splitting queries across GPUs trivially parallelizes without the
MaxP / cross-shard merge logic that BGE-M3 indexing required.

Verified-pipeline contract
--------------------------
Input results file: rrf_results.jsonl (or any retriever's top-K output) with
schema {source_id, dest_id, source_cluster_id, retrieved: [{cluster_id, score}]}.
Cluster text comes from corpus_chunks_cleaned.jsonl, keyed on cluster_id.

Outputs (data/processed/baseline/cleaned/)
------------------------------------------
reranker_results{.rank###}.jsonl  — per-query top-k re-ordered candidates
reranker_summary{.rank###}.json   — provenance + timings per rank
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_ENCODER_DIR = None  # None = use RERANKER_MODEL from HF hub; set to a path for fine-tuned
DEFAULT_TOP_K_INPUT = 100
DEFAULT_TOP_K_OUTPUT = 100
DEFAULT_MAX_LENGTH = 1024  # BAAI-recommended fine-tuning ceiling for bge-reranker-v2-m3 (model supports up to 8192)
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_CHUNKS_PER_CLUSTER = 2  # 2 chunks (~1200 words) fits 1024-token reranker budget after query + special tokens
DEFAULT_SCORE_MODE = "concat"

DEFAULT_CORPUS = Path("data/processed/baseline/corpus_chunks_cleaned.jsonl")
DEFAULT_GOLD = Path("data/processed/baseline/cleaned/gold_pairs_test.jsonl")
DEFAULT_INPUT_RESULTS = Path("data/processed/baseline/cleaned/rrf_results.jsonl")
DEFAULT_OUT_DIR = Path("data/processed/baseline/cleaned")


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("baseline_reranker")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[baseline_reranker] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


# ---------- I/O ----------


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


REQUIRED_GOLD_FIELDS = ("source_id", "dest_id", "source_cluster_id", "quote")


def _load_queries(gold_path: Path) -> list[dict[str, Any]]:
    """Load deduplicated queries from cleaned gold pairs.

    Dedup on (source_id, dest_id) — first occurrence wins. Mirrors the
    contract in baseline_bm25.py::_load_queries_verified and
    baseline_bge_m3.py::_load_queries_verified so all three retrieval +
    reranking stages see the same 20,877 query universe.
    """
    queries: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for line_no, r in enumerate(_iter_jsonl(Path(gold_path)), start=1):
        missing = [k for k in REQUIRED_GOLD_FIELDS if k not in r]
        if missing:
            raise ValueError(
                f"gold line {line_no} missing required fields: {missing}"
            )
        key = (int(r["source_id"]), int(r["dest_id"]))
        if key in seen:
            continue
        seen.add(key)
        queries.append({
            "source_id": key[0],
            "dest_id": key[1],
            "source_cluster_id": int(r["source_cluster_id"]),
            "query_text": r["quote"],
        })
    return queries


def _load_cluster_text_index(
    corpus_path: Path,
    *,
    max_chunks_per_cluster: int = DEFAULT_MAX_CHUNKS_PER_CLUSTER,
    cluster_filter: set[int] | None = None,
) -> dict[int, str]:
    """Build {cluster_id: concatenated_text} index from cleaned corpus.

    For each cluster_id, concatenates the first `max_chunks_per_cluster`
    chunks (in chunk_index order). Used by --score-mode=concat.
    """
    chunks_by_cluster: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for row in _iter_jsonl(Path(corpus_path)):
        cid = int(row["cluster_id"])
        if cluster_filter is not None and cid not in cluster_filter:
            continue
        if len(chunks_by_cluster[cid]) >= max_chunks_per_cluster:
            continue
        chunks_by_cluster[cid].append((int(row["chunk_index"]), str(row["text"])))

    index: dict[int, str] = {}
    for cid, chunks in chunks_by_cluster.items():
        chunks.sort(key=lambda t: t[0])
        index[cid] = " ".join(t for _, t in chunks)
    return index


def _load_cluster_chunks_index(
    corpus_path: Path,
    *,
    max_chunks_per_cluster: int = DEFAULT_MAX_CHUNKS_PER_CLUSTER,
    cluster_filter: set[int] | None = None,
) -> dict[int, list[str]]:
    """Build {cluster_id: [chunk_text, ...]} index for MaxP chunk-level reranking.

    Mirrors _load_cluster_text_index but preserves chunks as separate strings
    (instead of concatenating). Each chunk fits independently in the reranker
    max_length budget; the reranker scores each (query, chunk) pair and the
    cluster's MaxP score is max over its chunks (Dai & Callan 2019).
    """
    chunks_by_cluster: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for row in _iter_jsonl(Path(corpus_path)):
        cid = int(row["cluster_id"])
        if cluster_filter is not None and cid not in cluster_filter:
            continue
        if len(chunks_by_cluster[cid]) >= max_chunks_per_cluster:
            continue
        chunks_by_cluster[cid].append((int(row["chunk_index"]), str(row["text"])))

    index: dict[int, list[str]] = {}
    for cid, chunks in chunks_by_cluster.items():
        chunks.sort(key=lambda t: t[0])
        index[cid] = [t for _, t in chunks]
    return index


# ---------- sharding ----------


def _shard_range(n: int, rank: int, world_size: int) -> tuple[int, int]:
    """Largest-remainder shard allocation (mirrors baseline_bge_m3._shard_range)."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} out of range [0, {world_size})")
    base, rem = divmod(n, world_size)
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def _merge_shard_results(shard_paths: list[Path], merged_path: Path) -> None:
    """Concatenate per-rank reranker outputs in rank-order.

    No MaxP / cross-shard score merging needed — query-sharding means each
    query is handled by exactly one rank.
    """
    Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as fout:
        for shard in sorted(shard_paths):
            if not Path(shard).exists():
                continue
            with shard.open(encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)


# ---------- pure scoring helpers ----------


def _rerank_candidates_by_score(
    candidates: list[dict[str, Any]],
    scores: list[float],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Re-order candidates by descending score; deterministic tie-break by cluster_id ascending."""
    if len(candidates) != len(scores):
        raise ValueError(
            f"candidate / score length mismatch: {len(candidates)} vs {len(scores)}"
        )
    paired = list(zip(candidates, scores, strict=True))
    paired.sort(key=lambda cs: (-cs[1], int(cs[0]["cluster_id"])))
    return [
        {"cluster_id": int(c["cluster_id"]), "score": float(s)}
        for c, s in paired[:top_k]
    ]


def _maxp_aggregate(
    candidates: list[dict[str, Any]],
    chunk_scores_by_cluster: dict[int, list[float]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """MaxP aggregation: cluster score = max over its per-chunk scores.

    Industry-standard long-document reranking (Dai & Callan 2019, BERT-MaxP).
    Each chunk is scored independently against the query; the cluster's
    final relevance is the maximum chunk score.

    Clusters with no chunk scores (empty or missing) are dropped from output.
    Tie-break: ascending cluster_id (deterministic).
    """
    scored: list[tuple[int, float]] = []
    for c in candidates:
        cid = int(c["cluster_id"])
        chunk_scores = chunk_scores_by_cluster.get(cid)
        if not chunk_scores:
            continue
        scored.append((cid, max(chunk_scores)))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [
        {"cluster_id": cid, "score": float(score)}
        for cid, score in scored[:top_k]
    ]


# ---------- provenance ----------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _device_name(device: str) -> str:
    if device != "cuda":
        return device
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda-unknown"


def _seed_all(seed: int) -> None:
    import random as _r
    _r.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------- main ----------


def main(
    corpus_path: Path = DEFAULT_CORPUS,
    gold_path: Path = DEFAULT_GOLD,
    input_results_path: Path = DEFAULT_INPUT_RESULTS,
    out_dir: Path = DEFAULT_OUT_DIR,
    top_k_input: int = DEFAULT_TOP_K_INPUT,
    top_k_output: int = DEFAULT_TOP_K_OUTPUT,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_chunks_per_cluster: int = DEFAULT_MAX_CHUNKS_PER_CLUSTER,
    score_mode: str = DEFAULT_SCORE_MODE,
    encoder_dir: Path | None = DEFAULT_ENCODER_DIR,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Rerank candidates from input_results_path on this rank's query slice."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if score_mode not in ("concat", "maxp"):
        raise ValueError(f"score_mode must be 'concat' or 'maxp', got {score_mode!r}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} out of range [0, {world_size})")
    for p in (corpus_path, gold_path, input_results_path):
        if not Path(p).is_file():
            raise FileNotFoundError(f"input missing: {p}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(seed)

    device = _detect_device()
    device_name = _device_name(device)
    rank_suffix = f".rank{rank:03d}" if world_size > 1 else ""
    results_path = out_dir / f"reranker_results{rank_suffix}.jsonl"
    summary_path = out_dir / f"reranker_summary{rank_suffix}.json"

    logger.info("=" * 60)
    logger.info(
        f"MS4 cross-encoder reranking  "
        f"(rank={rank}/{world_size}, device={device_name}, "
        f"score_mode={score_mode}, top_k_in={top_k_input} -> top_k_out={top_k_output})"
    )
    logger.info("=" * 60)

    # --- Load inputs aligned by row order (RRF + gold both keyed by (source_id, dest_id)) ---
    logger.info("\n[1/4] Loading queries + input results")
    queries = _load_queries(gold_path)
    input_rows = list(_iter_jsonl(input_results_path))
    if len(queries) != len(input_rows):
        raise ValueError(
            f"queries ({len(queries):,}) and input_results ({len(input_rows):,}) "
            f"have different row counts — must be aligned"
        )
    for q, r in zip(queries[:1] + queries[-1:], input_rows[:1] + input_rows[-1:], strict=True):
        if (int(q["source_id"]), int(q["dest_id"])) != (int(r["source_id"]), int(r["dest_id"])):
            raise ValueError(
                f"row-order mismatch between queries and input_results "
                f"({q['source_id']},{q['dest_id']}) vs ({r['source_id']},{r['dest_id']})"
            )

    n_total = len(queries)
    shard_start, shard_end = _shard_range(n_total, rank, world_size)
    queries_slice = queries[shard_start:shard_end]
    input_slice = input_rows[shard_start:shard_end]
    logger.info(f"  total queries  : {n_total:,}")
    logger.info(f"  this shard     : [{shard_start:,}, {shard_end:,})  ({len(queries_slice):,} queries)")

    # --- Build cluster_id -> text index, filtered to candidates this rank needs ---
    logger.info(f"\n[2/4] Building cluster_id -> text index (mode={score_mode}, filtered)")
    needed_cluster_ids: set[int] = set()
    for r in input_slice:
        for hit in r["retrieved"][:top_k_input]:
            needed_cluster_ids.add(int(hit["cluster_id"]))
    logger.info(f"  unique candidate clusters this rank: {len(needed_cluster_ids):,}")
    t0 = time.perf_counter()
    if score_mode == "concat":
        cluster_text: dict[int, str] = _load_cluster_text_index(
            corpus_path,
            max_chunks_per_cluster=max_chunks_per_cluster,
            cluster_filter=needed_cluster_ids,
        )
        cluster_chunks: dict[int, list[str]] = {}  # unused
    else:  # maxp
        cluster_chunks = _load_cluster_chunks_index(
            corpus_path,
            max_chunks_per_cluster=max_chunks_per_cluster,
            cluster_filter=needed_cluster_ids,
        )
        cluster_text = {}
    text_index_seconds = time.perf_counter() - t0
    n_resolved = len(cluster_text) if score_mode == "concat" else len(cluster_chunks)
    n_missing = len(needed_cluster_ids) - n_resolved
    logger.info(f"  resolved {n_resolved:,} / {len(needed_cluster_ids):,} clusters in {text_index_seconds:.1f}s")
    if n_missing > 0:
        logger.info(f"  WARNING: {n_missing:,} candidate clusters not found in corpus — will skip those")

    # --- Load reranker model ---
    logger.info("\n[3/4] Loading reranker model")
    t0 = time.perf_counter()
    model_source = str(encoder_dir) if encoder_dir is not None else RERANKER_MODEL
    logger.info(f"  loading reranker from: {model_source}")
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source, torch_dtype=model_dtype,
    ).to(device).eval()
    encoder_load_seconds = time.perf_counter() - t0
    logger.info(f"  model loaded in {encoder_load_seconds:.1f}s ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    # --- Rerank ---
    logger.info(f"\n[4/4] Reranking {len(queries_slice):,} queries × up to {top_k_input} candidates each (mode={score_mode})")
    t0 = time.perf_counter()
    results_tmp = results_path.with_suffix(".jsonl.tmp")

    n_pairs_scored = 0
    log_interval = max(1, len(queries_slice) // 50)

    # --- Resume from existing .tmp if present ---
    already_done: set[tuple[int, int]] = set()
    if results_tmp.is_file():
        try:
            with results_tmp.open(encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    already_done.add((int(row["source_id"]), int(row["dest_id"])))
            logger.info(
                f"  RESUMING: {len(already_done):,} queries already in {results_tmp.name}, will skip"
            )
        except Exception as e:
            logger.info(f"  could not parse existing {results_tmp}: {e}; starting fresh")
            already_done = set()
            results_tmp.unlink(missing_ok=True)

    def _score_pairs(pairs: list[list[str]]) -> list[float]:
        """Score a list of [query, doc] pairs in mini-batches."""
        out: list[float] = []
        for bs_start in range(0, len(pairs), batch_size):
            bs_end = min(bs_start + batch_size, len(pairs))
            batch = pairs[bs_start:bs_end]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            ).to(device)
            logits = model(**inputs).logits.view(-1).float().cpu().tolist()
            out.extend(logits)
        return out

    with results_tmp.open("a", encoding="utf-8", buffering=1) as fout, torch.inference_mode():
        for qi, (q, ir) in enumerate(zip(queries_slice, input_slice, strict=True)):
            if (int(q["source_id"]), int(q["dest_id"])) in already_done:
                continue
            top_input = ir["retrieved"][:top_k_input]
            qtxt = q["query_text"]

            if score_mode == "concat":
                # Single (query, concat_text) pair per candidate
                candidates_with_text: list[dict[str, Any]] = []
                pair_texts: list[str] = []
                for hit in top_input:
                    cid = int(hit["cluster_id"])
                    txt = cluster_text.get(cid)
                    if txt is None:
                        continue
                    candidates_with_text.append({"cluster_id": cid, "score": hit["score"]})
                    pair_texts.append(txt)
                if not candidates_with_text:
                    fout.write(json.dumps({
                        "source_id": q["source_id"],
                        "dest_id": q["dest_id"],
                        "source_cluster_id": q["source_cluster_id"],
                        "retrieved": [],
                    }, allow_nan=False) + "\n")
                    continue
                pairs = [[qtxt, t] for t in pair_texts]
                scores = _score_pairs(pairs)
                n_pairs_scored += len(pairs)
                ranked = _rerank_candidates_by_score(
                    candidates_with_text, scores, top_k=top_k_output,
                )
            else:  # maxp
                # Score every (query, chunk) pair across all candidates at once
                pairs = []
                pair_to_cid: list[int] = []
                for hit in top_input:
                    cid = int(hit["cluster_id"])
                    chunks = cluster_chunks.get(cid)
                    if not chunks:
                        continue
                    for chunk in chunks:
                        pairs.append([qtxt, chunk])
                        pair_to_cid.append(cid)
                if not pairs:
                    fout.write(json.dumps({
                        "source_id": q["source_id"],
                        "dest_id": q["dest_id"],
                        "source_cluster_id": q["source_cluster_id"],
                        "retrieved": [],
                    }, allow_nan=False) + "\n")
                    continue
                scores = _score_pairs(pairs)
                n_pairs_scored += len(pairs)
                # Group scores by cluster_id
                chunk_scores_by_cluster: dict[int, list[float]] = defaultdict(list)
                for cid, s in zip(pair_to_cid, scores, strict=True):
                    chunk_scores_by_cluster[cid].append(s)
                ranked = _maxp_aggregate(
                    top_input, chunk_scores_by_cluster, top_k=top_k_output,
                )

            fout.write(json.dumps({
                "source_id": q["source_id"],
                "dest_id": q["dest_id"],
                "source_cluster_id": q["source_cluster_id"],
                "retrieved": ranked,
            }, allow_nan=False) + "\n")

            if (qi + 1) % log_interval == 0:
                elapsed = time.perf_counter() - t0
                logger.info(
                    f"    [{qi+1:,}/{len(queries_slice):,}] "
                    f"elapsed={elapsed:.0f}s  pairs_scored={n_pairs_scored:,}  "
                    f"throughput={n_pairs_scored/max(elapsed,1):.1f} pairs/sec"
                )

    os.replace(results_tmp, results_path)
    rerank_seconds = time.perf_counter() - t0
    logger.info(f"  rerank done in {rerank_seconds:.1f}s")

    # --- Summary ---
    results_hash = hashlib.sha256(results_path.read_bytes()).hexdigest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_queries_total": n_total,
        "n_queries_this_rank": len(queries_slice),
        "n_pairs_scored": n_pairs_scored,
        "top_k_input": top_k_input,
        "top_k_output": top_k_output,
        "max_length": max_length,
        "batch_size": batch_size,
        "max_chunks_per_cluster": max_chunks_per_cluster,
        "score_mode": score_mode,
        "reranker_model": model_source,
        "reranker_model_base": RERANKER_MODEL,
        "device": device,
        "device_name": device_name,
        "world_size": world_size,
        "shard_rank": rank,
        "shard_start": shard_start,
        "shard_end": shard_end,
        "n_clusters_needed": len(needed_cluster_ids),
        "n_clusters_resolved": n_resolved,
        "text_index_seconds": round(text_index_seconds, 3),
        "encoder_load_seconds": round(encoder_load_seconds, 3),
        "rerank_seconds": round(rerank_seconds, 3),
        "seed": seed,
        "git_sha": _git_sha(),
        "results_hash": results_hash,
    }
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(f"\nWrote summary -> {summary_path}")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MS4 cross-encoder reranker over fused candidates.")
    ap.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--gold-path", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--input-results-path", type=Path, default=DEFAULT_INPUT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--top-k-input", type=int, default=DEFAULT_TOP_K_INPUT)
    ap.add_argument("--top-k-output", type=int, default=DEFAULT_TOP_K_OUTPUT)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-chunks-per-cluster", type=int, default=DEFAULT_MAX_CHUNKS_PER_CLUSTER)
    ap.add_argument(
        "--score-mode",
        type=str,
        default=DEFAULT_SCORE_MODE,
        choices=("concat", "maxp"),
        help="concat: score (query, joined-chunks) once; maxp: score (query, each-chunk) and take max",
    )
    ap.add_argument("--encoder-dir", type=Path, default=DEFAULT_ENCODER_DIR,
                    help="Path to fine-tuned reranker (None = use RERANKER_MODEL from HF hub)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print(
            f"[baseline_reranker] DRY RUN  schema={SCHEMA_VERSION}  model={RERANKER_MODEL}  "
            f"score_mode={args.score_mode}  git_sha={_git_sha()}  args={vars(args)}"
        )
        sys.exit(0)
    main(
        corpus_path=args.corpus_path,
        gold_path=args.gold_path,
        input_results_path=args.input_results_path,
        out_dir=args.out_dir,
        top_k_input=args.top_k_input,
        top_k_output=args.top_k_output,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_chunks_per_cluster=args.max_chunks_per_cluster,
        score_mode=args.score_mode,
        encoder_dir=args.encoder_dir,
        rank=args.rank,
        world_size=args.world_size,
        seed=args.seed,
    )
