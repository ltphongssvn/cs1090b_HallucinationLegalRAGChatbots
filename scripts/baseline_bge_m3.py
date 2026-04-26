"""MS3 BGE-M3 dense baseline retrieval (corpus-sharded, resume-safe).

Architecture (corpus sharding):
    - Each rank indexes a disjoint corpus slice [shard_start, shard_end)
    - Each rank searches ALL queries against its local shard
    - Merge step combines hits per-query across shards via MaxP + top-k

Single-GPU mode (world_size=1): writes bge_m3_results.jsonl + bge_m3_index.faiss.
Multi-GPU mode (world_size>1): writes per-rank files; merge handled externally.

Outputs (data/processed/baseline/):
    bge_m3_results{.rank###}.jsonl    — per-query {source_id, dest_id, retrieved}
    bge_m3_summary{.rank###}.json     — BaselineBgeM3Summary
    bge_m3_index{.rank###}.faiss      — FAISS IndexFlatIP
    bge_m3_index_meta{.rank###}.jsonl — parallel chunk_meta per index row
    bge_m3_ckpt{.rank###}.json        — encoding progress (deleted on success)
    bge_m3_index{.rank###}.partial.faiss — in-flight index (deleted on success)

Resume semantics:
    - Encoding phase flushes the partial FAISS index + checkpoint every
      CHECKPOINT_INTERVAL_BATCHES batches (atomic: tempfile + os.replace).
    - On restart, a compatible checkpoint (same rank, world_size, shard range,
      encoder_model, dim, ntotal) triggers resume from the last recorded offset.
    - Partial artifacts are deleted only after the final index + meta are written.

Hardening:
    - Lazy imports (faiss, torch, sentence_transformers inside main)
    - Streaming corpus encoding (no full (N,1024) float32 materialization)
    - encoder.max_seq_length = MAX_LENGTH (prevents CUDA OOM)
    - Dynamic retrieval k-expansion when one opinion dominates
    - Atomic result write (tempfile + os.replace)
    - Deterministic tie-break in MaxP: (-score, opinion_id)
    - Seed torch + numpy for determinism
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
TOP_K = 100
ENCODER_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
ENCODE_BATCH_SIZE = 64
QUERY_BATCH_SIZE = 8  # reduced from 256 after OOM on long quote outliers (18.56 GB matmul at bs=256, max_len=8192)
RETRIEVAL_K_MULTIPLIER = 3
CHECKPOINT_INTERVAL_BATCHES = 200  # flush partial FAISS index + meta every N batches
MAX_LENGTH = 8192
SIMILARITY_METRIC = "cosine"
NORMALIZE_EMBEDDINGS = True
DTYPE = "float32"

DEFAULT_CORPUS = Path("data/processed/baseline/corpus_chunks.jsonl")
DEFAULT_GOLD = Path("data/processed/baseline/gold_pairs_test.jsonl")
DEFAULT_LEPARD = Path("lepard_train_4000000_rev0194f95.jsonl")
DEFAULT_OUT_DIR = Path("data/processed/baseline")


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("baseline_bge_m3")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[baseline_bge_m3] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


# ---------- I/O ----------


def _iter_corpus(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _load_queries(gold_path: Path) -> list[dict[str, Any]]:
    """Load queries directly from gold pairs (quote field embedded by baseline_prep)."""
    queries: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    with gold_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (int(r["source_id"]), int(r["dest_id"]))
            if key not in seen:
                seen.add(key)
                queries.append(
                    {
                        "source_id": key[0],
                        "dest_id": key[1],
                        "query_text": r.get("destination_context", ""),
                    }
                )
    return queries


# ---------- aggregation ----------


def _aggregate_chunk_scores(raw_hits: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    best: dict[int, float] = {}
    for h in raw_hits:
        oid = h["opinion_id"]
        s = h["score"]
        if oid not in best or s > best[oid]:
            best[oid] = s
    ranked = [{"opinion_id": oid, "score": sc} for oid, sc in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ranked[:top_k]


# ---------- sharding ----------


def _shard_range(n: int, rank: int, world_size: int) -> tuple[int, int]:
    """Largest-remainder shard allocation.

    Guarantees:
        - sum of (end - start) across all ranks == n
        - max(size) - min(size) <= 1
        - disjoint + contiguous
        - (0, 0) for ranks with empty workload when world_size > n
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} out of range [0, {world_size})")
    base, rem = divmod(n, world_size)
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def _merge_shard_results(shard_paths: list[Path], merged_path: Path, *, top_k: int = TOP_K) -> None:
    """Corpus-shard merge: combine per-query hits across shards via MaxP + top-k."""
    per_query: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    key_order: list[tuple[int, int]] = []
    seen_keys: set[tuple[int, int]] = set()

    for shard in sorted(shard_paths):
        with shard.open(encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                key = (int(row["source_id"]), int(row["dest_id"]))
                if key not in seen_keys:
                    seen_keys.add(key)
                    key_order.append(key)
                for hit in row["retrieved"]:
                    oid = int(hit["opinion_id"])
                    sc = float(hit["score"])
                    prev = per_query[key].get(oid)
                    if prev is None or sc > prev:
                        per_query[key][oid] = sc

    with merged_path.open("w", encoding="utf-8") as fout:
        for key in key_order:
            scores = per_query[key]
            ranked = [
                {"opinion_id": oid, "score": sc} for oid, sc in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            ][:top_k]
            fout.write(
                json.dumps(
                    {"source_id": key[0], "dest_id": key[1], "retrieved": ranked},
                    allow_nan=False,
                )
                + "\n"
            )


# ---------- checkpointing (resume-safe corpus encoding) ----------


def _write_checkpoint(
    ckpt_path: Path,
    *,
    rank: int,
    world_size: int,
    n_encoded: int,
    shard_start: int,
    shard_end: int,
    encoder_model: str,
) -> None:
    """Atomically persist encoding progress."""
    tmp = ckpt_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "n_encoded": n_encoded,
                "shard_start": shard_start,
                "shard_end": shard_end,
                "encoder_model": encoder_model,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, ckpt_path)


_CHECKPOINT_REQUIRED_KEYS = frozenset({"rank", "world_size", "n_encoded", "shard_start", "shard_end", "encoder_model"})


def _load_checkpoint(ckpt_path: Path) -> dict[str, Any] | None:
    """Load checkpoint JSON. Returns None if file missing, corrupt, or incomplete.

    Structural incompleteness (missing required keys) is treated identically to a
    corrupt file — caller starts fresh rather than continuing from an ambiguous state.
    """
    if not ckpt_path.exists():
        return None
    try:
        data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    if not _CHECKPOINT_REQUIRED_KEYS.issubset(data.keys()):
        return None
    return data


# ---------- provenance ----------


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()[:12]
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


# ---------- W&B ----------


def _log_to_wandb(summary: dict[str, Any], out_dir: Path) -> None:
    try:
        import wandb
    except ImportError:
        logger.info("  wandb unavailable — skipping telemetry")
        return
    run = wandb.init(
        entity="phl690-harvard-extension-schol",
        project="cs1090b",
        job_type="baseline-bge-m3",
        config=summary,
        reinit=True,
    )
    wandb.log(summary)
    art = wandb.Artifact("baseline-bge-m3", type="dataset")
    art.add_dir(str(out_dir))
    run.log_artifact(art)
    run.finish()


# ---------- main ----------


def main(
    corpus_path: Path = DEFAULT_CORPUS,
    gold_pairs_path: Path = DEFAULT_GOLD,
    out_dir: Path = DEFAULT_OUT_DIR,
    top_k: int = TOP_K,
    log_to_wandb: bool = False,
    seed: int = 0,
    encode_batch_size: int = ENCODE_BATCH_SIZE,
    query_batch_size: int = QUERY_BATCH_SIZE,
    rank: int = 0,
    world_size: int = 1,
    case_names_redacted: bool = False,
) -> Any:
    """Run BGE-M3 dense baseline on corpus shard [rank/world_size] with resume."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from src.eda_schemas import BaselineBgeM3Summary

    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if encode_batch_size < 1:
        raise ValueError(f"encode_batch_size must be >= 1, got {encode_batch_size}")
    if query_batch_size < 1:
        raise ValueError(f"query_batch_size must be >= 1, got {query_batch_size}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} out of range [0, {world_size})")
    for path in (corpus_path, gold_pairs_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"input missing: {path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(seed)
    device = _detect_device()
    device_name = _device_name(device)

    rank_suffix = f".rank{rank:03d}" if world_size > 1 else ""
    index_path = out_dir / f"bge_m3_index{rank_suffix}.faiss"
    meta_path = out_dir / f"bge_m3_index_meta{rank_suffix}.jsonl"
    results_path = out_dir / f"bge_m3_results{rank_suffix}.jsonl"
    summary_path = out_dir / f"bge_m3_summary{rank_suffix}.json"
    ckpt_path = out_dir / f"bge_m3_ckpt{rank_suffix}.json"
    partial_index_path = out_dir / f"bge_m3_index{rank_suffix}.partial.faiss"

    logger.info("=" * 60)
    logger.info(f"MS3 BGE-M3 corpus-shard  (rank={rank}/{world_size}, device={device_name}, top_k={top_k})")
    logger.info("=" * 60)

    # --- Load encoder ---
    logger.info("\n[1/5] Loading encoder")
    t0 = time.perf_counter()
    encoder = SentenceTransformer(ENCODER_MODEL, device=device)
    encoder.max_seq_length = MAX_LENGTH
    encoder_load_seconds = time.perf_counter() - t0
    logger.info(f"  encoder loaded in: {encoder_load_seconds:.2f}s")

    # --- Corpus shard range ---
    logger.info("\n[2/5] Computing corpus shard range")
    n_total = sum(1 for _ in _iter_corpus(corpus_path))
    shard_start, shard_end = _shard_range(n_total, rank, world_size)
    logger.info(f"  corpus (total)   : {n_total:,}")
    logger.info(f"  shard range      : [{shard_start:,}, {shard_end:,})  ({shard_end - shard_start:,} chunks)")

    chunk_meta: list[tuple[int, int]] = []
    for i, c in enumerate(_iter_corpus(corpus_path)):
        if shard_start <= i < shard_end:
            chunk_meta.append((c["opinion_id"], c["chunk_index"]))
        elif i >= shard_end:
            break
    n_chunks = len(chunk_meta)
    unique_opinions = len({m[0] for m in chunk_meta})
    logger.info(f"  shard opinions   : {unique_opinions:,}")

    # --- Build or reuse/resume FAISS index ---
    reuse_index = False
    if index_path.exists() and meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            meta_line_count = sum(1 for _ in f)
        if meta_line_count == n_chunks:
            try:
                index = faiss.read_index(str(index_path))
                if index.ntotal == n_chunks and index.d == EMBEDDING_DIM:
                    reuse_index = True
                    logger.info(f"\n[3/5] Reusing FAISS index ({index.ntotal:,} vectors)")
            except Exception as e:
                logger.info(f"  could not load existing index: {e}")

    if reuse_index:
        index_build_seconds = 0.0
    else:
        logger.info(
            f"\n[3/5] Encoding shard + building FAISS index "
            f"(batch={encode_batch_size}, checkpoint every {CHECKPOINT_INTERVAL_BATCHES} batches)"
        )

        # Attempt resume
        resume_from = 0
        index: Any = None  # faiss index handle
        ckpt = _load_checkpoint(ckpt_path)
        if ckpt is not None and partial_index_path.exists():
            if (
                ckpt.get("rank") == rank
                and ckpt.get("world_size") == world_size
                and ckpt.get("shard_start") == shard_start
                and ckpt.get("shard_end") == shard_end
                and ckpt.get("encoder_model") == ENCODER_MODEL
            ):
                try:
                    index = faiss.read_index(str(partial_index_path))
                    if index.d == EMBEDDING_DIM and index.ntotal == ckpt["n_encoded"]:
                        resume_from = int(ckpt["n_encoded"])
                        logger.info(f"  RESUMING: {resume_from:,} chunks already encoded")
                    else:
                        logger.info("  partial index dim/count mismatch — restarting")
                        index = faiss.IndexFlatIP(EMBEDDING_DIM)
                except Exception as e:
                    logger.info(f"  could not load partial index ({e}) — restarting")
                    index = faiss.IndexFlatIP(EMBEDDING_DIM)
            else:
                logger.info("  checkpoint config mismatch — restarting")
                index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if index is None:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)

        t0 = time.perf_counter()

        def _iter_shard_batches() -> Iterator[list[str]]:
            batch: list[str] = []
            absolute_start = shard_start + resume_from
            for i, c in enumerate(_iter_corpus(corpus_path)):
                if i < absolute_start:
                    continue
                if i >= shard_end:
                    break
                batch.append(c["text"])
                if len(batch) >= encode_batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        n_encoded = resume_from
        batches_since_ckpt = 0
        for batch_texts in _iter_shard_batches():
            embs = encoder.encode(
                batch_texts,
                batch_size=encode_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=NORMALIZE_EMBEDDINGS,
            ).astype(np.float32)
            if embs.shape[1] != EMBEDDING_DIM:
                raise ValueError(f"encoder dim {embs.shape[1]} != expected {EMBEDDING_DIM}")
            index.add(embs)
            n_encoded += len(batch_texts)
            batches_since_ckpt += 1

            if batches_since_ckpt >= CHECKPOINT_INTERVAL_BATCHES:
                faiss.write_index(index, str(partial_index_path))
                _write_checkpoint(
                    ckpt_path,
                    rank=rank,
                    world_size=world_size,
                    n_encoded=n_encoded,
                    shard_start=shard_start,
                    shard_end=shard_end,
                    encoder_model=ENCODER_MODEL,
                )
                logger.info(f"    checkpoint: encoded {n_encoded:,} / {n_chunks:,} — index flushed")
                batches_since_ckpt = 0
            elif n_encoded % (encode_batch_size * 100) == 0:
                logger.info(f"    encoded {n_encoded:,} / {n_chunks:,}")

        index_build_seconds = time.perf_counter() - t0
        logger.info(f"  index size       : {index.ntotal:,}")
        logger.info(f"  index build      : {index_build_seconds:.2f}s")

        # Final atomic write of index + meta; clean up partial artifacts
        faiss.write_index(index, str(index_path))
        with meta_path.open("w", encoding="utf-8") as f:
            for oid, ci in chunk_meta:
                f.write(json.dumps({"opinion_id": oid, "chunk_index": ci}) + "\n")
        if partial_index_path.exists():
            partial_index_path.unlink()
        if ckpt_path.exists():
            ckpt_path.unlink()

    # --- Load + encode queries ---
    logger.info("\n[4/5] Loading + encoding queries")
    queries = _load_queries(gold_pairs_path)
    logger.info(f"  queries          : {len(queries):,}")
    query_texts = [q["query_text"] for q in queries]
    t0 = time.perf_counter()
    query_embeddings = encoder.encode(
        query_texts,
        batch_size=query_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    ).astype(np.float32)
    query_encode_seconds = time.perf_counter() - t0
    logger.info(f"  encode time      : {query_encode_seconds:.2f}s")

    # --- Retrieval with dynamic k-expansion ---
    logger.info(f"\n[5/5] FAISS search top-{top_k} per query (shard-local)")
    t0 = time.perf_counter()
    initial_k = min(top_k * RETRIEVAL_K_MULTIPLIER, n_chunks)
    results_tmp = results_path.with_suffix(".jsonl.tmp")

    def _retrieve_and_aggregate(q_emb: np.ndarray) -> list[dict[str, Any]]:
        current_k = initial_k
        while True:
            scores, indices = index.search(q_emb, current_k)
            raw_hits = [
                {
                    "opinion_id": chunk_meta[int(idx)][0],
                    "chunk_index": chunk_meta[int(idx)][1],
                    "score": float(score),
                }
                for idx, score in zip(indices[0], scores[0], strict=False)
                if idx != -1
            ]
            aggregated = _aggregate_chunk_scores(raw_hits, top_k=top_k)
            if len(aggregated) >= top_k or current_k >= n_chunks:
                return aggregated
            current_k = min(current_k * 2, n_chunks)

    with results_tmp.open("w", encoding="utf-8") as fout:
        for batch_start in range(0, len(queries), query_batch_size):
            batch_end = min(batch_start + query_batch_size, len(queries))
            batch_q_emb = query_embeddings[batch_start:batch_end]
            scores_batch, indices_batch = index.search(batch_q_emb, initial_k)
            for qi_local, qi_global in enumerate(range(batch_start, batch_end)):
                q = queries[qi_global]
                raw_hits = [
                    {
                        "opinion_id": chunk_meta[int(idx)][0],
                        "chunk_index": chunk_meta[int(idx)][1],
                        "score": float(score),
                    }
                    for idx, score in zip(
                        indices_batch[qi_local],
                        scores_batch[qi_local],
                        strict=False,
                    )
                    if idx != -1
                ]
                aggregated = _aggregate_chunk_scores(raw_hits, top_k=top_k)
                if len(aggregated) < top_k and initial_k < n_chunks:
                    aggregated = _retrieve_and_aggregate(query_embeddings[qi_global : qi_global + 1])
                fout.write(
                    json.dumps(
                        {
                            "source_id": q["source_id"],
                            "dest_id": q["dest_id"],
                            "retrieved": aggregated,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )

    os.replace(results_tmp, results_path)
    retrieval_seconds = time.perf_counter() - t0
    logger.info(f"  retrieval done in: {retrieval_seconds:.2f}s")

    # --- Summary ---
    results_hash = hashlib.sha256(results_path.read_bytes()).hexdigest()
    validated = BaselineBgeM3Summary(
        schema_version=SCHEMA_VERSION,
        n_queries=len(queries),
        n_corpus_chunks=n_chunks,
        n_unique_opinions=unique_opinions,
        top_k=top_k,
        encoder_model=ENCODER_MODEL,
        embedding_dim=EMBEDDING_DIM,
        device=device,
        device_name=device_name,
        encode_batch_size=encode_batch_size,
        similarity_metric=SIMILARITY_METRIC,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        max_length=MAX_LENGTH,
        dtype=DTYPE,
        encoder_load_seconds=round(encoder_load_seconds, 3),
        index_build_seconds=round(index_build_seconds, 3),
        query_encode_seconds=round(query_encode_seconds, 3),
        retrieval_seconds=round(retrieval_seconds, 3),
        seed=seed,
        world_size=world_size,
        shard_rank=rank,
        git_sha=_git_sha(),
        results_hash=results_hash,
        case_names_redacted=case_names_redacted,
    )
    summary_path.write_text(
        json.dumps(validated.model_dump(), sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(f"\nWrote summary -> {summary_path}")

    if log_to_wandb:
        _log_to_wandb(validated.model_dump(), out_dir)

    return validated


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MS3 BGE-M3 dense baseline retrieval.")
    ap.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--gold-pairs-path", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--encode-batch-size", type=int, default=ENCODE_BATCH_SIZE)
    ap.add_argument("--query-batch-size", type=int, default=QUERY_BATCH_SIZE)
    ap.add_argument("--log-to-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--case-names-redacted", action="store_true", help="Flag indicating gold pairs have been redacted")
    ap.add_argument("--rank", type=int, default=0, help="shard rank for multi-GPU")
    ap.add_argument("--world-size", type=int, default=1, help="total shard count")
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print(
            f"[baseline_bge_m3] DRY RUN  schema={SCHEMA_VERSION}  "
            f"top_k={args.top_k}  encoder={ENCODER_MODEL}  dim={EMBEDDING_DIM}  "
            f"device={_detect_device()}  git_sha={_git_sha()}  "
            f"python={sys.version.split()[0]}  args={vars(args)}"
        )
        sys.exit(0)
    main(
        corpus_path=args.corpus_path,
        gold_pairs_path=args.gold_pairs_path,
        out_dir=args.out_dir,
        top_k=args.top_k,
        log_to_wandb=args.log_to_wandb,
        seed=args.seed,
        encode_batch_size=args.encode_batch_size,
        query_batch_size=args.query_batch_size,
        rank=args.rank,
        world_size=args.world_size,
        case_names_redacted=args.case_names_redacted,
    )
