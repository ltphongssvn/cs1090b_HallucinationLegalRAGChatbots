# scripts/rag_generate.py
"""MS5 RAG generation — Qwen2.5-7B-Instruct with retrieved-context grounding.

Architecture (query-shard, single rank per GPU)
-----------------------------------------------
Each rank handles a disjoint slice of queries [shard_start, shard_end).
For each query in its slice:
  1. Look up retrieved top-k cluster contexts (per --ablation)
  2. Build a chat-style prompt: <quote question, retrieved contexts>
  3. Generate an answer with Qwen2.5-7B-Instruct (greedy, seed=0)
  4. Emit {source_id, dest_id, source_cluster_id, generation, ablation} JSONL

Four ablation configs (controlled by --ablation flag):
  - none      : LLM alone, empty context (baseline hallucination floor)
  - bm25      : top-k from cleaned BM25 results
  - bge_m3    : top-k from cleaned BGE-M3 results
  - reranker  : top-k from cross-encoder reranker output (full pipeline)

Why query sharding
------------------
Generation is per-query work over a fixed model + retrieved context. Splitting
queries across GPUs trivially parallelizes — no cross-rank coordination at
generation time. Merge step concatenates per-rank output files in order.

Why plain transformers (not vLLM)
---------------------------------
The MS3 certified stack pins transformers==4.41.2. Adding vLLM would force
a transformers upgrade and risk breaking MS3 reproducibility. Plain
transformers + batch=64 measured 563.5 tok/s on L4 (smoke test 7) — fast
enough for ~2-5h per-ablation wall-time per GPU, parallelizable across 4× L4.

Outputs (data/processed/rag/<ablation>/)
----------------------------------------
generations{.rank###}.jsonl   — per-query generated answers
generation_summary{.rank###}.json — provenance + timings per rank
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
GENERATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TOP_K_CONTEXT = 5      # how many retrieved clusters to feed as context
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 4096
DEFAULT_MAX_CHUNKS_PER_CLUSTER = 2  # how many corpus chunks per retrieved cluster
DEFAULT_TEMPERATURE = 0.0       # greedy decoding for reproducibility
DEFAULT_SEED = 0

DEFAULT_CORPUS = Path("data/processed/baseline/corpus_chunks_cleaned.jsonl")
DEFAULT_GOLD = Path("data/processed/baseline/cleaned/gold_pairs_test.jsonl")
DEFAULT_RETRIEVAL_DIR = Path("data/processed/baseline/cleaned")
DEFAULT_OUT_ROOT = Path("data/processed/rag")

# Ablation registry: maps ablation name -> {results_filename, label}
# results_filename=None means no retrieval (LLM-alone baseline)
ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "none":     {"results_filename": None,                 "label": "no_rag"},
    "bm25":     {"results_filename": "bm25_results.jsonl", "label": "bm25_rag"},
    "bge_m3":   {"results_filename": "bge_m3_results.jsonl","label": "bge_m3_rag"},
    "rrf":      {"results_filename": "rrf_results.jsonl",  "label": "rrf_rag"},
    "reranker": {"results_filename": "reranker_results.jsonl","label": "reranker_rag"},
}


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("rag_generate")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[rag_generate] %(message)s"))
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

    Dedup on (source_id, dest_id) — first occurrence wins; matches the contract
    in baseline_bm25.py / baseline_bge_m3.py / baseline_reranker.py.
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
    """Build {cluster_id: concatenated_text} index from cleaned corpus."""
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


def _resolve_ablation(name: str) -> dict[str, Any]:
    if name not in ABLATION_CONFIGS:
        raise KeyError(
            f"unknown ablation {name!r}; valid: {sorted(ABLATION_CONFIGS.keys())}"
        )
    return ABLATION_CONFIGS[name]


# ---------- prompt construction ----------


def _build_prompt(quote: str, contexts: list[str]) -> str:
    """Build the user message text for the chat template.

    Format: legal-domain instruction + numbered retrieved contexts (if any) +
    the citing quote as the question. Greedy decoding makes the answer
    deterministic given (model, prompt).
    """
    if contexts:
        ctx_block = "\n\n".join(
            f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )
        body = (
            "You are a legal research assistant. Use only the retrieved "
            "court-opinion contexts below to answer the question; if the "
            "contexts are not sufficient, say so explicitly.\n\n"
            f"{ctx_block}\n\n"
            f"Question: {quote}\n\nAnswer:"
        )
    else:
        body = (
            "You are a legal research assistant. Answer the following "
            "question about U.S. federal law as accurately as you can.\n\n"
            f"Question: {quote}\n\nAnswer:"
        )
    return body


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
    *,
    ablation: str,
    corpus_path: Path = DEFAULT_CORPUS,
    gold_path: Path = DEFAULT_GOLD,
    retrieval_dir: Path = DEFAULT_RETRIEVAL_DIR,
    out_root: Path = DEFAULT_OUT_ROOT,
    top_k_context: int = DEFAULT_TOP_K_CONTEXT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    max_chunks_per_cluster: int = DEFAULT_MAX_CHUNKS_PER_CLUSTER,
    rank: int = 0,
    world_size: int = 1,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run RAG generation for one ablation on this rank's query slice."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = _resolve_ablation(ablation)
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} out of range [0, {world_size})")
    if not Path(gold_path).is_file():
        raise FileNotFoundError(f"gold missing: {gold_path}")

    out_dir = Path(out_root) / cfg["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(seed)

    device = _detect_device()
    device_name = _device_name(device)
    rank_suffix = f".rank{rank:03d}" if world_size > 1 else ""
    gen_path = out_dir / f"generations{rank_suffix}.jsonl"
    summary_path = out_dir / f"generation_summary{rank_suffix}.json"

    logger.info("=" * 60)
    logger.info(
        f"MS5 RAG generation  ablation={ablation} (label={cfg['label']})  "
        f"rank={rank}/{world_size}  device={device_name}"
    )
    logger.info("=" * 60)

    # --- Load queries (full set) and shard ---
    logger.info("\n[1/5] Loading queries")
    all_queries = _load_queries(Path(gold_path))
    n_total = len(all_queries)
    shard_start, shard_end = _shard_range(n_total, rank, world_size)
    queries = all_queries[shard_start:shard_end]
    logger.info(f"  total queries: {n_total:,}")
    logger.info(f"  this shard   : [{shard_start:,}, {shard_end:,})  ({len(queries):,})")

    # --- Load retrieval results (if ablation != none) and align by row order ---
    retrieval_rows: list[dict[str, Any]] = []
    if cfg["results_filename"] is not None:
        results_path = Path(retrieval_dir) / cfg["results_filename"]
        if not results_path.is_file():
            raise FileNotFoundError(
                f"retrieval results missing for ablation={ablation}: {results_path}"
            )
        logger.info(f"\n[2/5] Loading retrieval results from {results_path}")
        all_rows = list(_iter_jsonl(results_path))
        if len(all_rows) != n_total:
            raise ValueError(
                f"retrieval results count ({len(all_rows):,}) != queries "
                f"({n_total:,}) — must be aligned"
            )
        for q, r in zip(all_queries[:1] + all_queries[-1:], all_rows[:1] + all_rows[-1:], strict=True):
            if (int(q["source_id"]), int(q["dest_id"])) != (int(r["source_id"]), int(r["dest_id"])):
                raise ValueError("queries / retrieval-results row order mismatch")
        retrieval_rows = all_rows[shard_start:shard_end]
    else:
        logger.info("\n[2/5] Skipping retrieval (ablation=none, no-RAG baseline)")

    # --- Build cluster_id -> text index (filtered to this rank's candidates) ---
    cluster_text: dict[int, str] = {}
    text_index_seconds = 0.0
    if retrieval_rows:
        logger.info("\n[3/5] Building cluster_id -> text index (filtered)")
        needed: set[int] = set()
        for r in retrieval_rows:
            for hit in r["retrieved"][:top_k_context]:
                needed.add(int(hit["cluster_id"]))
        logger.info(f"  unique candidate clusters this rank: {len(needed):,}")
        if not Path(corpus_path).is_file():
            raise FileNotFoundError(f"corpus missing: {corpus_path}")
        t0 = time.perf_counter()
        cluster_text = _load_cluster_text_index(
            Path(corpus_path),
            max_chunks_per_cluster=max_chunks_per_cluster,
            cluster_filter=needed,
        )
        text_index_seconds = time.perf_counter() - t0
        logger.info(
            f"  resolved {len(cluster_text):,}/{len(needed):,} clusters in "
            f"{text_index_seconds:.1f}s"
        )
    else:
        logger.info("\n[3/5] Skipping cluster text index (no contexts needed)")

    # --- Load Qwen2.5-7B-Instruct ---
    logger.info("\n[4/5] Loading generator model")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL, torch_dtype=model_dtype,
    ).to(device).eval()
    encoder_load_seconds = time.perf_counter() - t0
    logger.info(
        f"  model loaded in {encoder_load_seconds:.1f}s "
        f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)"
    )

    # --- Generate ---
    logger.info(f"\n[5/5] Generating {len(queries):,} answers (batch_size={batch_size})")
    t0 = time.perf_counter()
    gen_tmp = gen_path.with_suffix(".jsonl.tmp")
    log_interval = max(1, len(queries) // 50)

    n_generated = 0
    n_total_tokens_out = 0

    with gen_tmp.open("w", encoding="utf-8") as fout, torch.inference_mode():
        for bs_start in range(0, len(queries), batch_size):
            bs_end = min(bs_start + batch_size, len(queries))
            batch_queries = queries[bs_start:bs_end]

            # Build per-query prompts
            prompts: list[str] = []
            for qi_local, q in enumerate(batch_queries):
                qi_global = bs_start + qi_local
                contexts: list[str] = []
                if retrieval_rows:
                    rr = retrieval_rows[qi_global]
                    for hit in rr["retrieved"][:top_k_context]:
                        cid = int(hit["cluster_id"])
                        txt = cluster_text.get(cid)
                        if txt:
                            contexts.append(txt)
                user_msg = _build_prompt(quote=q["query_text"], contexts=contexts)
                chat_prompt = tok.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                prompts.append(chat_prompt)

            inputs = tok(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            ).to(device)

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )

            # Slice off prompt; decode only the new tokens per item
            input_lens = inputs["input_ids"].shape[1]
            for qi_local, q in enumerate(batch_queries):
                new_tokens = out[qi_local][input_lens:]
                answer = tok.decode(new_tokens, skip_special_tokens=True).strip()
                n_total_tokens_out += int((new_tokens != tok.pad_token_id).sum().item())
                fout.write(json.dumps({
                    "source_id": q["source_id"],
                    "dest_id": q["dest_id"],
                    "source_cluster_id": q["source_cluster_id"],
                    "ablation": ablation,
                    "generation": answer,
                }, allow_nan=False) + "\n")
                n_generated += 1

            if (bs_end) % (log_interval * batch_size) < batch_size:
                elapsed = time.perf_counter() - t0
                logger.info(
                    f"    [{n_generated:,}/{len(queries):,}] elapsed={elapsed:.0f}s  "
                    f"tokens={n_total_tokens_out:,}  throughput={n_total_tokens_out/max(elapsed,1):.1f} tok/s"
                )

    os.replace(gen_tmp, gen_path)
    generation_seconds = time.perf_counter() - t0
    logger.info(f"  generation done in {generation_seconds:.1f}s")

    # --- Summary ---
    results_hash = hashlib.sha256(gen_path.read_bytes()).hexdigest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "ablation": ablation,
        "ablation_label": cfg["label"],
        "n_queries_total": n_total,
        "n_queries_this_rank": len(queries),
        "n_generated": n_generated,
        "n_total_tokens_out": n_total_tokens_out,
        "top_k_context": top_k_context,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "max_length": max_length,
        "max_chunks_per_cluster": max_chunks_per_cluster,
        "generator_model": GENERATOR_MODEL,
        "device": device,
        "device_name": device_name,
        "world_size": world_size,
        "shard_rank": rank,
        "shard_start": shard_start,
        "shard_end": shard_end,
        "encoder_load_seconds": round(encoder_load_seconds, 3),
        "text_index_seconds": round(text_index_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
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
    ap = argparse.ArgumentParser(
        description="MS5 RAG generation with Qwen2.5-7B-Instruct over four ablations.",
    )
    ap.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=sorted(ABLATION_CONFIGS.keys()),
        help="Which retrieval ablation to run.",
    )
    ap.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--gold-path", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--top-k-context", type=int, default=DEFAULT_TOP_K_CONTEXT)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--max-chunks-per-cluster", type=int, default=DEFAULT_MAX_CHUNKS_PER_CLUSTER)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print(
            f"[rag_generate] DRY RUN  schema={SCHEMA_VERSION}  "
            f"model={GENERATOR_MODEL}  ablation={args.ablation}  "
            f"git_sha={_git_sha()}  args={vars(args)}"
        )
        sys.exit(0)
    main(
        ablation=args.ablation,
        corpus_path=args.corpus_path,
        gold_path=args.gold_path,
        retrieval_dir=args.retrieval_dir,
        out_root=args.out_root,
        top_k_context=args.top_k_context,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_chunks_per_cluster=args.max_chunks_per_cluster,
        rank=args.rank,
        world_size=args.world_size,
        seed=args.seed,
    )
