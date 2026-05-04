# scripts/merge_bge_m3_shards.py
"""Merge per-rank BGE-M3 corpus-shard outputs into unified results + summary.

Called by scripts/baseline_bge_m3_multigpu.sbatch after all 4 corpus-shard
workers complete. Replaces the previously-inlined `python -c` heredoc to
avoid shell-quoting fragility and to make the merge step independently
testable / re-runnable.

Verified vs legacy mode is selected by --verified (mirrors baseline_bge_m3.py
flag): controls join key (source_cluster_id vs dest_id), match field
(cluster_id vs opinion_id), and which id field to read from per-rank
chunk_meta files for the unique-id count.

Usage
-----
    .venv/bin/python scripts/merge_bge_m3_shards.py \
        --out-dir data/processed/baseline/cleaned \
        --world-size 4 \
        --top-k 100 \
        --encode-batch-size 32 \
        --seed 0 \
        --verified

What it does
------------
1. Read all `bge_m3_results.rank{NNN}.jsonl` shards in --out-dir
2. Verify shard count matches --world-size
3. Cross-shard MaxP merge via _merge_shard_results (verified-mode aware)
4. Aggregate per-rank summaries: max-of-rank wall-times, sum corpus chunks,
   union of cluster_ids / opinion_ids across rank meta files
5. Write merged `bge_m3_results.jsonl` + Pydantic-validated `bge_m3_summary.json`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

# Ensure repo root on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.baseline_bge_m3 import (
    DTYPE,
    EMBEDDING_DIM,
    ENCODER_MODEL,
    MAX_LENGTH,
    NORMALIZE_EMBEDDINGS,
    SCHEMA_VERSION,
    SIMILARITY_METRIC,
    _git_sha,
    _merge_shard_results,
)


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("merge_bge_m3_shards")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[merge_bge_m3_shards] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


def merge(
    out_dir: Path,
    *,
    world_size: int,
    top_k: int,
    encode_batch_size: int,
    seed: int = 0,
    verified: bool = False,
) -> dict:
    from src.eda_schemas import BaselineBgeM3Summary

    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"out_dir does not exist: {out_dir}")

    shard_paths = sorted(out_dir.glob("bge_m3_results.rank*.jsonl"))
    if len(shard_paths) != world_size:
        raise RuntimeError(
            f"expected {world_size} bge_m3_results.rank*.jsonl shards, "
            f"got {len(shard_paths)}: {[p.name for p in shard_paths]}"
        )
    rank_summary_paths = sorted(out_dir.glob("bge_m3_summary.rank*.json"))
    if len(rank_summary_paths) != world_size:
        raise RuntimeError(
            f"expected {world_size} bge_m3_summary.rank*.json files, "
            f"got {len(rank_summary_paths)}"
        )
    rank_meta_paths = sorted(out_dir.glob("bge_m3_index_meta.rank*.jsonl"))
    if len(rank_meta_paths) != world_size:
        raise RuntimeError(
            f"expected {world_size} bge_m3_index_meta.rank*.jsonl files, "
            f"got {len(rank_meta_paths)}"
        )

    merged_results = out_dir / "bge_m3_results.jsonl"
    merged_summary_path = out_dir / "bge_m3_summary.json"

    logger.info(
        f"merging {len(shard_paths)} shards (verified={verified}) -> {merged_results}"
    )
    _merge_shard_results(shard_paths, merged_results, top_k=top_k, verified=verified)

    # Aggregate summaries
    rank_summaries = [json.loads(p.read_text()) for p in rank_summary_paths]
    rank0 = rank_summaries[0]
    n_chunks_total = sum(int(s["n_corpus_chunks"]) for s in rank_summaries)

    # Union of unique ids across rank meta files
    id_field = "cluster_id" if verified else "opinion_id"
    unique_ids: set[int] = set()
    for mp in rank_meta_paths:
        with mp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                unique_ids.add(int(row[id_field]))

    n_queries = sum(1 for _ in merged_results.open(encoding="utf-8"))
    results_hash = hashlib.sha256(merged_results.read_bytes()).hexdigest()

    # Worst-case (max) per-rank wall-times
    encoder_load_seconds = max(float(s["encoder_load_seconds"]) for s in rank_summaries)
    index_build_seconds = max(float(s["index_build_seconds"]) for s in rank_summaries)
    query_encode_seconds = max(float(s["query_encode_seconds"]) for s in rank_summaries)
    retrieval_seconds = max(float(s["retrieval_seconds"]) for s in rank_summaries)

    merged_summary = {
        "schema_version": SCHEMA_VERSION,
        "n_queries": n_queries,
        "n_corpus_chunks": n_chunks_total,
        "n_unique_opinions": len(unique_ids),
        "top_k": top_k,
        "encoder_model": ENCODER_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "device": rank0["device"],
        "device_name": rank0["device_name"],
        "encode_batch_size": encode_batch_size,
        "similarity_metric": SIMILARITY_METRIC,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "max_length": MAX_LENGTH,
        "dtype": DTYPE,
        "encoder_load_seconds": encoder_load_seconds,
        "index_build_seconds": index_build_seconds,
        "query_encode_seconds": query_encode_seconds,
        "retrieval_seconds": retrieval_seconds,
        "seed": seed,
        "world_size": world_size,
        "shard_rank": 0,
        "git_sha": _git_sha(),
        "results_hash": results_hash,
    }
    validated = BaselineBgeM3Summary.model_validate(merged_summary)
    merged_summary_path.write_text(
        json.dumps(validated.model_dump(), sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    logger.info(
        f"merged: n_queries={n_queries:,}  n_corpus_chunks={n_chunks_total:,}  "
        f"n_unique_{id_field}={len(unique_ids):,}  hash={results_hash[:16]}"
    )
    logger.info(f"wrote {merged_results}")
    logger.info(f"wrote {merged_summary_path}")
    return validated.model_dump()


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Merge per-rank BGE-M3 corpus-shard outputs into unified results + summary.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--top-k", type=int, required=True)
    ap.add_argument("--encode-batch-size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--verified",
        action="store_true",
        help="Verified mode (cluster_id keying); legacy=opinion_id keying",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    merge(
        args.out_dir,
        world_size=args.world_size,
        top_k=args.top_k,
        encode_batch_size=args.encode_batch_size,
        seed=args.seed,
        verified=args.verified,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
