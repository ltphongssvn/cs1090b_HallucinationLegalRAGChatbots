# scripts/merge_reranker_shards.py
"""Merge per-rank reranker output shards into a single results file + summary.

Called by scripts/baseline_reranker_multigpu.sbatch after all 4 query-shard
workers complete. Replaces the previously-inlined `python -c` heredoc to
avoid shell-quoting fragility and to make the merge step independently
testable / re-runnable.

Usage
-----
    .venv/bin/python scripts/merge_reranker_shards.py \
        --out-dir data/processed/baseline/cleaned \
        --world-size 4 \
        --seed 0

What it does
------------
1. Read all `reranker_results.rank{NNN}.jsonl` files in out-dir
2. Verify shard count matches --world-size
3. Concatenate them in rank order via _merge_shard_results
4. Cross-check merged row count == sum of per-rank n_queries_this_rank
5. Aggregate per-rank summaries (max-of-rank for wall-times, sum for pair counts)
6. Write merged `reranker_results.jsonl` + `reranker_summary.json`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

# Ensure repo root on sys.path so `scripts.baseline_reranker` resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.baseline_reranker import (
    _git_sha,
    _merge_shard_results,
    RERANKER_MODEL,
    SCHEMA_VERSION,
)


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("merge_reranker_shards")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[merge_reranker_shards] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


def merge(out_dir: Path, world_size: int, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"out_dir does not exist: {out_dir}")

    shard_paths = sorted(out_dir.glob("reranker_results.rank*.jsonl"))
    if len(shard_paths) != world_size:
        raise RuntimeError(
            f"expected {world_size} reranker_results.rank*.jsonl shards, "
            f"got {len(shard_paths)}: {[p.name for p in shard_paths]}"
        )
    rank_summary_paths = sorted(out_dir.glob("reranker_summary.rank*.json"))
    if len(rank_summary_paths) != world_size:
        raise RuntimeError(
            f"expected {world_size} reranker_summary.rank*.json files, "
            f"got {len(rank_summary_paths)}"
        )

    merged_results = out_dir / "reranker_results.jsonl"
    merged_summary = out_dir / "reranker_summary.json"

    logger.info(f"merging {len(shard_paths)} shards -> {merged_results}")
    _merge_shard_results(shard_paths, merged_results)

    # Aggregate summaries
    rank_summaries = [json.loads(p.read_text()) for p in rank_summary_paths]
    rank0 = rank_summaries[0]
    n_queries_total = int(rank0["n_queries_total"])
    n_pairs_scored_total = sum(int(s["n_pairs_scored"]) for s in rank_summaries)
    n_queries_emitted = sum(1 for _ in merged_results.open())
    if n_queries_emitted != n_queries_total:
        raise RuntimeError(
            f"merged row count {n_queries_emitted:,} != n_queries_total "
            f"{n_queries_total:,}"
        )

    results_hash = hashlib.sha256(merged_results.read_bytes()).hexdigest()

    # Worst-case (max) per-rank wall-times
    encoder_load_seconds = max(float(s["encoder_load_seconds"]) for s in rank_summaries)
    rerank_seconds = max(float(s["rerank_seconds"]) for s in rank_summaries)
    text_index_seconds = max(float(s["text_index_seconds"]) for s in rank_summaries)

    merged_summary_dict = {
        "schema_version": SCHEMA_VERSION,
        "n_queries_total": n_queries_total,
        "n_pairs_scored_total": n_pairs_scored_total,
        "top_k_input": int(rank0["top_k_input"]),
        "top_k_output": int(rank0["top_k_output"]),
        "max_length": int(rank0["max_length"]),
        "batch_size": int(rank0["batch_size"]),
        "max_chunks_per_cluster": int(rank0["max_chunks_per_cluster"]),
        "reranker_model": RERANKER_MODEL,
        "device": rank0["device"],
        "device_name": rank0["device_name"],
        "world_size": world_size,
        "encoder_load_seconds": encoder_load_seconds,
        "text_index_seconds": text_index_seconds,
        "rerank_seconds": rerank_seconds,
        "seed": seed,
        "git_sha": _git_sha(),
        "results_hash": results_hash,
    }
    merged_summary.write_text(
        json.dumps(merged_summary_dict, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    logger.info(
        f"merged: n_queries={n_queries_total:,}  pairs={n_pairs_scored_total:,}  "
        f"hash={results_hash[:16]}"
    )
    logger.info(f"wrote {merged_results}")
    logger.info(f"wrote {merged_summary}")
    return merged_summary_dict


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Merge per-rank reranker output shards into unified results + summary.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    merge(args.out_dir, args.world_size, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
