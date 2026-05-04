# scripts/run_mine_hard_negatives.py
"""Reproducible launcher for hard-negative mining on the verified pipeline."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GOLD = "data/processed/baseline/cleaned/gold_pairs_test.jsonl"
DEFAULT_RRF = "data/processed/baseline/cleaned/rrf_results.jsonl"
DEFAULT_CORPUS = "data/processed/baseline/corpus_chunks_cleaned.jsonl"
DEFAULT_OUT = "data/processed/finetune/hard_negatives.jsonl"
DEFAULT_N_NEG = 7
DEFAULT_NEG_RANK_MIN = 2
DEFAULT_NEG_RANK_MAX = 100
DEFAULT_MAX_CHUNKS_PER_CLUSTER = 2
DEFAULT_VAL_FRACTION = 0.05
DEFAULT_SEED = 0


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reproducible launcher for reranker hard-negative mining.",
    )
    ap.add_argument("--gold-path", default=DEFAULT_GOLD)
    ap.add_argument("--rrf-path", default=DEFAULT_RRF)
    ap.add_argument("--corpus-path", default=DEFAULT_CORPUS)
    ap.add_argument("--out-path", default=DEFAULT_OUT)
    ap.add_argument("--n-neg-per-pos", type=int, default=DEFAULT_N_NEG)
    ap.add_argument("--neg-rank-min", type=int, default=DEFAULT_NEG_RANK_MIN)
    ap.add_argument("--neg-rank-max", type=int, default=DEFAULT_NEG_RANK_MAX)
    ap.add_argument("--max-chunks-per-cluster", type=int, default=DEFAULT_MAX_CHUNKS_PER_CLUSTER)
    ap.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "mine_hard_negatives.py"),
        "--gold-path", args.gold_path,
        "--rrf-path", args.rrf_path,
        "--corpus-path", args.corpus_path,
        "--out-path", args.out_path,
        "--n-neg-per-pos", str(args.n_neg_per_pos),
        "--neg-rank-min", str(args.neg_rank_min),
        "--neg-rank-max", str(args.neg_rank_max),
        "--max-chunks-per-cluster", str(args.max_chunks_per_cluster),
        "--val-fraction", str(args.val_fraction),
        "--seed", str(args.seed),
    ]
    print(f"[run_mine_hard_negatives] launching: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
