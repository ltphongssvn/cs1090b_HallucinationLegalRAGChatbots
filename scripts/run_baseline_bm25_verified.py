"""Reproducible runner for BM25 verified retrieval on cleaned corpus.

Wraps scripts/baseline_bm25.py --verified with --index-dir defaulted so
that re-runs reuse the saved index (skips ~37min rebuild).

Usage
-----
    uv run python scripts/run_baseline_bm25_verified.py \\
        --corpus-path data/processed/baseline/corpus_chunks_cleaned.jsonl \\
        --gold-pairs-path data/processed/baseline/cleaned/gold_pairs_test.jsonl \\
        --out-dir data/processed/baseline/cleaned \\
        --index-dir data/processed/baseline/bm25_index_cleaned
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TOP_K = 100
DEFAULT_SEED = 0


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reproducible BM25 verified retrieval runner.",
    )
    ap.add_argument("--corpus-path", type=Path, required=True)
    ap.add_argument("--gold-pairs-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, required=True,
                    help="Saved BM25 index dir (skips rebuild on re-run).")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--log-to-wandb", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main() -> int:
    args = _build_arg_parser().parse_args()
    print("[run_baseline_bm25_verified] resolved config:")
    print(f"  corpus_path      : {args.corpus_path}")
    print(f"  gold_pairs_path  : {args.gold_pairs_path}")
    print(f"  out_dir          : {args.out_dir}")
    print(f"  index_dir        : {args.index_dir}")
    print(f"  top_k            : {args.top_k}")
    print(f"  seed             : {args.seed}")
    print(f"  log_to_wandb     : {args.log_to_wandb}")
    print(f"  cpu_count        : {os.cpu_count()}")
    if args.dry_run:
        print("[run_baseline_bm25_verified] DRY RUN")
        return 0

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "baseline_bm25.py"),
        "--verified",
        "--corpus-path", str(args.corpus_path),
        "--gold-pairs-path", str(args.gold_pairs_path),
        "--out-dir", str(args.out_dir),
        "--index-dir", str(args.index_dir),
        "--top-k", str(args.top_k),
        "--seed", str(args.seed),
    ]
    if args.log_to_wandb:
        cmd.append("--log-to-wandb")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    print(f"[run_baseline_bm25_verified] launching: {' '.join(cmd)}")
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
