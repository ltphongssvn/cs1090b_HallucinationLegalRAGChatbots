"""Reproducible runner for clean_corpus on the verified 27GB corpus.

Production defaults derived from empirical measurements on Harvard ODD
(Apr 26, 2026): eyecite + HyperscanTokenizer cleans ~65 rows/sec, so
25K-row shards complete in ~6.5min, well under the 1200s shard timeout.

Pre-warms the HyperscanTokenizer cache before spawning workers to avoid
first-run race conditions on the cache file.

Usage
-----
    uv run python scripts/run_clean_corpus.py \\
        --in-path data/processed/baseline/corpus_chunks_enriched.jsonl \\
        --out-path data/processed/baseline/corpus_chunks_cleaned.jsonl

    # All defaults overridable via CLI flags.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Production defaults (Harvard ODD 48-core node, Apr 2026)
DEFAULT_WORKERS = 48
DEFAULT_ROWS_PER_SHARD = 25_000
DEFAULT_SHARD_TIMEOUT_SEC = 1200.0


def _warmup_hyperscan_cache() -> None:
    """Pre-compile + cache HyperscanTokenizer DB so workers don't race."""
    print("[run_clean_corpus] warming hyperscan cache (first call ~5s)...")
    from clean_query import clean_destination_context

    clean_destination_context("Brown v. Board, 347 U.S. 483 (1954)")
    cache_dir = REPO_ROOT / ".hyperscan_cache"
    if cache_dir.exists():
        size_mb = sum(f.stat().st_size for f in cache_dir.iterdir()) / 1e6
        print(f"[run_clean_corpus]   .hyperscan_cache primed ({size_mb:.1f} MB)")


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reproducible runner for cleaning the 27GB corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in-path", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel workers (default {DEFAULT_WORKERS}).",
    )
    ap.add_argument(
        "--rows-per-shard",
        type=int,
        default=DEFAULT_ROWS_PER_SHARD,
        help=f"Rows per shard (default {DEFAULT_ROWS_PER_SHARD}).",
    )
    ap.add_argument(
        "--shard-timeout-sec",
        type=float,
        default=DEFAULT_SHARD_TIMEOUT_SEC,
        help=f"Per-shard timeout in seconds (default {DEFAULT_SHARD_TIMEOUT_SEC}).",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main() -> int:
    args = _build_arg_parser().parse_args()
    print("[run_clean_corpus] resolved config:")
    print(f"  in_path           : {args.in_path}")
    print(f"  out_path          : {args.out_path}")
    print(f"  workers           : {args.workers}")
    print(f"  rows_per_shard    : {args.rows_per_shard}")
    print(f"  shard_timeout_sec : {args.shard_timeout_sec}")
    print(f"  cpu_count         : {os.cpu_count()}")
    if args.dry_run:
        print("[run_clean_corpus] DRY RUN — exiting before warmup")
        return 0

    _warmup_hyperscan_cache()

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "clean_corpus.py"),
        "--in-path", str(args.in_path),
        "--out-path", str(args.out_path),
        "--workers", str(args.workers),
        "--mode", "subprocess",
        "--rows-per-shard", str(args.rows_per_shard),
        "--shard-timeout-sec", str(args.shard_timeout_sec),
    ]
    print(f"[run_clean_corpus] launching: {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
