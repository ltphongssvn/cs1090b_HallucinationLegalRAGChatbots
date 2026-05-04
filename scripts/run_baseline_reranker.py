# scripts/run_baseline_reranker.py
"""Reproducible launcher for the verified MS4 cross-encoder reranker stage.

Wraps scripts/run_slurm_job.py with the canonical env-var overrides for
the verified pipeline (max_length=1024 per BAAI fine-tuning recommendation
for bge-reranker-v2-m3, max_chunks_per_cluster=2 to fit the 1024-token
budget without mid-document truncation).

Usage
-----
    .venv/bin/python scripts/run_baseline_reranker.py
    .venv/bin/python scripts/run_baseline_reranker.py --no-poll
    .venv/bin/python scripts/run_baseline_reranker.py --max-length 2048

Defaults match the verified-pipeline contract; flags allow ablation runs
without editing the script.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORPUS = "data/processed/baseline/corpus_chunks_cleaned.jsonl"
DEFAULT_GOLD = "data/processed/baseline/cleaned/gold_pairs_test.jsonl"
DEFAULT_INPUT_RESULTS = "data/processed/baseline/cleaned/rrf_results.jsonl"
DEFAULT_OUT_DIR = "data/processed/baseline/cleaned"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_MAX_CHUNKS_PER_CLUSTER = 2
DEFAULT_BATCH_SIZE = 32


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reproducible launcher for verified MS4 reranker stage.",
    )
    ap.add_argument("--corpus-path", default=DEFAULT_CORPUS)
    ap.add_argument("--gold-path", default=DEFAULT_GOLD)
    ap.add_argument("--input-results", default=DEFAULT_INPUT_RESULTS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--max-chunks-per-cluster", type=int,
                    default=DEFAULT_MAX_CHUNKS_PER_CLUSTER)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--encoder-dir", default="",
                    help="Path to fine-tuned reranker (empty = HF hub bge-reranker-v2-m3)")
    ap.add_argument("--check-existing", action="store_true",
                    help="Skip submit if matching SLURM job already running")
    ap.add_argument("--no-poll", action="store_true",
                    help="Submit and exit; don't block on completion.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_slurm_job.py"),
        "--sbatch", str(REPO_ROOT / "scripts" / "baseline_reranker_multigpu.sbatch"),
        "--env", f"CORPUS_PATH={args.corpus_path}",
        "--env", f"GOLD_PATH={args.gold_path}",
        "--env", f"INPUT_RESULTS={args.input_results}",
        "--env", f"OUT_DIR={args.out_dir}",
        "--env", f"MAX_LENGTH={args.max_length}",
        "--env", f"MAX_CHUNKS_PER_CLUSTER={args.max_chunks_per_cluster}",
        "--env", f"BATCH_SIZE={args.batch_size}",
        "--env", f"ENCODER_DIR={args.encoder_dir}",
    ]
    if args.check_existing:
        cmd.append("--check-existing")
    if args.no_poll:
        cmd.append("--no-poll")
    else:
        cmd.extend(["--poll-interval-sec", "60", "--max-wait-min", "1320"])

    print(f"[run_baseline_reranker] launching: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
