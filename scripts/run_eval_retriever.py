# scripts/run_eval_retriever.py
"""Reproducible eval runner for any single retriever variant.

Runs both standard single eval (writes <results>.eval.json) and stratified
eval (writes <results>.stratified.json) on a retriever\'s output. Call this
after any reranker / RRF / BM25 / BGE-M3 SLURM job completes.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GOLD = "data/processed/baseline/cleaned/gold_pairs_test.jsonl"


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run single + stratified eval for one retriever.")
    ap.add_argument("--gold-path", default=DEFAULT_GOLD)
    ap.add_argument("--results-path", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-buckets", type=int, default=3)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")

    # 1. Standard single eval
    cmd1 = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_baseline_eval.py"), "single",
        "--gold-path", args.gold_path,
        "--results-path", args.results_path,
        "--label", args.label,
    ]
    print(f"[run_eval_retriever] >>> single eval: {args.label}")
    rc1 = subprocess.run(cmd1, env=env, check=False).returncode

    # 2. Stratified eval
    cmd2 = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "stratified_eval.py"),
        "--gold-path", args.gold_path,
        "--results-path", args.results_path,
        "--label", args.label,
        "--n-buckets", str(args.n_buckets),
    ]
    print(f"[run_eval_retriever] >>> stratified eval: {args.label}")
    rc2 = subprocess.run(cmd2, env=env, check=False).returncode

    return rc1 or rc2


if __name__ == "__main__":
    sys.exit(main())
