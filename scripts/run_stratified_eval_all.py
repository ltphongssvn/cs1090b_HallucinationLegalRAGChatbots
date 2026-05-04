# scripts/run_stratified_eval_all.py
"""Reproducible runner for stratified eval across all retriever variants.

Usage:
    .venv/bin/python scripts/run_stratified_eval_all.py
    .venv/bin/python scripts/run_stratified_eval_all.py --retrievers bm25 bge_m3 rrf reranker
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = "data/processed/baseline/cleaned/gold_pairs_test.jsonl"
DEFAULT_CLEANED_DIR = "data/processed/baseline/cleaned"
DEFAULT_RETRIEVERS = ["bm25", "bge_m3", "rrf", "reranker"]
DEFAULT_N_BUCKETS = 3


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run stratified eval (HEAD/TORSO/TAIL) across retriever variants.",
    )
    ap.add_argument("--gold-path", default=DEFAULT_GOLD)
    ap.add_argument("--cleaned-dir", default=DEFAULT_CLEANED_DIR)
    ap.add_argument("--retrievers", nargs="+", default=DEFAULT_RETRIEVERS)
    ap.add_argument("--n-buckets", type=int, default=DEFAULT_N_BUCKETS, choices=(2, 3))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    rc_total = 0
    for retriever in args.retrievers:
        results_path = Path(args.cleaned_dir) / f"{retriever}_results.jsonl"
        if not results_path.is_file():
            print(f"[run_stratified_eval_all] SKIP {retriever}: missing {results_path}")
            continue
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "stratified_eval.py"),
            "--gold-path", args.gold_path,
            "--results-path", str(results_path),
            "--label", retriever,
            "--n-buckets", str(args.n_buckets),
        ]
        print(f"[run_stratified_eval_all] >>> {retriever}")
        rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, check=False).returncode
        if rc != 0:
            print(f"[run_stratified_eval_all] FAILED {retriever} rc={rc}")
            rc_total = 1
    return rc_total


if __name__ == "__main__":
    sys.exit(main())
