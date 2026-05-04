# scripts/run_rag_ablations.py
"""Reproducible launcher for RAG generation across all 4 retrieval ablations.

Submits 4 SLURM jobs (none, bm25, bge_m3, rrf) in parallel via run_slurm_job.py.
The reranker_finetuned ablation is submitted separately after the fine-tuned
reranker is ready (run with --ablations reranker after Step 5 of the pipeline).
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
DEFAULT_OUT_ROOT = "data/processed/rag"
DEFAULT_ABLATIONS = ["none", "bm25", "bge_m3", "rrf"]
RETRIEVAL_DIR_FOR_ABLATION = {
    "none":     "data/processed/baseline/cleaned",
    "bm25":     "data/processed/baseline/cleaned",
    "bge_m3":   "data/processed/baseline/cleaned",
    "rrf":      "data/processed/baseline/cleaned",
    "reranker": "data/processed/baseline/cleaned/finetuned",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reproducible launcher for RAG generation across ablations.",
    )
    ap.add_argument("--ablations", nargs="+", default=DEFAULT_ABLATIONS,
                    choices=list(RETRIEVAL_DIR_FOR_ABLATION.keys()))
    ap.add_argument("--corpus-path", default=DEFAULT_CORPUS)
    ap.add_argument("--gold-path", default=DEFAULT_GOLD)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--no-poll", action="store_true", default=True,
                    help="Submit and exit (default; ablations run in parallel)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    rc_total = 0
    for ablation in args.ablations:
        retrieval_dir = RETRIEVAL_DIR_FOR_ABLATION[ablation]
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_slurm_job.py"),
            "--sbatch", str(REPO_ROOT / "scripts" / "rag_generate_multigpu.sbatch"),
            "--env", f"ABLATION={ablation}",
            "--env", f"CORPUS_PATH={args.corpus_path}",
            "--env", f"GOLD_PATH={args.gold_path}",
            "--env", f"RETRIEVAL_DIR={retrieval_dir}",
            "--env", f"OUT_ROOT={args.out_root}",
            "--no-poll",
        ]
        print(f"[run_rag_ablations] >>> ablation={ablation}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
        rc = subprocess.run(cmd, env=env, check=False).returncode
        if rc != 0:
            print(f"[run_rag_ablations] FAILED submission for {ablation} rc={rc}")
            rc_total = 1
    return rc_total


if __name__ == "__main__":
    sys.exit(main())
