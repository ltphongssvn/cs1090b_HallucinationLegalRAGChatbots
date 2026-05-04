# scripts/run_finetune_reranker.py
"""Reproducible launcher for legal-domain reranker fine-tuning (4× L4 DDP)."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TRAIN = "data/processed/finetune/hard_negatives.jsonl"
DEFAULT_VAL = "data/processed/finetune/hard_negatives.val.jsonl"
DEFAULT_OUTPUT_DIR = "data/processed/finetune/bge_reranker_legal"


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reproducible launcher for legal reranker fine-tuning.",
    )
    ap.add_argument("--train-path", default=DEFAULT_TRAIN)
    ap.add_argument("--val-path", default=DEFAULT_VAL)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--lr", default="2e-5")
    ap.add_argument("--epochs", default="2")
    ap.add_argument("--batch-size", default="4")
    ap.add_argument("--grad-accum", default="8")
    ap.add_argument("--max-length", default="1024")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--check-existing", action="store_true",
                    help="Skip submit if matching SLURM job already running")
    ap.add_argument("--no-poll", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_slurm_job.py"),
        "--sbatch", str(REPO_ROOT / "scripts" / "finetune_reranker.sbatch"),
        "--env", f"TRAIN_PATH={args.train_path}",
        "--env", f"VAL_PATH={args.val_path}",
        "--env", f"OUTPUT_DIR={args.output_dir}",
        "--env", f"LR={args.lr}",
        "--env", f"EPOCHS={args.epochs}",
        "--env", f"BATCH_SIZE={args.batch_size}",
        "--env", f"GRAD_ACCUM={args.grad_accum}",
        "--env", f"MAX_LENGTH={args.max_length}",
        "--env", f"SEED={args.seed}",
    ]
    if args.check_existing:
        cmd.append("--check-existing")
    if args.no_poll:
        cmd.append("--no-poll")
    else:
        cmd.extend(["--poll-interval-sec", "60", "--max-wait-min", "1320",
                    "--resume-on-timeout", "--max-cycles", "2"])
    print(f"[run_finetune_reranker] launching: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
