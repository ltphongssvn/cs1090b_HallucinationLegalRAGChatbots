# scripts/judge_watchdog.py
"""Watchdog: monitors parallel hallucination judge children, restarts on death.

Usage:
    nohup .venv/bin/python scripts/judge_watchdog.py > logs/judge_watchdog.log 2>&1 & disown
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK_INTERVAL_SEC = 900

LABEL_FOR_ABLATION = {
    "none": "no_rag", "bm25": "bm25_rag",
    "bge_m3": "bge_m3_rag", "rrf": "rrf_rag",
    "reranker": "reranker_rag",
}


def _build_arg_parser():
    ap = argparse.ArgumentParser(description="Watchdog for parallel judge processes.")
    ap.add_argument("--ablations", nargs="+",
                    default=["none", "bm25", "bge_m3", "rrf"])
    ap.add_argument("--check-interval-sec", type=int, default=CHECK_INTERVAL_SEC)
    ap.add_argument("--max-restarts", type=int, default=20)
    ap.add_argument("--judge-log", default="logs/judge_full2.log")
    ap.add_argument("--target-n", type=int, default=20877)
    ap.add_argument("--dry-run", action="store_true", help="Count children once, exit without launching")
    return ap


def _count_judge_children() -> int:
    """Count alive judge processes via /proc (ps -ef truncates cmdline)."""
    n = 0
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        cmdline_path = entry / "cmdline"
        try:
            cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        except (OSError, FileNotFoundError):
            continue
        if "scripts/hallucination_judge.py" in cmdline:
            n += 1
    return n


def _is_done(ablation: str, target_n: int = 20877) -> bool:
    label = LABEL_FOR_ABLATION[ablation]
    p = REPO_ROOT / "data" / "processed" / "hallucination" / label / "judgments.jsonl"
    if not p.is_file():
        return False
    with p.open() as f:
        n = sum(1 for _ in f)
    return n >= target_n


def _all_done(ablations: list[str], target_n: int = 20877) -> bool:
    return all(_is_done(a, target_n) for a in ablations)


def _launch_judge(ablations: list[str], judge_log: str) -> None:
    log_path = REPO_ROOT / judge_log
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_hallucination_judge_parallel.py"),
        "--ablations", *ablations,
        "--max-parallel", str(len(ablations)),
    ]
    print(f"[{datetime.now()}] launching: {' '.join(cmd)}")
    sys.stdout.flush()
    log_fh = log_path.open("a", encoding="utf-8")
    subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        start_new_session=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    print(f"[{datetime.now()}] watchdog started, check interval {args.check_interval_sec}s")
    sys.stdout.flush()

    if args.dry_run:
        n = _count_judge_children()
        done = _all_done(args.ablations, args.target_n)
        print(f"[dry-run] alive judge children: {n}")
        print(f"[dry-run] all ablations done   : {done}")
        for ab in args.ablations:
            print(f"[dry-run]   {ab}: done={_is_done(ab, args.target_n)}")
        return 0

    restarts = 0
    while True:
        if _all_done(args.ablations, args.target_n):
            print(f"[{datetime.now()}] all ablations complete; exiting")
            return 0

        n_alive = _count_judge_children()
        print(f"[{datetime.now()}] alive children: {n_alive}")
        sys.stdout.flush()

        if n_alive == 0:
            if restarts >= args.max_restarts:
                print(f"[{datetime.now()}] max_restarts {args.max_restarts} reached; exiting")
                return 1
            restarts += 1
            print(f"[{datetime.now()}] restart #{restarts}")
            _launch_judge(args.ablations, args.judge_log)
            time.sleep(60)
        time.sleep(args.check_interval_sec)


if __name__ == "__main__":
    sys.exit(main())
