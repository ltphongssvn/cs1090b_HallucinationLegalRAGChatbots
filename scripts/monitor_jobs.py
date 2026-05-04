#!/usr/bin/env python3
"""Hourly job monitor — flags TIMEOUT/FAIL/OOM/Traceback in active SLURM jobs.

Usage:
    .venv/bin/python scripts/monitor_jobs.py
    .venv/bin/python scripts/monitor_jobs.py --jobs 102965 102966 102967
    watch -n 3600 ".venv/bin/python scripts/monitor_jobs.py"
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"

ALERT_PATTERNS = re.compile(
    r"OutOfMemoryError|TIMEOUT|CANCELLED|FAILED|OUT_OF_MEMORY|Traceback",
    re.IGNORECASE,
)
TERMINAL_FAIL_STATES = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL")


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def _squeue_user(user: str) -> str:
    return _run(["squeue", "-u", user, "-o", "%.10i %.20j %.2t %.10M %R"])


def _sacct_state(job_id: str) -> str:
    out = _run(["sacct", "-j", job_id, "--format=State", "-n", "-P"])
    return out.splitlines()[0].strip() if out.strip() else "UNKNOWN"


def _scan_logs_for_alerts(job_id: str) -> list[tuple[Path, str]]:
    """Return list of (log_path, last_alert_lines) for any log mentioning failure patterns."""
    alerts: list[tuple[Path, str]] = []
    for log in LOGS_DIR.glob(f"*_{job_id}.log"):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if ALERT_PATTERNS.search(text):
            tail_lines = text.strip().splitlines()[-10:]
            alerts.append((log, "\n".join(tail_lines)))
    return alerts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Monitor active SLURM jobs for failures.")
    ap.add_argument("--jobs", nargs="*", default=None,
                    help="Specific job IDs to track (default: all user jobs)")
    ap.add_argument("--user", default=os.environ.get("USER", ""))
    args = ap.parse_args(argv)

    print(f"=== monitor {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    queue = _squeue_user(args.user)
    print(queue if queue.strip() else "  (no jobs in queue)")

    job_ids = args.jobs
    if not job_ids:
        # Auto-detect: pull job IDs from squeue
        job_ids = re.findall(r"^\s*(\d{4,})\s", queue, flags=re.MULTILINE)

    if not job_ids:
        print("  no job IDs to track")
        return 0

    print("--- sacct states ---")
    any_alert = False
    for j in job_ids:
        state = _sacct_state(j)
        marker = ""
        if any(s in state for s in TERMINAL_FAIL_STATES):
            marker = "  *** ALERT ***"
            any_alert = True
        print(f"  {j}: {state}{marker}")

    print("--- log scan for OutOfMemoryError / Traceback / FAIL / TIMEOUT ---")
    for j in job_ids:
        alerts = _scan_logs_for_alerts(j)
        if not alerts:
            continue
        any_alert = True
        for log, tail in alerts:
            print(f"  *** ALERT in {log} ***")
            for line in tail.splitlines():
                print(f"    {line}")

    if not any_alert:
        print("  no alerts")
    return 1 if any_alert else 0


if __name__ == "__main__":
    sys.exit(main())
