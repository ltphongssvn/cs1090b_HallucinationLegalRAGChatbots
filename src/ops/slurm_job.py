"""SLURM job status monitoring utility.

Wraps `sacct` to extract job runtime metrics (elapsed, time limit, state) and
surface walltime headroom. Designed for reuse in scripts, notebooks, and CI
without coupling to live-cluster state — all subprocess calls are patchable.

Example (CLI):
    $ python -m src.ops.slurm_job 95397
    job=95397  state=RUNNING  elapsed=00:30:15 (1815s)  limit=08:00:00 (28800s)  used=6.3%  remaining=27000s

    $ python -m src.ops.slurm_job 95397 --extended
    (adds JobName, ExitCode, MaxRSS, AllocTRES)

    $ python -m src.ops.slurm_job 95397 --json
    {"elapsed_fraction": 0.063, "elapsed_seconds": 1815, ...}

Example (import):
    from src.ops.slurm_job import get_job_status
    status = get_job_status(95397)
    if status.elapsed_fraction > 0.85:
        print("Job will likely hit walltime; extend or checkpoint soon.")
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    elapsed_seconds: int
    time_limit_seconds: int | None
    state: str

    @property
    def remaining_seconds(self) -> int | None:
        if self.time_limit_seconds is None:
            return None
        return max(self.time_limit_seconds - self.elapsed_seconds, 0)

    @property
    def elapsed_fraction(self) -> float:
        if self.time_limit_seconds is None or self.time_limit_seconds == 0:
            return 0.0
        return self.elapsed_seconds / self.time_limit_seconds


@dataclass(frozen=True)
class ExtendedStatus:
    """Diagnostic-rich view including exit code, memory, allocation, name."""

    job_id: str
    elapsed_seconds: int
    time_limit_seconds: int | None
    state: str
    exit_code: str
    max_rss: str
    alloc_tres: str
    job_name: str


_DURATION_RE = re.compile(r"^(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})$")


def _parse_duration(s: str) -> int | None:
    """Parse SLURM duration: HH:MM:SS or D-HH:MM:SS. UNLIMITED -> None."""
    s = s.strip()
    if s.upper() in {"UNLIMITED", "PARTITION_LIMIT", ""}:
        return None
    m = _DURATION_RE.match(s)
    if not m:
        raise ValueError(f"unparseable SLURM duration: {s!r}")
    days, hours, minutes, seconds = m.groups()
    total = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    if days:
        total += int(days) * 86400
    return total


def _parse_sacct_line(line: str) -> JobStatus | None:
    """Parse one pipe-delimited sacct line. Returns None for .batch/.extern."""
    parts = line.strip().split("|")
    if len(parts) != 4:
        raise ValueError(f"malformed sacct line (need 4 fields, got {len(parts)}): {line!r}")
    job_id, elapsed, time_limit, state = parts
    if "." in job_id:
        return None
    return JobStatus(
        job_id=job_id,
        elapsed_seconds=_parse_duration(elapsed) or 0,
        time_limit_seconds=_parse_duration(time_limit),
        state=state.strip(),
    )


def get_job_status(job_id: int | str) -> JobStatus:
    """Fetch basic job status. Raises LookupError if job not found."""
    cmd = [
        "sacct",
        "-j",
        str(job_id),
        "--format=JobID,Elapsed,TimeLimit,State",
        "-n",
        "-P",
    ]
    stdout = subprocess.check_output(cmd, text=True)
    for line in stdout.splitlines():
        parsed = _parse_sacct_line(line)
        if parsed is not None and str(parsed.job_id) == str(job_id):
            return parsed
    raise LookupError(f"job {job_id} not found in sacct output")


def get_extended_status(job_id: int | str) -> ExtendedStatus:
    """Diagnostic query including ExitCode, MaxRSS, AllocTRES, JobName."""
    cmd = [
        "sacct",
        "-j",
        str(job_id),
        "--format=JobID,Elapsed,TimeLimit,State,ExitCode,MaxRSS,AllocTRES,JobName",
        "-n",
        "-P",
    ]
    stdout = subprocess.check_output(cmd, text=True)
    for line in stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 8:
            continue
        if "." in parts[0]:
            continue
        if str(parts[0]) != str(job_id):
            continue
        return ExtendedStatus(
            job_id=parts[0],
            elapsed_seconds=_parse_duration(parts[1]) or 0,
            time_limit_seconds=_parse_duration(parts[2]),
            state=parts[3].strip(),
            exit_code=parts[4].strip(),
            max_rss=parts[5].strip(),
            alloc_tres=parts[6].strip(),
            job_name=parts[7].strip(),
        )
    raise LookupError(f"job {job_id} not found in sacct output")


def _fmt_hms(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check SLURM job runtime status.")
    ap.add_argument("job_id", type=str, help="SLURM job ID")
    ap.add_argument(
        "--warn-fraction",
        type=float,
        default=0.85,
        help="exit 2 if elapsed/limit exceeds this (default 0.85)",
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument(
        "--extended",
        action="store_true",
        help="include ExitCode, MaxRSS, AllocTRES, JobName",
    )
    args = ap.parse_args(argv)

    try:
        status = get_job_status(args.job_id)
    except LookupError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.json:
        import json as _json

        payload = {
            "job_id": status.job_id,
            "state": status.state,
            "elapsed_seconds": status.elapsed_seconds,
            "time_limit_seconds": status.time_limit_seconds,
            "elapsed_fraction": round(status.elapsed_fraction, 4),
            "remaining_seconds": status.remaining_seconds,
        }
        print(_json.dumps(payload, sort_keys=True))
        if status.elapsed_fraction > args.warn_fraction:
            return 2
        return 0

    remaining = f"{status.remaining_seconds}s" if status.remaining_seconds is not None else "unlimited"
    limit_str = (
        f"{_fmt_hms(status.time_limit_seconds)} ({status.time_limit_seconds}s)"
        if status.time_limit_seconds is not None
        else "UNLIMITED"
    )
    print(
        f"job={status.job_id}  state={status.state}  "
        f"elapsed={_fmt_hms(status.elapsed_seconds)} ({status.elapsed_seconds}s)  "
        f"limit={limit_str}  "
        f"used={status.elapsed_fraction * 100:.1f}%  remaining={remaining}"
    )

    if args.extended:
        try:
            ext = get_extended_status(args.job_id)
            print(
                f"  job_name={ext.job_name}  exit_code={ext.exit_code}  "
                f"max_rss={ext.max_rss}  alloc_tres={ext.alloc_tres}"
            )
        except LookupError:
            pass

    if status.elapsed_fraction > args.warn_fraction:
        print(
            f"WARNING: elapsed fraction {status.elapsed_fraction:.2%} exceeds warn threshold {args.warn_fraction:.2%}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
