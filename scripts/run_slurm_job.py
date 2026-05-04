# scripts/run_slurm_job.py
"""Reproducible Python wrapper for SLURM job submission + polling + resume.

Replaces ad-hoc terminal `sbatch ... .sbatch` invocations and inline
notebook subprocess logic with a single CLI runner that any team member or
TF reviewer can call to reproduce a multi-GPU stage.

Capabilities
------------
1. Submit a SLURM batch script with a typed list of env-var overrides
2. Poll squeue every N seconds and surface state transitions
3. On TIMEOUT, optionally resubmit automatically until success or max-cycles
4. Validate exit state via sacct after the job leaves the queue
5. All output written to logs/ in the canonical {job_name}_{job_id}.log path

Usage
-----
    # MS3 BGE-M3 verified retrieval (multi-cycle resume on TIMEOUT)
    .venv/bin/python scripts/run_slurm_job.py \
        --sbatch scripts/baseline_bge_m3_multigpu.sbatch \
        --env VERIFIED=1 \
        --env CORPUS_PATH=data/processed/baseline/corpus_chunks_cleaned.jsonl \
        --env GOLD_PATH=data/processed/baseline/cleaned/gold_pairs_test.jsonl \
        --env OUT_DIR=data/processed/baseline/cleaned \
        --resume-on-timeout \
        --max-cycles 5

    # MS4 reranker (single cycle, ~6h)
    .venv/bin/python scripts/run_slurm_job.py \
        --sbatch scripts/baseline_reranker_multigpu.sbatch \
        --env CORPUS_PATH=data/processed/baseline/corpus_chunks_cleaned.jsonl \
        --env GOLD_PATH=data/processed/baseline/cleaned/gold_pairs_test.jsonl \
        --env INPUT_RESULTS=data/processed/baseline/cleaned/rrf_results.jsonl \
        --env OUT_DIR=data/processed/baseline/cleaned

    # Submit only (don't poll)
    .venv/bin/python scripts/run_slurm_job.py \
        --sbatch scripts/baseline_reranker_multigpu.sbatch \
        --no-poll

    # Poll an already-running job (recover from session disconnect)
    .venv/bin/python scripts/run_slurm_job.py --poll-only --job-id 102631

Design notes
------------
- Idempotency: the underlying sbatch scripts already check for valid
  output artifacts and skip work. Resubmission after TIMEOUT is therefore
  safe — checkpointed work resumes from last flush.
- Polling cadence: 60s default, matches the existing notebook cells.
- Exit codes: 0 on COMPLETED, 1 on FAILED/CANCELLED/OOM, 2 on missing
  inputs / arg errors, 3 on max-cycles exhausted.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"

DEFAULT_POLL_INTERVAL_SEC = 60
DEFAULT_MAX_WAIT_MIN = 1320  # 22h ceiling per cycle (above the 20h SLURM walltime cap)
DEFAULT_MAX_CYCLES = 1


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("run_slurm_job")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[run_slurm_job] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


# ---------- shell + env ----------


def _resolve_bash() -> str:
    found = shutil.which("bash")
    if found:
        return found
    for c in ("/usr/bin/bash", "/bin/bash"):
        if Path(c).exists():
            return c
    raise RuntimeError("bash not found on PATH")


BASH = _resolve_bash()
_USER_LOCAL_BIN = str(Path.home() / ".local" / "bin")
_PATH = ":".join([
    _USER_LOCAL_BIN, "/opt/slurm/bin", "/usr/local/bin", "/usr/bin",
    "/bin", os.environ.get("PATH", ""),
])
_BASE_ENV = {**os.environ, "PATH": _PATH}


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True,
        env=env if env is not None else _BASE_ENV,
    )


def _parse_env_arg(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--env value must be KEY=VALUE, got {s!r}"
        )
    k, v = s.split("=", 1)
    if not k:
        raise argparse.ArgumentTypeError(f"--env key empty in {s!r}")
    return k, v


# ---------- submit ----------


def submit_job(sbatch_path: Path, env_overrides: dict[str, str]) -> str:
    """Submit sbatch with env-var overrides; return job_id."""
    sbatch_path = Path(sbatch_path).resolve()
    if not sbatch_path.is_file():
        raise FileNotFoundError(f"sbatch script missing: {sbatch_path}")

    # Build env-prefixed shell command:  KEY1=VAL1 KEY2=VAL2 sbatch <path>
    env_str = " ".join(f"{k}={v}" for k, v in env_overrides.items())
    submit_cmd = f"{env_str} sbatch {sbatch_path}".strip()
    logger.info(f"submit cmd: {submit_cmd}")

    proc = _run([BASH, "-c", submit_cmd])
    if proc.returncode != 0:
        logger.info(f"  [stderr] {proc.stderr}")
        raise RuntimeError(f"sbatch failed with exit {proc.returncode}")
    out = proc.stdout.strip()
    logger.info(f"  {out}")
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(f"could not parse job_id from sbatch output: {out!r}")
    return m.group(1)


# ---------- poll ----------


def _squeue_state(job_id: str) -> tuple[str, str] | None:
    """Return (state, elapsed) tuple or None if job is no longer in the queue."""
    proc = _run([BASH, "-c", f"squeue -h -j {job_id} -o '%T %M'"])
    line = proc.stdout.strip()
    if not line:
        return None
    parts = line.split(None, 1)
    state = parts[0]
    elapsed = parts[1] if len(parts) > 1 else ""
    return state, elapsed


def _find_running_job_for_sbatch(sbatch_path: Path, env_overrides: dict[str, str]) -> str | None:
    """Return job_id of an existing RUNNING/PENDING job matching this sbatch + env, or None.

    Matches by job name (from sbatch --job-name) AND on out_dir env var (so different
    ablations of the same sbatch don't collide). Prevents duplicate submissions.
    """
    job_name = None
    try:
        for line in sbatch_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#SBATCH --job-name="):
                job_name = line.split("=", 1)[1].strip()
                break
    except OSError:
        return None
    if not job_name:
        return None
    user = os.environ.get("USER", "")
    proc = _run([BASH, "-c", f"squeue -h -u {user} -n {job_name} -o '%i'"])
    candidate_ids = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    if not candidate_ids:
        return None
    # If env_overrides has OUT_DIR / OUT_ROOT / ABLATION, narrow by matching submit log
    discriminator_keys = ("OUT_DIR", "OUT_ROOT", "ABLATION")
    discriminators = {k: v for k, v in env_overrides.items() if k in discriminator_keys}
    if not discriminators:
        return candidate_ids[0]
    for jid in candidate_ids:
        log_path = LOGS_DIR / f"{job_name}_{jid}.log"
        if not log_path.is_file():
            continue
        try:
            content = log_path.read_text(errors="replace")
        except OSError:
            continue
        if all(f"{k}={v}" in content or f"{k.lower()}" in content.lower() and v in content for k, v in discriminators.items()):
            return jid
    # Conservative: if any candidate exists with matching name, assume duplicate
    return candidate_ids[0]


def _sacct_final_state(job_id: str) -> str:
    proc = _run([BASH, "-c", f"sacct -j {job_id} --format=State -n -P | head -1"])
    return proc.stdout.strip()


_TERMINAL_FAIL_STATES = ("FAILED", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL")


def poll_job(
    job_id: str,
    *,
    poll_interval_sec: int = DEFAULT_POLL_INTERVAL_SEC,
    max_wait_min: int = DEFAULT_MAX_WAIT_MIN,
) -> str:
    """Block until job leaves the queue. Return the sacct final state.

    Raises TimeoutError if max_wait_min exceeded.
    """
    start = time.time()
    last_state = ""
    while True:
        s = _squeue_state(job_id)
        if s is None:
            final = _sacct_final_state(job_id)
            elapsed_min = (time.time() - start) / 60
            logger.info(
                f"  job {job_id} left queue after {elapsed_min:.1f}min  sacct_state={final}"
            )
            return final

        state, elapsed = s
        if state != last_state:
            logger.info(
                f"  [{int((time.time() - start) / 60)}min] job {job_id}  "
                f"state={state}  elapsed={elapsed}"
            )
            last_state = state

        if (time.time() - start) / 60 > max_wait_min:
            raise TimeoutError(
                f"exceeded {max_wait_min}min wait ceiling for job {job_id}"
            )
        time.sleep(poll_interval_sec)


# ---------- main run-cycle loop ----------


def run(
    sbatch_path: Path,
    env_overrides: dict[str, str],
    *,
    poll: bool = True,
    check_existing: bool = False,
    resume_on_timeout: bool = False,
    max_cycles: int = DEFAULT_MAX_CYCLES,
    poll_interval_sec: int = DEFAULT_POLL_INTERVAL_SEC,
    max_wait_min: int = DEFAULT_MAX_WAIT_MIN,
) -> int:
    """Submit + (optionally) poll + (optionally) resubmit on TIMEOUT.

    Returns process-exit-code: 0 on COMPLETED, 1 on terminal failure,
    3 on max-cycles exhausted without success.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    cycle = 0
    last_job_id = ""
    while cycle < max_cycles:
        cycle += 1
        # Check for in-flight duplicate before submitting
        if check_existing:
            existing = _find_running_job_for_sbatch(sbatch_path, env_overrides)
            if existing:
                logger.info(
                    f"cycle {cycle}/{max_cycles}: existing job {existing} "
                    f"already running for {sbatch_path.name}; skipping submission"
                )
                if not poll:
                    return 0
                last_job_id = existing
                # Fall through to poll the existing one
                final = poll_job(
                    last_job_id,
                    poll_interval_sec=poll_interval_sec,
                    max_wait_min=max_wait_min,
                )
                if final.startswith("COMPLETED"):
                    return 0
                if any(s in final for s in _TERMINAL_FAIL_STATES):
                    return 1
                if "TIMEOUT" in final and resume_on_timeout and cycle < max_cycles:
                    continue
                return 1
        logger.info("=" * 60)
        logger.info(
            f"cycle {cycle}/{max_cycles}: submitting {sbatch_path.name}"
        )
        logger.info("=" * 60)
        last_job_id = submit_job(sbatch_path, env_overrides)
        if not poll:
            logger.info(f"  --no-poll: returning after submission, job_id={last_job_id}")
            return 0

        try:
            final = poll_job(
                last_job_id,
                poll_interval_sec=poll_interval_sec,
                max_wait_min=max_wait_min,
            )
        except TimeoutError as e:
            logger.info(f"  poll timeout: {e}")
            return 1

        if final.startswith("COMPLETED"):
            logger.info(f"  cycle {cycle} COMPLETED — done")
            return 0
        if "TIMEOUT" in final:
            if resume_on_timeout and cycle < max_cycles:
                logger.info(
                    f"  cycle {cycle} TIMEOUT — resubmitting "
                    f"(checkpointed work resumes; cycle {cycle + 1}/{max_cycles})"
                )
                continue
            logger.info(
                f"  cycle {cycle} TIMEOUT — resume disabled or max-cycles reached"
            )
            return 3
        if any(s in final for s in _TERMINAL_FAIL_STATES):
            logger.info(f"  cycle {cycle} terminal failure: {final}")
            return 1
        logger.info(f"  cycle {cycle} unknown final state: {final!r}")
        return 1

    logger.info(f"max_cycles={max_cycles} exhausted; last_job_id={last_job_id}")
    return 3


# ---------- CLI ----------


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reproducible SLURM submission + polling wrapper.",
    )
    ap.add_argument(
        "--sbatch",
        type=Path,
        help="Path to .sbatch script to submit (omit when --poll-only).",
    )
    ap.add_argument(
        "--env",
        type=_parse_env_arg,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable to set for the sbatch command (repeatable).",
    )
    ap.add_argument(
        "--no-poll",
        action="store_true",
        help="Submit and exit; don't wait for completion.",
    )
    ap.add_argument(
        "--resume-on-timeout",
        action="store_true",
        help="On TIMEOUT, resubmit identical command (checkpointed work resumes).",
    )
    ap.add_argument(
        "--max-cycles",
        type=int,
        default=DEFAULT_MAX_CYCLES,
        help=f"Max submit cycles when --resume-on-timeout (default {DEFAULT_MAX_CYCLES}).",
    )
    ap.add_argument(
        "--poll-interval-sec",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SEC,
    )
    ap.add_argument(
        "--max-wait-min",
        type=int,
        default=DEFAULT_MAX_WAIT_MIN,
        help=f"Per-cycle wait ceiling in minutes (default {DEFAULT_MAX_WAIT_MIN}).",
    )
    ap.add_argument(
        "--check-existing",
        action="store_true",
        help="Before submitting, check for running job with same sbatch name + env; skip if found",
    )
    ap.add_argument(
        "--poll-only",
        action="store_true",
        help="Skip submission; poll an existing job specified by --job-id.",
    )
    ap.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Required with --poll-only.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.poll_only:
        if not args.job_id:
            logger.info("--poll-only requires --job-id")
            return 2
        try:
            final = poll_job(
                args.job_id,
                poll_interval_sec=args.poll_interval_sec,
                max_wait_min=args.max_wait_min,
            )
        except TimeoutError as e:
            logger.info(f"poll timeout: {e}")
            return 1
        if final.startswith("COMPLETED"):
            return 0
        if any(s in final for s in _TERMINAL_FAIL_STATES + ("TIMEOUT",)):
            return 1
        return 1

    if not args.sbatch:
        logger.info("--sbatch is required (or use --poll-only)")
        return 2

    env_overrides = dict(args.env)
    return run(
        args.sbatch,
        env_overrides,
        poll=not args.no_poll,
        check_existing=args.check_existing,
        resume_on_timeout=args.resume_on_timeout,
        max_cycles=args.max_cycles,
        poll_interval_sec=args.poll_interval_sec,
        max_wait_min=args.max_wait_min,
    )


if __name__ == "__main__":
    sys.exit(main())
