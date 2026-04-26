"""Clean citation leakage from corpus_chunks_enriched.jsonl `text` field.

Strips citations, case names, supra/id refs from every chunk's text via
clean_query.clean_destination_context. Preserves opinion_id, cluster_id,
chunk_index. Writes summary JSON with sha256 + git_sha provenance.

Three parallel modes:
  - pool (default): mp.Pool.imap_unordered. Single-writer bottleneck.
  - sharded: split→N mp processes→concat. Hangs on eyecite catastrophic
    backtracking (cpython#96062 — Pool deadlocks on hung workers).
  - subprocess: each shard runs as independent OS subprocess with
    subprocess.run(timeout=N). Hung shards killed cleanly via SIGKILL;
    other shards unaffected. Industry pattern for fault-tolerant batch
    text cleaning at scale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clean_query import clean_destination_context  # noqa: E402

WORKER_SCRIPT = REPO_ROOT / "scripts" / "clean_corpus_worker.py"
DEFAULT_SHARD_TIMEOUT_SEC = 300.0
DEFAULT_ROWS_PER_SHARD = 50_000


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("clean_corpus")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[clean_corpus] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()[:12]
        )
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_one_line(line: str) -> str:
    line = line.rstrip("\n")
    if not line:
        return ""
    row = json.loads(line)
    if "text" in row:
        row["text"] = clean_destination_context(row["text"] or "")
    return json.dumps(row)


def _process_shard(args: tuple[Path, Path]) -> int:
    in_shard, out_shard = args
    n = 0
    with in_shard.open(encoding="utf-8") as fin, out_shard.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            cleaned = _clean_one_line(line)
            if cleaned:
                fout.write(cleaned + "\n")
                n += 1
    return n


def _split_into_shards(
    in_path: Path, shard_dir: Path, n_shards: int
) -> list[tuple[Path, Path]]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    with in_path.open(encoding="utf-8") as f:
        total = sum(1 for _ in f)
    rows_per_shard = (total + n_shards - 1) // n_shards
    shards: list[tuple[Path, Path]] = []
    with in_path.open(encoding="utf-8") as fin:
        for shard_idx in range(n_shards):
            in_shard = shard_dir / f"in_{shard_idx:04d}.jsonl"
            out_shard = shard_dir / f"out_{shard_idx:04d}.jsonl"
            with in_shard.open("w", encoding="utf-8") as fout:
                for _ in range(rows_per_shard):
                    line = fin.readline()
                    if not line:
                        break
                    fout.write(line)
            shards.append((in_shard, out_shard))
    return shards


def _split_by_rows_per_shard(
    in_path: Path, shard_dir: Path, rows_per_shard: int
) -> list[tuple[Path, Path]]:
    """Split input into shards of fixed row count. Returns list of (in, out) shards."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards: list[tuple[Path, Path]] = []
    with in_path.open(encoding="utf-8") as fin:
        shard_idx = 0
        current_in: Path | None = None
        current_fout = None
        rows_in_current = 0
        for line in fin:
            if current_fout is None or rows_in_current >= rows_per_shard:
                if current_fout is not None:
                    current_fout.close()
                in_shard = shard_dir / f"in_{shard_idx:04d}.jsonl"
                out_shard = shard_dir / f"out_{shard_idx:04d}.jsonl"
                shards.append((in_shard, out_shard))
                current_fout = in_shard.open("w", encoding="utf-8")
                rows_in_current = 0
                shard_idx += 1
            current_fout.write(line)
            rows_in_current += 1
        if current_fout is not None:
            current_fout.close()
    return shards


def _run_shard_subprocess(
    in_shard: Path,
    out_shard: Path,
    timeout_sec: float,
) -> tuple[bool, str]:
    """Run worker subprocess on one shard. Returns (success, error_msg)."""
    try:
        subprocess.run(
            [sys.executable, str(WORKER_SCRIPT), str(in_shard), str(out_shard)],
            timeout=timeout_sec,
            check=True,
            capture_output=True,
        )
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_sec}s"
    except subprocess.CalledProcessError as e:
        return False, f"exit {e.returncode}: {e.stderr.decode()[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main(
    in_path: Path,
    out_path: Path,
    *,
    workers: int = 1,
    log_every: int = 100_000,
    chunksize: int = 256,
    mode: str = "pool",
    shard_timeout_sec: float = DEFAULT_SHARD_TIMEOUT_SEC,
    rows_per_shard: int | None = None,
) -> dict:
    """Clean text field on every chunk row, stream JSONL → JSONL atomically.

    workers=1 (default): serial, deterministic.
    workers>1, mode="pool": mp.Pool.imap_unordered.
    workers>1, mode="sharded": split → N mp.Pool procs → concat (deadlock risk).
    workers>1, mode="subprocess": split → N OS subprocesses with timeout
        (recommended for production; fault-tolerant via SIGKILL).
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    if not in_path.exists():
        raise FileNotFoundError(f"missing input: {in_path}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if mode not in ("pool", "sharded", "subprocess"):
        raise ValueError(
            f"mode must be 'pool', 'sharded', or 'subprocess', got {mode!r}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Cleaning corpus: {in_path}")
    logger.info(f"Output:          {out_path}")
    logger.info(f"Workers:         {workers}")
    logger.info(f"Mode:            {mode}")
    logger.info("=" * 60)

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    n_total = 0
    shards_failed = 0
    failed_shard_paths: list[str] = []
    n_shards_used = 0

    if workers > 1 and mode == "subprocess":
        rps = rows_per_shard if rows_per_shard is not None else DEFAULT_ROWS_PER_SHARD
        with tempfile.TemporaryDirectory(dir=out_path.parent) as shard_root:
            shard_dir = Path(shard_root)
            shards = _split_by_rows_per_shard(in_path, shard_dir, rps)
            n_shards_used = len(shards)
            logger.info(
                f"  split into {n_shards_used} shards (rows_per_shard={rps}); "
                f"running {workers}-way subprocess pool with "
                f"shard_timeout={shard_timeout_sec}s"
            )
            # Use ThreadPoolExecutor to manage subprocess concurrency cleanly
            from concurrent.futures import ThreadPoolExecutor, as_completed

            shard_results: dict[Path, tuple[bool, str]] = {}
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {
                    ex.submit(
                        _run_shard_subprocess, in_s, out_s, shard_timeout_sec
                    ): (in_s, out_s)
                    for in_s, out_s in shards
                }
                for fut in as_completed(futures):
                    in_s, out_s = futures[fut]
                    ok, err = fut.result()
                    shard_results[in_s] = (ok, err)
                    if not ok:
                        logger.warning(f"  shard {in_s.name} FAILED: {err}")
                        shards_failed += 1
                        failed_shard_paths.append(str(in_s))

            # Concat successful out shards in original order
            with tmp.open("w", encoding="utf-8") as fout:
                for in_s, out_s in shards:
                    ok, _ = shard_results.get(in_s, (False, "missing"))
                    if not ok or not out_s.exists():
                        continue
                    with out_s.open(encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)
                            n_total += 1
        tmp.rename(out_path)
    elif workers > 1 and mode == "sharded":
        with tempfile.TemporaryDirectory(dir=out_path.parent) as shard_root:
            shard_dir = Path(shard_root)
            shards = _split_into_shards(in_path, shard_dir, workers)
            n_shards_used = len(shards)
            logger.info(f"  split into {n_shards_used} shards")
            with mp.Pool(processes=workers) as pool:
                counts = pool.map(_process_shard, shards)
            n_total = sum(counts)
            with tmp.open("w", encoding="utf-8") as fout:
                for _, out_shard in shards:
                    if out_shard.exists():
                        with out_shard.open(encoding="utf-8") as fin:
                            shutil.copyfileobj(fin, fout)
        tmp.rename(out_path)
    elif workers == 1:
        with in_path.open(encoding="utf-8") as fin, tmp.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                cleaned = _clean_one_line(line)
                if not cleaned:
                    continue
                fout.write(cleaned + "\n")
                n_total += 1
                if n_total % log_every == 0:
                    logger.info(f"  {n_total:,} rows cleaned")
        tmp.rename(out_path)
    else:
        # mode == "pool", workers > 1
        with in_path.open(encoding="utf-8") as fin, tmp.open(
            "w", encoding="utf-8"
        ) as fout, mp.Pool(processes=workers) as pool:
            for cleaned in pool.imap_unordered(
                _clean_one_line, fin, chunksize=chunksize
            ):
                if not cleaned:
                    continue
                fout.write(cleaned + "\n")
                n_total += 1
                if n_total % log_every == 0:
                    logger.info(f"  {n_total:,} rows cleaned")
        tmp.rename(out_path)

    summary = {
        "in_path": str(in_path),
        "out_path": str(out_path),
        "total_rows": n_total,
        "workers": workers,
        "mode": mode,
        "shard_timeout_sec": shard_timeout_sec,
        "n_shards": n_shards_used,
        "shards_failed": shards_failed,
        "failed_shard_paths": failed_shard_paths,
        "input_sha256": _sha256(in_path),
        "output_sha256": _sha256(out_path),
        "git_sha": _git_sha(),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        f"  total: {n_total:,} rows  shards_failed: {shards_failed}/{n_shards_used}"
    )
    logger.info(f"  summary -> {summary_path}")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Clean corpus_chunks JSONL.")
    ap.add_argument("--in-path", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default 1 = serial). Use os.cpu_count() for max.",
    )
    ap.add_argument(
        "--mode",
        choices=["pool", "sharded", "subprocess"],
        default="pool",
        help="pool (default), sharded (mp.Pool deadlock risk), "
        "subprocess (recommended; fault-tolerant via timeout).",
    )
    ap.add_argument(
        "--shard-timeout-sec",
        type=float,
        default=DEFAULT_SHARD_TIMEOUT_SEC,
        help=f"Per-shard subprocess timeout (default {DEFAULT_SHARD_TIMEOUT_SEC}s).",
    )
    ap.add_argument(
        "--rows-per-shard",
        type=int,
        default=None,
        help=f"Rows per shard for subprocess mode (default {DEFAULT_ROWS_PER_SHARD}).",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print("[clean_corpus] DRY RUN")
        print(f"  in_path           : {args.in_path}")
        print(f"  out_path          : {args.out_path}")
        print(f"  workers           : {args.workers}")
        print(f"  mode              : {args.mode}")
        print(f"  shard_timeout_sec : {args.shard_timeout_sec}")
        print(f"  rows_per_shard    : {args.rows_per_shard}")
        print(f"  cpu_count         : {os.cpu_count()}")
        print(f"  git_sha           : {_git_sha()}")
        sys.exit(0)
    try:
        main(
            in_path=args.in_path,
            out_path=args.out_path,
            workers=args.workers,
            mode=args.mode,
            shard_timeout_sec=args.shard_timeout_sec,
            rows_per_shard=args.rows_per_shard,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
