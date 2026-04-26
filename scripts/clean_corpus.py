"""Clean citation leakage from corpus_chunks_enriched.jsonl `text` field.

Strips citations, case names, supra/id refs from every chunk's text via
clean_query.clean_destination_context. Preserves opinion_id, cluster_id,
chunk_index. Writes summary JSON with sha256 + git_sha provenance.

Two parallel modes:
  - pool (default): mp.Pool.imap_unordered; main process reads + writes.
    Bottleneck on large corpora due to single-writer serialization.
  - sharded: split input into N contiguous file shards, each worker
    processes its shard independently, results concatenated in order.
    Better for 27GB+ corpora; preserves input order.

Usage
-----
    uv run python scripts/clean_corpus.py \\
        --in-path data/processed/baseline/corpus_chunks_enriched.jsonl \\
        --out-path data/processed/baseline/corpus_chunks_cleaned.jsonl \\
        --workers 32 --mode sharded
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
    """Worker function: parse, clean text, dump.

    Module-level so multiprocessing.Pool can pickle it.
    """
    line = line.rstrip("\n")
    if not line:
        return ""
    row = json.loads(line)
    if "text" in row:
        row["text"] = clean_destination_context(row["text"] or "")
    return json.dumps(row)


def _process_shard(args: tuple[Path, Path]) -> int:
    """Worker: read shard JSONL, write cleaned shard JSONL. Returns row count."""
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
    """Split input JSONL into n_shards contiguous files."""
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


def main(
    in_path: Path,
    out_path: Path,
    *,
    workers: int = 1,
    log_every: int = 100_000,
    chunksize: int = 256,
    mode: str = "pool",
) -> dict:
    """Clean text field on every chunk row, stream JSONL → JSONL atomically.

    workers=1 (default): serial, deterministic order.
    workers>1, mode="pool": parallel via mp.Pool.imap_unordered; order NOT preserved.
    workers>1, mode="sharded": split input → N independent procs → concat in order.
        Best for large corpora where eyecite warmup + IPC dominate.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    if not in_path.exists():
        raise FileNotFoundError(f"missing input: {in_path}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if mode not in ("pool", "sharded"):
        raise ValueError(f"mode must be 'pool' or 'sharded', got {mode!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Cleaning corpus: {in_path}")
    logger.info(f"Output:          {out_path}")
    logger.info(f"Workers:         {workers}")
    logger.info(f"Mode:            {mode}")
    logger.info("=" * 60)

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    n_total = 0

    if workers > 1 and mode == "sharded":
        with tempfile.TemporaryDirectory(dir=out_path.parent) as shard_root:
            shard_dir = Path(shard_root)
            shards = _split_into_shards(in_path, shard_dir, workers)
            logger.info(f"  split into {len(shards)} shards; processing in parallel")
            with mp.Pool(processes=workers) as pool:
                counts = pool.map(_process_shard, shards)
            n_total = sum(counts)
            logger.info(f"  shards complete: {n_total:,} rows total; concatenating")
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
        "input_sha256": _sha256(in_path),
        "output_sha256": _sha256(out_path),
        "git_sha": _git_sha(),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(f"  total: {n_total:,} rows")
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
        choices=["pool", "sharded"],
        default="pool",
        help="pool=mp.Pool.imap_unordered (default); sharded=split→N procs→concat",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print("[clean_corpus] DRY RUN")
        print(f"  in_path  : {args.in_path}")
        print(f"  out_path : {args.out_path}")
        print(f"  workers  : {args.workers}")
        print(f"  mode     : {args.mode}")
        print(f"  cpu_count: {os.cpu_count()}")
        print(f"  git_sha  : {_git_sha()}")
        sys.exit(0)
    try:
        main(
            in_path=args.in_path,
            out_path=args.out_path,
            workers=args.workers,
            mode=args.mode,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
