"""Clean citation leakage from corpus_chunks_enriched.jsonl `text` field.

Strips citations, case names, supra/id refs from every chunk's text via
clean_query.clean_destination_context. Preserves opinion_id, cluster_id,
chunk_index. Writes summary JSON with sha256 + git_sha provenance.

Supports parallel execution via --workers (multiprocessing.Pool with
imap_unordered + chunksize tuning). Order is NOT preserved when workers > 1
to maximize throughput; downstream consumers (BM25/BGE) are order-agnostic.

Usage
-----
    uv run python scripts/clean_corpus.py \\
        --in-path data/processed/baseline/corpus_chunks_enriched.jsonl \\
        --out-path data/processed/baseline/corpus_chunks_cleaned.jsonl \\
        --workers 32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
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


def main(
    in_path: Path,
    out_path: Path,
    *,
    workers: int = 1,
    log_every: int = 100_000,
    chunksize: int = 256,
) -> dict:
    """Clean text field on every chunk row, stream JSONL → JSONL atomically.

    workers=1 (default): serial, deterministic order.
    workers>1: parallel via mp.Pool.imap_unordered; order NOT preserved.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    if not in_path.exists():
        raise FileNotFoundError(f"missing input: {in_path}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Cleaning corpus: {in_path}")
    logger.info(f"Output:          {out_path}")
    logger.info(f"Workers:         {workers}")
    logger.info("=" * 60)

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    n_total = 0

    if workers == 1:
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
    else:
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
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print("[clean_corpus] DRY RUN")
        print(f"  in_path  : {args.in_path}")
        print(f"  out_path : {args.out_path}")
        print(f"  workers  : {args.workers}")
        print(f"  cpu_count: {os.cpu_count()}")
        print(f"  git_sha  : {_git_sha()}")
        sys.exit(0)
    try:
        main(in_path=args.in_path, out_path=args.out_path, workers=args.workers)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
