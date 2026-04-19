"""Subsample corpus_chunks.jsonl to one chunk per opinion (chunk_index=0).

Purpose: reduce 7.8M chunk corpus to ~1.47M unique-opinion entries for the
BGE-M3 MS3 baseline, since full chunk-level encoding is computationally
infeasible on a single L4 within the MS3 timeline. BM25 remains on full corpus;
BGE-M3 baseline operates at opinion-level granularity with MaxP aggregation
degenerating trivially (one chunk → one score per opinion).

This is a deliberate MS3-scope choice. Full 7.8M BGE-M3 is a post-MS3 extension.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def subsample_one_per_opinion(src_path: Path, dst_path: Path) -> tuple[int, int]:
    """Keep only chunk_index=0 per opinion_id. Returns (n_in, n_out).

    Writes via tempfile + os.replace for atomicity. Handles out-of-order streams
    by scanning for chunk_index=0 specifically, not the first row seen for an
    opinion.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")

    n_in = 0
    n_out = 0
    seen_opinions: set[int] = set()
    with tmp.open("w", encoding="utf-8") as fout:
        for row in _iter_jsonl(src_path):
            n_in += 1
            if row.get("chunk_index") != 0:
                continue
            oid = int(row["opinion_id"])
            if oid in seen_opinions:
                continue
            seen_opinions.add(oid)
            fout.write(json.dumps(row, allow_nan=False) + "\n")
            n_out += 1
    os.replace(tmp, dst_path)
    return n_in, n_out


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Subsample corpus to one chunk per opinion (chunk_index=0).")
    ap.add_argument(
        "--corpus-path",
        dest="corpus_path",
        type=Path,
        default=Path("data/processed/baseline/corpus_chunks.jsonl"),
    )
    ap.add_argument(
        "--out-path",
        dest="out_path",
        type=Path,
        default=Path("data/processed/baseline/corpus_chunks_opinion_sample.jsonl"),
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if not args.corpus_path.is_file():
        raise FileNotFoundError(args.corpus_path)
    n_in, n_out = subsample_one_per_opinion(args.corpus_path, args.out_path)
    print(f"read {n_in:,} chunks -> wrote {n_out:,} opinion-sample rows to {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
