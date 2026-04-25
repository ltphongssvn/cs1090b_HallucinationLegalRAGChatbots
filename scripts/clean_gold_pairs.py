"""Clean LePaRD destination_context in gold-pair JSONL files.

Strips citation strings, case names, supra/id references, and LePaRD
artifacts via scripts.clean_query.clean_destination_context. Writes
cleaned JSONL to --out-dir alongside a clean_gold_pairs_summary.json
manifest with input/output SHA256 hashes + git_sha.

Usage
-----
    uv run python scripts/clean_gold_pairs.py \\
        --in-dir data/processed/baseline \\
        --out-dir data/processed/baseline \\
        --files gold_pairs_test.jsonl gold_pairs_val.jsonl

    uv run python scripts/clean_gold_pairs.py --dry-run ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clean_query import clean_jsonl_field  # noqa: E402


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("clean_gold_pairs")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[clean_gold_pairs] %(message)s"))
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(
    in_dir: Path,
    out_dir: Path,
    files: list[str],
    field: str = "destination_context",
) -> dict:
    """Clean each named JSONL file in ``in_dir``, write to ``out_dir``."""
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    for fname in files:
        if not (in_dir / fname).exists():
            raise FileNotFoundError(f"missing {in_dir / fname}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Cleaning {len(files)} gold-pair file(s)")
    logger.info("=" * 60)

    input_sha: dict[str, str] = {}
    output_sha: dict[str, str] = {}
    total = 0
    for fname in files:
        src = in_dir / fname
        dst = out_dir / fname
        input_sha[fname] = _sha256(src)
        logger.info(f"  cleaning {fname} ({src.stat().st_size / 1e6:.1f} MB)")
        n = clean_jsonl_field(src, dst, field=field)
        output_sha[fname] = _sha256(dst)
        total += n
        logger.info(f"    {n:,} rows -> {dst}")

    summary = {
        "files_processed": list(files),
        "field_cleaned": field,
        "total_rows_cleaned": total,
        "input_sha256": input_sha,
        "output_sha256": output_sha,
        "git_sha": _git_sha(),
    }
    summary_path = out_dir / "clean_gold_pairs_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(f"  wrote {summary_path}")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Clean LePaRD gold-pair JSONLs.")
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="JSONL filenames inside --in-dir to clean",
    )
    ap.add_argument(
        "--field",
        type=str,
        default="destination_context",
        help="Field to clean (default: destination_context)",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print("[clean_gold_pairs] DRY RUN")
        print(f"  in_dir   : {args.in_dir}")
        print(f"  out_dir  : {args.out_dir}")
        print(f"  files    : {args.files}")
        print(f"  field    : {args.field}")
        print(f"  git_sha  : {_git_sha()}")
        sys.exit(0)
    try:
        main(
            in_dir=args.in_dir,
            out_dir=args.out_dir,
            files=args.files,
            field=args.field,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
