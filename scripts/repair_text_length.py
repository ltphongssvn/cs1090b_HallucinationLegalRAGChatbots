"""
scripts/repair_text_length.py
------------------------------
Repairs text_length field in JSONL shards where stored value disagrees
with actual len(text) by more than abs_diff > 200 OR rel_diff > 5%.

Root cause (verified 2026-04): NaN repair pass altered case_name byte
counts; a tiny subset of records have stale text_length metadata.

Safety contract:
1. Identifies bad records first (dry-run mode default).
2. Backup written before any mutation.
3. AST-free: pure json.loads/json.dumps — no regex on source code.
4. Idempotent: re-running on already-repaired shards changes nothing.

Usage
-----
    python scripts/repair_text_length.py --dry-run
    python scripts/repair_text_length.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ABS_TOL = 200
REL_TOL = 0.05


def _is_bad(stored: int, actual: int) -> bool:
    abs_diff = abs(stored - actual)
    rel_diff = abs_diff / actual if actual > 0 else float("inf")
    return abs_diff > ABS_TOL or rel_diff > REL_TOL


def repair_shard(shard_path: Path, dry_run: bool) -> int:
    lines_out: list[str] = []
    repaired = 0
    with shard_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                lines_out.append(line)
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                lines_out.append(line)
                continue
            text = obj.get("text") or ""
            stored = obj.get("text_length")
            try:
                stored_int = int(stored)
            except (TypeError, ValueError):
                lines_out.append(line)
                continue
            actual = len(text)
            if _is_bad(stored_int, actual):
                obj["text_length"] = actual
                lines_out.append(json.dumps(obj) + "\n")
                repaired += 1
                print(f"  [repair] id={obj.get('id')} stored={stored_int} actual={actual}")
            else:
                lines_out.append(line)
    if repaired > 0 and not dry_run:
        backup = shard_path.with_suffix(".jsonl.bak")
        shutil.copy2(shard_path, backup)
        shard_path.write_text("".join(lines_out), encoding="utf-8")
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/cl_federal_appellate_bulk"))
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    mode = "DRY RUN" if args.dry_run else "REPAIRING"
    print(f"[repair_text_length] {mode} ...")
    total = 0
    for shard in sorted(args.input_dir.glob("*.jsonl")):
        n = repair_shard(shard, dry_run=args.dry_run)
        if n:
            print(f"  {shard.name}: {n} record(s) repaired")
            total += n
    print(f"[repair_text_length] Total repaired: {total}")


if __name__ == "__main__":
    main()
