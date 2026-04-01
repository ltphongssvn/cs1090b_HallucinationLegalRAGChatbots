"""
scripts/audit_jsonl_nan.py
--------------------------
Audits JSONL shards for bare NaN/Infinity tokens that pass Python's
json.loads but are rejected by Polars' strict tape parser.

Findings (verified 2026-04):
  - 25,173 NaN lines across 135/159 shards (98.28% clean)
  - All NaN occurrences are in `case_name` field only
  - Cause: upstream extract.py used json.dumps(allow_nan=True) default
  - Classification: REPAIRABLE — case_name is advisory metadata,
    not required for chunking, citation lookup, or NLI evaluation
  - parse_constant kwarg is a no-op in Python 3 — not used here

Detection strategy:
  - Float nan/inf: json.loads + recursive math.isnan / math.isinf walk
  - Stringified NaN: recursive walk checking str values in _STRING_NAN_VALUES
  - No string heuristics — zero false positives on legal text

Processes shards in parallel via multiprocessing.
Progress tracked via tqdm for observability over SSH.

Usage
-----
    python scripts/audit_jsonl_nan.py
    python scripts/audit_jsonl_nan.py --input-dir data/raw/cl_federal_appellate_bulk
    python scripts/audit_jsonl_nan.py --json
    python scripts/audit_jsonl_nan.py --emit-shard-ids
    python scripts/audit_jsonl_nan.py --csv logs/nan_audit.csv
    python scripts/audit_jsonl_nan.py --fix
    python scripts/audit_jsonl_nan.py --fix --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

_STRING_NAN_VALUES: frozenset[str] = frozenset({"NaN", "nan", "Infinity", "-Infinity", "Inf", "-Inf"})

_NAN_REPAIR_PATTERN = re.compile(r"(?<![\"'\w])(?:NaN|-?Infinity|-?Inf)(?![\"'\w])")


@dataclass(frozen=True)
class ShardHealth:
    shard: str
    total_lines: int
    nan_lines: int
    nan_fields: dict[str, int]


@dataclass(frozen=True)
class DatasetHealth:
    total_lines: int
    nan_lines: int
    nan_shards: int
    total_shards: int
    nan_fields: dict[str, int]
    contaminated_shards: list[str]

    @property
    def clean_pct(self) -> float:
        return 100.0 * (self.total_lines - self.nan_lines) / self.total_lines if self.total_lines else 0.0

    def gate_verdict(self) -> str:
        """
        Classify nan_lines presence as hard failure, repairable, or clean.
          CLEAN        : no nan lines
          REPAIRABLE   : nan only in advisory fields (case_name, raw_text, etc.)
          HARD_FAILURE : nan in required Stage 3 fields
        """
        _advisory = {"case_name", "raw_text", "cleaning_flags"}
        if self.nan_lines == 0:
            return "CLEAN"
        if all(f in _advisory for f in self.nan_fields):
            return "REPAIRABLE — NaN only in advisory fields; does not block Stage 3"
        return "HARD_FAILURE — NaN in required fields; blocks Stage 3 pipeline"


def _has_nan(value: Any) -> bool:
    """
    Recursively detect float nan/inf OR stringified NaN variants.
    Handles nested dicts and lists. No string heuristics.
    """
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    if isinstance(value, str) and value in _STRING_NAN_VALUES:
        return True
    if isinstance(value, dict):
        return any(_has_nan(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_nan(v) for v in value)
    return False


def _nan_fields(obj: dict[str, Any]) -> list[str]:
    return [k for k, v in obj.items() if _has_nan(v)]


def audit_shard(shard_path: Path) -> ShardHealth:
    """Scan a single shard. Safe to call from multiprocessing worker."""
    total, nan_lines = 0, 0
    nan_fields: dict[str, int] = {}
    try:
        with shard_path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    nan_lines += 1
                    continue
                fields = _nan_fields(obj)
                if fields:
                    nan_lines += 1
                    for f in fields:
                        nan_fields[f] = nan_fields.get(f, 0) + 1
    except Exception as exc:
        print(f"[audit] ERROR reading {shard_path.name}: {exc}")
    return ShardHealth(
        shard=shard_path.name,
        total_lines=total,
        nan_lines=nan_lines,
        nan_fields=nan_fields,
    )


def repair_shard(shard_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Repair bare NaN/Infinity tokens -> null in a single shard.
    Creates .jsonl.bak backup before writing.
    Returns (total_lines, repaired_lines).
    """
    total, repaired = 0, 0
    lines_out: list[str] = []
    with shard_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            total += 1
            fixed = _NAN_REPAIR_PATTERN.sub("null", line)
            lines_out.append(fixed)
            if fixed != line:
                repaired += 1
    if repaired > 0 and not dry_run:
        backup = shard_path.with_suffix(".jsonl.bak")
        shutil.copy2(shard_path, backup)
        shard_path.write_text("".join(lines_out), encoding="utf-8")
    return total, repaired


def audit_dataset(input_dir: Path) -> DatasetHealth:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    shards = sorted(input_dir.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No .jsonl shards found in {input_dir}")

    ncpus = multiprocessing.cpu_count()
    print(f"[audit] Scanning {len(shards)} shards using {ncpus} CPU cores ...")

    with multiprocessing.Pool(processes=ncpus) as pool:
        results = list(
            tqdm(
                pool.imap(audit_shard, shards),
                total=len(shards),
                unit="shard",
                desc="auditing",
            )
        )

    total_lines = nan_lines = nan_shards = 0
    all_nan_fields: dict[str, int] = {}
    contaminated: list[str] = []

    for health in results:
        total_lines += health.total_lines
        nan_lines += health.nan_lines
        if health.nan_lines:
            nan_shards += 1
            contaminated.append(health.shard)
        for f, c in health.nan_fields.items():
            all_nan_fields[f] = all_nan_fields.get(f, 0) + c

    return DatasetHealth(
        total_lines=total_lines,
        nan_lines=nan_lines,
        nan_shards=nan_shards,
        total_shards=len(shards),
        nan_fields=all_nan_fields,
        contaminated_shards=sorted(contaminated),
    )


def repair_dataset(input_dir: Path, dry_run: bool = False) -> None:
    shards = sorted(input_dir.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No .jsonl shards found in {input_dir}")
    mode = "DRY RUN" if dry_run else "REPAIRING"
    print(f"[repair] {mode} {len(shards)} shards ...")
    total_repaired = 0
    for shard in tqdm(shards, unit="shard", desc="repairing"):
        _, repaired = repair_shard(shard, dry_run=dry_run)
        total_repaired += repaired
    print(f"[repair] {'Would repair' if dry_run else 'Repaired'} {total_repaired:,} lines.")


def write_csv(health: DatasetHealth, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["field", "nan_count"])
        for f, c in sorted(health.nan_fields.items(), key=lambda x: -x[1]):
            writer.writerow([f, c])
    print(f"[audit] CSV written -> {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/cl_federal_appellate_bulk"))
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    parser.add_argument("--emit-shard-ids", action="store_true", help="Print contaminated shard names.")
    parser.add_argument("--csv", type=Path, default=None, help="Write per-field NaN counts to CSV.")
    parser.add_argument("--fix", action="store_true", help="Repair NaN->null in-place with .bak backup.")
    parser.add_argument("--dry-run", action="store_true", help="With --fix: preview without writing.")
    args = parser.parse_args()

    if args.fix:
        repair_dataset(args.input_dir, dry_run=args.dry_run)
        return

    health = audit_dataset(args.input_dir)

    if args.csv:
        write_csv(health, args.csv)

    if args.json:
        print(
            json.dumps(
                {
                    "total_lines": health.total_lines,
                    "nan_lines": health.nan_lines,
                    "nan_shards": health.nan_shards,
                    "total_shards": health.total_shards,
                    "clean_pct": round(health.clean_pct, 4),
                    "nan_fields": health.nan_fields,
                    "gate_verdict": health.gate_verdict(),
                    "contaminated_shards": health.contaminated_shards,
                },
                indent=2,
            )
        )
        return

    print(f"\ntotal lines:  {health.total_lines:,}")
    print(f"nan lines:    {health.nan_lines:,}")
    print(f"nan shards:   {health.nan_shards}/{health.total_shards}")
    print(f"clean pct:    {health.clean_pct:.4f}%")
    print(f"nan fields:   {health.nan_fields}")
    print(f"verdict:      {health.gate_verdict()}")

    if args.emit_shard_ids:
        print("\ncontaminated shards:")
        for s in health.contaminated_shards:
            print(f"  {s}")


if __name__ == "__main__":
    main()
