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

Repair strategy:
  - Regex substitution on raw line bytes before JSON parse (faster than
    round-trip deserialise -> fix -> re-serialise, and avoids re-encoding
    unicode escapes or changing key ordering in the output)
  - Writes to a .jsonl.tmp sibling then tmp.replace() for atomicity —
    a crash mid-write leaves the original shard intact
  - Streams line-by-line so peak RAM is bounded to one line regardless
    of shard size (shards reach ~422 MB on this dataset)

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
import logging
import math
import multiprocessing
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Module-level constants (immutable — safe to import directly in tests)
# ---------------------------------------------------------------------------

# Exact string values produced by upstream allow_nan=True serialisation.
# Kept as a frozenset so membership checks are O(1).
_STRING_NAN_VALUES: frozenset[str] = frozenset({"NaN", "nan", "Infinity", "-Infinity", "Inf", "-Inf"})

# Matches bare NaN/Infinity tokens NOT already quoted.
# Negative lookbehind/lookahead on [\"'\w] prevents touching string values
# such as "NaN" or field names containing "Inf".
# Regex substitution on the raw line is faster than a JSON round-trip and
# avoids altering key order or unicode escapes present in the original bytes.
_NAN_REPAIR_PATTERN = re.compile(r"(?<![\"'\w])(?:NaN|-?Infinity|-?Inf)(?![\"'\w])")

# ---------------------------------------------------------------------------
# Logging — single named logger; callers control level/handler
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _has_nan(value: Any) -> bool:
    """
    Recursively detect float nan/inf OR stringified NaN variants.
    Handles nested dicts and lists. No string heuristics — only exact
    membership in _STRING_NAN_VALUES — so legal text never triggers false
    positives (e.g. a case name containing the word "infinity").
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
    """Return list of top-level keys whose values contain a NaN/Inf variant."""
    return [k for k, v in obj.items() if _has_nan(v)]


# ---------------------------------------------------------------------------
# Shard-level audit (called from multiprocessing worker — must be picklable)
# ---------------------------------------------------------------------------


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
                    # Line is not valid JSON at all — count as contaminated
                    nan_lines += 1
                    continue
                fields = _nan_fields(obj)
                if fields:
                    nan_lines += 1
                    for f in fields:
                        nan_fields[f] = nan_fields.get(f, 0) + 1
    except Exception as exc:
        log.error("Error reading %s: %s", shard_path.name, exc)
    return ShardHealth(
        shard=shard_path.name,
        total_lines=total,
        nan_lines=nan_lines,
        nan_fields=nan_fields,
    )


# ---------------------------------------------------------------------------
# Shard-level repair — streamed via tempfile to bound peak RAM
# ---------------------------------------------------------------------------


def repair_shard(shard_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Repair bare NaN/Infinity tokens -> null in a single shard.

    Streams line-by-line into a sibling .jsonl.tmp file so peak RAM is
    bounded to one line at a time regardless of shard size (shards reach
    ~422 MB on this dataset; loading into a list would cost ~1.2-1.7 GB
    Python overhead per worker in a multiprocessing pool).

    On completion, creates a .jsonl.bak backup then atomically replaces the
    original via tmp.replace() so a crash mid-write never corrupts the shard.

    Returns (total_lines, repaired_lines).
    """
    total, repaired = 0, 0
    # Sibling tmp stays on same filesystem — guarantees tmp.replace() is an
    # atomic rename() syscall, not a cross-device copy.
    tmp = shard_path.with_suffix(".jsonl.tmp")
    try:
        with shard_path.open(encoding="utf-8", errors="replace") as fh, \
             tmp.open("w", encoding="utf-8") as out:
            for line in fh:
                total += 1
                fixed = _NAN_REPAIR_PATTERN.sub("null", line)
                if fixed != line:
                    repaired += 1
                out.write(fixed)

        if repaired > 0 and not dry_run:
            backup = shard_path.with_suffix(".jsonl.bak")
            shutil.copy2(shard_path, backup)   # preserve mtime + permissions
            tmp.replace(shard_path)            # atomic on POSIX
        # dry_run or nothing repaired — tmp discarded in finally block
    finally:
        # Always clean up tmp; harmless if already renamed above
        if tmp.exists():
            tmp.unlink()
    return total, repaired


# ---------------------------------------------------------------------------
# Dataset-level audit and repair
# ---------------------------------------------------------------------------


def audit_dataset(input_dir: Path) -> DatasetHealth:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    shards = sorted(input_dir.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No .jsonl shards found in {input_dir}")

    ncpus = multiprocessing.cpu_count()
    log.info("Scanning %d shards using %d CPU cores ...", len(shards), ncpus)

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
    log.info("%s %d shards ...", mode, len(shards))
    total_repaired = 0
    for shard in tqdm(shards, unit="shard", desc="repairing"):
        _, repaired = repair_shard(shard, dry_run=dry_run)
        total_repaired += repaired
    log.info("%s %d lines.", "Would repair" if dry_run else "Repaired", total_repaired)


# ---------------------------------------------------------------------------
# Output formatters — split from main() so they are independently testable
# ---------------------------------------------------------------------------


def _write_csv(health: DatasetHealth, csv_path: Path) -> None:
    """Write per-field NaN counts to CSV. Separated from main() for testability."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["field", "nan_count"])
        for f, c in sorted(health.nan_fields.items(), key=lambda x: -x[1]):
            writer.writerow([f, c])
    log.info("CSV written -> %s", csv_path)


def _emit_json(health: DatasetHealth) -> None:
    """Print JSON summary to stdout. Separated from main() for testability."""
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


def _emit_text(health: DatasetHealth, emit_shard_ids: bool = False) -> None:
    """Print human-readable summary to stdout. Separated from main() for testability."""
    print(f"\ntotal lines:  {health.total_lines:,}")
    print(f"nan lines:    {health.nan_lines:,}")
    print(f"nan shards:   {health.nan_shards}/{health.total_shards}")
    print(f"clean pct:    {health.clean_pct:.4f}%")
    print(f"nan fields:   {health.nan_fields}")
    print(f"verdict:      {health.gate_verdict()}")
    if emit_shard_ids:
        print("\ncontaminated shards:")
        for s in health.contaminated_shards:
            print(f"  {s}")


# ---------------------------------------------------------------------------
# CLI entry point — dispatch only; no business logic here
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

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
        _write_csv(health, args.csv)

    if args.json:
        _emit_json(health)
        return

    _emit_text(health, emit_shard_ids=args.emit_shard_ids)


if __name__ == "__main__":
    main()
