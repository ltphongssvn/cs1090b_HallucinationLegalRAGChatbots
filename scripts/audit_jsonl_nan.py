"""
scripts/audit_jsonl_nan.py
--------------------------
Audits JSONL shards for bare NaN/Infinity tokens that pass Python's
json.loads but are rejected by Polars' strict tape parser.

Detection strategy:
  - Float nan/inf: json.loads + recursive math.isnan / math.isinf walk
  - Stringified NaN: recursive walk checking str values in _STRING_NAN_VALUES
  - No string heuristics — zero false positives on legal text
  - Contamination split into typed counters: nonfinite_lines,
    string_sentinel_lines, decode_error_lines for honest gate verdicts

Repair strategy:
  - Semantic: json.loads (with parse_constant intercept) -> recursive
    replace -> json.dumps(allow_nan=False). Quote-context safe — legal
    text containing NaN/Infinity inside strings is never modified.
  - Regex was rejected: it is not quote-context aware and corrupts legal
    strings containing spaced NaN/Infinity tokens (confirmed in testing).
  - Streams line-by-line into .jsonl.tmp then atomic rename — peak RAM
    is O(1) regardless of shard size (shards reach ~422 MB on this dataset).
  - Repair is idempotent: a clean shard touched twice changes 0 lines.

Configuration:
  - AuditSettings (pydantic-settings): advisory_fields, workers, etc.
    Override via AUDIT_* env vars.
  - load_audit_config(): load advisory_fields from a YAML file via OmegaConf.

Telemetry:
  - log_health_to_wandb(): log DatasetHealth fields to a W&B run (offline safe).

Usage
-----
    python scripts/audit_jsonl_nan.py
    python scripts/audit_jsonl_nan.py --input-dir data/raw/cl_federal_appellate_bulk
    python scripts/audit_jsonl_nan.py --json
    python scripts/audit_jsonl_nan.py --emit-shard-ids
    python scripts/audit_jsonl_nan.py --csv logs/nan_audit.csv
    python scripts/audit_jsonl_nan.py --fix
    python scripts/audit_jsonl_nan.py --fix --dry-run
    python scripts/audit_jsonl_nan.py --wandb
    python scripts/audit_jsonl_nan.py --workers 4
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import Field
from pydantic_settings import BaseSettings
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Module-level constants (immutable — safe to import directly in tests)
# ---------------------------------------------------------------------------

# Exact string values produced by upstream allow_nan=True serialisation.
# Kept as a frozenset so membership checks are O(1).
_STRING_NAN_VALUES: frozenset[str] = frozenset(
    {"NaN", "nan", "Infinity", "-Infinity", "Inf", "-Inf"}
)

# Retained for regex contract tests — NOT used in repair path.
# Repair uses semantic parse->walk->reserialize instead of regex because
# the regex is not quote-context aware: tokens surrounded by spaces inside
# quoted strings are corrupted (confirmed in testing 2026-04).
_NAN_REPAIR_PATTERN = re.compile(r"(?<![\"'\w])(?:NaN|-?Infinity|-?Inf)(?![\"'\w])")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — pydantic-settings + OmegaConf
# ---------------------------------------------------------------------------


class AuditSettings(BaseSettings):
    """Runtime configuration. Override any field via AUDIT_* env vars."""

    input_dir: Path = Path("data/raw/cl_federal_appellate_bulk")
    # Advisory fields: NaN here is REPAIRABLE, not a HARD_FAILURE
    advisory_fields: frozenset[str] = frozenset(
        {"case_name", "raw_text", "cleaning_flags"}
    )
    string_nan_values: frozenset[str] = _STRING_NAN_VALUES
    workers: int = Field(default=4, gt=0)
    dry_run: bool = False

    model_config = {"env_prefix": "AUDIT_"}


def load_audit_config(config_path: Path) -> AuditSettings:
    """
    Load advisory_fields (and other overrides) from a YAML file via OmegaConf,
    then merge into AuditSettings. Allows externalising policy from source.
    """
    raw: DictConfig = OmegaConf.load(config_path)
    overrides: dict[str, Any] = OmegaConf.to_container(raw, resolve=True)  # type: ignore[assignment]
    if "advisory_fields" in overrides:
        overrides["advisory_fields"] = frozenset(overrides["advisory_fields"])
    return AuditSettings(**overrides)


# ---------------------------------------------------------------------------
# Data classes — typed contamination counters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardHealth:
    shard: str
    total_lines: int
    nan_lines: int           # total contaminated lines (backward compat)
    nan_fields: dict[str, int]
    nonfinite_lines: int = 0       # float nan/inf
    string_sentinel_lines: int = 0 # stringified "NaN" / "Infinity" etc.
    decode_error_lines: int = 0    # JSONDecodeError — separate failure class


@dataclass(frozen=True)
class DatasetHealth:
    total_lines: int
    nan_lines: int
    nan_shards: int
    total_shards: int
    nan_fields: dict[str, int]
    contaminated_shards: list[str]
    nonfinite_lines: int = 0
    string_sentinel_lines: int = 0
    decode_error_lines: int = 0

    @property
    def clean_pct(self) -> float:
        return (
            100.0 * (self.total_lines - self.nan_lines) / self.total_lines
            if self.total_lines
            else 0.0
        )

    def gate_verdict(self, advisory: frozenset[str] | None = None) -> str:
        """
        Classify contamination as CLEAN, REPAIRABLE, HARD_FAILURE, or
        PARSE_FAILURE.

        Bug fix: empty nan_fields with nan_lines > 0 previously returned
        REPAIRABLE via vacuous-truth all(). Now correctly returns PARSE_FAILURE
        when decode errors are present but no field names were recorded.
        """
        _advisory = advisory or frozenset({"case_name", "raw_text", "cleaning_flags"})

        if self.nan_lines == 0:
            return "CLEAN"

        # Decode errors with no field mapping — cannot be classified as advisory
        if self.decode_error_lines > 0 and not self.nan_fields:
            return "PARSE_FAILURE — malformed JSON lines; manual inspection required"

        if self.nan_fields and all(f in _advisory for f in self.nan_fields):
            return "REPAIRABLE — NaN only in advisory fields; does not block Stage 3"

        return "HARD_FAILURE — NaN in required fields; blocks Stage 3 pipeline"


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _has_nan(value: Any) -> bool:
    """
    Recursively detect float nan/inf OR stringified NaN variants.
    Only exact membership in _STRING_NAN_VALUES — no substring heuristics —
    so legal text never triggers false positives.
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


def _is_nonfinite(value: Any) -> bool:
    """True only for float nan/inf — not string sentinels."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    if isinstance(value, dict):
        return any(_is_nonfinite(v) for v in value.values())
    if isinstance(value, list):
        return any(_is_nonfinite(v) for v in value)
    return False


def _is_string_sentinel(value: Any) -> bool:
    """True only for string sentinel values — not float nan/inf."""
    if isinstance(value, str) and value in _STRING_NAN_VALUES:
        return True
    if isinstance(value, dict):
        return any(_is_string_sentinel(v) for v in value.values())
    if isinstance(value, list):
        return any(_is_string_sentinel(v) for v in value)
    return False


# ---------------------------------------------------------------------------
# Shard-level audit
# ---------------------------------------------------------------------------


def audit_shard(shard_path: Path) -> ShardHealth:
    """Scan a single shard. Safe to call from multiprocessing worker."""
    total = nan_lines = nonfinite = sentinel = decode_err = 0
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
                    # Separate counter — not the same failure class as NaN
                    nan_lines += 1
                    decode_err += 1
                    continue
                fields = _nan_fields(obj)
                if fields:
                    nan_lines += 1
                    if _is_nonfinite(obj):
                        nonfinite += 1
                    if _is_string_sentinel(obj):
                        sentinel += 1
                    for f in fields:
                        nan_fields[f] = nan_fields.get(f, 0) + 1
    except Exception as exc:
        log.error("Error reading %s: %s", shard_path.name, exc)
    return ShardHealth(
        shard=shard_path.name,
        total_lines=total,
        nan_lines=nan_lines,
        nan_fields=nan_fields,
        nonfinite_lines=nonfinite,
        string_sentinel_lines=sentinel,
        decode_error_lines=decode_err,
    )


# ---------------------------------------------------------------------------
# Shard-level repair — semantic, quote-context safe, streaming
# ---------------------------------------------------------------------------


def _replace_nonfinite(obj: Any) -> Any:
    """
    Recursively replace float nan/inf with None.
    Does NOT replace string sentinels — parse_constant already handles
    bare tokens; quoted "NaN" strings are intentional string values.
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _replace_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_nonfinite(v) for v in obj]
    return obj


def _semantic_repair_line(line: str) -> tuple[str, bool]:
    """
    Repair a single raw JSON line via parse -> walk -> reserialize.

    parse_constant intercepts bare NaN/Infinity during json.loads and
    converts them to float('nan') so _replace_nonfinite can nullify them.
    json.dumps(allow_nan=False) guarantees strict JSON output.

    Returns (repaired_line, was_changed).
    Raises json.JSONDecodeError for truly malformed lines (not NaN-related).
    """
    def _intercept(token: str) -> float:
        # parse_constant fires for bare NaN / Infinity / -Infinity tokens
        return float("nan") if "nan" in token.lower() else float("inf")

    obj = json.loads(line, parse_constant=_intercept)
    cleaned = _replace_nonfinite(obj)
    repaired = json.dumps(cleaned, allow_nan=False)
    return repaired + "\n", repaired != line.rstrip("\n")


def repair_shard(shard_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Repair bare NaN/Infinity tokens -> null in a single shard.

    Uses semantic parse->walk->reserialize (not regex) so quoted legal
    strings containing NaN/Infinity are never modified.

    Streams line-by-line into a .jsonl.tmp sibling — peak RAM is O(1).
    Atomic rename on completion; .bak backup preserved before overwrite.
    Idempotent: a clean shard repaired twice changes 0 lines.

    Returns (total_lines, repaired_lines).
    """
    total, repaired = 0, 0
    tmp = shard_path.with_suffix(".jsonl.tmp")
    try:
        with shard_path.open(encoding="utf-8", errors="replace") as fh, \
             tmp.open("w", encoding="utf-8") as out:
            for raw_line in fh:
                total += 1
                try:
                    fixed, changed = _semantic_repair_line(raw_line.rstrip("\n"))
                    if changed:
                        repaired += 1
                    out.write(fixed)
                except json.JSONDecodeError:
                    # Malformed line — pass through unchanged
                    out.write(raw_line)

        if repaired > 0 and not dry_run:
            backup = shard_path.with_suffix(".jsonl.bak")
            shutil.copy2(shard_path, backup)
            tmp.replace(shard_path)   # atomic on POSIX
    finally:
        if tmp.exists():
            tmp.unlink()
    return total, repaired


# ---------------------------------------------------------------------------
# Dataset-level audit and repair
# ---------------------------------------------------------------------------


def audit_dataset(input_dir: Path, workers: int | None = None) -> DatasetHealth:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    shards = sorted(input_dir.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No .jsonl shards found in {input_dir}")

    ncpus = workers or multiprocessing.cpu_count()
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
    nonfinite = sentinel = decode_err = 0
    all_nan_fields: dict[str, int] = {}
    contaminated: list[str] = []

    for h in results:
        total_lines += h.total_lines
        nan_lines += h.nan_lines
        nonfinite += h.nonfinite_lines
        sentinel += h.string_sentinel_lines
        decode_err += h.decode_error_lines
        if h.nan_lines:
            nan_shards += 1
            contaminated.append(h.shard)
        for f, c in h.nan_fields.items():
            all_nan_fields[f] = all_nan_fields.get(f, 0) + c

    return DatasetHealth(
        total_lines=total_lines,
        nan_lines=nan_lines,
        nan_shards=nan_shards,
        total_shards=len(shards),
        nan_fields=all_nan_fields,
        contaminated_shards=sorted(contaminated),
        nonfinite_lines=nonfinite,
        string_sentinel_lines=sentinel,
        decode_error_lines=decode_err,
    )


def repair_dataset(
    input_dir: Path, dry_run: bool = False, workers: int | None = None
) -> None:
    shards = sorted(input_dir.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No .jsonl shards found in {input_dir}")
    mode = "DRY RUN" if dry_run else "REPAIRING"
    log.info("%s %d shards ...", mode, len(shards))
    total_repaired = 0
    for shard in tqdm(shards, unit="shard", desc="repairing"):
        _, repaired = repair_shard(shard, dry_run=dry_run)
        total_repaired += repaired
    log.info(
        "%s %d lines.",
        "Would repair" if dry_run else "Repaired",
        total_repaired,
    )


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _write_csv(health: DatasetHealth, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["field", "nan_count"])
        for f, c in sorted(health.nan_fields.items(), key=lambda x: -x[1]):
            writer.writerow([f, c])
    log.info("CSV written -> %s", csv_path)


def _emit_json(health: DatasetHealth) -> None:
    print(
        json.dumps(
            {
                "total_lines": health.total_lines,
                "nan_lines": health.nan_lines,
                "nonfinite_lines": health.nonfinite_lines,
                "string_sentinel_lines": health.string_sentinel_lines,
                "decode_error_lines": health.decode_error_lines,
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
    print(f"\ntotal lines:          {health.total_lines:,}")
    print(f"nan lines:            {health.nan_lines:,}")
    print(f"  nonfinite_lines:    {health.nonfinite_lines:,}")
    print(f"  sentinel_lines:     {health.string_sentinel_lines:,}")
    print(f"  decode_error_lines: {health.decode_error_lines:,}")
    print(f"nan shards:           {health.nan_shards}/{health.total_shards}")
    print(f"clean pct:            {health.clean_pct:.4f}%")
    print(f"nan fields:           {health.nan_fields}")
    print(f"verdict:              {health.gate_verdict()}")
    if emit_shard_ids:
        print("\ncontaminated shards:")
        for s in health.contaminated_shards:
            print(f"  {s}")


# ---------------------------------------------------------------------------
# W&B telemetry
# ---------------------------------------------------------------------------


def log_health_to_wandb(
    health: DatasetHealth,
    project: str = "audit-jsonl-nan",
    run_name: str | None = None,
) -> None:
    """
    Log DatasetHealth fields to a W&B run. Safe in offline mode.
    Set WANDB_MODE=offline to avoid network calls in CI/HPC environments.
    """
    import wandb

    run = wandb.init(project=project, name=run_name)
    run.log(
        {
            "data/total_lines": health.total_lines,
            "data/nan_lines": health.nan_lines,
            "data/nonfinite_lines": health.nonfinite_lines,
            "data/string_sentinel_lines": health.string_sentinel_lines,
            "data/decode_error_lines": health.decode_error_lines,
            "data/nan_shards": health.nan_shards,
            "data/total_shards": health.total_shards,
            "data/clean_pct": round(health.clean_pct, 4),
            "data/gate_verdict": health.gate_verdict(),
        }
    )
    run.finish()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/raw/cl_federal_appellate_bulk")
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    parser.add_argument(
        "--emit-shard-ids", action="store_true", help="Print contaminated shard names."
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="Write per-field NaN counts to CSV."
    )
    parser.add_argument(
        "--fix", action="store_true", help="Repair NaN->null in-place with .bak backup."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="With --fix: preview without writing."
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Worker processes (default: cpu_count)."
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Log health metrics to Weights & Biases."
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML config for advisory_fields etc."
    )
    args = parser.parse_args()

    if args.fix:
        repair_dataset(args.input_dir, dry_run=args.dry_run, workers=args.workers)
        return

    health = audit_dataset(args.input_dir, workers=args.workers)

    if args.csv:
        _write_csv(health, args.csv)

    if args.wandb:
        log_health_to_wandb(health)

    if args.json:
        _emit_json(health)
        return

    _emit_text(health, emit_shard_ids=args.emit_shard_ids)


if __name__ == "__main__":
    main()
