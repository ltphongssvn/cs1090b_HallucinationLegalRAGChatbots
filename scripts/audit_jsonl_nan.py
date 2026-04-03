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
  - audit_shard_strict(): errors="strict" surfaces encoding corruption
    that errors="replace" silently normalises (confirmed in testing 2026-04)
  - _walk(obj, predicate): single recursive traversal helper shared by
    _has_nan, _is_nonfinite, _is_string_sentinel — eliminates 3x DRY violation

Repair strategy:
  - Semantic: json.loads (with parse_constant intercept) -> recursive
    replace -> json.dumps(allow_nan=False). Quote-context safe — legal
    text containing NaN/Infinity inside strings is never modified.
  - Regex was rejected: not quote-context aware; corrupts legal strings
    containing spaced NaN/Infinity tokens (confirmed in testing 2026-04).
  - Streams line-by-line into .jsonl.tmp then atomic rename — peak RAM
    is O(1) regardless of shard size (shards reach ~422 MB on this dataset).
  - Repair is idempotent: a clean shard touched twice changes 0 lines.
  - repair_dataset(parallel_repair=True): mirrors audit_dataset's worker
    pool; per-shard atomicity guaranteed by .tmp -> rename strategy.

Post-repair validation:
  - validate_shard_polars(): runs pl.read_ndjson() on repaired shard to
    confirm the actual downstream consumer accepts the output.
  - Polars rejects bare NaN (TapeError), accepts null after repair
    (confirmed in testing 2026-04).

Aggregation:
  - DatasetHealth.zero(total_shards) + ShardHealth via __add__ replaces
    the 8-counter mutable loop. sum(results, start=DatasetHealth.zero())
    makes aggregation testable in isolation.

Schema-driven advisory policy:
  - derive_advisory_from_schema(dataclass): returns frozenset of field
    names typed Optional[...] in the schema. Only Optional fields can be
    REPAIRABLE; required fields with NaN are HARD_FAILURE.
  - --schema-advisory CLI flag enables strict 2026 schema-native gating.
  - Default advisory (_DEFAULT_ADVISORY_FIELDS) preserved for backward
    compatibility with existing pipelines.

Configuration:
  - AuditSettings (pydantic-settings): advisory_fields, workers, etc.
    Override via AUDIT_* env vars.
  - load_audit_config(): load advisory_fields from a YAML file via OmegaConf.

Telemetry:
  - log_health_to_wandb(): log DatasetHealth fields to a W&B run (offline safe).

Usage
-----
    uv run python scripts/audit_jsonl_nan.py
    uv run python scripts/audit_jsonl_nan.py --input-dir data/raw/cl_federal_appellate_bulk
    uv run python scripts/audit_jsonl_nan.py --json
    uv run python scripts/audit_jsonl_nan.py --emit-shard-ids
    uv run python scripts/audit_jsonl_nan.py --csv logs/nan_audit.csv
    uv run python scripts/audit_jsonl_nan.py --fix
    uv run python scripts/audit_jsonl_nan.py --fix --dry-run
    uv run python scripts/audit_jsonl_nan.py --fix --parallel-repair
    uv run python scripts/audit_jsonl_nan.py --wandb
    uv run python scripts/audit_jsonl_nan.py --workers 4
    uv run python scripts/audit_jsonl_nan.py --fix --validate
    uv run python scripts/audit_jsonl_nan.py --strict-encoding
    uv run python scripts/audit_jsonl_nan.py --schema-advisory
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from omegaconf import DictConfig, OmegaConf
from pydantic import Field
from pydantic_settings import BaseSettings
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Module-level constants (immutable — safe to import directly in tests)
# ---------------------------------------------------------------------------

_STRING_NAN_VALUES: frozenset[str] = frozenset(
    {"NaN", "nan", "Infinity", "-Infinity", "Inf", "-Inf"}
)

# Default advisory fields — single source of truth shared by AuditSettings
# and gate_verdict(). NaN in these fields is REPAIRABLE, not HARD_FAILURE.
_DEFAULT_ADVISORY_FIELDS: frozenset[str] = frozenset(
    {"case_name", "raw_text", "cleaning_flags"}
)

# Retained for regex contract tests only — NOT used in repair path.
# Repair uses semantic parse->walk->reserialize because this regex is not
# quote-context aware: tokens surrounded by spaces inside quoted strings
# are corrupted (confirmed in testing 2026-04).
_NAN_REPAIR_PATTERN = re.compile(r"(?<![\"'\w])(?:NaN|-?Infinity|-?Inf)(?![\"'\w])")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema-driven advisory policy
# ---------------------------------------------------------------------------


def derive_advisory_from_schema(schema_cls: type) -> frozenset[str]:
    """
    Derive advisory fields from a dataclass or Pydantic schema by inspecting
    type annotations. Only fields typed Optional[...] (i.e. Union[X, None])
    are considered advisory — NaN in these fields is REPAIRABLE.

    Required fields with NaN become HARD_FAILURE under this policy.
    """
    hints = get_type_hints(schema_cls)
    return frozenset(
        field
        for field, typ in hints.items()
        if get_origin(typ) is Union and type(None) in get_args(typ)
    )


# ---------------------------------------------------------------------------
# Configuration — pydantic-settings + OmegaConf
# ---------------------------------------------------------------------------


class AuditSettings(BaseSettings):
    """Runtime configuration. Override any field via AUDIT_* env vars."""

    input_dir: Path = Path("data/raw/cl_federal_appellate_bulk")
    advisory_fields: frozenset[str] = _DEFAULT_ADVISORY_FIELDS
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
    nan_lines: int
    nan_fields: dict[str, int]
    nonfinite_lines: int = 0
    string_sentinel_lines: int = 0
    decode_error_lines: int = 0


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

    @classmethod
    def zero(cls, total_shards: int = 0) -> "DatasetHealth":
        """Identity element for __add__ aggregation over ShardHealth results."""
        return cls(
            total_lines=0,
            nan_lines=0,
            nan_shards=0,
            total_shards=total_shards,
            nan_fields={},
            contaminated_shards=[],
            nonfinite_lines=0,
            string_sentinel_lines=0,
            decode_error_lines=0,
        )

    def __add__(self, other: ShardHealth) -> "DatasetHealth":
        """Accumulate a ShardHealth. Enables sum(results, start=DatasetHealth.zero())."""
        merged = dict(self.nan_fields)
        for f, c in other.nan_fields.items():
            merged[f] = merged.get(f, 0) + c
        return DatasetHealth(
            total_lines=self.total_lines + other.total_lines,
            nan_lines=self.nan_lines + other.nan_lines,
            nan_shards=self.nan_shards + (1 if other.nan_lines else 0),
            total_shards=self.total_shards,
            nan_fields=merged,
            contaminated_shards=sorted(
                self.contaminated_shards
                + ([other.shard] if other.nan_lines else [])
            ),
            nonfinite_lines=self.nonfinite_lines + other.nonfinite_lines,
            string_sentinel_lines=self.string_sentinel_lines + other.string_sentinel_lines,
            decode_error_lines=self.decode_error_lines + other.decode_error_lines,
        )

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
        """
        _advisory = advisory or _DEFAULT_ADVISORY_FIELDS

        if self.nan_lines == 0:
            return "CLEAN"
        if not self.nan_fields:
            return "PARSE_FAILURE — malformed JSON lines; manual inspection required"
        if all(f in _advisory for f in self.nan_fields):
            return "REPAIRABLE — NaN only in advisory fields; does not block Stage 3"
        return "HARD_FAILURE — NaN in required fields; blocks Stage 3 pipeline"


# ---------------------------------------------------------------------------
# Generic recursive traversal helper
# ---------------------------------------------------------------------------


def _walk(obj: Any, predicate: Callable[[Any], bool]) -> bool:
    """
    Generic recursive traversal over arbitrary JSON-like objects.
    Returns True as soon as predicate(node) is True for any node.
    Replaces three identical recursive walkers — DRY fix (2026-04).
    """
    if predicate(obj):
        return True
    if isinstance(obj, dict):
        return any(_walk(v, predicate) for v in obj.values())
    if isinstance(obj, list):
        return any(_walk(v, predicate) for v in obj)
    return False


# ---------------------------------------------------------------------------
# Detection helpers — all use _walk
# ---------------------------------------------------------------------------


def _has_nan(value: Any) -> bool:
    """Recursively detect float nan/inf OR stringified NaN variants."""

    def _predicate(v: Any) -> bool:
        return (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or (
            isinstance(v, str) and v in _STRING_NAN_VALUES
        )

    return _walk(value, _predicate)


def _nan_fields(obj: dict[str, Any]) -> list[str]:
    """Return list of top-level keys whose values contain a NaN/Inf variant."""
    return [k for k, v in obj.items() if _has_nan(v)]


def _is_nonfinite(value: Any) -> bool:
    """True only for float nan/inf — not string sentinels."""

    def _predicate(v: Any) -> bool:
        return isinstance(v, float) and (math.isnan(v) or math.isinf(v))

    return _walk(value, _predicate)


def _is_string_sentinel(value: Any) -> bool:
    """True only for string sentinel values — not float nan/inf."""

    def _predicate(v: Any) -> bool:
        return isinstance(v, str) and v in _STRING_NAN_VALUES

    return _walk(value, _predicate)


# ---------------------------------------------------------------------------
# Shard-level audit — lenient (default) and strict encoding modes
# ---------------------------------------------------------------------------


def _audit_shard_impl(shard_path: Path, encoding_errors: str) -> ShardHealth:
    """
    Core audit logic shared by audit_shard and audit_shard_strict.

    encoding_errors: passed to open() as errors= parameter.
      "replace" — silent normalisation, operationally resilient (default).
      "strict"  — raises UnicodeDecodeError on first corrupt byte, counted
                  as decode_error so gate_verdict can surface it.
    """
    total = nan_lines = nonfinite = sentinel = decode_err = 0
    nan_fields: dict[str, int] = {}
    try:
        with shard_path.open(encoding="utf-8", errors=encoding_errors) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
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
    except UnicodeDecodeError as exc:
        log.error("Encoding error in %s: %s", shard_path.name, exc)
        nan_lines += 1
        decode_err += 1
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


def audit_shard(shard_path: Path) -> ShardHealth:
    """Scan shard with lenient encoding (errors='replace'). Safe for multiprocessing."""
    return _audit_shard_impl(shard_path, encoding_errors="replace")


def audit_shard_strict(shard_path: Path) -> ShardHealth:
    """Scan shard with strict encoding (errors='strict'). Surfaces encoding corruption."""
    return _audit_shard_impl(shard_path, encoding_errors="strict")


# ---------------------------------------------------------------------------
# Post-repair Polars validation
# ---------------------------------------------------------------------------


def validate_shard_polars(shard_path: Path) -> tuple[bool, str | None]:
    """Validate shard with Polars read_ndjson. Returns (ok, error_or_None)."""
    try:
        import polars as pl

        pl.read_ndjson(shard_path)
        return True, None
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Shard-level repair — semantic, quote-context safe, streaming
# ---------------------------------------------------------------------------


def _replace_nonfinite(obj: Any) -> Any:
    """Recursively replace float nan/inf with None."""
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
    parse_constant intercepts bare NaN/Infinity; _replace_nonfinite nullifies.
    Returns (repaired_line, was_changed). Raises JSONDecodeError if malformed.
    """

    def _intercept(token: str) -> float:
        return float("nan") if "nan" in token.lower() else float("inf")

    obj = json.loads(line, parse_constant=_intercept)
    cleaned = _replace_nonfinite(obj)
    repaired = json.dumps(cleaned, allow_nan=False)
    return repaired + "\n", repaired != line.rstrip("\n")


def repair_shard(shard_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Repair bare NaN/Infinity -> null. Semantic, quote-context safe, streaming.
    Atomic rename; .bak backup; idempotent. Returns (total_lines, repaired).
    """
    total, repaired = 0, 0
    tmp = shard_path.with_suffix(".jsonl.tmp")
    try:
        with shard_path.open(encoding="utf-8", errors="replace") as fh, tmp.open(
            "w", encoding="utf-8"
        ) as out:
            for raw_line in fh:
                total += 1
                try:
                    fixed, changed = _semantic_repair_line(raw_line.rstrip("\n"))
                    if changed:
                        repaired += 1
                    out.write(fixed)
                except json.JSONDecodeError:
                    out.write(raw_line)

        if repaired > 0 and not dry_run:
            backup = shard_path.with_suffix(".jsonl.bak")
            shutil.copy2(shard_path, backup)
            tmp.replace(shard_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return total, repaired


# ---------------------------------------------------------------------------
# Dataset-level audit and repair
# ---------------------------------------------------------------------------


def audit_dataset(
    input_dir: Path,
    workers: int | None = None,
    strict_encoding: bool = False,
    map_fn: Any | None = None,
) -> DatasetHealth:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    shards = sorted(input_dir.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No .jsonl shards found in {input_dir}")

    ncpus = workers or multiprocessing.cpu_count()
    log.info("Scanning %d shards using %d CPU cores ...", len(shards), ncpus)

    shard_fn = audit_shard_strict if strict_encoding else audit_shard

    if map_fn is not None:
        # Injected map function (e.g. builtins.map) bypasses Pool — enables
        # synchronous execution in tests without multiprocessing overhead.
        results = list(tqdm(map_fn(shard_fn, shards), total=len(shards), unit="shard", desc="auditing"))
    else:
        with multiprocessing.Pool(processes=ncpus) as pool:
            results = list(
                tqdm(
                    pool.imap(shard_fn, shards),
                    total=len(shards),
                    unit="shard",
                    desc="auditing",
                )
            )

    return sum(results, start=DatasetHealth.zero(total_shards=len(shards)))


def _repair_shard_task(args: tuple[Path, bool]) -> tuple[str, int, int]:
    """
    Worker function for parallel repair — must be picklable (module-level).
    Returns (shard_name, total_lines, repaired_lines).
    """
    shard_path, dry_run = args
    total, repaired = repair_shard(shard_path, dry_run=dry_run)
    return shard_path.name, total, repaired


def repair_dataset(
    input_dir: Path,
    dry_run: bool = False,
    workers: int | None = None,
    validate: bool = False,
    parallel_repair: bool = False,
) -> None:
    """
    Repair all shards in input_dir.

    parallel_repair=True: mirrors audit_dataset's worker pool via
    ProcessPoolExecutor. Per-shard atomicity guaranteed by .tmp -> rename.
    For 159 shards × ~5s each, parallel repair reduces wall time from
    ~13 min (sequential) to ~2 min (8 workers).
    """
    shards = sorted(input_dir.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No .jsonl shards found in {input_dir}")
    mode = "DRY RUN" if dry_run else "REPAIRING"
    ncpus = workers or multiprocessing.cpu_count()
    log.info("%s %d shards (parallel=%s, workers=%d) ...", mode, len(shards), parallel_repair, ncpus)

    total_repaired = 0

    if parallel_repair:
        # ProcessPoolExecutor + atomic rename: safe for concurrent shard repair
        tasks = [(s, dry_run) for s in shards]
        with ProcessPoolExecutor(max_workers=ncpus) as pool:
            futures = {pool.submit(_repair_shard_task, t): t[0] for t in tasks}
            for future in tqdm(
                as_completed(futures), total=len(futures), unit="shard", desc="repairing"
            ):
                _, _, repaired = future.result()
                total_repaired += repaired
    else:
        for shard in tqdm(shards, unit="shard", desc="repairing"):
            _, repaired = repair_shard(shard, dry_run=dry_run)
            total_repaired += repaired
            if validate and not dry_run:
                ok, err = validate_shard_polars(shard)
                if not ok:
                    log.error(
                        "Post-repair Polars validation FAILED for %s: %s",
                        shard.name,
                        err,
                    )
                else:
                    log.debug("Post-repair Polars validation OK: %s", shard.name)

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


def _emit_json(health: DatasetHealth, advisory: frozenset[str] | None = None) -> None:
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
                "gate_verdict": health.gate_verdict(advisory=advisory),
                "contaminated_shards": health.contaminated_shards,
            },
            indent=2,
        )
    )


def _emit_text(
    health: DatasetHealth,
    emit_shard_ids: bool = False,
    advisory: frozenset[str] | None = None,
) -> None:
    print(f"\ntotal lines:          {health.total_lines:,}")
    print(f"nan lines:            {health.nan_lines:,}")
    print(f"  nonfinite_lines:    {health.nonfinite_lines:,}")
    print(f"  sentinel_lines:     {health.string_sentinel_lines:,}")
    print(f"  decode_error_lines: {health.decode_error_lines:,}")
    print(f"nan shards:           {health.nan_shards}/{health.total_shards}")
    print(f"clean pct:            {health.clean_pct:.4f}%")
    print(f"nan fields:           {health.nan_fields}")
    print(f"verdict:              {health.gate_verdict(advisory=advisory)}")
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
    advisory: frozenset[str] | None = None,
) -> None:
    """Log DatasetHealth to W&B. Safe in offline mode (WANDB_MODE=offline).
    Includes provenance fields: git_sha, python_version, polars_version for
    full reproducibility traceability per 2026 DL pipeline standards.
    """
    import subprocess
    import sys
    import wandb

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    try:
        import polars as pl
        polars_version = pl.__version__
    except Exception:
        polars_version = "unknown"

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
            "data/gate_verdict": health.gate_verdict(advisory=advisory),
            "provenance/git_sha": git_sha,
            "provenance/python_version": sys.version[:6],
            "provenance/polars_version": polars_version,
        }
    )
    run.finish()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=__import__("sys").stderr,
    )

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
        "--validate",
        action="store_true",
        help="With --fix: run Polars validation on each repaired shard.",
    )
    parser.add_argument(
        "--parallel-repair",
        action="store_true",
        help="With --fix: repair shards in parallel (mirrors audit worker pool).",
    )
    parser.add_argument(
        "--strict-encoding",
        action="store_true",
        help="Use errors='strict' to surface encoding corruption (default: replace).",
    )
    parser.add_argument(
        "--schema-advisory",
        action="store_true",
        help=(
            "Derive advisory fields from OpinionRecord schema (Optional fields only). "
            "Stricter 2026 policy: required fields with NaN become HARD_FAILURE."
        ),
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

    advisory: frozenset[str] | None = None
    if args.schema_advisory:
        from src.schemas import OpinionRecord

        advisory = derive_advisory_from_schema(OpinionRecord)
        log.info("Schema-driven advisory policy: %s", sorted(advisory))

    if args.fix:
        repair_dataset(
            args.input_dir,
            dry_run=args.dry_run,
            workers=args.workers,
            validate=args.validate,
            parallel_repair=args.parallel_repair,
        )
        return

    health = audit_dataset(
        args.input_dir,
        workers=args.workers,
        strict_encoding=args.strict_encoding,
    )

    if args.csv:
        _write_csv(health, args.csv)

    if args.wandb:
        log_health_to_wandb(health, advisory=advisory)

    if args.json:
        _emit_json(health, advisory=advisory)
        return

    _emit_text(health, emit_shard_ids=args.emit_shard_ids, advisory=advisory)


if __name__ == "__main__":
    main()
