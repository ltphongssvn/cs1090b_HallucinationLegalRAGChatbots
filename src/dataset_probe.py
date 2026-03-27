# src/dataset_probe.py
"""
Dataset readiness probe for local CourtListener federal appellate JSONL shards.

Gate dependency graph and failure semantics
-------------------------------------------
Gate independence:
  A7, A8, A9, A12, B6 — independent; run on the full sampled record set.
  A11 — independent; runs on a11_subsample_n records.
  A13 — depends on A8: caller pre-filters records before passing to gate_a13.
        gate_a13 trusts the caller's pre-filtered list and does not re-filter.
  schema — independent; presence/type/range/vocabulary/consistency checks.

Failure semantics:
  Any failed Category A gate with severity="blocking" (A7, A8, A11, A12, A13,
  schema) blocks Stage 3 pipeline entry and causes --ci-mode to exit 1.
  A9 severity="advisory" — distribution probe only, never blocks CI.
  B6 severity="advisory" — always passes; distribution-only gate.
  Gate failures are surfaced in report["summary"]["failed_blocking"] and
  report["summary"]["failed_advisory"].

Implements Category A dataset-readiness gates required before Stage 3:
  A7  — text_source breakdown (configurable known formats via ProbeConfig)
  A8  — text_length distribution + RAG viability threshold derivation
  A9  — citation_count distribution (advisory — not a hard corpus filter)
  A11 — tokenizer-aware chunk count (BGE-M3 + optional Mistral generative check)
  A12 — citation anchor survival + cross-validation vs citation_count field
  A13 — sentence density check via repo-certified spaCy pipeline (A8-filtered)
  B6  — text_entropy empirical distribution + spot-check for formula drift

Expected schema (23 fields):
  id, cluster_id, docket_id, court_id, court_name, case_name, date_filed,
  precedential_status, opinion_type, extracted_by_ocr, raw_text, text,
  text_length, text_source, cleaning_flags, source, token_count,
  paragraph_count, citation_count, text_hash, citation_density,
  is_precedential, text_entropy

CLI usage:
  uv run python -m src.dataset_probe \\
      --data-dir data/raw/cl_federal_appellate_bulk \\
      --subset 10000 \\
      --output logs/dataset_probe_report.json
  uv run python -m src.dataset_probe ... --ci-mode
  uv run python -m src.dataset_probe ... --full-scan
  uv run python -m src.dataset_probe ... --skip-generative-tokenizer
  uv run python -m src.dataset_probe ... --log-to-wandb \\
      --wandb-entity phl690-harvard-extension-schol \\
      --wandb-project cs1090b \\
      --wandb-name dataset_probe_v2.5.11_10k

No side effects on corpus shards — all output written to --output only.
W&B telemetry is exclusively a main() concern — _log_report_to_wandb is
never called from run_probe().
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import re
import statistics
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterator

from pydantic import BaseModel  # type: ignore[import]

try:
    import wandb  # type: ignore[import]
except ImportError:
    wandb = None  # type: ignore[assignment]

try:
    import polars as pl  # type: ignore[import]
except ImportError:
    pl = None  # type: ignore[assignment]

# spacy and transformers.AutoTokenizer are lazily imported inside the
# functions that use them (_load_spacy_nlp, gate_a11_tokenizer_chunk_count).
# This allows the probe to run schema/A7/A8/A9/B6 gates without requiring
# the 1GB+ model downloads in minimal CI environments.

# ---------------------------------------------------------------------------
# Probe version
# CHANGED 2.5.11:
#   - Specific exception types: OSError + CalledProcessError (obs 6)
#   - sample_records() public utility API docstring (obs 8)
#   - A11 chunk-count formula documented + regression constants (obs 15)
#   - stratify_by in ProbeConfig + _stratified_reservoir_sample (obs 18)
#   - ProbeReport(BaseModel) — run_probe returns typed report (obs 7/14)
#   - log_to_wandb removed from run_probe + CourtListenerDatasetProbe.run (obs 4/17)
#   - _log_report_to_wandb docstring: main()-only, no gate function calls (obs 3/13)
# ---------------------------------------------------------------------------

PROBE_VERSION = "2.5.11"

# ---------------------------------------------------------------------------
# Shared legal citation regex — OCR-resilient F\s*\.\s*\d+ pattern
# ---------------------------------------------------------------------------

_LEGAL_CITATION_RE = re.compile(
    r"(\d+\s+[A-Z][a-z]*\.?\s*(?:\d+d?|App\.?|Supp\.?)"
    r"|[A-Z][a-z]+\s+v\.\s+[A-Z]"
    r"|U\.S\.\s+\d+"
    r"|\d+\s+F\s*\.\s*\d+[a-z]?\s+\d+)",
    re.MULTILINE,
)

DOCUMENTED_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "cluster_id",
        "docket_id",
        "court_id",
        "court_name",
        "case_name",
        "date_filed",
        "precedential_status",
        "opinion_type",
        "extracted_by_ocr",
        "raw_text",
        "text",
        "text_length",
        "text_source",
        "cleaning_flags",
        "source",
        "token_count",
        "paragraph_count",
        "citation_count",
        "text_hash",
        "citation_density",
        "is_precedential",
        "text_entropy",
    }
)

KNOWN_TEXT_SOURCES: frozenset[str] = frozenset(
    {
        "plain_text",
        "html_with_citations",
        "html_lawbox",
        "html_columbia",
        "html_anon_2020",
        "xml_harvard",
        "direct_court_input",
        "pdf",
    }
)

MIN_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "court_id",
        "text",
        "text_length",
        "text_source",
        "citation_count",
        "citation_density",
        "is_precedential",
        "text_entropy",
        "token_count",
        "paragraph_count",
    }
)

REQUIRED_FIELDS: frozenset[str] = MIN_REQUIRED_FIELDS

# Fields explicitly required for Stage 3 chunking, metadata filtering,
# and citation lookup as described in the README Stage 3 section.
STAGE3_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "court_id",
        "court_name",
        "text",
        "text_length",
        "text_source",
        "citation_count",
        "citation_density",
        "is_precedential",
        "text_entropy",
        "token_count",
        "paragraph_count",
        "date_filed",
        "opinion_type",
        "precedential_status",
        "text_hash",
        "source",
    }
)


@dataclasses.dataclass(frozen=True)
class ProbeConfig:
    """
    All probe thresholds and sampling parameters.
    Explicit, versionable, injectable, and JSON-serializable for provenance.
    """

    min_text_length: int = 1500
    chunk_size_subwords: int = 1024
    chunk_overlap_subwords: int = 128
    min_sentence_count: int = 20
    encoder_model: str = "BAAI/bge-m3"
    spacy_model: str = "en_core_web_sm"
    a7_known_formats_pass_pct: float = 80.0
    a7_known_formats: frozenset[str] = dataclasses.field(
        default_factory=lambda: frozenset({"plain_text", "html_with_citations"})
    )
    a8_below_threshold_pass_pct: float = 25.0
    a9_zero_citation_pass_pct: float = 20.0
    a11_min_median_chunks: float = 2.0
    a12_min_pct_with_anchor: float = 60.0
    a12_text_cap_chars: int = 50_000
    a13_max_below_threshold_pct: float = 15.0
    quality_signals_sample_n: int = 500
    a11_subsample_n: int = 200
    a12_subsample_n: int = 500
    a13_subsample_n: int = 200
    a13_text_cap_chars: int = 50_000
    quality_signals_text_cap_chars: int = 50_000
    b6_entropy_spot_check_tolerance: float = 1.0
    b6_entropy_spot_check_sample_n: int = 10
    a11_generative_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    text_length_consistency_tolerance: int = 200
    text_length_relative_tolerance: float = 0.05
    quality_signals_html_pattern: str = r"<[a-zA-Z][^>]{0,100}>"
    quality_signals_boilerplate_phrases: tuple[str, ...] = dataclasses.field(
        default_factory=lambda: (
            "all rights reserved",
            "this page intentionally left blank",
            "unpublished disposition",
            "not for publication",
            "do not cite",
        )
    )
    # Optional stratification field for proportional court/source coverage.
    # When set, run_probe uses _stratified_reservoir_sample instead of uniform
    # reservoir sampling. Ensures minority circuits are represented.
    # Set to None (default) to use standard uniform reservoir sampling.
    stratify_by: str | None = None


def _probe_config_to_dict(cfg: ProbeConfig) -> dict[str, Any]:
    """Serialize ProbeConfig to a JSON-safe dict.
    Converts frozensets to sorted lists, tuples to lists, None preserved."""
    result: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        if isinstance(val, frozenset):
            result[f.name] = sorted(val)
        elif isinstance(val, tuple):
            result[f.name] = list(val)
        else:
            result[f.name] = val
    return result


# ---------------------------------------------------------------------------
# Module-level constants — literal values, NOT derived from ProbeConfig().
# No type annotations — must match test assertions exactly.
# Values must equal corresponding ProbeConfig field defaults.
# ---------------------------------------------------------------------------
PROVISIONAL_MIN_TEXT_LENGTH = 1500
CHUNK_SIZE_SUBWORDS = 1024
CHUNK_OVERLAP_SUBWORDS = 128
ENCODER_MODEL = "BAAI/bge-m3"
SPACY_MODEL = "en_core_web_sm"
SPACY_EXCLUDE = ["ner", "parser", "lemmatizer"]
MIN_SENTENCE_COUNT = 20


# ---------------------------------------------------------------------------
# GateResult — Pydantic v2 model for gate output contract
# ---------------------------------------------------------------------------


class GateResult(BaseModel):
    """
    Typed contract for gate results. All gates return dicts that satisfy
    this minimum contract. Extra gate-specific fields are allowed.
    frozen=True enforces immutability after construction.
    """

    gate: str
    severity: str
    model_config = {"extra": "allow", "frozen": True}


# ---------------------------------------------------------------------------
# ProbeReport — Pydantic v2 model for run_probe return value (obs 7/14)
# ---------------------------------------------------------------------------


class ProbeReport(BaseModel):
    """
    Typed return value for run_probe(). Replaces raw dict[str, Any].
    Supports dict-style access (report['gates']) and 'key in report'
    for backward compatibility via __getitem__ and __contains__.
    All fields are also accessible as attributes.
    """

    gates: dict[str, Any]
    summary: dict[str, Any]
    provenance: dict[str, Any]
    quality_signals: dict[str, Any]
    shard_audit: dict[str, Any]
    subset_n: int
    seed: int
    data_dir: str
    model_config = {"extra": "allow", "frozen": False}

    def __getitem__(self, key: str) -> Any:
        """Dict-style access for backward compatibility."""
        return self.model_dump()[key]

    def __contains__(self, key: object) -> bool:
        """Support 'key in report' for backward compatibility."""
        return key in self.model_dump()


# ---------------------------------------------------------------------------
# GATE_REGISTRY — populated after gate functions are defined
# ---------------------------------------------------------------------------

GATE_REGISTRY: list[dict[str, Any]] = []


def _safe_int(value: Any, fallback: int = 0) -> int:
    """
    Safely convert value to int, returning fallback on ValueError or TypeError.
    Gates A8/A9 exclude malformed values and report parse error counts separately.
    _safe_int is used in A12/A13/run_probe for filtering logic.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return fallback


def _get_git_sha() -> str:
    """Return current git commit SHA, or 'not-a-git-repo' if unavailable.
    Catches OSError (missing git binary) and CalledProcessError (non-zero
    exit when not in a git repo) — not bare Exception."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "not-a-git-repo"


def _get_text(row: dict[str, Any]) -> str:
    """Return the text field of a record as a string, or '' if missing/None."""
    val = row.get("text")
    if val is None:
        return ""
    return str(val)


def _percentile(sorted_values: list[Any], p: float) -> Any:
    """
    Return the p-th percentile of a pre-sorted list using the ceiling-index
    empirical convention defined for this probe. p must be in [0, 100].

    Convention: index = ceil(p/100 * n) - 1, clamped to [0, n-1].
    This is a deterministic empirical percentile rule — it does not use
    interpolation. Do not compare directly to other libraries' default
    percentile methods which may differ. p=0 returns min, p=100 returns max.
    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError("Cannot compute percentile of empty list")
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    idx = max(0, min(n - 1, int(math.ceil(p / 100.0 * n)) - 1))
    return sorted_values[idx]


def _shannon_entropy(text: str) -> float:
    """
    Compute word-level Shannon entropy in bits (not normalized by log2(|V|)).

    Tokenization contract (must match upstream text_entropy field computation):
      - Tokenization basis: whitespace split (str.split()) — no regex, no unicode
        segmentation. Tokens are substrings separated by any whitespace.
      - Case: preserved as-is — 'Court' and 'court' are distinct tokens.
      - Punctuation: included — 'held.' and 'held' are distinct tokens.
      - Stopwords: no removal — all whitespace-delimited tokens are counted.
      - Normalization: raw Shannon bits, NOT divided by log2(|V|). Formula:
          H = -sum(p_i * log2(p_i)) where p_i = count(w_i) / total_words.
      - Empty string: returns 0.0.
      - Single unique token: returns 0.0 (H = 0 for a deterministic signal).

    This contract is validated by B6 spot-check against stored text_entropy.
    If upstream changes tokenization basis, B6 spot_check.consistent will be False.
    """
    words = text.split()
    if not words:
        return 0.0
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    n = len(words)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


# ---------------------------------------------------------------------------
# Pure reservoir sampler — decoupled from file I/O
# ---------------------------------------------------------------------------


def _reservoir_sample(
    iterable: Iterator[dict[str, Any]],
    n: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Pure Vitter's reservoir sampling — accepts any iterator of dicts.
    Decoupled from file I/O: no Path, no file opening, no JSON parsing.
    Makes the sampling algorithm unit-testable without disk I/O.
    Returns a reservoir of up to n records sampled uniformly at random.
    """
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    for i, record in enumerate(iterable):
        if i < n:
            reservoir.append(record)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = record
    return reservoir


def _stratified_reservoir_sample(
    iterable: Iterator[dict[str, Any]],
    n: int,
    stratify_by: str,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Proportional stratified sampling by a record field (e.g., 'court_id').

    Consumes the full iterable to group records into strata, then samples
    proportionally from each stratum so all strata are represented.
    Proportional allocation: stratum_n = max(1, round(n * |stratum| / |total|)).
    If result exceeds n after rounding, trims randomly.

    Use when uniform reservoir sampling would under-represent minority courts
    or text_source types. Configured via ProbeConfig.stratify_by.
    """
    rng = random.Random(seed)
    all_records = list(iterable)
    if not all_records:
        return []

    strata: dict[str, list[dict[str, Any]]] = {}
    for r in all_records:
        key = str(r.get(stratify_by, "MISSING"))
        strata.setdefault(key, []).append(r)

    total = len(all_records)
    result: list[dict[str, Any]] = []
    for stratum_records in strata.values():
        stratum_n = max(1, round(n * len(stratum_records) / total))
        sampled = rng.sample(stratum_records, min(stratum_n, len(stratum_records)))
        result.extend(sampled)

    if len(result) > n:
        rng.shuffle(result)
        result = result[:n]

    return result


# ---------------------------------------------------------------------------
# Shard loaders
# ---------------------------------------------------------------------------


def iter_shards(data_dir: Path) -> Iterator[dict[str, Any]]:
    """
    Yield valid records from all .jsonl shards in sorted order.
    Silently drops blank lines and JSON parse errors with no counting —
    use iter_shards_with_audit() to obtain counts of skipped lines and
    parse errors.
    """
    for record, _ in _iter_shards_inner(data_dir):
        yield record


def iter_shards_with_audit(data_dir: Path) -> dict[str, Any]:
    """Load all shards and return audit summary with total_records_decoded."""
    shard_files = sorted(data_dir.glob("*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No .jsonl shards found in {data_dir}")

    records: list[dict[str, Any]] = []
    shard_errors: dict[str, int] = {}
    total_parse_errors = 0
    total_blank_lines = 0

    for shard_path in shard_files:
        errors = 0
        with open(shard_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    total_blank_lines += 1
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError:
                    errors += 1
                    total_parse_errors += 1
        if errors:
            shard_errors[shard_path.name] = errors

    return {
        "records": records,
        "total_records_decoded": len(records),
        "total_parse_errors": total_parse_errors,
        "total_blank_lines": total_blank_lines,
        "shard_errors": shard_errors,
        "shard_count": len(shard_files),
    }


def _iter_shards_inner(data_dir: Path) -> Generator[tuple[dict[str, Any], str], None, None]:
    """Internal generator yielding (record, shard_name) tuples."""
    shard_files = sorted(data_dir.glob("*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No .jsonl shards found in {data_dir}")
    for shard_path in shard_files:
        with open(shard_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped), shard_path.name
                except json.JSONDecodeError:
                    continue


def sample_records(data_dir: Path, n: int, seed: int = 0) -> list[dict[str, Any]]:
    """
    Public utility API: reservoir-sample n records from all shards.

    This is an intentional public API for callers who need lightweight
    corpus exploration (notebooks, ad-hoc analysis) without running the
    full probe pipeline or collecting parse-error audit stats.

    For the full probe pipeline with parse-error counting and shard-level
    audit, use the internal _reservoir_sample_with_audit() path, which is
    called automatically by run_probe().

    JSON parse errors are silently dropped without audit in this path —
    use iter_shards_with_audit() directly if parse error counts matter.
    """
    return _reservoir_sample(iter_shards(data_dir), n=n, seed=seed)


def _reservoir_sample_with_audit(
    data_dir: Path,
    n: int,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One-pass streaming: collect audit counters and reservoir sample simultaneously."""
    shard_files = sorted(data_dir.glob("*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No .jsonl shards found in {data_dir}")

    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    shard_errors: dict[str, int] = {}
    total_parse_errors = 0
    total_blank_lines = 0
    total_records_decoded = 0

    for shard_path in shard_files:
        errors = 0
        with open(shard_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    total_blank_lines += 1
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    errors += 1
                    total_parse_errors += 1
                    continue
                i = total_records_decoded
                if i < n:
                    reservoir.append(record)
                else:
                    j = rng.randint(0, i)
                    if j < n:
                        reservoir[j] = record
                total_records_decoded += 1
        if errors:
            shard_errors[shard_path.name] = errors

    audit = {
        "shard_count": len(shard_files),
        "total_records_decoded": total_records_decoded,
        "total_parse_errors": total_parse_errors,
        "total_blank_lines": total_blank_lines,
        "shard_errors": shard_errors,
    }
    return reservoir, audit


def _full_scan_with_polars(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load all records from all shards using Polars scan_ndjson for exact statistics.
    Used by run_probe when full_scan=True.
    """
    if pl is None:
        raise ImportError("polars is required for --full-scan mode. Install with: uv add polars")
    shard_files = sorted(data_dir.glob("*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No .jsonl shards found in {data_dir}")

    all_records: list[dict[str, Any]] = []
    total_parse_errors = 0
    shard_errors: dict[str, int] = {}

    for shard_path in shard_files:
        try:
            df = pl.scan_ndjson(shard_path).collect()
            all_records.extend(df.to_dicts())
        except Exception as exc:
            shard_errors[shard_path.name] = 1
            total_parse_errors += 1
            print(f"[dataset_probe] WARNING: Polars failed on {shard_path.name}: {exc}")

    audit = {
        "shard_count": len(shard_files),
        "total_records_decoded": len(all_records),
        "total_parse_errors": total_parse_errors,
        "total_blank_lines": 0,
        "shard_errors": shard_errors,
    }
    return all_records, audit


# ---------------------------------------------------------------------------
# validate_schema composable helpers
# ---------------------------------------------------------------------------


def _check_presence(records: list[dict[str, Any]]) -> dict[str, int]:
    """Return missing_counts for MIN_REQUIRED_FIELDS — {field: count_missing}."""
    missing: dict[str, int] = {f: 0 for f in MIN_REQUIRED_FIELDS}
    for r in records:
        for f in MIN_REQUIRED_FIELDS:
            if f not in r:
                missing[f] += 1
    return missing


def _check_types_and_ranges(
    records: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (type_errors, range_errors) dicts for numeric/bool fields."""
    type_errors: dict[str, int] = {}
    range_errors: dict[str, int] = {}

    for r in records:
        text_len = r.get("text_length")
        if text_len is not None and not isinstance(text_len, (int, float)):
            type_errors["text_length"] = type_errors.get("text_length", 0) + 1

        is_prec = r.get("is_precedential")
        if is_prec is not None and not isinstance(is_prec, bool):
            type_errors["is_precedential"] = type_errors.get("is_precedential", 0) + 1

        cite_count = r.get("citation_count")
        if cite_count is not None:
            try:
                if int(cite_count) < 0:
                    range_errors["citation_count"] = range_errors.get("citation_count", 0) + 1
            except (TypeError, ValueError):
                type_errors["citation_count"] = type_errors.get("citation_count", 0) + 1

        cite_density = r.get("citation_density")
        if cite_density is not None:
            if not isinstance(cite_density, (int, float)):
                type_errors["citation_density"] = type_errors.get("citation_density", 0) + 1
            elif float(cite_density) < 0:
                range_errors["citation_density"] = range_errors.get("citation_density", 0) + 1

        text_entropy = r.get("text_entropy")
        if text_entropy is not None:
            if not isinstance(text_entropy, (int, float)):
                type_errors["text_entropy"] = type_errors.get("text_entropy", 0) + 1
            elif float(text_entropy) < 0:
                range_errors["text_entropy"] = range_errors.get("text_entropy", 0) + 1

        para_count = r.get("paragraph_count")
        if para_count is not None:
            if not isinstance(para_count, int):
                type_errors["paragraph_count"] = type_errors.get("paragraph_count", 0) + 1
            elif para_count < 0:
                range_errors["paragraph_count"] = range_errors.get("paragraph_count", 0) + 1

        tok_count = r.get("token_count")
        if tok_count is not None:
            if not isinstance(tok_count, int):
                type_errors["token_count"] = type_errors.get("token_count", 0) + 1
            elif tok_count < 0:
                range_errors["token_count"] = range_errors.get("token_count", 0) + 1

    return type_errors, range_errors


def _check_vocabulary(records: list[dict[str, Any]]) -> dict[str, int]:
    """Return vocabulary_errors — {field: count_invalid} for enum fields."""
    vocab_errors: dict[str, int] = {}
    for r in records:
        text_source = r.get("text_source")
        if text_source is not None and str(text_source) not in KNOWN_TEXT_SOURCES:
            vocab_errors["text_source"] = vocab_errors.get("text_source", 0) + 1
    return vocab_errors


def _check_consistency(
    records: list[dict[str, Any]],
    tolerance: int = 200,
    relative_tolerance: float = 0.05,
) -> dict[str, int]:
    """
    Return consistency_errors — checks text_length vs len(text).
    Uses OR logic: fails if abs_diff > tolerance OR rel_diff > relative_tolerance.
    Handles short docs (200 chars = 40% of 500-char doc) and long docs
    (200 chars = 0.2% of 100K-char doc — acceptable rounding) correctly.
    """
    consistency_errors: dict[str, int] = {}
    for r in records:
        text_len = r.get("text_length")
        actual_text = r.get("text")
        if text_len is not None and isinstance(text_len, (int, float)) and actual_text is not None:
            actual_len = len(str(actual_text))
            abs_diff = abs(int(text_len) - actual_len)
            rel_diff = abs_diff / max(actual_len, 1)
            if abs_diff > tolerance or rel_diff > relative_tolerance:
                consistency_errors["text_length_consistency"] = consistency_errors.get("text_length_consistency", 0) + 1
    return consistency_errors


def _check_documented_coverage(records: list[dict[str, Any]]) -> dict[str, int]:
    """
    Return missing counts for DOCUMENTED_FIELDS not in MIN_REQUIRED_FIELDS.
    Advisory — does not affect pass/fail.
    """
    documented_only = DOCUMENTED_FIELDS - MIN_REQUIRED_FIELDS
    missing: dict[str, int] = {}
    for r in records:
        for f in documented_only:
            if f not in r:
                missing[f] = missing.get(f, 0) + 1
    return missing


def validate_schema(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """
    Check required field presence, type, range, vocabulary, consistency, and
    documented field coverage. Also reports stage3_pass and stage3_missing_counts.
    """
    cfg = config or ProbeConfig()

    if not records:
        return {
            "gate": "schema_validation",
            "severity": "blocking",
            "required_fields": sorted(MIN_REQUIRED_FIELDS),
            "missing_counts": {},
            "type_errors": {},
            "range_errors": {},
            "vocabulary_errors": {},
            "consistency_errors": {},
            "missing_documented_fields": {},
            "stage3_pass": True,
            "stage3_missing_counts": {},
            "pass": True,
            "note": "No records to validate.",
        }

    missing_by_field = _check_presence(records)
    type_errors, range_errors = _check_types_and_ranges(records)
    vocabulary_errors = _check_vocabulary(records)
    consistency_errors = _check_consistency(
        records,
        tolerance=cfg.text_length_consistency_tolerance,
        relative_tolerance=cfg.text_length_relative_tolerance,
    )
    missing_documented = _check_documented_coverage(records)

    stage3_missing: dict[str, int] = {}
    for r in records:
        for f in STAGE3_REQUIRED_FIELDS:
            if f not in r:
                stage3_missing[f] = stage3_missing.get(f, 0) + 1
    stage3_pass = not bool(stage3_missing)

    any_missing = any(v > 0 for v in missing_by_field.values())
    passed = (
        not any_missing and not type_errors and not range_errors and not vocabulary_errors and not consistency_errors
    )
    return {
        "gate": "schema_validation",
        "severity": "blocking",
        "required_fields": sorted(MIN_REQUIRED_FIELDS),
        "missing_counts": {k: v for k, v in missing_by_field.items() if v > 0},
        "type_errors": type_errors,
        "range_errors": range_errors,
        "vocabulary_errors": vocabulary_errors,
        "consistency_errors": consistency_errors,
        "missing_documented_fields": missing_documented,
        "stage3_pass": stage3_pass,
        "stage3_missing_counts": stage3_missing,
        "pass": passed,
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def gate_a7_text_source_breakdown(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """A7 — text_source breakdown. severity=blocking."""
    cfg = config or ProbeConfig()
    if not records:
        return {
            "gate": "A7_text_source_breakdown",
            "severity": "blocking",
            "sample_n": 0,
            "pass": False,
            "note": "No records.",
        }

    counts: dict[str, int] = {}
    for r in records:
        src = str(r.get("text_source", "MISSING"))
        counts[src] = counts.get(src, 0) + 1
    total = len(records)
    breakdown = {
        src: {"count": cnt, "pct": round(100.0 * cnt / total, 2)}
        for src, cnt in sorted(counts.items(), key=lambda x: -x[1])
    }
    known_pct = sum(v["pct"] for k, v in breakdown.items() if k in cfg.a7_known_formats)
    return {
        "gate": "A7_text_source_breakdown",
        "severity": "blocking",
        "sample_n": total,
        "total_records": total,
        "breakdown": breakdown,
        "known_formats_pct": round(known_pct, 2),
        "unknown_formats_pct": round(100.0 - known_pct, 2),
        "pass": known_pct >= cfg.a7_known_formats_pass_pct,
        "note": (
            "Inspect records from any source outside a7_known_formats "
            "to verify row_normalizer.py strips them cleanly before Stage 3."
        ),
    }


def gate_a8_text_length_distribution(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """
    A8 — text_length distribution. severity=blocking.
    Malformed text_length values are EXCLUDED from the distribution.
    text_length_parse_errors counts excluded records.
    sample_n reflects valid (parseable) records only.
    """
    cfg = config or ProbeConfig()
    if not records:
        return {
            "gate": "A8_text_length_distribution",
            "severity": "blocking",
            "sample_n": 0,
            "text_length_parse_errors": 0,
            "pass": False,
            "note": "No records.",
        }

    lengths: list[int] = []
    parse_errors = 0
    for r in records:
        raw = r.get("text_length", 0)
        try:
            lengths.append(int(raw))
        except (ValueError, TypeError):
            parse_errors += 1

    if not lengths:
        return {
            "gate": "A8_text_length_distribution",
            "severity": "blocking",
            "count": 0,
            "sample_n": 0,
            "text_length_parse_errors": parse_errors,
            "pass": False,
            "note": (f"All {parse_errors} records have unparseable text_length — cannot compute distribution."),
        }

    lengths_sorted = sorted(lengths)
    below_provisional = sum(1 for length in lengths if length < cfg.min_text_length)
    return {
        "gate": "A8_text_length_distribution",
        "severity": "blocking",
        "count": len(lengths),
        "sample_n": len(lengths),
        "text_length_parse_errors": parse_errors,
        "mean": round(statistics.mean(lengths), 1),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "p5": _percentile(lengths_sorted, 5),
        "p10": _percentile(lengths_sorted, 10),
        "p25": _percentile(lengths_sorted, 25),
        "p75": _percentile(lengths_sorted, 75),
        "p90": _percentile(lengths_sorted, 90),
        "p95": _percentile(lengths_sorted, 95),
        "provisional_min_chars": cfg.min_text_length,
        "below_provisional_count": below_provisional,
        "below_provisional_pct": round(100.0 * below_provisional / len(lengths), 2),
        "pass": below_provisional / len(lengths) < cfg.a8_below_threshold_pass_pct / 100.0,
        "note": (
            "~20% short-doc tail is expected (summary dispositions). "
            "Stage 3 applies text_length >= 1500 filter before chunking."
        ),
    }


def gate_a9_citation_count_distribution(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """
    A9 — citation_count distribution. severity=advisory.
    Malformed citation_count values are EXCLUDED from the distribution.
    citation_count_parse_errors counts excluded records.
    sample_n reflects valid (parseable) records only.
    """
    cfg = config or ProbeConfig()
    if not records:
        return {
            "gate": "A9_citation_count_distribution",
            "severity": "advisory",
            "sample_n": 0,
            "citation_count_parse_errors": 0,
            "pass": False,
            "note": "No records.",
        }

    counts: list[int] = []
    parse_errors = 0
    for r in records:
        raw = r.get("citation_count", 0)
        try:
            counts.append(int(raw))
        except (ValueError, TypeError):
            parse_errors += 1

    if not counts:
        return {
            "gate": "A9_citation_count_distribution",
            "severity": "advisory",
            "count": 0,
            "sample_n": 0,
            "citation_count_parse_errors": parse_errors,
            "pass": False,
            "note": (
                f"All {parse_errors} records have unparseable citation_count — "
                "cannot compute distribution. Advisory gate only."
            ),
        }

    n = len(counts)
    zero = sum(1 for c in counts if c == 0)
    above_5 = sum(1 for c in counts if c > 5)
    return {
        "gate": "A9_citation_count_distribution",
        "severity": "advisory",
        "count": n,
        "sample_n": n,
        "citation_count_parse_errors": parse_errors,
        "mean": round(statistics.mean(counts), 2),
        "median": statistics.median(counts),
        "min": min(counts),
        "max": max(counts),
        "zero_citation_count": zero,
        "zero_citation_pct": round(100.0 * zero / n, 2),
        "above_5_count": above_5,
        "above_5_pct": round(100.0 * above_5 / n, 2),
        "pass": zero / n < cfg.a9_zero_citation_pass_pct / 100.0,
        "note": (
            "Advisory probe only — does not hard-filter the corpus. "
            "Full 1.46M-opinion corpus is used unfiltered for final runs. "
            "Fast-iteration subset (~150K) may optionally filter citation_count > 5 "
            "to maximise Tier C utility."
        ),
    }


def gate_a11_tokenizer_chunk_count(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """
    A11 — Tokenizer-aware chunk count. severity=blocking.

    Chunk count formula (matches Stage 3 chunking pipeline exactly):
      stride = chunk_size_subwords - chunk_overlap_subwords  (= 896)
      n_chunks = max(1, ceil((total_tokens - chunk_overlap_subwords) / stride))

    This formula is validated by TestA11ChunkCountFormula regression tests
    against CHUNK_SIZE_SUBWORDS=1024 and CHUNK_OVERLAP_SUBWORDS=128.

    AutoTokenizer is lazily imported here — not at module top level — to allow
    schema/A7/A8/A9/B6 gates to run without 1GB+ model downloads.
    generative_token_check sub-dict has severity=advisory — diagnostic only.
    Catches OSError for tokenizer load failure (network/HF_TOKEN issue).
    """
    cfg = config or ProbeConfig()
    if not records:
        return {
            "gate": "A11_tokenizer_chunk_count",
            "severity": "blocking",
            "pass": False,
            "note": "No records.",
        }

    if tokenizer is None:
        try:
            from transformers import AutoTokenizer  # type: ignore[import]  # noqa: PLC0415

            tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_model)
            tokenizer_revision = getattr(tokenizer, "name_or_path", cfg.encoder_model)
        except OSError as exc:
            return {
                "gate": "A11_tokenizer_chunk_count",
                "severity": "blocking",
                "pass": False,
                "error": str(exc),
                "note": "AutoTokenizer load failed — check HF_TOKEN and network.",
            }
    else:
        tokenizer_revision = getattr(tokenizer, "name_or_path", cfg.encoder_model)

    chunk_counts: list[int] = []
    token_lengths: list[int] = []
    multi_chunk = 0

    for r in records:
        text = _get_text(r)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        total_tokens = len(ids)
        token_lengths.append(total_tokens)
        stride = cfg.chunk_size_subwords - cfg.chunk_overlap_subwords
        n_chunks = max(1, math.ceil(max(0, total_tokens - cfg.chunk_overlap_subwords) / stride))
        chunk_counts.append(n_chunks)
        if n_chunks > 1:
            multi_chunk += 1

    result: dict[str, Any] = {
        "gate": "A11_tokenizer_chunk_count",
        "severity": "blocking",
        "encoder_model": cfg.encoder_model,
        "tokenizer_revision": tokenizer_revision,
        "chunk_size_subwords": cfg.chunk_size_subwords,
        "overlap_subwords": cfg.chunk_overlap_subwords,
        "subsample_n": len(records),
        "mean_token_length": round(statistics.mean(token_lengths), 1),
        "median_token_length": statistics.median(token_lengths),
        "mean_chunks_per_doc": round(statistics.mean(chunk_counts), 2),
        "median_chunks_per_doc": statistics.median(chunk_counts),
        "multi_chunk_pct": round(100.0 * multi_chunk / len(records), 2),
        "pass": statistics.median(chunk_counts) >= cfg.a11_min_median_chunks,
        "note": "median_chunks_per_doc >= 2 confirms corpus supports multi-chunk splitting.",
    }

    if cfg.a11_generative_model:
        try:
            from transformers import AutoTokenizer  # type: ignore[import]  # noqa: PLC0415

            gen_tok = AutoTokenizer.from_pretrained(cfg.a11_generative_model)
            gen_lengths: list[int] = []
            mistral_limit = 32_768
            over_limit = 0
            for r in records:
                text = _get_text(r)
                gen_ids = gen_tok(text, add_special_tokens=False)["input_ids"]
                gen_lengths.append(len(gen_ids))
                if len(gen_ids) > mistral_limit:
                    over_limit += 1
            result["generative_token_check"] = {
                "severity": "advisory",
                "model": cfg.a11_generative_model,
                "mean_tokens": round(statistics.mean(gen_lengths), 1),
                "median_tokens": statistics.median(gen_lengths),
                "max_tokens": max(gen_lengths),
                "over_32k_count": over_limit,
                "over_32k_pct": round(100.0 * over_limit / len(records), 2),
                "note": "README asserts max(prompt_tokens) < 32768 using Mistral tokenizer.",
            }
        except OSError as exc:
            result["generative_token_check"] = {
                "severity": "advisory",
                "model": cfg.a11_generative_model,
                "error": str(exc),
                "note": "Generative tokenizer load failed — skipping secondary check.",
            }

    return result


def gate_a12_citation_anchor_survival(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """
    A12 — Citation anchor survival. severity=blocking.
    Reports records_text_capped — count of records where text > cap.
    """
    cfg = config or ProbeConfig()
    if not records:
        return {
            "gate": "A12_citation_anchor_survival",
            "severity": "blocking",
            "records_text_capped": 0,
            "pass": False,
            "note": "No records.",
        }

    has_anchor = 0
    citation_counts_found: list[int] = []
    field_nonzero_regex_zero = 0
    field_nonzero_total = 0
    records_text_capped = 0

    for r in records:
        raw_text = _get_text(r)
        if len(raw_text) > cfg.a12_text_cap_chars:
            records_text_capped += 1
        text = raw_text[: cfg.a12_text_cap_chars]
        matches = _LEGAL_CITATION_RE.findall(text)
        n_matches = len(matches)
        citation_counts_found.append(n_matches)
        if matches:
            has_anchor += 1
        field_count = _safe_int(r.get("citation_count", 0), fallback=0)
        if field_count > 0:
            field_nonzero_total += 1
            if n_matches == 0:
                field_nonzero_regex_zero += 1

    pct_with_anchor = 100.0 * has_anchor / len(records)
    field_nonzero_regex_zero_pct = (
        round(100.0 * field_nonzero_regex_zero / field_nonzero_total, 2) if field_nonzero_total > 0 else 0.0
    )

    return {
        "gate": "A12_citation_anchor_survival",
        "severity": "blocking",
        "subsample_n": len(records),
        "text_cap_chars": cfg.a12_text_cap_chars,
        "records_text_capped": records_text_capped,
        "records_with_citation_anchor": has_anchor,
        "pct_with_citation_anchor": round(pct_with_anchor, 2),
        "mean_anchors_per_doc": round(statistics.mean(citation_counts_found), 2),
        "citation_field_vs_regex": {
            "field_nonzero_total": field_nonzero_total,
            "field_nonzero_regex_zero_count": field_nonzero_regex_zero,
            "field_nonzero_regex_zero_pct": field_nonzero_regex_zero_pct,
            "note": (
                "Records where citation_count > 0 but regex finds 0 matches — "
                "may indicate normalization destroying citation anchors."
            ),
        },
        "pass": pct_with_anchor >= cfg.a12_min_pct_with_anchor,
        "note": (
            "Heuristic approximation via regex — not a precision extractor. "
            ">=60% of records must contain extractable citation anchors "
            "for Tier C SQLite lookup to be viable. "
            f"Text capped at {cfg.a12_text_cap_chars} chars before scan."
        ),
    }


# ---------------------------------------------------------------------------
# gate_a13 internal helpers
# ---------------------------------------------------------------------------


def _load_spacy_nlp(
    cfg: ProbeConfig,
    nlp: Any | None,
) -> tuple[Any, str]:
    """
    Load or return the spaCy NLP pipeline.
    spaCy is lazily imported here — not at module top level.
    Catches OSError specifically (missing model file or package).
    Returns (nlp_object, version_string).
    If nlp is injected, returns it with version='injected'.
    If loading fails with OSError, returns (None, 'load_failed').
    """
    if nlp is not None:
        return nlp, "injected"
    try:
        import spacy as _spacy  # type: ignore[import]  # noqa: PLC0415

        loaded = _spacy.load(cfg.spacy_model, exclude=SPACY_EXCLUDE)
        if "sentencizer" not in loaded.pipe_names:
            loaded.add_pipe("sentencizer")
        loaded.max_length = 2_000_000
        return loaded, _spacy.__version__
    except OSError:
        return None, "load_failed"


def _compute_sentence_counts(
    records: list[dict[str, Any]],
    nlp: Any,
    cfg: ProbeConfig,
) -> tuple[list[int], int]:
    """
    Run sentence segmentation on all records.
    Returns (sent_counts, below_threshold_count).
    """
    sent_counts: list[int] = []
    below_threshold = 0
    for r in records:
        text = _get_text(r)[: cfg.a13_text_cap_chars]
        doc = nlp(text)
        n_sents = sum(1 for _ in doc.sents)
        sent_counts.append(n_sents)
        if n_sents < cfg.min_sentence_count:
            below_threshold += 1
    return sent_counts, below_threshold


def gate_a13_sentence_density(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
    nlp: Any | None = None,
) -> dict[str, Any]:
    """
    A13 — Sentence density. severity=blocking.
    Trusts the caller's pre-filtered record list — no internal re-filter.
    spaCy is lazily imported via _load_spacy_nlp.
    """
    cfg = config or ProbeConfig()
    if not records:
        return {
            "gate": "A13_sentence_density",
            "severity": "blocking",
            "pass": False,
            "records_after_a8_filter": 0,
            "note": "No records.",
        }

    nlp_obj, spacy_version = _load_spacy_nlp(cfg, nlp)
    if nlp_obj is None:
        return {
            "gate": "A13_sentence_density",
            "severity": "blocking",
            "pass": False,
            "records_after_a8_filter": len(records),
            "error": "spaCy model load failed",
            "note": "spaCy model load failed — check spacy_model in ProbeConfig.",
        }

    sent_counts, below_threshold = _compute_sentence_counts(records, nlp_obj, cfg)

    return {
        "gate": "A13_sentence_density",
        "severity": "blocking",
        "spacy_model": cfg.spacy_model,
        "spacy_version": spacy_version,
        "min_sentence_threshold": cfg.min_sentence_count,
        "subsample_n": len(records),
        "records_after_a8_filter": len(records),
        "mean_sentences": round(statistics.mean(sent_counts), 1),
        "median_sentences": statistics.median(sent_counts),
        "min_sentences": min(sent_counts),
        "below_threshold_count": below_threshold,
        "below_threshold_pct": round(100.0 * below_threshold / len(records), 2),
        "pass": below_threshold / len(records) < cfg.a13_max_below_threshold_pct / 100.0,
        "note": (
            "Caller pre-filters to text_length >= 1500 records before passing here. "
            "Pass threshold: <15% below 20 sentences."
        ),
    }


def gate_b6_text_entropy_distribution(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """B6 — text_entropy distribution. severity=advisory — always passes."""
    cfg = config or ProbeConfig()
    if not records:
        return {
            "gate": "B6_text_entropy_distribution",
            "severity": "advisory",
            "sample_n": 0,
            "pass": True,
            "note": "No records.",
        }

    entropies = [float(r.get("text_entropy", 0.0)) for r in records]
    entropies_sorted = sorted(entropies)
    zero_entropy = sum(1 for e in entropies if e == 0.0)

    spot_sample = records[: cfg.b6_entropy_spot_check_sample_n]
    deviations: list[float] = []
    for r in spot_sample:
        computed = _shannon_entropy(_get_text(r))
        stored = float(r.get("text_entropy", 0.0))
        deviations.append(abs(computed - stored))
    max_deviation = max(deviations) if deviations else 0.0

    return {
        "gate": "B6_text_entropy_distribution",
        "severity": "advisory",
        "sample_n": len(entropies),
        "count": len(entropies),
        "mean": round(statistics.mean(entropies), 4),
        "median": round(float(statistics.median(entropies)), 4),
        "min": round(min(entropies), 4),
        "max": round(max(entropies), 4),
        "p5": _percentile(entropies_sorted, 5),
        "p10": _percentile(entropies_sorted, 10),
        "p25": _percentile(entropies_sorted, 25),
        "p75": _percentile(entropies_sorted, 75),
        "p90": _percentile(entropies_sorted, 90),
        "p95": _percentile(entropies_sorted, 95),
        "zero_entropy_count": zero_entropy,
        "zero_entropy_pct": round(100.0 * zero_entropy / len(entropies), 2),
        "spot_check": {
            "consistent": max_deviation <= cfg.b6_entropy_spot_check_tolerance,
            "max_deviation": round(max_deviation, 4),
            "tolerance": cfg.b6_entropy_spot_check_tolerance,
            "sample_n": len(spot_sample),
        },
        "pass": True,
        "note": "Use p10 as provisional low-entropy filter cutoff for Stage 3.",
    }


# ---------------------------------------------------------------------------
# Populate GATE_REGISTRY after all gate functions are defined
# ---------------------------------------------------------------------------

GATE_REGISTRY = [
    {"name": "A7", "fn": gate_a7_text_source_breakdown, "severity": "blocking"},
    {"name": "A8", "fn": gate_a8_text_length_distribution, "severity": "blocking"},
    {"name": "A9", "fn": gate_a9_citation_count_distribution, "severity": "advisory"},
    {"name": "A11", "fn": gate_a11_tokenizer_chunk_count, "severity": "blocking"},
    {"name": "A12", "fn": gate_a12_citation_anchor_survival, "severity": "blocking"},
    {"name": "A13", "fn": gate_a13_sentence_density, "severity": "blocking"},
    {"name": "B6", "fn": gate_b6_text_entropy_distribution, "severity": "advisory"},
]


class ModelQualitySignals:
    """
    Soft quality warnings for RAG pipeline rows.
    HTML pattern and boilerplate phrases are driven by ProbeConfig fields
    for full provenance auditability and testability.
    """

    @classmethod
    def check(
        cls,
        row: dict[str, Any],
        text_field: str = "text",
        config: ProbeConfig | None = None,
    ) -> list[tuple[str, str]]:
        """
        Return soft quality warning signals for a single record.
        Uses cfg.quality_signals_html_pattern and cfg.quality_signals_boilerplate_phrases.
        Text is capped at config.quality_signals_text_cap_chars before citation regex.
        """
        cfg = config or ProbeConfig()
        html_re = re.compile(cfg.quality_signals_html_pattern)
        signals: list[tuple[str, str]] = []
        text: str = _get_text(row) if text_field == "text" else str(row.get(text_field, ""))
        word_count_estimate = len(text.split())

        if word_count_estimate < 20:
            signals.append(("truncated_document", f"~{word_count_estimate} words — likely truncated"))
        if word_count_estimate > 100_000:
            signals.append(("gigantic_document", f"~{word_count_estimate} words — may exceed model context"))
        if html_re.search(text):
            signals.append(("html_remnants", "HTML tags detected — scraping artifact"))
        if unicodedata.normalize("NFC", text) != text:
            signals.append(("unicode_not_nfc", "Text is not NFC-normalized"))

        lower = text.lower()
        for phrase in cfg.quality_signals_boilerplate_phrases:
            if phrase in lower:
                signals.append(("boilerplate", f"Boilerplate phrase detected: {phrase!r}"))
                break

        capped_text = text[: cfg.quality_signals_text_cap_chars]
        citation_count = len(_LEGAL_CITATION_RE.findall(capped_text))
        if word_count_estimate > 100 and citation_count == 0:
            signals.append(("no_citations", "No legal citations found — may be non-opinion text"))

        return signals

    @classmethod
    def summarize(
        cls,
        records: list[dict[str, Any]],
        sample_n: int = 500,
        seed: int = 0,
        config: ProbeConfig | None = None,
    ) -> dict[str, Any]:
        """Return signal frequency counts and pct_clean. Reports records_text_capped."""
        cfg = config or ProbeConfig()
        rng = random.Random(seed)
        subsample = rng.sample(records, min(sample_n, len(records)))
        signal_counts: dict[str, int] = {}
        clean = 0
        records_text_capped = 0

        for row in subsample:
            text = _get_text(row)
            if len(text) > cfg.quality_signals_text_cap_chars:
                records_text_capped += 1
            sigs = cls.check(row, config=cfg)
            if not sigs:
                clean += 1
            for name, _ in sigs:
                signal_counts[name] = signal_counts.get(name, 0) + 1

        return {
            "subsample_n": len(subsample),
            "signal_counts": signal_counts,
            "pct_clean": round(100.0 * clean / len(subsample), 2) if subsample else 0.0,
            "records_text_capped": records_text_capped,
        }


class CourtListenerDatasetProbe:
    """Orchestrates all dataset-readiness gates."""

    def __init__(self, config: ProbeConfig | None = None) -> None:
        self.config = config or ProbeConfig()

    def run(
        self,
        data_dir: Path,
        subset: int,
        output: Path,
        seed: int = 0,
        skip_tokenizer: bool = False,
        skip_spacy: bool = False,
        full_scan: bool = False,
    ) -> "ProbeReport":
        """
        Run all gates. log_to_wandb removed — W&B is exclusively a main() concern.
        Returns a typed ProbeReport.
        """
        return run_probe(
            data_dir=data_dir,
            subset=subset,
            output=output,
            seed=seed,
            skip_tokenizer=skip_tokenizer,
            skip_spacy=skip_spacy,
            config=self.config,
            full_scan=full_scan,
        )


# ---------------------------------------------------------------------------
# run_probe orchestration helpers
# ---------------------------------------------------------------------------


def _prepare_samples(
    records: list[dict[str, Any]],
    cfg: ProbeConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Pre-compute subsamples for A11, A12, A13.
    A13 candidates are pre-filtered to records with text_length >= min_text_length.
    Returns (a11_sample, a12_sample, a13_sample).
    """
    rng = random.Random(seed + 1)
    a11_sample = rng.sample(records, min(cfg.a11_subsample_n, len(records)))
    a12_sample = rng.sample(records, min(cfg.a12_subsample_n, len(records)))
    a13_candidates = [r for r in records if _safe_int(r.get("text_length", 0)) >= cfg.min_text_length]
    a13_sample = rng.sample(a13_candidates, min(cfg.a13_subsample_n, len(a13_candidates)))
    return a11_sample, a12_sample, a13_sample


def _load_spacy_pipeline(
    cfg: ProbeConfig,
    skip_spacy: bool,
) -> tuple[Any | None, str, str]:
    """
    Load spaCy pipeline (or return None when skip_spacy=True).
    Returns (nlp|None, spacy_version, model_version).
    Catches OSError specifically for missing model/package.
    """
    if skip_spacy:
        try:
            import spacy as _spacy  # type: ignore[import]  # noqa: PLC0415

            return None, _spacy.__version__, "unknown"
        except OSError:
            return None, "unknown", "unknown"

    try:
        import spacy as _spacy  # type: ignore[import]  # noqa: PLC0415

        nlp = _spacy.load(cfg.spacy_model, exclude=SPACY_EXCLUDE)
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        nlp.max_length = 2_000_000
        return nlp, _spacy.__version__, nlp.meta.get("version", "unknown")
    except OSError:
        try:
            import spacy as _spacy  # type: ignore[import]  # noqa: PLC0415

            return None, _spacy.__version__, "unknown"
        except OSError:
            return None, "unknown", "unknown"


def _build_provenance(
    cfg: ProbeConfig,
    audit: dict[str, Any],
    spacy_version: str,
    spacy_model_version: str,
    full_scan: bool,
) -> dict[str, Any]:
    """Build the provenance dict for the probe report."""
    return {
        "probe_version": PROBE_VERSION,
        "git_sha": _get_git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spacy_version": spacy_version,
        "spacy_model_version": spacy_model_version,
        "full_scan": full_scan,
        "polars_version": pl.__version__ if pl is not None else None,
        "probe_config": _probe_config_to_dict(cfg),
    }


def _summarize_gates(gates: dict[str, Any]) -> dict[str, Any]:
    """
    Partition gate results into passed/failed_blocking/failed_advisory/skipped.
    Returns summary dict with all_passed = (no blocking failures).
    """
    passed: list[str] = []
    failed_blocking: list[str] = []
    failed_advisory: list[str] = []
    skipped: list[str] = []

    for k, v in gates.items():
        if v.get("skipped"):
            skipped.append(k)
        elif v.get("pass") is True:
            passed.append(k)
        elif v.get("pass") is False:
            if v.get("severity") == "advisory":
                failed_advisory.append(k)
            else:
                failed_blocking.append(k)

    return {
        "passed": passed,
        "failed_blocking": failed_blocking,
        "failed_advisory": failed_advisory,
        "failed": failed_blocking + failed_advisory,
        "skipped": skipped,
        "all_passed": len(failed_blocking) == 0,
    }


def _log_report_to_wandb(
    report: Any,
    entity: str,
    project: str,
    name: str,
    output: Path,
) -> None:
    """
    Log all gate metrics to W&B in a single wandb.log call and upload the
    full report as a W&B Artifact.

    Called exclusively from main() — never from run_probe().
    This function contains only W&B telemetry logic. It accesses report fields
    by key but never calls gate functions (gate_a7, gate_a8, etc.) directly.
    Captures wandb.init() return value as _run to satisfy mypy union-attr check.
    All metrics are consolidated into one dict — wandb.log is called exactly once.
    """
    if wandb is None:
        print("[dataset_probe] W&B not installed — skipping W&B logging.")
        return

    _run = wandb.init(
        project=project,
        entity=entity,
        job_type="dataset_probe",
        name=name,
        config=report["provenance"]["probe_config"],
        tags=["data_readiness", "courtlistener"],
    )

    metrics: dict[str, Any] = {
        "probe/all_passed": report["summary"]["all_passed"],
        "probe/passed_count": len(report["summary"]["passed"]),
        "probe/failed_blocking_count": len(report["summary"]["failed_blocking"]),
        "probe/failed_advisory_count": len(report["summary"]["failed_advisory"]),
        "probe/skipped_count": len(report["summary"]["skipped"]),
        "probe/total_records_decoded": report["shard_audit"]["total_records_decoded"],
        "probe/parse_errors": report["shard_audit"]["total_parse_errors"],
        "probe/subset_n": report["subset_n"],
        "probe/pct_clean": report["quality_signals"]["pct_clean"],
    }

    for gate_name, gate_result in report["gates"].items():
        if "pass" in gate_result:
            metrics[f"gate/{gate_name}/pass"] = int(gate_result["pass"])

    if "A8" in report["gates"]:
        a8 = report["gates"]["A8"]
        for key in (
            "p5",
            "p10",
            "p25",
            "p75",
            "p90",
            "p95",
            "mean",
            "median",
            "below_provisional_pct",
            "below_provisional_count",
        ):
            if key in a8:
                metrics[f"gate/A8/{key}"] = a8[key]

    if "A12" in report["gates"]:
        a12 = report["gates"]["A12"]
        metrics["gate/A12/pct_with_citation_anchor"] = a12.get("pct_with_citation_anchor", 0)
        metrics["gate/A12/mean_anchors_per_doc"] = a12.get("mean_anchors_per_doc", 0)
        if "citation_field_vs_regex" in a12:
            metrics["gate/A12/field_nonzero_regex_zero_pct"] = a12["citation_field_vs_regex"][
                "field_nonzero_regex_zero_pct"
            ]

    if "A11" in report["gates"] and "median_chunks_per_doc" in report["gates"]["A11"]:
        a11 = report["gates"]["A11"]
        metrics["gate/A11/median_chunks_per_doc"] = a11["median_chunks_per_doc"]
        metrics["gate/A11/mean_chunks_per_doc"] = a11["mean_chunks_per_doc"]
        metrics["gate/A11/multi_chunk_pct"] = a11["multi_chunk_pct"]
        metrics["gate/A11/mean_token_length"] = a11["mean_token_length"]

    if "A13" in report["gates"] and "median_sentences" in report["gates"]["A13"]:
        a13 = report["gates"]["A13"]
        metrics["gate/A13/median_sentences"] = a13["median_sentences"]
        metrics["gate/A13/below_threshold_pct"] = a13["below_threshold_pct"]
        metrics["gate/A13/records_after_a8_filter"] = a13["records_after_a8_filter"]

    if "B6" in report["gates"]:
        b6 = report["gates"]["B6"]
        for key in ("p5", "p10", "p25", "p75", "mean", "median", "zero_entropy_count"):
            if key in b6:
                metrics[f"gate/B6/{key}"] = b6[key]

    # Single consolidated log call
    wandb.log(metrics)

    artifact = wandb.Artifact(
        name="dataset_probe_report",
        type="probe_report",
        description="Full dataset readiness probe — CourtListener federal appellate corpus",
        metadata={
            "probe_version": report["provenance"]["probe_version"],
            "git_sha": report["provenance"]["git_sha"],
            "subset_n": report["subset_n"],
            "all_passed": report["summary"]["all_passed"],
        },
    )
    artifact.add_file(str(output))
    if _run is not None:
        _run.log_artifact(artifact)
    wandb.finish()
    print(f"[dataset_probe] W&B run complete — https://wandb.ai/{entity}/{project}")


def run_probe(
    data_dir: Path,
    subset: int,
    output: Path,
    seed: int = 0,
    skip_tokenizer: bool = False,
    skip_spacy: bool = False,
    config: ProbeConfig | None = None,
    full_scan: bool = False,
) -> ProbeReport:
    """
    Run all gates on a record subset. Returns a typed ProbeReport.
    No side effects on corpus shards — all output written to --output only.

    log_to_wandb is intentionally absent from this signature (obs 4/17).
    W&B telemetry is exclusively a main() concern — call _log_report_to_wandb
    after run_probe() returns from main() only.
    """
    cfg = config or ProbeConfig()

    if full_scan:
        print(f"[dataset_probe] Full scan mode — loading all records from {data_dir} via Polars ...")
        records, audit = _full_scan_with_polars(data_dir)
        print(f"[dataset_probe] Full scan loaded {len(records)} records.")
    elif cfg.stratify_by:
        print(f"[dataset_probe] Stratified sampling by '{cfg.stratify_by}' ({subset} records) from {data_dir} ...")
        records = _stratified_reservoir_sample(
            iter_shards(data_dir),
            n=subset,
            stratify_by=cfg.stratify_by,
            seed=seed,
        )
        audit = {
            "shard_count": len(sorted(data_dir.glob("*.jsonl"))),
            "total_records_decoded": len(records),
            "total_parse_errors": 0,
            "total_blank_lines": 0,
            "shard_errors": {},
            "stratified_by": cfg.stratify_by,
        }
        print(f"[dataset_probe] Loaded {len(records)} stratified records.")
    else:
        print(f"[dataset_probe] Sampling {subset} records from {data_dir} ...")
        records, audit = _reservoir_sample_with_audit(data_dir, n=subset, seed=seed)
        print(f"[dataset_probe] Loaded {len(records)} records.")

    a11_sample, a12_sample, a13_sample = _prepare_samples(records, cfg, seed)
    nlp_pipeline, spacy_version, spacy_model_version = _load_spacy_pipeline(cfg, skip_spacy)

    gates: dict[str, Any] = {}

    print("[dataset_probe] Gate: schema validation ...")
    gates["schema"] = validate_schema(records, config=cfg)
    print("[dataset_probe] Gate A7: text_source breakdown ...")
    gates["A7"] = gate_a7_text_source_breakdown(records, cfg)
    print("[dataset_probe] Gate A8: text_length distribution ...")
    gates["A8"] = gate_a8_text_length_distribution(records, cfg)
    print("[dataset_probe] Gate A9: citation_count distribution ...")
    gates["A9"] = gate_a9_citation_count_distribution(records, cfg)
    print("[dataset_probe] Gate A12: citation anchor survival ...")
    gates["A12"] = gate_a12_citation_anchor_survival(a12_sample, config=cfg)
    print("[dataset_probe] Gate B6: text_entropy distribution ...")
    gates["B6"] = gate_b6_text_entropy_distribution(records, cfg)

    if not skip_tokenizer:
        print("[dataset_probe] Gate A11: tokenizer-aware chunk count (BAAI/bge-m3) ...")
        gates["A11"] = gate_a11_tokenizer_chunk_count(a11_sample, config=cfg)
    else:
        gates["A11"] = {"gate": "A11_tokenizer_chunk_count", "skipped": True}

    if not skip_spacy:
        print("[dataset_probe] Gate A13: sentence density (spaCy) ...")
        gates["A13"] = gate_a13_sentence_density(a13_sample, config=cfg, nlp=nlp_pipeline)
    else:
        gates["A13"] = {"gate": "A13_sentence_density", "skipped": True}

    print("[dataset_probe] Quality signals ...")
    quality_signals = ModelQualitySignals.summarize(records, sample_n=cfg.quality_signals_sample_n, config=cfg)

    summary = _summarize_gates(gates)
    provenance = _build_provenance(cfg, audit, spacy_version, spacy_model_version, full_scan)

    shard_audit = {
        "shard_count": audit["shard_count"],
        "total_records_decoded": audit["total_records_decoded"],
        "total_parse_errors": audit["total_parse_errors"],
        "total_blank_lines": audit["total_blank_lines"],
        "shard_errors": audit["shard_errors"],
    }

    report = ProbeReport(
        gates=gates,
        summary=summary,
        provenance=provenance,
        quality_signals=quality_signals,
        shard_audit=shard_audit,
        subset_n=len(records),
        seed=seed,
        data_dir=str(data_dir),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    print(f"[dataset_probe] Report written → {output}")
    print(
        f"[dataset_probe] PASSED: {summary['passed']} | "
        f"FAILED_BLOCKING: {summary['failed_blocking']} | "
        f"FAILED_ADVISORY: {summary['failed_advisory']} | "
        f"SKIPPED: {summary['skipped']}"
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CourtListener dataset readiness probe (Category A + B6 gates).")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/cl_federal_appellate_bulk"))
    parser.add_argument("--subset", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("logs/dataset_probe_report.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-spacy", action="store_true")
    parser.add_argument("--ci-mode", action="store_true")
    parser.add_argument("--log-to-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default="phl690-harvard-extension-schol")
    parser.add_argument("--wandb-project", type=str, default="cs1090b")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument(
        "--skip-generative-tokenizer",
        action="store_true",
        help="Skip the Mistral-7B secondary tokenizer check in A11.",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Use Polars scan_ndjson to load all records for exact statistics.",
    )

    args = parser.parse_args()

    _probe_defaults = ProbeConfig()
    cfg = dataclasses.replace(
        _probe_defaults,
        a11_generative_model="" if args.skip_generative_tokenizer else _probe_defaults.a11_generative_model,
    )

    # W&B is not passed to run_probe — exclusively a main() concern.
    report = run_probe(
        data_dir=args.data_dir,
        subset=args.subset,
        output=args.output,
        seed=args.seed,
        skip_tokenizer=args.skip_tokenizer,
        skip_spacy=args.skip_spacy,
        config=cfg,
        full_scan=args.full_scan,
    )

    if args.log_to_wandb:
        run_name = args.wandb_name or f"dataset_probe_v{PROBE_VERSION}_{args.subset // 1000}k"
        _log_report_to_wandb(
            report=report,
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            output=args.output,
        )

    if args.ci_mode and not report["summary"]["all_passed"]:
        print(f"[dataset_probe] CI mode: blocking gate failures — {report['summary']['failed_blocking']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
