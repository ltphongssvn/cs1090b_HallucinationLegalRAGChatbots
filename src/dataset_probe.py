# src/dataset_probe.py
"""
Dataset readiness probe for local CourtListener federal appellate JSONL shards.

Gate dependency graph and failure semantics
-------------------------------------------
Gate independence:
  A7, A8, A9, A12, B6 — independent; run on the full sampled record set.
  A11 — independent; runs on a11_subsample_n records.
  A13 — depends on A8: evaluates sentence density only on records that pass
        the A8 text_length >= min_text_length filter.
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
  uv run python -m src.dataset_probe ... --skip-generative-tokenizer
  uv run python -m src.dataset_probe ... --log-to-wandb \\
      --wandb-entity phl690-harvard-extension-schol \\
      --wandb-project cs1090b \\
      --wandb-name dataset_probe_v2.5.5_10k

No side effects on corpus shards — all output written to --output only.
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

import spacy as spacy  # type: ignore[import]

try:
    import wandb  # type: ignore[import]
except ImportError:
    wandb = None  # type: ignore[assignment]

from transformers import AutoTokenizer  # type: ignore[import]

# ---------------------------------------------------------------------------
# Probe version
# CHANGED: bumped to 2.5.5 — added --skip-generative-tokenizer CLI flag;
# added warning print when log_to_wandb=True but wandb.run is None.
# ---------------------------------------------------------------------------

PROBE_VERSION = "2.5.5"

# ---------------------------------------------------------------------------
# Shared legal citation regex
#
# Match examples:
#   "123 F.3d 456"     — federal reporter, third series
#   "456 F.2d 789"     — federal reporter, second series
#   "123 F.Supp 456"   — federal supplement reporter
#   "347 U.S. 483"     — United States Reports
#   "Smith v. Jones"   — case name citation anchor
#   "Brown v. Board"   — landmark SCOTUS case name anchor
#
# Non-match examples:
#   "The defendant argued the motion." — pure prose
#   "Section 42 of the statute"        — bare number
# ---------------------------------------------------------------------------

_LEGAL_CITATION_RE = re.compile(
    r"(\d+\s+[A-Z][a-z]*\.?\s*(?:\d+d?|App\.?|Supp\.?)"
    r"|[A-Z][a-z]+\s+v\.\s+[A-Z]"
    r"|U\.S\.\s+\d+"
    r"|\d+\s+F\.\d+[a-z]?\s+\d+)",
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
    # Set to "" to skip the Mistral secondary tokenizer check in A11.
    # Use --skip-generative-tokenizer CLI flag to set this at runtime.
    a11_generative_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    text_length_consistency_tolerance: int = 200


def _probe_config_to_dict(cfg: ProbeConfig) -> dict[str, Any]:
    """Serialize ProbeConfig to a JSON-safe dict — converts frozensets to sorted lists."""
    result: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        result[f.name] = sorted(val) if isinstance(val, frozenset) else val
    return result


PROVISIONAL_MIN_TEXT_LENGTH = ProbeConfig().min_text_length
CHUNK_SIZE_SUBWORDS = ProbeConfig().chunk_size_subwords
CHUNK_OVERLAP_SUBWORDS = ProbeConfig().chunk_overlap_subwords
ENCODER_MODEL = ProbeConfig().encoder_model
SPACY_MODEL = ProbeConfig().spacy_model
SPACY_EXCLUDE = ["ner", "parser", "lemmatizer"]
MIN_SENTENCE_COUNT = ProbeConfig().min_sentence_count


def _get_git_sha() -> str:
    """Return current git commit SHA, or 'not-a-git-repo' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
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
    Reservoir-sample n records from all shards without loading full corpus.
    JSON parse errors are silently dropped with no counting in this path —
    use iter_shards_with_audit() to obtain parse error counts and audit stats.
    """
    reservoir: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for i, record in enumerate(iter_shards(data_dir)):
        if i < n:
            reservoir.append(record)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = record
    return reservoir


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


def validate_schema(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """
    Check required field presence, type, range, vocabulary, consistency, and
    documented field coverage. Accepts optional config for tolerance settings.
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
            "pass": True,
            "note": "No records to validate.",
        }

    missing_by_field: dict[str, int] = {f: 0 for f in MIN_REQUIRED_FIELDS}
    type_errors: dict[str, int] = {}
    range_errors: dict[str, int] = {}
    vocabulary_errors: dict[str, int] = {}
    consistency_errors: dict[str, int] = {}
    documented_only = DOCUMENTED_FIELDS - MIN_REQUIRED_FIELDS
    missing_documented: dict[str, int] = {}

    for r in records:
        for f in MIN_REQUIRED_FIELDS:
            if f not in r:
                missing_by_field[f] += 1

        for f in documented_only:
            if f not in r:
                missing_documented[f] = missing_documented.get(f, 0) + 1

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

        text_source = r.get("text_source")
        if text_source is not None and str(text_source) not in KNOWN_TEXT_SOURCES:
            vocabulary_errors["text_source"] = vocabulary_errors.get("text_source", 0) + 1

        actual_text = r.get("text")
        if text_len is not None and isinstance(text_len, (int, float)) and actual_text is not None:
            actual_len = len(str(actual_text))
            if abs(int(text_len) - actual_len) > cfg.text_length_consistency_tolerance:
                consistency_errors["text_length_consistency"] = consistency_errors.get("text_length_consistency", 0) + 1

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
        "pass": passed,
    }


def gate_a7_text_source_breakdown(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """A7 — text_source breakdown. severity=blocking."""
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "A7_text_source_breakdown", "severity": "blocking", "pass": False, "note": "No records."}

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
    """A8 — text_length distribution. severity=blocking."""
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "A8_text_length_distribution", "severity": "blocking", "pass": False, "note": "No records."}

    lengths = [int(r.get("text_length", 0)) for r in records]
    lengths_sorted = sorted(lengths)
    below_provisional = sum(1 for length in lengths if length < cfg.min_text_length)
    return {
        "gate": "A8_text_length_distribution",
        "severity": "blocking",
        "count": len(lengths),
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
    """A9 — citation_count distribution. severity=advisory."""
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "A9_citation_count_distribution", "severity": "advisory", "pass": False, "note": "No records."}

    counts = [int(r.get("citation_count", 0)) for r in records]
    n = len(counts)
    zero = sum(1 for c in counts if c == 0)
    above_5 = sum(1 for c in counts if c > 5)
    return {
        "gate": "A9_citation_count_distribution",
        "severity": "advisory",
        "count": n,
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
    """A11 — Tokenizer-aware chunk count. severity=blocking."""
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "A11_tokenizer_chunk_count", "severity": "blocking", "pass": False, "note": "No records."}

    tokenizer_revision = cfg.encoder_model
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_model)
            tokenizer_revision = getattr(tokenizer, "name_or_path", cfg.encoder_model)
        except Exception as exc:
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
                "model": cfg.a11_generative_model,
                "mean_tokens": round(statistics.mean(gen_lengths), 1),
                "median_tokens": statistics.median(gen_lengths),
                "max_tokens": max(gen_lengths),
                "over_32k_count": over_limit,
                "over_32k_pct": round(100.0 * over_limit / len(records), 2),
                "note": "README asserts max(prompt_tokens) < 32768 using Mistral tokenizer.",
            }
        except Exception as exc:
            result["generative_token_check"] = {
                "model": cfg.a11_generative_model,
                "error": str(exc),
                "note": "Generative tokenizer load failed — skipping secondary check.",
            }

    return result


def gate_a12_citation_anchor_survival(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """A12 — Citation anchor survival. severity=blocking. Text capped before regex."""
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "A12_citation_anchor_survival", "severity": "blocking", "pass": False, "note": "No records."}

    has_anchor = 0
    citation_counts_found: list[int] = []
    field_nonzero_regex_zero = 0
    field_nonzero_total = 0

    for r in records:
        text = _get_text(r)[: cfg.a12_text_cap_chars]
        matches = _LEGAL_CITATION_RE.findall(text)
        n_matches = len(matches)
        citation_counts_found.append(n_matches)
        if matches:
            has_anchor += 1
        field_count = int(r.get("citation_count", 0))
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


def gate_a13_sentence_density(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
    nlp: Any | None = None,
) -> dict[str, Any]:
    """A13 — Sentence density on A8-filtered records. severity=blocking."""
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "A13_sentence_density", "severity": "blocking", "pass": False, "note": "No records."}

    spacy_version = "injected"
    if nlp is None:
        try:
            nlp = spacy.load(cfg.spacy_model, exclude=SPACY_EXCLUDE)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            nlp.max_length = 2_000_000
            spacy_version = spacy.__version__
        except Exception as exc:
            return {
                "gate": "A13_sentence_density",
                "severity": "blocking",
                "pass": False,
                "error": str(exc),
                "note": f"spaCy model load failed: {exc}",
            }

    substantive = [r for r in records if int(r.get("text_length", 0)) >= cfg.min_text_length]
    if not substantive:
        return {
            "gate": "A13_sentence_density",
            "severity": "blocking",
            "pass": False,
            "records_after_a8_filter": 0,
            "note": "No records pass A8 length filter.",
        }

    sent_counts: list[int] = []
    below_threshold = 0
    for r in substantive:
        text = _get_text(r)[: cfg.a13_text_cap_chars]
        doc = nlp(text)
        n_sents = sum(1 for _ in doc.sents)
        sent_counts.append(n_sents)
        if n_sents < cfg.min_sentence_count:
            below_threshold += 1

    return {
        "gate": "A13_sentence_density",
        "severity": "blocking",
        "spacy_model": cfg.spacy_model,
        "spacy_version": spacy_version,
        "min_sentence_threshold": cfg.min_sentence_count,
        "subsample_n": len(substantive),
        "records_after_a8_filter": len(substantive),
        "mean_sentences": round(statistics.mean(sent_counts), 1),
        "median_sentences": statistics.median(sent_counts),
        "min_sentences": min(sent_counts),
        "below_threshold_count": below_threshold,
        "below_threshold_pct": round(100.0 * below_threshold / len(substantive), 2),
        "pass": below_threshold / len(substantive) < cfg.a13_max_below_threshold_pct / 100.0,
        "note": (
            "Evaluated on text_length >= 1500 records only (A8-filtered). Pass threshold: <15% below 20 sentences."
        ),
    }


def gate_b6_text_entropy_distribution(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """B6 — text_entropy distribution. severity=advisory — always passes."""
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "B6_text_entropy_distribution", "severity": "advisory", "pass": True, "note": "No records."}

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


class ModelQualitySignals:
    """
    Soft quality warnings for RAG pipeline rows.
    word_count_estimate: whitespace-split word count — NOT HF subword token count.
    HF subword tokens computed in gate_a11 via AutoTokenizer only.
    """

    HTML_RE = re.compile(r"<[a-zA-Z][^>]{0,100}>")
    BOILERPLATE_PHRASES = (
        "all rights reserved",
        "this page intentionally left blank",
        "unpublished disposition",
        "not for publication",
        "do not cite",
    )

    @classmethod
    def check(
        cls,
        row: dict[str, Any],
        text_field: str = "text",
        config: ProbeConfig | None = None,
    ) -> list[tuple[str, str]]:
        """
        Return soft quality warning signals for a single record.
        Text is capped at config.quality_signals_text_cap_chars before citation regex.
        """
        cfg = config or ProbeConfig()
        signals: list[tuple[str, str]] = []
        text: str = _get_text(row) if text_field == "text" else str(row.get(text_field, ""))
        word_count_estimate = len(text.split())

        if word_count_estimate < 20:
            signals.append(("truncated_document", f"~{word_count_estimate} words — likely truncated"))
        if word_count_estimate > 100_000:
            signals.append(("gigantic_document", f"~{word_count_estimate} words — may exceed model context"))
        if cls.HTML_RE.search(text):
            signals.append(("html_remnants", "HTML tags detected — scraping artifact"))
        if unicodedata.normalize("NFC", text) != text:
            signals.append(("unicode_not_nfc", "Text is not NFC-normalized"))

        lower = text.lower()
        for phrase in cls.BOILERPLATE_PHRASES:
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
    ) -> dict[str, Any]:
        """Return signal frequency counts and pct_clean for a record sample."""
        rng = random.Random(seed)
        subsample = rng.sample(records, min(sample_n, len(records)))
        signal_counts: dict[str, int] = {}
        clean = 0
        for row in subsample:
            sigs = cls.check(row)
            if not sigs:
                clean += 1
            for name, _ in sigs:
                signal_counts[name] = signal_counts.get(name, 0) + 1
        return {
            "subsample_n": len(subsample),
            "signal_counts": signal_counts,
            "pct_clean": round(100.0 * clean / len(subsample), 2) if subsample else 0.0,
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
        log_to_wandb: bool = False,
    ) -> dict[str, Any]:
        return run_probe(
            data_dir=data_dir,
            subset=subset,
            output=output,
            seed=seed,
            skip_tokenizer=skip_tokenizer,
            skip_spacy=skip_spacy,
            config=self.config,
            log_to_wandb=log_to_wandb,
        )


def _log_report_to_wandb(
    report: dict[str, Any],
    entity: str,
    project: str,
    name: str,
    output: Path,
) -> None:
    """Initialize a W&B run, log all gate metrics and the full report artifact."""
    if wandb is None:
        print("[dataset_probe] W&B not installed — skipping W&B logging.")
        return

    run = wandb.init(
        project=project,
        entity=entity,
        job_type="dataset_probe",
        name=name,
        config=report["provenance"]["probe_config"],
        tags=["data_readiness", "courtlistener"],
    )

    wandb.log(
        {
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
    )

    for gate_name, gate_result in report["gates"].items():
        if "pass" in gate_result:
            wandb.log({f"gate/{gate_name}/pass": int(gate_result["pass"])})

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
                wandb.log({f"gate/A8/{key}": a8[key]})

    if "A12" in report["gates"]:
        a12 = report["gates"]["A12"]
        wandb.log(
            {
                "gate/A12/pct_with_citation_anchor": a12.get("pct_with_citation_anchor", 0),
                "gate/A12/mean_anchors_per_doc": a12.get("mean_anchors_per_doc", 0),
                "gate/A12/field_nonzero_regex_zero_pct": (
                    a12["citation_field_vs_regex"]["field_nonzero_regex_zero_pct"]
                ),
            }
        )

    if "A11" in report["gates"] and "median_chunks_per_doc" in report["gates"]["A11"]:
        a11 = report["gates"]["A11"]
        wandb.log(
            {
                "gate/A11/median_chunks_per_doc": a11["median_chunks_per_doc"],
                "gate/A11/mean_chunks_per_doc": a11["mean_chunks_per_doc"],
                "gate/A11/multi_chunk_pct": a11["multi_chunk_pct"],
                "gate/A11/mean_token_length": a11["mean_token_length"],
            }
        )

    if "A13" in report["gates"] and "median_sentences" in report["gates"]["A13"]:
        a13 = report["gates"]["A13"]
        wandb.log(
            {
                "gate/A13/median_sentences": a13["median_sentences"],
                "gate/A13/below_threshold_pct": a13["below_threshold_pct"],
                "gate/A13/records_after_a8_filter": a13["records_after_a8_filter"],
            }
        )

    if "B6" in report["gates"]:
        b6 = report["gates"]["B6"]
        for key in ("p5", "p10", "p25", "p75", "mean", "median", "zero_entropy_count"):
            if key in b6:
                wandb.log({f"gate/B6/{key}": b6[key]})

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
    run.log_artifact(artifact)
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
    log_to_wandb: bool = False,
) -> dict[str, Any]:
    """Run all gates on a reservoir-sampled subset. No side effects on corpus shards."""
    cfg = config or ProbeConfig()

    print(f"[dataset_probe] Sampling {subset} records from {data_dir} ...")
    records, audit = _reservoir_sample_with_audit(data_dir, n=subset, seed=seed)
    print(f"[dataset_probe] Loaded {len(records)} records.")

    rng = random.Random(seed + 1)
    a11_sample = rng.sample(records, min(cfg.a11_subsample_n, len(records)))
    a12_sample = rng.sample(records, min(cfg.a12_subsample_n, len(records)))
    a13_candidates = [r for r in records if int(r.get("text_length", 0)) >= cfg.min_text_length]
    a13_sample = rng.sample(a13_candidates, min(cfg.a13_subsample_n, len(a13_candidates)))

    spacy_version = "unknown"
    spacy_model_version = "unknown"
    nlp_pipeline: Any | None = None

    if not skip_spacy:
        try:
            nlp_pipeline = spacy.load(cfg.spacy_model, exclude=SPACY_EXCLUDE)
            if "sentencizer" not in nlp_pipeline.pipe_names:
                nlp_pipeline.add_pipe("sentencizer")
            nlp_pipeline.max_length = 2_000_000
            spacy_version = spacy.__version__
            spacy_model_version = nlp_pipeline.meta.get("version", "unknown")
        except Exception:
            nlp_pipeline = None
    else:
        try:
            spacy_version = spacy.__version__
        except Exception:
            pass

    report: dict[str, Any] = {
        "data_dir": str(data_dir),
        "subset_n": len(records),
        "seed": seed,
        "shard_audit": {
            "shard_count": audit["shard_count"],
            "total_records_decoded": audit["total_records_decoded"],
            "total_parse_errors": audit["total_parse_errors"],
            "total_blank_lines": audit["total_blank_lines"],
            "shard_errors": audit["shard_errors"],
        },
        "provenance": {
            "probe_version": PROBE_VERSION,
            "git_sha": _get_git_sha(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "spacy_version": spacy_version,
            "spacy_model_version": spacy_model_version,
            "probe_config": _probe_config_to_dict(cfg),
        },
        "gates": {},
    }

    print("[dataset_probe] Gate: schema validation ...")
    report["gates"]["schema"] = validate_schema(records, config=cfg)
    print("[dataset_probe] Gate A7: text_source breakdown ...")
    report["gates"]["A7"] = gate_a7_text_source_breakdown(records, cfg)
    print("[dataset_probe] Gate A8: text_length distribution ...")
    report["gates"]["A8"] = gate_a8_text_length_distribution(records, cfg)
    print("[dataset_probe] Gate A9: citation_count distribution ...")
    report["gates"]["A9"] = gate_a9_citation_count_distribution(records, cfg)
    print("[dataset_probe] Gate A12: citation anchor survival ...")
    report["gates"]["A12"] = gate_a12_citation_anchor_survival(a12_sample, config=cfg)
    print("[dataset_probe] Gate B6: text_entropy distribution ...")
    report["gates"]["B6"] = gate_b6_text_entropy_distribution(records, cfg)

    if not skip_tokenizer:
        print("[dataset_probe] Gate A11: tokenizer-aware chunk count (BAAI/bge-m3) ...")
        report["gates"]["A11"] = gate_a11_tokenizer_chunk_count(a11_sample, config=cfg)
    else:
        report["gates"]["A11"] = {"gate": "A11_tokenizer_chunk_count", "skipped": True}

    if not skip_spacy:
        print("[dataset_probe] Gate A13: sentence density (spaCy) ...")
        report["gates"]["A13"] = gate_a13_sentence_density(a13_sample, config=cfg, nlp=nlp_pipeline)
    else:
        report["gates"]["A13"] = {"gate": "A13_sentence_density", "skipped": True}

    print("[dataset_probe] Quality signals ...")
    report["quality_signals"] = ModelQualitySignals.summarize(records, sample_n=cfg.quality_signals_sample_n)

    passed: list[str] = []
    failed_blocking: list[str] = []
    failed_advisory: list[str] = []
    skipped: list[str] = []

    for k, v in report["gates"].items():
        if v.get("skipped"):
            skipped.append(k)
        elif v.get("pass") is True:
            passed.append(k)
        elif v.get("pass") is False:
            if v.get("severity") == "advisory":
                failed_advisory.append(k)
            else:
                failed_blocking.append(k)

    report["summary"] = {
        "passed": passed,
        "failed_blocking": failed_blocking,
        "failed_advisory": failed_advisory,
        "failed": failed_blocking + failed_advisory,
        "skipped": skipped,
        "all_passed": len(failed_blocking) == 0,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[dataset_probe] Report written → {output}")
    print(
        f"[dataset_probe] PASSED: {passed} | "
        f"FAILED_BLOCKING: {failed_blocking} | "
        f"FAILED_ADVISORY: {failed_advisory} | "
        f"SKIPPED: {skipped}"
    )

    # CHANGED: guard wandb.run access — when log_to_wandb=True but wandb.run
    # is None (no active run), emit a visible warning rather than silently
    # doing nothing. This prevents confusion when callers expect W&B logs but
    # forgot to call wandb.init() before run_probe.
    if log_to_wandb:
        if wandb is None:
            print("[dataset_probe] WARNING: log_to_wandb=True but wandb is not installed — logging skipped.")
        elif wandb.run is None:
            print(
                "[dataset_probe] WARNING: log_to_wandb=True but no active wandb run detected. "
                "Call wandb.init() before run_probe, or use --log-to-wandb CLI flag which "
                "handles wandb.init() automatically."
            )
        else:
            wandb.log(
                {
                    "probe/passed_gates": len(passed),
                    "probe/failed_blocking": len(failed_blocking),
                    "probe/failed_advisory": len(failed_advisory),
                    "probe/all_passed": report["summary"]["all_passed"],
                    "probe/parse_errors": audit["total_parse_errors"],
                    **{f"probe/quality/{k}": v for k, v in report["quality_signals"]["signal_counts"].items()},
                    "probe/pct_clean": report["quality_signals"]["pct_clean"],
                }
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
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="Exit 1 if any BLOCKING gate fails. Advisory failures do not trigger exit 1.",
    )
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Log gate metrics and report artifact to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="phl690-harvard-extension-schol",
        help="W&B entity. Default: phl690-harvard-extension-schol",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cs1090b",
        help="W&B project name. Default: cs1090b",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run name. Default: dataset_probe_v{PROBE_VERSION}_{subset}k",
    )
    # ADDED: --skip-generative-tokenizer — sets a11_generative_model="" in the
    # ProbeConfig, skipping the Mistral-7B secondary tokenizer check in A11.
    # The Mistral tokenizer requires a ~1GB download and network access.
    # Use this flag in CI or minimal environments to avoid the download.
    # Consistent with --skip-tokenizer and --skip-spacy flag pattern.
    parser.add_argument(
        "--skip-generative-tokenizer",
        action="store_true",
        help=(
            "Skip the Mistral-7B secondary tokenizer check in A11. "
            "Sets a11_generative_model='' in ProbeConfig. "
            "Use in CI or minimal environments to avoid the ~1GB tokenizer download."
        ),
    )

    args = parser.parse_args()

    # Build config — apply --skip-generative-tokenizer if set
    cfg = ProbeConfig(a11_generative_model="" if args.skip_generative_tokenizer else ProbeConfig().a11_generative_model)

    report = run_probe(
        data_dir=args.data_dir,
        subset=args.subset,
        output=args.output,
        seed=args.seed,
        skip_tokenizer=args.skip_tokenizer,
        skip_spacy=args.skip_spacy,
        config=cfg,
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
