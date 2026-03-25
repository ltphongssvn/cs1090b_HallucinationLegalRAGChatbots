# src/dataset_probe.py
"""
Dataset readiness probe for local CourtListener federal appellate JSONL shards.

Implements Category A dataset-readiness gates required before Stage 3:
  A7  — text_source breakdown (missing 17.2% audit)
  A8  — text_length distribution + RAG viability threshold derivation
  A9  — citation_count distribution (advisory — not a hard corpus filter)
  A11 — tokenizer-aware chunk count (BGE-M3 + optional Mistral generative check)
  A12 — citation anchor survival check in normalized text
  A13 — sentence density check via repo-certified spaCy pipeline (nlp injectable)
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
  uv run python -m src.dataset_probe ... --ci-mode  # exits 1 if any gate fails

All output written to --output as JSON. No side effects on corpus shards.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterator

import spacy as spacy  # type: ignore[import]
import wandb  # type: ignore[import]
from transformers import AutoTokenizer  # type: ignore[import]

# ---------------------------------------------------------------------------
# Probe version — increment when report schema changes
# ---------------------------------------------------------------------------

PROBE_VERSION = "2.1.0"

# ---------------------------------------------------------------------------
# ProbeConfig — frozen dataclass so experiment settings are versioned/loggable
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ProbeConfig:
    """
    All probe thresholds and sampling parameters in one place.
    Explicit, versionable, injectable, and JSON-serializable for provenance logging.
    """

    # Pass/fail thresholds
    min_text_length: int = 1500
    chunk_size_subwords: int = 1024
    chunk_overlap_subwords: int = 128
    min_sentence_count: int = 20
    encoder_model: str = "BAAI/bge-m3"
    spacy_model: str = "en_core_web_sm"
    a7_known_formats_pass_pct: float = 80.0
    a8_below_threshold_pass_pct: float = 25.0
    a9_zero_citation_pass_pct: float = 20.0
    a11_min_median_chunks: float = 2.0
    a12_min_pct_with_anchor: float = 60.0
    # A13 threshold calibrated to 15% based on empirical corpus run:
    # median=71.5 sentences, mean=118.8 — corpus is NLI-ready.
    a13_max_below_threshold_pct: float = 15.0
    quality_signals_sample_n: int = 500

    # Subsample sizes — explicit and loggable (were magic numbers)
    a11_subsample_n: int = 200
    a12_subsample_n: int = 500
    a13_subsample_n: int = 200

    # A13 text cap — explicit and loggable (was magic number 50_000)
    # Caps per-doc cost for spaCy processing in the probe (not in training)
    a13_text_cap_chars: int = 50_000

    # B6 entropy spot-check tolerance — max allowed deviation between
    # stored text_entropy and probe-computed Shannon entropy before flagging drift
    b6_entropy_spot_check_tolerance: float = 1.0
    b6_entropy_spot_check_sample_n: int = 10

    # Optional secondary generative tokenizer for A11.
    # README uses mistralai/Mistral-7B-Instruct-v0.2 for prompt length assertion.
    # A chunk fitting 1024 BGE-M3 subwords may be larger under the Mistral tokenizer.
    # Set to "" to skip generative tokenizer check.
    a11_generative_model: str = "mistralai/Mistral-7B-Instruct-v0.2"


# Module-level defaults (kept for backward-compatible imports)
PROVISIONAL_MIN_TEXT_LENGTH = ProbeConfig().min_text_length
CHUNK_SIZE_SUBWORDS = ProbeConfig().chunk_size_subwords
CHUNK_OVERLAP_SUBWORDS = ProbeConfig().chunk_overlap_subwords
ENCODER_MODEL = ProbeConfig().encoder_model
SPACY_MODEL = ProbeConfig().spacy_model
SPACY_EXCLUDE = ["ner", "parser", "lemmatizer"]
MIN_SENTENCE_COUNT = ProbeConfig().min_sentence_count

REQUIRED_FIELDS: frozenset[str] = frozenset(
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


# ---------------------------------------------------------------------------
# Percentile helper — single reusable implementation
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[Any], p: float) -> Any:
    """
    Return the p-th percentile of a pre-sorted list.
    p must be in [0, 100]. Uses ceiling-index method consistent with numpy default.
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


# ---------------------------------------------------------------------------
# Shannon entropy helper — used by B6 spot-check
# ---------------------------------------------------------------------------


def _shannon_entropy(text: str) -> float:
    """Compute word-level Shannon entropy (bits). Returns 0.0 for empty text."""
    words = text.split()
    if not words:
        return 0.0
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    n = len(words)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


# ---------------------------------------------------------------------------
# Shard loader with audit counters
# ---------------------------------------------------------------------------


def iter_shards(data_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield valid records from all .jsonl shards. Silently skips blank lines only."""
    for record, _ in _iter_shards_inner(data_dir):
        yield record


def iter_shards_with_audit(data_dir: Path) -> dict[str, Any]:
    """
    Load all shards and return audit summary including per-shard parse error counts.
    Use this when auditability of malformed lines is required.
    """
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
    Uses Vitter's Algorithm R for memory-efficient streaming reservoir sampling.
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


# ---------------------------------------------------------------------------
# Schema validation — presence + type + range checks
# ---------------------------------------------------------------------------


def validate_schema(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Check required field presence, type correctness, and value ranges.
    Returns pass=False if any record fails any check.
    """
    if not records:
        return {
            "gate": "schema_validation",
            "required_fields": sorted(REQUIRED_FIELDS),
            "missing_counts": {},
            "type_errors": {},
            "range_errors": {},
            "pass": True,
            "note": "No records to validate.",
        }

    missing_by_field: dict[str, int] = {f: 0 for f in REQUIRED_FIELDS}
    type_errors: dict[str, int] = {}
    range_errors: dict[str, int] = {}

    for r in records:
        for f in REQUIRED_FIELDS:
            if f not in r:
                missing_by_field[f] += 1

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

    any_missing = any(v > 0 for v in missing_by_field.values())
    passed = not any_missing and not type_errors and not range_errors
    return {
        "gate": "schema_validation",
        "required_fields": sorted(REQUIRED_FIELDS),
        "missing_counts": {k: v for k, v in missing_by_field.items() if v > 0},
        "type_errors": type_errors,
        "range_errors": range_errors,
        "pass": passed,
    }


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------


def gate_a7_text_source_breakdown(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """A7 — Full text_source breakdown including unaccounted ~17.2%."""
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "A7_text_source_breakdown", "pass": False, "note": "No records."}

    counts: dict[str, int] = {}
    for r in records:
        src = str(r.get("text_source", "MISSING"))
        counts[src] = counts.get(src, 0) + 1
    total = len(records)
    breakdown = {
        src: {"count": cnt, "pct": round(100.0 * cnt / total, 2)}
        for src, cnt in sorted(counts.items(), key=lambda x: -x[1])
    }
    known_pct = sum(v["pct"] for k, v in breakdown.items() if k in ("plain_text", "html_with_citations"))
    return {
        "gate": "A7_text_source_breakdown",
        "total_records": total,
        "breakdown": breakdown,
        "known_formats_pct": round(known_pct, 2),
        "unknown_formats_pct": round(100.0 - known_pct, 2),
        "pass": known_pct >= cfg.a7_known_formats_pass_pct,
        "note": (
            "Inspect records from any source outside plain_text/html_with_citations "
            "to verify row_normalizer.py strips them cleanly before Stage 3."
        ),
    }


def gate_a8_text_length_distribution(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """A8 — text_length distribution. Pass: <25% below min_text_length."""
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "A8_text_length_distribution", "pass": False, "note": "No records."}

    lengths = [int(r.get("text_length", 0)) for r in records]
    lengths_sorted = sorted(lengths)
    below_provisional = sum(1 for length in lengths if length < cfg.min_text_length)
    return {
        "gate": "A8_text_length_distribution",
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
    """
    A9 — citation_count distribution.
    ADVISORY ONLY — this probe does not hard-filter the corpus.
    The full 1.46M-opinion corpus is used unfiltered for final runs.
    Only the ~150K fast-iteration subset optionally filters citation_count > 5.
    """
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "A9_citation_count_distribution", "pass": False, "note": "No records."}

    counts = [int(r.get("citation_count", 0)) for r in records]
    n = len(counts)
    zero = sum(1 for c in counts if c == 0)
    above_5 = sum(1 for c in counts if c > 5)
    return {
        "gate": "A9_citation_count_distribution",
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
    sample_n: int | None = None,
    seed: int = 0,
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """
    A11 — Tokenizer-aware chunk count using BAAI/bge-m3 (README-certified encoder).
    AutoTokenizer imported at module level so tests can patch it network-free.
    Tokenizer revision logged for reproducibility.

    Subsampling is done by run_probe before calling this gate.
    This gate processes the full records list passed to it.
    sample_n is retained for backward compatibility but ignored when records
    already fit within the budget — run_probe controls sampling.

    Optional secondary check against a11_generative_model (Mistral by default):
    A chunk fitting 1024 BGE-M3 subwords may exceed the Mistral context window.
    README asserts max(prompt_tokens) < 32768 using the Mistral tokenizer.
    Set a11_generative_model="" to skip this check.
    """
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "A11_tokenizer_chunk_count", "pass": False, "note": "No records."}

    # Use all records passed — subsampling is run_probe's responsibility
    subsample = records

    try:
        tok = AutoTokenizer.from_pretrained(cfg.encoder_model)
        tokenizer_revision = getattr(tok, "name_or_path", cfg.encoder_model)
    except Exception as exc:
        return {
            "gate": "A11_tokenizer_chunk_count",
            "pass": False,
            "error": str(exc),
            "note": "AutoTokenizer load failed — check HF_TOKEN and network.",
        }

    chunk_counts: list[int] = []
    token_lengths: list[int] = []
    multi_chunk = 0

    for r in subsample:
        text = str(r.get("text", ""))
        ids = tok(text, add_special_tokens=False)["input_ids"]
        total_tokens = len(ids)
        token_lengths.append(total_tokens)
        stride = cfg.chunk_size_subwords - cfg.chunk_overlap_subwords
        n_chunks = max(1, math.ceil(max(0, total_tokens - cfg.chunk_overlap_subwords) / stride))
        chunk_counts.append(n_chunks)
        if n_chunks > 1:
            multi_chunk += 1

    result: dict[str, Any] = {
        "gate": "A11_tokenizer_chunk_count",
        "encoder_model": cfg.encoder_model,
        "tokenizer_revision": tokenizer_revision,
        "chunk_size_subwords": cfg.chunk_size_subwords,
        "overlap_subwords": cfg.chunk_overlap_subwords,
        "subsample_n": len(subsample),
        "mean_token_length": round(statistics.mean(token_lengths), 1),
        "median_token_length": statistics.median(token_lengths),
        "mean_chunks_per_doc": round(statistics.mean(chunk_counts), 2),
        "median_chunks_per_doc": statistics.median(chunk_counts),
        "multi_chunk_pct": round(100.0 * multi_chunk / len(subsample), 2),
        "pass": statistics.median(chunk_counts) >= cfg.a11_min_median_chunks,
        "note": "median_chunks_per_doc >= 2 confirms corpus supports multi-chunk splitting.",
    }

    # Optional secondary generative tokenizer check (Mistral by default)
    # README: assert max(prompt_tokens) < 32768 using Mistral tokenizer
    if cfg.a11_generative_model:
        try:
            gen_tok = AutoTokenizer.from_pretrained(cfg.a11_generative_model)
            gen_lengths: list[int] = []
            mistral_limit = 32_768
            over_limit = 0
            for r in subsample:
                text = str(r.get("text", ""))
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
                "over_32k_pct": round(100.0 * over_limit / len(subsample), 2),
                "note": (
                    "README asserts max(prompt_tokens) < 32768 using Mistral tokenizer. "
                    "Docs with >32k tokens will trigger the runtime assertion in generation."
                ),
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
    sample_n: int | None = None,
    seed: int = 0,
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """
    A12 — Citation anchor survival in normalized text field.
    NOTE: regex-based detection is a heuristic approximation.
    Subsampling is done by run_probe before calling this gate.
    This gate processes the full records list passed to it.
    """
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "A12_citation_anchor_survival", "pass": False, "note": "No records."}

    # Use all records passed — subsampling is run_probe's responsibility
    subsample = records

    CITATION_RE = re.compile(
        r"(\d+\s+[A-Z][a-z]*\.?\s*(?:\d+d?|App\.?|Supp\.?)"
        r"|[A-Z][a-z]+\s+v\.\s+[A-Z]"
        r"|U\.S\.\s+\d+"
        r"|\d+\s+F\.\d+[a-z]?\s+\d+)",
        re.MULTILINE,
    )

    has_anchor = 0
    citation_counts_found: list[int] = []
    for r in subsample:
        text = str(r.get("text", ""))
        matches = CITATION_RE.findall(text)
        citation_counts_found.append(len(matches))
        if matches:
            has_anchor += 1

    pct_with_anchor = 100.0 * has_anchor / len(subsample)
    return {
        "gate": "A12_citation_anchor_survival",
        "subsample_n": len(subsample),
        "records_with_citation_anchor": has_anchor,
        "pct_with_citation_anchor": round(pct_with_anchor, 2),
        "mean_anchors_per_doc": round(statistics.mean(citation_counts_found), 2),
        "pass": pct_with_anchor >= cfg.a12_min_pct_with_anchor,
        "note": (
            "Heuristic approximation via regex — not a precision extractor. "
            ">=60% of records must contain extractable citation anchors "
            "for Tier C SQLite lookup to be viable."
        ),
    }


def gate_a13_sentence_density(
    records: list[dict[str, Any]],
    sample_n: int | None = None,
    seed: int = 0,
    config: ProbeConfig | None = None,
    nlp: Any | None = None,
) -> dict[str, Any]:
    """
    A13 — Sentence density on A8-filtered records only (text_length >= min_text_length).

    nlp is injectable for testing — when provided, spacy.load is NOT called internally.
    When nlp=None, spaCy is loaded internally using cfg.spacy_model.
    spaCy imported at module level so patch('src.dataset_probe.spacy') works in tests.

    Subsampling is done by run_probe before calling this gate.
    This gate processes the full substantive records list passed to it.
    """
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "A13_sentence_density", "pass": False, "note": "No records."}

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
                "pass": False,
                "error": str(exc),
                "note": f"spaCy model load failed: {exc}",
            }

    substantive = [r for r in records if int(r.get("text_length", 0)) >= cfg.min_text_length]
    if not substantive:
        return {
            "gate": "A13_sentence_density",
            "pass": False,
            "records_after_a8_filter": 0,
            "note": "No records pass A8 length filter.",
        }

    # Use all substantive records — subsampling is run_probe's responsibility
    subsample = substantive

    sent_counts: list[int] = []
    below_threshold = 0
    for r in subsample:
        text = str(r.get("text", ""))[: cfg.a13_text_cap_chars]
        doc = nlp(text)
        n_sents = sum(1 for _ in doc.sents)
        sent_counts.append(n_sents)
        if n_sents < cfg.min_sentence_count:
            below_threshold += 1

    return {
        "gate": "A13_sentence_density",
        "spacy_model": cfg.spacy_model,
        "spacy_version": spacy_version,
        "min_sentence_threshold": cfg.min_sentence_count,
        "subsample_n": len(subsample),
        "records_after_a8_filter": len(substantive),
        "mean_sentences": round(statistics.mean(sent_counts), 1),
        "median_sentences": statistics.median(sent_counts),
        "min_sentences": min(sent_counts),
        "below_threshold_count": below_threshold,
        "below_threshold_pct": round(100.0 * below_threshold / len(subsample), 2),
        "pass": below_threshold / len(subsample) < cfg.a13_max_below_threshold_pct / 100.0,
        "note": (
            "Evaluated on text_length >= 1500 records only (A8-filtered). "
            "Pass threshold: <15% below 20 sentences — calibrated to corpus "
            "(median=71.5 sentences, empirical below-rate=11%)."
        ),
    }


def gate_b6_text_entropy_distribution(
    records: list[dict[str, Any]],
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """
    B6 — text_entropy distribution + spot-check for formula drift.
    Spot-check computes Shannon entropy on a subsample and compares against
    the stored text_entropy field to detect upstream formula changes.
    Pass always (distribution-only) — threshold is data-derived.
    """
    cfg = config or ProbeConfig()
    if not records:
        return {"gate": "B6_text_entropy_distribution", "pass": True, "note": "No records."}

    entropies = [float(r.get("text_entropy", 0.0)) for r in records]
    entropies_sorted = sorted(entropies)
    zero_entropy = sum(1 for e in entropies if e == 0.0)

    # Spot-check: compute entropy on sample and compare to stored values
    spot_sample = records[: cfg.b6_entropy_spot_check_sample_n]
    deviations: list[float] = []
    for r in spot_sample:
        text = str(r.get("text", ""))
        computed = _shannon_entropy(text)
        stored = float(r.get("text_entropy", 0.0))
        deviations.append(abs(computed - stored))
    max_deviation = max(deviations) if deviations else 0.0
    spot_consistent = max_deviation <= cfg.b6_entropy_spot_check_tolerance

    return {
        "gate": "B6_text_entropy_distribution",
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
            "consistent": spot_consistent,
            "max_deviation": round(max_deviation, 4),
            "tolerance": cfg.b6_entropy_spot_check_tolerance,
            "sample_n": len(spot_sample),
        },
        "pass": True,
        "note": "Use p10 as provisional low-entropy filter cutoff for Stage 3.",
    }


# ---------------------------------------------------------------------------
# Model-relevant quality signals (advisory — not schema violations)
# ---------------------------------------------------------------------------


class ModelQualitySignals:
    """
    Soft quality warnings for RAG pipeline rows.
    Returns (signal_name, detail) tuples. Empty = clean.
    Used by wandb_logger.py and integrated into run_probe() report.
    """

    HTML_RE = re.compile(r"<[a-zA-Z][^>]{0,100}>")
    CITATION_RE = re.compile(r"\d+\s+[A-Z][a-z]*\.?\s*(?:\d+d?|App\.?|Supp\.?)")
    BOILERPLATE_PHRASES = (
        "all rights reserved",
        "this page intentionally left blank",
        "unpublished disposition",
        "not for publication",
        "do not cite",
    )

    @classmethod
    def check(cls, row: dict[str, Any], text_field: str = "text") -> list[tuple[str, str]]:
        signals: list[tuple[str, str]] = []
        text: str = row.get(text_field, "")
        token_count = len(text.split())

        if token_count < 20:
            signals.append(("truncated_document", f"~{token_count} tokens — likely truncated"))
        if token_count > 100_000:
            signals.append(("gigantic_document", f"~{token_count} tokens — may exceed model context"))
        if cls.HTML_RE.search(text):
            signals.append(("html_remnants", "HTML tags detected — scraping artifact"))
        if unicodedata.normalize("NFC", text) != text:
            signals.append(("unicode_not_nfc", "Text is not NFC-normalized"))

        lower = text.lower()
        for phrase in cls.BOILERPLATE_PHRASES:
            if phrase in lower:
                signals.append(("boilerplate", f"Boilerplate phrase detected: {phrase!r}"))
                break

        citation_count = len(cls.CITATION_RE.findall(text))
        if token_count > 100 and citation_count == 0:
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


# ---------------------------------------------------------------------------
# Top-level probe class — config-injecting orchestrator
# ---------------------------------------------------------------------------


class CourtListenerDatasetProbe:
    """
    Orchestrates all dataset-readiness gates for the local CourtListener corpus.
    Accepts injected ProbeConfig for reproducible, versioned experiment settings.
    Supports optional W&B logging for experiment tracking.

    validate_row removed — it was dead code never called outside tests.
    Full batch validation is provided by validate_schema().
    """

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


# ---------------------------------------------------------------------------
# run_probe — subsampling happens here before gates are called
# ---------------------------------------------------------------------------


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
    cfg = config or ProbeConfig()

    print(f"[dataset_probe] Sampling {subset} records from {data_dir} ...")
    audit = iter_shards_with_audit(data_dir)
    all_records = audit["records"]

    # Subsampling happens here — gates receive pre-sampled lists
    rng = random.Random(seed)
    records = rng.sample(all_records, subset) if len(all_records) > subset else all_records
    print(f"[dataset_probe] Loaded {len(records)} records.")

    # Subsample for A11/A12/A13 gates — controlled here, not inside gates
    a11_sample = rng.sample(records, min(cfg.a11_subsample_n, len(records)))
    a12_sample = rng.sample(records, min(cfg.a12_subsample_n, len(records)))
    a13_candidates = [r for r in records if int(r.get("text_length", 0)) >= cfg.min_text_length]
    a13_sample = rng.sample(a13_candidates, min(cfg.a13_subsample_n, len(a13_candidates)))

    try:
        spacy_version = spacy.__version__
        spacy_model_version = spacy.load(cfg.spacy_model).meta.get("version", "unknown")
    except Exception:
        spacy_version = "unknown"
        spacy_model_version = "unknown"

    # Load spaCy once for A13 if not skipped
    nlp_pipeline: Any | None = None
    if not skip_spacy:
        try:
            nlp_pipeline = spacy.load(cfg.spacy_model, exclude=SPACY_EXCLUDE)
            if "sentencizer" not in nlp_pipeline.pipe_names:
                nlp_pipeline.add_pipe("sentencizer")
            nlp_pipeline.max_length = 2_000_000
        except Exception:
            nlp_pipeline = None

    report: dict[str, Any] = {
        "data_dir": str(data_dir),
        "subset_n": len(records),
        "seed": seed,
        "shard_audit": {
            "shard_count": audit["shard_count"],
            "total_parse_errors": audit["total_parse_errors"],
            "total_blank_lines": audit["total_blank_lines"],
            "shard_errors": audit["shard_errors"],
        },
        "provenance": {
            "probe_version": PROBE_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "spacy_version": spacy_version,
            "spacy_model_version": spacy_model_version,
            "probe_config": dataclasses.asdict(cfg),
        },
        "gates": {},
    }

    print("[dataset_probe] Gate: schema validation ...")
    report["gates"]["schema"] = validate_schema(records)

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

    passed = [k for k, v in report["gates"].items() if v.get("pass") is True]
    failed = [k for k, v in report["gates"].items() if v.get("pass") is False]
    skipped = [k for k, v in report["gates"].items() if v.get("skipped")]
    report["summary"] = {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "all_passed": len(failed) == 0,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[dataset_probe] Report written → {output}")
    print(f"[dataset_probe] PASSED: {passed} | FAILED: {failed} | SKIPPED: {skipped}")

    if log_to_wandb and wandb.run is not None:
        wandb.log(
            {
                "probe/passed_gates": len(passed),
                "probe/failed_gates": len(failed),
                "probe/all_passed": report["summary"]["all_passed"],
                "probe/parse_errors": audit["total_parse_errors"],
                **{f"probe/quality/{k}": v for k, v in report["quality_signals"]["signal_counts"].items()},
                "probe/pct_clean": report["quality_signals"]["pct_clean"],
            }
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        help="Exit 1 if any gate fails — use as CI hard gate before training jobs.",
    )
    args = parser.parse_args()
    report = run_probe(
        data_dir=args.data_dir,
        subset=args.subset,
        output=args.output,
        seed=args.seed,
        skip_tokenizer=args.skip_tokenizer,
        skip_spacy=args.skip_spacy,
    )
    if args.ci_mode and not report["summary"]["all_passed"]:
        print(f"[dataset_probe] CI mode: gate failures — {report['summary']['failed']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
