# src/dataset_probe.py
"""
Dataset readiness probe for local CourtListener federal appellate JSONL shards.

Implements Category A dataset-readiness gates required before Stage 3:
  A7  — text_source breakdown (missing 17.2% audit)
  A8  — text_length distribution + RAG viability threshold derivation
  A9  — citation_count distribution (zero-citation filter for fast-iteration subset)
  A11 — tokenizer-aware chunk count verification (BAAI/bge-m3, 1024-subword budget)
  A12 — citation anchor survival check in normalized text
  A13 — sentence density check via repo-certified spaCy pipeline (not NLTK)
  B6  — text_entropy empirical distribution (threshold must be data-derived)

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

All output written to --output as JSON. No side effects on corpus shards.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterator

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

# RAG viability floor — Stage 3 applies this as a hard filter before chunking.
# Pass condition: <25% below threshold (corpus has known ~20% short-doc tail).
PROVISIONAL_MIN_TEXT_LENGTH = 1500

# Tokenizer chunk budget from README Stage 3 controlled design choice.
CHUNK_SIZE_SUBWORDS = 1024
CHUNK_OVERLAP_SUBWORDS = 128

# Encoder model for A11 tokenizer-aware chunk count check (README-certified).
ENCODER_MODEL = "BAAI/bge-m3"

# spaCy model for A13 sentence density check (repo-certified pipeline).
SPACY_MODEL = "en_core_web_sm"
SPACY_EXCLUDE = ["ner", "parser", "lemmatizer"]

# Minimum sentence count for Tier B NLI atomic-claim density.
# A13 only evaluates records that pass the A8 length filter (text_length >= 1500)
# so short summary dispositions do not pollute this gate.
MIN_SENTENCE_COUNT = 20


# ---------------------------------------------------------------------------
# Shard loader
# ---------------------------------------------------------------------------


def iter_shards(data_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield records from all .jsonl shard files in data_dir."""
    shard_files = sorted(data_dir.glob("*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No .jsonl shards found in {data_dir}")
    for shard_path in shard_files:
        with open(shard_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
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
# Gate implementations
# ---------------------------------------------------------------------------


def gate_a7_text_source_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    A7 — Full text_source breakdown including the unaccounted ~17.2%.
    plain_text + html_with_citations = 82.8%; remainder is unknown format risk.
    """
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
        "pass": known_pct >= 80.0,
        "note": (
            "Inspect records from any source outside plain_text/html_with_citations "
            "to verify row_normalizer.py strips them cleanly before Stage 3."
        ),
    }


def gate_a8_text_length_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    A8 — text_length distribution + RAG viability threshold.
    Pass condition: <25% below PROVISIONAL_MIN_TEXT_LENGTH.
    ~20% short-doc tail is expected (summary dispositions filtered in Stage 3).
    """
    lengths = [int(r.get("text_length", 0)) for r in records]
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    def pct(p: float) -> int:
        idx = max(0, min(n - 1, int(math.ceil(p / 100.0 * n)) - 1))
        return lengths_sorted[idx]

    below_provisional = sum(1 for length in lengths if length < PROVISIONAL_MIN_TEXT_LENGTH)
    return {
        "gate": "A8_text_length_distribution",
        "count": n,
        "mean": round(statistics.mean(lengths), 1),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "p5": pct(5),
        "p10": pct(10),
        "p25": pct(25),
        "p75": pct(75),
        "p90": pct(90),
        "p95": pct(95),
        "provisional_min_chars": PROVISIONAL_MIN_TEXT_LENGTH,
        "below_provisional_count": below_provisional,
        "below_provisional_pct": round(100.0 * below_provisional / n, 2),
        "pass": below_provisional / n < 0.25,
        "note": (
            "~20% short-doc tail is expected (summary dispositions). "
            "Stage 3 applies text_length >= 1500 filter before chunking."
        ),
    }


def gate_a9_citation_count_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    A9 — citation_count distribution.
    Zero-citation cases are procedural anomalies with no Tier C utility.
    """
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
        "pass": zero / n < 0.20,
        "note": ("For ~150K fast-iteration subset, filter citation_count > 5 to maximise Tier C utility."),
    }


def gate_a11_tokenizer_chunk_count(
    records: list[dict[str, Any]],
    sample_n: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """
    A11 — Tokenizer-aware chunk count using BAAI/bge-m3 (README-certified encoder).
    Verifies corpus supports multi-chunk splitting at 1024-subword budget.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore[import]

        tok = AutoTokenizer.from_pretrained(ENCODER_MODEL)
    except Exception as exc:
        return {
            "gate": "A11_tokenizer_chunk_count",
            "pass": False,
            "error": str(exc),
            "note": "AutoTokenizer load failed — check HF_TOKEN and network.",
        }

    rng = random.Random(seed)
    subsample = rng.sample(records, min(sample_n, len(records)))

    chunk_counts: list[int] = []
    token_lengths: list[int] = []
    multi_chunk = 0

    for r in subsample:
        text = str(r.get("text", ""))
        ids = tok(text, add_special_tokens=False)["input_ids"]
        total_tokens = len(ids)
        token_lengths.append(total_tokens)
        stride = CHUNK_SIZE_SUBWORDS - CHUNK_OVERLAP_SUBWORDS
        n_chunks = max(1, math.ceil(max(0, total_tokens - CHUNK_OVERLAP_SUBWORDS) / stride))
        chunk_counts.append(n_chunks)
        if n_chunks > 1:
            multi_chunk += 1

    return {
        "gate": "A11_tokenizer_chunk_count",
        "encoder_model": ENCODER_MODEL,
        "chunk_size_subwords": CHUNK_SIZE_SUBWORDS,
        "overlap_subwords": CHUNK_OVERLAP_SUBWORDS,
        "subsample_n": len(subsample),
        "mean_token_length": round(statistics.mean(token_lengths), 1),
        "median_token_length": statistics.median(token_lengths),
        "mean_chunks_per_doc": round(statistics.mean(chunk_counts), 2),
        "median_chunks_per_doc": statistics.median(chunk_counts),
        "multi_chunk_pct": round(100.0 * multi_chunk / len(subsample), 2),
        "pass": statistics.median(chunk_counts) >= 2,
        "note": "median_chunks_per_doc >= 2 confirms corpus supports multi-chunk splitting.",
    }


def gate_a12_citation_anchor_survival(
    records: list[dict[str, Any]],
    sample_n: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """
    A12 — Citation anchor survival in normalized text field.
    Checks reporter citations, case-name citations, SCOTUS, and Federal reporter.
    """
    CITATION_RE = re.compile(
        r"(\d+\s+[A-Z][a-z]*\.?\s*(?:\d+d?|App\.?|Supp\.?)"
        r"|[A-Z][a-z]+\s+v\.\s+[A-Z]"
        r"|U\.S\.\s+\d+"
        r"|\d+\s+F\.\d+[a-z]?\s+\d+)",
        re.MULTILINE,
    )
    rng = random.Random(seed)
    subsample = rng.sample(records, min(sample_n, len(records)))

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
        "pass": pct_with_anchor >= 60.0,
        "note": (">=60% of records must contain extractable citation anchors for Tier C SQLite lookup to be viable."),
    }


def gate_a13_sentence_density(
    records: list[dict[str, Any]],
    sample_n: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """
    A13 — Sentence density check via repo-certified spaCy pipeline (not NLTK).
    Only evaluates records that pass the A8 length filter (text_length >= 1500)
    so short summary dispositions do not pollute the NLI density gate.
    sentencizer added because parser is excluded (avoids spaCy E030).
    nlp.max_length set high to handle full federal appellate opinions safely.
    """
    try:
        import spacy  # type: ignore[import]

        nlp = spacy.load(SPACY_MODEL, exclude=SPACY_EXCLUDE)
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        nlp.max_length = 2_000_000
    except Exception as exc:
        return {
            "gate": "A13_sentence_density",
            "pass": False,
            "error": str(exc),
            "note": "spaCy model load failed — run: python -m spacy download en_core_web_sm",
        }

    # Filter to records that pass A8 length floor before sentence density check
    substantive = [r for r in records if int(r.get("text_length", 0)) >= PROVISIONAL_MIN_TEXT_LENGTH]
    if not substantive:
        return {
            "gate": "A13_sentence_density",
            "pass": False,
            "error": "No records pass A8 length filter — run A8 gate first.",
            "note": "Ensure text_length >= 1500 records exist in sample.",
        }

    rng = random.Random(seed)
    subsample = rng.sample(substantive, min(sample_n, len(substantive)))

    sent_counts: list[int] = []
    below_threshold = 0
    for r in subsample:
        text = str(r.get("text", ""))[:50_000]  # cap per-doc cost for probe
        doc = nlp(text)
        n_sents = sum(1 for _ in doc.sents)
        sent_counts.append(n_sents)
        if n_sents < MIN_SENTENCE_COUNT:
            below_threshold += 1

    return {
        "gate": "A13_sentence_density",
        "spacy_model": SPACY_MODEL,
        "min_sentence_threshold": MIN_SENTENCE_COUNT,
        "subsample_n": len(subsample),
        "records_after_a8_filter": len(substantive),
        "mean_sentences": round(statistics.mean(sent_counts), 1),
        "median_sentences": statistics.median(sent_counts),
        "min_sentences": min(sent_counts),
        "below_threshold_count": below_threshold,
        "below_threshold_pct": round(100.0 * below_threshold / len(subsample), 2),
        "pass": below_threshold / len(subsample) < 0.10,
        "note": (
            "Evaluated on text_length >= 1500 records only (A8-filtered). "
            ">=90% must have >20 sentences for Tier B NLI atomic-claim density."
        ),
    }


def gate_b6_text_entropy_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    B6 — text_entropy empirical distribution.
    Threshold must be derived from data — not hardcoded.
    """
    entropies = [float(r.get("text_entropy", 0.0)) for r in records]
    entropies_sorted = sorted(entropies)
    n = len(entropies_sorted)

    def pct(p: float) -> float:
        idx = max(0, min(n - 1, int(math.ceil(p / 100.0 * n)) - 1))
        return round(entropies_sorted[idx], 4)

    zero_entropy = sum(1 for e in entropies if e == 0.0)
    return {
        "gate": "B6_text_entropy_distribution",
        "count": n,
        "mean": round(statistics.mean(entropies), 4),
        "median": round(float(statistics.median(entropies)), 4),
        "min": round(min(entropies), 4),
        "max": round(max(entropies), 4),
        "p5": pct(5),
        "p10": pct(10),
        "p25": pct(25),
        "p75": pct(75),
        "p90": pct(90),
        "p95": pct(95),
        "zero_entropy_count": zero_entropy,
        "zero_entropy_pct": round(100.0 * zero_entropy / n, 2),
        "pass": True,  # distribution-only gate — no hardcoded threshold
        "note": ("Use p10 as provisional low-entropy filter cutoff for Stage 3. Adjust after reviewing distribution."),
    }


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_schema(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Check all required fields are present across sampled records."""
    missing_by_field: dict[str, int] = {f: 0 for f in REQUIRED_FIELDS}
    for r in records:
        for f in REQUIRED_FIELDS:
            if f not in r:
                missing_by_field[f] += 1
    any_missing = any(v > 0 for v in missing_by_field.values())
    return {
        "gate": "schema_validation",
        "required_fields": sorted(REQUIRED_FIELDS),
        "missing_counts": {k: v for k, v in missing_by_field.items() if v > 0},
        "pass": not any_missing,
    }


# ---------------------------------------------------------------------------
# Model-relevant quality signals (advisory — not schema violations)
# Used by src/wandb_logger.py to log quality signal distributions to W&B.
# ---------------------------------------------------------------------------


class ModelQualitySignals:
    """
    Model-relevant quality signals for RAG pipeline rows.
    Soft warnings returned as (signal_name, detail) tuples. Empty = clean.
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
        """Return quality signals for a row. Empty = clean."""
        signals: list[tuple[str, str]] = []
        text: str = row.get(text_field, "")
        token_count = len(text.split())

        if token_count < 20:
            signals.append(("truncated_document", f"~{token_count} tokens — likely truncated"))
        if token_count > 100_000:
            signals.append(("gigantic_document", f"~{token_count} tokens — may exceed model context"))
        if cls.HTML_RE.search(text):
            signals.append(("html_remnants", "HTML tags detected — scraping artifact"))

        import unicodedata

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


# ---------------------------------------------------------------------------
# Top-level probe class and run_probe function
# ---------------------------------------------------------------------------


class CourtListenerDatasetProbe:
    """
    Orchestrates all dataset-readiness gates for the local CourtListener corpus.
    Replaces the dropped pile-of-law/pile-of-law HuggingFace probe.
    """

    def validate_row(self, row: dict[str, Any]) -> list[str]:
        """Return validation errors for a single row. Empty list = valid."""
        errors: list[str] = []
        text = row.get("text", "")
        if not isinstance(text, str) or len(text) < 50:
            errors.append(f"text too short or missing: {len(str(text))} chars")
        return errors

    def run(
        self,
        data_dir: Path,
        subset: int,
        output: Path,
        seed: int = 0,
        skip_tokenizer: bool = False,
        skip_spacy: bool = False,
    ) -> dict[str, Any]:
        return run_probe(
            data_dir=data_dir,
            subset=subset,
            output=output,
            seed=seed,
            skip_tokenizer=skip_tokenizer,
            skip_spacy=skip_spacy,
        )


def run_probe(
    data_dir: Path,
    subset: int,
    output: Path,
    seed: int = 0,
    skip_tokenizer: bool = False,
    skip_spacy: bool = False,
) -> dict[str, Any]:
    print(f"[dataset_probe] Sampling {subset} records from {data_dir} ...")
    records = sample_records(data_dir, subset, seed=seed)
    print(f"[dataset_probe] Loaded {len(records)} records.")

    report: dict[str, Any] = {
        "data_dir": str(data_dir),
        "subset_n": len(records),
        "seed": seed,
        "gates": {},
    }

    print("[dataset_probe] Gate: schema validation ...")
    report["gates"]["schema"] = validate_schema(records)

    print("[dataset_probe] Gate A7: text_source breakdown ...")
    report["gates"]["A7"] = gate_a7_text_source_breakdown(records)

    print("[dataset_probe] Gate A8: text_length distribution ...")
    report["gates"]["A8"] = gate_a8_text_length_distribution(records)

    print("[dataset_probe] Gate A9: citation_count distribution ...")
    report["gates"]["A9"] = gate_a9_citation_count_distribution(records)

    print("[dataset_probe] Gate A12: citation anchor survival ...")
    report["gates"]["A12"] = gate_a12_citation_anchor_survival(records)

    print("[dataset_probe] Gate B6: text_entropy distribution ...")
    report["gates"]["B6"] = gate_b6_text_entropy_distribution(records)

    if not skip_tokenizer:
        print("[dataset_probe] Gate A11: tokenizer-aware chunk count (BAAI/bge-m3) ...")
        report["gates"]["A11"] = gate_a11_tokenizer_chunk_count(records)
    else:
        report["gates"]["A11"] = {"gate": "A11_tokenizer_chunk_count", "skipped": True}

    if not skip_spacy:
        print("[dataset_probe] Gate A13: sentence density (spaCy) ...")
        report["gates"]["A13"] = gate_a13_sentence_density(records)
    else:
        report["gates"]["A13"] = {"gate": "A13_sentence_density", "skipped": True}

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
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="CourtListener dataset readiness probe (Category A + B6 gates).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/cl_federal_appellate_bulk"),
        help="Directory containing .jsonl shard files.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=10_000,
        help="Number of records to reservoir-sample (default: 10000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/dataset_probe_report.json"),
        help="Output path for JSON report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reservoir sampling (default: 0).",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip Gate A11 (tokenizer chunk count) — requires HF model download.",
    )
    parser.add_argument(
        "--skip-spacy",
        action="store_true",
        help="Skip Gate A13 (spaCy sentence density).",
    )
    args = parser.parse_args()
    run_probe(
        data_dir=args.data_dir,
        subset=args.subset,
        output=args.output,
        seed=args.seed,
        skip_tokenizer=args.skip_tokenizer,
        skip_spacy=args.skip_spacy,
    )


if __name__ == "__main__":
    main()
