# src/extract.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/extract.py
"""Streaming CourtListener opinions extractor with sharded output and resume.

This module is the workhorse of the ingest pipeline. It consumes the
raw CourtListener opinions CSV (optionally bz2-compressed), joins each
row against pre-built cluster/docket/court metadata maps, normalises
the opinion text, and writes the results as numbered JSONL or Parquet
shards under :attr:`PipelineConfig.shard_dir`.

Architecture
------------
The module is organised as a thin orchestration function
(:func:`extract_opinions_to_shards`) over a set of **pure** helpers:

* :class:`ShardWriter` — buffered shard accumulator with JSONL/Parquet
  backends and ``start_index`` support for resume.
* Text-cleaning functions (:func:`_normalize_text` and friends) —
  stateless, regex-driven, easy to unit-test.
* Metric helpers (:func:`_count_citations`, :func:`_citation_density`,
  :func:`_text_entropy`) — stateless scalar functions used to build the
  per-record quality features.
* :func:`build_record` — assembles a :class:`OpinionRecord` from raw
  inputs and the metadata joins.

The orchestrator itself handles only: checkpoint load/save, CSV
streaming, per-row filter decisions, statistics aggregation, and
logging.

Resume semantics
----------------
A ``checkpoint.json`` is written every
:attr:`PipelineConfig.checkpoint_interval` rows and at the end of every
run. On re-entry the orchestrator skips the first ``scanned`` rows
(already processed), continues shard numbering at ``num_shards``, and
adds to the accumulated counters. This makes the extractor safe to kill
and restart on cluster-timeout boundaries.

Skip reasons
------------
Rows can be dropped for four reasons, each counted separately:

* ``bad_cluster_id`` — ``cluster_id`` is not an int (counted under
  ``skipped_parse``).
* ``bad_opinion_id`` — ``id`` is not an int, usually a sign of CSV
  field misalignment (counted under ``skipped_parse``).
* ``skipped_empty`` — no text field passed the minimum length floor.
* ``skipped_stub`` — text shrank below the floor *after* normalisation
  (e.g. only HTML boilerplate).

Skipped rows with parse failures are written to the optional
quarantine JSONL for later inspection.
"""

from __future__ import annotations

import bz2
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Set, Tuple, Union

from tqdm import tqdm

from src.config import PipelineConfig
from src.schemas import OpinionRecord

# ============================================================
# ShardWriter
# ============================================================


class ShardWriter:
    """Buffered writer that emits numbered JSONL or Parquet shards.

    Records are accumulated in an in-memory buffer and flushed whenever
    the buffer reaches ``shard_size``. The caller must invoke
    :meth:`flush` once at the end to write any remaining partial shard.

    Shards are numbered with a zero-padded 4-digit index and named
    ``shard_NNNN.<ext>``. Passing ``start_index`` lets resume runs
    continue the numbering without clobbering shards from a prior
    incarnation.

    Attributes:
        shard_count: Number of shards already flushed (read-only).
    """

    def __init__(
        self,
        shard_dir: Union[str, Path],
        shard_size: int,
        compress: bool = False,
        output_format: str = "jsonl",
        start_index: int = 0,
    ) -> None:
        """Initialise the writer.

        Args:
            shard_dir: Directory that will contain the shard files.
                Must already exist; the writer does not create it.
            shard_size: Number of records per flushed shard.
            compress: When ``True`` and ``output_format == "jsonl"``,
                shards are written as ``.jsonl.gz``. Ignored for
                Parquet.
            output_format: ``"jsonl"`` (default) or ``"parquet"``.
            start_index: Initial shard number, used for resume.
        """
        self._shard_dir = Path(shard_dir)
        self._shard_size = shard_size
        self._compress = compress
        self._output_format = output_format
        self._index: int = start_index
        self._buffer: List[Any] = []

    @property
    def shard_count(self) -> int:
        """Number of shards successfully flushed so far."""
        return self._index

    def _shard_path(self, index: int) -> Path:
        """Return the filesystem path for shard ``index``."""
        if self._output_format == "parquet":
            return self._shard_dir / f"shard_{index:04d}.parquet"
        ext = "jsonl.gz" if self._compress else "jsonl"
        return self._shard_dir / f"shard_{index:04d}.{ext}"

    def add(self, record: Any) -> None:
        """Append a record and auto-flush when the buffer fills.

        Args:
            record: Either an object with a ``to_dict()`` method
                (e.g. :class:`OpinionRecord`) or a plain dict.
        """
        self._buffer.append(record)
        if len(self._buffer) >= self._shard_size:
            self.flush()

    def flush(self) -> None:
        """Write the current buffer as a new shard, then reset.

        A no-op when the buffer is empty, so callers can invoke it
        unconditionally at cleanup time.
        """
        if not self._buffer:
            return
        shard_path = self._shard_path(self._index)
        if self._output_format == "parquet":
            self._flush_parquet(shard_path)
        else:
            self._flush_jsonl(shard_path)
        self._index += 1
        self._buffer = []

    def _flush_jsonl(self, shard_path: Path) -> None:
        """Write the buffer as newline-delimited JSON, gzipped if enabled."""
        import gzip as _gzip

        opener = _gzip.open if self._compress else open
        with opener(shard_path, "wt") as file_handle:
            for case in self._buffer:
                d: Dict[str, Any] = case.to_dict() if hasattr(case, "to_dict") else case
                file_handle.write(json.dumps(d) + "\n")

    def _flush_parquet(self, shard_path: Path) -> None:
        """Write the buffer as a single Parquet file via pyarrow."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows: List[Dict[str, Any]] = []
        for case in self._buffer:
            rows.append(case.to_dict() if hasattr(case, "to_dict") else case)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, shard_path)


# ============================================================
# Legal text cleaning — pure functions
# ============================================================

#: Matches any HTML tag. Used for tag stripping before citation-aware
#: reflow; does not attempt full HTML parsing.
_HTML_TAG_RE = re.compile(r"<[^>]+>")

#: Collapses runs of spaces/tabs into a single space.
_MULTI_SPACE_RE = re.compile(r"[ \t]+")

#: Matches 3+ consecutive newlines for paragraph-break normalisation.
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

#: Strips NUL bytes that occasionally survive bz2 decoding.
_NULL_BYTE_RE = re.compile(r"\x00")

#: Matches Latin-1 C1 control characters — common mojibake marker.
_MOJIBAKE_RE = re.compile(r"[\x80-\x9f]")

#: Matches Westlaw page-number markers of the form ``*NN`` on their
#: own line (with optional surrounding newlines).
_PAGE_STAR_RE = re.compile(r"\n?\*\d+\n?")

#: Matches the Westlaw reporter header block that prefixes unreported
#: opinions, which is legal boilerplate with no semantic content.
_WESTLAW_HEADER_RE = re.compile(r"^(Not Reported in .*?\n|Only the Westlaw citation.*?\n)", re.MULTILINE)


def _strip_html_preserve_citations(text: str) -> str:
    """Strip HTML tags while keeping inline citation text intact.

    Replaces tags with a single space (to avoid welding adjacent words
    together) then collapses runs of whitespace.
    """
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _remove_boilerplate(text: str) -> str:
    """Delete Westlaw headers and page-break star markers."""
    text = _WESTLAW_HEADER_RE.sub("", text)
    text = _PAGE_STAR_RE.sub(" ", text)
    return text.strip()


def _clean_encoding(text: str) -> str:
    """Repair encoding artifacts common in OCR and scraped opinions.

    Removes NUL bytes and C1 control chars; maps smart quotes and
    en/em dashes to their ASCII equivalents so downstream tokenisers
    do not see near-duplicate vocabulary.
    """
    text = _NULL_BYTE_RE.sub("", text)
    text = _MOJIBAKE_RE.sub("", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2014", "--").replace("\u2013", "-")
    return text


def _strip_html(text: str) -> str:
    """Plain HTML tag stripper — no citation preservation logic.

    Kept for callers that just want the text content of a single HTML
    field without the full :func:`_normalize_text` pipeline.
    """
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _normalize_text(raw_text: str, text_source: str) -> Tuple[str, List[str]]:
    """Apply the full cleaning pipeline and record which stages fired.

    Pipeline order:

    1. Encoding repair.
    2. HTML stripping (only when ``text_source != "plain_text"``).
    3. Boilerplate removal.
    4. Newline collapse.
    5. Final strip.

    Args:
        raw_text: Unmodified text from the chosen CSV field.
        text_source: The field name the text came from; used to decide
            whether HTML stripping is needed.

    Returns:
        ``(cleaned_text, flags)`` where ``flags`` is a list of stage
        names that actually mutated the text (for per-corpus cleaning
        statistics).
    """
    flags: List[str] = []
    cleaned = _clean_encoding(raw_text)
    if cleaned != raw_text:
        flags.append("encoding_cleaned")
    if text_source != "plain_text":
        cleaned = _strip_html_preserve_citations(cleaned)
        flags.append("html_stripped")
    before_boilerplate = cleaned
    cleaned = _remove_boilerplate(cleaned)
    if cleaned != before_boilerplate:
        flags.append("boilerplate_removed")
    if _MULTI_NEWLINE_RE.search(cleaned):
        cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
        flags.append("newlines_collapsed")
    cleaned = cleaned.strip()
    if cleaned != raw_text.strip():
        flags.append("whitespace_trimmed")
    return cleaned, flags


# ============================================================
# Pure functions — extracted for testability
# ============================================================


def select_best_text(
    row: Dict[str, Any],
    text_source_fields: Tuple[str, ...],
    min_length: int = 50,
) -> Tuple[str, str]:
    """Pick the highest-priority text field that meets the length floor.

    Iterates ``text_source_fields`` in order and returns the first
    whose stripped value is at least ``min_length`` characters long.

    Args:
        row: A CSV row as a dict.
        text_source_fields: Ordered priority list of column names.
        min_length: Minimum character count to accept.

    Returns:
        ``(text, source_field_name)``, or ``("", "")`` when no field
        qualifies.
    """
    for source_field in text_source_fields:
        candidate = (row.get(source_field) or "").strip()
        if len(candidate) >= min_length:
            return candidate, source_field
    return "", ""


def _attr(obj: Any, key: str, default: Any = "") -> Any:
    """Attribute/key accessor that works on both dataclasses and dicts.

    Used in :func:`build_record` so cluster/docket metadata can be
    supplied as either typed dataclasses or raw dicts without branching.
    """
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _extract_year(date_filed: Optional[str]) -> str:
    """Return the 4-digit year prefix of a date string, or ``"unknown"``."""
    if not date_filed or date_filed == "nan":
        return "unknown"
    match = re.match(r"(\d{4})", str(date_filed))
    return match.group(1) if match else "unknown"


#: Matches U.S. legal reporter citations (U.S., S.Ct., L.Ed., F.Nd,
#: F.Supp.N). Intentionally loose — false positives are cheap, misses
#: hurt corpus metrics.
_CITATION_RE = re.compile(r"\d+\s+(?:U\.S\.|S\.Ct\.|L\.Ed\.|F\.\d[a-z]*|F\.Supp\.\d*)")


def _count_citations(text: str) -> int:
    """Return the number of legal reporter citations in ``text``."""
    return len(_CITATION_RE.findall(text))


def _count_paragraphs(text: str) -> int:
    """Return the paragraph count, bottoming out at 1 for non-empty text."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return max(len(paragraphs), 1)


def _citation_density(citation_count: int, text_length: int) -> float:
    """Citations per 1K tokens, using a 4-chars-per-token estimate.

    The 4:1 estimate is accurate enough for corpus-level density
    statistics; per-document token counts use the actual tokenizer.
    """
    token_count = text_length // 4
    if token_count == 0:
        return 0.0
    return (citation_count / token_count) * 1000


def _text_entropy(text: str) -> float:
    """Shannon entropy of the character distribution in ``text``.

    Used as a cheap gibberish detector: clean English prose scores
    around 4.0–4.5 bits/char, while OCR garble drops below 3.0 or
    spikes above 5.0 depending on the failure mode.
    """
    if not text:
        return 0.0
    counts: Counter[str] = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy


def parse_opinion_id(raw: str) -> Optional[int]:
    """Parse an opinion ID from a CSV cell, returning ``None`` on failure.

    A ``None`` return is the main signal of CSV field misalignment
    (e.g. an embedded newline shifted subsequent columns).
    """
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def build_record(
    opinion_id: int,
    cluster_id: int,
    raw_text: str,
    normalized_text: str,
    text_source: str,
    cleaning_flags: List[str],
    opinion_type: str,
    extracted_by_ocr: str,
    cluster_meta: Dict[int, Any],
    docket_meta: Dict[int, Any],
    court_name_map: Dict[str, str],
) -> OpinionRecord:
    """Assemble a fully-joined :class:`OpinionRecord` for a single opinion.

    Performs the cluster → docket → court join, computes the quality
    metric features (citation count/density, entropy, token estimate,
    paragraph count), and hashes the normalised text for dedup.

    Args:
        opinion_id: Parsed opinion ID from the CSV.
        cluster_id: Parsed cluster ID, already verified as a federal
            appellate cluster.
        raw_text: Unmodified text from the chosen source field.
        normalized_text: Output of :func:`_normalize_text`.
        text_source: Name of the source field.
        cleaning_flags: Flags emitted by :func:`_normalize_text`.
        opinion_type: CourtListener opinion type (``"010combined"``, etc.).
        extracted_by_ocr: Raw string from the OCR flag column.
        cluster_meta: Map of cluster_id → cluster metadata object/dict.
        docket_meta: Map of docket_id → docket metadata object/dict.
        court_name_map: Map of court_id → human-readable court name.

    Returns:
        A fully-populated :class:`OpinionRecord` ready for shard output.
    """
    empty: Dict[str, Any] = {}
    cluster_metadata = cluster_meta.get(cluster_id, empty)
    docket_id: Optional[int] = _attr(cluster_metadata, "docket_id", None)
    docket_metadata = docket_meta.get(docket_id, empty) if docket_id else empty
    court_id: str = _attr(docket_metadata, "court_id", "")
    precedential_status: str = _attr(cluster_metadata, "precedential_status", "")

    cit_count = _count_citations(normalized_text)
    text_len = len(normalized_text)

    return OpinionRecord(
        id=opinion_id,
        cluster_id=cluster_id,
        docket_id=docket_id,
        court_id=court_id,
        court_name=court_name_map.get(court_id, ""),
        case_name=_attr(cluster_metadata, "case_name", "") or _attr(docket_metadata, "case_name", ""),
        date_filed=_attr(cluster_metadata, "date_filed", "") or _attr(docket_metadata, "date_filed", ""),
        precedential_status=precedential_status,
        opinion_type=opinion_type,
        extracted_by_ocr=extracted_by_ocr,
        raw_text=raw_text,
        text=normalized_text,
        text_length=text_len,
        text_source=text_source,
        cleaning_flags=cleaning_flags,
        source="courtlistener_bulk",
        token_count=text_len // 4,
        paragraph_count=_count_paragraphs(normalized_text),
        citation_count=cit_count,
        text_hash=hashlib.sha256(normalized_text.encode()).hexdigest(),
        citation_density=_citation_density(cit_count, text_len),
        is_precedential=precedential_status.lower() in ("published", "precedential"),
        text_entropy=_text_entropy(normalized_text),
    )


# ============================================================
# CSV streaming
# ============================================================


def _open_csv(filepath: Union[str, Path]) -> IO[str]:
    """Open a CourtListener opinions CSV, transparently handling bz2.

    Uses ``errors="replace"`` because the raw dump contains occasional
    invalid UTF-8 sequences that the downstream :func:`_clean_encoding`
    step will strip anyway.

    Args:
        filepath: Path to a ``.csv`` or ``.csv.bz2`` file.

    Returns:
        A text-mode file handle ready for :class:`csv.DictReader`.
    """
    filepath = str(filepath)
    if filepath.endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(filepath, "rb"), encoding="utf-8", errors="replace")
    return open(filepath, "r", encoding="utf-8", errors="replace", newline="")


# ============================================================
# Quarantine — dead letter queue for skipped rows
# ============================================================


def _write_quarantine(quarantine_path: Optional[Path], reason: str, row: Dict[str, Any]) -> None:
    """Append a rejected row to the quarantine JSONL, if configured.

    A no-op when ``quarantine_path`` is ``None``, making the call safe
    inside the hot row loop.
    """
    if quarantine_path is None:
        return
    with open(quarantine_path, "a") as fh:
        fh.write(json.dumps({"reason": reason, "row": dict(row)}) + "\n")


# ============================================================
# Checkpoint — resume long-running extraction after timeout
# ============================================================


def _load_checkpoint(shard_dir: Path) -> Optional[Dict[str, Any]]:
    """Load ``checkpoint.json`` from ``shard_dir``, or ``None`` if absent."""
    cp_path = shard_dir / "checkpoint.json"
    if cp_path.exists():
        return json.loads(cp_path.read_text())
    return None


def _save_checkpoint(shard_dir: Path, data: Dict[str, Any]) -> None:
    """Persist the extraction state to ``checkpoint.json`` for resume."""
    cp_path = shard_dir / "checkpoint.json"
    cp_path.write_text(json.dumps(data))


# ============================================================
# Main extraction — thin orchestration over pure functions
# ============================================================


def extract_opinions_to_shards(
    opinions_path: Union[str, Path],
    cluster_meta: Dict[int, Any],
    docket_meta: Dict[int, Any],
    court_name_map: Dict[str, str],
    config: Optional[PipelineConfig] = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """Stream the opinions CSV into shards, with full stats and resume.

    Workflow per row:

    1. Parse ``cluster_id``; drop on failure.
    2. Skip rows whose cluster is not in the federal appellate set.
    3. Select the best text field via :func:`select_best_text`; drop
       if none qualifies.
    4. Normalise via :func:`_normalize_text`; drop stubs that shrank
       below the floor.
    5. Parse ``opinion_id``; drop on failure.
    6. Build the full :class:`OpinionRecord` and enqueue it.
    7. Update running statistics.
    8. Every ``checkpoint_interval`` rows, flush the writer and save a
       checkpoint.

    Uses ``escapechar='\\\\'`` in the CSV reader to match the
    PostgreSQL ``COPY`` format used by the CourtListener dump.

    Args:
        opinions_path: Path to the opinions CSV (plain or ``.bz2``).
        cluster_meta: Map of cluster_id → cluster metadata, pre-filtered
            to federal appellate clusters only.
        docket_meta: Map of docket_id → docket metadata.
        court_name_map: Map of court_id → human-readable court name.
        config: Pipeline configuration. A default :class:`PipelineConfig`
            is constructed when ``None``.
        logger: Optional logger for progress and summary output.

    Returns:
        A statistics dict containing scan/extract/skip counters, text
        source breakdown, cleaning flag counts, court/year/type
        distributions, text-length and token-count percentiles,
        duplicate counts, OCR counts, and a schema sample of the first
        emitted record.
    """
    if config is None:
        config = PipelineConfig()

    # Remove 128KB CSV field size limit — legal opinions routinely exceed this
    csv.field_size_limit(sys.maxsize)

    config.shard_dir.mkdir(parents=True, exist_ok=True)
    federal_cluster_ids: Set[int] = set(cluster_meta.keys())

    # Load checkpoint for resume
    checkpoint = _load_checkpoint(config.shard_dir)
    resume_from: int = 0
    start_shard_index: int = 0
    if checkpoint is not None:
        resume_from = checkpoint.get("scanned", 0)
        start_shard_index = checkpoint.get("num_shards", 0)
        if logger:
            logger.info(f"  Resuming from checkpoint: scanned={resume_from:,}, shards={start_shard_index}")

    writer = ShardWriter(config.shard_dir, config.shard_size, start_index=start_shard_index)

    extracted_total: int = checkpoint.get("extracted", 0) if checkpoint else 0
    skipped_empty: int = checkpoint.get("skipped_empty", 0) if checkpoint else 0
    skipped_parse: int = checkpoint.get("skipped_parse", 0) if checkpoint else 0
    skipped_parse_reasons: Counter[str] = (
        Counter(checkpoint.get("skipped_parse_reasons", {})) if checkpoint else Counter()
    )
    skipped_stub: int = checkpoint.get("skipped_stub", 0) if checkpoint else 0
    scanned: int = 0
    text_source_counts: Dict[str, int] = {field: 0 for field in config.text_source_fields}
    cleaning_flag_counts: Counter[str] = Counter()

    court_distribution: Counter[str] = Counter()
    opinion_type_distribution: Counter[str] = Counter()
    precedential_status_distribution: Counter[str] = Counter()
    year_distribution: Counter[str] = Counter()

    text_lengths: List[int] = []
    ocr_extracted_count: int = 0

    seen_opinion_ids: Set[int] = set()
    duplicate_opinion_ids: int = 0
    seen_text_hashes: Set[int] = set()
    duplicate_texts: int = 0

    schema_sample: Optional[List[str]] = None

    with _open_csv(opinions_path) as file_handle:
        reader = csv.DictReader(file_handle, escapechar="\\")
        for row in tqdm(reader, desc="Scanning opinions"):
            scanned += 1

            # Skip rows already processed in previous run
            if scanned <= resume_from:
                continue

            try:
                cluster_id = int(row.get("cluster_id", ""))
            except (ValueError, TypeError):
                skipped_parse += 1
                skipped_parse_reasons["bad_cluster_id"] += 1
                _write_quarantine(config.quarantine_path, "bad_cluster_id", row)
                if logger:
                    logger.debug(f"  Skipped row {scanned}: bad_cluster_id={row.get('cluster_id', '')!r}")
                continue

            if cluster_id not in federal_cluster_ids:
                continue

            raw_text, text_source = select_best_text(row, config.text_source_fields, config.min_text_length)

            if not raw_text:
                skipped_empty += 1
                continue

            normalized_text, cleaning_flags = _normalize_text(raw_text, text_source)
            for flag in cleaning_flags:
                cleaning_flag_counts[flag] += 1

            # Post-normalization stub filter
            if len(normalized_text) < config.min_text_length:
                skipped_stub += 1
                continue

            parsed_id = parse_opinion_id(row.get("id", ""))
            if parsed_id is None:
                skipped_parse += 1
                skipped_parse_reasons["bad_opinion_id"] += 1
                _write_quarantine(config.quarantine_path, "bad_opinion_id", row)
                if logger:
                    logger.debug(f"  Skipped row {scanned}: bad_opinion_id={row.get('id', '')!r}")
                continue
            opinion_id = parsed_id

            # Track text source AFTER all filters pass (fixes >100% percentages)
            text_source_counts[text_source] += 1

            record = build_record(
                opinion_id=opinion_id,
                cluster_id=cluster_id,
                raw_text=raw_text,
                normalized_text=normalized_text,
                text_source=text_source,
                cleaning_flags=cleaning_flags,
                opinion_type=row.get("type", ""),
                extracted_by_ocr=row.get("extracted_by_ocr", ""),
                cluster_meta=cluster_meta,
                docket_meta=docket_meta,
                court_name_map=court_name_map,
            )

            # Duplicate detection
            if opinion_id in seen_opinion_ids:
                duplicate_opinion_ids += 1
            seen_opinion_ids.add(opinion_id)

            text_hash = hash(normalized_text[:500])
            if text_hash in seen_text_hashes:
                duplicate_texts += 1
            seen_text_hashes.add(text_hash)

            # Collect stats
            court_distribution[record.court_id] += 1
            opinion_type_distribution[record.opinion_type] += 1
            precedential_status_distribution[record.precedential_status] += 1
            year_distribution[_extract_year(record.date_filed)] += 1
            text_lengths.append(record.text_length)
            if record.extracted_by_ocr in ("True", "true", True):
                ocr_extracted_count += 1
            if schema_sample is None:
                record_dict: Dict[str, Any] = record.to_dict()
                schema_sample = list(record_dict.keys())

            writer.add(record)
            extracted_total += 1

            if extracted_total % config.log_interval == 0 and logger:
                logger.info(f"  Extracted {extracted_total:,} (scanned {scanned:,})")

            # Periodic checkpoint
            if scanned % config.checkpoint_interval == 0:
                writer.flush()
                _save_checkpoint(
                    config.shard_dir,
                    {
                        "scanned": scanned,
                        "extracted": extracted_total,
                        "skipped_empty": skipped_empty,
                        "skipped_parse": skipped_parse,
                        "skipped_parse_reasons": dict(skipped_parse_reasons),
                        "skipped_stub": skipped_stub,
                        "num_shards": writer.shard_count,
                    },
                )

    writer.flush()

    # Final checkpoint
    _save_checkpoint(
        config.shard_dir,
        {
            "scanned": scanned,
            "extracted": extracted_total,
            "skipped_empty": skipped_empty,
            "skipped_parse": skipped_parse,
            "skipped_parse_reasons": dict(skipped_parse_reasons),
            "skipped_stub": skipped_stub,
            "num_shards": writer.shard_count,
        },
    )

    text_length_stats: Dict[str, int] = {}
    if text_lengths:
        sorted_lengths = sorted(text_lengths)
        n = len(sorted_lengths)
        text_length_stats = {
            "count": n,
            "mean": int(sum(sorted_lengths) / n),
            "min": sorted_lengths[0],
            "max": sorted_lengths[-1],
            "median": sorted_lengths[n // 2],
            "p25": sorted_lengths[n // 4],
            "p75": sorted_lengths[3 * n // 4],
            "p90": sorted_lengths[int(n * 0.9)],
            "p95": sorted_lengths[int(n * 0.95)],
            "p99": sorted_lengths[int(n * 0.99)],
        }

    token_stats: Dict[str, int] = {}
    if text_lengths:
        token_lengths = [length // 4 for length in sorted_lengths]
        n = len(token_lengths)
        token_stats = {
            "mean": int(sum(token_lengths) / n),
            "median": token_lengths[n // 2],
            "p95": token_lengths[int(n * 0.95)],
            "above_512": sum(1 for t in token_lengths if t > 512),
            "above_2048": sum(1 for t in token_lengths if t > 2048),
            "above_4096": sum(1 for t in token_lengths if t > 4096),
        }

    stats: Dict[str, Any] = {
        "extracted_total": extracted_total,
        "skipped_empty": skipped_empty,
        "skipped_parse": skipped_parse,
        "skipped_parse_reasons": dict(skipped_parse_reasons),
        "skipped_stub": skipped_stub,
        "scanned": scanned,
        "num_shards": writer.shard_count,
        "text_source_counts": text_source_counts,
        "cleaning_flag_counts": dict(cleaning_flag_counts),
        "court_distribution": dict(court_distribution),
        "opinion_type_distribution": dict(opinion_type_distribution),
        "precedential_status_distribution": dict(precedential_status_distribution),
        "year_distribution": dict(year_distribution),
        "text_length_stats": text_length_stats,
        "token_stats": token_stats,
        "ocr_extracted_count": ocr_extracted_count,
        "duplicate_opinion_ids": duplicate_opinion_ids,
        "duplicate_texts": duplicate_texts,
        "schema": schema_sample,
    }

    if logger:
        logger.info(f"\n  Scanned:          {scanned:,}")
        logger.info(f"  Extracted:        {extracted_total:,}")
        logger.info(f"  Skipped empty:    {skipped_empty:,}")
        logger.info(f"  Skipped parse:    {skipped_parse:,}")
        logger.info(f"  Skipped stub:     {skipped_stub:,}")
        if skipped_parse_reasons:
            for reason, count in skipped_parse_reasons.most_common():
                logger.info(f"    {reason:<20} {count:>8,}")
        logger.info(f"  Duplicate IDs:    {duplicate_opinion_ids:,}")
        logger.info(f"  Duplicate texts:  {duplicate_texts:,}")
        logger.info(f"  OCR-extracted:    {ocr_extracted_count:,}")
        logger.info(f"  Shards:           {writer.shard_count}")
        if text_length_stats:
            logger.info(
                f"\n  Text length (chars): mean={text_length_stats['mean']:,} "
                f"median={text_length_stats['median']:,} "
                f"p95={text_length_stats['p95']:,} "
                f"max={text_length_stats['max']:,}"
            )
        if token_stats:
            logger.info(
                f"  Token estimate:      mean={token_stats['mean']:,} "
                f"median={token_stats['median']:,} "
                f">512={token_stats['above_512']:,} "
                f">4096={token_stats['above_4096']:,}"
            )
        logger.info("\n  Text sources:")
        for source_name, count in text_source_counts.items():
            if count > 0:
                pct = count / extracted_total * 100 if extracted_total else 0
                logger.info(f"    {source_name:<25} {count:>8,} ({pct:.1f}%)")
        if cleaning_flag_counts:
            logger.info("\n  Cleaning flags:")
            for flag, count in cleaning_flag_counts.most_common():
                logger.info(f"    {flag:<25} {count:>8,}")
        logger.info("\n  Courts:")
        for court_name, count in court_distribution.most_common():
            logger.info(f"    {court_name:<10} {count:>8,}")
        logger.info("\n  Top years:")
        for year, count in sorted(year_distribution.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"    {year:<6} {count:>8,}")

    return stats
