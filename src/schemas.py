# src/schemas.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/schemas.py
"""Typed dataclass schemas for every pipeline data structure.

Centralising these definitions gives the pipeline a single source of
truth for field names and types. Replacing ``Dict[str, Any]`` with
these dataclasses catches typos at test time rather than at the
manifest-audit step days later.

Schemas
-------
* :class:`DocketMeta` — slim metadata held per federal-appellate docket.
* :class:`ClusterMeta` — slim metadata held per opinion cluster.
* :class:`OpinionRecord` — the canonical per-opinion output row written
  to every JSONL/Parquet shard.
* :class:`ManifestData` — typed view of the run manifest.
* :class:`FilterResult` — bundle returned by the three-stage filter
  chain (courts → dockets → clusters).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class DocketMeta:
    """Minimal docket metadata retained after the federal-court filter.

    Attributes:
        court_id: CourtListener court slug (e.g. ``"ca9"``).
        case_name: Human-readable case caption.
        date_filed: Filing date as a string; format varies upstream.
    """

    court_id: str
    case_name: str
    date_filed: str


@dataclass
class ClusterMeta:
    """Minimal opinion-cluster metadata retained after the docket filter.

    Attributes:
        docket_id: Parent docket ID (join key into :class:`DocketMeta`).
        case_name: Cluster-level case caption; may differ from the
            docket's caption for consolidated cases.
        date_filed: Cluster filing date.
        precedential_status: CourtListener status string
            (``"Published"``, ``"Unpublished"``, etc.), used
            downstream to derive :attr:`OpinionRecord.is_precedential`.
    """

    docket_id: int
    case_name: str
    date_filed: str
    precedential_status: str


@dataclass
class OpinionRecord:
    """Canonical per-opinion row emitted to every output shard.

    Combines the joined court/docket/cluster provenance with the
    normalised text and the quality-metric features computed by
    :func:`src.extract.build_record`.

    Attributes:
        id: CourtListener opinion ID (primary key).
        cluster_id: Parent cluster ID.
        docket_id: Parent docket ID, resolved via the cluster join.
            ``None`` only when the cluster record lacks a docket link.
        court_id: Court slug from the docket.
        court_name: Human-readable court name from the courts table.
        case_name: Case caption, preferring the cluster's value.
        date_filed: Filing date, preferring the cluster's value.
        precedential_status: Raw status string from the cluster.
        opinion_type: CourtListener opinion type code.
        extracted_by_ocr: Raw OCR flag string from the source row.
        raw_text: Unmodified text of the chosen source field.
        text: Text after the :func:`_normalize_text` pipeline.
        text_length: ``len(text)`` — character count.
        text_source: Name of the chosen CSV text column.
        cleaning_flags: Normalisation stages that fired for this row.
        source: Origin tag, always ``"courtlistener_bulk"``.
        token_count: Estimated token count (``text_length // 4``).
        paragraph_count: Paragraphs separated by blank lines (min 1).
        citation_count: Matches of :data:`_CITATION_RE` in ``text``.
        text_hash: SHA-256 hex of the normalised text for dedup.
        citation_density: Citations per 1K estimated tokens.
        is_precedential: Derived boolean; ``True`` for published or
            precedential statuses.
        text_entropy: Shannon entropy of the character distribution.
    """

    id: int
    cluster_id: int
    docket_id: Optional[int]
    court_id: str
    court_name: str
    case_name: str
    date_filed: str
    precedential_status: str
    opinion_type: str
    extracted_by_ocr: str
    raw_text: str
    text: str
    text_length: int
    text_source: str
    cleaning_flags: List[str]
    source: str
    token_count: int = 0
    paragraph_count: int = 1
    citation_count: int = 0
    text_hash: str = ""
    citation_density: float = 0.0
    is_precedential: bool = False
    text_entropy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict copy for JSON/Parquet serialisation."""
        return asdict(self)


@dataclass
class ManifestData:
    """Typed view of the run manifest written by :mod:`src.manifest`.

    Mirrors the JSON layout so typed code paths can consume manifests
    without ``Dict[str, Any]`` plumbing. Field defaults match the
    empty-state a fresh manifest would have before any stage runs.

    Attributes:
        num_cases: Total extracted opinion count.
        num_shards: Number of shard files emitted.
        shard_size: Rows per shard used by the writer.
        version: Manifest schema version; bump on layout changes.
        source_files: Label → source CSV basename.
        source_checksums: Label → SHA-256 of the source CSV.
        federal_courts: Sorted list of court slugs used.
        filter_chain: Stage-by-stage count summary.
        text_source_counts: Per-text-field row counts.
        court_distribution: Per-court row counts.
        text_length_stats: Percentile summary of text lengths.
        checksum: Shard filename → SHA-256 map.
        run_metadata: Timestamp, Python version, git SHA, config snapshot.
    """

    num_cases: int
    num_shards: int
    shard_size: int
    version: int = 2
    source_files: Dict[str, str] = field(default_factory=dict)
    source_checksums: Dict[str, str] = field(default_factory=dict)
    federal_courts: List[str] = field(default_factory=list)
    filter_chain: Dict[str, Any] = field(default_factory=dict)
    text_source_counts: Dict[str, int] = field(default_factory=dict)
    court_distribution: Dict[str, int] = field(default_factory=dict)
    text_length_stats: Dict[str, int] = field(default_factory=dict)
    checksum: Dict[str, str] = field(default_factory=dict)
    run_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Combined result of the three-stage federal appellate filter chain.

    Replaces the earlier stringly-typed ``Dict[str, Any]`` return from
    :func:`src.filter_chain.build_federal_appellate_filter`, letting
    the extractor consume named attributes with IDE completion.

    Attributes:
        fed_court_ids: Set of federal appellate court slugs.
        court_name_map: Court slug → human-readable name.
        docket_meta: Docket ID → docket metadata dict.
        cluster_meta: Cluster ID → cluster metadata dict.
    """

    fed_court_ids: Set[str]
    court_name_map: Dict[str, str]
    docket_meta: Dict[int, Dict[str, Any]]
    cluster_meta: Dict[int, Dict[str, Any]]
