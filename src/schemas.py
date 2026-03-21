# src/schemas.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/src/schemas.py
# SRP: Typed schemas for pipeline data structures.

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class DocketMeta:
    court_id: str
    case_name: str
    date_filed: str


@dataclass
class ClusterMeta:
    docket_id: int
    case_name: str
    date_filed: str
    precedential_status: str


@dataclass
class OpinionRecord:
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
        return asdict(self)


@dataclass
class ManifestData:
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
    """Typed result from filter chain — replaces stringly-typed Dict[str, Any]."""

    fed_court_ids: Set[str]
    court_name_map: Dict[str, str]
    docket_meta: Dict[int, Dict[str, Any]]
    cluster_meta: Dict[int, Dict[str, Any]]
