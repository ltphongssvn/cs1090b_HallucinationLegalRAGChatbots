# src/config.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/src/config.py
# SRP: Single typed config object. Supports pinned snapshots for reproducibility.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple


@dataclass
class PipelineConfig:
    """All pipeline parameters. Passed explicitly to every function."""

    # Storage
    bulk_dir: Path = Path("data/raw/cl_bulk")
    shard_dir: Path = Path("data/raw/cl_federal_appellate_bulk")
    shard_size: int = 10000

    # S3
    s3_bucket_url: str = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com"
    s3_bucket_name: str = "com-courtlistener-storage"
    s3_prefix: str = "bulk-data/"

    # Pinned snapshot for reproducibility
    pinned_courts: Optional[str] = None
    pinned_dockets: Optional[str] = None
    pinned_clusters: Optional[str] = None
    pinned_opinions: Optional[str] = None

    # Extraction
    min_text_length: int = 50
    text_source_fields: Tuple[str, ...] = (
        "plain_text",
        "html_with_citations",
        "html",
        "html_lawbox",
        "html_columbia",
    )
    log_interval: int = 50_000
    checksum_buffer_size: int = 8192
    csv_chunksize: int = 500_000
    quarantine_path: Optional[Path] = None
    checkpoint_interval: int = 50_000

    # Courts
    federal_appellate_court_ids: FrozenSet[str] = frozenset(
        {
            "ca1",
            "ca2",
            "ca3",
            "ca4",
            "ca5",
            "ca6",
            "ca7",
            "ca8",
            "ca9",
            "ca10",
            "ca11",
            "cadc",
            "cafc",
        }
    )

    # Validation
    schema_audit_per_shard: int = 50
    min_expected_total: int = 10000

    @property
    def manifest_path(self) -> Path:
        return self.shard_dir / "manifest.json"

    @property
    def needed_files(self) -> Dict[str, str]:
        return {
            "courts": "courts-",
            "dockets": "dockets-",
            "clusters": "opinion-clusters-",
            "opinions": "opinions-",
        }

    @property
    def has_pinned_snapshot(self) -> bool:
        return all([self.pinned_courts, self.pinned_dockets, self.pinned_clusters, self.pinned_opinions])

    @property
    def pinned_files(self) -> Optional[Dict[str, Dict[str, Any]]]:
        if not self.has_pinned_snapshot:
            return None
        return {
            "courts": {"key": self.pinned_courts, "size": 0, "size_mb": 0},
            "dockets": {"key": self.pinned_dockets, "size": 0, "size_mb": 0},
            "clusters": {"key": self.pinned_clusters, "size": 0, "size_mb": 0},
            "opinions": {"key": self.pinned_opinions, "size": 0, "size_mb": 0},
        }
