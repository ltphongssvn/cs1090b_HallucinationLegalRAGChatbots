# src/config.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/config.py
"""Typed pipeline configuration for the CourtListener ingest pipeline.

This module defines a single :class:`PipelineConfig` dataclass that is
passed explicitly to every function in the ingestion layer. Centralising
configuration in one immutable-by-convention object keeps the pipeline
functional and testable: no module-level globals, no environment lookups
scattered through business logic, and every knob is discoverable via
``help(PipelineConfig)``.

Reproducibility
---------------
The ``pinned_*`` fields support **snapshot pinning**: when all four are
set, :attr:`PipelineConfig.has_pinned_snapshot` becomes ``True`` and
:attr:`PipelineConfig.pinned_files` yields a dict in the same shape as
the discovery output of :mod:`src.s3_discovery`. This lets an experiment
re-fetch the exact same CourtListener dump used in a prior run by
recording the four object keys in a config file rather than relying on
"latest" semantics.

SRP
---
This module only *describes* configuration. It performs no I/O, no
validation of remote resources, and no file-system mutation. Pure data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple


@dataclass
class PipelineConfig:
    """All parameters controlling the CourtListener ingest pipeline.

    The dataclass is intentionally **not** ``frozen=True`` so that call
    sites can selectively override fields for tests, but production code
    should treat instances as immutable and construct a new
    :class:`PipelineConfig` rather than mutating fields.

    Attributes:
        bulk_dir: Directory into which the raw CourtListener bulk CSVs
            are downloaded (see :mod:`src.bulk_download`).
        shard_dir: Directory holding extracted/normalised shards plus
            the run manifest.
        shard_size: Number of rows per output shard during extraction.
            10,000 is tuned to keep per-shard parquet files under
            ~100 MB for the opinions table.
        s3_bucket_url: Public HTTPS endpoint of the CourtListener bucket.
        s3_bucket_name: Bucket name only, used by the ``aws`` CLI path.
        s3_prefix: Key prefix under which the dated bulk dumps live.
        pinned_courts: Optional exact S3 key of a courts CSV dump.
        pinned_dockets: Optional exact S3 key of a dockets CSV dump.
        pinned_clusters: Optional exact S3 key of an opinion-clusters
            CSV dump.
        pinned_opinions: Optional exact S3 key of an opinions CSV dump.
        min_text_length: Rows whose extracted opinion text is shorter
            than this many characters are dropped as noise. 50 is a
            conservative threshold that removes stub/placeholder rows
            without losing genuine per-curiam orders.
        text_source_fields: Ordered tuple of CourtListener opinion
            columns from which to draw the canonical text. The first
            non-empty field wins; HTML variants are parsed downstream.
        log_interval: Emit a progress log line every N rows during
            extraction.
        checksum_buffer_size: Block size for SHA-256 streaming over
            downloaded files.
        csv_chunksize: Rows per ``pandas.read_csv`` chunk — sized to fit
            comfortably in ~1 GB of RAM for the opinions table.
        quarantine_path: Optional path to a JSONL file that receives
            rows rejected by the validator. ``None`` discards them.
        checkpoint_interval: Rows between checkpoint writes during
            extraction, enabling crash-resume.
        federal_appellate_court_ids: Frozen set of CourtListener court
            slugs that constitute the federal appellate corpus. Held as
            a frozenset both for O(1) membership tests and to signal
            immutability.
        schema_audit_per_shard: Number of rows per shard that are
            deep-validated against :mod:`src.schemas`. Full validation
            of every row would dominate pipeline wall time.
        min_expected_total: Lower bound on the total row count of a
            healthy run. Used by the post-run sanity check.
    """

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
            "ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7",
            "ca8", "ca9", "ca10", "ca11", "cadc", "cafc",
        }
    )

    # Validation
    schema_audit_per_shard: int = 50
    min_expected_total: int = 10000

    def __post_init__(self) -> None:
        """Coerce string paths to :class:`~pathlib.Path` instances.

        When a :class:`PipelineConfig` is instantiated from a YAML/JSON
        dict (as in :mod:`hydra`/:mod:`omegaconf` workflows), path
        fields arrive as plain strings. Coercing here lets the rest of
        the codebase rely on :class:`Path` methods unconditionally.
        """
        if not isinstance(self.bulk_dir, Path):
            self.bulk_dir = Path(self.bulk_dir)
        if not isinstance(self.shard_dir, Path):
            self.shard_dir = Path(self.shard_dir)
        if self.quarantine_path is not None and not isinstance(self.quarantine_path, Path):
            self.quarantine_path = Path(self.quarantine_path)

    @property
    def manifest_path(self) -> Path:
        """Path to the run manifest JSON inside :attr:`shard_dir`."""
        return self.shard_dir / "manifest.json"

    @property
    def needed_files(self) -> Dict[str, str]:
        """Mapping of corpus label → filename prefix on the S3 bucket.

        Consumed by :mod:`src.s3_discovery` when scanning
        ``s3_prefix`` for the newest dump of each table.
        """
        return {
            "courts": "courts-",
            "dockets": "dockets-",
            "clusters": "opinion-clusters-",
            "opinions": "opinions-",
        }

    @property
    def has_pinned_snapshot(self) -> bool:
        """``True`` iff all four ``pinned_*`` fields are populated.

        A partial pin is treated as *not pinned* — it is safer to fall
        back to discovery than to mix a pinned opinions dump with a
        live dockets dump and silently get a cross-snapshot inconsistency.
        """
        return all(
            [
                self.pinned_courts,
                self.pinned_dockets,
                self.pinned_clusters,
                self.pinned_opinions,
            ]
        )

    @property
    def pinned_files(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Return the pinned snapshot in discovery-output shape, or ``None``.

        The shape matches the dict returned by
        :func:`src.s3_discovery.discover_latest_bulk_files` so that
        :func:`src.bulk_download.download_bulk_csvs` can consume either
        a pinned snapshot or a fresh discovery without branching.

        The ``size`` and ``size_mb`` fields are filled with sentinel
        zeros because a pinned config does not carry object metadata;
        downstream code must not rely on them when a pin is in use.
        """
        if not self.has_pinned_snapshot:
            return None
        return {
            "courts": {"key": self.pinned_courts, "size": 0, "size_mb": 0},
            "dockets": {"key": self.pinned_dockets, "size": 0, "size_mb": 0},
            "clusters": {"key": self.pinned_clusters, "size": 0, "size_mb": 0},
            "opinions": {"key": self.pinned_opinions, "size": 0, "size_mb": 0},
        }
