# src/filter_chain.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/filter_chain.py
"""Federal appellate filter chain: courts → dockets → clusters.

Builds the cluster-ID allow-list that :mod:`src.extract` uses to decide
which opinion rows to keep. The chain runs in three stages, each narrowing
the next:

1. **Courts** — load the courts CSV and retain only the 13 federal
   appellate courts listed in
   :attr:`PipelineConfig.federal_appellate_court_ids`.
2. **Dockets** — stream the dockets CSV in chunks and keep dockets whose
   ``court_id`` is in the federal set, indexing them by docket ID.
3. **Clusters** — stream the opinion-clusters CSV in chunks and keep
   clusters whose ``docket_id`` is in the federal docket set, indexing
   them by cluster ID.

The result is returned as a :class:`~src.schemas.FilterResult` holding
all four maps (court IDs, court names, docket meta, cluster meta) so the
downstream extractor can perform the final cluster→docket→court join
without re-reading any CSV.

Design notes
------------
* **Chunked reads**: dockets and clusters are too large to fit in RAM at
  once. :func:`pandas.read_csv` is called with ``chunksize`` so memory
  stays bounded regardless of dump size.
* **PostgreSQL COPY format**: the CourtListener dump uses ``\\`` as an
  escape character; every read passes ``escapechar="\\\\"`` via
  :data:`CSV_READ_KWARGS`.
* **Soft error tolerance**: ``on_bad_lines="skip"`` and
  ``encoding_errors="replace"`` keep a malformed row from aborting the
  whole chain — bad rows are simply excluded from the filter set.
* **Smoke test**: each large CSV is probed with a 10-row read before the
  chunked scan, surfacing format/column issues in milliseconds rather
  than after 20 minutes of streaming.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from tqdm import tqdm

from src.config import PipelineConfig
from src.schemas import FilterResult

#: Shared :func:`pandas.read_csv` kwargs for all large CourtListener CSVs.
#:
#: * ``on_bad_lines="skip"`` — tolerate malformed rows rather than abort.
#: * ``escapechar="\\\\"`` — match PostgreSQL ``COPY`` escaping.
#: * ``encoding_errors="replace"`` — survive occasional invalid UTF-8.
#: * ``low_memory=False`` — suppress dtype-guessing warnings on wide chunks.
CSV_READ_KWARGS: Dict[str, Any] = {
    "on_bad_lines": "skip",
    "escapechar": "\\",
    "encoding_errors": "replace",
    "low_memory": False,
}


def _smoke_test_csv(
    filepath: Union[str, Path],
    usecols: List[str],
    logger: Any = None,
) -> bool:
    """Read the first 10 rows of ``filepath`` to validate format early.

    Used by the loaders below as a fail-fast probe before launching the
    full chunked scan — catches missing columns, wrong separators, and
    truncated files in milliseconds.

    Args:
        filepath: Path to the CSV to probe.
        usecols: Columns that must be present; surfaces missing-column
            errors at probe time.
        logger: Optional logger. When supplied, emits one OK or FAIL
            line per probe.

    Returns:
        ``True`` on successful read, ``False`` on any exception.
    """
    try:
        sample = pd.read_csv(filepath, usecols=usecols, nrows=10, escapechar="\\", encoding_errors="replace")
        if logger:
            logger.info(f"  Smoke test OK: {Path(filepath).name} ({len(sample.columns)} cols)")
        return True
    except Exception as error:
        if logger:
            logger.error(f"  Smoke test FAIL: {Path(filepath).name} — {error}")
        return False


def load_federal_courts(
    courts_path: Union[str, Path],
    config: Optional[PipelineConfig] = None,
    logger: Any = None,
) -> Tuple[Set[str], Dict[str, str]]:
    """Load the courts CSV and return the federal appellate subset.

    The courts file is small enough to load whole — no chunking needed.
    Filters rows whose ``id`` is in
    :attr:`PipelineConfig.federal_appellate_court_ids` and builds a
    court_id → ``full_name`` map for downstream display.

    Args:
        courts_path: Path to the courts CSV.
        config: Pipeline configuration (defaults to :class:`PipelineConfig`).
        logger: Optional logger for per-court listing.

    Returns:
        ``(federal_court_ids, court_name_map)`` where the set holds the
        matched court slugs and the map goes from slug to display name.
    """
    if config is None:
        config = PipelineConfig()

    courts_dataframe = pd.read_csv(courts_path, escapechar="\\", encoding_errors="replace")
    federal_mask = courts_dataframe["id"].isin(config.federal_appellate_court_ids)
    federal_courts = courts_dataframe[federal_mask]
    federal_court_ids: Set[str] = set(federal_courts["id"].values)
    court_name_map: Dict[str, str] = dict(
        zip(federal_courts["id"], federal_courts.get("full_name", federal_courts["id"]))
    )

    if logger:
        logger.info(f"  Federal appellate courts: {len(federal_court_ids)}")
        for court_id in sorted(federal_court_ids):
            logger.info(f"    {court_id:<10} {court_name_map[court_id]}")

    return federal_court_ids, court_name_map


def load_federal_dockets(
    dockets_path: Union[str, Path],
    federal_court_ids: Set[str],
    config: Optional[PipelineConfig] = None,
    logger: Any = None,
) -> Dict[int, Dict[str, Any]]:
    """Stream the dockets CSV and index federal-appellate rows by docket ID.

    Reads in ``csv_chunksize`` chunks so RAM usage is bounded regardless
    of dump size. Each surviving row is stored as a minimal metadata
    dict (court, case name, filing date) for the later cluster-level join.

    Args:
        dockets_path: Path to the dockets CSV (plain or ``.bz2``).
        federal_court_ids: Allow-list produced by
            :func:`load_federal_courts`.
        config: Pipeline configuration.
        logger: Optional logger for total/matched counts.

    Returns:
        Mapping of docket ID → ``{"court_id", "case_name", "date_filed"}``.
    """
    if config is None:
        config = PipelineConfig()
    columns = ["id", "court_id", "case_name", "date_filed"]
    _smoke_test_csv(dockets_path, columns, logger=logger)
    chunks = pd.read_csv(dockets_path, usecols=columns, chunksize=config.csv_chunksize, **CSV_READ_KWARGS)

    docket_metadata_map: Dict[int, Dict[str, Any]] = {}
    total_rows: int = 0
    for chunk in tqdm(chunks, desc="Scanning dockets"):
        total_rows += len(chunk)
        matched = chunk[chunk["court_id"].isin(federal_court_ids)]
        for row in matched.itertuples(index=False):
            docket_metadata_map[row.id] = {
                "court_id": row.court_id,
                "case_name": getattr(row, "case_name", ""),
                "date_filed": str(getattr(row, "date_filed", "")),
            }

    if logger:
        logger.info(f"  Total dockets: {total_rows:,}")
        logger.info(f"  Federal appellate dockets: {len(docket_metadata_map):,}")

    return docket_metadata_map


def load_federal_clusters(
    clusters_path: Union[str, Path],
    federal_docket_ids: Set[int],
    config: Optional[PipelineConfig] = None,
    logger: Any = None,
) -> Dict[int, Dict[str, Any]]:
    """Stream the opinion-clusters CSV and keep clusters in federal dockets.

    Same chunked-read pattern as :func:`load_federal_dockets`. Rows whose
    ``docket_id`` cannot be parsed as an int are silently dropped — a
    cluster without a valid docket reference cannot be joined downstream
    anyway.

    Args:
        clusters_path: Path to the opinion-clusters CSV (plain or ``.bz2``).
        federal_docket_ids: Allow-list of docket IDs from
            :func:`load_federal_dockets`.
        config: Pipeline configuration.
        logger: Optional logger for total/matched counts.

    Returns:
        Mapping of cluster ID → ``{"docket_id", "case_name",
        "date_filed", "precedential_status"}``.
    """
    if config is None:
        config = PipelineConfig()
    columns = ["id", "docket_id", "case_name", "date_filed", "precedential_status"]
    _smoke_test_csv(clusters_path, columns, logger=logger)
    chunks = pd.read_csv(clusters_path, usecols=columns, chunksize=config.csv_chunksize, **CSV_READ_KWARGS)

    cluster_metadata_map: Dict[int, Dict[str, Any]] = {}
    total_rows: int = 0
    for chunk in tqdm(chunks, desc="Scanning clusters"):
        total_rows += len(chunk)
        matched = chunk[chunk["docket_id"].isin(federal_docket_ids)]
        for row in matched.itertuples(index=False):
            try:
                cluster_metadata_map[row.id] = {
                    "docket_id": int(row.docket_id),
                    "case_name": getattr(row, "case_name", ""),
                    "date_filed": str(getattr(row, "date_filed", "")),
                    "precedential_status": getattr(row, "precedential_status", ""),
                }
            except (ValueError, TypeError):
                continue

    if logger:
        logger.info(f"  Total clusters: {total_rows:,}")
        logger.info(f"  Federal appellate clusters: {len(cluster_metadata_map):,}")

    return cluster_metadata_map


def build_federal_appellate_filter(
    local_paths: Dict[str, Any],
    config: Optional[PipelineConfig] = None,
    logger: Any = None,
) -> FilterResult:
    """Run the full three-stage filter chain and return its combined result.

    Orchestrates :func:`load_federal_courts` →
    :func:`load_federal_dockets` → :func:`load_federal_clusters` with
    the IDs from each stage feeding the next. The returned
    :class:`FilterResult` contains everything :mod:`src.extract` needs
    for the final opinion-level join.

    Args:
        local_paths: Mapping with keys ``"courts"``, ``"dockets"``,
            ``"clusters"`` (produced by
            :func:`src.bulk_download.download_bulk_csvs`).
        config: Pipeline configuration.
        logger: Optional logger for stage progress.

    Returns:
        A :class:`FilterResult` bundling the court ID set, court name
        map, docket metadata map, and cluster metadata map.
    """
    if config is None:
        config = PipelineConfig()

    if logger:
        logger.info("Loading courts...")
    federal_court_ids, court_name_map = load_federal_courts(local_paths["courts"], config=config, logger=logger)

    if logger:
        logger.info("\nLoading dockets...")
    docket_metadata_map = load_federal_dockets(local_paths["dockets"], federal_court_ids, config=config, logger=logger)

    if logger:
        logger.info("\nLoading opinion clusters...")
    cluster_metadata_map = load_federal_clusters(
        local_paths["clusters"], set(docket_metadata_map.keys()), config=config, logger=logger
    )

    return FilterResult(
        fed_court_ids=federal_court_ids,
        court_name_map=court_name_map,
        docket_meta=docket_metadata_map,
        cluster_meta=cluster_metadata_map,
    )
