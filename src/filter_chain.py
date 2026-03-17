# src/filter_chain.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/src/filter_chain.py
# SRP: Build federal appellate filter chain: courts → dockets → clusters.

from pathlib import Path  # used: file paths
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd  # used: CSV loading and filtering
from tqdm import tqdm  # used: progress on large chunks

from src.config import PipelineConfig
from src.schemas import FilterResult

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
    """Read first 10 rows to catch format issues early."""
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
    """Load courts CSV, return federal appellate metadata."""
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
    """Load dockets CSV in chunks, filter by federal appellate courts."""
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
    """Load opinion clusters CSV in chunks, filter by federal appellate dockets."""
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
    """Run full filter chain: courts → dockets → clusters."""
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
