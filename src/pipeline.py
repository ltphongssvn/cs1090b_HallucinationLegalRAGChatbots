# src/pipeline.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/src/pipeline.py
# SRP: Orchestrate bulk data acquisition. Supports pinned snapshots.

from typing import Any, Dict, Optional

from src.bulk_download import download_bulk_csvs
from src.config import PipelineConfig
from src.exceptions import PipelineError
from src.extract import extract_opinions_to_shards
from src.filter_chain import build_federal_appellate_filter
from src.manifest import read_manifest, validate_manifest_shards, write_manifest
from src.s3_discovery import discover_latest_bulk_files
from src.validation import run_contract_tests


def run_pipeline(
    config: Optional[PipelineConfig] = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """Run full bulk data acquisition. Returns manifest dict."""
    if config is None:
        config = PipelineConfig()

    existing = read_manifest(config.manifest_path)
    if existing and validate_manifest_shards(existing, config.shard_dir):
        if logger:
            logger.info(
                f"✓ Already complete: {existing['num_cases']:,} cases, {existing['num_shards']} shards verified"
            )
        return existing

    if config.has_pinned_snapshot:
        if logger:
            logger.info("STEP 1: Using pinned snapshot (reproducible)...")
        latest_files = config.pinned_files
    else:
        if logger:
            logger.info("STEP 1: Discovering bulk files on S3...")
        latest_files = discover_latest_bulk_files(config=config)

    if latest_files is None:
        raise PipelineError("No pinned files and discovery returned None")

    if logger:
        for label, info in latest_files.items():
            logger.info(f"  {label:<12} {info['key']}")

    if logger:
        logger.info("\nSTEP 2: Downloading bulk CSVs...")
    local_paths = download_bulk_csvs(latest_files, config=config, logger=logger)

    if logger:
        logger.info("\nSTEP 3: Building filter chain...")
    filter_result = build_federal_appellate_filter(local_paths, config=config, logger=logger)

    if logger:
        logger.info(f"\nSTEP 4: Extracting → shards (size={config.shard_size:,})...")
    stats = extract_opinions_to_shards(
        opinions_path=local_paths["opinions"],
        cluster_meta=filter_result.cluster_meta,
        docket_meta=filter_result.docket_meta,
        court_name_map=filter_result.court_name_map,
        config=config,
        logger=logger,
    )

    manifest_data = write_manifest(
        manifest_path=config.manifest_path,
        shard_dir=config.shard_dir,
        extraction_stats=stats,
        local_paths=local_paths,
        fed_court_ids=filter_result.fed_court_ids,
        docket_count=len(filter_result.docket_meta),
        cluster_count=len(filter_result.cluster_meta),
        shard_size=config.shard_size,
        config=config,
    )
    if logger:
        logger.info(f"Manifest: {config.manifest_path}")

    return manifest_data


def validate_pipeline(
    config: Optional[PipelineConfig] = None,
    manifest_data: Optional[Dict[str, Any]] = None,
    logger: Any = None,
    shard_strategy: str = "sample",
) -> bool:
    """Run TDD contract tests."""
    if config is None:
        config = PipelineConfig()
    passed = run_contract_tests(
        config=config,
        manifest_data=manifest_data,
        logger=logger,
        shard_strategy=shard_strategy,
    )
    if not passed:
        raise PipelineError("TDD contract tests failed")
    return True
