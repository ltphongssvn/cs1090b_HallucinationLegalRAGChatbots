#!/usr/bin/env python3
# scripts/verify_gcs_data.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/verify_gcs_data.py
"""Verify that required data files are present in GCS bucket.

This script checks for the presence of CourtListener bulk CSVs, processed
shards, and LePaRD dataset files in a Google Cloud Storage bucket. It's
designed to be run before starting a Colab training session to ensure all
required data is available for streaming.

Usage:
    python scripts/verify_gcs_data.py --bucket my-bucket --project my-project --check-all
    python scripts/verify_gcs_data.py --bucket my-bucket --project my-project --check-bulk
    python scripts/verify_gcs_data.py --bucket my-bucket --project my-project --check-shards
    python scripts/verify_gcs_data.py --bucket my-bucket --project my-project --check-lepard

Exit codes:
    0: All requested checks passed
    1: One or more checks failed
    2: Configuration error (authentication failed)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig
from src.gcs_utils import (
    GCSConfig,
    authenticate_gcs_colab,
    gcs_exists,
    gcs_get_info,
    gcs_list_files,
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the verification script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="  %(message)s",
    )
    return logging.getLogger("verify_gcs")


def verify_bulk_csvs(gcs_cfg: GCSConfig, pipeline_cfg: PipelineConfig, logger: logging.Logger) -> bool:
    """Verify CourtListener bulk CSVs are present in GCS.

    Args:
        gcs_cfg: GCS configuration.
        pipeline_cfg: Pipeline configuration with GCS prefixes.
        logger: Logger instance.

    Returns:
        True if all bulk CSVs are present, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("Verifying CourtListener bulk CSVs in GCS")
    logger.info("=" * 60)
    logger.info(f"Bucket: gs://{gcs_cfg.bucket_name}/{pipeline_cfg.gcs_cl_bulk_prefix}")

    required_files = [
        "courts-2026-03-31.csv.bz2",
        "dockets-2026-03-31.csv.bz2",
        "opinion-clusters-2026-03-31.csv.bz2",
        "opinions-2026-03-31.csv.bz2",
    ]

    all_present = True
    for filename in required_files:
        blob_path = pipeline_cfg.gcs_cl_bulk_prefix + filename

        if gcs_exists(blob_path, gcs_cfg):
            try:
                info = gcs_get_info(blob_path, gcs_cfg)
                size_gb = info["size"] / 1e9
                logger.info(f"  ✓ {filename}  {size_gb:.2f} GB")
            except Exception as e:
                logger.warning(f"  ✓ {filename}  (size check failed: {e})")
        else:
            logger.error(f"  ✗ {filename} NOT FOUND")
            all_present = False

    if all_present:
        logger.info(f"\n✓ All {len(required_files)} bulk CSVs present in GCS")
    else:
        logger.error("\n✗ Some bulk CSVs are missing from GCS")

    return all_present


def verify_shards(gcs_cfg: GCSConfig, pipeline_cfg: PipelineConfig, logger: logging.Logger) -> bool:
    """Verify processed shards are present in GCS.

    Args:
        gcs_cfg: GCS configuration.
        pipeline_cfg: Pipeline configuration with GCS prefixes.
        logger: Logger instance.

    Returns:
        True if manifest and shards are present, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("Verifying processed shards in GCS")
    logger.info("=" * 60)
    logger.info(f"Bucket: gs://{gcs_cfg.bucket_name}/{pipeline_cfg.gcs_cl_shards_prefix}")

    manifest_blob = pipeline_cfg.gcs_cl_shards_prefix + "manifest.json"

    if not gcs_exists(manifest_blob, gcs_cfg):
        logger.error("  ✗ manifest.json NOT FOUND")
        logger.error("\n✗ Shards not found in GCS - pipeline needs to be run first")
        return False

    logger.info("  ✓ manifest.json found")

    try:
        shard_files = [
            f
            for f in gcs_list_files(pipeline_cfg.gcs_cl_shards_prefix, gcs_cfg)
            if f.endswith(".jsonl")
        ]

        if not shard_files:
            logger.error("  ✗ No .jsonl shard files found")
            return False

        logger.info(f"  ✓ {len(shard_files)} shards found")

        for f in sorted(shard_files)[:5]:
            try:
                info = gcs_get_info(f, gcs_cfg)
                size_mb = info["size"] / 1e6
                filename = f.split("/")[-1]
                logger.info(f"    {filename}  {size_mb:.1f} MB")
            except Exception:
                pass

        if len(shard_files) > 5:
            logger.info(f"    ... and {len(shard_files) - 5} more shards")

        logger.info("\n✓ Shards ready for streaming from GCS")
        return True

    except Exception as e:
        logger.error(f"  ✗ Error listing shards: {e}")
        return False


def verify_lepard(gcs_cfg: GCSConfig, pipeline_cfg: PipelineConfig, logger: logging.Logger) -> bool:
    """Verify LePaRD dataset is present in GCS.

    Args:
        gcs_cfg: GCS configuration.
        pipeline_cfg: Pipeline configuration with GCS prefixes.
        logger: Logger instance.

    Returns:
        True if LePaRD files are present, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("Verifying LePaRD dataset in GCS")
    logger.info("=" * 60)
    logger.info(f"Bucket: gs://{gcs_cfg.bucket_name}/{pipeline_cfg.gcs_lepard_prefix}")

    lepard_jsonl = pipeline_cfg.gcs_lepard_prefix + "lepard_train_4000000_rev0194f95.jsonl"

    if not gcs_exists(lepard_jsonl, gcs_cfg):
        logger.error("  ✗ lepard_train_4000000_rev0194f95.jsonl NOT FOUND")
        logger.error("\n✗ LePaRD dataset not found in GCS")
        return False

    logger.info("  ✓ LePaRD JSONL found")

    try:
        lepard_files = gcs_list_files(pipeline_cfg.gcs_lepard_prefix, gcs_cfg)

        for f in lepard_files:
            try:
                info = gcs_get_info(f, gcs_cfg)
                size = info["size"]
                filename = f.split("/")[-1]

                if size > 1e6:
                    size_gb = size / 1e9
                    logger.info(f"    {filename}  {size_gb:.2f} GB")
                else:
                    logger.info(f"    {filename}  ({size} B)")
            except Exception:
                pass

        logger.info("\n✓ LePaRD ready for streaming from GCS")
        return True

    except Exception as e:
        logger.error(f"  ✗ Error listing LePaRD files: {e}")
        return False


def main() -> int:
    """Main entry point for GCS data verification."""
    parser = argparse.ArgumentParser(
        description="Verify required data files are present in GCS bucket"
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="GCS bucket name (without gs:// prefix)",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="GCP project ID",
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Check all data types (bulk CSVs, shards, LePaRD)",
    )
    parser.add_argument(
        "--check-bulk",
        action="store_true",
        help="Check CourtListener bulk CSVs only",
    )
    parser.add_argument(
        "--check-shards",
        action="store_true",
        help="Check processed shards only",
    )
    parser.add_argument(
        "--check-lepard",
        action="store_true",
        help="Check LePaRD dataset only",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Skip Colab authentication (use if already authenticated)",
    )

    args = parser.parse_args()

    if not any([args.check_all, args.check_bulk, args.check_shards, args.check_lepard]):
        args.check_all = True

    logger = setup_logging(args.verbose)

    pipeline_cfg = PipelineConfig(
        use_gcs_streaming=True,
        gcs_bucket_name=args.bucket,
        gcs_project_id=args.project,
    )

    gcs_cfg = GCSConfig(
        bucket_name=args.bucket,
        project_id=args.project,
        use_gcs_streaming=True,
    )

    if not args.no_auth:
        try:
            logger.info("Authenticating with GCS...")
            authenticate_gcs_colab()
            logger.info("✓ Authentication successful\n")
        except ImportError:
            logger.warning("Not in Colab environment - skipping auth\n")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return 2

    results = []

    if args.check_all or args.check_bulk:
        results.append(verify_bulk_csvs(gcs_cfg, pipeline_cfg, logger))
        logger.info("")

    if args.check_all or args.check_shards:
        results.append(verify_shards(gcs_cfg, pipeline_cfg, logger))
        logger.info("")

    if args.check_all or args.check_lepard:
        results.append(verify_lepard(gcs_cfg, pipeline_cfg, logger))
        logger.info("")

    if all(results):
        logger.info("=" * 60)
        logger.info("✓ All verification checks passed")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("=" * 60)
        logger.error("✗ Some verification checks failed")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
