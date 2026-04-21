#!/bin/bash
# scripts/upload_data_to_gcs.sh
# Upload all datasets to GCS: CourtListener bulk, processed shards, and LePaRD
# Mirrors the structure from notebooks/Project_Group_#43_Submission_v01.ipynb
# Usage: GCS_BUCKET=bucket-name ./scripts/upload_data_to_gcs.sh
#
# Resilience strategy for Colab (runtime disconnects mid-transfer):
#   - Completed local downloads are uploaded to GCS directly (no re-download)
#   - Missing files are downloaded via src.bulk_download (aws CLI → requests
#     fallback), so AWS CLI is NOT required (Colab ships without it)
#   - Per-file failures are logged but never abort the run, so any file already
#     in GCS or completed on disk still reaches GCS

set -e

# Use .venv Python
PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Virtual environment not found at .venv/"
    echo "Run: uv sync"
    exit 1
fi

# Check environment variables
if [ -z "$GCS_BUCKET" ]; then
    echo "ERROR: GCS_BUCKET environment variable not set"
    echo ""
    exit 1
fi

echo "========================================"
echo "Upload All Datasets to GCS"
echo "========================================"
echo "GCS Bucket: gs://$GCS_BUCKET/"
echo ""

# Test bucket write access
echo "→ Testing bucket write access..."
TEST_FILE="/tmp/gcs_write_test_$$.txt"
echo "test" > "$TEST_FILE"

if gsutil cp "$TEST_FILE" "gs://$GCS_BUCKET/.write_test" 2>/dev/null; then
    gsutil rm "gs://$GCS_BUCKET/.write_test" 2>/dev/null
    rm -f "$TEST_FILE"
    echo "  ✓ Bucket is writable"
else
    rm -f "$TEST_FILE"
    echo ""
    echo "ERROR: Cannot write to gs://$GCS_BUCKET/"
    echo "Check:"
    echo "  1. Bucket exists: gsutil ls gs://$GCS_BUCKET/"
    echo "  2. You have write permissions"
    echo "  3. You're authenticated: gcloud auth list"
    exit 1
fi

echo ""
echo "Datasets:"
echo "  1. Processed federal appellate shards"
echo "  2. LePaRD training data"
echo "  3. CourtListener bulk CSVs"
echo ""

# ============================================================
# 1. Processed Federal Appellate Shards
# ============================================================
echo "========================================"
echo "1. Processed Federal Appellate Shards"
echo "========================================"

GCS_SHARDS="cs1090b_cl_federal_appellate_bulk"
STAGING_SHARDS="/content/drive/MyDrive/cs1090b_shards_staging"

echo "→ Staging dir: $STAGING_SHARDS"
echo "→ Target:      gs://$GCS_BUCKET/$GCS_SHARDS/"
echo ""

# Check if shards already exist in GCS
if gsutil -q stat "gs://$GCS_BUCKET/$GCS_SHARDS/manifest.json" 2>/dev/null; then
    echo "✓ Shards already in GCS"

    # Count shards
    SHARD_COUNT=$(gsutil ls "gs://$GCS_BUCKET/$GCS_SHARDS/*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$SHARD_COUNT" -gt 0 ]; then
        echo "  $SHARD_COUNT shards available for streaming"
    fi
elif [ -f "$STAGING_SHARDS/manifest.json" ]; then
    # Shards exist in Drive staging, upload them
    echo "→ Uploading shards from Drive staging..."
    gsutil -m rsync -r "$STAGING_SHARDS/" "gs://$GCS_BUCKET/$GCS_SHARDS/"
    echo "  ✓ Shards uploaded"
else
    # Shards don't exist - need to run pipeline
    echo "⚠️  Shards not found in GCS or Drive staging"
    echo ""
    echo "→ Running pipeline to generate shards..."
    echo "  This will:"
    echo "  1. Download CourtListener bulk CSVs (if not present)"
    echo "  2. Filter to federal appellate courts"
    echo "  3. Extract and shard opinions"
    echo "  4. Generate manifest with SHA-256 checksums"
    echo "  5. Stage to Drive: $STAGING_SHARDS"
    echo ""

    # Create staging directory
    mkdir -p "$STAGING_SHARDS"

    # Run the pipeline (unbuffered output for real-time progress)
    if PIPELINE_OUTPUT_DIR="$STAGING_SHARDS" $PYTHON -u -c "
import logging
import os
import sys
from src.pipeline import run_pipeline
from src.config import PipelineConfig

# Unbuffered logging for real-time output
logging.basicConfig(
    level=logging.INFO,
    format='  %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

try:
    # Override output directory to use Drive staging
    output_dir = os.environ.get('PIPELINE_OUTPUT_DIR', 'data/raw/cl_federal_appellate_bulk')
    config = PipelineConfig()
    config.output_dir = output_dir
    logger.info(f'Starting pipeline (output: {output_dir})...')
    manifest = run_pipeline(config=config, logger=logger)
    logger.info(f'✓ Pipeline complete: {manifest[\"num_cases\"]:,} cases, {manifest[\"num_shards\"]} shards')
    sys.exit(0)
except Exception as e:
    logger.error(f'✗ Pipeline failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"; then
        echo ""
        echo "  ✓ Pipeline completed successfully"
        echo ""

        # Upload generated shards to GCS
        if [ -f "$STAGING_SHARDS/manifest.json" ]; then
            echo "→ Uploading generated shards to GCS..."
            gsutil -m rsync -r "$STAGING_SHARDS/" "gs://$GCS_BUCKET/$GCS_SHARDS/"
            echo "  ✓ Shards uploaded"
        else
            echo "  ✗ Pipeline completed but manifest not found"
        fi
    else
        echo "  ✗ Pipeline failed"
        echo "     Check logs and try again"
    fi
fi

echo ""

# ============================================================
# 2. LePaRD Training Data
# ============================================================
echo "========================================"
echo "2. LePaRD Training Data"
echo "========================================"

GCS_LEPARD="cs1090b_lepard"
STAGING_LEPARD="/content/drive/MyDrive/cs1090b_lepard_staging"

echo "→ Staging dir: $STAGING_LEPARD"
echo "→ Target:      gs://$GCS_BUCKET/$GCS_LEPARD/"
echo ""

# Check if already in GCS
if gsutil -q stat "gs://$GCS_BUCKET/$GCS_LEPARD/lepard_train_4000000_rev0194f95.jsonl" 2>/dev/null; then
    echo "✓ LePaRD already in GCS"
elif [ -d "$STAGING_LEPARD" ] && [ "$(ls -A $STAGING_LEPARD 2>/dev/null)" ]; then
    # LePaRD exists in Drive staging, upload it
    echo "→ Uploading LePaRD from Drive staging..."
    gsutil -m rsync -r "$STAGING_LEPARD/" "gs://$GCS_BUCKET/$GCS_LEPARD/"
    echo "  ✓ LePaRD uploaded"
else
    # Need to download LePaRD to Drive staging
    echo "⚠️  LePaRD not found in GCS or Drive staging"
    echo ""
    echo "→ Downloading LePaRD to Drive staging..."
    echo "  This may take several minutes for 4M training pairs..."
    echo ""

    # Create staging directory
    mkdir -p "$STAGING_LEPARD"

    # Use scripts/ingest_lepard.py with Drive staging directory
    if LEPARD_OUTPUT_DIR="$STAGING_LEPARD" $PYTHON -u scripts/ingest_lepard.py; then
        echo ""
        echo "  ✓ LePaRD download complete"
        echo ""
        echo "→ Uploading LePaRD to GCS..."
        gsutil -m rsync -r "$STAGING_LEPARD/" "gs://$GCS_BUCKET/$GCS_LEPARD/"
        echo "  ✓ LePaRD uploaded"
    else
        echo "  ✗ LePaRD ingest failed"
        echo "     Skipping LePaRD upload"
    fi
fi

echo ""

# ============================================================
# 3. CourtListener Bulk CSVs (from S3)
# ============================================================
# Hybrid Colab-resilient transfer (inlined, no external helper script):
#   - Pure-Python HTTPS download with Range-based resume (no aws CLI)
#   - Per-file: skip-if-in-GCS → resumable download → immediate upload → cleanup
#   - Staging dir persists across runtime disconnects (defaults to Google Drive
#     when $CL_BULK_STAGING_DIR is unset and /content/drive is mounted)
echo "========================================"
echo "3. CourtListener Bulk CSVs"
echo "========================================"

export GCS_BUCKET
export GCS_CL_BULK_PREFIX="${GCS_CL_BULK_PREFIX:-cs1090b_cl_bulk}"

export CL_BULK_STAGING_DIR="/content/drive/MyDrive/cs1090b_cl_bulk_staging"

echo "→ Staging dir: $CL_BULK_STAGING_DIR"
echo "→ Target:      gs://$GCS_BUCKET/$GCS_CL_BULK_PREFIX/"
echo ""

# Per-file loop: skip if already in GCS, else download into the Drive staging
# dir via src.bulk_download (idempotent — skips files already on disk), then
# upload to GCS. Drive-persistent files survive Colab runtime disconnects.
set +e
$PYTHON -u - <<'PYEOF'
import logging, os, subprocess, sys
from pathlib import Path

from src.bulk_download import download_file
from src.config import PipelineConfig
from src.s3_discovery import discover_latest_bulk_files

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("cl_bulk")

bucket  = os.environ["GCS_BUCKET"]
prefix  = os.environ.get("GCS_CL_BULK_PREFIX", "cs1090b_cl_bulk").rstrip("/")
staging = Path(os.environ["CL_BULK_STAGING_DIR"]).expanduser()
staging.mkdir(parents=True, exist_ok=True)

config = PipelineConfig()
latest = discover_latest_bulk_files(config)

failed = []
for label in ("courts", "dockets", "clusters", "opinions"):
    if label not in latest:
        continue
    key      = str(latest[label]["key"])
    filename = Path(key).name
    uri      = f"gs://{bucket}/{prefix}/{filename}"
    dest     = staging / filename

    log.info("\n[%s] %s", label, filename)

    # 1. Skip if already in GCS.
    if subprocess.run(["gsutil", "-q", "stat", uri]).returncode == 0:
        log.info("  ✓ already in GCS, skipping")
        continue

    # 2. Download directly into Drive (no-op if file is already there).
    try:
        download_file(key, dest, config=config, logger=log)
    except Exception as e:
        log.error("  ✗ download failed: %s", e)
        failed.append(filename)
        continue

    # 3. Upload Drive → GCS.
    log.info("  → uploading to %s", uri)
    if subprocess.run(["gsutil", "-m", "cp", str(dest), uri]).returncode != 0:
        log.error("  ✗ upload failed")
        failed.append(filename)
        continue
    log.info("  ✓ %s now in GCS", filename)

if failed:
    log.error("\n%d file(s) failed: %s", len(failed), ", ".join(failed))
    sys.exit(1)
PYEOF
CL_BULK_RC=$?
set -e

if [ $CL_BULK_RC -ne 0 ]; then
    echo ""
    echo "⚠️  One or more CourtListener bulk files failed to transfer (exit $CL_BULK_RC)"
    echo "     Re-run this script to resume from the last completed byte"
else
    echo ""
    echo "✓ All CourtListener bulk transfers completed successfully"
fi

# Expose the prefix under the old var name so the summary below keeps working.
GCS_CL_BULK="$GCS_CL_BULK_PREFIX"

echo ""

# ============================================================
# Summary
# ============================================================
echo "========================================"
echo "✓ Upload Complete"
echo "========================================"
echo ""
echo "GCS Structure:"
echo "  gs://$GCS_BUCKET/$GCS_SHARDS/"
echo "  gs://$GCS_BUCKET/$GCS_LEPARD/"
echo "  gs://$GCS_BUCKET/$GCS_CL_BULK/"
echo ""
