#!/bin/bash
# scripts/upload_data_to_gcs.sh
# Upload all datasets to GCS: CourtListener bulk, processed shards, and LePaRD
# Mirrors the structure from notebooks/Project_Group_#43_Submission_v01.ipynb
# Usage: GCS_BUCKET=bucket-name ./scripts/upload_data_to_gcs.sh

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
LOCAL_SHARDS="data/raw/cl_federal_appellate_bulk"

# Check if shards already exist in GCS
if gsutil -q stat "gs://$GCS_BUCKET/$GCS_SHARDS/manifest.json" 2>/dev/null; then
    echo "✓ Shards already in GCS"

    # Count shards
    SHARD_COUNT=$(gsutil ls "gs://$GCS_BUCKET/$GCS_SHARDS/*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$SHARD_COUNT" -gt 0 ]; then
        echo "  $SHARD_COUNT shards available for streaming"
    fi
elif [ -f "$LOCAL_SHARDS/manifest.json" ]; then
    # Shards exist locally, upload them
    echo "→ Uploading shards from $LOCAL_SHARDS..."
    gsutil -m rsync -r "$LOCAL_SHARDS/" "gs://$GCS_BUCKET/$GCS_SHARDS/"
    echo "  ✓ Shards uploaded"
else
    # Shards don't exist - need to run pipeline
    echo "⚠️  Shards not found in GCS or locally"
    echo ""
    echo "→ Running pipeline to generate shards..."
    echo "  This will:"
    echo "  1. Download CourtListener bulk CSVs (if not present)"
    echo "  2. Filter to federal appellate courts"
    echo "  3. Extract and shard opinions"
    echo "  4. Generate manifest with SHA-256 checksums"
    echo ""

    # Run the pipeline (unbuffered output for real-time progress)
    if $PYTHON -u -c "
import logging
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
    config = PipelineConfig()
    logger.info('Starting pipeline...')
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
        if [ -f "$LOCAL_SHARDS/manifest.json" ]; then
            echo "→ Uploading generated shards to GCS..."
            gsutil -m rsync -r "$LOCAL_SHARDS/" "gs://$GCS_BUCKET/$GCS_SHARDS/"
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
LOCAL_LEPARD="data/raw/lepard"

# Use scripts/ingest_lepard.py to ensure artifact is ready (follows notebook pattern)
echo "→ Ensuring LePaRD artifact is ready..."

# Step 1: Fast-path verify
if $PYTHON scripts/ingest_lepard.py --verify-only 2>/dev/null; then
    echo "  ✓ LePaRD artifact valid (fast-path)"
else
    # Step 2: Self-heal or re-ingest (with progress output)
    echo "  → Running ingest_lepard.py (self-heal/download)..."
    echo "     This may take several minutes for 4M training pairs..."
    echo ""
    # Use unbuffered Python and show progress
    if ! $PYTHON -u scripts/ingest_lepard.py; then
        echo "  ✗ LePaRD ingest failed"
        echo "     Skipping LePaRD upload"
    fi
fi

# Upload all files from data/raw/lepard to GCS
if [ -d "$LOCAL_LEPARD" ] && [ "$(ls -A $LOCAL_LEPARD 2>/dev/null)" ]; then
    echo ""
    echo "→ Uploading LePaRD to GCS..."

    # Use rsync to upload entire directory
    gsutil -m rsync -r "$LOCAL_LEPARD/" "gs://$GCS_BUCKET/$GCS_LEPARD/"

    echo "  ✓ LePaRD uploaded"
else
    echo "  ⚠️  $LOCAL_LEPARD is empty or not found"
fi

echo ""

# ============================================================
# 3. CourtListener Bulk CSVs (from S3)
# ============================================================
echo "========================================"
echo "3. CourtListener Bulk CSVs"
echo "========================================"

GCS_CL_BULK="cs1090b_cl_bulk"
TEMP_DIR="/tmp/courtlistener_bulk"

echo "→ Discovering latest bulk files on S3..."
$PYTHON -c "
import sys
import json
from src.s3_discovery import discover_latest_bulk_files
from src.config import PipelineConfig

try:
    config = PipelineConfig()
    latest = discover_latest_bulk_files(config)
    print(json.dumps(latest, indent=2))
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" > /tmp/latest_files.json

if [ $? -ne 0 ]; then
    echo "✗ Failed to discover files"
    exit 1
fi

# Parse discovered files
FILES=$($PYTHON -c "
import json
with open('/tmp/latest_files.json') as f:
    data = json.load(f)
    for label in ['courts', 'dockets', 'clusters', 'opinions']:
        if label in data:
            key = data[label]['key']
            filename = key.split('/')[-1]
            print(filename)
")

mkdir -p "$TEMP_DIR"

COUNT=0
TOTAL=$(echo "$FILES" | wc -l | tr -d ' ')

echo "Files to transfer: $TOTAL (parallel)"
echo ""

# Function to transfer a single file
transfer_file() {
    local file=$1
    local num=$2
    local S3_PATH="s3://com-courtlistener-storage/bulk-data/$file"
    local GCS_PATH="gs://$GCS_BUCKET/$GCS_CL_BULK/$file"
    local LOCAL_PATH="$TEMP_DIR/$file"

    echo "[$num/$TOTAL] $file"

    if gsutil -q stat "$GCS_PATH" 2>/dev/null; then
        echo "  ✓ Already in GCS, skipping"
        return 0
    fi

    echo "  → Downloading from S3..."
    # Show progress for large files (remove --quiet to show progress)
    if ! aws s3 cp "$S3_PATH" "$LOCAL_PATH" --no-sign-request; then
        echo "  ✗ Download failed"
        return 1
    fi

    echo "  → Uploading to GCS..."
    if ! gsutil -m cp "$LOCAL_PATH" "$GCS_PATH" 2>/dev/null; then
        echo "  ✗ Upload failed"
        rm -f "$LOCAL_PATH"
        return 1
    fi

    rm -f "$LOCAL_PATH"
    echo "  ✓ Done"
    return 0
}

export -f transfer_file
export GCS_BUCKET GCS_CL_BULK TEMP_DIR TOTAL

# Launch parallel transfers
echo "Starting parallel transfers..."
echo ""
PIDS=()
for file in $FILES; do
    COUNT=$((COUNT + 1))
    echo "  Launching transfer: $file"
    transfer_file "$file" "$COUNT" &
    PIDS+=($!)
done

echo ""
echo "Waiting for all transfers to complete..."
echo ""

# Wait for all transfers to complete
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -gt 0 ]; then
    echo "⚠️  $FAILED file(s) failed to transfer"
else
    echo "✓ All transfers completed successfully"
fi

rm -f /tmp/latest_files.json
rmdir "$TEMP_DIR" 2>/dev/null || true

echo ""
echo "✓ CourtListener bulk CSVs uploaded"
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
