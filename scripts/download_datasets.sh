#!/usr/bin/env bash
# scripts/download_datasets.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/download_datasets.sh
# Responsibility: download and verify legal datasets from Hugging Face Hub.
# Run after bash setup.sh: bash scripts/download_datasets.sh
# Or as a setup step:      STEP=download_datasets bash setup.sh
#
# Datasets:
#   freelaw/opinions          — CourtListener federal appellate opinions (RAG corpus)
#   nguyen-brat/legal-ner     — Legal NER for entity extraction validation
#   pile-of-law/pile-of-law   — Multi-source legal text for pretraining context
#
# Modes:
#   NO_DOWNLOAD=1             — skip all downloads (idempotent re-runs)
#   OFFLINE=1                 — fail if cache absent
#   HF_DATASETS_CACHE         — override HF cache dir (default: ~/.cache/huggingface)
#
# Cache: all datasets cached in HF_DATASETS_CACHE — subsequent runs are no-ops.

# Sourced by setup.sh if called via STEP= — otherwise run standalone.
# When run standalone, source lib.sh for messaging helpers.
if [ -z "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    source "$PROJECT_ROOT/scripts/lib.sh"
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
fi

# Dataset registry — add/remove entries here to control what is downloaded.
# Format: "hf_dataset_id|config_or_NA|split|description"
LEGAL_DATASETS=(
    "freelaw/opinions|NA|train|CourtListener federal appellate opinions (RAG corpus)"
    "nguyen-brat/legal-ner|NA|train|Legal NER — entity extraction validation"
    "pile-of-law/pile-of-law|r_courtlistener_opinions|train|Pile of Law — CourtListener subset"
)

# Minimum expected sizes (bytes) — guards against truncated downloads
declare -A DATASET_MIN_SIZES=(
    ["freelaw/opinions"]=1000000
    ["nguyen-brat/legal-ner"]=10000
    ["pile-of-law/pile-of-law"]=100000
)

_hf_dataset_cached() {
    # Returns 0 if dataset is already in HF cache, 1 otherwise
    local dataset_id="$1"
    _require_python
    $PYTHON -c "
import sys
try:
    from datasets import load_dataset
    ds = load_dataset('${dataset_id}', trust_remote_code=False, download_mode='reuse_cache_if_exists')
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

_download_single_dataset() {
    local dataset_id="$1" config="$2" split="$3" description="$4"
    _require_python

    _msg_info "Dataset: ${dataset_id} (${description})"

    # Idempotent: skip if already cached
    if _hf_dataset_cached "$dataset_id"; then
        _msg_ok "${dataset_id} — already cached, skipping download"
        return 0
    fi

    if [ "${OFFLINE:-0}" = "1" ]; then
        _msg_error "Offline mode: dataset not cached" \
            "${dataset_id} not found in HF cache" \
            "Cannot download in OFFLINE=1 mode" \
            "Pre-cache by running: bash scripts/download_datasets.sh   then retry with OFFLINE=1"
        exit 1
    fi

    _msg_info "Downloading ${dataset_id} (split=${split})..."
    $PYTHON -c "
import sys
from datasets import load_dataset
from tqdm.auto import tqdm

dataset_id = '${dataset_id}'
config     = None if '${config}' == 'NA' else '${config}'
split      = '${split}'

try:
    ds = load_dataset(
        dataset_id,
        config,
        split=split,
        trust_remote_code=False,
        download_mode='reuse_cache_if_exists',
    )
    n = len(ds) if hasattr(ds, '__len__') else '?'
    cols = list(ds.features.keys()) if hasattr(ds, 'features') else []
    print(f'  \033[0;32m✓\033[0m {dataset_id}: {n} rows | columns: {cols}')
except Exception as e:
    print(f'\033[0;31m  ✗ Failed to download {dataset_id}: {e}\033[0m')
    print(f'    Fix: check HF_DATASETS_CACHE permissions and network connectivity')
    sys.exit(1)
" || {
        _msg_error "Dataset download failed" \
            "${dataset_id} download exited non-zero" \
            "RAG pipeline requires this corpus — pipeline cells will fail without it" \
            "Retry: bash scripts/download_datasets.sh   |   check network connectivity"
        exit 1
    }
}

_verify_dataset_cache() {
    # Verify cache integrity — ensures no truncated downloads
    _require_python
    _msg_info "Verifying dataset cache integrity..."
    $PYTHON -c "
import os, sys
from pathlib import Path

cache_dir = Path(os.environ.get('HF_DATASETS_CACHE',
    Path.home() / '.cache' / 'huggingface' / 'datasets'))

if not cache_dir.exists():
    print(f'  \033[0;33m⚠\033[0m Cache dir not found: {cache_dir}')
    sys.exit(0)

total_gb = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / 1e9
dataset_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
print(f'  \033[0;32m✓\033[0m HF cache: {cache_dir}')
print(f'    Datasets: {len(dataset_dirs)} | Total size: {total_gb:.2f}GB')
for d in dataset_dirs:
    size_mb = sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / 1e6
    print(f'    • {d.name}: {size_mb:.1f}MB')
"
}

download_datasets() {
    echo " Downloading legal datasets from Hugging Face Hub..."

    if [ "${NO_DOWNLOAD:-0}" = "1" ]; then
        _msg_skip "NO_DOWNLOAD=1 — skipping dataset downloads"
        _msg_info "To download: bash scripts/download_datasets.sh"
        return 0
    fi

    if _is_dry_run; then
        _msg_dry_run "download legal datasets" "HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
        for entry in "${LEGAL_DATASETS[@]}"; do
            IFS='|' read -r dataset_id config split description <<< "$entry"
            _msg_dry_run "download" "${dataset_id} (split=${split}) — ${description}"
        done
        step_end "download_datasets" "DRY"; return
    fi

    _require_python

    # Log HF cache location
    local hf_cache="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
    _msg_info "HF_DATASETS_CACHE: ${hf_cache}"

    # Download each dataset
    local failed=0
    for entry in "${LEGAL_DATASETS[@]}"; do
        IFS='|' read -r dataset_id config split description <<< "$entry"
        _download_single_dataset "$dataset_id" "$config" "$split" "$description" || \
            failed=$(( failed + 1 ))
    done

    if [ "$failed" -gt 0 ]; then
        _msg_error "Dataset downloads incomplete" \
            "${failed} dataset(s) failed to download" \
            "RAG pipeline cells require all datasets to be present" \
            "Retry: bash scripts/download_datasets.sh"
        exit 1
    fi

    _verify_dataset_cache
    _msg_ok "All legal datasets downloaded and verified"
}

# ===========================================================================
# Standalone execution (when not sourced by setup.sh)
# ===========================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    [ "${DEBUG:-0}" = "1" ] && set -x
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s)
    _step_start_time=$(date +%s)
    _CURRENT_STEP="download_datasets"
    download_datasets
    _msg_ok "Dataset download complete. Run: jupyter lab"
fi
