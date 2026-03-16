#!/usr/bin/env bash
# setup.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/setup.sh
#
# Usage:        bash setup.sh
# Debug:        DEBUG=1 bash setup.sh
# Skip GPU:     SKIP_GPU=1 bash setup.sh
# Dry run:      DRY_RUN=1 bash setup.sh
set -euo pipefail
[ "${DEBUG:-0}" = "1" ] && set -x

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

trap 'echo "ERROR: setup.sh failed at line $LINENO — environment may be incomplete"' ERR

check_uv() {
    if ! command -v uv &>/dev/null && ! command -v ~/.local/bin/uv &>/dev/null; then
        echo "ERROR: uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    UV=$(command -v uv 2>/dev/null || echo ~/.local/bin/uv)
    echo " uv: $($UV --version)"
}

check_lockfile() {
    if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
        echo "ERROR: pyproject.toml not found — are you in the project root?"
        exit 1
    fi
    if [ ! -f "$PROJECT_ROOT/uv.lock" ]; then
        echo "ERROR: uv.lock not found — lockfile must be committed to the repo."
        echo "       To generate deliberately: uv lock && git add uv.lock && git commit -m 'chore: pin uv.lock'"
        exit 1
    fi
    echo " uv.lock sha256: $(sha256sum "$PROJECT_ROOT/uv.lock" | cut -d' ' -f1)"
}

log_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        echo " GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        echo " VRAM:         $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
        echo " Driver:       $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        echo " CUDA runtime: $(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}')"
    else
        echo "WARNING: nvidia-smi not found — GPU training unavailable"
    fi
    if command -v nvcc &>/dev/null; then
        echo " CUDA toolkit: $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')"
    else
        echo "WARNING: nvcc not on PATH — CUDA toolkit version unverified"
    fi
    $PYTHON -c "
import torch
if torch.cuda.is_available():
    print(f'  torch CUDA:  {torch.version.cuda}')
    print(f'  cuDNN:       {torch.backends.cudnn.version()}')
    print(f'  Compute cap: {torch.cuda.get_device_capability()}')
" 2>/dev/null || true
}

ensure_venv() {
    if [ -f "$PYTHON" ] && $PYTHON -c "import sys; sys.exit(0 if sys.version_info[:3] == (3,11,9) else 1)" 2>/dev/null; then
        echo " .venv already exists with Python 3.11.9 — skipping creation"
        return
    fi

    if [ -d "$PROJECT_ROOT/.venv" ]; then
        echo "WARNING: .venv exists but is NOT Python 3.11.9 — it will be removed."
        echo "         Contents: $(du -sh "$PROJECT_ROOT/.venv" 2>/dev/null | cut -f1) on disk"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "DRY_RUN=1 — skipping .venv removal. Exiting."
            exit 0
        fi
        echo "         Aborting in 5 seconds — press Ctrl+C to cancel..."
        sleep 5
        echo " Removing stale .venv..."
        rm -rf "$PROJECT_ROOT/.venv"
    fi

    echo " Creating .venv with Python 3.11.9..."
    $UV venv .venv --python 3.11.9 --seed
}

verify_python() {
    $PYTHON -c "import sys; assert sys.version_info[:3] == (3,11,9), f'Expected 3.11.9 got {sys.version}'"
    echo " Python: $($PYTHON --version)"
    echo " Executable: $($PYTHON -c 'import sys; print(sys.executable)')"
}

sync_dependencies() {
    echo " Syncing dependencies from uv.lock (--frozen)..."
    $UV sync --frozen --dev
}

run_env_smoke_tests() {
    echo " Running environment smoke tests..."
    $PYTHON -c "
import torch
ver = torch.__version__
assert ver.startswith('2.') and 'cu' in ver, f'Expected torch 2.x+cuXXX got {ver}'
print(f'  torch {ver}')
"
    $PYTHON -c "import transformers; print(f'  transformers {transformers.__version__}')"
    $PYTHON -c "import faiss; print(f'  faiss ok')"
    $PYTHON -c "import spacy; print(f'  spacy {spacy.__version__}')"
}

run_gpu_smoke_tests() {
    if [ "${SKIP_GPU:-0}" = "1" ]; then
        echo " SKIP_GPU=1 — skipping GPU smoke tests (CPU-only mode)"
        return
    fi
    echo " Running GPU smoke tests..."
    $PYTHON -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — wrong torch wheel'
assert torch.version.cuda.startswith('11.7'), f'Expected CUDA 11.7 got {torch.version.cuda}'
cap = torch.cuda.get_device_capability()
assert cap >= (8, 0), f'L4 requires compute capability >=8.0, got {cap}'
print(f'  CUDA {torch.version.cuda} ok — compute capability {cap}')
"
}

download_nlp_models() {
    echo " Downloading spaCy en_core_web_sm..."
    $PYTHON -m spacy download en_core_web_sm --quiet
}

write_manifest() {
    echo " Writing environment manifest..."
    mkdir -p "$PROJECT_ROOT/logs"
    $PYTHON -c "
import json, torch, transformers, spacy, sys, platform, subprocess
from datetime import datetime

def get_nvcc_version():
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if 'release' in line:
                return line.strip()
    except FileNotFoundError:
        return 'nvcc not found'
    return 'unknown'

manifest = {
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'python': sys.version,
    'torch': torch.__version__,
    'cuda_runtime': torch.version.cuda,
    'cudnn': str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
    'cuda_toolkit_nvcc': get_nvcc_version(),
    'cuda_available': torch.cuda.is_available(),
    'compute_capability': torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'vram_gb': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else None,
    'transformers': transformers.__version__,
    'spacy': spacy.__version__,
    'platform': platform.platform(),
}
with open('logs/environment_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print('  manifest written to logs/environment_manifest.json')
"
}

register_kernel() {
    echo " Registering Jupyter kernel..."
    $PYTHON -m ipykernel install --user --name hallucination-legal-rag --display-name "HallucinationLegalRAG (3.11.9)"
    $PYTHON -c "
import subprocess, json
result = subprocess.run(['jupyter', 'kernelspec', 'list', '--json'], capture_output=True, text=True)
kernels = json.loads(result.stdout).get('kernelspecs', {})
assert 'hallucination-legal-rag' in kernels, 'Kernel registration failed'
print('  kernel verified in kernelspec list')
" || echo "WARNING: Could not verify kernel — jupyter may not be on PATH"
}

verify_tests() {
    echo " Verifying pytest test collection..."
    $UV run pytest --co -q || echo "WARNING: No tests collected yet"
}

echo "============================================================"
echo " cs1090b_HallucinationLegalRAGChatbots — Environment Bootstrap"
echo "============================================================"
check_uv
check_lockfile
log_gpu
ensure_venv
verify_python
sync_dependencies
run_env_smoke_tests
run_gpu_smoke_tests
download_nlp_models
write_manifest
register_kernel
verify_tests
echo "============================================================"
echo " Environment ready."
echo " Activate:  source .venv/bin/activate"
echo " Kernel:    HallucinationLegalRAG (3.11.9)"
echo " Manifest:  logs/environment_manifest.json"
echo " CPU mode:  SKIP_GPU=1 bash setup.sh"
echo " Dry run:   DRY_RUN=1 bash setup.sh"
echo "============================================================"
