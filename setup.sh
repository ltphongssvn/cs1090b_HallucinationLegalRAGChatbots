#!/usr/bin/env bash
# setup.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/setup.sh
#
# Usage:        bash setup.sh
# Debug:        DEBUG=1 bash setup.sh
# Skip GPU:     SKIP_GPU=1 bash setup.sh
# Dry run:      DRY_RUN=1 bash setup.sh
# Offline:      OFFLINE=1 bash setup.sh  (requires cached wheel in .cache/spacy/)
#
# Hardware target: 4x NVIDIA L4 (23034MiB each, compute cap 8.9, CUDA driver 12.8)
# Torch wheel:     2.0.1+cu117 (compiled against CUDA runtime 11.7 — compatible via driver forward-compat)
set -euo pipefail
[ "${DEBUG:-0}" = "1" ] && set -x

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Pinned spaCy model — update deliberately and commit when upgrading
SPACY_MODEL="en_core_web_sm"
SPACY_MODEL_VERSION="3.8.0"
SPACY_MODEL_URL="https://github.com/explosion/spacy-models/releases/download/${SPACY_MODEL}-${SPACY_MODEL_VERSION}/${SPACY_MODEL}-${SPACY_MODEL_VERSION}-py3-none-any.whl"
SPACY_MODEL_SHA256="5e97b9ec4f95153b992896c5c45b1a00c3fcde7f764426c5370f2f11e71abef2"
SPACY_CACHE_DIR="$PROJECT_ROOT/.cache/spacy"
SPACY_WHEEL="$SPACY_CACHE_DIR/${SPACY_MODEL}-${SPACY_MODEL_VERSION}-py3-none-any.whl"

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
        echo " --- nvidia-smi per-GPU summary ---"
        nvidia-smi --query-gpu=index,name,memory.total,driver_version \
            --format=csv,noheader | while IFS=',' read -r idx name mem drv; do
            echo "  GPU $idx:$(echo "$name" | xargs) | VRAM:$(echo "$mem" | xargs) | Driver:$(echo "$drv" | xargs)"
        done
        DRIVER_CUDA=$(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}')
        echo " Driver CUDA (max supported): $DRIVER_CUDA"
        echo " NOTE: torch wheel is compiled against CUDA runtime 11.7 (cu117)."
        echo "       Driver CUDA $DRIVER_CUDA is forward-compatible — this is expected and correct."
    else
        echo "WARNING: nvidia-smi not found — GPU training unavailable"
    fi
    if command -v nvcc &>/dev/null; then
        echo " CUDA toolkit (nvcc): $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')"
    else
        echo "WARNING: nvcc not on PATH — CUDA toolkit version unverified"
    fi
    if [ -f "$PYTHON" ]; then
        $PYTHON -c "
import torch
if torch.cuda.is_available():
    print(f'  torch CUDA runtime: {torch.version.cuda}')
    print(f'  cuDNN:              {torch.backends.cudnn.version()}')
    print(f'  visible GPUs:       {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        vram = round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)
        print(f'    [{i}] {name} | cap {cap} | {vram} GB')
" 2>/dev/null || true
    fi
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
    # --dev is the default in uv >=0.4 but kept explicit here to:
    # (1) document intent — dev tools (pytest, mypy, hypothesis) are required,
    # (2) guard against future uv default changes,
    # (3) signal to collaborators that this is a deliberate inclusion.
    $UV sync --frozen --dev
}

download_nlp_models() {
    echo " Installing spaCy ${SPACY_MODEL} ${SPACY_MODEL_VERSION} (pinned)..."
    mkdir -p "$SPACY_CACHE_DIR"

    # Check if model version already installed — skip if so
    if $PYTHON -c "
import spacy, sys
try:
    nlp = spacy.load('${SPACY_MODEL}')
    meta = nlp.meta
    if meta.get('version') == '${SPACY_MODEL_VERSION}':
        sys.exit(0)
    else:
        print(f'  installed version {meta.get(\"version\")} != pinned ${SPACY_MODEL_VERSION} — reinstalling')
        sys.exit(1)
except OSError:
    sys.exit(1)
" 2>/dev/null; then
        echo " ${SPACY_MODEL} ${SPACY_MODEL_VERSION} already installed — skipping"
        return
    fi

    # Obtain wheel: use cache if present, else download (unless OFFLINE=1)
    if [ ! -f "$SPACY_WHEEL" ]; then
        if [ "${OFFLINE:-0}" = "1" ]; then
            echo "ERROR: OFFLINE=1 but cached wheel not found at $SPACY_WHEEL"
            echo "       To cache: mkdir -p .cache/spacy && wget -O $SPACY_WHEEL $SPACY_MODEL_URL"
            exit 1
        fi
        echo " Downloading ${SPACY_MODEL} wheel..."
        curl -fsSL -o "$SPACY_WHEEL" "$SPACY_MODEL_URL"
    else
        echo " Using cached wheel: $SPACY_WHEEL"
    fi

    # Checksum verification — always, regardless of source
    ACTUAL_SHA=$(sha256sum "$SPACY_WHEEL" | cut -d' ' -f1)
    if [ "$ACTUAL_SHA" != "$SPACY_MODEL_SHA256" ]; then
        echo "ERROR: spaCy model wheel checksum mismatch!"
        echo "       expected: $SPACY_MODEL_SHA256"
        echo "       actual:   $ACTUAL_SHA"
        echo "       Deleting corrupted wheel: $SPACY_WHEEL"
        rm -f "$SPACY_WHEEL"
        exit 1
    fi
    echo " Checksum verified: $ACTUAL_SHA"

    # Install from verified local wheel
    $PYTHON -m pip install --quiet "$SPACY_WHEEL"
    echo " ${SPACY_MODEL} ${SPACY_MODEL_VERSION} installed from pinned wheel"
}

run_env_smoke_tests() {
    echo " Running environment smoke tests..."

    # torch: version check + functional tensor op on CPU
    $PYTHON -c "
import torch
ver = torch.__version__
assert ver.startswith('2.') and 'cu' in ver, f'Expected torch 2.x+cuXXX got {ver}'
t = torch.tensor([1.0, 2.0, 3.0])
assert torch.allclose(t.mean(), torch.tensor(2.0)), 'torch tensor op failed'
print(f'  torch {ver} — tensor op ok')
"

    # transformers: version check + tokenizer instantiation (no model download)
    $PYTHON -c "
import transformers
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
ids = tok('hello world', return_tensors='pt')
assert ids['input_ids'].shape[1] > 0, 'tokenizer produced empty output'
print(f'  transformers {transformers.__version__} — tokenizer ok')
"

    # faiss: import + functional flat index add/search
    $PYTHON -c "
import faiss, numpy as np
dim = 64
index = faiss.IndexFlatL2(dim)
vecs = np.random.rand(10, dim).astype('float32')
index.add(vecs)
assert index.ntotal == 10, 'faiss index add failed'
D, I = index.search(vecs[:1], 3)
assert I.shape == (1, 3), 'faiss search returned wrong shape'
print(f'  faiss ok — index add/search functional')
"

    # spacy: version check + model load + pipeline run
    $PYTHON -c "
import spacy
nlp = spacy.load('${SPACY_MODEL}')
assert nlp.meta.get('version') == '${SPACY_MODEL_VERSION}', \
    f'spaCy model version mismatch: expected ${SPACY_MODEL_VERSION} got {nlp.meta.get(\"version\")}'
doc = nlp('The Supreme Court ruled in favor of the plaintiff.')
assert len(doc) > 0, 'spacy pipeline produced empty doc'
ents = [ent.label_ for ent in doc.ents]
print(f'  spacy {spacy.__version__} | model ${SPACY_MODEL_VERSION} — pipeline ok, entities: {ents}')
"
}

run_gpu_smoke_tests() {
    if [ "${SKIP_GPU:-0}" = "1" ]; then
        echo " SKIP_GPU=1 — skipping GPU smoke tests (CPU-only mode)"
        return
    fi
    echo " Running GPU smoke tests (target: 4x NVIDIA L4, cu117 wheel, driver CUDA 12.8)..."
    $PYTHON -c "
import torch

# 1. CUDA must be available
assert torch.cuda.is_available(), 'CUDA not available — wrong torch wheel'

# 2. torch runtime must be cu117 — this is the pinned wheel for this project
assert torch.version.cuda.startswith('11.7'), (
    f'Expected torch CUDA runtime 11.7 (cu117 wheel) got {torch.version.cuda}. '
    f'NOTE: driver CUDA (12.8) != torch runtime CUDA (11.7) — this is expected.'
)

# 3. Must see all 4 L4 GPUs
n = torch.cuda.device_count()
assert n >= 4, f'Expected 4x NVIDIA L4 GPUs, got {n}'

# 4. Every visible GPU must be NVIDIA L4 with compute cap 8.9 and >=22GB VRAM
for i in range(n):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
    assert 'L4' in name, f'GPU {i}: expected NVIDIA L4, got {name}'
    assert cap >= (8, 9), f'GPU {i}: expected compute cap (8,9) for L4, got {cap}'
    assert vram_gb >= 22.0, f'GPU {i}: expected >=22GB VRAM, got {vram_gb:.1f}GB'
    print(f'  GPU [{i}] {name} | cap {cap} | {vram_gb:.1f} GB — ok')

# 5. Functional: allocate a tensor on GPU 0 and verify round-trip
t = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
assert t.device.type == 'cuda', 'tensor not on CUDA device'
assert torch.allclose(t.mean().cpu(), torch.tensor(2.0)), 'GPU tensor op failed'
print(f'  CUDA {torch.version.cuda} — GPU tensor op on cuda:0 ok')
"
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

def get_driver_cuda():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if 'CUDA Version' in line:
                return line.strip().split()[-1]
    except FileNotFoundError:
        return 'nvidia-smi not found'
    return 'unknown'

gpus = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            'index': i,
            'name': props.name,
            'vram_gb': round(props.total_memory / 1e9, 2),
            'compute_capability': list(torch.cuda.get_device_capability(i)),
        })

nlp = spacy.load('${SPACY_MODEL}')
manifest = {
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'python': sys.version,
    'torch': torch.__version__,
    'torch_cuda_runtime': torch.version.cuda,
    'driver_cuda': get_driver_cuda(),
    'cudnn': str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
    'cuda_toolkit_nvcc': get_nvcc_version(),
    'cuda_available': torch.cuda.is_available(),
    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'gpus': gpus,
    'transformers': transformers.__version__,
    'spacy': spacy.__version__,
    'spacy_model': '${SPACY_MODEL}',
    'spacy_model_version': nlp.meta.get('version'),
    'spacy_model_sha256': '${SPACY_MODEL_SHA256}',
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
echo " Target: 4x NVIDIA L4 | Python 3.11.9 | torch 2.0.1+cu117"
echo " Driver CUDA: 12.8 (forward-compat) | torch runtime: 11.7"
echo " spaCy model: ${SPACY_MODEL} ${SPACY_MODEL_VERSION} (pinned + checksummed)"
echo " Offline mode: OFFLINE=1 bash setup.sh (requires .cache/spacy/ wheel)"
echo "============================================================"
check_uv
check_lockfile
log_gpu
ensure_venv
verify_python
sync_dependencies
# Note: download_nlp_models must run before run_env_smoke_tests
# because spacy smoke test requires en_core_web_sm to be installed
download_nlp_models
run_env_smoke_tests
run_gpu_smoke_tests
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
echo " Offline:   OFFLINE=1 bash setup.sh"
echo "============================================================"
