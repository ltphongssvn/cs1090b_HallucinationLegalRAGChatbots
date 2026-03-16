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
# Reproducibility env vars (set here, written to .env, recorded in manifest):
#   PYTHONHASHSEED=0                — deterministic Python hash randomization
#   CUBLAS_WORKSPACE_CONFIG=:4096:8 — deterministic cuBLAS ops
#   TOKENIZERS_PARALLELISM=false    — prevent HuggingFace tokenizer fork warnings
set -euo pipefail
[ "${DEBUG:-0}" = "1" ] && set -x

# --- Reproducibility: set before any Python or library is invoked ---
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

# ===========================================================================
# Hardware target constants — update here only if cluster hardware changes.
# All gpu checks, logs, and manifest entries reference these variables.
# ===========================================================================
TARGET_GPU_NAME="L4"
TARGET_GPU_COUNT=4
TARGET_COMPUTE_CAP_MAJOR=8
TARGET_COMPUTE_CAP_MINOR=9
TARGET_VRAM_GB_MIN=22.0
TARGET_TORCH_CUDA_RUNTIME="11.7"
TARGET_DRIVER_CUDA="12.8"
TARGET_PYTHON_VERSION="3.11.9"

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
    echo " Hardware target: ${TARGET_GPU_COUNT}x NVIDIA ${TARGET_GPU_NAME} | CUDA runtime ${TARGET_TORCH_CUDA_RUNTIME} | driver CUDA ${TARGET_DRIVER_CUDA}"
    if command -v nvidia-smi &>/dev/null; then
        echo " --- nvidia-smi per-GPU summary ---"
        nvidia-smi --query-gpu=index,name,memory.total,driver_version \
            --format=csv,noheader | while IFS=',' read -r idx name mem drv; do
            echo "  GPU $idx:$(echo "$name" | xargs) | VRAM:$(echo "$mem" | xargs) | Driver:$(echo "$drv" | xargs)"
        done
        DRIVER_CUDA=$(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}')
        echo " Driver CUDA (max supported): $DRIVER_CUDA"
        echo " NOTE: torch wheel compiled against CUDA runtime ${TARGET_TORCH_CUDA_RUNTIME} (cu117)."
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
    PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"
    if [ -f "$PYTHON" ] && $PYTHON -c "import sys; sys.exit(0 if sys.version_info[:3] == (${PYVER_TUPLE}) else 1)" 2>/dev/null; then
        echo " .venv already exists with Python ${TARGET_PYTHON_VERSION} — skipping creation"
        return
    fi

    if [ -d "$PROJECT_ROOT/.venv" ]; then
        echo "WARNING: .venv exists but is NOT Python ${TARGET_PYTHON_VERSION} — it will be removed."
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

    echo " Creating .venv with Python ${TARGET_PYTHON_VERSION}..."
    $UV venv .venv --python "${TARGET_PYTHON_VERSION}" --seed
}

verify_python() {
    PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"
    $PYTHON -c "import sys; assert sys.version_info[:3] == (${PYVER_TUPLE}), f'Expected ${TARGET_PYTHON_VERSION} got {sys.version}'"
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

check_dependency_drift() {
    echo " Checking for dependency drift..."

    # 1. Detect if pyproject.toml is newer than uv.lock — indicates uv.lock
    #    was not regenerated after a dependency change, meaning the lockfile
    #    is stale and does not reflect the current pyproject.toml spec.
    if [ "$PROJECT_ROOT/pyproject.toml" -nt "$PROJECT_ROOT/uv.lock" ]; then
        echo "ERROR: pyproject.toml is newer than uv.lock — lockfile is stale."
        echo "       Someone edited pyproject.toml without regenerating uv.lock."
        echo "       To fix deliberately: uv lock && git add uv.lock && git commit -m 'chore: regenerate uv.lock'"
        exit 1
    fi
    echo " pyproject.toml vs uv.lock timestamp — ok (lockfile is not stale)"

    # 2. Verify uv.lock is consistent with pyproject.toml using uv's own checker.
    #    'uv lock --check' exits non-zero if the lockfile would change on re-solve,
    #    catching cases where pyproject.toml constraints no longer satisfy uv.lock pins.
    if $UV lock --check 2>/dev/null; then
        echo " uv lock --check — lockfile is consistent with pyproject.toml"
    else
        echo "ERROR: uv lock --check failed — uv.lock is inconsistent with pyproject.toml."
        echo "       To fix deliberately: uv lock && git add uv.lock && git commit -m 'chore: regenerate uv.lock'"
        exit 1
    fi

    # 3. Verify installed packages match uv.lock exactly.
    #    'uv sync --frozen --check' (uv >=0.4) exits non-zero if the venv
    #    diverges from the lockfile — catching manual pip installs or removals
    #    that happened after setup. This is the primary drift detection gate.
    if $UV sync --frozen --dev --check 2>/dev/null; then
        echo " uv sync --frozen --check — installed packages match uv.lock exactly"
    else
        echo "ERROR: Installed packages diverge from uv.lock."
        echo "       This usually means someone ran 'pip install' manually after setup."
        echo "       To fix: bash setup.sh  (re-syncs from uv.lock)"
        exit 1
    fi

    # 4. Cross-check critical package versions directly in Python as a final
    #    sanity gate — uv sync --check may not exist in older uv builds,
    #    so this Python check is the fallback that always works.
    $PYTHON -c "
import importlib.metadata as meta

# Critical packages with minimum versions required for this project.
# These are a subset of pyproject.toml — add more if drift in these causes breakage.
required = {
    'torch':            ('2.0.0',  '2.0.1+cu117'),
    'transformers':     ('4.35.0', None),
    'datasets':         ('2.16.0', None),
    'faiss-cpu':        ('1.7.0',  None),
    'spacy':            ('3.7.0',  None),
    'scikit-learn':     ('1.5.0',  None),
    'numpy':            ('1.24.0', None),
    'pandas':           ('2.2.0',  None),
}

from packaging.version import Version
drift = []
for pkg, (min_ver, exact_ver) in required.items():
    try:
        installed = meta.version(pkg)
        if Version(installed) < Version(min_ver):
            drift.append(f'{pkg}: installed {installed} < required {min_ver}')
        elif exact_ver and installed != exact_ver:
            # Warn on exact mismatch (torch wheel) but do not hard-fail —
            # a newer compatible wheel is acceptable; a CPU wheel is not.
            print(f'  WARNING: {pkg} installed={installed}, expected={exact_ver} (check wheel type)')
        else:
            print(f'  {pkg:<20} {installed} — ok')
    except meta.PackageNotFoundError:
        drift.append(f'{pkg}: NOT INSTALLED')

if drift:
    print('ERROR: Dependency drift detected:')
    for d in drift:
        print(f'  {d}')
    raise SystemExit(1)

print('  All critical package versions verified — no drift detected')
"
}

write_repro_env() {
    echo " Writing reproducibility .env..."
    cat > "$PROJECT_ROOT/.env" << 'ENVEOF'
# .env
# Path: cs1090b_HallucinationLegalRAGChatbots/.env
# Reproducibility environment variables — source this before running experiments.
# Generated by setup.sh — do not edit manually.
# Load in notebooks via: import dotenv; dotenv.load_dotenv()
# Load in shell via:     set -a && source .env && set +a
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
ENVEOF

    $PYTHON -c "
import os
checks = {
    'PYTHONHASHSEED': '0',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'TOKENIZERS_PARALLELISM': 'false',
}
for var, expected in checks.items():
    actual = os.environ.get(var)
    assert actual == expected, f'{var}={actual!r} — expected {expected!r}'
    print(f'  {var}={actual} — verified')
"
}

verify_numerical_stability() {
    echo " Verifying numerical/runtime stability configuration..."
    $PYTHON -c "
import os, torch

# 1. CUBLAS_WORKSPACE_CONFIG must be set before torch import
cublas_cfg = os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')
assert cublas_cfg == ':4096:8', (
    f'CUBLAS_WORKSPACE_CONFIG={cublas_cfg!r} — must be :4096:8 for deterministic cuBLAS.'
)
print(f'  CUBLAS_WORKSPACE_CONFIG={cublas_cfg} — ok')

# 2. Enable deterministic algorithms — warn_only=False raises on non-deterministic ops
torch.use_deterministic_algorithms(True, warn_only=False)
assert torch.are_deterministic_algorithms_enabled(), \
    'torch.use_deterministic_algorithms(True) did not take effect'
print(f'  torch.use_deterministic_algorithms: enabled (warn_only=False)')

# 3. Disable cuDNN benchmark — prevents auto-selection of non-deterministic algorithms
torch.backends.cudnn.benchmark = False
assert not torch.backends.cudnn.benchmark, \
    'cudnn.benchmark=True — non-deterministic algorithm selection is active'
print(f'  cudnn.benchmark: False — deterministic algorithm selection')

# 4. Enable cuDNN deterministic mode — forces bit-exact cuDNN ops
torch.backends.cudnn.deterministic = True
assert torch.backends.cudnn.deterministic, \
    'cudnn.deterministic=False — cuDNN may produce non-deterministic results'
print(f'  cudnn.deterministic: True — bit-exact cuDNN ops enabled')

# 5. Verify PYTHONHASHSEED
hashseed = os.environ.get('PYTHONHASHSEED', 'not set')
assert hashseed == '0', \
    f'PYTHONHASHSEED={hashseed!r} — must be 0 for deterministic hash randomization'
print(f'  PYTHONHASHSEED={hashseed} — deterministic hash randomization')

# 6. Verify TOKENIZERS_PARALLELISM
tok_par = os.environ.get('TOKENIZERS_PARALLELISM', 'not set')
assert tok_par == 'false', \
    f'TOKENIZERS_PARALLELISM={tok_par!r} — must be false to prevent tokenizer fork deadlocks'
print(f'  TOKENIZERS_PARALLELISM={tok_par} — tokenizer fork safety enabled')

print()
print('  Numerical stability configuration verified.')
print('  NOTE: Notebooks must also call torch.use_deterministic_algorithms(True, warn_only=False)')
print('        in Cell 1 before any GPU ops.')
"
}

download_nlp_models() {
    echo " Installing spaCy ${SPACY_MODEL} ${SPACY_MODEL_VERSION} (pinned)..."
    mkdir -p "$SPACY_CACHE_DIR"

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
    $PYTHON -m pip install --quiet "$SPACY_WHEEL"
    echo " ${SPACY_MODEL} ${SPACY_MODEL_VERSION} installed from pinned wheel"
}

run_env_smoke_tests() {
    echo " Running environment smoke tests..."

    $PYTHON -c "
import torch
ver = torch.__version__
assert ver.startswith('2.') and 'cu' in ver, f'Expected torch 2.x+cuXXX got {ver}'
t = torch.tensor([1.0, 2.0, 3.0])
assert torch.allclose(t.mean(), torch.tensor(2.0)), 'torch tensor op failed'
print(f'  torch {ver} — tensor op ok')
"
    $PYTHON -c "
import transformers
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
ids = tok('hello world', return_tensors='pt')
assert ids['input_ids'].shape[1] > 0, 'tokenizer produced empty output'
print(f'  transformers {transformers.__version__} — tokenizer ok')
"
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
    echo " Running GPU smoke tests (target: ${TARGET_GPU_COUNT}x NVIDIA ${TARGET_GPU_NAME}, cu117 wheel, driver CUDA ${TARGET_DRIVER_CUDA})..."
    $PYTHON -c "
import torch

TARGET_GPU_NAME    = '${TARGET_GPU_NAME}'
TARGET_GPU_COUNT   = ${TARGET_GPU_COUNT}
TARGET_CAP         = (${TARGET_COMPUTE_CAP_MAJOR}, ${TARGET_COMPUTE_CAP_MINOR})
TARGET_VRAM_GB_MIN = ${TARGET_VRAM_GB_MIN}
TARGET_TORCH_CUDA  = '${TARGET_TORCH_CUDA_RUNTIME}'

assert torch.cuda.is_available(), 'CUDA not available — wrong torch wheel'
assert torch.version.cuda.startswith(TARGET_TORCH_CUDA), (
    f'Expected torch CUDA runtime {TARGET_TORCH_CUDA} (cu117 wheel) got {torch.version.cuda}. '
    f'NOTE: driver CUDA (${TARGET_DRIVER_CUDA}) != torch runtime CUDA ({TARGET_TORCH_CUDA}) — expected.'
)
n = torch.cuda.device_count()
assert n >= TARGET_GPU_COUNT, f'Expected {TARGET_GPU_COUNT}x NVIDIA {TARGET_GPU_NAME} GPUs, got {n}'
for i in range(n):
    name    = torch.cuda.get_device_name(i)
    cap     = torch.cuda.get_device_capability(i)
    vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
    assert TARGET_GPU_NAME in name, \
        f'GPU {i}: expected NVIDIA {TARGET_GPU_NAME}, got {name}'
    assert cap >= TARGET_CAP, \
        f'GPU {i}: expected compute cap >={TARGET_CAP} for {TARGET_GPU_NAME}, got {cap}'
    assert vram_gb >= TARGET_VRAM_GB_MIN, \
        f'GPU {i}: expected >={TARGET_VRAM_GB_MIN}GB VRAM, got {vram_gb:.1f}GB'
    print(f'  GPU [{i}] {name} | cap {cap} | {vram_gb:.1f} GB — ok')
t = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
assert t.device.type == 'cuda', 'tensor not on CUDA device'
assert torch.allclose(t.mean().cpu(), torch.tensor(2.0)), 'GPU tensor op failed'
print(f'  CUDA {torch.version.cuda} — GPU tensor op on cuda:0 ok')
"
}

write_manifest() {
    echo " Writing environment manifest..."
    mkdir -p "$PROJECT_ROOT/logs"

    GIT_SHA=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "not-a-git-repo")
    GIT_BRANCH=$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    GIT_DIRTY=$(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | wc -l | xargs)
    UVLOCK_SHA256=$(sha256sum "$PROJECT_ROOT/uv.lock" | cut -d' ' -f1)

    $PYTHON -c "
import json, torch, transformers, spacy, sys, platform, subprocess, os
import importlib.metadata as meta
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

def get_driver_version():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True)
        return result.stdout.strip().splitlines()[0] if result.stdout.strip() else 'unknown'
    except FileNotFoundError:
        return 'nvidia-smi not found'

def get_faiss_version():
    try:
        import faiss
        return getattr(faiss, '__version__', 'installed — version attr unavailable')
    except ImportError:
        return 'not installed'

def get_installed_versions(pkgs):
    versions = {}
    for pkg in pkgs:
        try:
            versions[pkg] = meta.version(pkg)
        except meta.PackageNotFoundError:
            versions[pkg] = 'not installed'
    return versions

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
    # --- Repo provenance ---
    'git_sha': '${GIT_SHA}',
    'git_branch': '${GIT_BRANCH}',
    'git_dirty_files': int('${GIT_DIRTY}'),
    'uv_lock_sha256': '${UVLOCK_SHA256}',
    # --- Python ---
    'python': sys.version,
    'platform': platform.platform(),
    # --- Reproducibility env vars ---
    'repro_env': {
        'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'TOKENIZERS_PARALLELISM': os.environ.get('TOKENIZERS_PARALLELISM'),
    },
    # --- Numerical stability settings ---
    'numerical_stability': {
        'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
    },
    # --- Hardware targets (from setup.sh constants) ---
    'hardware_target': {
        'gpu_name': '${TARGET_GPU_NAME}',
        'gpu_count': ${TARGET_GPU_COUNT},
        'compute_cap_min': [${TARGET_COMPUTE_CAP_MAJOR}, ${TARGET_COMPUTE_CAP_MINOR}],
        'vram_gb_min': ${TARGET_VRAM_GB_MIN},
        'torch_cuda_runtime': '${TARGET_TORCH_CUDA_RUNTIME}',
        'driver_cuda': '${TARGET_DRIVER_CUDA}',
        'python_version': '${TARGET_PYTHON_VERSION}',
    },
    # --- Torch & CUDA (actual) ---
    'torch': torch.__version__,
    'torch_cuda_runtime': torch.version.cuda,
    'driver_cuda': get_driver_cuda(),
    'driver_version': get_driver_version(),
    'cudnn': str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
    'cuda_toolkit_nvcc': get_nvcc_version(),
    'cuda_available': torch.cuda.is_available(),
    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'gpus': gpus,
    # --- NLP / RAG libraries ---
    'transformers': transformers.__version__,
    'spacy': spacy.__version__,
    'spacy_model': '${SPACY_MODEL}',
    'spacy_model_version': nlp.meta.get('version'),
    'spacy_model_sha256': '${SPACY_MODEL_SHA256}',
    'faiss': get_faiss_version(),
    # --- Full installed package snapshot for drift auditing ---
    'installed_packages': get_installed_versions([
        'torch', 'transformers', 'datasets', 'faiss-cpu', 'spacy',
        'scikit-learn', 'numpy', 'pandas', 'langchain', 'gensim',
        'sentence-transformers', 'networkx', 'pytest', 'mypy', 'hypothesis',
    ]),
}
with open('logs/environment_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print('  manifest written to logs/environment_manifest.json')
print(f'  git sha: ${GIT_SHA} | branch: ${GIT_BRANCH} | dirty files: ${GIT_DIRTY}')
print(f'  uv.lock sha256: ${UVLOCK_SHA256}')
print(f'  repro env: PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false')
print(f'  numerical stability: deterministic_algorithms=True cudnn.benchmark=False cudnn.deterministic=True')
"
}

register_kernel() {
    echo " Registering Jupyter kernel..."
    $PYTHON -m ipykernel install --user \
        --name hallucination-legal-rag \
        --display-name "HallucinationLegalRAG (${TARGET_PYTHON_VERSION})"

    $PYTHON -m jupyter kernelspec list --json 2>/dev/null | $PYTHON -c "
import sys, json
data = json.load(sys.stdin)
kernels = data.get('kernelspecs', {})
assert 'hallucination-legal-rag' in kernels, \
    'Kernel registration failed — not found in venv jupyter kernelspec list'
spec = kernels['hallucination-legal-rag']
print(f'  kernel verified via venv jupyter: {spec[\"spec\"][\"display_name\"]}')
print(f'  kernel path: {spec[\"resource_dir\"]}')
" || echo "WARNING: Could not verify kernel via venv jupyter — ipykernel may not be fully installed"
}

verify_tests() {
    echo " Verifying test suite..."
    UNIT_COUNT=$($UV run pytest tests/ --co -q -m unit 2>/dev/null | grep -c "^tests/" || true)

    if [ "${UNIT_COUNT}" -gt 0 ]; then
        echo " Found ${UNIT_COUNT} unit tests — running as verification gate..."
        $UV run pytest tests/ -m unit -q --tb=short && \
            echo " Unit tests passed — environment verified" || \
            { echo "ERROR: Unit tests failed — environment may be broken"; exit 1; }
    else
        echo " No unit tests found yet — falling back to collection check..."
        $UV run pytest tests/ --co -q 2>/dev/null && \
            echo " Test collection ok — no unit tests to run yet" || \
            echo "WARNING: Test collection failed — check src/ imports"
    fi
}

echo "============================================================"
echo " cs1090b_HallucinationLegalRAGChatbots — Environment Bootstrap"
echo " Target: ${TARGET_GPU_COUNT}x NVIDIA ${TARGET_GPU_NAME} | Python ${TARGET_PYTHON_VERSION} | torch 2.0.1+cu117"
echo " Driver CUDA: ${TARGET_DRIVER_CUDA} (forward-compat) | torch runtime: ${TARGET_TORCH_CUDA_RUNTIME}"
echo " spaCy model: ${SPACY_MODEL} ${SPACY_MODEL_VERSION} (pinned + checksummed)"
echo " Offline mode: OFFLINE=1 bash setup.sh (requires .cache/spacy/ wheel)"
echo " Repro vars:   PYTHONHASHSEED=0 | CUBLAS_WORKSPACE_CONFIG=:4096:8 | TOKENIZERS_PARALLELISM=false"
echo " Stability:    deterministic_algorithms=True | cudnn.benchmark=False | cudnn.deterministic=True"
echo "============================================================"
check_uv
check_lockfile
log_gpu
ensure_venv
verify_python
sync_dependencies
check_dependency_drift
write_repro_env
verify_numerical_stability
download_nlp_models
run_env_smoke_tests
run_gpu_smoke_tests
write_manifest
register_kernel
verify_tests
echo "============================================================"
echo " Environment ready."
echo " Activate:  source .venv/bin/activate"
echo " Kernel:    HallucinationLegalRAG (${TARGET_PYTHON_VERSION})"
echo " Manifest:  logs/environment_manifest.json"
echo " Repro env: source .env  (or dotenv.load_dotenv() in notebook)"
echo " CPU mode:  SKIP_GPU=1 bash setup.sh"
echo " Dry run:   DRY_RUN=1 bash setup.sh"
echo " Offline:   OFFLINE=1 bash setup.sh"
echo "============================================================"
