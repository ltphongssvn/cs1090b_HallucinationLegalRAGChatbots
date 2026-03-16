#!/usr/bin/env bash
# setup.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/setup.sh
#
# Usage:        bash setup.sh
# Debug:        DEBUG=1 bash setup.sh
# Skip GPU:     SKIP_GPU=1 bash setup.sh
# Dry run:      DRY_RUN=1 bash setup.sh
# Offline:      OFFLINE=1 bash setup.sh  (requires cached wheel in .cache/spacy/)
# Single step:  STEP=<name> bash setup.sh
#               e.g. STEP=download_nlp_models bash setup.sh
#               e.g. STEP=register_kernel bash setup.sh
#               e.g. STEP=run_gpu_smoke_tests bash setup.sh
#
# Execution order (fail-fast design):
#   1. preflight_fast_checks()  — cheap gates: disk, nvidia-smi, python, lockfile (seconds)
#   2. check_uv / check_lockfile — tool availability
#   3. log_gpu                  — pre-venv driver-level info
#   4. ensure_venv              — expensive: build venv
#   5. verify_python / sync_dependencies — expensive: package sync
#   6. check_dependency_drift   — post-sync integrity
#   7. detect_hardware          — post-venv torch-level detection
#   8. ... rest of setup
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
# ===========================================================================
TARGET_GPU_NAME="L4"
TARGET_GPU_COUNT=4
TARGET_COMPUTE_CAP_MAJOR=8
TARGET_COMPUTE_CAP_MINOR=9
TARGET_VRAM_GB_MIN=22.0
TARGET_TORCH_CUDA_RUNTIME="11.7"
TARGET_DRIVER_CUDA="12.8"
TARGET_PYTHON_VERSION="3.11.9"
TARGET_MIN_DISK_GB=50

# ===========================================================================
# Reproducibility constants — single source of truth for ALL seeding.
# Change RANDOM_SEED here to run seed sensitivity experiments.
# ===========================================================================
RANDOM_SEED=0
REPRO_PYTHONHASHSEED=0
REPRO_CUBLAS_CFG=":4096:8"
REPRO_TOKENIZERS_PAR="false"

# Pinned spaCy model
SPACY_MODEL="en_core_web_sm"
SPACY_MODEL_VERSION="3.8.0"
SPACY_MODEL_URL="https://github.com/explosion/spacy-models/releases/download/${SPACY_MODEL}-${SPACY_MODEL_VERSION}/${SPACY_MODEL}-${SPACY_MODEL_VERSION}-py3-none-any.whl"
SPACY_MODEL_SHA256="5e97b9ec4f95153b992896c5c45b1a00c3fcde7f764426c5370f2f11e71abef2"
SPACY_CACHE_DIR="$PROJECT_ROOT/.cache/spacy"
SPACY_WHEEL="$SPACY_CACHE_DIR/${SPACY_MODEL}-${SPACY_MODEL_VERSION}-py3-none-any.whl"

# Runtime-detected hardware — populated by detect_hardware()
DETECTED_GPU_NAME=""
DETECTED_GPU_COUNT=""
DETECTED_DRIVER_CUDA=""
DETECTED_TORCH_CUDA=""
DETECTED_CUDNN=""
HARDWARE_MATCH="true"

# ===========================================================================
# Research ergonomics: color output, per-step timing, summary table
# ===========================================================================
# Colors — disabled automatically if not a TTY (e.g. CI log files)
if [ -t 1 ]; then
    C_RESET="\033[0m"
    C_BOLD="\033[1m"
    C_GREEN="\033[0;32m"
    C_YELLOW="\033[0;33m"
    C_RED="\033[0;31m"
    C_CYAN="\033[0;36m"
    C_DIM="\033[2m"
else
    C_RESET=""; C_BOLD=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_CYAN=""; C_DIM=""
fi

# Summary table: arrays of step names, statuses, and durations
SUMMARY_STEPS=()
SUMMARY_STATUS=()
SUMMARY_DURATION=()

_step_start_time=0

step_begin() {
    # Called at the start of each named step — prints header, records start time
    local name="$1"
    _step_start_time=$(date +%s)
    echo -e "${C_BOLD}${C_CYAN}▶ ${name}${C_RESET}"
}

step_end() {
    # Called at the end of each named step — records duration and status
    local name="$1"
    local status="${2:-PASS}"  # PASS | WARN | SKIP
    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - _step_start_time ))
    SUMMARY_STEPS+=("$name")
    SUMMARY_DURATION+=("${duration}s")
    case "$status" in
        PASS) SUMMARY_STATUS+=("${C_GREEN}PASS${C_RESET}") ;;
        WARN) SUMMARY_STATUS+=("${C_YELLOW}WARN${C_RESET}") ;;
        SKIP) SUMMARY_STATUS+=("${C_DIM}SKIP${C_RESET}") ;;
        *)    SUMMARY_STATUS+=("${C_RED}FAIL${C_RESET}") ;;
    esac
    echo -e "  ${C_DIM}(${duration}s)${C_RESET}"
}

print_summary() {
    echo ""
    echo -e "${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_BOLD} Setup Summary${C_RESET}"
    echo -e "${C_BOLD}============================================================${C_RESET}"
    printf "  %-40s %-8s %s\n" "Step" "Status" "Duration"
    printf "  %-40s %-8s %s\n" "----" "------" "--------"
    for i in "${!SUMMARY_STEPS[@]}"; do
        printf "  %-40s " "${SUMMARY_STEPS[$i]}"
        echo -ne "${SUMMARY_STATUS[$i]}"
        printf "   %s\n" "${SUMMARY_DURATION[$i]}"
    done
    echo -e "${C_BOLD}============================================================${C_RESET}"
}

# Single-step mode: STEP=<fn_name> bash setup.sh
# Allows researchers to re-run just one phase without full setup.
# Example: STEP=download_nlp_models bash setup.sh
# Example: STEP=register_kernel bash setup.sh
run_step() {
    local fn="$1"
    shift
    if [ -n "${STEP:-}" ] && [ "$STEP" != "$fn" ]; then
        # Skip this step in single-step mode
        SUMMARY_STEPS+=("$fn")
        SUMMARY_STATUS+=("${C_DIM}SKIP${C_RESET}")
        SUMMARY_DURATION+=("-")
        return 0
    fi
    step_begin "$fn"
    "$fn" "$@"
    step_end "$fn" "PASS"
}

trap 'echo -e "${C_RED}ERROR: setup.sh failed at line $LINENO — environment may be incomplete${C_RESET}"; print_summary' ERR

# ===========================================================================
# Step functions
# ===========================================================================

preflight_fast_checks() {
    echo " Running preflight fast checks (pre-venv, seconds)..."
    local failures=()

    FREE_GB=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
    if [ "${FREE_GB:-0}" -lt "$TARGET_MIN_DISK_GB" ]; then
        failures+=("Disk: only ${FREE_GB}GB free on $PROJECT_ROOT, need ${TARGET_MIN_DISK_GB}GB")
    else
        echo -e "  ${C_GREEN}✓${C_RESET} disk: ${FREE_GB}GB free >= ${TARGET_MIN_DISK_GB}GB"
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        failures+=("nvidia-smi not found — not a GPU node. Request a GPU allocation.")
    else
        SYSFS_GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | xargs)
        if [ "${SYSFS_GPU_COUNT:-0}" -lt "$TARGET_GPU_COUNT" ]; then
            failures+=("GPU count (nvidia-smi): detected ${SYSFS_GPU_COUNT}, need ${TARGET_GPU_COUNT}.")
        else
            echo -e "  ${C_GREEN}✓${C_RESET} nvidia-smi GPU count: ${SYSFS_GPU_COUNT} >= ${TARGET_GPU_COUNT}"
        fi

        SYSFS_GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')
        if ! echo "$SYSFS_GPU_NAMES" | grep -q "$TARGET_GPU_NAME"; then
            failures+=("GPU name (nvidia-smi): detected '${SYSFS_GPU_NAMES}', expected '${TARGET_GPU_NAME}'.")
        else
            echo -e "  ${C_GREEN}✓${C_RESET} nvidia-smi GPU name: contains '${TARGET_GPU_NAME}'"
        fi

        SYSFS_DRIVER_CUDA=$(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}')
        if [ "${SYSFS_DRIVER_CUDA}" != "${TARGET_DRIVER_CUDA}" ]; then
            echo -e "  ${C_YELLOW}WARNING${C_RESET}: driver CUDA ${SYSFS_DRIVER_CUDA} != target ${TARGET_DRIVER_CUDA}. Update TARGET_DRIVER_CUDA if intentional."
        else
            echo -e "  ${C_GREEN}✓${C_RESET} driver CUDA: ${SYSFS_DRIVER_CUDA}"
        fi
    fi

    if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
        failures+=("pyproject.toml not found at $PROJECT_ROOT")
    else
        echo -e "  ${C_GREEN}✓${C_RESET} pyproject.toml: present"
    fi

    if [ ! -f "$PROJECT_ROOT/uv.lock" ]; then
        failures+=("uv.lock not found — run: uv lock && git add uv.lock && git commit -m 'chore: pin uv.lock'")
    else
        echo -e "  ${C_GREEN}✓${C_RESET} uv.lock: present"
    fi

    if ! command -v uv &>/dev/null && ! command -v ~/.local/bin/uv &>/dev/null; then
        failures+=("uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh")
    else
        UV_BIN=$(command -v uv 2>/dev/null || echo ~/.local/bin/uv)
        echo -e "  ${C_GREEN}✓${C_RESET} uv: $($UV_BIN --version)"
    fi

    if [ ${#failures[@]} -gt 0 ]; then
        echo ""
        echo -e "${C_RED}${C_BOLD}============================================================${C_RESET}"
        echo -e "${C_RED}${C_BOLD} PREFLIGHT FAILED — ${#failures[@]} issue(s). Fix before retrying.${C_RESET}"
        echo -e "${C_RED}${C_BOLD}============================================================${C_RESET}"
        for i in "${!failures[@]}"; do
            echo -e "  ${C_RED}[$((i+1))]${C_RESET} ${failures[$i]}"
        done
        echo ""
        echo -e "${C_DIM} These checks are fast (pre-venv). Failing here saves venv build + sync time on the wrong node.${C_RESET}"
        exit 1
    fi

    echo -e "  ${C_GREEN}✓${C_RESET} Preflight fast checks passed — proceeding to expensive operations."
}

check_uv() {
    if ! command -v uv &>/dev/null && ! command -v ~/.local/bin/uv &>/dev/null; then
        echo -e "${C_RED}ERROR${C_RESET}: uv not found."
        exit 1
    fi
    UV=$(command -v uv 2>/dev/null || echo ~/.local/bin/uv)
    echo -e "  ${C_GREEN}✓${C_RESET} uv: $($UV --version)"
}

check_lockfile() {
    if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
        echo -e "${C_RED}ERROR${C_RESET}: pyproject.toml not found."
        exit 1
    fi
    if [ ! -f "$PROJECT_ROOT/uv.lock" ]; then
        echo -e "${C_RED}ERROR${C_RESET}: uv.lock not found."
        echo "       To generate deliberately: uv lock && git add uv.lock && git commit -m 'chore: pin uv.lock'"
        exit 1
    fi
    echo -e "  ${C_GREEN}✓${C_RESET} uv.lock sha256: $(sha256sum "$PROJECT_ROOT/uv.lock" | cut -d' ' -f1)"
}

log_gpu() {
    echo " Hardware target: ${TARGET_GPU_COUNT}x NVIDIA ${TARGET_GPU_NAME} | CUDA runtime ${TARGET_TORCH_CUDA_RUNTIME} | driver CUDA ${TARGET_DRIVER_CUDA}"
    if command -v nvidia-smi &>/dev/null; then
        echo " --- nvidia-smi per-GPU summary (pre-venv) ---"
        nvidia-smi --query-gpu=index,name,memory.total,driver_version \
            --format=csv,noheader | while IFS=',' read -r idx name mem drv; do
            echo "  GPU $idx:$(echo "$name" | xargs) | VRAM:$(echo "$mem" | xargs) | Driver:$(echo "$drv" | xargs)"
        done
        DRIVER_CUDA_DETECTED=$(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}')
        echo " Driver CUDA (max supported): $DRIVER_CUDA_DETECTED"
        echo -e " ${C_DIM}NOTE: torch wheel compiled against CUDA runtime ${TARGET_TORCH_CUDA_RUNTIME} (cu117). Driver CUDA $DRIVER_CUDA_DETECTED is forward-compatible.${C_RESET}"
    else
        echo -e " ${C_YELLOW}WARNING${C_RESET}: nvidia-smi not found"
    fi
    if command -v nvcc &>/dev/null; then
        echo " CUDA toolkit (nvcc): $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')"
    else
        echo -e " ${C_YELLOW}WARNING${C_RESET}: nvcc not on PATH"
    fi
}

detect_hardware() {
    echo " Detecting hardware (post-venv, torch-level)..."

    DETECTED_HW=$($PYTHON -c "
import torch, json
result = {
    'cuda_available': torch.cuda.is_available(),
    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'torch_cuda': torch.version.cuda or 'unknown',
    'cudnn': str(torch.backends.cudnn.version()) if torch.cuda.is_available() else 'N/A',
    'gpus': [],
}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        result['gpus'].append({
            'index': i,
            'name': props.name,
            'vram_gb': round(props.total_memory / 1e9, 2),
            'compute_capability': list(torch.cuda.get_device_capability(i)),
        })
print(json.dumps(result))
")

    DETECTED_GPU_COUNT=$(echo "$DETECTED_HW" | $PYTHON -c "import json,sys; d=json.load(sys.stdin); print(d['gpu_count'])")
    DETECTED_TORCH_CUDA=$(echo "$DETECTED_HW" | $PYTHON -c "import json,sys; d=json.load(sys.stdin); print(d['torch_cuda'])")
    DETECTED_CUDNN=$(echo "$DETECTED_HW" | $PYTHON -c "import json,sys; d=json.load(sys.stdin); print(d['cudnn'])")
    DETECTED_GPU_NAME=$(echo "$DETECTED_HW" | $PYTHON -c "
import json,sys
d=json.load(sys.stdin)
names = list({g['name'] for g in d['gpus']})
print(', '.join(names) if names else 'N/A')
")
    if command -v nvidia-smi &>/dev/null; then
        DETECTED_DRIVER_CUDA=$(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}')
    else
        DETECTED_DRIVER_CUDA="unknown"
    fi

    echo " --- Detected hardware vs targets ---"
    printf "  %-20s %-30s %s\n" "Property" "Detected" "Target"
    printf "  %-20s %-30s %s\n" "--------" "--------" "------"
    printf "  %-20s %-30s %s\n" "GPU name"  "'${DETECTED_GPU_NAME}'" "contains '${TARGET_GPU_NAME}'"
    printf "  %-20s %-30s %s\n" "GPU count" "${DETECTED_GPU_COUNT}" "${TARGET_GPU_COUNT}"
    printf "  %-20s %-30s %s\n" "torch CUDA" "${DETECTED_TORCH_CUDA}" "${TARGET_TORCH_CUDA_RUNTIME}"
    printf "  %-20s %-30s %s\n" "driver CUDA" "${DETECTED_DRIVER_CUDA}" "${TARGET_DRIVER_CUDA}"
    printf "  %-20s %-30s\n"    "cuDNN" "${DETECTED_CUDNN}"

    echo "$DETECTED_HW" | $PYTHON -c "
import json, sys
d = json.load(sys.stdin)
for g in d['gpus']:
    print(f\"  GPU[{g['index']}]: {g['name']} | {g['vram_gb']}GB | cap {g['compute_capability']}\")
"

    if [ "${TARGET_GPU_NAME}" != "" ] && ! echo "${DETECTED_GPU_NAME}" | grep -q "${TARGET_GPU_NAME}"; then
        echo -e " ${C_YELLOW}WARNING${C_RESET}: GPU name mismatch — '${DETECTED_GPU_NAME}' vs target '${TARGET_GPU_NAME}'."
        HARDWARE_MATCH="false"
    fi
    if [ "${DETECTED_GPU_COUNT}" != "${TARGET_GPU_COUNT}" ]; then
        echo -e " ${C_YELLOW}WARNING${C_RESET}: GPU count mismatch — detected ${DETECTED_GPU_COUNT}, target ${TARGET_GPU_COUNT}."
        HARDWARE_MATCH="false"
    fi
    if ! echo "${DETECTED_TORCH_CUDA}" | grep -q "^${TARGET_TORCH_CUDA_RUNTIME}"; then
        echo -e " ${C_YELLOW}WARNING${C_RESET}: torch CUDA mismatch — detected ${DETECTED_TORCH_CUDA}, target ${TARGET_TORCH_CUDA_RUNTIME}."
        HARDWARE_MATCH="false"
    fi

    if [ "$HARDWARE_MATCH" = "true" ]; then
        echo -e " ${C_GREEN}✓${C_RESET} Hardware detection complete — all values match targets."
    else
        echo -e " ${C_YELLOW}WARNING${C_RESET}: Hardware mismatches detected — run_gpu_smoke_tests() will hard-fail."
        step_end "detect_hardware" "WARN"
        return
    fi
}

ensure_venv() {
    PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"
    if [ -f "$PYTHON" ] && $PYTHON -c "import sys; sys.exit(0 if sys.version_info[:3] == (${PYVER_TUPLE}) else 1)" 2>/dev/null; then
        echo -e "  ${C_GREEN}✓${C_RESET} .venv already exists with Python ${TARGET_PYTHON_VERSION} — skipping creation"
        return
    fi

    if [ -d "$PROJECT_ROOT/.venv" ]; then
        echo -e "  ${C_YELLOW}WARNING${C_RESET}: .venv exists but is NOT Python ${TARGET_PYTHON_VERSION} — it will be removed."
        echo "         Contents: $(du -sh "$PROJECT_ROOT/.venv" 2>/dev/null | cut -f1) on disk"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo " DRY_RUN=1 — skipping .venv removal. Exiting."
            exit 0
        fi
        echo "         Aborting in 5 seconds — press Ctrl+C to cancel..."
        sleep 5
        rm -rf "$PROJECT_ROOT/.venv"
    fi

    echo " Creating .venv with Python ${TARGET_PYTHON_VERSION}..."
    $UV venv .venv --python "${TARGET_PYTHON_VERSION}" --seed
}

verify_python() {
    PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"
    $PYTHON -c "import sys; assert sys.version_info[:3] == (${PYVER_TUPLE}), f'Expected ${TARGET_PYTHON_VERSION} got {sys.version}'"
    echo -e "  ${C_GREEN}✓${C_RESET} Python: $($PYTHON --version)"
    echo "  Executable: $($PYTHON -c 'import sys; print(sys.executable)')"
}

sync_dependencies() {
    echo " Syncing dependencies from uv.lock (--frozen)..."
    # --dev is the default in uv >=0.4 but kept explicit here to:
    # (1) document intent — dev tools (pytest, mypy, hypothesis) are required,
    # (2) guard against future uv default changes,
    # (3) signal to collaborators that this is a deliberate inclusion.
    $UV sync --frozen --dev
    echo -e "  ${C_GREEN}✓${C_RESET} Dependencies synced"
}

check_dependency_drift() {
    echo " Checking for dependency drift..."

    if [ "$PROJECT_ROOT/pyproject.toml" -nt "$PROJECT_ROOT/uv.lock" ]; then
        echo -e "${C_RED}ERROR${C_RESET}: pyproject.toml newer than uv.lock — lockfile stale."
        echo "       Fix: uv lock && git add uv.lock && git commit -m 'chore: regenerate uv.lock'"
        exit 1
    fi
    echo -e "  ${C_GREEN}✓${C_RESET} pyproject.toml vs uv.lock timestamp — ok"

    if $UV lock --check 2>/dev/null; then
        echo -e "  ${C_GREEN}✓${C_RESET} uv lock --check — consistent"
    else
        echo -e "${C_RED}ERROR${C_RESET}: uv lock --check failed."
        exit 1
    fi

    if $UV sync --frozen --dev --check 2>/dev/null; then
        echo -e "  ${C_GREEN}✓${C_RESET} uv sync --check — installed packages match uv.lock"
    else
        echo -e "${C_RED}ERROR${C_RESET}: Installed packages diverge from uv.lock. Fix: bash setup.sh"
        exit 1
    fi

    $PYTHON -c "
import importlib.metadata as meta
from packaging.version import Version

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
drift = []
for pkg, (min_ver, exact_ver) in required.items():
    try:
        installed = meta.version(pkg)
        if Version(installed) < Version(min_ver):
            drift.append(f'{pkg}: installed {installed} < required {min_ver}')
        elif exact_ver and installed != exact_ver:
            print(f'  WARNING: {pkg} installed={installed}, expected={exact_ver}')
        else:
            print(f'  \033[0;32m✓\033[0m {pkg:<20} {installed}')
    except meta.PackageNotFoundError:
        drift.append(f'{pkg}: NOT INSTALLED')
if drift:
    for d in drift:
        print(f'  \033[0;31mERROR\033[0m {d}')
    raise SystemExit(1)
print('  All critical package versions verified')
"
}

write_repro_env() {
    echo " Writing reproducibility .env..."
    cat > "$PROJECT_ROOT/.env" << ENVEOF
# .env
# Path: cs1090b_HallucinationLegalRAGChatbots/.env
# Reproducibility environment variables — source this before running experiments.
# Generated by setup.sh — do not edit manually.
# Load in notebooks via: import dotenv; dotenv.load_dotenv()
# Load in shell via:     set -a && source .env && set +a
export PYTHONHASHSEED=${REPRO_PYTHONHASHSEED}
export CUBLAS_WORKSPACE_CONFIG=${REPRO_CUBLAS_CFG}
export TOKENIZERS_PARALLELISM=${REPRO_TOKENIZERS_PAR}
export RANDOM_SEED=${RANDOM_SEED}
ENVEOF

    $PYTHON -c "
import os
checks = {
    'PYTHONHASHSEED':          '${REPRO_PYTHONHASHSEED}',
    'CUBLAS_WORKSPACE_CONFIG': '${REPRO_CUBLAS_CFG}',
    'TOKENIZERS_PARALLELISM':  '${REPRO_TOKENIZERS_PAR}',
    'RANDOM_SEED':             '${RANDOM_SEED}',
}
for var, expected in checks.items():
    actual = os.environ.get(var)
    assert actual == expected, f'{var}={actual!r} != {expected!r}'
    print(f'  \033[0;32m✓\033[0m {var}={actual}')
"
}

write_repro_module() {
    echo " Writing src/repro.py — notebook/CLI parity module..."
    mkdir -p "$PROJECT_ROOT/src"
    cat > "$PROJECT_ROOT/src/repro.py" << PYEOF
# src/repro.py
# Path: cs1090b_HallucinationLegalRAGChatbots/src/repro.py
#
# Canonical reproducibility configuration module.
# Import and call configure() as the FIRST statement in every notebook Cell 1
# and every CLI training/evaluation script to guarantee notebook/CLI parity.
#
# Usage:
#   from src.repro import configure
#   repro_cfg = configure()
#
# RANDOM_SEED is injected from setup.sh constants block.
# To change: update RANDOM_SEED in setup.sh, re-run bash setup.sh, commit .env + src/repro.py.
import os
import random
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_EXPECTED_PYTHONHASHSEED  = "${REPRO_PYTHONHASHSEED}"
_EXPECTED_CUBLAS_CFG      = "${REPRO_CUBLAS_CFG}"
_EXPECTED_TOKENIZERS_PAR  = "${REPRO_TOKENIZERS_PAR}"
_RANDOM_SEED              = ${RANDOM_SEED}


def _load_dotenv(project_root: Optional[Path] = None) -> None:
    root = project_root or Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f".env not found at {env_path}. Run bash setup.sh.")
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key = key.replace("export ", "").strip()
                    val = val.strip()
                    if key not in os.environ:
                        os.environ[key] = val


def _apply_torch_flags() -> None:
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _EXPECTED_CUBLAS_CFG)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _verify() -> dict:
    import torch
    checks = {}
    for var, expected in [
        ("PYTHONHASHSEED",          _EXPECTED_PYTHONHASHSEED),
        ("CUBLAS_WORKSPACE_CONFIG", _EXPECTED_CUBLAS_CFG),
        ("TOKENIZERS_PARALLELISM",  _EXPECTED_TOKENIZERS_PAR),
    ]:
        actual = os.environ.get(var)
        assert actual == expected, (
            f"{var}={actual!r} — expected {expected!r}. "
            f"Call configure() before importing torch."
        )
        checks[var] = actual
    assert torch.are_deterministic_algorithms_enabled(), \
        "torch.use_deterministic_algorithms not enabled"
    checks["deterministic_algorithms"] = True
    assert not torch.backends.cudnn.benchmark, "cudnn.benchmark=True"
    checks["cudnn_benchmark"] = False
    assert torch.backends.cudnn.deterministic, "cudnn.deterministic=False"
    checks["cudnn_deterministic"] = True
    checks["random_seed"] = _RANDOM_SEED
    return checks


def configure(
    project_root: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Apply and verify all reproducibility settings.
    Call as the FIRST statement in every notebook Cell 1 and CLI script.
    Returns a config dict for logging/manifest inclusion.
    """
    _load_dotenv(project_root)
    _apply_torch_flags()
    _seed_all(_RANDOM_SEED)
    cfg = _verify()
    if verbose:
        import torch
        print("  [repro] Reproducibility configured:")
        print(f"    PYTHONHASHSEED={cfg['PYTHONHASHSEED']}")
        print(f"    CUBLAS_WORKSPACE_CONFIG={cfg['CUBLAS_WORKSPACE_CONFIG']}")
        print(f"    TOKENIZERS_PARALLELISM={cfg['TOKENIZERS_PARALLELISM']}")
        print(f"    RANDOM_SEED={_RANDOM_SEED}")
        print(f"    deterministic_algorithms=True | cudnn.benchmark=False | cudnn.deterministic=True")
        print(f"    seeds: random={_RANDOM_SEED}, numpy={_RANDOM_SEED}, torch={_RANDOM_SEED}")
        if torch.cuda.is_available():
            print(f"    torch.cuda.manual_seed_all({_RANDOM_SEED}) → {torch.cuda.device_count()} GPU(s)")
    return cfg
PYEOF
    echo -e "  ${C_GREEN}✓${C_RESET} src/repro.py written (RANDOM_SEED=${RANDOM_SEED})"

    $PYTHON -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from src.repro import configure
cfg = configure(verbose=True)
assert cfg['random_seed'] == ${RANDOM_SEED}
print('  \033[0;32m✓\033[0m src/repro.configure() verified')
"
}

verify_numerical_stability() {
    echo " Verifying numerical/runtime stability..."
    $PYTHON -c "
import os, torch

checks = [
    ('CUBLAS_WORKSPACE_CONFIG', os.environ.get('CUBLAS_WORKSPACE_CONFIG'), '${REPRO_CUBLAS_CFG}'),
    ('PYTHONHASHSEED',          os.environ.get('PYTHONHASHSEED'),          '${REPRO_PYTHONHASHSEED}'),
    ('TOKENIZERS_PARALLELISM',  os.environ.get('TOKENIZERS_PARALLELISM'),  '${REPRO_TOKENIZERS_PAR}'),
    ('RANDOM_SEED',             os.environ.get('RANDOM_SEED'),             '${RANDOM_SEED}'),
]
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

for name, actual, expected in checks:
    assert actual == expected, f'{name}={actual!r} != {expected!r}'
    print(f'  \033[0;32m✓\033[0m {name}={actual}')
assert torch.are_deterministic_algorithms_enabled()
assert not torch.backends.cudnn.benchmark
assert torch.backends.cudnn.deterministic
print('  \033[0;32m✓\033[0m torch deterministic flags — ok')
print()
print('  NOTE: Notebooks must call: from src.repro import configure; configure()')
"
}

download_nlp_models() {
    echo " Installing spaCy ${SPACY_MODEL} ${SPACY_MODEL_VERSION} (pinned)..."
    mkdir -p "$SPACY_CACHE_DIR"

    if $PYTHON -c "
import spacy, sys
try:
    nlp = spacy.load('${SPACY_MODEL}')
    sys.exit(0 if nlp.meta.get('version') == '${SPACY_MODEL_VERSION}' else 1)
except OSError:
    sys.exit(1)
" 2>/dev/null; then
        echo -e "  ${C_GREEN}✓${C_RESET} ${SPACY_MODEL} ${SPACY_MODEL_VERSION} already installed — skipping"
        return
    fi

    if [ ! -f "$SPACY_WHEEL" ]; then
        if [ "${OFFLINE:-0}" = "1" ]; then
            echo -e "${C_RED}ERROR${C_RESET}: OFFLINE=1 but wheel not at $SPACY_WHEEL"
            exit 1
        fi
        echo " Downloading ${SPACY_MODEL} wheel..."
        curl -fsSL -o "$SPACY_WHEEL" "$SPACY_MODEL_URL"
    else
        echo " Using cached wheel: $SPACY_WHEEL"
    fi

    ACTUAL_SHA=$(sha256sum "$SPACY_WHEEL" | cut -d' ' -f1)
    if [ "$ACTUAL_SHA" != "$SPACY_MODEL_SHA256" ]; then
        echo -e "${C_RED}ERROR${C_RESET}: checksum mismatch! expected=$SPACY_MODEL_SHA256 actual=$ACTUAL_SHA"
        rm -f "$SPACY_WHEEL"
        exit 1
    fi
    echo -e "  ${C_GREEN}✓${C_RESET} Checksum verified"
    $PYTHON -m pip install --quiet "$SPACY_WHEEL"
    echo -e "  ${C_GREEN}✓${C_RESET} ${SPACY_MODEL} ${SPACY_MODEL_VERSION} installed"
}

run_env_smoke_tests() {
    echo " Running environment smoke tests..."
    $PYTHON -c "
import torch
ver = torch.__version__
assert ver.startswith('2.') and 'cu' in ver, f'Expected torch 2.x+cuXXX got {ver}'
t = torch.tensor([1.0, 2.0, 3.0])
assert torch.allclose(t.mean(), torch.tensor(2.0))
print(f'  \033[0;32m✓\033[0m torch {ver} — tensor op ok')
"
    $PYTHON -c "
import transformers
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
ids = tok('hello world', return_tensors='pt')
assert ids['input_ids'].shape[1] > 0
print(f'  \033[0;32m✓\033[0m transformers {transformers.__version__} — tokenizer ok')
"
    $PYTHON -c "
import faiss, numpy as np
index = faiss.IndexFlatL2(64)
vecs = np.random.rand(10, 64).astype('float32')
index.add(vecs)
D, I = index.search(vecs[:1], 3)
assert I.shape == (1, 3)
print('  \033[0;32m✓\033[0m faiss — index add/search ok')
"
    $PYTHON -c "
import spacy
nlp = spacy.load('${SPACY_MODEL}')
assert nlp.meta.get('version') == '${SPACY_MODEL_VERSION}'
doc = nlp('The Supreme Court ruled in favor of the plaintiff.')
ents = [e.label_ for e in doc.ents]
print(f'  \033[0;32m✓\033[0m spacy {spacy.__version__} | model ${SPACY_MODEL_VERSION} | entities: {ents}')
"
}

run_gpu_smoke_tests() {
    if [ "${SKIP_GPU:-0}" = "1" ]; then
        echo -e " ${C_YELLOW}SKIP_GPU=1${C_RESET} — skipping GPU smoke tests"
        step_end "run_gpu_smoke_tests" "SKIP"
        return
    fi
    echo " Running GPU smoke tests — enforcing TARGET_* constraints..."
    [ "$HARDWARE_MATCH" = "false" ] && echo -e " ${C_YELLOW}WARNING${C_RESET}: hardware mismatches flagged above — hard-failing now."

    $PYTHON -c "
import torch

TARGET_GPU_NAME    = '${TARGET_GPU_NAME}'
TARGET_GPU_COUNT   = ${TARGET_GPU_COUNT}
TARGET_CAP         = (${TARGET_COMPUTE_CAP_MAJOR}, ${TARGET_COMPUTE_CAP_MINOR})
TARGET_VRAM_GB_MIN = ${TARGET_VRAM_GB_MIN}
TARGET_TORCH_CUDA  = '${TARGET_TORCH_CUDA_RUNTIME}'

assert torch.cuda.is_available(), 'CUDA not available'
assert torch.version.cuda.startswith(TARGET_TORCH_CUDA), \
    f'torch CUDA {torch.version.cuda} != target {TARGET_TORCH_CUDA}'
n = torch.cuda.device_count()
assert n >= TARGET_GPU_COUNT, f'Expected {TARGET_GPU_COUNT}x {TARGET_GPU_NAME}, got {n}'
for i in range(n):
    name    = torch.cuda.get_device_name(i)
    cap     = torch.cuda.get_device_capability(i)
    vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
    assert TARGET_GPU_NAME in name, f'GPU {i}: expected {TARGET_GPU_NAME}, got {name}'
    assert cap >= TARGET_CAP, f'GPU {i}: cap {cap} < {TARGET_CAP}'
    assert vram_gb >= TARGET_VRAM_GB_MIN, f'GPU {i}: {vram_gb:.1f}GB < {TARGET_VRAM_GB_MIN}GB'
    print(f'  \033[0;32m✓\033[0m GPU[{i}] {name} | cap {cap} | {vram_gb:.1f}GB')
t = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
assert torch.allclose(t.mean().cpu(), torch.tensor(2.0))
print(f'  \033[0;32m✓\033[0m CUDA {torch.version.cuda} — tensor op on cuda:0 ok')
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
    'timestamp':       datetime.utcnow().isoformat() + 'Z',
    'git_sha':         '${GIT_SHA}',
    'git_branch':      '${GIT_BRANCH}',
    'git_dirty_files': int('${GIT_DIRTY}'),
    'uv_lock_sha256':  '${UVLOCK_SHA256}',
    'python':          sys.version,
    'platform':        platform.platform(),
    'repro_env': {
        'PYTHONHASHSEED':          os.environ.get('PYTHONHASHSEED'),
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'TOKENIZERS_PARALLELISM':  os.environ.get('TOKENIZERS_PARALLELISM'),
        'RANDOM_SEED':             os.environ.get('RANDOM_SEED'),
    },
    'numerical_stability': {
        'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark':                  torch.backends.cudnn.benchmark,
        'cudnn_deterministic':              torch.backends.cudnn.deterministic,
    },
    'parity_module': 'src/repro.py',
    'parity_usage':  'from src.repro import configure; configure()',
    'hardware_target': {
        'gpu_name':           '${TARGET_GPU_NAME}',
        'gpu_count':          ${TARGET_GPU_COUNT},
        'compute_cap_min':    [${TARGET_COMPUTE_CAP_MAJOR}, ${TARGET_COMPUTE_CAP_MINOR}],
        'vram_gb_min':        ${TARGET_VRAM_GB_MIN},
        'torch_cuda_runtime': '${TARGET_TORCH_CUDA_RUNTIME}',
        'driver_cuda':        '${TARGET_DRIVER_CUDA}',
        'python_version':     '${TARGET_PYTHON_VERSION}',
        'min_disk_gb':        ${TARGET_MIN_DISK_GB},
    },
    'hardware_detected': {
        'gpu_name':           '${DETECTED_GPU_NAME}',
        'gpu_count':          '${DETECTED_GPU_COUNT}',
        'torch_cuda_runtime': '${DETECTED_TORCH_CUDA}',
        'driver_cuda':        '${DETECTED_DRIVER_CUDA}',
        'cudnn':              '${DETECTED_CUDNN}',
        'hardware_match':     '${HARDWARE_MATCH}',
    },
    'torch':               torch.__version__,
    'torch_cuda_runtime':  torch.version.cuda,
    'driver_cuda':         get_driver_cuda(),
    'driver_version':      get_driver_version(),
    'cudnn':               str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
    'cuda_toolkit_nvcc':   get_nvcc_version(),
    'cuda_available':      torch.cuda.is_available(),
    'gpu_count':           torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'gpus':                gpus,
    'transformers':        transformers.__version__,
    'spacy':               spacy.__version__,
    'spacy_model':         '${SPACY_MODEL}',
    'spacy_model_version': nlp.meta.get('version'),
    'spacy_model_sha256':  '${SPACY_MODEL_SHA256}',
    'faiss':               get_faiss_version(),
    'installed_packages':  get_installed_versions([
        'torch', 'transformers', 'datasets', 'faiss-cpu', 'spacy',
        'scikit-learn', 'numpy', 'pandas', 'langchain', 'gensim',
        'sentence-transformers', 'networkx', 'pytest', 'mypy', 'hypothesis',
    ]),
}
with open('logs/environment_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print('  \033[0;32m✓\033[0m manifest → logs/environment_manifest.json')
print(f'  git: ${GIT_SHA} | branch: ${GIT_BRANCH} | dirty: ${GIT_DIRTY}')
print(f'  uv.lock sha256: ${UVLOCK_SHA256}')
print(f'  hardware_match: ${HARDWARE_MATCH} | detected: ${DETECTED_GPU_COUNT}x ${DETECTED_GPU_NAME}')
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
assert 'hallucination-legal-rag' in kernels, 'Kernel not found in kernelspec list'
spec = kernels['hallucination-legal-rag']
print(f'  \033[0;32m✓\033[0m kernel: {spec[\"spec\"][\"display_name\"]}')
print(f'    path: {spec[\"resource_dir\"]}')
" || echo -e "  ${C_YELLOW}WARNING${C_RESET}: Could not verify kernel via venv jupyter"
}

verify_tests() {
    echo " Verifying test suite..."
    UNIT_COUNT=$($UV run pytest tests/ --co -q -m unit 2>/dev/null | grep -c "^tests/" || true)

    if [ "${UNIT_COUNT}" -gt 0 ]; then
        echo " Found ${UNIT_COUNT} unit tests — running as verification gate..."
        $UV run pytest tests/ -m unit -q --tb=short && \
            echo -e "  ${C_GREEN}✓${C_RESET} Unit tests passed" || \
            { echo -e "  ${C_RED}ERROR${C_RESET}: Unit tests failed"; exit 1; }
    else
        echo " No unit tests yet — falling back to collection check..."
        $UV run pytest tests/ --co -q 2>/dev/null && \
            echo -e "  ${C_GREEN}✓${C_RESET} Test collection ok" || \
            echo -e "  ${C_YELLOW}WARNING${C_RESET}: Test collection failed — check src/ imports"
    fi
}

# ===========================================================================
# Main execution
# ===========================================================================

if [ -n "${STEP:-}" ]; then
    echo -e "${C_BOLD}${C_CYAN}Single-step mode: STEP=${STEP}${C_RESET}"
    echo -e "${C_DIM}Only '${STEP}' will execute. All other steps skipped.${C_RESET}"
    echo ""
fi

echo -e "${C_BOLD}============================================================${C_RESET}"
echo -e "${C_BOLD} cs1090b_HallucinationLegalRAGChatbots — Environment Bootstrap${C_RESET}"
echo -e " Target: ${TARGET_GPU_COUNT}x NVIDIA ${TARGET_GPU_NAME} | Python ${TARGET_PYTHON_VERSION} | torch 2.0.1+cu117"
echo -e " Driver CUDA: ${TARGET_DRIVER_CUDA} (forward-compat) | torch runtime: ${TARGET_TORCH_CUDA_RUNTIME}"
echo -e " spaCy model: ${SPACY_MODEL} ${SPACY_MODEL_VERSION} (pinned + checksummed)"
echo -e " Repro:        PYTHONHASHSEED=${REPRO_PYTHONHASHSEED} | CUBLAS=${REPRO_CUBLAS_CFG} | RANDOM_SEED=${RANDOM_SEED}"
echo -e " Fail-fast:    preflight_fast_checks() runs FIRST"
echo -e " Single step:  STEP=<fn_name> bash setup.sh"
echo -e "${C_BOLD}============================================================${C_RESET}"

run_step preflight_fast_checks   # FIRST: cheap gates
run_step check_uv
run_step check_lockfile
run_step log_gpu                  # pre-venv: driver-level only
run_step ensure_venv              # expensive: build venv
run_step verify_python
run_step sync_dependencies        # expensive: package sync
run_step check_dependency_drift
run_step detect_hardware          # post-venv: torch-level, DETECTED_* vars
run_step write_repro_env
run_step write_repro_module
run_step verify_numerical_stability
run_step download_nlp_models
run_step run_env_smoke_tests
run_step run_gpu_smoke_tests      # hard gate
run_step write_manifest
run_step register_kernel
run_step verify_tests

print_summary

echo -e "${C_BOLD}============================================================${C_RESET}"
echo -e "${C_GREEN}${C_BOLD} Environment ready.${C_RESET}"
echo -e " Activate:   source .venv/bin/activate"
echo -e " Kernel:     HallucinationLegalRAG (${TARGET_PYTHON_VERSION})"
echo -e " Manifest:   logs/environment_manifest.json"
echo -e " Repro env:  source .env  (or dotenv.load_dotenv() in notebook)"
echo -e " Parity:     from src.repro import configure; configure()"
echo -e " Single step: STEP=<fn_name> bash setup.sh"
echo -e "   Available: preflight_fast_checks | check_uv | check_lockfile | log_gpu"
echo -e "              ensure_venv | verify_python | sync_dependencies | check_dependency_drift"
echo -e "              detect_hardware | write_repro_env | write_repro_module"
echo -e "              verify_numerical_stability | download_nlp_models | run_env_smoke_tests"
echo -e "              run_gpu_smoke_tests | write_manifest | register_kernel | verify_tests"
echo -e " CPU mode:   SKIP_GPU=1 bash setup.sh"
echo -e " Dry run:    DRY_RUN=1 bash setup.sh"
echo -e " Offline:    OFFLINE=1 bash setup.sh"
echo -e " Seed expt:  Edit RANDOM_SEED in setup.sh, re-run, commit .env + src/repro.py"
echo -e "${C_BOLD}============================================================${C_RESET}"
