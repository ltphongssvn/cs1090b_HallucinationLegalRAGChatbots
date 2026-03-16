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
#
# Execution order (fail-fast design):
#   1. preflight_fast_checks()  — cheap gates (seconds)
#   2. check_uv / check_lockfile
#   3. log_gpu                  — pre-venv driver-level info
#   4. ensure_venv              — expensive: build venv
#   5. verify_python / sync_dependencies
#   6. check_dependency_drift
#   7. detect_hardware          — post-venv torch-level detection
#   8. ... rest of setup
#
# Function decomposition (SRP): each function has one responsibility.
# Private helpers prefixed _. Public steps are thin orchestrators.
#
# Defensive validation:
#   _require_uv()    — asserts $UV is set before any uv invocation
#   _require_python()— asserts $PYTHON exists before any venv invocation
#   _require_hardware_detected() — asserts DETECTED_* populated before use
#   _require_project_root()      — asserts PROJECT_ROOT contains pyproject.toml
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
# Hardware target constants
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
# Reproducibility constants
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
# Sentinel value "UNDETECTED" is used instead of "" so that defensive guards
# can distinguish "not yet detected" from "detected as empty/N/A".
DETECTED_GPU_NAME="UNDETECTED"
DETECTED_GPU_COUNT="UNDETECTED"
DETECTED_DRIVER_CUDA="UNDETECTED"
DETECTED_TORCH_CUDA="UNDETECTED"
DETECTED_CUDNN="UNDETECTED"
HARDWARE_MATCH="true"

# $UV is set by check_uv() — all other functions must call _require_uv() first
UV=""

# ===========================================================================
# Ergonomics: color output, per-step timing, summary table
# ===========================================================================
if [ -t 1 ]; then
    C_RESET="\033[0m"; C_BOLD="\033[1m"; C_GREEN="\033[0;32m"
    C_YELLOW="\033[0;33m"; C_RED="\033[0;31m"; C_CYAN="\033[0;36m"; C_DIM="\033[2m"
else
    C_RESET=""; C_BOLD=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_CYAN=""; C_DIM=""
fi

SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
_step_start_time=0

step_begin() { _step_start_time=$(date +%s); echo -e "${C_BOLD}${C_CYAN}▶ $1${C_RESET}"; }

step_end() {
    local name="$1" status="${2:-PASS}" duration=$(( $(date +%s) - _step_start_time ))
    SUMMARY_STEPS+=("$name"); SUMMARY_DURATION+=("${duration}s")
    case "$status" in
        PASS) SUMMARY_STATUS+=("${C_GREEN}PASS${C_RESET}") ;;
        WARN) SUMMARY_STATUS+=("${C_YELLOW}WARN${C_RESET}") ;;
        SKIP) SUMMARY_STATUS+=("${C_DIM}SKIP${C_RESET}") ;;
        *)    SUMMARY_STATUS+=("${C_RED}FAIL${C_RESET}") ;;
    esac
    echo -e "  ${C_DIM}(${duration}s)${C_RESET}"
}

print_summary() {
    echo -e "\n${C_BOLD}============================================================${C_RESET}"
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

run_step() {
    local fn="$1"; shift
    if [ -n "${STEP:-}" ] && [ "$STEP" != "$fn" ]; then
        SUMMARY_STEPS+=("$fn"); SUMMARY_STATUS+=("${C_DIM}SKIP${C_RESET}"); SUMMARY_DURATION+=("-")
        return 0
    fi
    step_begin "$fn"
    "$fn" "$@"
    step_end "$fn" "PASS"
}

trap 'echo -e "${C_RED}ERROR at line $LINENO${C_RESET}"; print_summary' ERR

# ===========================================================================
# Defensive guards — called at the top of any function that requires a
# prerequisite to be satisfied. Produce actionable error messages.
# ===========================================================================

_require_project_root() {
    # Guards: PROJECT_ROOT is actually the project root (contains pyproject.toml)
    # Catches: wrong working directory, symlink issues, running from subdir
    if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
        echo -e "${C_RED}ERROR${C_RESET}: PROJECT_ROOT='$PROJECT_ROOT' does not contain pyproject.toml."
        echo "       Are you running setup.sh from the correct directory?"
        echo "       Expected: cs1090b_HallucinationLegalRAGChatbots/"
        exit 1
    fi
}

_require_uv() {
    # Guards: $UV is set and points to a working uv binary.
    # Catches: functions called via STEP= without check_uv() having run first.
    if [ -z "${UV:-}" ]; then
        # Attempt auto-resolve before failing — tolerates STEP= usage patterns
        if command -v uv &>/dev/null; then
            UV=$(command -v uv)
        elif [ -x "$HOME/.local/bin/uv" ]; then
            UV="$HOME/.local/bin/uv"
        else
            echo -e "${C_RED}ERROR${C_RESET}: \$UV is not set and uv binary not found."
            echo "       Run check_uv first, or install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
    fi
    if ! "$UV" --version &>/dev/null; then
        echo -e "${C_RED}ERROR${C_RESET}: \$UV='$UV' exists but does not execute. Check permissions."
        exit 1
    fi
}

_require_python() {
    # Guards: $PYTHON venv binary exists and is the correct version.
    # Catches: functions called via STEP= before ensure_venv() has run,
    #          or after venv was accidentally deleted.
    if [ ! -x "$PYTHON" ]; then
        echo -e "${C_RED}ERROR${C_RESET}: venv Python not found at '$PYTHON'."
        echo "       Run ensure_venv first: STEP=ensure_venv bash setup.sh"
        echo "       Or run full setup:     bash setup.sh"
        exit 1
    fi
    local PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"
    if ! "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info[:3] == (${PYVER_TUPLE}) else 1)" 2>/dev/null; then
        local actual; actual=$("$PYTHON" --version 2>&1)
        echo -e "${C_RED}ERROR${C_RESET}: venv Python version mismatch."
        echo "       Expected: Python ${TARGET_PYTHON_VERSION}"
        echo "       Actual:   ${actual}"
        echo "       Fix: rm -rf .venv && bash setup.sh"
        exit 1
    fi
}

_require_hardware_detected() {
    # Guards: DETECTED_* vars are populated (detect_hardware() has run).
    # Catches: write_manifest() or run_gpu_smoke_tests() called via STEP=
    #          before detect_hardware(), which would silently embed sentinel
    #          values into the manifest or produce misleading GPU check output.
    if [ "${DETECTED_GPU_COUNT}" = "UNDETECTED" ] || \
       [ "${DETECTED_TORCH_CUDA}" = "UNDETECTED" ] || \
       [ "${DETECTED_GPU_NAME}" = "UNDETECTED" ]; then
        echo -e "${C_YELLOW}WARNING${C_RESET}: Hardware not yet detected (DETECTED_* = UNDETECTED)."
        echo "       Running detect_hardware() now to populate DETECTED_* vars..."
        detect_hardware
    fi
}

_require_repro_env() {
    # Guards: .env exists and RANDOM_SEED is exported in the current process.
    # Catches: verify_numerical_stability or write_manifest called before
    #          write_repro_env(), which would cause RANDOM_SEED checks to fail.
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        echo -e "${C_RED}ERROR${C_RESET}: .env not found at $PROJECT_ROOT/.env."
        echo "       Run write_repro_env first: STEP=write_repro_env bash setup.sh"
        exit 1
    fi
    if [ -z "${RANDOM_SEED:-}" ]; then
        echo -e "${C_RED}ERROR${C_RESET}: RANDOM_SEED is not set in the current process."
        echo "       This should never happen — RANDOM_SEED is a setup.sh constant."
        exit 1
    fi
}

# ===========================================================================
# Private helpers — single responsibility, not run_step targets
# ===========================================================================

_check_disk_space() {
    local free_gb msg
    free_gb=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
    if [ "${free_gb:-0}" -lt "$TARGET_MIN_DISK_GB" ]; then
        msg="Disk: only ${free_gb}GB free on $PROJECT_ROOT, need ${TARGET_MIN_DISK_GB}GB"
        echo -e "  ${C_RED}✗${C_RESET} $msg"
        echo "$msg"; return 1
    fi
    echo -e "  ${C_GREEN}✓${C_RESET} disk: ${free_gb}GB free >= ${TARGET_MIN_DISK_GB}GB"
}

_check_nvidia_smi_present() {
    if ! command -v nvidia-smi &>/dev/null; then
        local msg="nvidia-smi not found — not a GPU node. Request a GPU allocation."
        echo -e "  ${C_RED}✗${C_RESET} $msg"; echo "$msg"; return 1
    fi
    echo -e "  ${C_GREEN}✓${C_RESET} nvidia-smi: present"
}

_check_gpu_count_smi() {
    local count
    count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | xargs)
    if [ "${count:-0}" -lt "$TARGET_GPU_COUNT" ]; then
        local msg="GPU count (nvidia-smi): detected ${count}, need ${TARGET_GPU_COUNT}. Check CUDA_VISIBLE_DEVICES."
        echo -e "  ${C_RED}✗${C_RESET} $msg"; echo "$msg"; return 1
    fi
    echo -e "  ${C_GREEN}✓${C_RESET} nvidia-smi GPU count: ${count} >= ${TARGET_GPU_COUNT}"
}

_check_gpu_name_smi() {
    local names
    names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')
    if ! echo "$names" | grep -q "$TARGET_GPU_NAME"; then
        local msg="GPU name (nvidia-smi): detected '${names}', expected '${TARGET_GPU_NAME}'. Wrong node."
        echo -e "  ${C_RED}✗${C_RESET} $msg"; echo "$msg"; return 1
    fi
    echo -e "  ${C_GREEN}✓${C_RESET} nvidia-smi GPU name: contains '${TARGET_GPU_NAME}'"
}

_check_driver_cuda_smi() {
    local driver_cuda
    driver_cuda=$(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}')
    if [ "${driver_cuda}" != "${TARGET_DRIVER_CUDA}" ]; then
        echo -e "  ${C_YELLOW}WARNING${C_RESET}: driver CUDA ${driver_cuda} != target ${TARGET_DRIVER_CUDA}. Update TARGET_DRIVER_CUDA if intentional."
    else
        echo -e "  ${C_GREEN}✓${C_RESET} driver CUDA: ${driver_cuda}"
    fi
}

_check_lockfile_present() {
    local ok=true
    if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
        echo -e "  ${C_RED}✗${C_RESET} pyproject.toml not found"
        echo "pyproject.toml not found at $PROJECT_ROOT"; ok=false
    fi
    if [ ! -f "$PROJECT_ROOT/uv.lock" ]; then
        echo -e "  ${C_RED}✗${C_RESET} uv.lock not found"
        echo "uv.lock not found — run: uv lock && git add uv.lock && git commit"; ok=false
    fi
    [ "$ok" = "true" ] && echo -e "  ${C_GREEN}✓${C_RESET} pyproject.toml + uv.lock: present"
    [ "$ok" = "false" ] && return 1
}

_check_uv_present() {
    if ! command -v uv &>/dev/null && ! command -v ~/.local/bin/uv &>/dev/null; then
        local msg="uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo -e "  ${C_RED}✗${C_RESET} $msg"; echo "$msg"; return 1
    fi
    local uv_bin; uv_bin=$(command -v uv 2>/dev/null || echo ~/.local/bin/uv)
    echo -e "  ${C_GREEN}✓${C_RESET} uv: $($uv_bin --version)"
}

_query_torch_hardware() {
    _require_python
    $PYTHON -c "
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
"
}

_parse_detected_hardware() {
    local hw_json="$1"
    # Validate JSON is non-empty before parsing
    if [ -z "$hw_json" ]; then
        echo -e "${C_RED}ERROR${C_RESET}: _query_torch_hardware returned empty output."
        echo "       torch may not be installed or venv may be broken."
        exit 1
    fi
    DETECTED_GPU_COUNT=$(echo "$hw_json" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['gpu_count'])")
    DETECTED_TORCH_CUDA=$(echo "$hw_json" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['torch_cuda'])")
    DETECTED_CUDNN=$(echo "$hw_json" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['cudnn'])")
    DETECTED_GPU_NAME=$(echo "$hw_json" | $PYTHON -c "
import json,sys
d=json.load(sys.stdin)
names=list({g['name'] for g in d['gpus']})
print(', '.join(names) if names else 'N/A')
")
    # Validate parsed values are non-empty — guard against silent parse failures
    for var_name in DETECTED_GPU_COUNT DETECTED_TORCH_CUDA DETECTED_CUDNN DETECTED_GPU_NAME; do
        local val="${!var_name}"
        if [ -z "$val" ] || [ "$val" = "UNDETECTED" ]; then
            echo -e "${C_RED}ERROR${C_RESET}: _parse_detected_hardware: ${var_name} is empty after parse."
            echo "       Raw hardware JSON: $hw_json"
            exit 1
        fi
    done
    if command -v nvidia-smi &>/dev/null; then
        DETECTED_DRIVER_CUDA=$(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}')
        [ -z "$DETECTED_DRIVER_CUDA" ] && DETECTED_DRIVER_CUDA="parse-failed"
    else
        DETECTED_DRIVER_CUDA="nvidia-smi-not-found"
    fi
}

_print_hardware_table() {
    local hw_json="$1"
    echo " --- Detected hardware vs targets ---"
    printf "  %-20s %-30s %s\n" "Property" "Detected" "Target"
    printf "  %-20s %-30s %s\n" "--------" "--------" "------"
    printf "  %-20s %-30s %s\n" "GPU name"    "'${DETECTED_GPU_NAME}'"  "contains '${TARGET_GPU_NAME}'"
    printf "  %-20s %-30s %s\n" "GPU count"   "${DETECTED_GPU_COUNT}"   "${TARGET_GPU_COUNT}"
    printf "  %-20s %-30s %s\n" "torch CUDA"  "${DETECTED_TORCH_CUDA}"  "${TARGET_TORCH_CUDA_RUNTIME}"
    printf "  %-20s %-30s %s\n" "driver CUDA" "${DETECTED_DRIVER_CUDA}" "${TARGET_DRIVER_CUDA}"
    printf "  %-20s %-30s\n"    "cuDNN"        "${DETECTED_CUDNN}"
    echo "$hw_json" | $PYTHON -c "
import json,sys
for g in json.load(sys.stdin)['gpus']:
    print(f\"  GPU[{g['index']}]: {g['name']} | {g['vram_gb']}GB | cap {g['compute_capability']}\")
"
}

_compare_hardware_to_targets() {
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
        echo -e " ${C_YELLOW}WARNING${C_RESET}: Mismatches detected — run_gpu_smoke_tests() will hard-fail."
    fi
}

_assert_cuda_available() {
    _require_python
    $PYTHON -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — wrong torch wheel'
assert torch.version.cuda.startswith('${TARGET_TORCH_CUDA_RUNTIME}'), \
    f'torch CUDA {torch.version.cuda} != target ${TARGET_TORCH_CUDA_RUNTIME}'
print(f'  \033[0;32m✓\033[0m CUDA available | torch runtime {torch.version.cuda}')
"
}

_assert_gpu_count() {
    _require_python
    $PYTHON -c "
import torch
n = torch.cuda.device_count()
assert n >= ${TARGET_GPU_COUNT}, f'Expected ${TARGET_GPU_COUNT}x ${TARGET_GPU_NAME}, got {n}'
print(f'  \033[0;32m✓\033[0m GPU count: {n} >= ${TARGET_GPU_COUNT}')
"
}

_assert_per_gpu_specs() {
    _require_python
    $PYTHON -c "
import torch
TARGET_GPU_NAME    = '${TARGET_GPU_NAME}'
TARGET_CAP         = (${TARGET_COMPUTE_CAP_MAJOR}, ${TARGET_COMPUTE_CAP_MINOR})
TARGET_VRAM_GB_MIN = ${TARGET_VRAM_GB_MIN}
for i in range(torch.cuda.device_count()):
    name    = torch.cuda.get_device_name(i)
    cap     = torch.cuda.get_device_capability(i)
    vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
    assert TARGET_GPU_NAME in name, f'GPU {i}: expected {TARGET_GPU_NAME}, got {name}'
    assert cap >= TARGET_CAP, f'GPU {i}: cap {cap} < {TARGET_CAP}'
    assert vram_gb >= TARGET_VRAM_GB_MIN, f'GPU {i}: {vram_gb:.1f}GB < {TARGET_VRAM_GB_MIN}GB'
    print(f'  \033[0;32m✓\033[0m GPU[{i}] {name} | cap {cap} | {vram_gb:.1f}GB')
"
}

_assert_gpu_tensor_op() {
    _require_python
    $PYTHON -c "
import torch
t = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
assert t.device.type == 'cuda'
assert torch.allclose(t.mean().cpu(), torch.tensor(2.0))
print(f'  \033[0;32m✓\033[0m CUDA ${TARGET_TORCH_CUDA_RUNTIME} — tensor round-trip on cuda:0 ok')
"
}

_collect_manifest_data() {
    _require_python
    _require_hardware_detected
    local git_sha="$1" git_branch="$2" git_dirty="$3" uvlock_sha256="$4"
    # Validate git/lock args are non-empty before embedding in manifest
    for arg_name in git_sha git_branch git_dirty uvlock_sha256; do
        local val="${!arg_name}"
        if [ -z "$val" ]; then
            echo -e "${C_YELLOW}WARNING${C_RESET}: _collect_manifest_data: ${arg_name} is empty — using 'unknown'"
            eval "${arg_name}=unknown"
        fi
    done
    $PYTHON -c "
import json, torch, transformers, spacy, sys, platform, subprocess, os
import importlib.metadata as meta
from datetime import datetime

def _get_nvcc():
    try:
        r = subprocess.run(['nvcc','--version'], capture_output=True, text=True)
        for l in r.stdout.splitlines():
            if 'release' in l: return l.strip()
    except FileNotFoundError: return 'nvcc not found'
    return 'unknown'

def _get_driver_cuda():
    try:
        r = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        for l in r.stdout.splitlines():
            if 'CUDA Version' in l: return l.strip().split()[-1]
    except FileNotFoundError: return 'nvidia-smi not found'
    return 'unknown'

def _get_driver_version():
    try:
        r = subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                           capture_output=True, text=True)
        return r.stdout.strip().splitlines()[0] if r.stdout.strip() else 'unknown'
    except FileNotFoundError: return 'nvidia-smi not found'

def _get_faiss():
    try:
        import faiss
        return getattr(faiss,'__version__','installed — version attr unavailable')
    except ImportError: return 'not installed'

def _get_pkgs(pkgs):
    out = {}
    for p in pkgs:
        try: out[p] = meta.version(p)
        except meta.PackageNotFoundError: out[p] = 'not installed'
    return out

gpus = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({'index':i,'name':props.name,
                     'vram_gb':round(props.total_memory/1e9,2),
                     'compute_capability':list(torch.cuda.get_device_capability(i))})

nlp = spacy.load('${SPACY_MODEL}')
data = {
    'timestamp':       datetime.utcnow().isoformat()+'Z',
    'git_sha':         '${git_sha}',
    'git_branch':      '${git_branch}',
    'git_dirty_files': int('${git_dirty}') if '${git_dirty}'.isdigit() else -1,
    'uv_lock_sha256':  '${uvlock_sha256}',
    'python':          sys.version,
    'platform':        platform.platform(),
    'repro_env': {
        'PYTHONHASHSEED':          os.environ.get('PYTHONHASHSEED', 'NOT SET'),
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'NOT SET'),
        'TOKENIZERS_PARALLELISM':  os.environ.get('TOKENIZERS_PARALLELISM', 'NOT SET'),
        'RANDOM_SEED':             os.environ.get('RANDOM_SEED', 'NOT SET'),
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
        'compute_cap_min':    [${TARGET_COMPUTE_CAP_MAJOR},${TARGET_COMPUTE_CAP_MINOR}],
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
    'driver_cuda':         _get_driver_cuda(),
    'driver_version':      _get_driver_version(),
    'cudnn':               str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
    'cuda_toolkit_nvcc':   _get_nvcc(),
    'cuda_available':      torch.cuda.is_available(),
    'gpu_count':           torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'gpus':                gpus,
    'transformers':        transformers.__version__,
    'spacy':               spacy.__version__,
    'spacy_model':         '${SPACY_MODEL}',
    'spacy_model_version': nlp.meta.get('version'),
    'spacy_model_sha256':  '${SPACY_MODEL_SHA256}',
    'faiss':               _get_faiss(),
    'installed_packages':  _get_pkgs([
        'torch','transformers','datasets','faiss-cpu','spacy',
        'scikit-learn','numpy','pandas','langchain','gensim',
        'sentence-transformers','networkx','pytest','mypy','hypothesis',
    ]),
}
print(json.dumps(data))
"
}

_write_manifest_file() {
    local manifest_json="$1"
    # Defensive: validate JSON before writing to disk
    if [ -z "$manifest_json" ]; then
        echo -e "${C_RED}ERROR${C_RESET}: _write_manifest_file: manifest JSON is empty."
        exit 1
    fi
    # Validate it's parseable JSON before writing
    echo "$manifest_json" | $PYTHON -c "import json,sys; json.load(sys.stdin)" 2>/dev/null || {
        echo -e "${C_RED}ERROR${C_RESET}: _write_manifest_file: manifest JSON is malformed."
        echo "       First 200 chars: ${manifest_json:0:200}"
        exit 1
    }
    mkdir -p "$PROJECT_ROOT/logs"
    echo "$manifest_json" | $PYTHON -c "
import json, sys
data = json.load(sys.stdin)
with open('logs/environment_manifest.json', 'w') as f:
    json.dump(data, f, indent=2)
print('  \033[0;32m✓\033[0m manifest → logs/environment_manifest.json')
"
}

# ===========================================================================
# Public step functions
# ===========================================================================

preflight_fast_checks() {
    # Defensive: validate PROJECT_ROOT before any file checks
    _require_project_root
    echo " Running preflight fast checks (pre-venv, seconds)..."
    # Collect failures into array — single-pass, no double-invocation of helpers
    local failures=()
    local output

    output=$(_check_disk_space 2>&1);       echo "$output" | grep -v "^Disk:" | cat
    echo "$output" | grep "^Disk:" | grep -q "." && failures+=("$(echo "$output" | grep '^Disk:')")

    output=$(_check_uv_present 2>&1);       echo "$output" | grep -v "^uv not" | cat
    echo "$output" | grep "^uv not" | grep -q "." && failures+=("$(echo "$output" | grep '^uv not')")

    output=$(_check_lockfile_present 2>&1); echo "$output" | grep -v "^pyproject\|^uv.lock not" | cat
    echo "$output" | grep "^pyproject\|^uv.lock not" | while read -r line; do failures+=("$line"); done

    output=$(_check_nvidia_smi_present 2>&1); echo "$output" | grep -v "^nvidia-smi not" | cat
    echo "$output" | grep "^nvidia-smi not" | grep -q "." && failures+=("$(echo "$output" | grep '^nvidia-smi not')")

    if command -v nvidia-smi &>/dev/null; then
        output=$(_check_gpu_count_smi 2>&1); echo "$output" | grep -v "^GPU count" | cat
        echo "$output" | grep "^GPU count" | grep -q "detected" && failures+=("$(echo "$output" | grep '^GPU count')")

        output=$(_check_gpu_name_smi 2>&1);  echo "$output" | grep -v "^GPU name" | cat
        echo "$output" | grep "^GPU name" | grep -q "detected" && failures+=("$(echo "$output" | grep '^GPU name')")

        _check_driver_cuda_smi
    fi

    if [ ${#failures[@]} -gt 0 ]; then
        echo -e "\n${C_RED}${C_BOLD}PREFLIGHT FAILED — ${#failures[@]} issue(s):${C_RESET}"
        for i in "${!failures[@]}"; do
            echo -e "  ${C_RED}[$((i+1))]${C_RESET} ${failures[$i]}"
        done
        echo -e "\n${C_DIM}Fast checks (pre-venv) save venv build+sync time on wrong nodes.${C_RESET}"
        exit 1
    fi
    echo -e "  ${C_GREEN}✓${C_RESET} Preflight fast checks passed."
}

check_uv() {
    if ! command -v uv &>/dev/null && ! command -v ~/.local/bin/uv &>/dev/null; then
        echo -e "${C_RED}ERROR${C_RESET}: uv not found."; exit 1
    fi
    UV=$(command -v uv 2>/dev/null || echo ~/.local/bin/uv)
    # Validate UV actually works
    if ! "$UV" --version &>/dev/null; then
        echo -e "${C_RED}ERROR${C_RESET}: uv binary at '$UV' does not execute."
        exit 1
    fi
    echo -e "  ${C_GREEN}✓${C_RESET} uv: $($UV --version)"
}

check_lockfile() {
    _require_project_root
    [ ! -f "$PROJECT_ROOT/pyproject.toml" ] && {
        echo -e "${C_RED}ERROR${C_RESET}: pyproject.toml not found."; exit 1
    }
    [ ! -f "$PROJECT_ROOT/uv.lock" ] && {
        echo -e "${C_RED}ERROR${C_RESET}: uv.lock not found."
        echo "       To generate deliberately: uv lock && git add uv.lock && git commit -m 'chore: pin uv.lock'"
        exit 1
    }
    echo -e "  ${C_GREEN}✓${C_RESET} uv.lock sha256: $(sha256sum "$PROJECT_ROOT/uv.lock" | cut -d' ' -f1)"
}

log_gpu() {
    echo " Hardware target: ${TARGET_GPU_COUNT}x NVIDIA ${TARGET_GPU_NAME} | CUDA runtime ${TARGET_TORCH_CUDA_RUNTIME} | driver CUDA ${TARGET_DRIVER_CUDA}"
    if command -v nvidia-smi &>/dev/null; then
        echo " --- nvidia-smi per-GPU summary (pre-venv) ---"
        nvidia-smi --query-gpu=index,name,memory.total,driver_version \
            --format=csv,noheader | while IFS=',' read -r idx name mem drv; do
            echo "  GPU $idx:$(echo "$name"|xargs) | VRAM:$(echo "$mem"|xargs) | Driver:$(echo "$drv"|xargs)"
        done
        echo -e " ${C_DIM}NOTE: torch wheel compiled against CUDA ${TARGET_TORCH_CUDA_RUNTIME}. Driver CUDA $(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}') is forward-compatible.${C_RESET}"
    else
        echo -e " ${C_YELLOW}WARNING${C_RESET}: nvidia-smi not found"
    fi
    command -v nvcc &>/dev/null && \
        echo " CUDA toolkit (nvcc): $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')" || \
        echo -e " ${C_YELLOW}WARNING${C_RESET}: nvcc not on PATH"
}

detect_hardware() {
    _require_python
    echo " Detecting hardware (post-venv, torch-level)..."
    local hw_json
    hw_json=$(_query_torch_hardware)
    _parse_detected_hardware "$hw_json"
    _print_hardware_table "$hw_json"
    _compare_hardware_to_targets
    [ "$HARDWARE_MATCH" = "false" ] && step_end "detect_hardware" "WARN" && return
}

ensure_venv() {
    _require_uv
    local PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"
    if [ -f "$PYTHON" ] && $PYTHON -c "import sys; sys.exit(0 if sys.version_info[:3] == (${PYVER_TUPLE}) else 1)" 2>/dev/null; then
        echo -e "  ${C_GREEN}✓${C_RESET} .venv already exists with Python ${TARGET_PYTHON_VERSION}"
        return
    fi
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        echo -e "  ${C_YELLOW}WARNING${C_RESET}: .venv exists but wrong Python version — will remove."
        echo "         Contents: $(du -sh "$PROJECT_ROOT/.venv" 2>/dev/null | cut -f1) on disk"
        [ "${DRY_RUN:-0}" = "1" ] && echo " DRY_RUN=1 — skipping removal. Exiting." && exit 0
        echo "         Aborting in 5 seconds — Ctrl+C to cancel..."; sleep 5
        rm -rf "$PROJECT_ROOT/.venv"
    fi
    echo " Creating .venv with Python ${TARGET_PYTHON_VERSION}..."
    "$UV" venv .venv --python "${TARGET_PYTHON_VERSION}" --seed
}

verify_python() {
    _require_python
    local PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"
    $PYTHON -c "import sys; assert sys.version_info[:3] == (${PYVER_TUPLE}), f'Expected ${TARGET_PYTHON_VERSION} got {sys.version}'"
    echo -e "  ${C_GREEN}✓${C_RESET} Python: $($PYTHON --version)"
    echo "  Executable: $($PYTHON -c 'import sys; print(sys.executable)')"
}

sync_dependencies() {
    _require_uv
    _require_python
    echo " Syncing dependencies from uv.lock (--frozen)..."
    # --dev kept explicit: documents intent, guards against uv default changes,
    # signals dev tools (pytest, mypy, hypothesis) are required.
    "$UV" sync --frozen --dev
    echo -e "  ${C_GREEN}✓${C_RESET} Dependencies synced"
}

check_dependency_drift() {
    _require_uv
    _require_python
    echo " Checking for dependency drift..."
    if [ "$PROJECT_ROOT/pyproject.toml" -nt "$PROJECT_ROOT/uv.lock" ]; then
        echo -e "${C_RED}ERROR${C_RESET}: pyproject.toml newer than uv.lock — stale lockfile."
        echo "       Fix: uv lock && git add uv.lock && git commit -m 'chore: regenerate uv.lock'"
        exit 1
    fi
    echo -e "  ${C_GREEN}✓${C_RESET} pyproject.toml vs uv.lock timestamp — ok"

    "$UV" lock --check 2>/dev/null && \
        echo -e "  ${C_GREEN}✓${C_RESET} uv lock --check — consistent" || \
        { echo -e "${C_RED}ERROR${C_RESET}: uv lock --check failed."; exit 1; }

    "$UV" sync --frozen --dev --check 2>/dev/null && \
        echo -e "  ${C_GREEN}✓${C_RESET} uv sync --check — matches uv.lock" || \
        { echo -e "${C_RED}ERROR${C_RESET}: packages diverge from uv.lock. Fix: bash setup.sh"; exit 1; }

    $PYTHON -c "
import importlib.metadata as meta
from packaging.version import Version
required = {
    'torch':('2.0.0','2.0.1+cu117'),'transformers':('4.35.0',None),
    'datasets':('2.16.0',None),'faiss-cpu':('1.7.0',None),
    'spacy':('3.7.0',None),'scikit-learn':('1.5.0',None),
    'numpy':('1.24.0',None),'pandas':('2.2.0',None),
}
drift=[]
for pkg,(min_v,exact_v) in required.items():
    try:
        inst=meta.version(pkg)
        if Version(inst)<Version(min_v): drift.append(f'{pkg}: {inst} < {min_v}')
        elif exact_v and inst!=exact_v: print(f'  WARNING: {pkg} {inst} != expected {exact_v}')
        else: print(f'  \033[0;32m✓\033[0m {pkg:<20} {inst}')
    except meta.PackageNotFoundError: drift.append(f'{pkg}: NOT INSTALLED')
if drift:
    [print(f'  \033[0;31mERROR\033[0m {d}') for d in drift]; raise SystemExit(1)
print('  All critical versions verified')
"
}

write_repro_env() {
    _require_project_root
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
    # Verify .env was actually written
    [ ! -f "$PROJECT_ROOT/.env" ] && {
        echo -e "${C_RED}ERROR${C_RESET}: .env was not created at $PROJECT_ROOT/.env"; exit 1
    }
    _require_python
    $PYTHON -c "
import os
for var,exp in [('PYTHONHASHSEED','${REPRO_PYTHONHASHSEED}'),
                ('CUBLAS_WORKSPACE_CONFIG','${REPRO_CUBLAS_CFG}'),
                ('TOKENIZERS_PARALLELISM','${REPRO_TOKENIZERS_PAR}'),
                ('RANDOM_SEED','${RANDOM_SEED}')]:
    act=os.environ.get(var)
    assert act==exp, f'{var}={act!r} != {exp!r}'
    print(f'  \033[0;32m✓\033[0m {var}={act}')
"
}

write_repro_module() {
    _require_python
    _require_repro_env
    echo " Writing src/repro.py — notebook/CLI parity module..."
    mkdir -p "$PROJECT_ROOT/src"
    cat > "$PROJECT_ROOT/src/repro.py" << PYEOF
# src/repro.py
# Path: cs1090b_HallucinationLegalRAGChatbots/src/repro.py
#
# Canonical reproducibility configuration module.
# Call configure() as the FIRST statement in every notebook Cell 1 and CLI script.
#
# Usage:
#   from src.repro import configure
#   repro_cfg = configure()
#
# RANDOM_SEED injected from setup.sh constants block.
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
    """Load .env. Single responsibility: env var loading only."""
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
    """Apply deterministic torch settings. Single responsibility: flag application only."""
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _EXPECTED_CUBLAS_CFG)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_all(seed: int) -> None:
    """Seed all RNGs. Single responsibility: seeding only."""
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
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
    """Verify all settings are active. Single responsibility: verification only."""
    import torch
    checks = {}
    for var, expected in [
        ("PYTHONHASHSEED",          _EXPECTED_PYTHONHASHSEED),
        ("CUBLAS_WORKSPACE_CONFIG", _EXPECTED_CUBLAS_CFG),
        ("TOKENIZERS_PARALLELISM",  _EXPECTED_TOKENIZERS_PAR),
    ]:
        actual = os.environ.get(var)
        assert actual == expected, f"{var}={actual!r} — expected {expected!r}. Call configure() first."
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
    Thin orchestrator: load → apply → seed → verify.
    Call as the FIRST statement in every notebook Cell 1 and CLI script.
    Returns verified config dict for logging/manifest inclusion.
    """
    _load_dotenv(project_root)
    _apply_torch_flags()
    _seed_all(_RANDOM_SEED)
    cfg = _verify()
    if verbose:
        import torch
        print("  [repro] Reproducibility configured:")
        for k, v in cfg.items():
            print(f"    {k}={v}")
        if torch.cuda.is_available():
            print(f"    torch.cuda.manual_seed_all({_RANDOM_SEED}) → {torch.cuda.device_count()} GPU(s)")
    return cfg
PYEOF
    # Verify file was actually written
    [ ! -f "$PROJECT_ROOT/src/repro.py" ] && {
        echo -e "${C_RED}ERROR${C_RESET}: src/repro.py was not created."; exit 1
    }
    echo -e "  ${C_GREEN}✓${C_RESET} src/repro.py written (RANDOM_SEED=${RANDOM_SEED})"
    $PYTHON -c "
import sys; sys.path.insert(0,'${PROJECT_ROOT}')
from src.repro import configure
cfg = configure(verbose=True)
assert cfg['random_seed'] == ${RANDOM_SEED}, f'random_seed {cfg[\"random_seed\"]} != ${RANDOM_SEED}'
print('  \033[0;32m✓\033[0m src/repro.configure() verified')
"
}

verify_numerical_stability() {
    _require_python
    _require_repro_env
    echo " Verifying numerical/runtime stability..."
    $PYTHON -c "
import os, torch
for name,actual,exp in [
    ('CUBLAS_WORKSPACE_CONFIG', os.environ.get('CUBLAS_WORKSPACE_CONFIG'), '${REPRO_CUBLAS_CFG}'),
    ('PYTHONHASHSEED',          os.environ.get('PYTHONHASHSEED'),          '${REPRO_PYTHONHASHSEED}'),
    ('TOKENIZERS_PARALLELISM',  os.environ.get('TOKENIZERS_PARALLELISM'),  '${REPRO_TOKENIZERS_PAR}'),
    ('RANDOM_SEED',             os.environ.get('RANDOM_SEED'),             '${RANDOM_SEED}'),
]:
    assert actual==exp, f'{name}={actual!r} != {exp!r}'
    print(f'  \033[0;32m✓\033[0m {name}={actual}')
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
assert torch.are_deterministic_algorithms_enabled()
assert not torch.backends.cudnn.benchmark
assert torch.backends.cudnn.deterministic
print('  \033[0;32m✓\033[0m torch deterministic flags — ok')
print()
print('  NOTE: Notebooks must call: from src.repro import configure; configure()')
"
}

download_nlp_models() {
    _require_python
    echo " Installing spaCy ${SPACY_MODEL} ${SPACY_MODEL_VERSION} (pinned)..."
    mkdir -p "$SPACY_CACHE_DIR"
    if $PYTHON -c "
import spacy,sys
try:
    nlp=spacy.load('${SPACY_MODEL}')
    sys.exit(0 if nlp.meta.get('version')=='${SPACY_MODEL_VERSION}' else 1)
except OSError: sys.exit(1)
" 2>/dev/null; then
        echo -e "  ${C_GREEN}✓${C_RESET} ${SPACY_MODEL} ${SPACY_MODEL_VERSION} already installed"
        return
    fi
    if [ ! -f "$SPACY_WHEEL" ]; then
        [ "${OFFLINE:-0}" = "1" ] && {
            echo -e "${C_RED}ERROR${C_RESET}: OFFLINE=1 but wheel not at $SPACY_WHEEL"; exit 1
        }
        echo " Downloading ${SPACY_MODEL} wheel..."
        curl -fsSL -o "$SPACY_WHEEL" "$SPACY_MODEL_URL"
        # Verify download succeeded
        [ ! -f "$SPACY_WHEEL" ] && {
            echo -e "${C_RED}ERROR${C_RESET}: Download failed — wheel not present at $SPACY_WHEEL"; exit 1
        }
    else
        echo " Using cached wheel: $SPACY_WHEEL"
    fi
    ACTUAL_SHA=$(sha256sum "$SPACY_WHEEL" | cut -d' ' -f1)
    if [ "$ACTUAL_SHA" != "$SPACY_MODEL_SHA256" ]; then
        echo -e "${C_RED}ERROR${C_RESET}: checksum mismatch! expected=$SPACY_MODEL_SHA256 actual=$ACTUAL_SHA"
        rm -f "$SPACY_WHEEL"; exit 1
    fi
    echo -e "  ${C_GREEN}✓${C_RESET} Checksum verified"
    $PYTHON -m pip install --quiet "$SPACY_WHEEL"
    # Verify installation succeeded
    $PYTHON -c "
import spacy, sys
nlp = spacy.load('${SPACY_MODEL}')
assert nlp.meta.get('version') == '${SPACY_MODEL_VERSION}', \
    f'Post-install version check failed: {nlp.meta.get(\"version\")}'
" || { echo -e "${C_RED}ERROR${C_RESET}: spaCy model post-install verification failed."; exit 1; }
    echo -e "  ${C_GREEN}✓${C_RESET} ${SPACY_MODEL} ${SPACY_MODEL_VERSION} installed and verified"
}

run_env_smoke_tests() {
    _require_python
    echo " Running environment smoke tests..."
    $PYTHON -c "
import torch
ver=torch.__version__
assert ver.startswith('2.') and 'cu' in ver, f'Expected torch 2.x+cuXXX got {ver}'
t=torch.tensor([1.0,2.0,3.0])
assert torch.allclose(t.mean(),torch.tensor(2.0))
print(f'  \033[0;32m✓\033[0m torch {ver} — tensor op ok')
"
    $PYTHON -c "
import transformers; from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('bert-base-uncased',local_files_only=False)
ids=tok('hello world',return_tensors='pt')
assert ids['input_ids'].shape[1]>0
print(f'  \033[0;32m✓\033[0m transformers {transformers.__version__} — tokenizer ok')
"
    $PYTHON -c "
import faiss,numpy as np
idx=faiss.IndexFlatL2(64); vecs=np.random.rand(10,64).astype('float32')
idx.add(vecs); D,I=idx.search(vecs[:1],3)
assert I.shape==(1,3)
print('  \033[0;32m✓\033[0m faiss — index add/search ok')
"
    $PYTHON -c "
import spacy; nlp=spacy.load('${SPACY_MODEL}')
assert nlp.meta.get('version')=='${SPACY_MODEL_VERSION}', \
    f'Model version {nlp.meta.get(\"version\")} != ${SPACY_MODEL_VERSION}'
doc=nlp('The Supreme Court ruled in favor of the plaintiff.')
ents=[e.label_ for e in doc.ents]
print(f'  \033[0;32m✓\033[0m spacy {spacy.__version__} | model ${SPACY_MODEL_VERSION} | entities: {ents}')
"
}

run_gpu_smoke_tests() {
    if [ "${SKIP_GPU:-0}" = "1" ]; then
        echo -e " ${C_YELLOW}SKIP_GPU=1${C_RESET} — skipping GPU smoke tests"
        step_end "run_gpu_smoke_tests" "SKIP"; return
    fi
    _require_python
    _require_hardware_detected
    echo " Running GPU smoke tests — enforcing TARGET_* constraints..."
    [ "$HARDWARE_MATCH" = "false" ] && \
        echo -e " ${C_YELLOW}WARNING${C_RESET}: hardware mismatches flagged — hard-failing now."
    _assert_cuda_available
    _assert_gpu_count
    _assert_per_gpu_specs
    _assert_gpu_tensor_op
}

write_manifest() {
    _require_python
    _require_hardware_detected
    echo " Writing environment manifest..."
    local git_sha git_branch git_dirty uvlock_sha256 manifest_json
    git_sha=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "not-a-git-repo")
    git_branch=$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    git_dirty=$(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | wc -l | xargs)
    # Defensive: uv.lock must exist for sha256
    [ ! -f "$PROJECT_ROOT/uv.lock" ] && {
        echo -e "${C_RED}ERROR${C_RESET}: uv.lock not found — cannot compute sha256 for manifest."; exit 1
    }
    uvlock_sha256=$(sha256sum "$PROJECT_ROOT/uv.lock" | cut -d' ' -f1)
    manifest_json=$(_collect_manifest_data "$git_sha" "$git_branch" "$git_dirty" "$uvlock_sha256")
    _write_manifest_file "$manifest_json"
    echo "  git: ${git_sha} | branch: ${git_branch} | dirty: ${git_dirty}"
    echo "  uv.lock sha256: ${uvlock_sha256}"
    echo "  hardware_match: ${HARDWARE_MATCH} | detected: ${DETECTED_GPU_COUNT}x ${DETECTED_GPU_NAME}"
}

register_kernel() {
    _require_python
    echo " Registering Jupyter kernel..."
    $PYTHON -m ipykernel install --user \
        --name hallucination-legal-rag \
        --display-name "HallucinationLegalRAG (${TARGET_PYTHON_VERSION})"
    $PYTHON -m jupyter kernelspec list --json 2>/dev/null | $PYTHON -c "
import sys,json
data=json.load(sys.stdin)
kernels=data.get('kernelspecs',{})
assert 'hallucination-legal-rag' in kernels, 'Kernel not found in kernelspec list'
spec=kernels['hallucination-legal-rag']
print(f'  \033[0;32m✓\033[0m kernel: {spec[\"spec\"][\"display_name\"]}')
print(f'    path: {spec[\"resource_dir\"]}')
" || echo -e "  ${C_YELLOW}WARNING${C_RESET}: Could not verify kernel via venv jupyter"
}

verify_tests() {
    _require_uv
    _require_python
    echo " Verifying test suite..."
    UNIT_COUNT=$("$UV" run pytest tests/ --co -q -m unit 2>/dev/null | grep -c "^tests/" || true)
    if [ "${UNIT_COUNT}" -gt 0 ]; then
        echo " Found ${UNIT_COUNT} unit tests — running as verification gate..."
        "$UV" run pytest tests/ -m unit -q --tb=short && \
            echo -e "  ${C_GREEN}✓${C_RESET} Unit tests passed" || \
            { echo -e "  ${C_RED}ERROR${C_RESET}: Unit tests failed"; exit 1; }
    else
        echo " No unit tests yet — collection check..."
        "$UV" run pytest tests/ --co -q 2>/dev/null && \
            echo -e "  ${C_GREEN}✓${C_RESET} Test collection ok" || \
            echo -e "  ${C_YELLOW}WARNING${C_RESET}: Test collection failed"
    fi
}

# ===========================================================================
# Main execution
# ===========================================================================

[ -n "${STEP:-}" ] && echo -e "${C_BOLD}${C_CYAN}Single-step mode: STEP=${STEP}${C_RESET}\n"

echo -e "${C_BOLD}============================================================${C_RESET}"
echo -e "${C_BOLD} cs1090b_HallucinationLegalRAGChatbots — Environment Bootstrap${C_RESET}"
echo -e " Target: ${TARGET_GPU_COUNT}x NVIDIA ${TARGET_GPU_NAME} | Python ${TARGET_PYTHON_VERSION} | torch 2.0.1+cu117"
echo -e " Driver CUDA: ${TARGET_DRIVER_CUDA} (forward-compat) | torch runtime: ${TARGET_TORCH_CUDA_RUNTIME}"
echo -e " Repro: PYTHONHASHSEED=${REPRO_PYTHONHASHSEED} | CUBLAS=${REPRO_CUBLAS_CFG} | RANDOM_SEED=${RANDOM_SEED}"
echo -e " Fail-fast: preflight_fast_checks() first | Single step: STEP=<fn> bash setup.sh"
echo -e "${C_BOLD}============================================================${C_RESET}"

run_step preflight_fast_checks
run_step check_uv
run_step check_lockfile
run_step log_gpu
run_step ensure_venv
run_step verify_python
run_step sync_dependencies
run_step check_dependency_drift
run_step detect_hardware
run_step write_repro_env
run_step write_repro_module
run_step verify_numerical_stability
run_step download_nlp_models
run_step run_env_smoke_tests
run_step run_gpu_smoke_tests
run_step write_manifest
run_step register_kernel
run_step verify_tests

print_summary

echo -e "${C_BOLD}============================================================${C_RESET}"
echo -e "${C_GREEN}${C_BOLD} Environment ready.${C_RESET}"
echo -e " Activate:    source .venv/bin/activate"
echo -e " Kernel:      HallucinationLegalRAG (${TARGET_PYTHON_VERSION})"
echo -e " Manifest:    logs/environment_manifest.json"
echo -e " Repro:       from src.repro import configure; configure()"
echo -e " Single step: STEP=<fn_name> bash setup.sh"
echo -e " CPU mode:    SKIP_GPU=1 bash setup.sh"
echo -e " Dry run:     DRY_RUN=1 bash setup.sh"
echo -e " Offline:     OFFLINE=1 bash setup.sh"
echo -e " Seed expt:   Edit RANDOM_SEED in setup.sh, re-run, commit .env + src/repro.py"
echo -e "${C_BOLD}============================================================${C_RESET}"
