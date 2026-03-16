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
#   _require_uv()                — asserts $UV is set before any uv invocation
#   _require_python()            — asserts $PYTHON exists before any venv invocation
#   _require_hardware_detected() — asserts DETECTED_* populated before use
#   _require_project_root()      — asserts PROJECT_ROOT contains pyproject.toml
#   _require_repro_env()         — asserts .env exists and RANDOM_SEED is set
set -euo pipefail
[ "${DEBUG:-0}" = "1" ] && set -x

# --- Reproducibility: set before any Python or library is invoked ---
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
SETUP_START_TIME=$(date +%s)

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

# Runtime-detected hardware — sentinel "UNDETECTED" distinguishes not-yet-detected from N/A
DETECTED_GPU_NAME="UNDETECTED"
DETECTED_GPU_COUNT="UNDETECTED"
DETECTED_DRIVER_CUDA="UNDETECTED"
DETECTED_TORCH_CUDA="UNDETECTED"
DETECTED_CUDNN="UNDETECTED"
HARDWARE_MATCH="true"

# $UV set by check_uv() — all callers must invoke _require_uv() first
UV=""

# ===========================================================================
# Ergonomics: color output, per-step timing, summary table
# ===========================================================================
if [ -t 1 ]; then
    C_RESET="\033[0m"; C_BOLD="\033[1m"; C_GREEN="\033[0;32m"
    C_YELLOW="\033[0;33m"; C_RED="\033[0;31m"; C_CYAN="\033[0;36m"; C_DIM="\033[2m"
    C_BLUE="\033[0;34m"; C_MAGENTA="\033[0;35m"
else
    C_RESET=""; C_BOLD=""; C_GREEN=""; C_YELLOW=""; C_RED=""
    C_CYAN=""; C_DIM=""; C_BLUE=""; C_MAGENTA=""
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
    local total_elapsed=$(( $(date +%s) - SETUP_START_TIME ))
    local total_mm=$(( total_elapsed / 60 ))
    local total_ss=$(( total_elapsed % 60 ))
    echo -e "\n${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_BOLD} Setup Summary  ${C_DIM}(total: ${total_mm}m ${total_ss}s)${C_RESET}"
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

# ===========================================================================
# Messaging helpers — consistent, actionable user messages
# ===========================================================================

# _msg_error <topic> <what-happened> <why-it-matters> <how-to-fix>
# Produces a structured 4-line error block: no raw tracebacks surfaced without context.
_msg_error() {
    local topic="$1" what="$2" why="$3" fix="$4"
    echo -e ""
    echo -e "${C_RED}${C_BOLD}  ✗ ERROR — ${topic}${C_RESET}"
    echo -e "${C_RED}    What:  ${what}${C_RESET}"
    echo -e "${C_DIM}    Why:   ${why}${C_RESET}"
    echo -e "${C_CYAN}    Fix:   ${fix}${C_RESET}"
    echo -e ""
}

# _msg_warn <topic> <what-happened> <severity> <action>
# severity: "informational" | "action-required"
_msg_warn() {
    local topic="$1" what="$2" severity="$3" action="$4"
    local tag
    [ "$severity" = "action-required" ] && tag="${C_YELLOW}[ACTION REQUIRED]${C_RESET}" || tag="${C_DIM}[informational]${C_RESET}"
    echo -e "${C_YELLOW}  ⚠ WARNING — ${topic}${C_RESET} ${tag}"
    echo -e "${C_YELLOW}    ${what}${C_RESET}"
    echo -e "${C_CYAN}    → ${action}${C_RESET}"
}

# _msg_ok <message>
_msg_ok() { echo -e "  ${C_GREEN}✓${C_RESET} $1"; }

# _msg_info <message>
_msg_info() { echo -e "  ${C_BLUE}ℹ${C_RESET} $1"; }

# _msg_skip <message>
_msg_skip() { echo -e "  ${C_DIM}⊘ $1${C_RESET}"; }

# ===========================================================================
# ERR trap — rich context on unexpected failure
# ===========================================================================
_on_error() {
    local line="$1" cmd="$2"
    echo -e ""
    echo -e "${C_RED}${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_RED}${C_BOLD}  SETUP FAILED — unexpected error${C_RESET}"
    echo -e "${C_RED}${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_RED}  Line:     ${line}${C_RESET}"
    echo -e "${C_RED}  Command:  ${cmd}${C_RESET}"
    echo -e "${C_DIM}  Hint:     Run with DEBUG=1 bash setup.sh for full trace.${C_RESET}"
    echo -e "${C_DIM}  Hint:     Re-run just the failing step: STEP=<fn_name> bash setup.sh${C_RESET}"
    echo -e "${C_DIM}  Hint:     Check logs/environment_manifest.json if it exists.${C_RESET}"
    echo -e ""
    print_summary
}
trap '_on_error "$LINENO" "$BASH_COMMAND"' ERR

# ===========================================================================
# Defensive guards
# ===========================================================================

_require_project_root() {
    if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
        _msg_error \
            "Wrong directory" \
            "PROJECT_ROOT='$PROJECT_ROOT' has no pyproject.toml" \
            "setup.sh must run from the cs1090b_HallucinationLegalRAGChatbots/ project root" \
            "cd ~/cs1090b_HallucinationLegalRAGChatbots && bash setup.sh"
        exit 1
    fi
}

_require_uv() {
    if [ -z "${UV:-}" ]; then
        if command -v uv &>/dev/null; then
            UV=$(command -v uv)
        elif [ -x "$HOME/.local/bin/uv" ]; then
            UV="$HOME/.local/bin/uv"
        else
            _msg_error \
                "uv not found" \
                "\$UV is unset and uv binary not found on PATH" \
                "uv is required to create the venv and sync packages from uv.lock" \
                "curl -LsSf https://astral.sh/uv/install.sh | sh   then re-run setup.sh"
            exit 1
        fi
    fi
    if ! "$UV" --version &>/dev/null; then
        _msg_error \
            "uv binary broken" \
            "\$UV='$UV' exists but does not execute" \
            "A corrupt or incompatible uv binary will silently break all package installs" \
            "Re-install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

_require_python() {
    if [ ! -x "$PYTHON" ]; then
        _msg_error \
            "venv Python not found" \
            "No executable at '$PYTHON'" \
            "All Python-dependent steps will fail without an activated venv" \
            "Run ensure_venv first: STEP=ensure_venv bash setup.sh   or: bash setup.sh"
        exit 1
    fi
    local PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"
    if ! "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info[:3] == (${PYVER_TUPLE}) else 1)" 2>/dev/null; then
        local actual; actual=$("$PYTHON" --version 2>&1)
        _msg_error \
            "Wrong Python version in venv" \
            "Expected Python ${TARGET_PYTHON_VERSION}, got '${actual}'" \
            "CUDA wheel compatibility and reproducibility depend on the exact Python version" \
            "rm -rf .venv && bash setup.sh   (venv will be rebuilt with ${TARGET_PYTHON_VERSION})"
        exit 1
    fi
}

_require_hardware_detected() {
    if [ "${DETECTED_GPU_COUNT}" = "UNDETECTED" ] || \
       [ "${DETECTED_TORCH_CUDA}" = "UNDETECTED" ] || \
       [ "${DETECTED_GPU_NAME}" = "UNDETECTED" ]; then
        _msg_warn \
            "Hardware not yet detected" \
            "DETECTED_* vars are still at sentinel 'UNDETECTED' — detect_hardware() has not run" \
            "action-required" \
            "Running detect_hardware() now to populate DETECTED_* before proceeding..."
        detect_hardware
    fi
}

_require_repro_env() {
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        _msg_error \
            ".env not found" \
            "No .env at $PROJECT_ROOT/.env" \
            "Reproducibility env vars (PYTHONHASHSEED, CUBLAS, RANDOM_SEED) will not be set in notebook/CLI processes" \
            "STEP=write_repro_env bash setup.sh"
        exit 1
    fi
    if [ -z "${RANDOM_SEED:-}" ]; then
        _msg_error \
            "RANDOM_SEED not set" \
            "RANDOM_SEED is empty in the current process" \
            "This is a bug in setup.sh — RANDOM_SEED must be a constant" \
            "Check the RANDOM_SEED= line in the constants block at the top of setup.sh"
        exit 1
    fi
}

# ===========================================================================
# Private helpers
# ===========================================================================

_check_disk_space() {
    local free_gb msg
    free_gb=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
    if [ "${free_gb:-0}" -lt "$TARGET_MIN_DISK_GB" ]; then
        msg="Disk: only ${free_gb}GB free on $PROJECT_ROOT, need ${TARGET_MIN_DISK_GB}GB"
        echo -e "  ${C_RED}✗${C_RESET} $msg"
        echo "$msg"; return 1
    fi
    _msg_ok "disk: ${free_gb}GB free >= ${TARGET_MIN_DISK_GB}GB"
}

_check_nvidia_smi_present() {
    if ! command -v nvidia-smi &>/dev/null; then
        local msg="nvidia-smi not found — not a GPU node"
        echo -e "  ${C_RED}✗${C_RESET} $msg"
        echo "$msg"; return 1
    fi
    _msg_ok "nvidia-smi: present"
}

_check_gpu_count_smi() {
    local count
    count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | xargs)
    if [ "${count:-0}" -lt "$TARGET_GPU_COUNT" ]; then
        local msg="GPU count (nvidia-smi): detected ${count}, need ${TARGET_GPU_COUNT}"
        echo -e "  ${C_RED}✗${C_RESET} $msg"
        echo "$msg"; return 1
    fi
    _msg_ok "nvidia-smi GPU count: ${count} >= ${TARGET_GPU_COUNT}"
}

_check_gpu_name_smi() {
    local names
    names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')
    if ! echo "$names" | grep -q "$TARGET_GPU_NAME"; then
        local msg="GPU name (nvidia-smi): detected '${names}', expected '${TARGET_GPU_NAME}'"
        echo -e "  ${C_RED}✗${C_RESET} $msg"
        echo "$msg"; return 1
    fi
    _msg_ok "nvidia-smi GPU name: contains '${TARGET_GPU_NAME}'"
}

_check_driver_cuda_smi() {
    local driver_cuda
    driver_cuda=$(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}')
    if [ "${driver_cuda}" != "${TARGET_DRIVER_CUDA}" ]; then
        _msg_warn \
            "Driver CUDA version mismatch" \
            "Detected driver CUDA ${driver_cuda}, target is ${TARGET_DRIVER_CUDA}" \
            "informational" \
            "Update TARGET_DRIVER_CUDA in setup.sh constants block if this node is intentionally different"
    else
        _msg_ok "driver CUDA: ${driver_cuda}"
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
    [ "$ok" = "true" ] && _msg_ok "pyproject.toml + uv.lock: present"
    [ "$ok" = "false" ] && return 1
}

_check_uv_present() {
    if ! command -v uv &>/dev/null && ! command -v ~/.local/bin/uv &>/dev/null; then
        local msg="uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo -e "  ${C_RED}✗${C_RESET} $msg"; echo "$msg"; return 1
    fi
    local uv_bin; uv_bin=$(command -v uv 2>/dev/null || echo ~/.local/bin/uv)
    _msg_ok "uv: $($uv_bin --version)"
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
            'index': i, 'name': props.name,
            'vram_gb': round(props.total_memory / 1e9, 2),
            'compute_capability': list(torch.cuda.get_device_capability(i)),
        })
print(json.dumps(result))
"
}

_parse_detected_hardware() {
    local hw_json="$1"
    if [ -z "$hw_json" ]; then
        _msg_error \
            "Hardware query returned empty output" \
            "_query_torch_hardware produced no JSON" \
            "torch may not be installed correctly, or the venv Python cannot import torch" \
            "STEP=sync_dependencies bash setup.sh   then: STEP=detect_hardware bash setup.sh"
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
    for var_name in DETECTED_GPU_COUNT DETECTED_TORCH_CUDA DETECTED_CUDNN DETECTED_GPU_NAME; do
        local val="${!var_name}"
        if [ -z "$val" ] || [ "$val" = "UNDETECTED" ]; then
            _msg_error \
                "Hardware parse failure" \
                "${var_name} is empty after parsing torch hardware JSON" \
                "Manifest and GPU smoke tests will use incorrect values, silently breaking reproducibility records" \
                "Run with DEBUG=1 bash setup.sh to see raw output, or STEP=detect_hardware bash setup.sh"
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
    _msg_info "Detected hardware vs targets:"
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
        _msg_warn \
            "GPU name mismatch" \
            "Detected '${DETECTED_GPU_NAME}', target requires '${TARGET_GPU_NAME}'" \
            "action-required" \
            "You may be on the wrong cluster node. run_gpu_smoke_tests() will hard-fail with details."
        HARDWARE_MATCH="false"
    fi
    if [ "${DETECTED_GPU_COUNT}" != "${TARGET_GPU_COUNT}" ]; then
        _msg_warn \
            "GPU count mismatch" \
            "Detected ${DETECTED_GPU_COUNT} GPUs, target requires ${TARGET_GPU_COUNT}" \
            "action-required" \
            "Check: echo \$CUDA_VISIBLE_DEVICES   or request ${TARGET_GPU_COUNT}x GPU allocation"
        HARDWARE_MATCH="false"
    fi
    if ! echo "${DETECTED_TORCH_CUDA}" | grep -q "^${TARGET_TORCH_CUDA_RUNTIME}"; then
        _msg_warn \
            "torch CUDA runtime mismatch" \
            "Detected torch CUDA ${DETECTED_TORCH_CUDA}, target is ${TARGET_TORCH_CUDA_RUNTIME} (cu117 wheel)" \
            "action-required" \
            "Wrong torch wheel installed. Fix: rm -rf .venv && bash setup.sh"
        HARDWARE_MATCH="false"
    fi
    if [ "$HARDWARE_MATCH" = "true" ]; then
        _msg_ok "Hardware detection complete — all values match targets."
    else
        _msg_warn \
            "Hardware mismatches detected" \
            "One or more DETECTED_* values do not match TARGET_* constants" \
            "action-required" \
            "See warnings above. run_gpu_smoke_tests() will hard-fail — fix hardware issues first."
    fi
}

_assert_cuda_available() {
    _require_python
    $PYTHON -c "
import torch, sys

if not torch.cuda.is_available():
    print('\033[0;31m  ✗ ERROR — CUDA not available\033[0m')
    print('    What:  torch.cuda.is_available() returned False')
    print('    Why:   Without CUDA, all GPU training cells will fail at runtime')
    print('    Fix:   Verify torch wheel: .venv/bin/python -c \"import torch; print(torch.__version__)\"')
    print('           Expected: 2.0.1+cu117   If CPU-only wheel: rm -rf .venv && bash setup.sh')
    sys.exit(1)

actual_cuda = torch.version.cuda or 'unknown'
if not actual_cuda.startswith('${TARGET_TORCH_CUDA_RUNTIME}'):
    print(f'\033[0;31m  ✗ ERROR — torch CUDA runtime mismatch\033[0m')
    print(f'    What:  torch.version.cuda={actual_cuda!r}, expected starts with ${TARGET_TORCH_CUDA_RUNTIME}')
    print(f'    Why:   cu117 wheel required for compatibility with this cluster CUDA driver')
    print(f'    Fix:   rm -rf .venv && bash setup.sh  (will reinstall correct wheel from uv.lock)')
    sys.exit(1)

print(f'  \033[0;32m✓\033[0m CUDA available | torch runtime {actual_cuda}')
"
}

_assert_gpu_count() {
    _require_python
    $PYTHON -c "
import torch, sys
n = torch.cuda.device_count()
if n < ${TARGET_GPU_COUNT}:
    print(f'\033[0;31m  ✗ ERROR — Insufficient GPU count\033[0m')
    print(f'    What:  torch sees {n} GPU(s), need ${TARGET_GPU_COUNT}x ${TARGET_GPU_NAME}')
    print(f'    Why:   Multi-GPU training is configured for ${TARGET_GPU_COUNT} GPUs; fewer will cause allocation errors')
    print(f'    Fix:   Check CUDA_VISIBLE_DEVICES env var, or re-request a ${TARGET_GPU_COUNT}x GPU allocation')
    sys.exit(1)
print(f'  \033[0;32m✓\033[0m GPU count: {n} >= ${TARGET_GPU_COUNT}')
"
}

_assert_per_gpu_specs() {
    _require_python
    $PYTHON -c "
import torch, sys
TARGET_GPU_NAME    = '${TARGET_GPU_NAME}'
TARGET_CAP         = (${TARGET_COMPUTE_CAP_MAJOR}, ${TARGET_COMPUTE_CAP_MINOR})
TARGET_VRAM_GB_MIN = ${TARGET_VRAM_GB_MIN}
failed = False
for i in range(torch.cuda.device_count()):
    name    = torch.cuda.get_device_name(i)
    cap     = torch.cuda.get_device_capability(i)
    vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
    if TARGET_GPU_NAME not in name:
        print(f'\033[0;31m  ✗ ERROR — GPU[{i}] wrong hardware\033[0m')
        print(f'    What:  GPU[{i}] is \"{name}\", expected NVIDIA {TARGET_GPU_NAME}')
        print(f'    Why:   Benchmark results, memory estimates, and compute cap checks are calibrated for {TARGET_GPU_NAME}')
        print(f'    Fix:   Request a node with {TARGET_GPU_COUNT}x NVIDIA {TARGET_GPU_NAME}')
        failed = True
    elif cap < TARGET_CAP:
        print(f'\033[0;31m  ✗ ERROR — GPU[{i}] compute capability too low\033[0m')
        print(f'    What:  cap={cap}, need >={TARGET_CAP} for {TARGET_GPU_NAME}')
        print(f'    Why:   Operations like bf16 and flash attention require compute cap >= {TARGET_CAP}')
        print(f'    Fix:   Request a node with NVIDIA {TARGET_GPU_NAME} (compute cap {TARGET_CAP})')
        failed = True
    elif vram_gb < TARGET_VRAM_GB_MIN:
        print(f'\033[0;31m  ✗ ERROR — GPU[{i}] insufficient VRAM\033[0m')
        print(f'    What:  {vram_gb:.1f}GB VRAM, need >={TARGET_VRAM_GB_MIN}GB')
        print(f'    Why:   Legal-BERT fine-tuning + FAISS index exceed {TARGET_VRAM_GB_MIN}GB per GPU')
        print(f'    Fix:   Request {TARGET_GPU_COUNT}x NVIDIA {TARGET_GPU_NAME} (each has ~{TARGET_VRAM_GB_MIN+1:.0f}GB VRAM)')
        failed = True
    else:
        print(f'  \033[0;32m✓\033[0m GPU[{i}] {name} | cap {cap} | {vram_gb:.1f}GB')
if failed:
    sys.exit(1)
"
}

_assert_gpu_tensor_op() {
    _require_python
    $PYTHON -c "
import torch, sys
try:
    t = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
    assert t.device.type == 'cuda'
    result = t.mean().cpu()
    assert torch.allclose(result, torch.tensor(2.0)), f'mean={result.item()}, expected 2.0'
    print(f'  \033[0;32m✓\033[0m CUDA ${TARGET_TORCH_CUDA_RUNTIME} — tensor round-trip on cuda:0 ok')
except Exception as e:
    print(f'\033[0;31m  ✗ ERROR — GPU functional test failed\033[0m')
    print(f'    What:  {e}')
    print(f'    Why:   If basic tensor ops fail, all training will fail at the same point')
    print(f'    Fix:   Check GPU health: nvidia-smi   |   Try: STEP=run_gpu_smoke_tests bash setup.sh')
    sys.exit(1)
"
}

_collect_manifest_data() {
    _require_python
    _require_hardware_detected
    local git_sha="$1" git_branch="$2" git_dirty="$3" uvlock_sha256="$4"
    for arg_name in git_sha git_branch git_dirty uvlock_sha256; do
        local val="${!arg_name}"
        if [ -z "$val" ]; then
            _msg_warn \
                "Manifest arg '${arg_name}' is empty" \
                "_collect_manifest_data received empty '${arg_name}'" \
                "informational" \
                "Manifest will record 'unknown' for this field — non-blocking"
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
    if [ -z "$manifest_json" ]; then
        _msg_error \
            "Empty manifest JSON" \
            "_write_manifest_file received empty input" \
            "An empty manifest cannot be used for reproducibility auditing or result comparison" \
            "Run STEP=write_manifest bash setup.sh   to retry manifest generation"
        exit 1
    fi
    if ! echo "$manifest_json" | $PYTHON -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
        _msg_error \
            "Malformed manifest JSON" \
            "Manifest JSON failed to parse — likely a bash variable interpolation issue" \
            "A corrupt manifest will silently record wrong reproducibility state" \
            "Run DEBUG=1 bash setup.sh and inspect the _collect_manifest_data output"
        echo "  First 300 chars of output: ${manifest_json:0:300}"
        exit 1
    fi
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
    _require_project_root
    echo " Running preflight fast checks (pre-venv, seconds)..."
    _msg_info "These checks run before any expensive operation. Failure here saves venv build + sync time on the wrong node."
    local failures=()
    local output

    output=$(_check_disk_space 2>&1)
    echo "$output" | grep -E "^  [✓⚠]" | cat
    echo "$output" | grep "^Disk:" | grep -q "." && failures+=("$(echo "$output" | grep '^Disk:')")

    output=$(_check_uv_present 2>&1)
    echo "$output" | grep -E "^  [✓⚠]" | cat
    echo "$output" | grep "^uv not" | grep -q "." && failures+=("$(echo "$output" | grep '^uv not')")

    output=$(_check_lockfile_present 2>&1)
    echo "$output" | grep -E "^  [✓⚠]" | cat
    while IFS= read -r line; do
        [[ "$line" =~ ^(pyproject|uv\.lock\ not) ]] && failures+=("$line")
    done <<< "$output"

    output=$(_check_nvidia_smi_present 2>&1)
    echo "$output" | grep -E "^  [✓⚠]" | cat
    echo "$output" | grep "^nvidia-smi not" | grep -q "." && failures+=("$(echo "$output" | grep '^nvidia-smi not')")

    if command -v nvidia-smi &>/dev/null; then
        output=$(_check_gpu_count_smi 2>&1)
        echo "$output" | grep -E "^  [✓⚠]" | cat
        echo "$output" | grep "^GPU count" | grep -q "detected" && failures+=("$(echo "$output" | grep '^GPU count')")

        output=$(_check_gpu_name_smi 2>&1)
        echo "$output" | grep -E "^  [✓⚠]" | cat
        echo "$output" | grep "^GPU name" | grep -q "detected" && failures+=("$(echo "$output" | grep '^GPU name')")

        _check_driver_cuda_smi
    fi

    if [ ${#failures[@]} -gt 0 ]; then
        echo -e ""
        echo -e "${C_RED}${C_BOLD}============================================================${C_RESET}"
        echo -e "${C_RED}${C_BOLD}  PREFLIGHT FAILED — ${#failures[@]} issue(s) must be fixed${C_RESET}"
        echo -e "${C_RED}${C_BOLD}============================================================${C_RESET}"
        for i in "${!failures[@]}"; do
            echo -e "  ${C_RED}[$((i+1))]${C_RESET} ${failures[$i]}"
        done
        echo -e ""
        echo -e "${C_CYAN}  Next steps:${C_RESET}"
        echo -e "${C_CYAN}  • GPU node issues: request a new allocation with ${TARGET_GPU_COUNT}x NVIDIA ${TARGET_GPU_NAME}${C_RESET}"
        echo -e "${C_CYAN}  • Missing files:   verify you are in cs1090b_HallucinationLegalRAGChatbots/${C_RESET}"
        echo -e "${C_CYAN}  • After fixing:    bash setup.sh${C_RESET}"
        echo -e ""
        exit 1
    fi
    _msg_ok "All preflight fast checks passed — proceeding to expensive operations."
}

check_uv() {
    if ! command -v uv &>/dev/null && ! command -v ~/.local/bin/uv &>/dev/null; then
        _msg_error \
            "uv not found" \
            "uv binary not found on PATH or at ~/.local/bin/uv" \
            "uv is the package manager for this project — all installs require it" \
            "curl -LsSf https://astral.sh/uv/install.sh | sh   then: source ~/.bashrc && bash setup.sh"
        exit 1
    fi
    UV=$(command -v uv 2>/dev/null || echo ~/.local/bin/uv)
    if ! "$UV" --version &>/dev/null; then
        _msg_error \
            "uv binary broken" \
            "\$UV='$UV' does not execute" \
            "Broken uv will silently fail to install packages" \
            "Re-install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    _msg_ok "uv: $("$UV" --version)"
}

check_lockfile() {
    _require_project_root
    [ ! -f "$PROJECT_ROOT/pyproject.toml" ] && {
        _msg_error "pyproject.toml not found" \
            "No pyproject.toml at $PROJECT_ROOT" \
            "Cannot resolve dependencies without it" \
            "cd ~/cs1090b_HallucinationLegalRAGChatbots && bash setup.sh"
        exit 1
    }
    [ ! -f "$PROJECT_ROOT/uv.lock" ] && {
        _msg_error "uv.lock not found" \
            "No uv.lock at $PROJECT_ROOT/uv.lock" \
            "Without uv.lock, uv sync --frozen cannot run and package versions are not pinned" \
            "uv lock && git add uv.lock && git commit -m 'chore: pin uv.lock'   then: bash setup.sh"
        exit 1
    }
    _msg_ok "uv.lock sha256: $(sha256sum "$PROJECT_ROOT/uv.lock" | cut -d' ' -f1)"
}

log_gpu() {
    _msg_info "Hardware target: ${TARGET_GPU_COUNT}x NVIDIA ${TARGET_GPU_NAME} | CUDA runtime ${TARGET_TORCH_CUDA_RUNTIME} | driver CUDA ${TARGET_DRIVER_CUDA}"
    _msg_info "torch wheel 2.0.1+cu117 compiled against CUDA ${TARGET_TORCH_CUDA_RUNTIME} — driver CUDA ${TARGET_DRIVER_CUDA} is forward-compatible (expected)"
    if command -v nvidia-smi &>/dev/null; then
        echo " --- nvidia-smi per-GPU summary (pre-venv) ---"
        nvidia-smi --query-gpu=index,name,memory.total,driver_version \
            --format=csv,noheader | while IFS=',' read -r idx name mem drv; do
            echo "  GPU $idx:$(echo "$name"|xargs) | VRAM:$(echo "$mem"|xargs) | Driver:$(echo "$drv"|xargs)"
        done
    else
        _msg_warn \
            "nvidia-smi not found" \
            "Cannot log pre-venv GPU details" \
            "informational" \
            "If this is a GPU node, nvidia-smi should be on PATH. Check module loads."
    fi
    command -v nvcc &>/dev/null && \
        _msg_ok "CUDA toolkit (nvcc): $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')" || \
        _msg_warn "nvcc not on PATH" "CUDA toolkit version cannot be verified" "informational" "nvcc is not required for runtime but useful for forensic reproducibility"
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
        _msg_ok ".venv already exists with Python ${TARGET_PYTHON_VERSION} — skipping creation"
        return
    fi
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        _msg_warn \
            "Stale .venv detected" \
            ".venv exists but contains wrong Python version (need ${TARGET_PYTHON_VERSION})" \
            "action-required" \
            "Removing stale .venv in 5 seconds — press Ctrl+C to cancel"
        echo "         Contents: $(du -sh "$PROJECT_ROOT/.venv" 2>/dev/null | cut -f1) on disk"
        [ "${DRY_RUN:-0}" = "1" ] && _msg_skip "DRY_RUN=1 — skipping removal. Exiting." && exit 0
        sleep 5
        rm -rf "$PROJECT_ROOT/.venv"
    fi
    _msg_info "Creating .venv with Python ${TARGET_PYTHON_VERSION}..."
    "$UV" venv .venv --python "${TARGET_PYTHON_VERSION}" --seed
}

verify_python() {
    _require_python
    local PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"
    $PYTHON -c "import sys; assert sys.version_info[:3] == (${PYVER_TUPLE}), f'Expected ${TARGET_PYTHON_VERSION} got {sys.version}'"
    _msg_ok "Python: $($PYTHON --version)"
    _msg_info "Executable: $($PYTHON -c 'import sys; print(sys.executable)')"
}

sync_dependencies() {
    _require_uv
    _require_python
    _msg_info "Syncing from uv.lock (--frozen) — this may take a few minutes on first run..."
    _msg_info "--dev is explicit: ensures pytest/mypy/hypothesis are always installed (not left to uv defaults)"
    "$UV" sync --frozen --dev
    _msg_ok "Dependencies synced from uv.lock"
}

check_dependency_drift() {
    _require_uv
    _require_python
    echo " Checking for dependency drift..."
    if [ "$PROJECT_ROOT/pyproject.toml" -nt "$PROJECT_ROOT/uv.lock" ]; then
        _msg_error \
            "Stale uv.lock" \
            "pyproject.toml is newer than uv.lock" \
            "Someone edited pyproject.toml without regenerating uv.lock. Collaborators will install different versions." \
            "uv lock && git add uv.lock && git commit -m 'chore: regenerate uv.lock'   then: bash setup.sh"
        exit 1
    fi
    _msg_ok "pyproject.toml vs uv.lock timestamp — ok (lockfile is not stale)"

    "$UV" lock --check 2>/dev/null && \
        _msg_ok "uv lock --check — lockfile consistent with pyproject.toml" || {
        _msg_error \
            "uv.lock inconsistency" \
            "uv lock --check failed — lockfile would change on re-solve" \
            "pyproject.toml constraints no longer satisfy uv.lock pins, causing silent version drift" \
            "uv lock && git add uv.lock && git commit -m 'chore: regenerate uv.lock'"
        exit 1
    }

    "$UV" sync --frozen --dev --check 2>/dev/null && \
        _msg_ok "uv sync --check — installed packages match uv.lock exactly" || {
        _msg_error \
            "Package drift detected" \
            "Installed packages in .venv diverge from uv.lock" \
            "Someone ran 'pip install' manually after setup, or venv was partially modified. Results may not be reproducible." \
            "bash setup.sh   (re-syncs .venv from uv.lock — manual pip installs will be overwritten)"
        exit 1
    }

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
        if Version(inst)<Version(min_v):
            drift.append(f'{pkg}: installed {inst} < minimum {min_v}')
        elif exact_v and inst!=exact_v:
            print(f'  \033[0;33m⚠ WARNING\033[0m {pkg} installed={inst}, expected={exact_v} — check wheel type (CPU vs CUDA)')
        else:
            print(f'  \033[0;32m✓\033[0m {pkg:<20} {inst}')
    except meta.PackageNotFoundError:
        drift.append(f'{pkg}: NOT INSTALLED — run bash setup.sh')
if drift:
    print('\n  \033[0;31mDrift detected — these packages are below minimum required versions:\033[0m')
    for d in drift:
        print(f'  \033[0;31m  • {d}\033[0m')
    print('  \033[0;36m  Fix: bash setup.sh  (re-syncs from uv.lock)\033[0m')
    raise SystemExit(1)
print('  All critical package versions verified — no drift')
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
    [ ! -f "$PROJECT_ROOT/.env" ] && {
        _msg_error ".env not created" \
            "cat heredoc to .env silently failed" \
            "Without .env, notebooks cannot load reproducibility settings at runtime" \
            "Check disk space and permissions on $PROJECT_ROOT"
        exit 1
    }
    _require_python
    $PYTHON -c "
import os
checks = [
    ('PYTHONHASHSEED','${REPRO_PYTHONHASHSEED}'),
    ('CUBLAS_WORKSPACE_CONFIG','${REPRO_CUBLAS_CFG}'),
    ('TOKENIZERS_PARALLELISM','${REPRO_TOKENIZERS_PAR}'),
    ('RANDOM_SEED','${RANDOM_SEED}'),
]
failed = []
for var,exp in checks:
    act=os.environ.get(var)
    if act != exp:
        failed.append(f'{var}={act!r} != {exp!r}')
    else:
        print(f'  \033[0;32m✓\033[0m {var}={act}')
if failed:
    print('\033[0;31m  ✗ ERROR — Reproducibility env var mismatch\033[0m')
    for f in failed:
        print(f'  \033[0;31m    {f}\033[0m')
    print('  \033[0;36m  Fix: Check PYTHONHASHSEED/CUBLAS_WORKSPACE_CONFIG are not overridden in ~/.bashrc\033[0m')
    raise SystemExit(1)
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
# Why this exists:
#   setup.sh sets env vars and torch flags in the shell process.
#   The Jupyter kernel launches in a FRESH process — it inherits nothing.
#   This module is the single source of truth that both the notebook AND CLI
#   import to get identical reproducibility settings regardless of launch method.
#
# Usage:
#   from src.repro import configure
#   repro_cfg = configure()  # must be first — before any torch import
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
        raise FileNotFoundError(
            f".env not found at {env_path}.\n"
            f"  Why: This file is required for reproducibility settings in notebook/CLI processes.\n"
            f"  Fix: Run bash setup.sh from the project root to regenerate it."
        )
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
        if actual != expected:
            raise AssertionError(
                f"{var}={actual!r} — expected {expected!r}.\n"
                f"  Why: This env var must be set before importing torch to take effect.\n"
                f"  Fix: Call configure() as the VERY FIRST statement in Cell 1, before any imports."
            )
        checks[var] = actual
    if not torch.are_deterministic_algorithms_enabled():
        raise AssertionError(
            "torch.use_deterministic_algorithms is not enabled.\n"
            "  Why: Non-deterministic ops will produce different results across runs.\n"
            "  Fix: Call configure() before any torch import, or re-run Cell 1."
        )
    checks["deterministic_algorithms"] = True
    if torch.backends.cudnn.benchmark:
        raise AssertionError(
            "cudnn.benchmark=True — auto-selects non-deterministic algorithms.\n"
            "  Fix: Call configure() before any torch import, or re-run Cell 1."
        )
    checks["cudnn_benchmark"] = False
    if not torch.backends.cudnn.deterministic:
        raise AssertionError(
            "cudnn.deterministic=False — cuDNN ops are non-deterministic.\n"
            "  Fix: Call configure() before any torch import, or re-run Cell 1."
        )
    checks["cudnn_deterministic"] = True
    checks["random_seed"] = _RANDOM_SEED
    return checks


def configure(
    project_root: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Thin orchestrator: load → apply → seed → verify.
    Each sub-step has its own single-responsibility helper.
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
    [ ! -f "$PROJECT_ROOT/src/repro.py" ] && {
        _msg_error "src/repro.py not created" \
            "File was not found after write" \
            "Notebooks will fail to import configure() and will run without reproducibility settings" \
            "Check disk space and permissions on $PROJECT_ROOT/src/"
        exit 1
    }
    _msg_ok "src/repro.py written (RANDOM_SEED=${RANDOM_SEED})"
    $PYTHON -c "
import sys; sys.path.insert(0,'${PROJECT_ROOT}')
from src.repro import configure
cfg = configure(verbose=True)
assert cfg['random_seed'] == ${RANDOM_SEED}, f'random_seed {cfg[\"random_seed\"]} != ${RANDOM_SEED}'
print('  \033[0;32m✓\033[0m src/repro.configure() verified')
" || {
        _msg_error "src/repro.py verification failed" \
            "configure() raised an error after writing src/repro.py" \
            "A broken repro module means notebooks will run without reproducibility settings" \
            "Run DEBUG=1 STEP=write_repro_module bash setup.sh to see the full traceback"
        exit 1
    }
}

verify_numerical_stability() {
    _require_python
    _require_repro_env
    echo " Verifying numerical/runtime stability..."
    $PYTHON -c "
import os, torch, sys
checks = [
    ('CUBLAS_WORKSPACE_CONFIG', os.environ.get('CUBLAS_WORKSPACE_CONFIG'), '${REPRO_CUBLAS_CFG}'),
    ('PYTHONHASHSEED',          os.environ.get('PYTHONHASHSEED'),          '${REPRO_PYTHONHASHSEED}'),
    ('TOKENIZERS_PARALLELISM',  os.environ.get('TOKENIZERS_PARALLELISM'),  '${REPRO_TOKENIZERS_PAR}'),
    ('RANDOM_SEED',             os.environ.get('RANDOM_SEED'),             '${RANDOM_SEED}'),
]
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
failed = []
for name,actual,exp in checks:
    if actual != exp:
        failed.append(f'{name}={actual!r} != {exp!r}')
    else:
        print(f'  \033[0;32m✓\033[0m {name}={actual}')
if not torch.are_deterministic_algorithms_enabled():
    failed.append('torch.use_deterministic_algorithms not enabled')
if torch.backends.cudnn.benchmark:
    failed.append('cudnn.benchmark=True')
if not torch.backends.cudnn.deterministic:
    failed.append('cudnn.deterministic=False')
if failed:
    print('\033[0;31m  ✗ Numerical stability check FAILED:\033[0m')
    for f in failed: print(f'    • {f}')
    print('\033[0;36m  Fix: Ensure PYTHONHASHSEED/CUBLAS_WORKSPACE_CONFIG are not overridden in ~/.bashrc\033[0m')
    print('\033[0;36m       Then: bash setup.sh\033[0m')
    sys.exit(1)
print('  \033[0;32m✓\033[0m torch deterministic flags — ok')
print()
print('  \033[2mNOTE: Notebooks must call: from src.repro import configure; configure()\033[0m')
print('  \033[2m      This is the ONLY correct way to ensure notebook/CLI parity.\033[0m')
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
        _msg_ok "${SPACY_MODEL} ${SPACY_MODEL_VERSION} already installed — skipping download"
        return
    fi
    if [ ! -f "$SPACY_WHEEL" ]; then
        if [ "${OFFLINE:-0}" = "1" ]; then
            _msg_error "Offline mode: wheel not cached" \
                "OFFLINE=1 but wheel not found at $SPACY_WHEEL" \
                "Cannot download in offline mode" \
                "Pre-cache the wheel: mkdir -p .cache/spacy && wget -O $SPACY_WHEEL $SPACY_MODEL_URL   then: OFFLINE=1 bash setup.sh"
            exit 1
        fi
        _msg_info "Downloading ${SPACY_MODEL} ${SPACY_MODEL_VERSION} wheel (pinned, will be cached)..."
        curl -fsSL -o "$SPACY_WHEEL" "$SPACY_MODEL_URL"
        [ ! -f "$SPACY_WHEEL" ] && {
            _msg_error "Download failed" \
                "curl succeeded but wheel not present at $SPACY_WHEEL" \
                "spaCy NER model is required for entity extraction in the RAG pipeline" \
                "Check network connectivity: curl -I $SPACY_MODEL_URL"
            exit 1
        }
    else
        _msg_info "Using cached wheel: $SPACY_WHEEL"
    fi
    ACTUAL_SHA=$(sha256sum "$SPACY_WHEEL" | cut -d' ' -f1)
    if [ "$ACTUAL_SHA" != "$SPACY_MODEL_SHA256" ]; then
        _msg_error "spaCy wheel checksum mismatch" \
            "expected=$SPACY_MODEL_SHA256  actual=$ACTUAL_SHA" \
            "A corrupted or tampered wheel could produce wrong NER results, silently degrading RAG quality" \
            "Cached wheel deleted. Re-run: bash setup.sh   (will re-download and re-verify)"
        rm -f "$SPACY_WHEEL"; exit 1
    fi
    _msg_ok "Checksum verified: $ACTUAL_SHA"
    $PYTHON -m pip install --quiet "$SPACY_WHEEL"
    $PYTHON -c "
import spacy, sys
nlp = spacy.load('${SPACY_MODEL}')
v = nlp.meta.get('version')
if v != '${SPACY_MODEL_VERSION}':
    print(f'\033[0;31m  ✗ Post-install version mismatch: {v} != ${SPACY_MODEL_VERSION}\033[0m')
    print('  Fix: rm -rf .venv && bash setup.sh')
    sys.exit(1)
print(f'  \033[0;32m✓\033[0m ${SPACY_MODEL} {v} installed and verified')
" || { _msg_error "spaCy post-install check failed" "Model loaded wrong version" "Entity extraction will silently use wrong model" "rm -rf .venv && bash setup.sh"; exit 1; }
}

run_env_smoke_tests() {
    _require_python
    echo " Running environment smoke tests..."
    $PYTHON -c "
import torch, sys
ver=torch.__version__
if not (ver.startswith('2.') and 'cu' in ver):
    print(f'\033[0;31m  ✗ ERROR — Wrong torch build\033[0m')
    print(f'    What:  torch version is {ver!r}, expected 2.x+cuXXX')
    print(f'    Why:   CPU-only torch wheel will silently run all ops on CPU')
    print(f'    Fix:   rm -rf .venv && bash setup.sh  (uv.lock pins cu117 wheel)')
    sys.exit(1)
t=torch.tensor([1.0,2.0,3.0])
assert torch.allclose(t.mean(),torch.tensor(2.0))
print(f'  \033[0;32m✓\033[0m torch {ver} — tensor op ok')
"
    $PYTHON -c "
import transformers, sys
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('bert-base-uncased',local_files_only=False)
ids=tok('hello world',return_tensors='pt')
if ids['input_ids'].shape[1] == 0:
    print('\033[0;31m  ✗ ERROR — tokenizer produced empty output\033[0m')
    print('    Fix: STEP=sync_dependencies bash setup.sh')
    sys.exit(1)
print(f'  \033[0;32m✓\033[0m transformers {transformers.__version__} — tokenizer ok')
"
    $PYTHON -c "
import faiss,numpy as np,sys
idx=faiss.IndexFlatL2(64); vecs=np.random.rand(10,64).astype('float32')
idx.add(vecs); D,I=idx.search(vecs[:1],3)
if I.shape != (1,3):
    print(f'\033[0;31m  ✗ ERROR — faiss search returned shape {I.shape}, expected (1,3)\033[0m')
    print('    Fix: STEP=sync_dependencies bash setup.sh')
    sys.exit(1)
print('  \033[0;32m✓\033[0m faiss — index add/search ok')
"
    $PYTHON -c "
import spacy, sys
nlp=spacy.load('${SPACY_MODEL}')
v=nlp.meta.get('version')
if v != '${SPACY_MODEL_VERSION}':
    print(f'\033[0;31m  ✗ ERROR — spaCy model version {v} != ${SPACY_MODEL_VERSION}\033[0m')
    print('    Fix: STEP=download_nlp_models bash setup.sh')
    sys.exit(1)
doc=nlp('The Supreme Court ruled in favor of the plaintiff.')
ents=[e.label_ for e in doc.ents]
print(f'  \033[0;32m✓\033[0m spacy {spacy.__version__} | model {v} | entities: {ents}')
"
}

run_gpu_smoke_tests() {
    if [ "${SKIP_GPU:-0}" = "1" ]; then
        _msg_skip "SKIP_GPU=1 — GPU smoke tests skipped (CPU-only mode)"
        step_end "run_gpu_smoke_tests" "SKIP"; return
    fi
    _require_python
    _require_hardware_detected
    echo " Running GPU smoke tests — enforcing TARGET_* constraints..."
    if [ "$HARDWARE_MATCH" = "false" ]; then
        _msg_warn \
            "Hardware mismatches were flagged by detect_hardware()" \
            "One or more DETECTED_* values do not match TARGET_* constants" \
            "action-required" \
            "Assertions below will fail with specific details. Fix hardware before re-running."
    fi
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
    [ ! -f "$PROJECT_ROOT/uv.lock" ] && {
        _msg_error "uv.lock missing for manifest" \
            "Cannot compute uv.lock sha256 — file not found" \
            "Manifest will not record the exact dependency snapshot" \
            "uv lock && git add uv.lock && git commit -m 'chore: pin uv.lock'"
        exit 1
    }
    uvlock_sha256=$(sha256sum "$PROJECT_ROOT/uv.lock" | cut -d' ' -f1)
    manifest_json=$(_collect_manifest_data "$git_sha" "$git_branch" "$git_dirty" "$uvlock_sha256")
    _write_manifest_file "$manifest_json"
    _msg_info "git: ${git_sha} | branch: ${git_branch} | dirty: ${git_dirty}"
    _msg_info "uv.lock sha256: ${uvlock_sha256}"
    _msg_info "hardware_match: ${HARDWARE_MATCH} | detected: ${DETECTED_GPU_COUNT}x ${DETECTED_GPU_NAME}"
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
if 'hallucination-legal-rag' not in kernels:
    print('\033[0;31m  ✗ ERROR — Kernel not found after registration\033[0m')
    print('    Why: JupyterLab will default to the wrong kernel, missing venv packages')
    print('    Fix: STEP=register_kernel bash setup.sh   or: jupyter kernelspec list')
    sys.exit(1)
spec=kernels['hallucination-legal-rag']
print(f'  \033[0;32m✓\033[0m kernel registered: {spec[\"spec\"][\"display_name\"]}')
print(f'    path: {spec[\"resource_dir\"]}')
" || _msg_warn \
        "Kernel verification skipped" \
        "Could not verify kernel via venv jupyter — jupyter may not be fully installed" \
        "informational" \
        "Manually verify: .venv/bin/python -m jupyter kernelspec list"
}

verify_tests() {
    _require_uv
    _require_python
    echo " Verifying test suite..."
    UNIT_COUNT=$("$UV" run pytest tests/ --co -q -m unit 2>/dev/null | grep -c "^tests/" || true)
    if [ "${UNIT_COUNT}" -gt 0 ]; then
        _msg_info "Found ${UNIT_COUNT} unit tests — running as environment verification gate..."
        "$UV" run pytest tests/ -m unit -q --tb=short && \
            _msg_ok "Unit tests passed — environment verified end-to-end" || {
            _msg_error "Unit tests failed" \
                "${UNIT_COUNT} unit tests collected but one or more failed" \
                "Failing unit tests indicate the environment is broken — do not proceed to training" \
                "Run: STEP=verify_tests DEBUG=1 bash setup.sh   or: .venv/bin/pytest tests/ -m unit -v --tb=long"
            exit 1
        }
    else
        _msg_info "No unit tests found yet — falling back to collection check..."
        "$UV" run pytest tests/ --co -q 2>/dev/null && \
            _msg_ok "Test collection ok — no unit tests to run yet" || \
            _msg_warn \
                "Test collection failed" \
                "pytest --co could not collect tests — likely a src/ import error" \
                "informational" \
                "Check: .venv/bin/python -c 'import src.environment; import src.repro'"
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
echo -e ""
echo -e " ${C_BOLD}Quick reference:${C_RESET}"
echo -e "   Activate venv:     source .venv/bin/activate"
echo -e "   Jupyter kernel:    HallucinationLegalRAG (${TARGET_PYTHON_VERSION})"
echo -e "   Manifest:          logs/environment_manifest.json"
echo -e ""
echo -e " ${C_BOLD}Notebook Cell 1 (required first line):${C_RESET}"
echo -e "   ${C_CYAN}from src.repro import configure; repro_cfg = configure()${C_RESET}"
echo -e ""
echo -e " ${C_BOLD}Re-run a single step:${C_RESET}"
echo -e "   STEP=<fn_name> bash setup.sh"
echo -e "   Available steps: preflight_fast_checks | check_uv | check_lockfile | log_gpu"
echo -e "                    ensure_venv | verify_python | sync_dependencies | check_dependency_drift"
echo -e "                    detect_hardware | write_repro_env | write_repro_module"
echo -e "                    verify_numerical_stability | download_nlp_models | run_env_smoke_tests"
echo -e "                    run_gpu_smoke_tests | write_manifest | register_kernel | verify_tests"
echo -e ""
echo -e " ${C_BOLD}Other modes:${C_RESET}"
echo -e "   CPU mode:   SKIP_GPU=1 bash setup.sh"
echo -e "   Debug:      DEBUG=1 bash setup.sh"
echo -e "   Dry run:    DRY_RUN=1 bash setup.sh"
echo -e "   Offline:    OFFLINE=1 bash setup.sh"
echo -e "   Seed expt:  Edit RANDOM_SEED in setup.sh, re-run, commit .env + src/repro.py"
echo -e "${C_BOLD}============================================================${C_RESET}"
