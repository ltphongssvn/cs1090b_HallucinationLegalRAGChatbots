#!/usr/bin/env bash
# scripts/lib.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/lib.sh
# Shared constants, color helpers, step framework, messaging, and defensive guards.
# Sourced by setup.sh and all scripts/*.sh — defines functions only, no execution.
#
# Log levels (set via LOG_LEVEL env var):
#   LOG_LEVEL=0 — quiet:   ERROR only
#   LOG_LEVEL=1 — normal:  ERROR + WARN + OK  (default)
#   LOG_LEVEL=2 — verbose: all messages including INFO (LOG_LEVEL=2 or VERBOSE=1)
#
# Example: LOG_LEVEL=0 bash setup.sh   (CI — errors only)
#          LOG_LEVEL=2 bash setup.sh   (debugging — all output)
#          VERBOSE=1   bash setup.sh   (alias for LOG_LEVEL=2)

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

# ===========================================================================
# Pinned spaCy model
# ===========================================================================
SPACY_MODEL="en_core_web_sm"
SPACY_MODEL_VERSION="3.8.0"
SPACY_MODEL_URL="https://github.com/explosion/spacy-models/releases/download/${SPACY_MODEL}-${SPACY_MODEL_VERSION}/${SPACY_MODEL}-${SPACY_MODEL_VERSION}-py3-none-any.whl"
SPACY_MODEL_SHA256="5e97b9ec4f95153b992896c5c45b1a00c3fcde7f764426c5370f2f11e71abef2"

# ===========================================================================
# Runtime state
# ===========================================================================
DETECTED_GPU_NAME="UNDETECTED"
DETECTED_GPU_COUNT="UNDETECTED"
DETECTED_DRIVER_CUDA="UNDETECTED"
DETECTED_TORCH_CUDA="UNDETECTED"
DETECTED_CUDNN="UNDETECTED"
HARDWARE_MATCH="true"
UV=""

# ===========================================================================
# Log level — controls message verbosity
# LOG_LEVEL=0: quiet  (ERROR only)         — use for CI / automated pipelines
# LOG_LEVEL=1: normal (ERROR+WARN+OK)      — default
# LOG_LEVEL=2: verbose (all + INFO)        — use for debugging
# VERBOSE=1 is an alias for LOG_LEVEL=2
# ===========================================================================
if [ "${VERBOSE:-0}" = "1" ]; then
    LOG_LEVEL=2
fi
LOG_LEVEL="${LOG_LEVEL:-1}"

# ===========================================================================
# Colors — disabled if not a TTY or LOG_LEVEL=0
# ===========================================================================
if [ -t 1 ] && [ "${LOG_LEVEL}" -gt 0 ]; then
    C_RESET="\033[0m"; C_BOLD="\033[1m"; C_GREEN="\033[0;32m"
    C_YELLOW="\033[0;33m"; C_RED="\033[0;31m"; C_CYAN="\033[0;36m"
    C_DIM="\033[2m"; C_BLUE="\033[0;34m"; C_MAGENTA="\033[0;35m"
else
    C_RESET=""; C_BOLD=""; C_GREEN=""; C_YELLOW=""; C_RED=""
    C_CYAN=""; C_DIM=""; C_BLUE=""; C_MAGENTA=""
fi

# ===========================================================================
# Step framework
# ===========================================================================
SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
_step_start_time=0
SETUP_START_TIME=$(date +%s)
_CURRENT_STEP="(none)"

step_begin() {
    _CURRENT_STEP="$1"
    _step_start_time=$(date +%s)
    # Always print step header regardless of LOG_LEVEL — lets user track progress
    echo -e "${C_BOLD}${C_CYAN}▶ $1${C_RESET}"
}

step_end() {
    local name="$1" status="${2:-PASS}" duration=$(( $(date +%s) - _step_start_time ))
    _CURRENT_STEP="(none)"
    SUMMARY_STEPS+=("$name"); SUMMARY_DURATION+=("${duration}s")
    case "$status" in
        PASS) SUMMARY_STATUS+=("${C_GREEN}PASS${C_RESET}") ;;
        WARN) SUMMARY_STATUS+=("${C_YELLOW}WARN${C_RESET}") ;;
        SKIP) SUMMARY_STATUS+=("${C_DIM}SKIP${C_RESET}") ;;
        DRY)  SUMMARY_STATUS+=("${C_MAGENTA}DRY${C_RESET}") ;;
        *)    SUMMARY_STATUS+=("${C_RED}FAIL${C_RESET}") ;;
    esac
    # Print duration only at verbose level — avoids clutter at normal/quiet
    [ "${LOG_LEVEL}" -ge 2 ] && echo -e "  ${C_DIM}(${duration}s)${C_RESET}"
}

print_summary() {
    local total_elapsed=$(( $(date +%s) - SETUP_START_TIME ))
    local mm=$(( total_elapsed / 60 )) ss=$(( total_elapsed % 60 ))
    echo -e "\n${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_BOLD} Setup Summary  ${C_DIM}(total: ${mm}m ${ss}s)${C_RESET}"
    [ "${DRY_RUN:-0}" = "1" ] && \
        echo -e "${C_MAGENTA}${C_BOLD} DRY RUN — no files written, no packages installed${C_RESET}"
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
    step_begin "$fn"; "$fn" "$@"; step_end "$fn" "PASS"
}

# ===========================================================================
# Messaging helpers — respect LOG_LEVEL
#
# Level mapping:
#   _msg_error → always printed (LOG_LEVEL 0+) — errors must never be silenced
#   _msg_warn  → printed at LOG_LEVEL 1+        — warnings visible in normal mode
#   _msg_ok    → printed at LOG_LEVEL 1+        — pass confirmations in normal mode
#   _msg_info  → printed at LOG_LEVEL 2 only    — verbose detail, silent in CI
#   _msg_skip  → printed at LOG_LEVEL 1+        — skip notices in normal mode
#   _msg_dry_run → printed at LOG_LEVEL 1+      — dry-run previews in normal mode
# ===========================================================================

# ERROR — always printed regardless of LOG_LEVEL
_msg_error() {
    local topic="$1" what="$2" why="$3" fix="$4"
    echo -e "\n${C_RED}${C_BOLD}  ✗ ERROR — ${topic}${C_RESET}"
    echo -e "${C_RED}    What:  ${what}${C_RESET}"
    echo -e "${C_DIM}    Why:   ${why}${C_RESET}"
    echo -e "${C_CYAN}    Fix:   ${fix}${C_RESET}\n"
}

# WARN — printed at LOG_LEVEL 1+ (normal and verbose)
_msg_warn() {
    [ "${LOG_LEVEL}" -lt 1 ] && return 0
    local topic="$1" what="$2" severity="$3" action="$4"
    local tag
    [ "$severity" = "action-required" ] && \
        tag="${C_YELLOW}[ACTION REQUIRED]${C_RESET}" || \
        tag="${C_DIM}[informational]${C_RESET}"
    echo -e "${C_YELLOW}  ⚠ WARNING — ${topic}${C_RESET} ${tag}"
    echo -e "${C_YELLOW}    ${what}${C_RESET}"
    echo -e "${C_CYAN}    → ${action}${C_RESET}"
}

# OK — printed at LOG_LEVEL 1+ (normal and verbose)
_msg_ok() {
    [ "${LOG_LEVEL}" -lt 1 ] && return 0
    echo -e "  ${C_GREEN}✓${C_RESET} $1"
}

# INFO — printed at LOG_LEVEL 2 only (verbose)
# Silent in normal (LOG_LEVEL=1) and quiet (LOG_LEVEL=0) modes.
# Use for: progress detail, intermediate values, diagnostic context.
_msg_info() {
    [ "${LOG_LEVEL}" -lt 2 ] && return 0
    echo -e "  ${C_BLUE}ℹ${C_RESET} $1"
}

# SKIP — printed at LOG_LEVEL 1+
_msg_skip() {
    [ "${LOG_LEVEL}" -lt 1 ] && return 0
    echo -e "  ${C_DIM}⊘ $1${C_RESET}"
}

# DRY RUN — printed at LOG_LEVEL 1+
_msg_dry_run() {
    [ "${LOG_LEVEL}" -lt 1 ] && return 0
    local action="$1" target="$2"
    echo -e "  ${C_MAGENTA}⊡ DRY RUN${C_RESET} — would ${action}: ${C_DIM}${target}${C_RESET}"
}

_is_dry_run() { [ "${DRY_RUN:-0}" = "1" ]; }

# ===========================================================================
# Strict failure handling — ERR trap + signal handlers
# ===========================================================================

_on_error() {
    local line="$1" cmd="$2"
    echo -e "\n${C_RED}${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_RED}${C_BOLD}  SETUP FAILED — unexpected error${C_RESET}"
    echo -e "${C_RED}${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_RED}  Line:         ${line}${C_RESET}"
    echo -e "${C_RED}  Command:      ${cmd}${C_RESET}"
    echo -e "${C_RED}  Active step:  ${_CURRENT_STEP}${C_RESET}"
    echo -e "${C_DIM}  Hint: DEBUG=1 bash setup.sh for full set -x trace${C_RESET}"
    echo -e "${C_DIM}  Hint: STEP=${_CURRENT_STEP} bash setup.sh to re-run just the failing step${C_RESET}"
    echo -e "${C_DIM}  Hint: LOG_LEVEL=2 bash setup.sh for verbose output${C_RESET}\n"
    print_summary
}

_on_sigint() {
    echo -e "\n\n${C_YELLOW}${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_YELLOW}${C_BOLD}  SETUP INTERRUPTED — Ctrl+C received${C_RESET}"
    echo -e "${C_YELLOW}${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_YELLOW}  Interrupted during step: ${_CURRENT_STEP}${C_RESET}"
    echo -e "${C_DIM}  The environment may be in a partial state.${C_RESET}"
    echo -e "${C_CYAN}  To resume: STEP=${_CURRENT_STEP} bash setup.sh${C_RESET}"
    echo -e "${C_CYAN}  To restart: bash setup.sh${C_RESET}\n"
    print_summary
    exit 130
}

_on_sigterm() {
    echo -e "\n\n${C_RED}${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_RED}${C_BOLD}  SETUP TERMINATED — SIGTERM received${C_RESET}"
    echo -e "${C_RED}${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_RED}  Terminated during step: ${_CURRENT_STEP}${C_RESET}"
    echo -e "${C_DIM}  Possible causes: cluster job timeout, OOM killer, or external kill${C_RESET}"
    echo -e "${C_CYAN}  Check cluster logs, then: STEP=${_CURRENT_STEP} bash setup.sh${C_RESET}\n"
    print_summary
    exit 143
}

trap '_on_sigint'  INT
trap '_on_sigterm' TERM

# ===========================================================================
# Defensive guards
# ===========================================================================
_require_project_root() {
    if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
        _msg_error "Wrong directory" \
            "PROJECT_ROOT='${PROJECT_ROOT}' has no pyproject.toml" \
            "setup.sh must run from the cs1090b_HallucinationLegalRAGChatbots/ project root" \
            "cd ~/cs1090b_HallucinationLegalRAGChatbots && bash setup.sh"
        exit 1
    fi
}

_require_uv() {
    if [ -z "${UV:-}" ]; then
        if command -v uv &>/dev/null; then UV=$(command -v uv)
        elif [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"
        else
            _msg_error "uv not found" "\$UV unset and uv not on PATH" \
                "uv is required to create the venv and sync packages" \
                "curl -LsSf https://astral.sh/uv/install.sh | sh   then re-run setup.sh"
            exit 1
        fi
    fi
    if ! "$UV" --version &>/dev/null; then
        _msg_error "uv binary broken" "\$UV='$UV' does not execute" \
            "Broken uv will silently fail all package installs" \
            "Re-install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

_require_python() {
    if [ ! -x "${PYTHON}" ]; then
        _msg_error "venv Python not found" "No executable at '${PYTHON}'" \
            "All Python-dependent steps will fail without an activated venv" \
            "STEP=ensure_venv bash setup.sh   or: bash setup.sh"
        exit 1
    fi
    local PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"
    if ! "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info[:3] == (${PYVER_TUPLE}) else 1)" 2>/dev/null; then
        local actual; actual=$("$PYTHON" --version 2>&1)
        _msg_error "Wrong Python version in venv" "Expected ${TARGET_PYTHON_VERSION}, got '${actual}'" \
            "CUDA wheel compatibility and reproducibility depend on exact Python version" \
            "rm -rf .venv && bash setup.sh"
        exit 1
    fi
}

_require_hardware_detected() {
    if [ "${DETECTED_GPU_COUNT}" = "UNDETECTED" ] || \
       [ "${DETECTED_TORCH_CUDA}" = "UNDETECTED" ] || \
       [ "${DETECTED_GPU_NAME}" = "UNDETECTED" ]; then
        _msg_warn "Hardware not yet detected" \
            "DETECTED_* vars are sentinel 'UNDETECTED' — detect_hardware() has not run" \
            "action-required" \
            "Running detect_hardware() now..."
        detect_hardware
    fi
}

_require_repro_env() {
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        _msg_error ".env not found" "No .env at ${PROJECT_ROOT}/.env" \
            "Reproducibility env vars will not be set in notebook/CLI processes" \
            "STEP=write_repro_env bash setup.sh"
        exit 1
    fi
    if [ -z "${RANDOM_SEED:-}" ]; then
        _msg_error "RANDOM_SEED not set" "RANDOM_SEED is empty in current process" \
            "Bug in setup.sh — RANDOM_SEED must be a constant" \
            "Check RANDOM_SEED= in scripts/lib.sh constants block"
        exit 1
    fi
}
