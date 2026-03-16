#!/usr/bin/env bash
# setup.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/setup.sh
#
# Thin orchestrator — sources modular scripts, then runs each step.
# All logic lives in scripts/*.sh; this file contains no implementation.
#
# Usage:        bash setup.sh
# Debug:        DEBUG=1 bash setup.sh
# Skip GPU:     SKIP_GPU=1 bash setup.sh
# Dry run:      DRY_RUN=1 bash setup.sh
# Offline:      OFFLINE=1 bash setup.sh
# No download:  NO_DOWNLOAD=1 bash setup.sh
# No Jupyter:   NO_JUPYTER=1 bash setup.sh
# Single step:  STEP=<name> bash setup.sh
#
# Log level:    LOG_LEVEL=<0|1|2> bash setup.sh
#               0 = quiet   — ERROR only (CI / automated pipelines)
#               1 = normal  — ERROR + WARN + OK (default)
#               2 = verbose — all messages including INFO
#   Alias:      VERBOSE=1 bash setup.sh  (same as LOG_LEVEL=2)
#
# Module map:
#   scripts/lib.sh            — constants, colors, step framework, messaging, guards
#   scripts/bootstrap_env.sh  — uv, lockfile, venv, deps, drift
#   scripts/validate_gpu.sh   — GPU detection, hardware policy, smoke tests
#   scripts/setup_nlp.sh      — spaCy model download and verification
#   scripts/setup_notebook.sh — repro env, repro module, stability, kernel
#   scripts/validate_tests.sh — env smoke tests, unit test gate
#   scripts/manifest.sh       — environment manifest collection and write
#   src/manifest_collector.py — manifest Python logic (lintable/typed/testable)
#   src/drift_check.py        — drift check Python logic (lintable/typed/testable)
set -euo pipefail
[ "${DEBUG:-0}" = "1" ] && set -x

# --- Reproducibility: set before any Python or library is invoked ---
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Source all modules — lib.sh must be first (defines shared state + log level)
# shellcheck source=scripts/lib.sh
source "$PROJECT_ROOT/scripts/lib.sh"
source "$PROJECT_ROOT/scripts/bootstrap_env.sh"
source "$PROJECT_ROOT/scripts/validate_gpu.sh"
source "$PROJECT_ROOT/scripts/setup_nlp.sh"
source "$PROJECT_ROOT/scripts/setup_notebook.sh"
source "$PROJECT_ROOT/scripts/validate_tests.sh"
source "$PROJECT_ROOT/scripts/manifest.sh"

trap '_on_error "$LINENO" "$BASH_COMMAND"' ERR

# ===========================================================================
# preflight_fast_checks — spans bootstrap_env + validate_gpu helpers
# ===========================================================================
preflight_fast_checks() {
    _require_project_root
    echo " Running preflight fast checks (pre-venv, seconds)..."
    _msg_info "These run before any expensive operation. Failure saves venv build+sync time on wrong nodes."
    local failures=()
    local output

    output=$(_check_disk_space 2>&1)
    echo "$output" | grep -E "^  [✓⚠ℹ]" | cat
    echo "$output" | grep "^Disk:" | grep -q "." && failures+=("$(echo "$output" | grep '^Disk:')")

    output=$(_check_uv_present 2>&1)
    echo "$output" | grep -E "^  [✓⚠ℹ]" | cat
    echo "$output" | grep "^uv not" | grep -q "." && failures+=("$(echo "$output" | grep '^uv not')")

    output=$(_check_lockfile_present 2>&1)
    echo "$output" | grep -E "^  [✓⚠ℹ]" | cat
    while IFS= read -r line; do
        [[ "$line" =~ ^(pyproject|uv\.lock\ not) ]] && failures+=("$line")
    done <<< "$output"

    output=$(_check_nvidia_smi_present 2>&1)
    echo "$output" | grep -E "^  [✓⚠ℹ]" | cat
    echo "$output" | grep "^nvidia-smi not" | grep -q "." && failures+=("$(echo "$output" | grep '^nvidia-smi not')")

    if command -v nvidia-smi &>/dev/null; then
        output=$(_check_gpu_count_smi 2>&1)
        echo "$output" | grep -E "^  [✓⚠ℹ]" | cat
        echo "$output" | grep "^GPU count" | grep -q "detected" && failures+=("$(echo "$output" | grep '^GPU count')")

        output=$(_check_gpu_name_smi 2>&1)
        echo "$output" | grep -E "^  [✓⚠ℹ]" | cat
        echo "$output" | grep "^GPU name" | grep -q "detected" && failures+=("$(echo "$output" | grep '^GPU name')")

        _check_driver_cuda_smi
    fi

    if [ ${#failures[@]} -gt 0 ]; then
        echo -e "\n${C_RED}${C_BOLD}============================================================${C_RESET}"
        echo -e "${C_RED}${C_BOLD}  PREFLIGHT FAILED — ${#failures[@]} issue(s) must be fixed${C_RESET}"
        echo -e "${C_RED}${C_BOLD}============================================================${C_RESET}"
        for i in "${!failures[@]}"; do
            echo -e "  ${C_RED}[$((i+1))]${C_RESET} ${failures[$i]}"
        done
        echo -e "\n${C_CYAN}  Next steps:${C_RESET}"
        echo -e "${C_CYAN}  • GPU issues:    request ${TARGET_GPU_COUNT}x NVIDIA ${TARGET_GPU_NAME} allocation${C_RESET}"
        echo -e "${C_CYAN}  • Missing files: verify you are in cs1090b_HallucinationLegalRAGChatbots/${C_RESET}"
        echo -e "${C_CYAN}  • After fixing:  bash setup.sh${C_RESET}\n"
        exit 1
    fi
    _msg_ok "All preflight fast checks passed."
}

# ===========================================================================
# Main execution
# ===========================================================================
[ -n "${STEP:-}" ] && echo -e "${C_BOLD}${C_CYAN}Single-step mode: STEP=${STEP}${C_RESET}\n"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo -e "${C_MAGENTA}${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_MAGENTA}${C_BOLD} DRY RUN MODE — no files written, no packages installed${C_RESET}"
    echo -e "${C_MAGENTA}${C_BOLD}============================================================${C_RESET}"
    echo -e "${C_MAGENTA} Mutating steps previewed: ensure_venv | sync_dependencies |${C_RESET}"
    echo -e "${C_MAGENTA}   download_nlp_models | write_repro_env | write_repro_module |${C_RESET}"
    echo -e "${C_MAGENTA}   register_kernel | write_manifest${C_RESET}"
    echo -e "${C_DIM} Read-only steps run normally.${C_RESET}"
    echo -e "${C_MAGENTA}${C_BOLD}============================================================${C_RESET}\n"
fi

echo -e "${C_BOLD}============================================================${C_RESET}"
echo -e "${C_BOLD} cs1090b_HallucinationLegalRAGChatbots — Environment Bootstrap${C_RESET}"
echo -e " Target: ${TARGET_GPU_COUNT}x NVIDIA ${TARGET_GPU_NAME} | Python ${TARGET_PYTHON_VERSION} | torch 2.0.1+cu117"
echo -e " Driver CUDA: ${TARGET_DRIVER_CUDA} (forward-compat) | torch runtime: ${TARGET_TORCH_CUDA_RUNTIME}"
echo -e " Repro: PYTHONHASHSEED=${REPRO_PYTHONHASHSEED} | CUBLAS=${REPRO_CUBLAS_CFG} | RANDOM_SEED=${RANDOM_SEED}"
echo -e " Modes: DRY_RUN=${DRY_RUN:-0} | SKIP_GPU=${SKIP_GPU:-0} | NO_DOWNLOAD=${NO_DOWNLOAD:-0} | NO_JUPYTER=${NO_JUPYTER:-0} | LOG_LEVEL=${LOG_LEVEL}"
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
echo -e "   Activate venv:  source .venv/bin/activate"
echo -e "   Jupyter kernel: HallucinationLegalRAG (${TARGET_PYTHON_VERSION})"
echo -e "   Manifest:       logs/environment_manifest.json"
echo -e ""
echo -e " ${C_BOLD}Notebook Cell 1 (required first line):${C_RESET}"
echo -e "   ${C_CYAN}from src.repro import configure; repro_cfg = configure()${C_RESET}"
echo -e ""
echo -e " ${C_BOLD}Available modes:${C_RESET}"
echo -e "   LOG_LEVEL=0    Quiet — errors only (CI)"
echo -e "   LOG_LEVEL=2    Verbose — all messages including INFO"
echo -e "   VERBOSE=1      Alias for LOG_LEVEL=2"
echo -e "   DRY_RUN=1      Preview all side effects (no writes)"
echo -e "   NO_DOWNLOAD=1  Skip spaCy model download/install"
echo -e "   NO_JUPYTER=1   Skip Jupyter kernel registration"
echo -e "   OFFLINE=1      Use cached wheels; fail if not present"
echo -e "   SKIP_GPU=1     Skip GPU smoke tests"
echo -e "   DEBUG=1        Full shell trace (set -x)"
echo -e "   STEP=<fn>      Run one step only"
echo -e ""
echo -e " ${C_BOLD}Module map (edit the right file):${C_RESET}"
echo -e "   Hardware policy:    scripts/validate_gpu.sh"
echo -e "   NLP model:          scripts/setup_nlp.sh"
echo -e "   Notebook/repro:     scripts/setup_notebook.sh"
echo -e "   Constants/seeds:    scripts/lib.sh"
echo -e "   Manifest Python:    src/manifest_collector.py"
echo -e "   Drift check Python: src/drift_check.py"
echo -e " ${C_BOLD}Seed expt:${C_RESET}   Edit RANDOM_SEED in scripts/lib.sh, re-run, commit"
echo -e "${C_BOLD}============================================================${C_RESET}"
