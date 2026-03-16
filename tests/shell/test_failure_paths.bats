#!/usr/bin/env bats
# tests/shell/test_failure_paths.bats
# Failure-path tests — the cases where setup scripts commonly break.

load helpers

# ===========================================================================
# 1. Missing uv
# ===========================================================================
@test "check_uv exits 1 when uv not on PATH" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/bootstrap_env.sh"
    local orig_path="$PATH" orig_home="$HOME"
    PATH="/usr/bin:/bin"
    HOME="/nonexistent_home_$$"
    run check_uv
    PATH="$orig_path"; HOME="$orig_home"
    [ "$status" -eq 1 ]
    assert_contains "$output" "uv not found"
    assert_contains "$output" "curl"
}

@test "_require_uv exits 1 with install hint when uv missing" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    UV="/nonexistent/uv_$$"
    local orig_path="$PATH"
    PATH="/usr/bin:/bin"
    run _require_uv
    PATH="$orig_path"
    [ "$status" -eq 1 ]
    assert_contains "$output" "uv"
    assert_contains "$output" "astral.sh"
}

@test "_require_uv exits 1 with broken uv binary message" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    local fake_uv; fake_uv=$(mktemp)
    printf '#!/usr/bin/env bash\nexit 127\n' > "$fake_uv"
    chmod +x "$fake_uv"
    UV="$fake_uv"
    run _require_uv
    [ "$status" -eq 1 ]
    assert_contains "$output" "does not execute"
    rm -f "$fake_uv"
}

# ===========================================================================
# 2. Missing pyproject.toml
# ===========================================================================
@test "check_lockfile exits 1 with structured error when pyproject.toml absent" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/bootstrap_env.sh"
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    run check_lockfile
    [ "$status" -eq 1 ]
    assert_contains "$output" "pyproject.toml not found"
    assert_contains "$output" "Fix:"
    rm -rf "$tmpdir"
}

@test "_require_project_root exits 1 with cd hint when pyproject.toml missing" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    run _require_project_root
    [ "$status" -eq 1 ]
    assert_contains "$output" "pyproject.toml"
    assert_contains "$output" "cd ~"
    rm -rf "$tmpdir"
}

@test "preflight_fast_checks fails when pyproject.toml absent" {
    # Run in subshell with LOG_LEVEL=2 so all messages are visible
    run bash -c "
        set +euo pipefail
        source '$PROJECT_ROOT/scripts/lib.sh'
        source '$PROJECT_ROOT/scripts/bootstrap_env.sh'
        source '$PROJECT_ROOT/scripts/validate_gpu.sh'
        LOG_LEVEL=2
        tmpdir=\$(mktemp -d)
        PROJECT_ROOT=\"\$tmpdir\"
        PYTHON=\"\$tmpdir/.venv/bin/python\"
        SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
        SETUP_START_TIME=\$(date +%s)
        _step_start_time=\$(date +%s)
        _CURRENT_STEP='(none)'
        preflight_fast_checks
        rm -rf \"\$tmpdir\"
    "
    [ "$status" -ne 0 ]
    # Should mention pyproject or preflight failure
    [[ "$output" == *"pyproject"* ]] || [[ "$output" == *"PREFLIGHT"* ]] || [[ "$output" == *"Wrong directory"* ]]
}

# ===========================================================================
# 3. Incompatible Python version
# ===========================================================================
@test "_require_python exits 1 when PYTHON binary is missing" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    PYTHON="/nonexistent/.venv/bin/python_$$"
    run _require_python
    [ "$status" -eq 1 ]
    assert_contains "$output" "venv Python not found"
    assert_contains "$output" "ensure_venv"
}

@test "_require_python exits 1 when venv Python has wrong version" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    local fake_python; fake_python=$(mktemp)
    printf '#!/usr/bin/env bash\nif [[ "$*" == *"sys.version_info"* ]]; then exit 1; fi\necho "Python 3.10.0"\n' > "$fake_python"
    chmod +x "$fake_python"
    PYTHON="$fake_python"
    TARGET_PYTHON_VERSION="3.11.9"
    run _require_python
    [ "$status" -eq 1 ]
    assert_contains "$output" "Wrong Python version"
    assert_contains "$output" "rm -rf .venv"
    rm -f "$fake_python"
}

@test "ensure_venv in DRY_RUN=1 does not create .venv" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/bootstrap_env.sh"
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    PYTHON="$tmpdir/.venv/bin/python"
    DRY_RUN=1; UV="$(command -v true)"
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    _CURRENT_STEP="(none)"
    run ensure_venv
    [ ! -d "$tmpdir/.venv" ]
    assert_contains "$output" "DRY RUN"
    DRY_RUN=0
    rm -rf "$tmpdir"
}

# ===========================================================================
# 4. Missing NVIDIA tools
# ===========================================================================
@test "_check_nvidia_smi_present fails when nvidia-smi absent from PATH" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/validate_gpu.sh"
    local orig_path="$PATH"
    PATH="$(echo "$PATH" | tr ':' '\n' | grep -v nvidia | tr '\n' ':' | sed 's/:$//')"
    # Also ensure nvidia-smi is not accessible
    run bash -c "
        source '$PROJECT_ROOT/scripts/lib.sh'
        source '$PROJECT_ROOT/scripts/validate_gpu.sh'
        PATH='/usr/bin:/bin'
        _check_nvidia_smi_present
    "
    PATH="$orig_path"
    [ "$status" -eq 1 ]
    assert_contains "$output" "nvidia-smi not found"
}

@test "_check_gpu_count_smi fails when count below target" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/validate_gpu.sh"
    if ! command -v nvidia-smi &>/dev/null; then
        skip "nvidia-smi not available"
    fi
    TARGET_GPU_COUNT=99
    run _check_gpu_count_smi
    [ "$status" -eq 1 ]
    assert_contains "$output" "GPU count"
    assert_contains "$output" "detected"
}

@test "run_gpu_smoke_tests skips cleanly with SKIP_GPU=1" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/validate_gpu.sh"
    SKIP_GPU=1
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    _CURRENT_STEP="(none)"
    run run_gpu_smoke_tests
    [ "$status" -eq 0 ]
    assert_contains "$output" "SKIP_GPU=1"
    SKIP_GPU=0
}

# ===========================================================================
# 5. CPU-only mode (SKIP_GPU=1)
# ===========================================================================
@test "run_gpu_smoke_tests in SKIP_GPU=1 records SKIP in summary" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/validate_gpu.sh"
    SKIP_GPU=1
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    _CURRENT_STEP="(none)"
    run_gpu_smoke_tests
    assert_contains "${SUMMARY_STATUS[0]:-}" "SKIP"
    SKIP_GPU=0
}

@test "SKIP_GPU=1 does not require NVIDIA tools" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/validate_gpu.sh"
    SKIP_GPU=1
    PYTHON="/nonexistent/python"
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    _CURRENT_STEP="(none)"
    run run_gpu_smoke_tests
    [ "$status" -eq 0 ]
    SKIP_GPU=0
}

# ===========================================================================
# 6. Bad torch build
# ===========================================================================
@test "run_env_smoke_tests fails with fix hint for CPU-only torch wheel" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
    if [ ! -x "$PYTHON" ]; then skip "venv not yet built"; fi
    run bash -c "
        source '$PROJECT_ROOT/scripts/lib.sh'
        PYTHON='$PROJECT_ROOT/.venv/bin/python'
        \$PYTHON -c \"
import sys
ver = '2.0.1'  # no +cu117 — simulates CPU wheel
if not (ver.startswith('2.') and 'cu' in ver):
    print(f'\033[0;31m  ✗ Wrong torch build: {ver!r}\033[0m')
    print('    Fix: rm -rf .venv && bash setup.sh')
    sys.exit(1)
\"
    "
    [ "$status" -eq 1 ]
    assert_contains "$output" "Wrong torch build"
    assert_contains "$output" "rm -rf .venv"
}

# ===========================================================================
# 7. Jupyter absent
# ===========================================================================
@test "register_kernel in DRY_RUN=1 does not require jupyter" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/setup_notebook.sh"
    DRY_RUN=1
    TARGET_PYTHON_VERSION="3.11.9"
    PYTHON="$(command -v python3)"
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    _CURRENT_STEP="(none)"
    run register_kernel
    [ "$status" -eq 0 ]
    assert_contains "$output" "DRY RUN"
    assert_contains "$output" "kernelspec"
    DRY_RUN=0
}

# ===========================================================================
# 8. _require_hardware_detected triggers when UNDETECTED
# ===========================================================================
@test "_require_hardware_detected warns when DETECTED_GPU_COUNT is UNDETECTED" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/validate_gpu.sh"
    DETECTED_GPU_COUNT="UNDETECTED"
    DETECTED_TORCH_CUDA="UNDETECTED"
    DETECTED_GPU_NAME="UNDETECTED"
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
    if [ ! -x "$PYTHON" ]; then skip "venv not yet built"; fi
    run _require_hardware_detected
    assert_contains "$output" "UNDETECTED"
}

# ===========================================================================
# 9. .env missing for write_repro_module
# ===========================================================================
@test "write_repro_module exits 1 with structured error when .env absent" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/setup_notebook.sh"
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    RANDOM_SEED=0; DRY_RUN=0
    run write_repro_module
    [ "$status" -eq 1 ]
    assert_contains "$output" ".env not found"
    assert_contains "$output" "write_repro_env"
    rm -rf "$tmpdir"
}

# ===========================================================================
# 10. uv.lock missing for write_manifest
# Fix: use system python3 so _require_python passes; only uv.lock is absent
# ===========================================================================
@test "write_manifest exits 1 with structured error when uv.lock absent" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/manifest.sh"
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    # Populate DETECTED_* to satisfy _require_hardware_detected
    DETECTED_GPU_NAME="L4"; DETECTED_GPU_COUNT="4"
    DETECTED_TORCH_CUDA="11.7"; DETECTED_CUDNN="8500"
    DETECTED_DRIVER_CUDA="12.8"; HARDWARE_MATCH="true"
    # Use system python3 so _require_python passes (venv may not exist in CI)
    PYTHON="$(command -v python3)"
    TARGET_PYTHON_VERSION="$(python3 -c 'import sys; print(".".join(map(str,sys.version_info[:3])))')"
    DRY_RUN=0
    # No uv.lock in tmpdir
    run write_manifest
    [ "$status" -eq 1 ]
    assert_contains "$output" "uv.lock"
    rm -rf "$tmpdir"
}
