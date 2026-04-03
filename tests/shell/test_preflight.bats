#!/usr/bin/env bats
# tests/shell/test_preflight.bats
# Path: cs1090b_HallucinationLegalRAGChatbots/tests/shell/test_preflight.bats
# Integration tests for preflight_fast_checks and DRY_RUN propagation across
# mutating steps. Tests source setup.sh in a temp environment.

load helpers

# ===========================================================================
# _check_disk_space
# ===========================================================================

@test "_check_disk_space passes on a real filesystem with space" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    TARGET_MIN_DISK_GB=1  # conservative — passes on any real filesystem
    run _check_disk_space
    [ "$status" -eq 0 ]
    assert_contains "$output" "free"
    rm -rf "$tmpdir"
}

@test "_check_disk_space fails when TARGET_MIN_DISK_GB exceeds available" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    # Set an absurdly high threshold to force failure
    TARGET_MIN_DISK_GB=999999
    run _check_disk_space
    [ "$status" -eq 1 ]
    assert_contains "$output" "Disk:"
    rm -rf "$tmpdir"
}

# ===========================================================================
# DRY_RUN propagation — mutating steps must not write files
# ===========================================================================

@test "write_repro_env in DRY_RUN=1 does not write .env" {
    # Source setup_notebook.sh helpers
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/setup_notebook.sh"
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    touch "$tmpdir/pyproject.toml"  # satisfy _require_project_root
    DRY_RUN=1
    RANDOM_SEED=0; REPRO_PYTHONHASHSEED=0
    REPRO_CUBLAS_CFG=":4096:8"; REPRO_TOKENIZERS_PAR="false"
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    _step_start_time=$(date +%s); SETUP_START_TIME=$(date +%s)
    run write_repro_env
    [ ! -f "$tmpdir/.env" ]
    assert_contains "$output" "DRY RUN"
    DRY_RUN=0
    rm -rf "$tmpdir"
}

@test "write_repro_module in DRY_RUN=1 does not write src/repro.py" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/setup_notebook.sh"
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    touch "$tmpdir/.env"  # satisfy _require_repro_env
    DRY_RUN=1
    RANDOM_SEED=0; REPRO_PYTHONHASHSEED=0
    REPRO_CUBLAS_CFG=":4096:8"; REPRO_TOKENIZERS_PAR="false"
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    _step_start_time=$(date +%s); SETUP_START_TIME=$(date +%s)
    run write_repro_module
    [ ! -f "$tmpdir/src/repro.py" ]
    assert_contains "$output" "DRY RUN"
    DRY_RUN=0
    rm -rf "$tmpdir"
}

@test "write_manifest in DRY_RUN=1 does not write logs/environment_manifest.json" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/manifest.sh"
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    echo "lock" > "$tmpdir/uv.lock"
    # populate DETECTED_* to satisfy _require_hardware_detected
    DETECTED_GPU_NAME="L4"; DETECTED_GPU_COUNT="4"
    DETECTED_TORCH_CUDA="11.7"; DETECTED_CUDNN="8500"
    DETECTED_DRIVER_CUDA="12.8"; HARDWARE_MATCH="true"
    DRY_RUN=1
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    _step_start_time=$(date +%s); SETUP_START_TIME=$(date +%s)
    PYTHON="$(command -v python3)"
    run write_manifest
    [ ! -f "$tmpdir/logs/environment_manifest.json" ]
    assert_contains "$output" "DRY RUN"
    DRY_RUN=0
    rm -rf "$tmpdir"
}

@test "register_kernel in DRY_RUN=1 does not invoke ipykernel install" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/setup_notebook.sh"
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    DRY_RUN=1
    TARGET_PYTHON_VERSION="3.11.9"
    PYTHON="$(command -v python3)"  # valid python, but won't be called
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    _step_start_time=$(date +%s); SETUP_START_TIME=$(date +%s)
    run register_kernel
    assert_contains "$output" "DRY RUN"
    assert_contains "$output" "kernelspec"
    DRY_RUN=0
    rm -rf "$tmpdir"
}

# ===========================================================================
# Golden output — error messages contain required remediation fields
# ===========================================================================

@test "check_lockfile error message contains uv lock command" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    source "$PROJECT_ROOT/scripts/bootstrap_env.sh"
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    touch "$tmpdir/pyproject.toml"
    run check_lockfile
    assert_contains "$output" "uv lock"
    assert_contains "$output" "git add uv.lock"
    assert_contains "$output" "git commit"
    rm -rf "$tmpdir"
}

@test "_require_project_root error message contains cd command" {
    source "$PROJECT_ROOT/scripts/lib.sh"
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    run _require_project_root
    assert_contains "$output" "cd ~"
    rm -rf "$tmpdir"
}
