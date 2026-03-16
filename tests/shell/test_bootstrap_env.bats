#!/usr/bin/env bats
# tests/shell/test_bootstrap_env.bats
# Path: cs1090b_HallucinationLegalRAGChatbots/tests/shell/test_bootstrap_env.bats
# Unit tests for scripts/bootstrap_env.sh — lockfile checks, venv, drift helpers.

load helpers

# ===========================================================================
# _check_lockfile_present
# ===========================================================================

@test "_check_lockfile_present fails when both files missing" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    run _check_lockfile_present
    [ "$status" -eq 1 ]
    assert_contains "$output" "pyproject.toml not found"
    rm -rf "$tmpdir"
}

@test "_check_lockfile_present fails when only uv.lock missing" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    touch "$tmpdir/pyproject.toml"
    run _check_lockfile_present
    [ "$status" -eq 1 ]
    assert_contains "$output" "uv.lock not found"
    rm -rf "$tmpdir"
}

@test "_check_lockfile_present fails when only pyproject.toml missing" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    touch "$tmpdir/uv.lock"
    run _check_lockfile_present
    [ "$status" -eq 1 ]
    assert_contains "$output" "pyproject.toml not found"
    rm -rf "$tmpdir"
}

@test "_check_lockfile_present passes when both files present" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    make_valid_project_root "$tmpdir"
    run _check_lockfile_present
    [ "$status" -eq 0 ]
    assert_contains "$output" "present"
    rm -rf "$tmpdir"
}

# ===========================================================================
# check_lockfile (public step)
# ===========================================================================

@test "check_lockfile exits 1 with structured error when uv.lock missing" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    touch "$tmpdir/pyproject.toml"
    run check_lockfile
    [ "$status" -eq 1 ]
    assert_contains "$output" "uv.lock not found"
    assert_contains "$output" "uv lock"
    assert_contains "$output" "git commit"
    rm -rf "$tmpdir"
}

@test "check_lockfile exits 1 with structured error when pyproject.toml missing" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    run check_lockfile
    [ "$status" -eq 1 ]
    assert_contains "$output" "pyproject.toml not found"
    rm -rf "$tmpdir"
}

@test "check_lockfile passes and prints sha256 when both files present" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    echo "test content" > "$tmpdir/pyproject.toml"
    echo "lock content" > "$tmpdir/uv.lock"
    run check_lockfile
    [ "$status" -eq 0 ]
    assert_contains "$output" "sha256"
    rm -rf "$tmpdir"
}

# ===========================================================================
# _check_uv_present
# ===========================================================================

@test "_check_uv_present passes when uv is on PATH" {
    load_bootstrap_env
    if command -v uv &>/dev/null || [ -x "$HOME/.local/bin/uv" ]; then
        run _check_uv_present
        [ "$status" -eq 0 ]
    else
        skip "uv not installed"
    fi
}

# ===========================================================================
# ensure_venv DRY_RUN behaviour
# ===========================================================================

@test "ensure_venv in DRY_RUN=1 does not create .venv" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    PYTHON="$tmpdir/.venv/bin/python"  # non-existent
    DRY_RUN=1
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    _step_start_time=$(date +%s)
    SETUP_START_TIME=$(date +%s)
    # stub UV so we don't need real uv
    UV="echo"
    run ensure_venv
    # .venv must NOT have been created
    [ ! -d "$tmpdir/.venv" ]
    assert_contains "$output" "DRY RUN"
    assert_contains "$output" "create .venv"
    DRY_RUN=0
    rm -rf "$tmpdir"
}

# ===========================================================================
# sync_dependencies DRY_RUN behaviour
# ===========================================================================

@test "sync_dependencies in DRY_RUN=1 prints preview and does not invoke uv sync" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    echo "" > "$tmpdir/uv.lock"  # minimal lockfile
    DRY_RUN=1
    UV="$(command -v true)"  # stub UV to a no-op
    PYTHON="$(command -v python3)"  # any valid python
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    _step_start_time=$(date +%s)
    SETUP_START_TIME=$(date +%s)
    run sync_dependencies
    assert_contains "$output" "DRY RUN"
    assert_contains "$output" "sync packages from uv.lock"
    DRY_RUN=0
    rm -rf "$tmpdir"
}
