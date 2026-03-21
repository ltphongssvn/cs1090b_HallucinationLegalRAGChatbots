#!/usr/bin/env bats
# tests/shell/test_bootstrap_env.bats
# Unit tests for scripts/bootstrap_env.sh

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
    # Call directly (not via run) to avoid set -e subshell issues with && patterns
    _check_lockfile_present
    local ret=$?
    [ "$ret" -eq 0 ]
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

@test "check_lockfile exits 1 when pyproject.toml missing (_require_project_root fires first)" {
    # check_lockfile calls _require_project_root first — emits "Wrong directory" error
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    run check_lockfile
    [ "$status" -eq 1 ]
    # _require_project_root fires before lockfile check — assert on actual message
    assert_contains "$output" "ERROR"
    assert_contains "$output" "pyproject.toml"
    rm -rf "$tmpdir"
}

@test "check_lockfile passes and prints sha256 when both files present" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    echo "test" > "$tmpdir/pyproject.toml"
    echo "lock" > "$tmpdir/uv.lock"
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
# ensure_venv DRY_RUN
# ===========================================================================
@test "ensure_venv in DRY_RUN=1 does not create .venv" {
    load_bootstrap_env
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
    assert_contains "$output" "create .venv"
    DRY_RUN=0
    rm -rf "$tmpdir"
}

# ===========================================================================
# sync_dependencies DRY_RUN
# DRY_RUN guard is now BEFORE _require_python, so no venv needed in dry-run.
# ===========================================================================
@test "sync_dependencies in DRY_RUN=1 prints preview without requiring venv Python" {
    load_bootstrap_env
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    echo "" > "$tmpdir/uv.lock"
    DRY_RUN=1
    UV="$(command -v true)"
    # PYTHON deliberately points to nonexistent path — DRY_RUN must not invoke it
    PYTHON="/nonexistent/.venv/bin/python_$$"
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    _CURRENT_STEP="(none)"
    run sync_dependencies
    [ "$status" -eq 0 ]
    assert_contains "$output" "DRY RUN"
    assert_contains "$output" "sync packages from uv.lock"
    DRY_RUN=0
    rm -rf "$tmpdir"
}
