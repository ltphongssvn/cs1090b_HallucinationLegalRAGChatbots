#!/usr/bin/env bats
# tests/shell/test_lib.bats
# Path: cs1090b_HallucinationLegalRAGChatbots/tests/shell/test_lib.bats
# Unit tests for scripts/lib.sh — guards, messaging, dry-run, step framework.

load helpers

# ===========================================================================
# _is_dry_run
# ===========================================================================

@test "_is_dry_run returns 0 (true) when DRY_RUN=1" {
    load_lib
    DRY_RUN=1
    run _is_dry_run
    [ "$status" -eq 0 ]
}

@test "_is_dry_run returns 1 (false) when DRY_RUN=0" {
    load_lib
    DRY_RUN=0
    run _is_dry_run
    [ "$status" -eq 1 ]
}

@test "_is_dry_run returns 1 (false) when DRY_RUN is unset" {
    load_lib
    unset DRY_RUN
    run _is_dry_run
    [ "$status" -eq 1 ]
}

@test "_is_dry_run returns 1 (false) when DRY_RUN is empty string" {
    load_lib
    DRY_RUN=""
    run _is_dry_run
    [ "$status" -eq 1 ]
}

# ===========================================================================
# _msg_dry_run
# ===========================================================================

@test "_msg_dry_run outputs DRY RUN prefix and both arguments" {
    load_lib
    run _msg_dry_run "write file" "/tmp/test.txt"
    [ "$status" -eq 0 ]
    assert_contains "$output" "DRY RUN"
    assert_contains "$output" "write file"
    assert_contains "$output" "/tmp/test.txt"
}

# ===========================================================================
# _msg_ok / _msg_info / _msg_skip
# ===========================================================================

@test "_msg_ok outputs checkmark and message" {
    load_lib
    run _msg_ok "everything is fine"
    [ "$status" -eq 0 ]
    assert_contains "$output" "everything is fine"
}

@test "_msg_info outputs info symbol and message" {
    load_lib
    run _msg_info "just so you know"
    [ "$status" -eq 0 ]
    assert_contains "$output" "just so you know"
}

@test "_msg_skip outputs skip symbol and message" {
    load_lib
    run _msg_skip "skipping this"
    [ "$status" -eq 0 ]
    assert_contains "$output" "skipping this"
}

# ===========================================================================
# _msg_error
# ===========================================================================

@test "_msg_error outputs all four structured fields" {
    load_lib
    run _msg_error "Test topic" "what happened" "why it matters" "how to fix"
    [ "$status" -eq 0 ]
    assert_contains "$output" "Test topic"
    assert_contains "$output" "what happened"
    assert_contains "$output" "why it matters"
    assert_contains "$output" "how to fix"
}

@test "_msg_error outputs What/Why/Fix labels" {
    load_lib
    run _msg_error "T" "W" "Y" "F"
    assert_contains "$output" "What:"
    assert_contains "$output" "Why:"
    assert_contains "$output" "Fix:"
}

# ===========================================================================
# _msg_warn
# ===========================================================================

@test "_msg_warn action-required outputs ACTION REQUIRED tag" {
    load_lib
    run _msg_warn "Topic" "what" "action-required" "do this"
    [ "$status" -eq 0 ]
    assert_contains "$output" "ACTION REQUIRED"
    assert_contains "$output" "Topic"
    assert_contains "$output" "do this"
}

@test "_msg_warn informational outputs informational tag" {
    load_lib
    run _msg_warn "Topic" "what" "informational" "fyi"
    [ "$status" -eq 0 ]
    assert_contains "$output" "informational"
}

# ===========================================================================
# _require_project_root
# ===========================================================================

@test "_require_project_root exits 1 when pyproject.toml missing" {
    load_lib
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    run _require_project_root
    [ "$status" -eq 1 ]
    assert_contains "$output" "pyproject.toml"
    rm -rf "$tmpdir"
}

@test "_require_project_root passes when pyproject.toml present" {
    load_lib
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    touch "$tmpdir/pyproject.toml"
    run _require_project_root
    [ "$status" -eq 0 ]
    rm -rf "$tmpdir"
}

# ===========================================================================
# _require_uv
# ===========================================================================

@test "_require_uv auto-resolves when uv is on PATH" {
    load_lib
    UV=""
    # uv is on PATH in the venv environment — should resolve successfully
    if command -v uv &>/dev/null || [ -x "$HOME/.local/bin/uv" ]; then
        run _require_uv
        [ "$status" -eq 0 ]
    else
        skip "uv not installed — cannot test auto-resolve"
    fi
}

@test "_require_uv exits 1 when UV is broken path" {
    load_lib
    UV="/nonexistent/path/to/uv"
    run _require_uv
    [ "$status" -eq 1 ]
    assert_contains "$output" "does not execute"
}

# ===========================================================================
# _require_repro_env
# ===========================================================================

@test "_require_repro_env exits 1 when .env missing" {
    load_lib
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    RANDOM_SEED=0
    run _require_repro_env
    [ "$status" -eq 1 ]
    assert_contains "$output" ".env not found"
    rm -rf "$tmpdir"
}

@test "_require_repro_env exits 1 when RANDOM_SEED is empty" {
    load_lib
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    touch "$tmpdir/.env"
    RANDOM_SEED=""
    run _require_repro_env
    [ "$status" -eq 1 ]
    assert_contains "$output" "RANDOM_SEED not set"
    rm -rf "$tmpdir"
}

@test "_require_repro_env passes when .env present and RANDOM_SEED set" {
    load_lib
    local tmpdir; tmpdir=$(mktemp -d)
    PROJECT_ROOT="$tmpdir"
    touch "$tmpdir/.env"
    RANDOM_SEED=0
    run _require_repro_env
    [ "$status" -eq 0 ]
    rm -rf "$tmpdir"
}

# ===========================================================================
# step_end summary recording
# ===========================================================================

@test "step_end PASS records step in SUMMARY_STEPS" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    _step_start_time=$(date +%s)
    step_end "my_step" "PASS"
    [ "${SUMMARY_STEPS[0]}" = "my_step" ]
}

@test "step_end SKIP records step with SKIP status" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    _step_start_time=$(date +%s)
    step_end "skipped_step" "SKIP"
    [ "${SUMMARY_STEPS[0]}" = "skipped_step" ]
    assert_contains "${SUMMARY_STATUS[0]}" "SKIP"
}

@test "step_end DRY records step with DRY status" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    _step_start_time=$(date +%s)
    step_end "dry_step" "DRY"
    assert_contains "${SUMMARY_STATUS[0]}" "DRY"
}

# ===========================================================================
# run_step with STEP= single-step mode
# ===========================================================================

@test "run_step skips function when STEP is set to different name" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    STEP="other_step"
    my_fn() { echo "I should not run"; }
    run_step my_fn
    # Function should NOT have executed — status entry should be SKIP
    assert_contains "${SUMMARY_STATUS[0]:-}" "SKIP"
    unset STEP
}

@test "run_step executes function when STEP matches function name" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    STEP="my_fn"
    _step_start_time=$(date +%s)
    my_fn() { echo "I ran"; }
    run run_step my_fn
    assert_contains "$output" "I ran"
    unset STEP
}

@test "run_step executes function when STEP is unset" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    unset STEP
    _step_start_time=$(date +%s)
    SETUP_START_TIME=$(date +%s)
    my_fn() { echo "executed"; }
    run run_step my_fn
    assert_contains "$output" "executed"
}
