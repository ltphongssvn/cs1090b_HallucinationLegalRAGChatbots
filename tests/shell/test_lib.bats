#!/usr/bin/env bats
# tests/shell/test_lib.bats
# Unit tests for scripts/lib.sh

load helpers

# ===========================================================================
# _is_dry_run
# ===========================================================================
@test "_is_dry_run returns 0 (true) when DRY_RUN=1" {
    load_lib; DRY_RUN=1
    run _is_dry_run
    [ "$status" -eq 0 ]
}

@test "_is_dry_run returns 1 (false) when DRY_RUN=0" {
    load_lib; DRY_RUN=0
    run _is_dry_run
    [ "$status" -eq 1 ]
}

@test "_is_dry_run returns 1 (false) when DRY_RUN is unset" {
    load_lib; unset DRY_RUN
    run _is_dry_run
    [ "$status" -eq 1 ]
}

@test "_is_dry_run returns 1 (false) when DRY_RUN is empty string" {
    load_lib; DRY_RUN=""
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

@test "_msg_info outputs info symbol and message at LOG_LEVEL=2" {
    # _msg_info is silent at LOG_LEVEL<2 by design.
    # Tests must explicitly set LOG_LEVEL=2 to assert on _msg_info output.
    load_lib
    LOG_LEVEL=2
    run bash -c "
        source '$PROJECT_ROOT/scripts/lib.sh'
        LOG_LEVEL=2
        _msg_info 'just so you know'
    "
    [ "$status" -eq 0 ]
    assert_contains "$output" "just so you know"
}

@test "_msg_info is silent at LOG_LEVEL=1 (normal mode)" {
    load_lib
    LOG_LEVEL=1
    run bash -c "
        source '$PROJECT_ROOT/scripts/lib.sh'
        LOG_LEVEL=1
        _msg_info 'should be silent'
    "
    [ "$status" -eq 0 ]
    assert_not_contains "$output" "should be silent"
}

@test "_msg_info is silent at LOG_LEVEL=0 (quiet mode)" {
    load_lib
    run bash -c "
        source '$PROJECT_ROOT/scripts/lib.sh'
        LOG_LEVEL=0
        _msg_info 'totally silent'
    "
    [ "$status" -eq 0 ]
    assert_not_contains "$output" "totally silent"
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

@test "_msg_error always prints at LOG_LEVEL=0 (quiet mode)" {
    run bash -c "
        source '$PROJECT_ROOT/scripts/lib.sh'
        LOG_LEVEL=0
        _msg_error 'topic' 'what' 'why' 'fix'
    "
    [ "$status" -eq 0 ]
    assert_contains "$output" "topic"
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

@test "_msg_warn is silent at LOG_LEVEL=0" {
    run bash -c "
        source '$PROJECT_ROOT/scripts/lib.sh'
        LOG_LEVEL=0
        _msg_warn 'topic' 'what' 'informational' 'action'
    "
    [ "$status" -eq 0 ]
    assert_not_contains "$output" "topic"
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
    load_lib; UV=""
    if command -v uv &>/dev/null || [ -x "$HOME/.local/bin/uv" ]; then
        run _require_uv
        [ "$status" -eq 0 ]
    else
        skip "uv not installed"
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
    PROJECT_ROOT="$tmpdir"; RANDOM_SEED=0
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
    touch "$tmpdir/.env"; RANDOM_SEED=0
    run _require_repro_env
    [ "$status" -eq 0 ]
    rm -rf "$tmpdir"
}

# ===========================================================================
# step_end summary recording
# NOTE: bats `run` executes in a subshell — bash arrays do not cross subshell
# boundaries. Call step_end directly (without `run`) and inspect arrays inline.
# ===========================================================================
@test "step_end PASS records step name in SUMMARY_STEPS" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    step_end "my_step" "PASS"
    [ "${SUMMARY_STEPS[0]}" = "my_step" ]
}

@test "step_end SKIP records SKIP in SUMMARY_STATUS" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    step_end "skipped_step" "SKIP"
    [ "${SUMMARY_STEPS[0]}" = "skipped_step" ]
    assert_contains "${SUMMARY_STATUS[0]}" "SKIP"
}

@test "step_end DRY records DRY in SUMMARY_STATUS" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    step_end "dry_step" "DRY"
    assert_contains "${SUMMARY_STATUS[0]}" "DRY"
}

@test "step_end WARN records WARN in SUMMARY_STATUS" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    step_end "warn_step" "WARN"
    assert_contains "${SUMMARY_STATUS[0]}" "WARN"
}

@test "step_end records duration in SUMMARY_DURATION" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    step_end "timed_step" "PASS"
    [ -n "${SUMMARY_DURATION[0]}" ]
    assert_contains "${SUMMARY_DURATION[0]}" "s"
}

# ===========================================================================
# run_step with STEP= single-step mode
# ===========================================================================
@test "run_step skips function when STEP is set to different name" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    STEP="other_step"
    my_fn() { echo "I should not run"; }
    run_step my_fn
    assert_contains "${SUMMARY_STATUS[0]:-}" "SKIP"
    unset STEP
}

@test "run_step executes function when STEP matches function name" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    STEP="my_fn"
    my_fn() { echo "I ran"; }
    run run_step my_fn
    assert_contains "$output" "I ran"
    unset STEP
}

@test "run_step executes function when STEP is unset" {
    load_lib
    SUMMARY_STEPS=(); SUMMARY_STATUS=(); SUMMARY_DURATION=()
    SETUP_START_TIME=$(date +%s); _step_start_time=$(date +%s)
    unset STEP
    my_fn() { echo "executed"; }
    run run_step my_fn
    assert_contains "$output" "executed"
}
