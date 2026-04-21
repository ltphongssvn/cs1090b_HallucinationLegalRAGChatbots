#!/usr/bin/env bats
# Tests for scripts/verify_dvc.sh — DVC pipeline reproducibility checker.
#
# Strategy: every behavioral test runs in a sandboxed tmpdir with:
#   - mocked `dvc` binary in $PATH
#   - minimal fake repo files (a few .dvc pointers, no large files)
# This isolates tests from the real 58GB repo and keeps runtime <5s total.

setup() {
    SCRIPT="$BATS_TEST_DIRNAME/../../scripts/verify_dvc.sh"
    MOCK_BIN="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$MOCK_BIN"
    SANDBOX_PATH="$MOCK_BIN:/usr/bin:/bin"

    # Test sandbox — tiny fake repo to run verify_dvc.sh against
    SANDBOX="$BATS_TEST_TMPDIR/repo"
    mkdir -p "$SANDBOX/data"
    cd "$SANDBOX"
}

mock_dvc() {
    cat > "$MOCK_BIN/dvc" << 'MOCK'
#!/bin/bash
case "$1 ${2:-}" in
    "remote list") echo "${MOCK_DVC_REMOTE_LIST-s3-dvc s3://bucket/dvc}"; exit "${MOCK_DVC_REMOTE_EXIT:-0}" ;;
    "status ") echo "${MOCK_DVC_STATUS:-Data and pipelines are up to date.}"; exit "${MOCK_DVC_STATUS_EXIT:-0}" ;;
    "status --cloud") echo "${MOCK_DVC_CLOUD:-Cache and remote in sync.}"; exit "${MOCK_DVC_CLOUD_EXIT:-0}" ;;
    *) echo "unexpected dvc invocation: $*" >&2; exit 99 ;;
esac
MOCK
    chmod +x "$MOCK_BIN/dvc"
}

# ---------- contract ----------

@test "script exists and is executable" {
    [ -f "$SCRIPT" ] && [ -x "$SCRIPT" ]
}

@test "shebang + strict mode" {
    head -1 "$SCRIPT" | grep -qE '^#!/usr/bin/env bash|^#!/bin/bash'
    grep -q 'set -euo pipefail' "$SCRIPT"
}

@test "--help advertises all three flag names" {
    run bash "$SCRIPT" --help
    [ "$status" -eq 0 ]
    [[ "$output" == *"--help"* ]]
    [[ "$output" == *"--json"* ]]
    [[ "$output" == *"--strict"* ]]
}

@test "unknown flag rejected with nonzero exit" {
    run bash "$SCRIPT" --bogus-flag
    [ "$status" -ne 0 ]
}

# ---------- preflight ----------

@test "exits 2 when dvc not on PATH" {
    run env PATH="/usr/bin:/bin" bash "$SCRIPT"
    [ "$status" -eq 2 ]
    [[ "$output" == *"dvc"* ]]
}

@test "accepts VERIFY_DVC_BIN for dependency injection" {
    mock_dvc
    mv "$MOCK_BIN/dvc" "$MOCK_BIN/custom-dvc"
    run env PATH="/usr/bin:/bin" VERIFY_DVC_BIN="$MOCK_BIN/custom-dvc" bash "$SCRIPT"
    [ "$status" -eq 0 ]
}

# ---------- verification scenarios ----------

@test "all-green: exits 0 with default (non-strict) mode" {
    mock_dvc
    run env PATH="$SANDBOX_PATH" bash "$SCRIPT"
    [ "$status" -eq 0 ]
}

@test "all-green + --strict: exits 0" {
    mock_dvc
    run env PATH="$SANDBOX_PATH" bash "$SCRIPT" --strict
    [ "$status" -eq 0 ]
}

@test "cloud drift: exits 0 in default mode, 1 in --strict mode" {
    mock_dvc
    MOCK_DVC_CLOUD="new: data/foo.jsonl" run env PATH="$SANDBOX_PATH" bash "$SCRIPT"
    [ "$status" -eq 0 ]

    MOCK_DVC_CLOUD="new: data/foo.jsonl" run env PATH="$SANDBOX_PATH" bash "$SCRIPT" --strict
    [ "$status" -eq 1 ]
}

@test "no remote configured: exits 2" {
    mock_dvc
    MOCK_DVC_REMOTE_LIST="" run env PATH="$SANDBOX_PATH" bash "$SCRIPT"
    [ "$status" -eq 2 ]
}

@test "dvc remote list fails: exits 2" {
    mock_dvc
    MOCK_DVC_REMOTE_EXIT=3 run env PATH="$SANDBOX_PATH" bash "$SCRIPT"
    [ "$status" -eq 2 ]
}

@test "stderr contains actionable message when dvc missing" {
    run env PATH="/usr/bin:/bin" bash "$SCRIPT"
    [ "$status" -eq 2 ]
    [[ "$output" == *"dvc"* ]]
    [[ "$output" == *"install"* || "$output" == *"PATH"* || "$output" == *"not found"* ]]
}

# ---------- JSON mode ----------

@test "--json emits parseable JSON with expected top-level keys" {
    mock_dvc
    run env PATH="$SANDBOX_PATH" bash "$SCRIPT" --json
    [ "$status" -eq 0 ]
    echo "$output" | jq -e '.dvc_remote != null and (.pointers_found | type == "number") and (.cloud_drift | type == "boolean") and (.untracked_large_files | type == "array")' > /dev/null
}

@test "--json cloud_drift is true on drift" {
    mock_dvc
    MOCK_DVC_CLOUD="new: data/foo.jsonl" run env PATH="$SANDBOX_PATH" bash "$SCRIPT" --json
    echo "$output" | jq -e '.cloud_drift == true' > /dev/null
}

@test "--json schema_version matches semver pattern" {
    mock_dvc
    run env PATH="$SANDBOX_PATH" bash "$SCRIPT" --json
    [ "$status" -eq 0 ]
    echo "$output" | jq -e '.schema_version | test("^[0-9]+\\.[0-9]+\\.[0-9]+$")' > /dev/null
}

# ---------- pointer enumeration ----------

@test "pointers_found counts .dvc files in working directory" {
    mock_dvc
    # Create 3 fake .dvc pointers
    touch "$SANDBOX/a.dvc" "$SANDBOX/data/b.dvc" "$SANDBOX/data/c.dvc"
    run env PATH="$SANDBOX_PATH" bash "$SCRIPT" --json
    [ "$status" -eq 0 ]
    actual=$(echo "$output" | jq -r '.pointers_found')
    [ "$actual" -eq 3 ]
}

# ---------- large-file false-positive guards ----------

@test "untracked_large_files empty when no files exceed threshold" {
    mock_dvc
    # Empty sandbox — no files > 100MB
    run env PATH="$SANDBOX_PATH" bash "$SCRIPT" --json
    [ "$status" -eq 0 ]
    count=$(echo "$output" | jq -r '.untracked_large_files | length')
    [ "$count" -eq 0 ]
}

@test "untracked_large_files scan excludes .git .venv .dvc" {
    mock_dvc
    # Simulate a suspicious large file inside .git (should be ignored)
    mkdir -p "$SANDBOX/.git"
    # Use sparse file to avoid actually consuming disk
    dd if=/dev/zero of="$SANDBOX/.git/bigpack" bs=1 count=0 seek=200M 2>/dev/null
    run env PATH="$SANDBOX_PATH" bash "$SCRIPT" --json
    [ "$status" -eq 0 ]
    ! echo "$output" | jq -r '.untracked_large_files[]' | grep -qE '\.git/|\.venv/|\.dvc/'
}

@test "files tracked by sibling .dvc pointer are excluded from untracked_large_files" {
    mock_dvc
    # Create a tracked large file: foo.jsonl + foo.jsonl.dvc
    dd if=/dev/zero of="$SANDBOX/foo.jsonl" bs=1 count=0 seek=200M 2>/dev/null
    touch "$SANDBOX/foo.jsonl.dvc"
    run env PATH="$SANDBOX_PATH" bash "$SCRIPT" --json
    [ "$status" -eq 0 ]
    ! echo "$output" | jq -r '.untracked_large_files[]' | grep -q 'foo.jsonl$'
}
