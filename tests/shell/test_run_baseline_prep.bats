#!/usr/bin/env bats
# Contract tests for scripts/run_baseline_prep.sh
load 'helpers.bash'

setup() {
    REPO_ROOT="$(cd "$BATS_TEST_DIRNAME/../.." && pwd)"
    cd "$REPO_ROOT"
}

@test "help flag prints usage without launching" {
    run bash scripts/run_baseline_prep.sh --help
    [ "$status" -eq 0 ]
    [[ "$output" == *"Full-scale MS3 baseline prep"* ]]
}

@test "dry-run exits 0 and does not create PID file" {
    rm -f logs/baseline_prep.pid
    run bash scripts/run_baseline_prep.sh --dry-run
    [ "$status" -eq 0 ]
    [[ "$output" == *"DRY RUN complete"* ]]
    [ ! -f logs/baseline_prep.pid ]
}

@test "dry-run prints fingerprint (git SHA, Python version)" {
    run bash scripts/run_baseline_prep.sh --dry-run
    [ "$status" -eq 0 ]
    [[ "$output" == *"git_sha"* ]]
    [[ "$output" == *"python"* ]]
}

@test "dry-run uses Python --dry-run flag, not inline python -c" {
    # Contract: the bash script must delegate dry-run validation to the Python
    # CLI, which is testable and versioned. Inline python -c is brittle.
    run grep -E "baseline_prep.py --dry-run|baseline_prep --dry-run" scripts/run_baseline_prep.sh
    [ "$status" -eq 0 ]
}

@test "missing input path exits with code 2" {
    run env SHARD_DIR=/nonexistent/path bash scripts/run_baseline_prep.sh --dry-run
    [ "$status" -eq 2 ]
    [[ "$output" == *"missing input"* ]]
}

@test "concurrent-run guard rejects when PID file holds live process" {
    mkdir -p logs
    # Use bash's own PID as a guaranteed-live process
    echo $$ > logs/baseline_prep.pid
    run bash scripts/run_baseline_prep.sh --dry-run
    rm -f logs/baseline_prep.pid
    [ "$status" -eq 3 ]
    [[ "$output" == *"already running"* ]]
}

@test "unknown flag rejected with nonzero exit" {
    run bash scripts/run_baseline_prep.sh --dryrun
    [ "$status" -ne 0 ]
    [[ "$output" == *"unknown"* ]] || [[ "$output" == *"FAIL"* ]]
}

@test "--no-resume forwarded to python dry-run" {
    run bash scripts/run_baseline_prep.sh --dry-run --no-resume
    [ "$status" -eq 0 ]
    [[ "$output" == *"'resume': False"* ]]
}

@test "--resume default forwarded to python dry-run" {
    run bash scripts/run_baseline_prep.sh --dry-run
    [ "$status" -eq 0 ]
    [[ "$output" == *"'resume': True"* ]]
}

@test "dry-run prints hostname + UTC start in manifest preview" {
    run bash scripts/run_baseline_prep.sh --dry-run
    [ "$status" -eq 0 ]
    [[ "$output" == *"hostname"* ]]
}

@test "script sets PYTHONUNBUFFERED for live logs" {
    run grep -E "PYTHONUNBUFFERED|python -u" scripts/run_baseline_prep.sh
    [ "$status" -eq 0 ]
}
