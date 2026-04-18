#!/usr/bin/env bats
load 'helpers.bash'

setup() {
    REPO_ROOT="$(cd "$BATS_TEST_DIRNAME/../.." && pwd)"
    cd "$REPO_ROOT"
}

@test "help flag works" {
    run bash scripts/monitor_baseline.sh --help
    [ "$status" -eq 0 ]
    [[ "$output" == *"Monitor"* ]]
}

@test "missing PID file produces clear message, exits 0" {
    rm -f logs/baseline_prep.pid
    run bash scripts/monitor_baseline.sh
    [ "$status" -eq 0 ]
    [[ "$output" == *"no PID file"* ]] || [[ "$output" == *"not running"* ]]
}

@test "missing log directory produces clear message" {
    tmpdir=$(mktemp -d)
    run bash -c "cd $tmpdir && bash $REPO_ROOT/scripts/monitor_baseline.sh"
    rm -rf "$tmpdir"
    [[ "$output" == *"no log"* ]] || [[ "$output" == *"No artifacts"* ]]
}

@test "auto-discovers latest log via glob, not hardcoded filename" {
    run grep -E "ls -t logs/baseline_prep_\*.log|logs/baseline_prep_\*\.log" scripts/monitor_baseline.sh
    [ "$status" -eq 0 ]
}

@test "reads PID dynamically from PID file" {
    run grep -E "cat.*logs/baseline_prep.pid|cat \"\\\$PID_FILE\"" scripts/monitor_baseline.sh
    [ "$status" -eq 0 ]
}

@test "uses %mem in ps format (human-readable)" {
    run grep -E "%mem|%MEM" scripts/monitor_baseline.sh
    [ "$status" -eq 0 ]
}
