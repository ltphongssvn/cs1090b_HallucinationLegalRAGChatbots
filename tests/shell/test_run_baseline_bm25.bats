#!/usr/bin/env bats
load 'helpers.bash'

setup() {
    REPO_ROOT="$(cd "$BATS_TEST_DIRNAME/../.." && pwd)"
    cd "$REPO_ROOT"
}

@test "help flag works" {
    run bash scripts/run_baseline_bm25.sh --help
    [ "$status" -eq 0 ]
}

@test "dry-run exits 0" {
    rm -f logs/bm25.pid
    run bash scripts/run_baseline_bm25.sh --dry-run
    [ "$status" -eq 0 ]
    [[ "$output" == *"DRY RUN complete"* ]]
}

@test "missing input exits 2" {
    run env CORPUS=/nonexistent/path bash scripts/run_baseline_bm25.sh --dry-run
    [ "$status" -eq 2 ]
}

@test "unknown flag exits 5" {
    run bash scripts/run_baseline_bm25.sh --notaflag
    [ "$status" -eq 5 ]
}

@test "concurrent run guard exits 3" {
    mkdir -p logs
    echo $$ > logs/bm25.pid
    run bash scripts/run_baseline_bm25.sh --dry-run
    rm -f logs/bm25.pid
    [ "$status" -eq 3 ]
}

@test "auto-detects cores via nproc" {
    run grep -E "nproc|N_THREADS.*nproc" scripts/run_baseline_bm25.sh
    [ "$status" -eq 0 ]
}

@test "warns when WANDB_API_KEY missing and log_to_wandb expected" {
    run grep -E "WANDB_API_KEY.*warn|warn.*WANDB" scripts/run_baseline_bm25.sh
    [ "$status" -eq 0 ]
}

@test "dry-run prints hostname + UTC" {
    run bash scripts/run_baseline_bm25.sh --dry-run
    [[ "$output" == *"hostname"* ]]
    [[ "$output" == *"utc_start"* ]]
}

@test "sets PYTHONUNBUFFERED" {
    run grep -E "PYTHONUNBUFFERED" scripts/run_baseline_bm25.sh
    [ "$status" -eq 0 ]
}

@test ".env sourced before defaults (override works)" {
    run grep -B1 "source .env" scripts/run_baseline_bm25.sh
    # .env source must appear BEFORE N_THREADS/TOP_K defaults
    env_line=$(grep -n "source .env" scripts/run_baseline_bm25.sh | head -1 | cut -d: -f1)
    defaults_line=$(grep -n 'TOP_K="\${TOP_K:' scripts/run_baseline_bm25.sh | head -1 | cut -d: -f1)
    [ "$env_line" -lt "$defaults_line" ]
}

@test "uses uv run --locked for reproducibility" {
    run grep -E "uv run --locked" scripts/run_baseline_bm25.sh
    [ "$status" -eq 0 ]
}

@test "rejects non-integer TOP_K" {
    run env TOP_K=abc bash scripts/run_baseline_bm25.sh --dry-run
    [ "$status" -eq 2 ]
}

@test "rejects non-integer N_THREADS" {
    run env N_THREADS=xyz bash scripts/run_baseline_bm25.sh --dry-run
    [ "$status" -eq 2 ]
}

@test "writes launch manifest with git_sha + pid + paths" {
    run grep -E "MANIFEST.*manifest.json|run_id.*bm25_" scripts/run_baseline_bm25.sh
    [ "$status" -eq 0 ]
}
