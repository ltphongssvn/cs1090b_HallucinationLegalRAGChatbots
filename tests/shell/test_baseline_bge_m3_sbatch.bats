#!/usr/bin/env bats
load 'helpers.bash'

setup() {
    REPO_ROOT="$(cd "$BATS_TEST_DIRNAME/../.." && pwd)"
    cd "$REPO_ROOT"
}

@test "sbatch file exists" {
    [ -f scripts/baseline_bge_m3.sbatch ]
}

@test "requests 48 cpus" {
    run grep -E "^#SBATCH --cpus-per-task=48" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "requests GPU via gres" {
    run grep -E "^#SBATCH --gres=gpu:" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "requests 160G memory" {
    run grep -E "^#SBATCH --mem=160G" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "declares gpu partition" {
    run grep -E "^#SBATCH --partition=gpu" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "output uses %j for collision prevention" {
    run grep -E "^#SBATCH --output=.*%j" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "uses uv run --locked" {
    run grep -E "uv run --locked" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "has exit-code-aware failure trap" {
    run grep -E "_fail_handler|scancel.*SLURM_JOB_ID" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "logs git_sha for provenance" {
    run grep -E "git_sha.*rev-parse" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "validates final summary schema" {
    run grep -E "summary schema VALID|BaselineBgeM3Summary" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "paths parameterized via env vars" {
    run grep -E 'CORPUS_PATH="\$\{CORPUS_PATH:-' scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "requeue enabled" {
    run grep -E "^#SBATCH --requeue" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "sources .env for secrets" {
    run grep -E "source .env" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "HF_HOME pinned to /shared (prevents quota failures)" {
    run grep -E "HF_HOME.*shared" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "nvidia-smi dmon GPU profiler enabled" {
    run grep -E "nvidia-smi dmon" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "dry-run preflight before full run" {
    run grep -E "preflight: dry-run|--dry-run" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "idempotency: skips on valid summary" {
    run grep -E "idempotent.*VALID|model_validate_json" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "reduces OMP threads for GPU workload" {
    run grep -E "OMP_NUM_THREADS=[0-9]" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "input preflight checks all three paths" {
    run grep -E 'for p in.*CORPUS_PATH.*GOLD_PATH.*LEPARD_PATH' scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "semantic idempotency verifies top_k + seed + batch_size + hash" {
    run grep -E "expected_top_k.*expected_seed.*expected_ebs" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "validates all result lines against BaselineBgeM3ResultLine" {
    run grep -E "BaselineBgeM3ResultLine.model_validate_json" scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}

@test "checks all four artifacts exist non-empty post-run" {
    run grep -E 'for f in.*SUMMARY.*RESULTS.*INDEX.*META' scripts/baseline_bge_m3.sbatch
    [ "$status" -eq 0 ]
}
