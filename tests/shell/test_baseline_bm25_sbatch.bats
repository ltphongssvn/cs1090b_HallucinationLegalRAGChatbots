#!/usr/bin/env bats
load 'helpers.bash'

setup() {
    REPO_ROOT="$(cd "$BATS_TEST_DIRNAME/../.." && pwd)"
    cd "$REPO_ROOT"
}

@test "sbatch file exists" {
    [ -f scripts/baseline_bm25.sbatch ]
}

@test "requests 48 cpus" {
    run grep -E "^#SBATCH --cpus-per-task=48" scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
}

@test "requests 128G memory" {
    run grep -E "^#SBATCH --mem=160G" scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
}

@test "declares partition" {
    run grep -E "^#SBATCH --partition=" scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
}

@test "output uses %j for collision prevention" {
    run grep -E "^#SBATCH --output=.*%j" scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
}

@test "uses uv run --locked" {
    run grep -E "uv run --locked" scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
}

@test "has failure trap" {
    run grep -E "trap.*ERR|FAILED utc_end" scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
}

@test "logs git_sha for provenance" {
    run grep -E "git_sha.*rev-parse" scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
}

@test "validates final summary schema" {
    run grep -E "summary schema VALID|model_validate_json" scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
}

@test "paths parameterized via env vars" {
    run grep -E 'CORPUS_PATH="\$\{CORPUS_PATH:-' scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
}

@test "requeue enabled" {
    run grep -E "^#SBATCH --requeue" scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
}

@test "sources .env for secrets" {
    run grep -E "source .env" scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
}

@test "splits polars/rayon threads (16 vs 48)" {
    run grep -E "POLARS_MAX_THREADS=16" scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
    run grep -E "RAYON_NUM_THREADS=48" scripts/baseline_bm25.sbatch
    [ "$status" -eq 0 ]
}
