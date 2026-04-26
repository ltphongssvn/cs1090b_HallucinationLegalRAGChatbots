#!/bin/bash
#SBATCH --job-name=bm25_baseline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/bm25_slurm_%j.log
#SBATCH --error=logs/bm25_slurm_%j.log

set -euo pipefail
trap 'echo "=== FAILED utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$?"' ERR
cd "$SLURM_SUBMIT_DIR"

# Source .env for HF_TOKEN / WANDB_API_KEY (SLURM strips login env)
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Thread allocation: POLARS=16 (I/O only), RAYON=48 (BM25 compute, sequential phase)
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export POLARS_MAX_THREADS=16
export RAYON_NUM_THREADS=48
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"

CORPUS_PATH="${CORPUS_PATH:-data/processed/baseline/corpus_chunks.jsonl}"
GOLD_PATH="${GOLD_PATH:-data/processed/baseline/gold_pairs_test.jsonl}"
OUT_DIR="${OUT_DIR:-data/processed/baseline}"
TOP_K="${TOP_K:-100}"
SEED="${SEED:-0}"

# Check for redaction marker
REDACTED_FLAG=""
if [[ -f "data/processed/baseline/.redacted" ]]; then
    REDACTED_FLAG="--case-names-redacted"
fi

echo "=== SLURM BM25 job ==="
echo "  job_id        : $SLURM_JOB_ID"
echo "  node          : $(hostname)"
echo "  partition     : $SLURM_JOB_PARTITION"
echo "  cpus          : $SLURM_CPUS_PER_TASK"
echo "  mem           : ${SLURM_MEM_PER_NODE:-unset} MB"
echo "  utc_start     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  polars_threads: $POLARS_MAX_THREADS"
echo "  rayon_threads : $RAYON_NUM_THREADS"
echo "  corpus        : $CORPUS_PATH"
echo "  top_k         : $TOP_K"
echo "  seed          : $SEED"
echo "  git_sha       : $(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
echo "  uv_version    : $(uv --version 2>&1 | head -1)"

# Idempotency: skip if valid summary already exists
SUMMARY="$OUT_DIR/bm25_summary.json"
if [[ -f "$SUMMARY" ]]; then
    if uv run --locked python -c "
from src.eda_schemas import BaselineBM25Summary
from pathlib import Path
BaselineBM25Summary.model_validate_json(Path('$SUMMARY').read_bytes())
" 2>/dev/null; then
        echo "  [idempotent] VALID summary exists at $SUMMARY — skipping"
        echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
        exit 0
    fi
fi

# Dry-run preflight (catches import / CLI errors before 30+ min compute)
echo "=== preflight: dry-run ==="
uv run --locked python scripts/baseline_bm25.py \
    --corpus-path "$CORPUS_PATH" \
    --gold-pairs-path "$GOLD_PATH" \
    --out-dir "$OUT_DIR" \
    --top-k "$TOP_K" \
    --seed "$SEED" \
    $REDACTED_FLAG \
    --dry-run

echo "=== full run ==="
uv run --locked python scripts/baseline_bm25.py \
    --corpus-path "$CORPUS_PATH" \
    --gold-pairs-path "$GOLD_PATH" \
    --out-dir "$OUT_DIR" \
    --top-k "$TOP_K" \
    --seed "$SEED" \
    $REDACTED_FLAG

# Final artifact + schema validation
if uv run --locked python -c "
from src.eda_schemas import BaselineBM25Summary
from pathlib import Path
BaselineBM25Summary.model_validate_json(Path('$OUT_DIR/bm25_summary.json').read_bytes())
" 2>/dev/null; then
    echo "=== summary schema VALID ==="
else
    echo "=== summary schema INVALID ==="
    exit 10
fi

echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
