#!/bin/bash
#SBATCH --job-name=bge_m3_baseline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/bge_m3_slurm_%j.log
#SBATCH --error=logs/bge_m3_slurm_%j.log

set -euo pipefail
_fail_handler() {
    local ec=$?
    echo "=== FAILED utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$ec"
    # Deterministic Python errors (exit 1) should NOT requeue — cancel to avoid OOM loops
    # Signal exits (128+N) or SLURM preemption can safely requeue
    if [[ $ec -ge 1 && $ec -lt 128 ]]; then
        echo "=== deterministic failure — cancelling to prevent requeue loop ==="
        scancel --signal=TERM "$SLURM_JOB_ID" 2>/dev/null || true
    fi
}
trap _fail_handler ERR
cd "$SLURM_SUBMIT_DIR"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# BGE-M3 is GPU-bound; minimize CPU thread contention
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
# Pin HF cache to /shared (11TB available) — prevents quota failures on ~ mount
export HF_HOME="${HF_HOME:-/shared/home/$USER/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

CORPUS_PATH="${CORPUS_PATH:-data/processed/baseline/corpus_chunks.jsonl}"
GOLD_PATH="${GOLD_PATH:-data/processed/baseline/gold_pairs_test.jsonl}"
LEPARD_PATH="${LEPARD_PATH:-lepard_train_4000000_rev0194f95.jsonl}"
OUT_DIR="${OUT_DIR:-data/processed/baseline}"
TOP_K="${TOP_K:-100}"
SEED="${SEED:-0}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-64}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-256}"

echo "=== SLURM BGE-M3 job ==="
echo "  job_id            : $SLURM_JOB_ID"
echo "  node              : $(hostname)"
echo "  partition         : $SLURM_JOB_PARTITION"
echo "  cpus              : $SLURM_CPUS_PER_TASK"
echo "  gpus              : ${SLURM_GPUS_ON_NODE:-unset}"
echo "  mem               : ${SLURM_MEM_PER_NODE:-unset} MB"
echo "  utc_start         : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  corpus            : $CORPUS_PATH"
echo "  top_k             : $TOP_K"
echo "  seed              : $SEED"
echo "  encode_batch_size : $ENCODE_BATCH_SIZE"
echo "  query_batch_size  : $QUERY_BATCH_SIZE"
echo "  git_sha           : $(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
echo "  uv_version        : $(uv --version 2>&1 | head -1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "  (no nvidia-smi)"

# Idempotency: skip if valid summary exists
SUMMARY="$OUT_DIR/bge_m3_summary.json"
if [[ -f "$SUMMARY" ]]; then
    if uv run --locked python -c "
from src.eda_schemas import BaselineBgeM3Summary
from pathlib import Path
BaselineBgeM3Summary.model_validate_json(Path('$SUMMARY').read_bytes())
" 2>/dev/null; then
        echo "  [idempotent] VALID summary exists at $SUMMARY — skipping"
        echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
        exit 0
    fi
fi

# Dry-run preflight
echo "=== preflight: dry-run ==="
uv run --locked python scripts/baseline_bge_m3.py \
    --corpus-path "$CORPUS_PATH" \
    --gold-pairs-path "$GOLD_PATH" \
    --lepard-path "$LEPARD_PATH" \
    --out-dir "$OUT_DIR" \
    --top-k "$TOP_K" \
    --seed "$SEED" \
    --encode-batch-size "$ENCODE_BATCH_SIZE" \
    --query-batch-size "$QUERY_BATCH_SIZE" \
    --dry-run

echo "=== full run ==="
# GPU telemetry every 10s (SM util, memory, power) — kills with main process
nvidia-smi dmon -d 10 -s pumct > "logs/gpu_dmon_${SLURM_JOB_ID}.txt" 2>&1 &
GPU_DMON_PID=$!
trap "kill $GPU_DMON_PID 2>/dev/null || true" EXIT

uv run --locked python scripts/baseline_bge_m3.py \
    --corpus-path "$CORPUS_PATH" \
    --gold-pairs-path "$GOLD_PATH" \
    --lepard-path "$LEPARD_PATH" \
    --out-dir "$OUT_DIR" \
    --top-k "$TOP_K" \
    --seed "$SEED" \
    --encode-batch-size "$ENCODE_BATCH_SIZE" \
    --query-batch-size "$QUERY_BATCH_SIZE"

# Final artifact + schema validation
if uv run --locked python -c "
from src.eda_schemas import BaselineBgeM3Summary
from pathlib import Path
BaselineBgeM3Summary.model_validate_json(Path('$OUT_DIR/bge_m3_summary.json').read_bytes())
" 2>/dev/null; then
    echo "=== summary schema VALID ==="
else
    echo "=== summary schema INVALID ==="
    exit 10
fi

echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
