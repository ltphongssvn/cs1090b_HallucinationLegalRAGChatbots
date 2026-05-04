#!/bin/bash
#SBATCH --job-name=bge_m3_multigpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --output=logs/bge_m3_multigpu_%j.log
#SBATCH --error=logs/bge_m3_multigpu_%j.log

set -euo pipefail
_fail_handler() {
    local ec=$?
    echo "=== FAILED utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$ec"
    if [[ $ec -ge 1 && $ec -lt 128 ]]; then
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

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HOME="${HF_HOME:-/shared/home/$USER/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

CORPUS_PATH="${CORPUS_PATH:-data/processed/baseline/corpus_chunks.jsonl}"
GOLD_PATH="${GOLD_PATH:-data/processed/baseline/gold_pairs_test.jsonl}"
LEPARD_PATH="${LEPARD_PATH:-lepard_train_4000000_rev0194f95.jsonl}"
OUT_DIR="${OUT_DIR:-data/processed/baseline}"
TOP_K="${TOP_K:-100}"
SEED="${SEED:-0}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-8}"
VERIFIED="${VERIFIED:-0}"
WORLD_SIZE=4

echo "=== SLURM BGE-M3 multi-GPU job (corpus-shard) ==="
echo "  job_id     : $SLURM_JOB_ID"
echo "  node       : $(hostname)"
echo "  world_size : $WORLD_SIZE"
echo "  utc_start  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  git_sha    : $(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

if [[ "$VERIFIED" == "1" ]]; then
    PREFLIGHT_PATHS=("$CORPUS_PATH" "$GOLD_PATH")
else
    PREFLIGHT_PATHS=("$CORPUS_PATH" "$GOLD_PATH" "$LEPARD_PATH")
fi
for p in "${PREFLIGHT_PATHS[@]}"; do
    if [[ ! -f "$p" ]]; then
        echo "FAIL: missing input: $p" >&2
        exit 2
    fi
done
echo "OK preflight (verified=$VERIFIED)"

# Idempotency
SUMMARY="$OUT_DIR/bge_m3_summary.json"
RESULTS="$OUT_DIR/bge_m3_results.jsonl"
if [[ -s "$SUMMARY" && -s "$RESULTS" ]]; then
    if uv run --locked python -c "
from src.eda_schemas import BaselineBgeM3Summary
from pathlib import Path
BaselineBgeM3Summary.model_validate_json(Path('$SUMMARY').read_bytes())
" 2>/dev/null; then
        echo "  [idempotent] merged summary VALID — skipping"
        echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
        exit 0
    fi
fi

# GPU telemetry
nvidia-smi dmon -d 15 -s pumct > "logs/gpu_dmon_${SLURM_JOB_ID}.txt" 2>&1 &
GPU_DMON_PID=$!
trap "kill $GPU_DMON_PID 2>/dev/null || true" EXIT

# Launch 4 parallel workers, each on its own GPU with its own corpus shard
echo "=== launching 4 corpus-shard workers ==="
pids=()
for rank in 0 1 2 3; do
    (
        export CUDA_VISIBLE_DEVICES=$rank
        VERIFIED_FLAG=""
        if [[ "$VERIFIED" == "1" ]]; then
            VERIFIED_FLAG="--verified"
        fi
        uv run --locked python scripts/baseline_bge_m3.py \
            --corpus-path "$CORPUS_PATH" \
            --gold-pairs-path "$GOLD_PATH" \
            --lepard-path "$LEPARD_PATH" \
            --out-dir "$OUT_DIR" \
            --top-k "$TOP_K" \
            --seed "$SEED" \
            --encode-batch-size "$ENCODE_BATCH_SIZE" \
            --query-batch-size "$QUERY_BATCH_SIZE" \
            --rank "$rank" \
            --world-size "$WORLD_SIZE" \
            $VERIFIED_FLAG \
            > "logs/bge_m3_rank${rank}_${SLURM_JOB_ID}.log" 2>&1
    ) &
    pids+=($!)
    echo "  rank $rank -> GPU $rank  (PID ${pids[-1]})"
done

fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "worker PID $pid failed"
        fail=1
    fi
done
if [[ $fail -ne 0 ]]; then
    echo "=== one or more workers failed ==="
    exit 12
fi
echo "=== all 4 workers completed ==="

# Merge per-rank results (delegated to scripts/merge_bge_m3_shards.py for testability)
echo "=== merging shards ==="
VERIFIED_FLAG_MERGE=""
if [[ "$VERIFIED" == "1" ]]; then
    VERIFIED_FLAG_MERGE="--verified"
fi
uv run --locked python scripts/merge_bge_m3_shards.py \
    --out-dir "$OUT_DIR" \
    --world-size "$WORLD_SIZE" \
    --top-k "$TOP_K" \
    --encode-batch-size "$ENCODE_BATCH_SIZE" \
    --seed "$SEED" \
    $VERIFIED_FLAG_MERGE


echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
