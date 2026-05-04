#!/bin/bash
# scripts/finetune_reranker.sbatch — 4× L4 DDP fine-tuning of bge-reranker-v2-m3

#SBATCH --job-name=finetune_reranker
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --output=logs/finetune_reranker_%j.log
#SBATCH --error=logs/finetune_reranker_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [[ -f .env ]]; then
    set -a
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

TRAIN_PATH="${TRAIN_PATH:-data/processed/finetune/hard_negatives.jsonl}"
VAL_PATH="${VAL_PATH:-data/processed/finetune/hard_negatives.val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/finetune/bge_reranker_legal}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
SEED="${SEED:-0}"

mkdir -p logs "$OUTPUT_DIR"

echo "=== finetune_reranker  job=$SLURM_JOB_ID  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "  train_path : $TRAIN_PATH"
echo "  val_path   : $VAL_PATH"
echo "  output_dir : $OUTPUT_DIR"
echo "  lr=$LR epochs=$EPOCHS batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM max_length=$MAX_LENGTH"

/usr/bin/nvidia-smi --query-gpu=index,name,memory.total --format=csv

for p in "$TRAIN_PATH" "$VAL_PATH"; do
    if [[ ! -f "$p" ]]; then echo "FAIL: missing $p" >&2; exit 2; fi
done

# Idempotency
if [[ -s "$OUTPUT_DIR/training_summary.json" ]]; then
    echo "  [idempotent] training_summary.json present — skipping"
    exit 0
fi

# torchrun for 4-way DDP
.venv/bin/torchrun \
    --standalone --nproc_per_node=4 \
    scripts/finetune_reranker.py \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --max-length "$MAX_LENGTH" \
    --seed "$SEED"

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
