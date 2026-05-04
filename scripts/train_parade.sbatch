#!/bin/bash
# scripts/train_parade.sbatch — train PARADE aggregator on fine-tuned reranker

#SBATCH --job-name=train_parade
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --output=logs/train_parade_%j.log
#SBATCH --error=logs/train_parade_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f .env ]] && { set -a; source .env; set +a; }

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HOME="${HF_HOME:-/shared/home/$USER/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

TRAIN_PATH="${TRAIN_PATH:-data/processed/finetune/hard_negatives.jsonl}"
VAL_PATH="${VAL_PATH:-data/processed/finetune/hard_negatives.val.jsonl}"
ENCODER_DIR="${ENCODER_DIR:-data/processed/finetune/bge_reranker_legal}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/finetune/parade}"
MAX_PASSAGES="${MAX_PASSAGES:-8}"
LR="${LR:-5e-5}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SEED="${SEED:-0}"

mkdir -p logs "$OUTPUT_DIR"

echo "=== train_parade  job=$SLURM_JOB_ID  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "  encoder_dir : $ENCODER_DIR"
echo "  output_dir  : $OUTPUT_DIR"
echo "  max_passages=$MAX_PASSAGES lr=$LR epochs=$EPOCHS batch=$BATCH_SIZE"

/usr/bin/nvidia-smi --query-gpu=index,name,memory.total --format=csv

for p in "$TRAIN_PATH" "$VAL_PATH" "$ENCODER_DIR/training_summary.json"; do
    [[ -e "$p" ]] || { echo "FAIL: missing $p" >&2; exit 2; }
done

if [[ -s "$OUTPUT_DIR/parade_summary.json" ]]; then
    echo "  [idempotent] parade_summary.json present — skipping"
    exit 0
fi

.venv/bin/torchrun --standalone --nproc_per_node=4 \
    scripts/parade_aggregator.py \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --encoder-dir "$ENCODER_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-passages "$MAX_PASSAGES" \
    --lr "$LR" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
    --max-length "$MAX_LENGTH" --seed "$SEED"

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
