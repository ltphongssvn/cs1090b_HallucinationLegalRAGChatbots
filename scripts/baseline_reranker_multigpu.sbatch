#!/bin/bash
# scripts/baseline_reranker_multigpu.sbatch
#
# MS4 cross-encoder reranker — 4-way query-sharded across 4× L4 GPUs.
# Each rank handles a disjoint slice of queries; merge step concatenates
# per-rank results in rank order (no MaxP cross-shard merge needed since
# query-sharding is exclusive partitioning).
#
# Usage:
#   CORPUS_PATH=... GOLD_PATH=... INPUT_RESULTS=... OUT_DIR=... \
#       sbatch scripts/baseline_reranker_multigpu.sbatch
#
# Defaults target the verified pipeline:
#   CORPUS_PATH    = data/processed/baseline/corpus_chunks_cleaned.jsonl
#   GOLD_PATH      = data/processed/baseline/cleaned/gold_pairs_test.jsonl
#   INPUT_RESULTS  = data/processed/baseline/cleaned/rrf_results.jsonl
#   OUT_DIR        = data/processed/baseline/cleaned

#SBATCH --job-name=reranker_multigpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --output=logs/reranker_multigpu_%j.log
#SBATCH --error=logs/reranker_multigpu_%j.log

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

CORPUS_PATH="${CORPUS_PATH:-data/processed/baseline/corpus_chunks_cleaned.jsonl}"
GOLD_PATH="${GOLD_PATH:-data/processed/baseline/cleaned/gold_pairs_test.jsonl}"
INPUT_RESULTS="${INPUT_RESULTS:-data/processed/baseline/cleaned/rrf_results.jsonl}"
OUT_DIR="${OUT_DIR:-data/processed/baseline/cleaned}"
TOP_K_INPUT="${TOP_K_INPUT:-100}"
TOP_K_OUTPUT="${TOP_K_OUTPUT:-100}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_CHUNKS_PER_CLUSTER="${MAX_CHUNKS_PER_CLUSTER:-3}"
SCORE_MODE="${SCORE_MODE:-concat}"
ENCODER_DIR="${ENCODER_DIR:-}"
SEED="${SEED:-0}"
WORLD_SIZE=4

mkdir -p logs

echo "=== SLURM cross-encoder reranker (query-shard, 4× GPU) ==="
echo "  job_id        : $SLURM_JOB_ID"
echo "  node          : $(hostname)"
echo "  world_size    : $WORLD_SIZE"
echo "  utc_start     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  git_sha       : $(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
echo "  corpus_path   : $CORPUS_PATH"
echo "  gold_path     : $GOLD_PATH"
echo "  input_results : $INPUT_RESULTS"
echo "  out_dir       : $OUT_DIR"
echo "  top_k_input   : $TOP_K_INPUT"
echo "  top_k_output  : $TOP_K_OUTPUT"
echo "  max_length    : $MAX_LENGTH"
echo "  batch_size    : $BATCH_SIZE"

/usr/bin/nvidia-smi --query-gpu=index,name,memory.total --format=csv

for p in "$CORPUS_PATH" "$GOLD_PATH" "$INPUT_RESULTS"; do
    if [[ ! -f "$p" ]]; then
        echo "FAIL: missing input: $p" >&2
        exit 2
    fi
done
echo "OK preflight"

# Idempotency: skip if merged summary + results already exist and parse
SUMMARY="$OUT_DIR/reranker_summary.json"
RESULTS="$OUT_DIR/reranker_results.jsonl"
if [[ -s "$SUMMARY" && -s "$RESULTS" ]]; then
    if .venv/bin/python -c "
import json
from pathlib import Path
d = json.loads(Path('$SUMMARY').read_text())
assert 'results_hash' in d and d.get('n_queries_total', 0) > 0
" 2>/dev/null; then
        echo "  [idempotent] merged summary VALID — skipping"
        echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
        exit 0
    fi
fi

# GPU telemetry
/usr/bin/nvidia-smi dmon -d 15 -s pumct > "logs/gpu_dmon_${SLURM_JOB_ID}.txt" 2>&1 &
GPU_DMON_PID=$!
trap "kill $GPU_DMON_PID 2>/dev/null || true" EXIT

# Launch 4 parallel workers
echo "=== launching 4 query-shard workers ==="
pids=()
for rank in 0 1 2 3; do
    (
        export CUDA_VISIBLE_DEVICES=$rank
        .venv/bin/python scripts/baseline_reranker.py \
            --corpus-path "$CORPUS_PATH" \
            --gold-path "$GOLD_PATH" \
            --input-results-path "$INPUT_RESULTS" \
            --out-dir "$OUT_DIR" \
            --top-k-input "$TOP_K_INPUT" \
            --top-k-output "$TOP_K_OUTPUT" \
            --max-length "$MAX_LENGTH" \
            --batch-size "$BATCH_SIZE" \
            --max-chunks-per-cluster "$MAX_CHUNKS_PER_CLUSTER" \
            --score-mode "$SCORE_MODE" \
            ${ENCODER_DIR:+--encoder-dir "$ENCODER_DIR"} \
            --rank "$rank" \
            --world-size "$WORLD_SIZE" \
            --seed "$SEED" \
            > "logs/reranker_rank${rank}_${SLURM_JOB_ID}.log" 2>&1
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

# Merge per-rank results (delegated to scripts/merge_reranker_shards.py for testability)
echo "=== merging shards ==="
.venv/bin/python scripts/merge_reranker_shards.py \
    --out-dir "$OUT_DIR" \
    --world-size "$WORLD_SIZE" \
    --seed "$SEED"


echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
