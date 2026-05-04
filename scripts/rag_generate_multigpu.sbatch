#!/bin/bash
# scripts/rag_generate_multigpu.sbatch
#
# MS5 RAG generation — 4-way query-sharded across 4× L4 GPUs.
# Each rank runs scripts/rag_generate.py on a disjoint slice of queries
# under one ABLATION config. Merge step concatenates per-rank outputs in
# rank order via scripts/merge_rag_generations.py.
#
# Usage (one ablation per submission):
#   ABLATION=reranker CORPUS_PATH=... GOLD_PATH=... \
#       sbatch scripts/rag_generate_multigpu.sbatch
#
# Defaults (verified pipeline):
#   ABLATION       = reranker  (override per-call: none|bm25|bge_m3|reranker)
#   CORPUS_PATH    = data/processed/baseline/corpus_chunks_cleaned.jsonl
#   GOLD_PATH      = data/processed/baseline/cleaned/gold_pairs_test.jsonl
#   RETRIEVAL_DIR  = data/processed/baseline/cleaned
#   OUT_ROOT       = data/processed/rag

#SBATCH --job-name=rag_generate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --output=logs/rag_generate_%j.log
#SBATCH --error=logs/rag_generate_%j.log

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

ABLATION="${ABLATION:-reranker}"
CORPUS_PATH="${CORPUS_PATH:-data/processed/baseline/corpus_chunks_cleaned.jsonl}"
GOLD_PATH="${GOLD_PATH:-data/processed/baseline/cleaned/gold_pairs_test.jsonl}"
RETRIEVAL_DIR="${RETRIEVAL_DIR:-data/processed/baseline/cleaned}"
OUT_ROOT="${OUT_ROOT:-data/processed/rag}"
TOP_K_CONTEXT="${TOP_K_CONTEXT:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
MAX_CHUNKS_PER_CLUSTER="${MAX_CHUNKS_PER_CLUSTER:-2}"
SEED="${SEED:-0}"
WORLD_SIZE=4

# Map ablation -> output label (must match scripts/rag_generate.py::ABLATION_CONFIGS)
case "$ABLATION" in
    none)     ABLATION_LABEL="no_rag" ;;
    bm25)     ABLATION_LABEL="bm25_rag" ;;
    bge_m3)   ABLATION_LABEL="bge_m3_rag" ;;
    rrf)      ABLATION_LABEL="rrf_rag" ;;
    reranker) ABLATION_LABEL="reranker_rag" ;;
    *) echo "FAIL: unknown ABLATION=$ABLATION (valid: none|bm25|bge_m3|rrf|reranker)" >&2; exit 2 ;;
esac
ABLATION_OUT_DIR="$OUT_ROOT/$ABLATION_LABEL"

mkdir -p logs "$ABLATION_OUT_DIR"

echo "=== SLURM RAG generation (query-shard, 4× GPU) ==="
echo "  job_id           : $SLURM_JOB_ID"
echo "  node             : $(hostname)"
echo "  world_size       : $WORLD_SIZE"
echo "  utc_start        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  git_sha          : $(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
echo "  ablation         : $ABLATION (label=$ABLATION_LABEL)"
echo "  corpus_path      : $CORPUS_PATH"
echo "  gold_path        : $GOLD_PATH"
echo "  retrieval_dir    : $RETRIEVAL_DIR"
echo "  out_root         : $OUT_ROOT"
echo "  ablation_out_dir : $ABLATION_OUT_DIR"
echo "  top_k_context    : $TOP_K_CONTEXT"
echo "  max_new_tokens   : $MAX_NEW_TOKENS"
echo "  batch_size       : $BATCH_SIZE"

/usr/bin/nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Preflight
PREFLIGHT_PATHS=("$CORPUS_PATH" "$GOLD_PATH")
if [[ "$ABLATION" != "none" ]]; then
    case "$ABLATION" in
        bm25)     RETRIEVAL_FILE="$RETRIEVAL_DIR/bm25_results.jsonl" ;;
        bge_m3)   RETRIEVAL_FILE="$RETRIEVAL_DIR/bge_m3_results.jsonl" ;;
        rrf)      RETRIEVAL_FILE="$RETRIEVAL_DIR/rrf_results.jsonl" ;;
        reranker) RETRIEVAL_FILE="$RETRIEVAL_DIR/reranker_results.jsonl" ;;
    esac
    PREFLIGHT_PATHS+=("$RETRIEVAL_FILE")
fi
for p in "${PREFLIGHT_PATHS[@]}"; do
    if [[ ! -f "$p" ]]; then
        echo "FAIL: missing input: $p" >&2
        exit 2
    fi
done
echo "OK preflight (ablation=$ABLATION)"

# Idempotency: skip if merged generations + summary already exist and parse
SUMMARY="$ABLATION_OUT_DIR/generation_summary.json"
RESULTS="$ABLATION_OUT_DIR/generations.jsonl"
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
        .venv/bin/python scripts/rag_generate.py \
            --ablation "$ABLATION" \
            --corpus-path "$CORPUS_PATH" \
            --gold-path "$GOLD_PATH" \
            --retrieval-dir "$RETRIEVAL_DIR" \
            --out-root "$OUT_ROOT" \
            --top-k-context "$TOP_K_CONTEXT" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --batch-size "$BATCH_SIZE" \
            --max-length "$MAX_LENGTH" \
            --max-chunks-per-cluster "$MAX_CHUNKS_PER_CLUSTER" \
            --rank "$rank" \
            --world-size "$WORLD_SIZE" \
            --seed "$SEED" \
            > "logs/rag_${ABLATION}_rank${rank}_${SLURM_JOB_ID}.log" 2>&1
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

# Merge per-rank generations (delegated to scripts/merge_rag_generations.py)
echo "=== merging shards ==="
.venv/bin/python scripts/merge_rag_generations.py \
    --ablation "$ABLATION" \
    --out-root "$OUT_ROOT" \
    --world-size "$WORLD_SIZE" \
    --seed "$SEED"

echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
