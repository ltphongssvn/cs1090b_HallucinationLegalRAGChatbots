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
OUT_DIR="${OUT_DIR:-data/processed/baseline}"
TOP_K="${TOP_K:-100}"
SEED="${SEED:-0}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-8}"
WORLD_SIZE=4

echo "=== SLURM BGE-M3 multi-GPU job (corpus-shard) ==="
echo "  job_id     : $SLURM_JOB_ID"
echo "  node       : $(hostname)"
echo "  world_size : $WORLD_SIZE"
echo "  utc_start  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  git_sha    : $(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

for p in "$CORPUS_PATH" "$GOLD_PATH"; do
    if [[ ! -f "$p" ]]; then
        echo "FAIL: missing input: $p" >&2
        exit 2
    fi
done
echo "OK preflight"

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
        uv run --locked python scripts/baseline_bge_m3.py \
            --corpus-path "$CORPUS_PATH" \
            --gold-pairs-path "$GOLD_PATH" \
            --out-dir "$OUT_DIR" \
            --top-k "$TOP_K" \
            --seed "$SEED" \
            --encode-batch-size "$ENCODE_BATCH_SIZE" \
            --query-batch-size "$QUERY_BATCH_SIZE" \
            --rank "$rank" \
            --world-size "$WORLD_SIZE" \
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

# Merge per-rank results into unified top-k via cross-shard MaxP
echo "=== merging shards ==="
uv run --locked python -c "
import hashlib
import json
from pathlib import Path
from scripts.baseline_bge_m3 import (
    _merge_shard_results, SCHEMA_VERSION, _git_sha,
    ENCODER_MODEL, EMBEDDING_DIM, MAX_LENGTH, SIMILARITY_METRIC,
    NORMALIZE_EMBEDDINGS, DTYPE,
)
from src.eda_schemas import BaselineBgeM3Summary

out_dir = Path('$OUT_DIR')
shard_paths = sorted(out_dir.glob('bge_m3_results.rank*.jsonl'))
assert len(shard_paths) == $WORLD_SIZE, f'expected $WORLD_SIZE shards, got {len(shard_paths)}'
merged = out_dir / 'bge_m3_results.jsonl'
_merge_shard_results(shard_paths, merged, top_k=$TOP_K)

# Aggregate summary: take one rank's timings + corpus totals + recomputed hash
rank_summaries = sorted(out_dir.glob('bge_m3_summary.rank*.json'))
rank0 = json.loads(rank_summaries[0].read_text())
n_chunks_total = sum(
    json.loads(p.read_text())['n_corpus_chunks'] for p in rank_summaries
)
# Unique opinions across shards requires reading all meta files
all_opinions: set[int] = set()
for m in sorted(out_dir.glob('bge_m3_index_meta.rank*.jsonl')):
    with m.open() as f:
        for line in f:
            all_opinions.add(json.loads(line)['opinion_id'])

n_queries = sum(1 for _ in merged.open())
results_hash = hashlib.sha256(merged.read_bytes()).hexdigest()

merged_summary = {
    'schema_version': SCHEMA_VERSION,
    'n_queries': n_queries,
    'n_corpus_chunks': n_chunks_total,
    'n_unique_opinions': len(all_opinions),
    'top_k': $TOP_K,
    'encoder_model': ENCODER_MODEL,
    'embedding_dim': EMBEDDING_DIM,
    'device': rank0['device'],
    'device_name': rank0['device_name'],
    'encode_batch_size': $ENCODE_BATCH_SIZE,
    'similarity_metric': SIMILARITY_METRIC,
    'normalize_embeddings': NORMALIZE_EMBEDDINGS,
    'max_length': MAX_LENGTH,
    'dtype': DTYPE,
    'encoder_load_seconds': max(json.loads(p.read_text())['encoder_load_seconds'] for p in rank_summaries),
    'index_build_seconds': max(json.loads(p.read_text())['index_build_seconds'] for p in rank_summaries),
    'query_encode_seconds': max(json.loads(p.read_text())['query_encode_seconds'] for p in rank_summaries),
    'retrieval_seconds': max(json.loads(p.read_text())['retrieval_seconds'] for p in rank_summaries),
    'seed': $SEED,
    'world_size': $WORLD_SIZE,
    'shard_rank': 0,
    'git_sha': _git_sha(),
    'results_hash': results_hash,
}
validated = BaselineBgeM3Summary.model_validate(merged_summary)
(out_dir / 'bge_m3_summary.json').write_text(
    json.dumps(validated.model_dump(), sort_keys=True, indent=2, allow_nan=False),
    encoding='utf-8',
)
print(f'merged: n_queries={n_queries:,}  n_corpus_chunks={n_chunks_total:,}  hash={results_hash[:16]}')
"

echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
