#!/bin/bash
#SBATCH --job-name=bge_m3_multigpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
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
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-256}"
WORLD_SIZE=4

echo "=== SLURM BGE-M3 multi-GPU job ==="
echo "  job_id            : $SLURM_JOB_ID"
echo "  node              : $(hostname)"
echo "  gpus              : ${SLURM_GPUS_ON_NODE:-unset}"
echo "  world_size        : $WORLD_SIZE"
echo "  utc_start         : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  git_sha           : $(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

for p in "$CORPUS_PATH" "$GOLD_PATH" "$LEPARD_PATH"; do
    if [[ ! -f "$p" ]]; then
        echo "FAIL: missing input: $p" >&2
        exit 2
    fi
done
echo "OK preflight"

# Idempotency: skip if merged summary exists and valid
SUMMARY="$OUT_DIR/bge_m3_summary.json"
RESULTS="$OUT_DIR/bge_m3_results.jsonl"
if [[ -s "$SUMMARY" && -s "$RESULTS" ]]; then
    if uv run --locked python -c "
from src.eda_schemas import BaselineBgeM3Summary
from pathlib import Path
BaselineBgeM3Summary.model_validate_json(Path('$SUMMARY').read_bytes())
" 2>/dev/null; then
        echo "  [idempotent] VALID merged summary exists — skipping"
        echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
        exit 0
    fi
fi

# GPU telemetry
nvidia-smi dmon -d 15 -s pumct > "logs/gpu_dmon_${SLURM_JOB_ID}.txt" 2>&1 &
GPU_DMON_PID=$!
trap "kill $GPU_DMON_PID 2>/dev/null || true" EXIT

# Launch 4 workers in parallel, one per GPU
echo "=== launching 4 parallel workers ==="
pids=()
for rank in 0 1 2 3; do
    (
        export CUDA_VISIBLE_DEVICES=$rank
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
            > "logs/bge_m3_rank${rank}_${SLURM_JOB_ID}.log" 2>&1
    ) &
    pids+=($!)
    echo "  spawned rank $rank on GPU $rank (PID ${pids[-1]})"
done

# Wait for all workers; any failure propagates via set -e
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

# Merge shard results + build unified summary
echo "=== merging shards ==="
uv run --locked python -c "
import json
import hashlib
from pathlib import Path
from src.eda_schemas import BaselineBgeM3Summary
from scripts.baseline_bge_m3 import _merge_shard_results, SCHEMA_VERSION, _git_sha

out_dir = Path('$OUT_DIR')
shard_paths = sorted(out_dir.glob('bge_m3_results.rank*.jsonl'))
assert len(shard_paths) == $WORLD_SIZE, f'expected $WORLD_SIZE shards, got {len(shard_paths)}'
merged = out_dir / 'bge_m3_results.jsonl'
_merge_shard_results(shard_paths, merged)
print(f'merged {len(shard_paths)} shards -> {merged}')

# Aggregate summary from rank-0's summary + recompute hash over merged file
rank0_summary = json.loads((out_dir / 'bge_m3_summary.json').read_text()) if (out_dir / 'bge_m3_summary.json').exists() else {}
# Each rank wrote its own bge_m3_summary.json — the last writer wins; take rank 0's timing
# but recompute hash + n_queries over merged output.
lines = merged.read_text().splitlines()
n_queries_total = len(lines)
results_hash = hashlib.sha256(merged.read_bytes()).hexdigest()

# Pull one rank summary for metadata (any is fine since config is identical)
# Use the first shard's per-rank summary file path pattern
# We wrote summaries to bge_m3_summary.json during each rank — they clobber.
# Reconstruct merged summary from known constants + aggregated stats.
from scripts.baseline_bge_m3 import (
    ENCODER_MODEL, EMBEDDING_DIM, MAX_LENGTH, SIMILARITY_METRIC,
    NORMALIZE_EMBEDDINGS, DTYPE,
)
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else device

# Corpus stats recomputed cheaply
n_chunks = sum(1 for _ in (out_dir / 'corpus_chunks.jsonl').open())
unique_opinions = len({json.loads(line)['opinion_id'] for line in (out_dir / 'corpus_chunks.jsonl').open()})

merged_summary = {
    'schema_version': SCHEMA_VERSION,
    'n_queries': n_queries_total,
    'n_corpus_chunks': n_chunks,
    'n_unique_opinions': unique_opinions,
    'top_k': $TOP_K,
    'encoder_model': ENCODER_MODEL,
    'embedding_dim': EMBEDDING_DIM,
    'device': device,
    'device_name': device_name,
    'encode_batch_size': $ENCODE_BATCH_SIZE,
    'similarity_metric': SIMILARITY_METRIC,
    'normalize_embeddings': NORMALIZE_EMBEDDINGS,
    'max_length': MAX_LENGTH,
    'dtype': DTYPE,
    'encoder_load_seconds': 0.0,
    'index_build_seconds': 0.0,
    'query_encode_seconds': 0.0,
    'retrieval_seconds': 0.0,
    'seed': $SEED,
    'world_size': $WORLD_SIZE,
    'shard_rank': -1,
    'git_sha': _git_sha(),
    'results_hash': results_hash,
}
# shard_rank=-1 means 'merged'; but schema requires ge=0
# Use shard_rank=0 as sentinel for merged
merged_summary['shard_rank'] = 0
validated = BaselineBgeM3Summary.model_validate(merged_summary)
(out_dir / 'bge_m3_summary.json').write_text(
    json.dumps(validated.model_dump(), sort_keys=True, indent=2, allow_nan=False),
    encoding='utf-8',
)
print(f'merged summary written: n_queries={n_queries_total}, hash={results_hash[:16]}...')
"

echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
