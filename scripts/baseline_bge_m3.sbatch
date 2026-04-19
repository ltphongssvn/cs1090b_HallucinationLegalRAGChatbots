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
    # Deterministic Python errors (exit 1-127) should NOT requeue — cancel to avoid OOM loops
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

# --- Preflight: inputs exist before allocating encoder ---
for p in "$CORPUS_PATH" "$GOLD_PATH" "$LEPARD_PATH"; do
    if [[ ! -f "$p" ]]; then
        echo "FAIL: missing input: $p" >&2
        exit 2
    fi
done
echo "OK preflight: all inputs present"

# --- Semantic idempotency: verify config match + result-line schema + hash integrity ---
SUMMARY="$OUT_DIR/bge_m3_summary.json"
RESULTS="$OUT_DIR/bge_m3_results.jsonl"
INDEX="$OUT_DIR/bge_m3_index.faiss"
META="$OUT_DIR/bge_m3_index_meta.jsonl"
if [[ -s "$SUMMARY" && -s "$RESULTS" && -s "$INDEX" && -s "$META" ]]; then
    if uv run --locked python - "$SUMMARY" "$RESULTS" "$TOP_K" "$SEED" "$ENCODE_BATCH_SIZE" <<'PYEOF' 2>/dev/null
import hashlib
import json
import sys
from pathlib import Path
from src.eda_schemas import BaselineBgeM3Summary, BaselineBgeM3ResultLine

summary_path, results_path, expected_top_k, expected_seed, expected_ebs = sys.argv[1:6]
s = BaselineBgeM3Summary.model_validate_json(Path(summary_path).read_bytes())
actual_hash = hashlib.sha256(Path(results_path).read_bytes()).hexdigest()
assert s.results_hash == actual_hash, "results_hash mismatch"
assert s.top_k == int(expected_top_k), f"top_k {s.top_k} != {expected_top_k}"
assert s.seed == int(expected_seed), f"seed {s.seed} != {expected_seed}"
assert s.encode_batch_size == int(expected_ebs), (
    f"encode_batch_size {s.encode_batch_size} != {expected_ebs}"
)
# Sample first + last result lines (full validation would be O(n))
lines = Path(results_path).read_text().splitlines()
for ln in (lines[0], lines[-1]):
    BaselineBgeM3ResultLine.model_validate_json(ln)
PYEOF
    then
        echo "  [idempotent] complete VALID artifacts match config — skipping"
        echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
        exit 0
    else
        echo "  [idempotent] existing summary does not match current config — re-running"
    fi
fi

# --- Dry-run preflight ---
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

# --- Full artifact + schema validation ---
for f in "$SUMMARY" "$RESULTS" "$INDEX" "$META"; do
    if [[ ! -s "$f" ]]; then
        echo "=== artifact missing or empty: $f ==="
        exit 11
    fi
done

if uv run --locked python - "$SUMMARY" "$RESULTS" <<'PYEOF'
import hashlib
import sys
from pathlib import Path
from src.eda_schemas import BaselineBgeM3Summary, BaselineBgeM3ResultLine

summary_path, results_path = sys.argv[1:3]
s = BaselineBgeM3Summary.model_validate_json(Path(summary_path).read_bytes())
actual_hash = hashlib.sha256(Path(results_path).read_bytes()).hexdigest()
assert s.results_hash == actual_hash, f"hash mismatch {s.results_hash[:12]} vs {actual_hash[:12]}"
# Validate every result line
for i, ln in enumerate(Path(results_path).read_text().splitlines(), 1):
    BaselineBgeM3ResultLine.model_validate_json(ln)
print(f"OK: {i} result lines valid, hash matches summary")
PYEOF
then
    echo "=== summary schema + hash + result lines VALID ==="
else
    echo "=== summary schema INVALID ==="
    exit 10
fi

echo "=== done utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
