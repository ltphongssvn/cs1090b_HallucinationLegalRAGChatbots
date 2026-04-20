#!/usr/bin/env bash
# Full-scale MS3 BM25 baseline runner.
#
# Uses 48 physical cores on AMD EPYC 7R13 (Harvard ODD).
# bm25s (NumPy/SciPy, parallelized via n_threads) + Polars thread pools
#
# Usage:
#   bash scripts/run_baseline_bm25.sh              # full run (~15-30 min)
#   bash scripts/run_baseline_bm25.sh --dry-run    # validate inputs only
#   N_THREADS=16 bash scripts/run_baseline_bm25.sh # override thread count
#
# Monitor:
#   tail -f logs/bm25_run_*.log
#   kill $(cat logs/bm25.pid)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Source .env BEFORE resolving defaults so .env overrides work as expected
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

mkdir -p logs

CORPUS="${CORPUS:-data/processed/baseline/corpus_chunks.jsonl}"
GOLD="${GOLD:-data/processed/baseline/gold_pairs_test.jsonl}"
LEPARD="${LEPARD:-lepard_train_4000000_rev0194f95.jsonl}"
OUT_DIR="${OUT_DIR:-data/processed/baseline}"
TOP_K="${TOP_K:-100}"
SEED="${SEED:-0}"
N_THREADS="${N_THREADS:-$(nproc --all 2>/dev/null || echo 48)}"

# Integer validation
for var in TOP_K SEED N_THREADS; do
    val="${!var}"
    case "$val" in
        ''|*[!0-9]*)
            echo "FAIL: $var must be non-negative integer (got: '$val')" >&2
            exit 2
            ;;
    esac
done

# Thread allocation avoids oversubscription on 48-core AMD EPYC:
#   RAYON=48 for bm25s (Rust) — primary compute
#   POLARS=48 for DataFrame ops (sequential with bm25s, never concurrent)
#   OMP/MKL=16 for incidental NumPy ops (no hot numpy path here)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-$N_THREADS}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$N_THREADS}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            grep '^#' "$0" | head -n 15
            exit 0
            ;;
        *)
            echo "FAIL: unknown flag: $arg" >&2
            exit 5
            ;;
    esac
done

echo "=== MS3 BM25 baseline runner ==="
echo "  repo_root   : $REPO_ROOT"
echo "  corpus      : $CORPUS"
echo "  gold        : $GOLD"
echo "  lepard      : $LEPARD"
echo "  out_dir     : $OUT_DIR"
echo "  top_k       : $TOP_K"
echo "  seed        : $SEED"
echo "  n_threads   : $N_THREADS"
echo "  dry_run     : $DRY_RUN"
echo "  hostname    : $(hostname)"
echo "  utc_start   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

for path in "$CORPUS" "$GOLD" "$LEPARD"; do
    if [[ ! -f "$path" ]]; then
        echo "FAIL: missing input: $path" >&2
        exit 2
    fi
done
echo "OK preflight"

PID_FILE="logs/bm25.pid"
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "FAIL: already running (PID=$OLD_PID)" >&2
        exit 3
    else
        rm -f "$PID_FILE"
    fi
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "  [warn] WANDB_API_KEY not set — W&B logging will be skipped if --log-to-wandb passed"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "=== DRY RUN ==="
    PYTHONPATH="$REPO_ROOT" uv run --locked python scripts/baseline_bm25.py \
        --corpus-path "$CORPUS" --gold-pairs-path "$GOLD" \
        --lepard-path "$LEPARD" --out-dir "$OUT_DIR" \
        --top-k "$TOP_K" --seed "$SEED" --dry-run
    echo "DRY RUN complete"
    exit 0
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/bm25_run_${TIMESTAMP}.log"

echo
echo "=== Launching full-scale run in background ==="
echo "  log  : $LOG_FILE"
echo "  pid  : $PID_FILE"

# setsid detaches from controlling terminal (survives SSH disconnect).
# nohup ignores SIGHUP. The <>/dev/null redirect fully decouples stdin/stdout.
setsid nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_ROOT" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    POLARS_MAX_THREADS="$POLARS_MAX_THREADS" \
    RAYON_NUM_THREADS="$RAYON_NUM_THREADS" \
    TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
    HF_TOKEN="${HF_TOKEN:-}" \
    WANDB_API_KEY="${WANDB_API_KEY:-}" \
    uv run python scripts/baseline_bm25.py \
    --corpus-path "$CORPUS" \
    --gold-pairs-path "$GOLD" \
    --lepard-path "$LEPARD" \
    --out-dir "$OUT_DIR" \
    --top-k "$TOP_K" \
    --seed "$SEED" \
    > "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PID_FILE"
ln -sfn "$(basename "$LOG_FILE")" "logs/bm25.current_log"

# Launch manifest sidecar (provenance for reproducibility)
GIT_SHA=$(git rev-parse HEAD 2>/dev/null | head -c 12 || echo unknown)
MANIFEST="logs/bm25_run_${TIMESTAMP}.manifest.json"
cat > "$MANIFEST" << MANIFEST_EOF
{
  "run_id": "bm25_${TIMESTAMP}_${GIT_SHA}",
  "pid": $PID,
  "log_file": "$LOG_FILE",
  "corpus_path": "$CORPUS",
  "gold_path": "$GOLD",
  "lepard_path": "$LEPARD",
  "out_dir": "$OUT_DIR",
  "top_k": $TOP_K,
  "seed": $SEED,
  "n_threads": $N_THREADS,
  "git_sha": "$GIT_SHA",
  "hostname": "$(hostname)",
  "utc_start": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST_EOF
echo "OK manifest: $MANIFEST"
echo "OK launched: PID=$PID"

sleep 10
if ! kill -0 "$PID" 2>/dev/null; then
    echo "FAIL: process died within 3s — check $LOG_FILE" >&2
    tail -n 20 "$LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 4
fi
echo "OK liveness check: still running after 10s"

echo
echo "Monitor:  tail -f $LOG_FILE"
echo "Kill   :  kill \$(cat $PID_FILE)"
