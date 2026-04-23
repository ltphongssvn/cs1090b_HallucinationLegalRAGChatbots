#!/usr/bin/env bash
# Full-scale MS3 baseline prep runner.
#
# Hardening:
#   - concurrent-run guard (kill -0 check on existing PID file)
#   - uv run (no hardcoded .venv/bin/python)
#   - env-overridable paths (${VAR:-default}) for SLURM/Docker reuse
#   - sources .env so HF_TOKEN/WANDB_API_KEY reach nohup subprocess
#   - thread caps (OMP/POLARS/RAYON=16) to avoid 48-core thrash on ODD
#   - trap EXIT cleans PID file on Ctrl-C before full launch
#   - post-launch liveness check (3s) catches instant failures
#
# Usage:
#   bash scripts/run_baseline_prep.sh              # full run (~2-3hr)
#   bash scripts/run_baseline_prep.sh --dry-run    # validate inputs only
#   bash scripts/run_baseline_prep.sh --no-resume  # force restart
#   SHARD_DIR=/path/to/shards bash scripts/run_baseline_prep.sh
#
# Monitor:
#   tail -f logs/baseline_prep_*.log
#   kill $(cat logs/baseline_prep.pid)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Env-overridable paths (SLURM/Docker friendly)
SHARD_DIR="${SHARD_DIR:-data/raw/cl_federal_appellate_bulk}"
VERIFIED_SUBSET="${VERIFIED_SUBSET:-data/processed/lepard_cl_verified_subset.jsonl}"
OUT_DIR="${OUT_DIR:-data/processed/baseline}"
SEED="${SEED:-0}"

# Thread caps — prevent Rust-backed Polars/tokenizers from spawning
# threads equal to logical cores (48 on ODD) which causes thrash
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-16}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DRY_RUN=0
RESUME_FLAG="--resume"
for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN=1 ;;
        --no-resume) RESUME_FLAG="--no-resume" ;;
        -h|--help)
            grep '^#' "$0" | head -n 25
            exit 0
            ;;
        *)
            echo "FAIL: unknown flag: $arg" >&2
            exit 5
            ;;
    esac
done

# --- Preflight ---
echo "=== MS3 baseline prep runner ==="
echo "  repo_root       : $REPO_ROOT"
echo "  shard_dir       : $SHARD_DIR"
echo "  verified_subset : $VERIFIED_SUBSET"
echo "  out_dir         : $OUT_DIR"
echo "  seed            : $SEED"
echo "  resume          : $RESUME_FLAG"
echo "  dry_run         : $DRY_RUN"
echo "  threads (OMP)   : $OMP_NUM_THREADS"
echo "  threads (POLARS): $POLARS_MAX_THREADS"
echo "  hostname        : $(hostname)"
echo "  utc_start       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

for path in "$SHARD_DIR" "$VERIFIED_SUBSET"; do
    if [[ ! -e "$path" ]]; then
        echo "FAIL: missing input: $path" >&2
        exit 2
    fi
done

SHARD_COUNT=$(find "$SHARD_DIR" -maxdepth 1 -name 'shard_*.jsonl' | wc -l)
if [[ "$SHARD_COUNT" -eq 0 ]]; then
    echo "FAIL: no shards under $SHARD_DIR" >&2
    exit 2
fi
echo "OK preflight: $SHARD_COUNT shards, subset $(du -h "$VERIFIED_SUBSET" | cut -f1)"

# Concurrent-run guard
PID_FILE="logs/baseline_prep.pid"
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "FAIL: already running (PID=$OLD_PID from $PID_FILE)" >&2
        echo "      kill $OLD_PID  # to abort prior run" >&2
        exit 3
    else
        echo "NOTE: stale PID file ($OLD_PID not running) — removing"
        rm -f "$PID_FILE"
    fi
fi

# Load .env so HF_TOKEN / WANDB_API_KEY reach the nohup subprocess
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "OK .env sourced"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "=== DRY RUN: delegating to scripts/baseline_prep.py --dry-run ==="
    PYTHONPATH="$REPO_ROOT" uv run python scripts/baseline_prep.py \
        --shard-dir "$SHARD_DIR" \
        --verified-subset-path "$VERIFIED_SUBSET" \
        --out-dir "$OUT_DIR" \
        --seed "$SEED" \
        $RESUME_FLAG \
        --dry-run
    echo
    echo "DRY RUN complete — safe to launch full run with:"
    echo "  bash scripts/run_baseline_prep.sh"
    exit 0
fi

# --- Full launch ---
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/baseline_prep_${TIMESTAMP}.log"

# Clean PID file if script exits before reaching the end of launch
trap 'if [[ -z "${LAUNCHED:-}" ]]; then rm -f "$PID_FILE"; fi' EXIT

echo
echo "=== Launching full-scale run in background ==="
echo "  log  : $LOG_FILE"
echo "  pid  : $PID_FILE"

nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_ROOT" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    POLARS_MAX_THREADS="$POLARS_MAX_THREADS" \
    RAYON_NUM_THREADS="$RAYON_NUM_THREADS" \
    TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
    HF_TOKEN="${HF_TOKEN:-}" \
    WANDB_API_KEY="${WANDB_API_KEY:-}" \
    uv run python scripts/baseline_prep.py \
    --shard-dir "$SHARD_DIR" \
    --verified-subset-path "$VERIFIED_SUBSET" \
    --out-dir "$OUT_DIR" \
    --seed "$SEED" \
    $RESUME_FLAG \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
# Symlink binds monitor to this specific run's log (not newest-mtime heuristic)
ln -sfn "$(basename "$LOG_FILE")" "logs/baseline_prep.current_log"
LAUNCHED=1
echo "OK launched: PID=$PID"

# Post-launch liveness check — catches instant failures (bad imports, etc.)
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
    echo "FAIL: process died within 3s — check $LOG_FILE" >&2
    tail -n 20 "$LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 4
fi
echo "OK liveness check: process still running after 3s"

echo
echo "Monitor:  tail -f $LOG_FILE"
echo "Kill   :  kill \$(cat $PID_FILE)"
