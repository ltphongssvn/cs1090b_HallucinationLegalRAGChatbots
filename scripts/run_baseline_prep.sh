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
LEPARD="${LEPARD:-lepard_train_4000000_rev0194f95.jsonl}"
CL_IDS="${CL_IDS:-data/processed/cl_ids.txt.gz}"
COURT_MAP="${COURT_MAP:-data/processed/cl_matched_courts.json}"
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
    esac
done

# --- Preflight ---
echo "=== MS3 baseline prep runner ==="
echo "  repo_root       : $REPO_ROOT"
echo "  shard_dir       : $SHARD_DIR"
echo "  lepard          : $LEPARD"
echo "  cl_ids          : $CL_IDS"
echo "  court_map       : $COURT_MAP"
echo "  out_dir         : $OUT_DIR"
echo "  seed            : $SEED"
echo "  resume          : $RESUME_FLAG"
echo "  dry_run         : $DRY_RUN"
echo "  threads (OMP)   : $OMP_NUM_THREADS"
echo "  threads (POLARS): $POLARS_MAX_THREADS"
echo

for path in "$SHARD_DIR" "$LEPARD" "$CL_IDS" "$COURT_MAP"; do
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
echo "OK preflight: $SHARD_COUNT shards, LePaRD $(du -h "$LEPARD" | cut -f1)"

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
    echo "=== DRY RUN: validating script import + CLI parse ==="
    PYTHONPATH="$REPO_ROOT" uv run python -c "
from scripts import baseline_prep
ap = baseline_prep._build_arg_parser()
args = ap.parse_args([
    '--shard-dir', '$SHARD_DIR',
    '--lepard-path', '$LEPARD',
    '--cl-ids-path', '$CL_IDS',
    '--court-map-path', '$COURT_MAP',
    '--out-dir', '$OUT_DIR',
    '--seed', '$SEED',
])
print(f'OK parsed args: {vars(args)}')
print(f'OK constants: CHUNK={baseline_prep.CHUNK_SIZE_SUBWORDS} OVERLAP={baseline_prep.CHUNK_OVERLAP_SUBWORDS}')
print(f'OK VAL_SIZE={baseline_prep.VAL_SIZE} TEST_SIZE={baseline_prep.TEST_SIZE}')
"
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
    --lepard-path "$LEPARD" \
    --cl-ids-path "$CL_IDS" \
    --court-map-path "$COURT_MAP" \
    --out-dir "$OUT_DIR" \
    $RESUME_FLAG \
    --seed "$SEED" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
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
