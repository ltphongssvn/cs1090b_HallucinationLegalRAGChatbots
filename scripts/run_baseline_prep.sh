#!/usr/bin/env bash
# Full-scale MS3 baseline prep runner.
#
# Wraps scripts/baseline_prep.py with:
#   - repo-root-anchored PYTHONPATH (fixes src/ imports)
#   - nohup + background launch (survives SSH disconnect)
#   - logs/ output with timestamped filename
#   - preflight input validation (fail-fast before 2-3hr run)
#   - PID file for later monitoring / kill
#
# Usage:
#   bash scripts/run_baseline_prep.sh             # full run (~2-3hr)
#   bash scripts/run_baseline_prep.sh --dry-run   # validate inputs only
#   bash scripts/run_baseline_prep.sh --no-resume # force restart
#
# Monitor:
#   tail -f logs/baseline_prep_*.log
#   kill $(cat logs/baseline_prep.pid)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SHARD_DIR="data/raw/cl_federal_appellate_bulk"
LEPARD="lepard_train_4000000_rev0194f95.jsonl"
CL_IDS="data/processed/cl_ids.txt.gz"
COURT_MAP="data/processed/cl_matched_courts.json"
OUT_DIR="data/processed/baseline"

DRY_RUN=0
RESUME_FLAG="--resume"
for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN=1 ;;
        --no-resume) RESUME_FLAG="--no-resume" ;;
        -h|--help)
            grep '^#' "$0" | head -n 20
            exit 0
            ;;
    esac
done

# --- Preflight ---
echo "=== MS3 baseline prep runner ==="
echo "  repo_root  : $REPO_ROOT"
echo "  shard_dir  : $SHARD_DIR"
echo "  lepard     : $LEPARD"
echo "  cl_ids     : $CL_IDS"
echo "  court_map  : $COURT_MAP"
echo "  out_dir    : $OUT_DIR"
echo "  resume     : $RESUME_FLAG"
echo "  dry_run    : $DRY_RUN"
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

if [[ ! -d .venv ]]; then
    echo "FAIL: .venv not found — run 'bash setup.sh' first" >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "=== DRY RUN: validating script import + CLI parse ==="
    PYTHONPATH="$REPO_ROOT" .venv/bin/python -c "
from scripts import baseline_prep
ap = baseline_prep._build_arg_parser()
args = ap.parse_args([
    '--shard-dir', '$SHARD_DIR',
    '--lepard-path', '$LEPARD',
    '--cl-ids-path', '$CL_IDS',
    '--court-map-path', '$COURT_MAP',
    '--out-dir', '$OUT_DIR',
    '--seed', '0',
])
print(f'OK parsed args: {vars(args)}')
print(f'OK module constants: CHUNK={baseline_prep.CHUNK_SIZE_SUBWORDS} OVERLAP={baseline_prep.CHUNK_OVERLAP_SUBWORDS}')
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
PID_FILE="logs/baseline_prep.pid"

echo
echo "=== Launching full-scale run in background ==="
echo "  log  : $LOG_FILE"
echo "  pid  : $PID_FILE"

nohup env PYTHONPATH="$REPO_ROOT" .venv/bin/python scripts/baseline_prep.py \
    --shard-dir "$SHARD_DIR" \
    --lepard-path "$LEPARD" \
    --cl-ids-path "$CL_IDS" \
    --court-map-path "$COURT_MAP" \
    --out-dir "$OUT_DIR" \
    $RESUME_FLAG \
    --seed 0 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "OK launched: PID=$PID"
echo
echo "Monitor:  tail -f $LOG_FILE"
echo "Kill   :  kill \$(cat $PID_FILE)"
