#!/usr/bin/env bash
# Monitor MS3 baseline_prep run — auto-discovers PID + latest log + semantic progress + ETA.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PID_FILE="logs/baseline_prep.pid"
OUT_DIR="data/processed/baseline"
SHARD_DIR="${SHARD_DIR:-data/raw/cl_federal_appellate_bulk}"
WC_SIZE_GUARD_GB="${WC_SIZE_GUARD_GB:-5}"

JSON_MODE=0
STRICT_MODE=0
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Monitor MS3 baseline_prep run."
            echo "Usage: bash scripts/monitor_baseline.sh [--json] [--strict]"
            exit 0
            ;;
        --json)   JSON_MODE=1 ;;
        --strict) STRICT_MODE=1 ;;
    esac
done

# --- Gather state ---
LOG_FILE=$(ls -t logs/baseline_prep_*.log 2>/dev/null | head -1)
PROC_ETIME_SEC=0
PROC_STATUS="not_running"
PROC_PID=""
if [[ -f "$PID_FILE" ]]; then
    PROC_PID=$(cat "$PID_FILE")
    if kill -0 "$PROC_PID" 2>/dev/null; then
        PS_ARGS=$(ps -p "$PROC_PID" -o args= 2>/dev/null)
        if [[ "$PS_ARGS" == *"baseline_prep"* ]]; then
            PROC_STATUS="running"
        else
            PROC_STATUS="pid_reuse_warning"
        fi
        ETIME_RAW=$(ps -p "$PROC_PID" -o etimes= 2>/dev/null | tr -d ' ')
        PROC_ETIME_SEC="${ETIME_RAW:-0}"
    else
        PROC_STATUS="stale_pid"
    fi
fi

CKPT="$OUT_DIR/chunking_checkpoint.json"
SUMMARY="$OUT_DIR/summary.json"
VAL="$OUT_DIR/gold_pairs_val.jsonl"
TEST="$OUT_DIR/gold_pairs_test.jsonl"
CORPUS="$OUT_DIR/corpus_chunks.jsonl"

TOTAL_SHARDS=$(find "$SHARD_DIR" -maxdepth 1 -name 'shard_*.jsonl' 2>/dev/null | wc -l)
SHARDS_DONE=0
if [[ -f "$CKPT" ]]; then
    SHARDS_DONE=$(PYTHONPATH="$REPO_ROOT" uv run python -c "
import json
try:
    print(len(json.load(open('$CKPT'))['completed']))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
fi

VAL_LINES=0
TEST_LINES=0
CORPUS_SIZE=0
[[ -f "$VAL" ]] && VAL_LINES=$(wc -l < "$VAL")
[[ -f "$TEST" ]] && TEST_LINES=$(wc -l < "$TEST")
[[ -f "$CORPUS" ]] && CORPUS_SIZE=$(stat -c '%s' "$CORPUS" 2>/dev/null || echo 0)

SUMMARY_PRESENT=0
SUMMARY_VALID="n/a"
if [[ -f "$SUMMARY" ]]; then
    SUMMARY_PRESENT=1
    SUMMARY_VALID=$(PYTHONPATH="$REPO_ROOT" uv run python -c "
from src.eda_schemas import BaselinePrepSummary
from pathlib import Path
try:
    BaselinePrepSummary.model_validate_json(Path('$SUMMARY').read_bytes())
    print('VALID')
except Exception as e:
    print(f'INVALID: {e}')
" 2>/dev/null || echo "check-failed")
fi

# ETA
VELOCITY_SPM=0
ETA_MIN=0
if [[ "$SHARDS_DONE" -gt 0 ]] && [[ "$PROC_ETIME_SEC" -gt 0 ]] && [[ "$TOTAL_SHARDS" -gt "$SHARDS_DONE" ]]; then
    VELOCITY_SPM=$(( SHARDS_DONE * 60 / PROC_ETIME_SEC ))
    if [[ "$VELOCITY_SPM" -gt 0 ]]; then
        ETA_MIN=$(( (TOTAL_SHARDS - SHARDS_DONE) / VELOCITY_SPM ))
    fi
fi

# --- JSON mode ---
if [[ "$JSON_MODE" -eq 1 ]]; then
    PYTHONPATH="$REPO_ROOT" uv run python -c "
import json
print(json.dumps({
    'timestamp_utc': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
    'log_file': '${LOG_FILE:-}',
    'proc_pid': '${PROC_PID}',
    'proc_status': '$PROC_STATUS',
    'proc_etime_sec': $PROC_ETIME_SEC,
    'shards_done': $SHARDS_DONE,
    'total_shards': $TOTAL_SHARDS,
    'velocity_shards_per_min': $VELOCITY_SPM,
    'eta_min': $ETA_MIN,
    'val_lines': $VAL_LINES,
    'test_lines': $TEST_LINES,
    'corpus_size_bytes': $CORPUS_SIZE,
    'summary_present': bool($SUMMARY_PRESENT),
    'summary_valid': '$SUMMARY_VALID',
}, indent=2))
"
    # Strict exit logic
    if [[ "$STRICT_MODE" -eq 1 ]]; then
        if [[ "$SUMMARY_PRESENT" -eq 1 ]] && [[ "$SUMMARY_VALID" != "VALID" ]]; then
            exit 10
        fi
        if [[ "$PROC_STATUS" == "stale_pid" ]] && [[ "$SUMMARY_PRESENT" -eq 0 ]]; then
            exit 11
        fi
    fi
    exit 0
fi

# --- Human mode ---
echo "=== baseline_prep monitor  ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="

if [[ -z "${LOG_FILE:-}" ]]; then
    echo "  [log] no log files under logs/baseline_prep_*.log"
else
    echo "  [log] $LOG_FILE"
    echo "  --- last 20 lines ---"
    tail -n 20 "$LOG_FILE" 2>/dev/null || echo "  [warn] log read failed"
fi

echo "---"

case "$PROC_STATUS" in
    running)
        ps -p "$PROC_PID" -o pid,etime,%mem,rss,stat,args 2>/dev/null | tail -n 1 | sed 's/^/  [proc] /'
        ;;
    pid_reuse_warning)
        echo "  [warn] PID $PROC_PID does not look like baseline_prep — possible PID reuse"
        ;;
    stale_pid)
        echo "  [proc] PID $PROC_PID not running (stale PID file or job finished)"
        ;;
    *)
        echo "  [proc] no PID file at $PID_FILE — job not running"
        ;;
esac

echo "---"

if [[ "$SHARDS_DONE" -gt 0 ]]; then
    echo "  [progress] checkpoint: $SHARDS_DONE / $TOTAL_SHARDS shards completed"
    if [[ "$VELOCITY_SPM" -gt 0 ]]; then
        echo "  [progress] velocity: $VELOCITY_SPM shards/min, elapsed ${PROC_ETIME_SEC}s, ETA ~${ETA_MIN}min"
    fi
else
    echo "  [progress] no chunking_checkpoint.json yet ($TOTAL_SHARDS shards available)"
fi

[[ "$VAL_LINES" -gt 0 ]]  && echo "  [progress] gold_pairs_val.jsonl  : $VAL_LINES lines"
[[ "$TEST_LINES" -gt 0 ]] && echo "  [progress] gold_pairs_test.jsonl : $TEST_LINES lines"
if [[ "$CORPUS_SIZE" -gt 0 ]]; then
    THRESHOLD=$((WC_SIZE_GUARD_GB * 1024 * 1024 * 1024))
    if [[ "$CORPUS_SIZE" -lt "$THRESHOLD" ]]; then
        echo "  [progress] corpus_chunks.jsonl   : $(wc -l < "$CORPUS") chunks, $(du -h "$CORPUS" | cut -f1)"
    else
        echo "  [progress] corpus_chunks.jsonl   : $(du -h "$CORPUS" | cut -f1) (skipping wc -l, > ${WC_SIZE_GUARD_GB}GB)"
    fi
fi

if [[ "$SUMMARY_PRESENT" -eq 1 ]]; then
    echo "  [progress] summary.json          : PRESENT"
    echo "  [progress] summary schema        : $SUMMARY_VALID"
else
    echo "  [progress] summary.json          : not yet written"
fi

echo "---"

if [[ -d "$OUT_DIR" ]]; then
    echo "  [artifacts] $OUT_DIR:"
    ls -lh "$OUT_DIR" 2>/dev/null | tail -n +2 | sed 's/^/    /'
else
    echo "  [artifacts] $OUT_DIR does not exist"
fi

# Strict mode exit
if [[ "$STRICT_MODE" -eq 1 ]]; then
    if [[ "$SUMMARY_PRESENT" -eq 1 ]] && [[ "$SUMMARY_VALID" != "VALID" ]]; then
        exit 10
    fi
    if [[ "$PROC_STATUS" == "stale_pid" ]] && [[ "$SUMMARY_PRESENT" -eq 0 ]]; then
        exit 11
    fi
fi
