#!/usr/bin/env bash
# Monitor MS3 baseline_prep run — auto-discovers PID + latest log + semantic progress + ETA.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PID_FILE="logs/baseline_prep.pid"
OUT_DIR="data/processed/baseline"
SHARD_DIR="${SHARD_DIR:-data/raw/cl_federal_appellate_bulk}"
WC_SIZE_GUARD_GB="${WC_SIZE_GUARD_GB:-5}"  # skip wc -l on files > this size

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Monitor MS3 baseline_prep run."
            echo "Usage: bash scripts/monitor_baseline.sh"
            exit 0
            ;;
    esac
done

echo "=== baseline_prep monitor  ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="

# --- Section 1: log tail (independent) ---
LOG_FILE=$(ls -t logs/baseline_prep_*.log 2>/dev/null | head -1)
if [[ -z "${LOG_FILE:-}" ]]; then
    echo "  [log] no log files under logs/baseline_prep_*.log"
else
    echo "  [log] $LOG_FILE"
    echo "  --- last 20 lines ---"
    tail -n 20 "$LOG_FILE" 2>/dev/null || echo "  [warn] log read failed"
fi

echo "---"

# --- Section 2: process (independent) ---
PROC_ETIME_SEC=0
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        PS_OUT=$(ps -p "$PID" -o pid,etime,%mem,rss,stat,args 2>/dev/null | tail -n 1)
        echo "  [proc] $PS_OUT"
        if [[ "$PS_OUT" != *"baseline_prep"* ]]; then
            echo "  [warn] PID $PID does not look like baseline_prep — possible PID reuse"
        fi
        # Parse etime (format: [[dd-]hh:]mm:ss) to seconds for ETA
        ETIME_RAW=$(ps -p "$PID" -o etimes= 2>/dev/null | tr -d ' ')
        PROC_ETIME_SEC="${ETIME_RAW:-0}"
    else
        echo "  [proc] PID $PID not running (stale PID file or job finished)"
    fi
else
    echo "  [proc] no PID file at $PID_FILE — job not running"
fi

echo "---"

# --- Section 3: semantic progress (independent) ---
CKPT="$OUT_DIR/chunking_checkpoint.json"
SUMMARY="$OUT_DIR/summary.json"
VAL="$OUT_DIR/gold_pairs_val.jsonl"
TEST="$OUT_DIR/gold_pairs_test.jsonl"
CORPUS="$OUT_DIR/corpus_chunks.jsonl"

# Dynamic total shards (no hardcoded 159)
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
    echo "  [progress] checkpoint: $SHARDS_DONE / $TOTAL_SHARDS shards completed"

    # ETA calculation (velocity = shards / elapsed sec)
    if [[ "$SHARDS_DONE" -gt 0 ]] && [[ "$PROC_ETIME_SEC" -gt 0 ]]; then
        REMAINING=$((TOTAL_SHARDS - SHARDS_DONE))
        if [[ "$REMAINING" -gt 0 ]]; then
            # Shards per minute (int math, so multiply first)
            SPM=$(( SHARDS_DONE * 60 / PROC_ETIME_SEC ))
            if [[ "$SPM" -gt 0 ]]; then
                ETA_MIN=$((REMAINING / SPM))
                echo "  [progress] velocity: $SPM shards/min, elapsed ${PROC_ETIME_SEC}s, ETA ~${ETA_MIN}min"
            fi
        fi
    fi
else
    echo "  [progress] no chunking_checkpoint.json yet ($TOTAL_SHARDS shards available)"
fi

if [[ -f "$VAL" ]]; then
    echo "  [progress] gold_pairs_val.jsonl  : $(wc -l < "$VAL") lines"
fi
if [[ -f "$TEST" ]]; then
    echo "  [progress] gold_pairs_test.jsonl : $(wc -l < "$TEST") lines"
fi
if [[ -f "$CORPUS" ]]; then
    # Size guard: avoid wc -l on multi-GB files
    CORPUS_SIZE=$(stat -c '%s' "$CORPUS" 2>/dev/null || echo 0)
    THRESHOLD=$((WC_SIZE_GUARD_GB * 1024 * 1024 * 1024))
    if [[ "$CORPUS_SIZE" -lt "$THRESHOLD" ]]; then
        echo "  [progress] corpus_chunks.jsonl   : $(wc -l < "$CORPUS") chunks, $(du -h "$CORPUS" | cut -f1)"
    else
        echo "  [progress] corpus_chunks.jsonl   : $(du -h "$CORPUS" | cut -f1) (skipping wc -l, size > ${WC_SIZE_GUARD_GB}GB)"
    fi
fi

if [[ -f "$SUMMARY" ]]; then
    echo "  [progress] summary.json          : PRESENT (job complete)"
    VALID=$(PYTHONPATH="$REPO_ROOT" uv run python -c "
from src.eda_schemas import BaselinePrepSummary
from pathlib import Path
try:
    BaselinePrepSummary.model_validate_json(Path('$SUMMARY').read_bytes())
    print('VALID')
except Exception as e:
    print(f'INVALID: {e}')
" 2>/dev/null || echo "check-failed")
    echo "  [progress] summary schema        : $VALID"
else
    echo "  [progress] summary.json          : not yet written"
fi

echo "---"

# --- Section 4: artifact listing ---
if [[ -d "$OUT_DIR" ]]; then
    echo "  [artifacts] $OUT_DIR:"
    ls -lh "$OUT_DIR" 2>/dev/null | tail -n +2 | sed 's/^/    /'
else
    echo "  [artifacts] $OUT_DIR does not exist"
fi
