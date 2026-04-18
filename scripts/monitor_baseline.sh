#!/usr/bin/env bash
# Monitor MS3 baseline_prep run — auto-discovers PID + latest log + semantic progress.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PID_FILE="logs/baseline_prep.pid"
OUT_DIR="data/processed/baseline"

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
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        PS_OUT=$(ps -p "$PID" -o pid,etime,%mem,rss,stat,args 2>/dev/null | tail -n 1)
        echo "  [proc] $PS_OUT"
        if [[ "$PS_OUT" != *"baseline_prep"* ]]; then
            echo "  [warn] PID $PID does not look like baseline_prep — possible PID reuse"
        fi
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

if [[ -f "$CKPT" ]]; then
    SHARDS_DONE=$(python3 -c "import json; print(len(json.load(open('$CKPT'))['completed']))" 2>/dev/null || echo "?")
    echo "  [progress] checkpoint: $SHARDS_DONE / 159 shards completed"
else
    echo "  [progress] no chunking_checkpoint.json yet"
fi

if [[ -f "$VAL" ]]; then
    echo "  [progress] gold_pairs_val.jsonl  : $(wc -l < "$VAL") lines"
fi
if [[ -f "$TEST" ]]; then
    echo "  [progress] gold_pairs_test.jsonl : $(wc -l < "$TEST") lines"
fi
if [[ -f "$CORPUS" ]]; then
    echo "  [progress] corpus_chunks.jsonl   : $(wc -l < "$CORPUS") chunks, $(du -h "$CORPUS" | cut -f1)"
fi

if [[ -f "$SUMMARY" ]]; then
    echo "  [progress] summary.json          : PRESENT (job complete)"
    if command -v uv >/dev/null 2>&1; then
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
    fi
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
