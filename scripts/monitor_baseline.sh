#!/usr/bin/env bash
# Monitor MS3 baseline_prep run — auto-discovers PID + latest log.
#
# Usage:
#   bash scripts/monitor_baseline.sh          # one-shot status
#   watch -n 10 bash scripts/monitor_baseline.sh   # live refresh
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

# Latest log (auto-discover via glob, not hardcoded filename)
LOG_FILE=$(ls -t logs/baseline_prep_*.log 2>/dev/null | head -1)

echo "=== baseline_prep monitor  ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
if [[ -z "${LOG_FILE:-}" ]]; then
    echo "  no log files under logs/baseline_prep_*.log"
else
    echo "  log: $LOG_FILE"
    echo "  --- last 8 lines ---"
    tail -n 8 "$LOG_FILE"
fi

echo
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "  process (PID=$PID):"
        ps -p "$PID" -o pid,etime,%mem,rss,stat,cmd 2>&1 | head -2
    else
        echo "  PID $PID not running (stale PID file or job finished)"
    fi
else
    echo "  no PID file at $PID_FILE — job not running"
fi

echo
if [[ -d "$OUT_DIR" ]]; then
    echo "  artifacts in $OUT_DIR:"
    ls -lh "$OUT_DIR" 2>/dev/null | tail -n +2
else
    echo "  No artifacts yet in $OUT_DIR"
fi
