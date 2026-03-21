#!/usr/bin/env bash
echo "========================================"
echo "  COVERAGE GATE - 80% Per-File Min"
echo "========================================"
uv run pytest tests/ -m "unit or contract" -q --tb=short --cov=src --cov-report=term-missing --no-header && echo "PASSED - Coverage gate green" || (echo "FAILED - Fix coverage before push" && exit 1)
