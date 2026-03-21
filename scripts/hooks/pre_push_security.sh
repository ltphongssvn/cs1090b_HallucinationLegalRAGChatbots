#!/usr/bin/env bash
mkdir -p logs
uv run pip-audit --format=json --progress-spinner=off --output=logs/pip-audit-precommit.json 2>&1 || true
uv run python scripts/ci_audit_report.py logs/pip-audit-precommit.json
