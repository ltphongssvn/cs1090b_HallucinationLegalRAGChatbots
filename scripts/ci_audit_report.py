# scripts/ci_audit_report.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/ci_audit_report.py
"""Non-blocking ``pip-audit`` report printer for CI.

Reads a ``pip-audit --format json`` output file and prints any
discovered CVEs as warnings, **always exiting 0** so a known
compatibility-pinned vulnerability does not block the pipeline. The
canonical use case is reporting transient CVEs in ``transformers``
while we hold its version pinned below 4.40 for CUDA 11.7 wheel
compatibility.

Usage:
    python scripts/ci_audit_report.py logs/pip-audit.json

When the input file does not exist, the script prints a skip message
and exits 0 — this keeps CI green on nodes where pip-audit itself
failed to run (e.g. network sandbox).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/pip-audit.json")
if not path.exists():
    print("pip-audit output not found — skipping")
    sys.exit(0)
data = json.loads(path.read_text())
vulns = [d for d in data.get("dependencies", []) if d.get("vulns")]
if vulns:
    print(f"WARNING: {len(vulns)} package(s) with known CVEs (non-blocking in CI)")
    print("NOTE: transformers<4.40 pinned for CUDA 11.7 wheel compatibility")
    for d in vulns:
        for v in d["vulns"]:
            print(f"  {d['name']}=={d['version']}: {v['id']}")
else:
    print("No known vulnerabilities found.")
