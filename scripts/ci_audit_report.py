# scripts/ci_audit_report.py
# Non-blocking pip-audit report for CI. Prints CVEs as warnings, never exits 1.
# Usage: python scripts/ci_audit_report.py logs/pip-audit.json
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
