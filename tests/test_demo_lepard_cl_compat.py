"""Tests for scripts/demo_lepard_cl_compat.py (TDD Red-first)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO = REPO_ROOT / "scripts" / "demo_lepard_cl_compat.py"


class TestDemoScript:
    def test_demo_script_exists(self):
        assert DEMO.exists(), f"demo script not found at {DEMO}"

    def test_demo_runs_and_prints_report(self):
        r = subprocess.run(
            [sys.executable, str(DEMO), "--no-narrative"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "LePaRD" in r.stdout
        assert "USABLE GOLD" in r.stdout

    def test_demo_json_mode_emits_valid_json(self):
        r = subprocess.run(
            [sys.executable, str(DEMO), "--json"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        data = json.loads(r.stdout)
        assert "id_overlap" in data
        assert "pair_overlap" in data
        assert data["pair_overlap"]["unique_pairs"] == 454
        assert data["pair_overlap"]["both_in_cl"] == 13

    def test_demo_narrative_mode_includes_context(self):
        r = subprocess.run(
            [sys.executable, str(DEMO)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert r.returncode == 0
        assert "Colab" in r.stdout or "CourtListener" in r.stdout
        assert "Interpretation" in r.stdout
