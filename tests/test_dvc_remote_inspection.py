"""Test the DVC remote inspection script returns valid summary."""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.contract
class TestDVCRemoteInspection:
    def test_script_exists(self):
        assert Path("scripts/inspect_dvc_remote.py").is_file(), "scripts/inspect_dvc_remote.py must exist"

    def test_script_runs_and_reports_totals(self):
        """Script must run successfully and report artifact count + total size."""
        proc = subprocess.run(
            [".venv/bin/python", "scripts/inspect_dvc_remote.py"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert proc.returncode == 0, f"script failed: {proc.stderr}"
        assert "Total artifacts:" in proc.stdout, "must report 'Total artifacts:' count"
        assert "Total size:" in proc.stdout, "must report 'Total size:' aggregate"
        assert "Remote:" in proc.stdout, "must report remote URL"
