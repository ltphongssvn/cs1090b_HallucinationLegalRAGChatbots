"""Test W&B sync diagnostics script."""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.contract
class TestWandbSyncDiagnostics:
    def test_script_exists(self):
        assert Path("scripts/wandb_sync_diagnostics.py").is_file(), "scripts/wandb_sync_diagnostics.py must exist"

    def test_script_runs_check_only(self):
        """Script must support --check-only mode (no mutations) and report state."""
        proc = subprocess.run(
            [".venv/bin/python", "scripts/wandb_sync_diagnostics.py", "--check-only"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert proc.returncode == 0, f"script failed: {proc.stderr}"
        for marker in [
            "OFFLINE-RUN QUEUE",
            "NETWORK REACHABILITY",
            "AUTHENTICATION",
            "Pending offline runs",
        ]:
            assert marker in proc.stdout, f"missing '{marker}' in output"

    def test_script_supports_sync_and_clean_flags(self):
        """Script must declare --sync and --clean flags."""
        proc = subprocess.run(
            [".venv/bin/python", "scripts/wandb_sync_diagnostics.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert proc.returncode == 0
        assert "--check-only" in proc.stdout
        assert "--sync" in proc.stdout
        assert "--clean" in proc.stdout
