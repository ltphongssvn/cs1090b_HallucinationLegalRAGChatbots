"""Test wandb_sync_diagnostics --verify-cloud mode."""

import subprocess

import pytest


@pytest.mark.contract
class TestWandbCloudVerification:
    def test_script_supports_verify_cloud_flag(self):
        """Script must declare --verify-cloud flag."""
        proc = subprocess.run(
            [".venv/bin/python", "scripts/wandb_sync_diagnostics.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert proc.returncode == 0
        assert "--verify-cloud" in proc.stdout, "missing --verify-cloud flag"

    def test_verify_cloud_runs_and_reports(self):
        """--verify-cloud must query W&B API and report cloud run inventory."""
        proc = subprocess.run(
            [".venv/bin/python", "scripts/wandb_sync_diagnostics.py", "--verify-cloud"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert proc.returncode == 0, f"script failed: {proc.stderr}"
        for marker in [
            "CLOUD VERIFICATION",
            "Total cloud runs",
            "PROJECT",
        ]:
            assert marker in proc.stdout, f"missing '{marker}' in output"
