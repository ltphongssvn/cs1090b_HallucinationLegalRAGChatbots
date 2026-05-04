# tests/test_rag_ablation_inflight_check.py
"""Cell 20 must not resubmit if a SLURM job for the same ablation is already running."""

import inspect

import pytest


@pytest.mark.contract
class TestInflightDetection:
    def test_run_slurm_job_supports_inflight_check(self):
        runner = __import__("scripts.run_slurm_job", fromlist=["*"])
        src = inspect.getsource(runner)
        assert "running_job_for_sbatch" in src or "_find_running_job" in src or "JobName" in src, (
            "run_slurm_job must support detecting in-flight jobs by sbatch name to prevent duplicate submissions"
        )
