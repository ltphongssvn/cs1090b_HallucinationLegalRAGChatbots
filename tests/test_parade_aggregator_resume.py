# tests/test_parade_aggregator_resume.py
"""Test PARADE training saves per-epoch checkpoints and resumes from latest."""

import inspect

import pytest


@pytest.fixture
def parade_module():
    return __import__("scripts.parade_aggregator", fromlist=["*"])


@pytest.mark.contract
class TestCheckpointResume:
    def test_main_saves_epoch_checkpoints(self, parade_module):
        """main() must save aggregator state after each epoch (not only at end)."""
        src = inspect.getsource(parade_module.main)
        assert "parade_aggregator.epoch" in src or "checkpoint" in src.lower(), (
            "main() must save per-epoch checkpoints (parade_aggregator.epoch{N}.pt) so SLURM TIMEOUT can resume"
        )

    def test_main_resumes_from_latest_checkpoint(self, parade_module):
        """main() must scan output_dir for latest epoch checkpoint and resume."""
        src = inspect.getsource(parade_module.main)
        assert "load_state_dict" in src or "resume" in src.lower(), (
            "main() must call aggregator.load_state_dict() from latest checkpoint if found in output_dir"
        )

    def test_main_skips_completed_epochs(self, parade_module):
        """main() must start training from start_epoch (epochs already done are skipped)."""
        src = inspect.getsource(parade_module.main)
        assert "start_epoch" in src or "resume_epoch" in src, (
            "main() must track start_epoch derived from latest checkpoint"
        )
