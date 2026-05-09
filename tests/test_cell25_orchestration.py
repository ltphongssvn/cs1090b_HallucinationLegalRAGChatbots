"""Test src/notebook_cells/cell25_wandb_repro.py extraction + accumulated time."""

from pathlib import Path

import pytest


@pytest.mark.contract
class TestCell25Extraction:
    def test_module_exists(self):
        assert Path("src/notebook_cells/__init__.py").is_file()
        assert Path("src/notebook_cells/cell25_wandb_repro.py").is_file()

    def test_module_exports_run(self):
        """Module must expose a single run() entry point."""
        from src.notebook_cells import cell25_wandb_repro

        assert callable(getattr(cell25_wandb_repro, "run", None)), "cell25_wandb_repro.run() must be defined"

    def test_run_returns_dict_with_required_keys(self):
        """run() must return a dict with diagnostic + return code."""
        from src.notebook_cells.cell25_wandb_repro import run

        result = run(skip_pytest=True, skip_network=True)
        assert isinstance(result, dict)
        for key in ["wandb_config", "helper_counts", "lineage", "queue", "rc"]:
            assert key in result, f"missing key: {key}"

    def test_accumulated_time_helper(self):
        """src.timer must expose accumulated time across cells."""
        from src.timer import format_accumulated, get_accumulated_seconds

        assert callable(get_accumulated_seconds)
        assert callable(format_accumulated)
        # Format should be MM:SS or HH:MM:SS
        formatted = format_accumulated(125.4)
        assert "2m" in formatted and "05" in formatted, f"unexpected format: {formatted}"
