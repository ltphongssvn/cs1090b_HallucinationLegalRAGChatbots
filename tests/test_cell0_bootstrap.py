"""Test src/notebook_cells/cell0_bootstrap.py extraction."""

from pathlib import Path

import pytest


@pytest.mark.contract
class TestCell0Bootstrap:
    def test_module_exists(self):
        assert Path("src/notebook_cells/cell0_bootstrap.py").is_file()

    def test_module_exports_run(self):
        from src.notebook_cells import cell0_bootstrap

        assert callable(getattr(cell0_bootstrap, "run", None))

    def test_run_returns_dict_with_required_keys(self):
        from src.notebook_cells.cell0_bootstrap import run

        result = run()
        for key in ["repo_root", "cwd", "sys_path_0"]:
            assert key in result, f"missing key: {key}"
        assert Path(result["repo_root"]).is_absolute()
        assert (Path(result["repo_root"]) / "src" / "__init__.py").is_file()

    def test_run_resets_accumulator(self):
        """run() must call reset_accumulated() so kernel restart starts fresh."""
        from src.notebook_cells.cell0_bootstrap import run
        from src.timer import cell_timer, get_accumulated_seconds

        with cell_timer("warmup", _override_elapsed=10.0):
            pass
        assert get_accumulated_seconds() >= 10.0
        run()
        assert get_accumulated_seconds() == 0.0
