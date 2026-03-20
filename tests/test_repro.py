# tests/test_repro.py
# Unit tests for src/repro.py — pure functions only, no GPU required.
import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


class TestConstants:
    def test_random_seed_is_zero(self) -> None:
        from src.repro import _RANDOM_SEED

        assert _RANDOM_SEED == 0

    def test_pythonhashseed_is_string_zero(self) -> None:
        from src.repro import _EXPECTED_PYTHONHASHSEED

        assert _EXPECTED_PYTHONHASHSEED == "0"

    def test_cublas_cfg_starts_with_colon(self) -> None:
        from src.repro import _EXPECTED_CUBLAS_CFG

        assert _EXPECTED_CUBLAS_CFG.startswith(":")

    def test_tokenizers_par_is_false_string(self) -> None:
        from src.repro import _EXPECTED_TOKENIZERS_PAR

        assert _EXPECTED_TOKENIZERS_PAR == "false"


class TestLoadDotenv:
    def test_raises_when_env_missing(self, tmp_path: Path) -> None:
        from src.repro import _load_dotenv

        with pytest.raises(FileNotFoundError, match=".env not found"):
            _load_dotenv(project_root=tmp_path)

    def test_loads_env_file_without_dotenv_package(self, tmp_path: Path) -> None:
        from src.repro import _load_dotenv

        env_file = tmp_path / ".env"
        env_file.write_text("export TEST_VAR_REPRO=hello\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_VAR_REPRO", None)
            with patch("builtins.__import__", side_effect=ImportError):
                _load_dotenv(project_root=tmp_path)
            assert os.environ.get("TEST_VAR_REPRO") == "hello"
            os.environ.pop("TEST_VAR_REPRO", None)

    def test_does_not_override_existing_env_vars(self, tmp_path: Path) -> None:
        from src.repro import _load_dotenv

        env_file = tmp_path / ".env"
        env_file.write_text("export EXISTING_VAR=new_value\n")
        with patch.dict(os.environ, {"EXISTING_VAR": "original"}, clear=False):
            with patch("builtins.__import__", side_effect=ImportError):
                _load_dotenv(project_root=tmp_path)
            assert os.environ["EXISTING_VAR"] == "original"


class TestSeedAll:
    def test_seeds_random_module(self) -> None:
        import random

        from src.repro import _seed_all

        _seed_all(42)
        v1 = random.random()
        _seed_all(42)
        v2 = random.random()
        assert v1 == v2

    def test_seeds_numpy(self) -> None:
        import numpy as np

        from src.repro import _seed_all

        _seed_all(42)
        v1 = np.random.rand()
        _seed_all(42)
        v2 = np.random.rand()
        assert v1 == v2

    def test_different_seeds_produce_different_values(self) -> None:
        import random

        from src.repro import _seed_all

        _seed_all(1)
        v1 = random.random()
        _seed_all(2)
        v2 = random.random()
        assert v1 != v2


class TestApplyTorchFlags:
    def test_sets_cublas_env_var(self) -> None:
        from src.repro import _apply_torch_flags

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            _apply_torch_flags()
            assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

    def test_deterministic_algorithms_enabled(self) -> None:
        import torch

        from src.repro import _apply_torch_flags

        _apply_torch_flags()
        assert torch.are_deterministic_algorithms_enabled()

    def test_cudnn_benchmark_disabled(self) -> None:
        import torch

        from src.repro import _apply_torch_flags

        _apply_torch_flags()
        assert torch.backends.cudnn.benchmark is False

    def test_cudnn_deterministic_enabled(self) -> None:
        import torch

        from src.repro import _apply_torch_flags

        _apply_torch_flags()
        assert torch.backends.cudnn.deterministic is True


class TestSeedAllTorch:
    def test_seeds_torch(self) -> None:
        import torch

        from src.repro import _seed_all

        _seed_all(42)
        v1 = torch.rand(1).item()
        _seed_all(42)
        v2 = torch.rand(1).item()
        assert v1 == v2


class TestVerify:
    def test_verify_passes_with_correct_env(self) -> None:
        from src.repro import _apply_torch_flags, _verify

        with patch.dict(
            os.environ,
            {
                "PYTHONHASHSEED": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "TOKENIZERS_PARALLELISM": "false",
            },
            clear=False,
        ):
            _apply_torch_flags()
            result = _verify()
            assert result["deterministic_algorithms"] is True
            assert result["cudnn_benchmark"] is False
            assert result["cudnn_deterministic"] is True

    def test_verify_raises_on_wrong_pythonhashseed(self) -> None:
        from src.repro import _verify

        with patch.dict(os.environ, {"PYTHONHASHSEED": "99"}, clear=False):
            with pytest.raises(AssertionError, match="PYTHONHASHSEED"):
                _verify()


class TestConfigure:
    def test_configure_verbose_prints(self, tmp_path: Path, capsys) -> None:
        from src.repro import configure

        env_file = tmp_path / ".env"
        env_file.write_text("PYTHONHASHSEED=0\nCUBLAS_WORKSPACE_CONFIG=:4096:8\nTOKENIZERS_PARALLELISM=false\n")
        with patch.dict(
            os.environ,
            {
                "PYTHONHASHSEED": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "TOKENIZERS_PARALLELISM": "false",
            },
            clear=False,
        ):
            configure(project_root=tmp_path, verbose=True)
        captured = capsys.readouterr()
        assert "Reproducibility configured" in captured.out

    def test_configure_returns_dict(self, tmp_path: Path) -> None:
        from src.repro import configure

        env_file = tmp_path / ".env"
        env_file.write_text("PYTHONHASHSEED=0\nCUBLAS_WORKSPACE_CONFIG=:4096:8\nTOKENIZERS_PARALLELISM=false\n")
        with patch.dict(
            os.environ,
            {
                "PYTHONHASHSEED": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "TOKENIZERS_PARALLELISM": "false",
            },
            clear=False,
        ):
            result = configure(project_root=tmp_path, verbose=False)
        assert isinstance(result, dict)
        assert "deterministic_algorithms" in result
