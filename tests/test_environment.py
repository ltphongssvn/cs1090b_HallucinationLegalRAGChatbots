# tests/test_environment.py
# Project: HallucinationLegalRAGChatbots
# TDD RED: Environment contract tests — runnable via pytest.

from unittest.mock import MagicMock

import pytest

from src.environment import (
    REQUIRED_DEPS,
    _check_constraint,
    _get_version,
    get_environment_summary,
)

pytestmark = pytest.mark.unit


class TestCheckConstraint:
    def test_passes_when_version_meets_minimum(self) -> None:
        ok, _ = _check_constraint("2.1.0", ">=2.0")
        assert ok

    def test_fails_when_version_below_minimum(self) -> None:
        ok, msg = _check_constraint("1.9.0", ">=2.0")
        assert not ok
        assert "1.9.0" in msg

    def test_passes_upper_bound(self) -> None:
        ok, _ = _check_constraint("4.39.0", ">=4.35,<4.41")
        assert ok

    def test_fails_upper_bound(self) -> None:
        ok, msg = _check_constraint("4.41.0", ">=4.35,<4.41")
        assert not ok

    def test_passes_exact_minimum(self) -> None:
        ok, _ = _check_constraint("2.0.0", ">=2.0")
        assert ok


class TestGetVersion:
    def test_reads_version_attr(self) -> None:
        mod = MagicMock()
        mod.__version__ = "1.2.3"
        assert _get_version(mod) == "1.2.3"

    def test_strips_cuda_suffix(self) -> None:
        mod = MagicMock()
        mod.__version__ = "2.0.1+cu117"
        assert _get_version(mod) == "2.0.1"

    def test_returns_none_when_no_version_attr(self) -> None:
        mod = MagicMock(spec=[])
        assert _get_version(mod) is None


class TestRequiredDeps:
    def test_required_deps_contains_torch(self) -> None:
        assert "torch" in REQUIRED_DEPS

    def test_required_deps_contains_transformers(self) -> None:
        assert "transformers" in REQUIRED_DEPS

    def test_required_deps_is_dict(self) -> None:
        assert isinstance(REQUIRED_DEPS, dict)


class TestGetEnvironmentSummary:
    def test_returns_dict_with_python(self) -> None:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("No GPU available in CI — get_environment_summary requires CUDA")
        summary = get_environment_summary()
        assert isinstance(summary, dict)
        assert "python" in summary

    def test_summary_contains_required_keys(self) -> None:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("No GPU available in CI — get_environment_summary requires CUDA")
        summary = get_environment_summary()
        assert "gpu" in summary
        assert "cuda" in summary
        assert "torch" in summary
