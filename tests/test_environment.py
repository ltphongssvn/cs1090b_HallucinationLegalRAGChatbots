import pytest

pytestmark = pytest.mark.unit

# tests/test_environment.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_environment.py
# TDD RED: Environment contract tests — runnable via pytest.

from unittest.mock import MagicMock

import pytest

from src.environment import (
    REQUIRED_DEPS,
    _check_constraint,
    _get_version,
    get_environment_summary,
)


class TestGetVersion:
    def test_extracts_dunder_version(self):
        mod = MagicMock(__version__="1.2.3")
        assert _get_version(mod) == "1.2.3"

    def test_strips_cuda_suffix(self):
        mod = MagicMock(__version__="2.0.1+cu117")
        assert _get_version(mod) == "2.0.1"

    def test_returns_none_if_no_version(self):
        mod = MagicMock(spec=[])
        assert _get_version(mod) is None


class TestCheckConstraint:
    def test_ge_passes(self):
        ok, _ = _check_constraint("2.1.0", ">=2.0")
        assert ok

    def test_ge_fails(self):
        ok, reason = _check_constraint("1.9.0", ">=2.0")
        assert not ok
        assert "1.9.0" in reason

    def test_lt_passes(self):
        ok, _ = _check_constraint("4.35.0", ">=4.35,<4.41")
        assert ok

    def test_lt_fails(self):
        ok, reason = _check_constraint("4.42.0", ">=4.35,<4.41")
        assert not ok

    def test_exact_boundary_ge(self):
        ok, _ = _check_constraint("2.0", ">=2.0")
        assert ok

    def test_exact_boundary_lt(self):
        ok, _ = _check_constraint("4.41", ">=4.35,<4.41")
        assert not ok


class TestRequiredDeps:
    def test_all_deps_have_constraint_or_none(self):
        for pkg, constraint in REQUIRED_DEPS.items():
            assert constraint is None or constraint.startswith(">="), f"{pkg}: constraint must start with >= or be None"

    def test_expected_packages_present(self):
        expected = {"torch", "transformers", "datasets", "numpy", "pandas"}
        assert expected.issubset(set(REQUIRED_DEPS.keys()))


class TestGetEnvironmentSummary:
    def test_returns_dict_with_python(self):
        summary = get_environment_summary()
        assert "python" in summary
        assert "." in summary["python"]
