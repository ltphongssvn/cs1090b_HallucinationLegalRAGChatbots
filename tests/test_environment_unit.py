# tests/test_environment_unit.py
# Unit tests for src/environment.py — mock-based, no GPU required.
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestCheckDeps:
    def test_check_constraint_ge_pass(self) -> None:
        from src.environment import _check_constraint

        ok, _ = _check_constraint("2.0.0", ">=2.0")
        assert ok

    def test_check_constraint_ge_fail(self) -> None:
        from src.environment import _check_constraint

        ok, msg = _check_constraint("1.0.0", ">=2.0")
        assert not ok
        assert "1.0.0" in msg

    def test_check_constraint_lt_pass(self) -> None:
        from src.environment import _check_constraint

        ok, _ = _check_constraint("4.39.0", "<4.41")
        assert ok

    def test_check_constraint_lt_fail(self) -> None:
        from src.environment import _check_constraint

        ok, _ = _check_constraint("4.41.0", "<4.41")
        assert not ok

    def test_check_constraint_combined(self) -> None:
        from src.environment import _check_constraint

        ok, _ = _check_constraint("4.38.0", ">=4.35,<4.41")
        assert ok

    def test_passes_when_all_deps_importable(self) -> None:
        from src.environment import _check_deps

        _check_deps()


class TestGetVersion:
    def test_strips_plus_suffix(self) -> None:
        from src.environment import _get_version

        mod = MagicMock()
        mod.__version__ = "2.0.1+cu117"
        assert _get_version(mod) == "2.0.1"

    def test_falls_back_to_VERSION_attr(self) -> None:
        from src.environment import _get_version

        mod = MagicMock(spec=["VERSION"])
        mod.VERSION = "3.0.0"
        assert _get_version(mod) == "3.0.0"

    def test_returns_none_if_no_attr(self) -> None:
        from src.environment import _get_version

        assert _get_version(MagicMock(spec=[])) is None


class TestRunEnvironmentChecks:
    @patch("src.environment._check_compat")
    @patch("src.environment._check_pytorch_cuda")
    @patch("src.environment._check_gpu_memory")
    @patch("src.environment._check_gpu_available")
    @patch("src.environment._check_deps")
    def test_returns_bool(self, *mocks) -> None:
        from src.environment import run_environment_checks

        result = run_environment_checks(logger=None)
        assert isinstance(result, bool)

    @patch("src.environment._check_compat")
    @patch("src.environment._check_pytorch_cuda")
    @patch("src.environment._check_gpu_memory")
    @patch("src.environment._check_gpu_available")
    @patch("src.environment._check_deps")
    def test_passes_with_mock_logger(self, *mocks) -> None:
        from src.environment import run_environment_checks

        logger = MagicMock()
        result = run_environment_checks(logger=logger)
        assert isinstance(result, bool)


class TestCheckCompat:
    def test_error_severity_rule_raises(self) -> None:
        from src.environment import CompatRule, _check_compat

        error_rule = CompatRule(name="test_error", check=lambda: True, message="hard blocker", severity="error")
        with patch("src.environment._build_compat_rules", return_value=[error_rule]):
            with pytest.raises(AssertionError, match="hard blocker"):
                _check_compat()

    def test_warn_severity_rule_does_not_raise(self) -> None:
        from src.environment import CompatRule, _check_compat

        warn_rule = CompatRule(name="test_warn", check=lambda: True, message="warn msg", severity="warn")
        with patch("src.environment._build_compat_rules", return_value=[warn_rule]):
            _check_compat()

    def test_no_rules_fire_does_not_raise(self) -> None:
        from src.environment import CompatRule, _check_compat

        never = CompatRule(name="never", check=lambda: False, message="x", severity="error")
        with patch("src.environment._build_compat_rules", return_value=[never]):
            _check_compat()

    def test_known_good_cluster_rule_is_warn(self) -> None:
        from src.environment import _build_compat_rules

        rules = _build_compat_rules()
        for rule in rules:
            if "torch_transformers" in rule.name:
                assert rule.severity == "warn"


class TestRequiredDeps:
    def test_all_keys_are_strings(self) -> None:
        from src.environment import REQUIRED_DEPS

        assert all(isinstance(k, str) for k in REQUIRED_DEPS)

    def test_contains_core_packages(self) -> None:
        from src.environment import REQUIRED_DEPS

        for pkg in ["torch", "transformers", "faiss", "spacy", "numpy"]:
            assert pkg in REQUIRED_DEPS

    def test_wandb_present(self) -> None:
        from src.environment import REQUIRED_DEPS

        assert "wandb" in REQUIRED_DEPS

    def test_accelerate_present(self) -> None:
        from src.environment import REQUIRED_DEPS

        assert "accelerate" in REQUIRED_DEPS


class TestCompatRuleDataclass:
    def test_compat_rule_fields(self) -> None:
        from src.environment import CompatRule

        rule = CompatRule(name="t", check=lambda: True, message="m", severity="error")
        assert rule.name == "t"
        assert rule.severity == "error"
        assert rule.check() is True

    def test_compat_rule_warn_severity(self) -> None:
        from src.environment import CompatRule

        rule = CompatRule(name="r", check=lambda: False, message="m", severity="warn")
        assert rule.severity == "warn"
        assert rule.check() is False
