# tests/test_environment_unit.py
# Unit tests for src/environment.py — mock-based, no GPU required.
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestCheckDeps:
    def test_passes_when_all_deps_importable(self) -> None:
        from src.environment import _check_deps

        _check_deps()

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
    def test_returns_bool(self) -> None:
        from src.environment import run_environment_checks

        result = run_environment_checks(logger=None)
        assert isinstance(result, bool)

    def test_passes_with_mock_logger(self) -> None:
        from src.environment import run_environment_checks

        logger = MagicMock()
        result = run_environment_checks(logger=logger)
        assert isinstance(result, bool)


class TestCheckCompat:
    def test_no_error_for_known_good_cluster_combo(self) -> None:
        # torch==2.0.1+cu117 + transformers<4.41 is the actual cluster config.
        # The compat rule is severity="warn" for this combo — must NOT raise.
        # Tests the logic in isolation, not the live cluster packages.

        mock_torch = MagicMock()
        mock_torch.__version__ = "2.0.1+cu117"
        mock_tf = MagicMock()
        mock_tf.__version__ = "4.39.3"

        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
            from src.environment import _build_compat_rules

            rules = _build_compat_rules()
            # The torch/transformers rule must be "warn", not "error"
            for rule in rules:
                if "torch_transformers" in rule.name:
                    assert rule.severity == "warn", (
                        f"Rule '{rule.name}' must be severity='warn' for the "
                        "validated L4/CUDA11.7 cluster — not 'error'"
                    )

    def test_error_severity_rule_raises(self) -> None:
        # Verify that "error"-severity rules DO raise AssertionError.
        # Uses a synthetic rule injected via _build_compat_rules mock.
        from src.environment import CompatRule, _check_compat

        fake_error_rule = CompatRule(
            name="test_hard_blocker",
            check=lambda: True,  # always fires
            message="synthetic hard blocker for test",
            severity="error",
        )
        with patch("src.environment._build_compat_rules", return_value=[fake_error_rule]):
            with pytest.raises(AssertionError, match="synthetic hard blocker"):
                _check_compat()

    def test_warn_severity_rule_does_not_raise(self) -> None:
        # Verify that "warn"-severity rules do NOT raise — they only print/log.
        from src.environment import CompatRule, _check_compat

        fake_warn_rule = CompatRule(
            name="test_known_risk",
            check=lambda: True,  # always fires
            message="synthetic warn for test",
            severity="warn",
        )
        with patch("src.environment._build_compat_rules", return_value=[fake_warn_rule]):
            # Must not raise
            _check_compat()

    def test_no_rules_fire_does_not_raise(self) -> None:
        from src.environment import CompatRule, _check_compat

        never_fires = CompatRule(
            name="never",
            check=lambda: False,
            message="should never fire",
            severity="error",
        )
        with patch("src.environment._build_compat_rules", return_value=[never_fires]):
            _check_compat()  # must not raise


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

        rule = CompatRule(
            name="test",
            check=lambda: True,
            message="msg",
            severity="error",
        )
        assert rule.name == "test"
        assert rule.severity == "error"
        assert rule.check() is True

    def test_compat_rule_warn_severity(self) -> None:
        from src.environment import CompatRule

        rule = CompatRule(name="r", check=lambda: False, message="m", severity="warn")
        assert rule.severity == "warn"
        assert rule.check() is False
