# tests/test_environment_coverage.py
# Project: HallucinationLegalRAGChatbots
import logging

import pytest

pytestmark = pytest.mark.unit
from unittest.mock import MagicMock, patch

from src.environment import (
    _check_compat,
    _check_deps,
    _check_gpu_available,
    _check_gpu_memory,
    _check_pytorch_cuda,
    get_environment_summary,
    run_environment_checks,
)


class TestCheckDeps:
    @patch("src.environment.importlib.import_module", side_effect=ImportError("missing"))
    def test_missing_dep_raises(self, mock_import):
        with pytest.raises(AssertionError, match="not installed"):
            _check_deps()

    @patch("src.environment.importlib.import_module")
    def test_undetectable_version_raises(self, mock_import):
        mod = MagicMock(spec=[])  # no __version__
        mock_import.return_value = mod
        with pytest.raises(AssertionError, match="undetectable"):
            _check_deps()


class TestCheckGpu:
    @patch("torch.cuda.is_available", return_value=False)
    def test_no_gpu_raises(self, mock_cuda):
        with pytest.raises(AssertionError, match="No CUDA"):
            _check_gpu_available()

    @patch("torch.cuda.get_device_properties")
    def test_low_memory_raises(self, mock_props):
        mock_props.return_value = MagicMock(total_memory=4e9)
        with pytest.raises(AssertionError):
            _check_gpu_memory()

    @patch("torch.version", MagicMock(cuda=None))
    def test_no_cuda_build_raises(self):
        with pytest.raises(AssertionError, match="without CUDA"):
            _check_pytorch_cuda()


class TestCheckCompat:
    def test_compat_warn_rule_does_not_raise(self):
        # _check_compat with a warn-severity rule must NOT raise.
        # Uses the CompatRule mock pattern — tests logic, not live packages.
        from src.environment import CompatRule

        warn_rule = CompatRule(name="test_warn", check=lambda: True, message="warn msg", severity="warn")
        with patch("src.environment._build_compat_rules", return_value=[warn_rule]):
            _check_compat()  # must not raise

    def test_compat_error_rule_raises(self):
        # _check_compat with an error-severity rule MUST raise AssertionError.
        from src.environment import CompatRule

        error_rule = CompatRule(name="test_error", check=lambda: True, message="hard blocker", severity="error")
        with patch("src.environment._build_compat_rules", return_value=[error_rule]):
            with pytest.raises(AssertionError, match="hard blocker"):
                _check_compat()

    def test_compat_rule_not_firing_does_not_raise(self):
        from src.environment import CompatRule

        never = CompatRule(name="never", check=lambda: False, message="never fires", severity="error")
        with patch("src.environment._build_compat_rules", return_value=[never]):
            _check_compat()


class TestRunEnvironmentChecks:
    @patch("src.environment._check_compat")
    @patch("src.environment._check_pytorch_cuda")
    @patch("src.environment._check_gpu_memory")
    @patch("src.environment._check_gpu_available")
    @patch("src.environment._check_deps")
    def test_all_pass(self, *mocks):
        assert run_environment_checks() is True

    @patch("src.environment._check_compat")
    @patch("src.environment._check_pytorch_cuda")
    @patch("src.environment._check_gpu_memory")
    @patch("src.environment._check_gpu_available", side_effect=AssertionError("no gpu"))
    @patch("src.environment._check_deps")
    def test_failure_returns_false(self, *mocks):
        assert run_environment_checks() is False

    @patch("src.environment._check_compat")
    @patch("src.environment._check_pytorch_cuda")
    @patch("src.environment._check_gpu_memory")
    @patch("src.environment._check_gpu_available")
    @patch("src.environment._check_deps")
    def test_logs_pass(self, *mocks):
        logger = logging.getLogger("test_env")
        msgs: list = []
        h = logging.Handler()
        h.emit = lambda r: msgs.append(r.getMessage())  # type: ignore
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
        run_environment_checks(logger=logger)
        assert any("PASS" in m for m in msgs)


class TestGetEnvironmentSummary:
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.get_device_name", return_value="TestGPU")
    @patch("torch.cuda.is_available", return_value=True)
    def test_summary_has_gpu(self, mock_avail, mock_name, mock_props):
        mock_props.return_value = MagicMock(total_memory=24e9)
        import torch

        with patch.object(torch.version, "cuda", "11.7"):
            summary = get_environment_summary()
        assert summary["gpu"] == "TestGPU"
        assert summary["python"]


class TestCheckDepsSuccess:
    @patch("src.environment.importlib.import_module")
    def test_all_deps_pass(self, mock_import):
        versions = {
            "torch": "2.0.1",
            "transformers": "4.40.0",
            "datasets": "2.16.0",
            "gensim": "4.3.0",
            "spacy": "3.7.0",
            "faiss": "1.7.0",
            "langchain": "0.1.0",
            "sklearn": "1.3.0",
            "numpy": "1.24.0",
            "pandas": "2.1.0",
        }

        def side_effect(name):
            m = MagicMock()
            m.__version__ = versions.get(name, "1.0.0")
            return m

        mock_import.side_effect = side_effect
        _check_deps()


class TestCheckCompatSuccess:
    def test_compat_passes_with_no_firing_rules(self):
        # Proves _check_compat passes when no rules fire.
        from src.environment import CompatRule

        no_fire = CompatRule(name="no_fire", check=lambda: False, message="won't fire", severity="error")
        with patch("src.environment._build_compat_rules", return_value=[no_fire]):
            _check_compat()
