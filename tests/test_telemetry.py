"""
Telemetry contract tests for scripts/audit_jsonl_nan.py
Tests W&B payload mapping, lazy import guard, and job_type tagging.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts.audit_jsonl_nan import DatasetHealth, log_health_to_wandb


class TestWandbPayloadMapping:
    def test_init_called_with_job_type(self):
        with patch("wandb.init") as mock_init:
            mock_init.return_value = MagicMock()
            h = DatasetHealth(1000, 0, 0, 5, {}, [])
            log_health_to_wandb(h, project="test-project")
            _, kwargs = mock_init.call_args
            assert kwargs.get("job_type") == "data-quality-gate"

    def test_init_called_with_correct_project(self):
        with patch("wandb.init") as mock_init:
            mock_init.return_value = MagicMock()
            h = DatasetHealth(1000, 0, 0, 5, {}, [])
            log_health_to_wandb(h, project="my-project")
            _, kwargs = mock_init.call_args
            assert kwargs.get("project") == "my-project"

    def test_metrics_keys_use_slash_grouping(self):
        logged = {}
        with patch("wandb.init") as mock_init:
            mock_run = MagicMock()
            mock_run.log = lambda d: logged.update(d)
            mock_init.return_value = mock_run
            h = DatasetHealth(1000, 50, 1, 5, {"case_name": 50}, ["s.jsonl"], nonfinite_lines=50, decode_error_lines=5)
            log_health_to_wandb(h, project="test")
        assert "data/nonfinite_lines" in logged
        assert "data/decode_error_lines" in logged
        assert "data/gate_verdict" in logged

    def test_wandb_finish_called(self):
        with patch("wandb.init") as mock_init:
            mock_run = MagicMock()
            mock_init.return_value = mock_run
            h = DatasetHealth(1000, 0, 0, 5, {}, [])
            log_health_to_wandb(h, project="test")
            mock_run.finish.assert_called_once()


class TestLazyImportGuard:
    def test_graceful_when_wandb_missing(self, caplog):
        import importlib
        import sys

        # temporarily hide wandb
        real_wandb = sys.modules.get("wandb")
        sys.modules["wandb"] = None  # type: ignore
        try:
            import importlib

            import scripts.audit_jsonl_nan as m

            importlib.reload(m)
            # should not raise
            h = DatasetHealth(100, 0, 0, 5, {}, [])
            m.log_health_to_wandb(h, project="test")
        except Exception as e:
            pytest.fail(f"Should not raise when wandb missing: {e}")
        finally:
            if real_wandb is not None:
                sys.modules["wandb"] = real_wandb
