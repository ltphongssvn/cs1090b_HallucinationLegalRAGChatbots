"""RED tests for src.wandb_lineage — must fail before module exists.

Properties to encode:
  1. Module exists and exports link_input_artifacts.
  2. Returns [] when wandb is not installed (silent skip).
  3. Returns [] when no wandb run is active.
  4. Returns [] when wandb run is offline (W&B issue #5309).
  5. Calls run.use_artifact(name, type=...) once per name when run is live.
  6. Returns the list of resolved Artifact objects in input order.
  7. Single failed use_artifact does not abort remaining calls.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestWandbLineageContract:
    def test_module_exposes_link_input_artifacts(self) -> None:
        from src.wandb_lineage import link_input_artifacts

        assert callable(link_input_artifacts)

    def test_returns_empty_when_wandb_missing(self) -> None:
        from src.wandb_lineage import link_input_artifacts

        # simulate ImportError on `import wandb`
        with patch.dict(sys.modules, {"wandb": None}):
            result = link_input_artifacts(["a:latest", "b:latest"])
        assert result == []

    def test_returns_empty_when_no_active_run(self) -> None:
        from src.wandb_lineage import link_input_artifacts

        fake_wandb = MagicMock()
        fake_wandb.run = None
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            result = link_input_artifacts(["a:latest"])
        assert result == []

    def test_returns_empty_when_run_offline(self) -> None:
        from src.wandb_lineage import link_input_artifacts

        fake_run = MagicMock()
        fake_run.offline = True
        fake_wandb = MagicMock()
        fake_wandb.run = fake_run
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            result = link_input_artifacts(["a:latest"])
        assert result == []
        fake_run.use_artifact.assert_not_called()

    def test_calls_use_artifact_per_name_when_live(self) -> None:
        from src.wandb_lineage import link_input_artifacts

        fake_run = MagicMock()
        fake_run.offline = False
        fake_run.use_artifact.side_effect = lambda name, type: f"art({name}, {type})"
        fake_wandb = MagicMock()
        fake_wandb.run = fake_run
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            result = link_input_artifacts(["bm25:latest", "bge-m3:latest"], artifact_type="dataset")
        assert fake_run.use_artifact.call_count == 2
        assert result == ["art(bm25:latest, dataset)", "art(bge-m3:latest, dataset)"]

    def test_partial_failure_is_logged_and_skipped(self) -> None:
        from src.wandb_lineage import link_input_artifacts

        fake_run = MagicMock()
        fake_run.offline = False

        def _maybe_fail(name, type):
            if name == "bad:latest":
                raise RuntimeError("artifact not found")
            return f"art({name})"

        fake_run.use_artifact.side_effect = _maybe_fail
        fake_wandb = MagicMock()
        fake_wandb.run = fake_run
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            result = link_input_artifacts(["good:latest", "bad:latest", "ok:latest"])
        assert result == ["art(good:latest)", "art(ok:latest)"]
