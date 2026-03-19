# tests/test_wandb_logger.py
from unittest.mock import MagicMock, patch

import pytest

from src.dataset_config import DatasetConfig
from src.dataset_loader import DatasetLoader

pytestmark = pytest.mark.unit


@pytest.fixture
def loader() -> DatasetLoader:
    return DatasetLoader(DatasetConfig())


@pytest.fixture
def valid_row() -> dict:
    return {
        "text": "The court held that the defendant failed. " * 3,
        "created_timestamp": "2022-01-15",
        "downloaded_timestamp": "2022-06-01",
        "url": "https://courtlistener.com/opinion/1/",
    }


class TestSetupWandbAuth:
    @patch("wandb.login")
    def test_uses_api_key_from_env(self, mock_login, monkeypatch) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "test-key-abc")
        monkeypatch.delenv("WANDB_MODE", raising=False)
        from src.wandb_logger import setup_wandb_auth

        setup_wandb_auth()
        mock_login.assert_called_once_with(key="test-key-abc", relogin=False)

    @patch("wandb.login")
    def test_skips_login_in_offline_mode(self, mock_login, monkeypatch) -> None:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_MODE", "offline")
        from src.wandb_logger import setup_wandb_auth

        setup_wandb_auth()
        mock_login.assert_not_called()

    @patch("wandb.login")
    def test_skips_login_in_disabled_mode(self, mock_login, monkeypatch) -> None:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_MODE", "disabled")
        from src.wandb_logger import setup_wandb_auth

        setup_wandb_auth()
        mock_login.assert_not_called()

    @patch("wandb.login")
    def test_falls_back_to_cached_credentials(self, mock_login, monkeypatch) -> None:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        from src.wandb_logger import setup_wandb_auth

        setup_wandb_auth()
        mock_login.assert_called_once_with(relogin=False)


class TestLoadArtifact:
    @patch("wandb.Api")
    def test_downloads_artifact_to_local_path(self, mock_api_cls, tmp_path) -> None:
        from src.wandb_logger import load_artifact

        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_artifact = MagicMock()
        mock_artifact.download.return_value = str(tmp_path)
        mock_api.artifact.return_value = mock_artifact
        result = load_artifact("entity/project/dataset:v1", str(tmp_path))
        mock_api.artifact.assert_called_once_with("entity/project/dataset:v1")
        mock_artifact.download.assert_called_once_with(root=str(tmp_path))
        assert result == str(tmp_path)


class TestLogRunStart:
    @patch("wandb.init")
    @patch("wandb.summary", new_callable=dict)
    def test_initializes_wandb_run(self, mock_summary, mock_init, loader) -> None:
        from src.wandb_logger import log_run_start

        mock_run = MagicMock()
        mock_init.return_value = mock_run
        run = log_run_start(loader, run_name="test-run", project="test-project")
        mock_init.assert_called_once()
        assert run is mock_run

    @patch("wandb.init")
    @patch("wandb.summary", new_callable=dict)
    def test_config_includes_provenance(self, mock_summary, mock_init, loader) -> None:
        from src.wandb_logger import log_run_start

        mock_init.return_value = MagicMock()
        log_run_start(loader)
        config = mock_init.call_args.kwargs["config"]
        assert "revision" in config and "dataset" in config and "reproducible" in config

    @patch("wandb.init")
    @patch("wandb.summary", new_callable=dict)
    def test_extra_config_merged(self, mock_summary, mock_init, loader) -> None:
        from src.wandb_logger import log_run_start

        mock_init.return_value = MagicMock()
        log_run_start(loader, extra={"model": "bert-base", "lr": 2e-5})
        config = mock_init.call_args.kwargs["config"]
        assert config["model"] == "bert-base" and config["lr"] == 2e-5


class TestLogDatasetStats:
    @patch("wandb.log")
    @patch("wandb.Table")
    @patch("wandb.plot")
    def test_logs_required_keys(self, mock_plot, mock_table, mock_log, loader, valid_row) -> None:
        from src.wandb_logger import log_dataset_stats

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1] * 42)
        mock_tokenizer.name_or_path = "bert-base"
        stats = log_dataset_stats(loader, mock_tokenizer, [valid_row])
        assert stats["n_valid"] == 1 and mock_log.called

    @patch("wandb.log")
    @patch("wandb.Table")
    @patch("wandb.plot")
    def test_returns_stats_dict(self, mock_plot, mock_table, mock_log, loader, valid_row) -> None:
        from src.wandb_logger import log_dataset_stats

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1] * 10)
        stats = log_dataset_stats(loader, mock_tokenizer, [valid_row])
        assert "avg_token_length" in stats and "court_distribution" in stats


class TestLogQualitySignals:
    @patch("wandb.log")
    def test_logs_signal_counts(self, mock_log, valid_row) -> None:
        from src.wandb_logger import log_quality_signals

        row_with_html = {**valid_row, "text": "<p>Court held.</p> " * 5}
        result = log_quality_signals([row_with_html])
        assert isinstance(result, dict) and mock_log.called

    @patch("wandb.log")
    def test_clean_rows_no_signals(self, mock_log) -> None:
        from src.wandb_logger import log_quality_signals

        result = log_quality_signals([])
        assert result == {} and not mock_log.called
