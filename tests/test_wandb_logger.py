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


class TestLogRunStart:
    @patch("wandb.init")
    @patch("wandb.summary", new_callable=dict)
    def test_initializes_wandb_run(self, mock_summary, mock_init, loader: DatasetLoader) -> None:
        from src.wandb_logger import log_run_start

        mock_run = MagicMock()
        mock_init.return_value = mock_run
        run = log_run_start(loader, run_name="test-run", project="test-project")
        mock_init.assert_called_once()
        assert run is mock_run

    @patch("wandb.init")
    @patch("wandb.summary", new_callable=dict)
    def test_config_includes_provenance(self, mock_summary, mock_init, loader: DatasetLoader) -> None:
        from src.wandb_logger import log_run_start

        mock_init.return_value = MagicMock()
        log_run_start(loader)
        config = mock_init.call_args.kwargs["config"]
        assert "revision" in config
        assert "dataset" in config
        assert "reproducible" in config

    @patch("wandb.init")
    @patch("wandb.summary", new_callable=dict)
    def test_extra_config_merged(self, mock_summary, mock_init, loader: DatasetLoader) -> None:
        from src.wandb_logger import log_run_start

        mock_init.return_value = MagicMock()
        log_run_start(loader, extra={"model": "bert-base", "lr": 2e-5})
        config = mock_init.call_args.kwargs["config"]
        assert config["model"] == "bert-base"
        assert config["lr"] == 2e-5


class TestLogDatasetStats:
    @patch("wandb.log")
    @patch("wandb.Table")
    @patch("wandb.plot")
    def test_logs_required_keys(self, mock_plot, mock_table, mock_log, loader: DatasetLoader, valid_row: dict) -> None:
        from src.wandb_logger import log_dataset_stats

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1] * 42)
        mock_tokenizer.name_or_path = "bert-base"
        stats = log_dataset_stats(loader, mock_tokenizer, [valid_row])
        assert stats["n_valid"] == 1
        assert mock_log.called

    @patch("wandb.log")
    @patch("wandb.Table")
    @patch("wandb.plot")
    def test_returns_stats_dict(self, mock_plot, mock_table, mock_log, loader: DatasetLoader, valid_row: dict) -> None:
        from src.wandb_logger import log_dataset_stats

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1] * 10)
        stats = log_dataset_stats(loader, mock_tokenizer, [valid_row])
        assert "avg_token_length" in stats
        assert "court_distribution" in stats


class TestLogQualitySignals:
    @patch("wandb.log")
    def test_logs_signal_counts(self, mock_log, valid_row: dict) -> None:
        from src.wandb_logger import log_quality_signals

        row_with_html = {**valid_row, "text": "<p>Court held.</p> " * 5}
        result = log_quality_signals([row_with_html])
        assert isinstance(result, dict)
        assert mock_log.called

    @patch("wandb.log")
    def test_clean_rows_no_signals(self, mock_log, valid_row: dict) -> None:
        from src.wandb_logger import log_quality_signals

        result = log_quality_signals([])
        assert result == {}
        assert not mock_log.called
