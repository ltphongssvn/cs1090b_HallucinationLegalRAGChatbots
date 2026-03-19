# tests/test_ingest.py
import json
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def valid_row() -> dict:
    return {
        "text": "The court held that the defendant failed to establish. " * 3,
        "created_timestamp": "2022-01-15",
        "downloaded_timestamp": "2022-06-01",
        "url": "https://courtlistener.com/opinion/1/",
    }


def _mock_hf_source(rows):
    mock = MagicMock()
    mock.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    return mock


class TestRunIngestion:
    @patch("wandb.finish")
    @patch("wandb.log")
    @patch("wandb.summary", new_callable=dict)
    @patch("wandb.Table")
    @patch("wandb.init")
    @patch("datasets.load_dataset")
    def test_dry_run_returns_counts(
        self, mock_load, mock_init, mock_table, mock_summary, mock_log, mock_finish, valid_row, tmp_path
    ) -> None:
        from src.ingest import run_ingestion

        mock_init.return_value = MagicMock()
        mock_load.return_value = _mock_hf_source([valid_row, valid_row])
        result = run_ingestion(str(tmp_path), max_samples=10, dry_run=True, wandb_mode="disabled")
        assert result["dry_run"] is True
        assert result["n_valid"] == 2
        assert result["n_rejected"] == 0

    @patch("wandb.finish")
    @patch("wandb.Artifact")
    @patch("wandb.log")
    @patch("wandb.summary", new_callable=dict)
    @patch("wandb.Table")
    @patch("wandb.init")
    @patch("datasets.load_dataset")
    @patch("datasets.Dataset")
    def test_full_run_writes_manifest(
        self,
        mock_ds_cls,
        mock_load,
        mock_init,
        mock_table,
        mock_summary,
        mock_log,
        mock_artifact,
        mock_finish,
        valid_row,
        tmp_path,
    ) -> None:
        from src.ingest import run_ingestion

        mock_run = MagicMock()
        mock_init.return_value = mock_run
        mock_load.return_value = _mock_hf_source([valid_row])
        mock_ds_instance = MagicMock()
        mock_ds_cls.from_list.return_value = mock_ds_instance
        result = run_ingestion(str(tmp_path), max_samples=10, wandb_mode="disabled")
        assert "n_valid" in result
        assert "checksum" in result
        manifest_path = tmp_path / "artifact_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "row_count" in manifest
        assert "revision" in manifest

    @patch("wandb.finish")
    @patch("wandb.log")
    @patch("wandb.summary", new_callable=dict)
    @patch("wandb.Table")
    @patch("wandb.init")
    @patch("datasets.load_dataset")
    def test_rejected_rows_tracked(
        self, mock_load, mock_init, mock_table, mock_summary, mock_log, mock_finish, valid_row, tmp_path
    ) -> None:
        from src.ingest import run_ingestion

        mock_init.return_value = MagicMock()
        bad_row = {"url": "x"}  # missing required fields
        mock_load.return_value = _mock_hf_source([valid_row, bad_row])
        result = run_ingestion(str(tmp_path), max_samples=10, dry_run=True, wandb_mode="disabled")
        assert result["n_rejected"] == 1
        assert result["n_valid"] == 1

    @patch("wandb.finish")
    @patch("wandb.log")
    @patch("wandb.summary", new_callable=dict)
    @patch("wandb.Table")
    @patch("wandb.init")
    @patch("datasets.load_dataset")
    def test_wandb_init_called_with_ingestion_job_type(
        self, mock_load, mock_init, mock_table, mock_summary, mock_log, mock_finish, valid_row, tmp_path
    ) -> None:
        from src.ingest import run_ingestion

        mock_init.return_value = MagicMock()
        mock_load.return_value = _mock_hf_source([])
        run_ingestion(str(tmp_path), dry_run=True, wandb_mode="disabled")
        assert mock_init.call_args.kwargs["job_type"] == "ingestion"
