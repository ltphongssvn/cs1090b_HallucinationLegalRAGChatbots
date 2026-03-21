# tests/test_bulk_download.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_bulk_download.py
import pytest

pytestmark = pytest.mark.unit

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.bulk_download import (
    _download_via_aws_cli,
    _download_via_requests,
    download_bulk_csvs,
    download_file,
)
from src.config import PipelineConfig


class TestDownloadViaAwsCli:
    @patch("src.bulk_download.subprocess.run")
    def test_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        assert _download_via_aws_cli("bucket", "key", tmp_path / "f.csv") is True

    @patch("src.bulk_download.subprocess.run")
    def test_failure(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        assert _download_via_aws_cli("bucket", "key", tmp_path / "f.csv") is False

    @patch("src.bulk_download.subprocess.run", side_effect=FileNotFoundError)
    def test_aws_not_installed(self, mock_run: MagicMock, tmp_path: Path) -> None:
        assert _download_via_aws_cli("bucket", "key", tmp_path / "f.csv") is False

    @patch("src.bulk_download.subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("aws", 10))
    def test_timeout(self, mock_run: MagicMock, tmp_path: Path) -> None:
        assert _download_via_aws_cli("bucket", "key", tmp_path / "f.csv") is False


class TestDownloadViaRequests:
    @patch("src.bulk_download.requests.get")
    def test_writes_content(self, mock_get: MagicMock, tmp_path: Path) -> None:
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "5"}
        mock_resp.iter_content.return_value = [b"hello"]
        mock_get.return_value = mock_resp
        out = tmp_path / "test.csv"
        _download_via_requests("http://example.com", "key", out)
        assert out.read_bytes() == b"hello"


class TestDownloadFile:
    def test_skips_existing(self, tmp_path: Path) -> None:
        existing = tmp_path / "exists.csv"
        existing.write_text("data")
        logger = logging.getLogger("test_dl")
        msgs: list = []
        h = logging.Handler()
        h.emit = lambda r: msgs.append(r.getMessage())  # type: ignore
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
        download_file("key", existing, logger=logger)
        assert any("exists, skipping" in m for m in msgs)

    @patch("src.bulk_download._download_via_aws_cli", return_value=True)
    def test_uses_aws_cli(self, mock_aws: MagicMock, tmp_path: Path) -> None:
        out = tmp_path / "new.csv"
        download_file("key", out)
        mock_aws.assert_called_once()
        assert not out.exists()  # aws cli is mocked, no real file

    @patch("src.bulk_download._download_via_requests")
    @patch("src.bulk_download._download_via_aws_cli", return_value=False)
    def test_falls_back_to_requests(self, mock_aws: MagicMock, mock_req: MagicMock, tmp_path: Path) -> None:
        out = tmp_path / "new.csv"
        download_file("key", out)
        mock_req.assert_called_once()

    @patch("src.bulk_download._download_via_aws_cli", return_value=True)
    def test_logs_aws_success(self, mock_aws: MagicMock, tmp_path: Path) -> None:
        logger = logging.getLogger("test_dl_aws")
        msgs: list = []
        h = logging.Handler()
        h.emit = lambda r: msgs.append(r.getMessage())  # type: ignore
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
        download_file("key", tmp_path / "new.csv", logger=logger)
        assert any("aws CLI" in m for m in msgs)

    @patch("src.bulk_download._download_via_requests")
    @patch("src.bulk_download._download_via_aws_cli", return_value=False)
    def test_logs_requests_fallback(self, mock_aws: MagicMock, mock_req: MagicMock, tmp_path: Path) -> None:
        logger = logging.getLogger("test_dl_req")
        msgs: list = []
        h = logging.Handler()
        h.emit = lambda r: msgs.append(r.getMessage())  # type: ignore
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
        download_file("key", tmp_path / "new.csv", logger=logger)
        assert any("requests" in m.lower() for m in msgs)


class TestDownloadBulkCsvs:
    @patch("src.bulk_download.download_file")
    def test_creates_bulk_dir(self, mock_dl: MagicMock, tmp_path: Path) -> None:
        config = PipelineConfig(bulk_dir=tmp_path / "bulk")
        latest = {"courts": {"key": "bulk-data/courts-2025-01-01.csv.bz2"}}
        result = download_bulk_csvs(latest, config=config)
        assert (tmp_path / "bulk").exists()
        assert "courts" in result

    @patch("src.bulk_download.download_file")
    def test_returns_local_paths(self, mock_dl: MagicMock, tmp_path: Path) -> None:
        config = PipelineConfig(bulk_dir=tmp_path / "bulk")
        latest = {
            "courts": {"key": "bulk-data/courts-2025-01-01.csv.bz2"},
            "dockets": {"key": "bulk-data/dockets-2025-01-01.csv.bz2"},
        }
        result = download_bulk_csvs(latest, config=config)
        assert len(result) == 2
        assert result["courts"].name == "courts-2025-01-01.csv.bz2"
