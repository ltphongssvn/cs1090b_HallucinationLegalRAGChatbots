# tests/test_pipeline.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_pipeline.py
import pytest

pytestmark = pytest.mark.unit

import logging
from pathlib import Path
from unittest.mock import patch

from src.config import PipelineConfig
from src.exceptions import PipelineError
from src.pipeline import run_pipeline, validate_pipeline
from src.schemas import FilterResult


class TestRunPipelineSkipsIfComplete:
    @patch("src.pipeline.validate_manifest_shards", return_value=True)
    @patch("src.pipeline.read_manifest", return_value={"num_cases": 100, "num_shards": 1})
    def test_returns_existing_manifest(self, mock_read, mock_validate, tmp_path):
        config = PipelineConfig(shard_dir=tmp_path)
        result = run_pipeline(config=config)
        assert result["num_cases"] == 100
        mock_read.assert_called_once()

    @patch("src.pipeline.validate_manifest_shards", return_value=True)
    @patch("src.pipeline.read_manifest", return_value={"num_cases": 50, "num_shards": 1})
    def test_logs_skip(self, mock_read, mock_validate, tmp_path):
        config = PipelineConfig(shard_dir=tmp_path)
        logger = logging.getLogger("test_skip")
        msgs: list = []
        h = logging.Handler()
        h.emit = lambda r: msgs.append(r.getMessage())  # type: ignore
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
        run_pipeline(config=config, logger=logger)
        assert any("Already complete" in m for m in msgs)


class TestRunPipelinePinned:
    @patch("src.pipeline.write_manifest", return_value={"version": 2, "num_cases": 5})
    @patch(
        "src.pipeline.extract_opinions_to_shards",
        return_value={
            "extracted_total": 5,
            "num_shards": 1,
            "text_source_counts": {},
            "skipped_empty": 0,
            "skipped_parse": 0,
            "scanned": 10,
        },
    )
    @patch("src.pipeline.build_federal_appellate_filter")
    @patch(
        "src.pipeline.download_bulk_csvs",
        return_value={
            "courts": Path("c.csv"),
            "dockets": Path("d.csv"),
            "clusters": Path("cl.csv"),
            "opinions": Path("o.csv"),
        },
    )
    @patch("src.pipeline.validate_manifest_shards", return_value=False)
    @patch("src.pipeline.read_manifest", return_value={})
    def test_pinned_snapshot_skips_discovery(
        self, mock_read, mock_validate, mock_dl, mock_filter, mock_extract, mock_write, tmp_path
    ):
        config = PipelineConfig(
            shard_dir=tmp_path,
            pinned_courts="bulk-data/courts-2025-12-31.csv.bz2",
            pinned_dockets="bulk-data/dockets-2025-12-31.csv.bz2",
            pinned_clusters="bulk-data/opinion-clusters-2025-12-31.csv.bz2",
            pinned_opinions="bulk-data/opinions-2025-12-31.csv.bz2",
        )
        mock_filter.return_value = FilterResult(
            fed_court_ids={"ca1"},
            court_name_map={"ca1": "1st"},
            docket_meta={1: {}},
            cluster_meta={100: {}},
        )
        result = run_pipeline(config=config)
        assert result["version"] == 2
        mock_dl.assert_called_once()

    @patch("src.pipeline.write_manifest", return_value={"version": 2, "num_cases": 0})
    @patch(
        "src.pipeline.extract_opinions_to_shards",
        return_value={
            "extracted_total": 0,
            "num_shards": 0,
            "text_source_counts": {},
            "skipped_empty": 0,
            "skipped_parse": 0,
            "scanned": 0,
        },
    )
    @patch("src.pipeline.build_federal_appellate_filter")
    @patch(
        "src.pipeline.download_bulk_csvs",
        return_value={
            "courts": Path("c"),
            "dockets": Path("d"),
            "clusters": Path("cl"),
            "opinions": Path("o"),
        },
    )
    @patch(
        "src.pipeline.discover_latest_bulk_files",
        return_value={
            "courts": {"key": "k"},
            "dockets": {"key": "k"},
            "clusters": {"key": "k"},
            "opinions": {"key": "k"},
        },
    )
    @patch("src.pipeline.validate_manifest_shards", return_value=False)
    @patch("src.pipeline.read_manifest", return_value={})
    def test_unpinned_calls_discovery(
        self, mock_read, mock_validate, mock_discover, mock_dl, mock_filter, mock_extract, mock_write, tmp_path
    ):
        config = PipelineConfig(shard_dir=tmp_path)
        mock_filter.return_value = FilterResult(
            fed_court_ids={"ca1"},
            court_name_map={"ca1": "1st"},
            docket_meta={1: {}},
            cluster_meta={100: {}},
        )
        run_pipeline(config=config)
        mock_discover.assert_called_once()


class TestRunPipelineNoPinnedFiles:
    @patch("src.pipeline.validate_manifest_shards", return_value=False)
    @patch("src.pipeline.read_manifest", return_value={})
    def test_raises_on_none_pinned(self, mock_read, mock_validate, tmp_path):
        config = PipelineConfig(shard_dir=tmp_path)
        config.pinned_courts = "x"  # partial pin → has_pinned_snapshot=False
        with patch("src.pipeline.discover_latest_bulk_files", return_value=None):
            with pytest.raises(PipelineError, match="No pinned files"):
                run_pipeline(config=config)


class TestValidatePipeline:
    @patch("src.pipeline.run_contract_tests", return_value=True)
    def test_passes(self, mock_tests, tmp_path):
        assert validate_pipeline(config=PipelineConfig(shard_dir=tmp_path)) is True

    @patch("src.pipeline.run_contract_tests", return_value=False)
    def test_raises_on_failure(self, mock_tests, tmp_path):
        with pytest.raises(PipelineError, match="contract tests failed"):
            validate_pipeline(config=PipelineConfig(shard_dir=tmp_path))


class TestRunPipelineWithLogger:
    @patch("src.pipeline.write_manifest", return_value={"version": 2, "num_cases": 5})
    @patch(
        "src.pipeline.extract_opinions_to_shards",
        return_value={
            "extracted_total": 5,
            "num_shards": 1,
            "text_source_counts": {},
            "skipped_empty": 0,
            "skipped_parse": 0,
            "scanned": 10,
        },
    )
    @patch("src.pipeline.build_federal_appellate_filter")
    @patch(
        "src.pipeline.download_bulk_csvs",
        return_value={
            "courts": Path("c"),
            "dockets": Path("d"),
            "clusters": Path("cl"),
            "opinions": Path("o"),
        },
    )
    @patch(
        "src.pipeline.discover_latest_bulk_files",
        return_value={
            "courts": {"key": "k"},
            "dockets": {"key": "k"},
            "clusters": {"key": "k"},
            "opinions": {"key": "k"},
        },
    )
    @patch("src.pipeline.validate_manifest_shards", return_value=False)
    @patch("src.pipeline.read_manifest", return_value={})
    def test_logs_all_steps(
        self, mock_read, mock_validate, mock_discover, mock_dl, mock_filter, mock_extract, mock_write, tmp_path
    ):
        from src.schemas import FilterResult

        config = PipelineConfig(shard_dir=tmp_path)
        mock_filter.return_value = FilterResult(
            fed_court_ids={"ca1"},
            court_name_map={"ca1": "1st"},
            docket_meta={1: {}},
            cluster_meta={100: {}},
        )
        logger = logging.getLogger("test_steps")
        msgs: list = []
        h = logging.Handler()
        h.emit = lambda r: msgs.append(r.getMessage())  # type: ignore
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
        run_pipeline(config=config, logger=logger)
        assert any("STEP 1" in m for m in msgs)
        assert any("STEP 2" in m for m in msgs)
        assert any("STEP 3" in m for m in msgs)
        assert any("STEP 4" in m for m in msgs)
        assert any("Manifest" in m for m in msgs)
