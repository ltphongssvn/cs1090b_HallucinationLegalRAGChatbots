# tests/test_hf_push.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_hf_push.py
# TDD RED: HuggingFace Datasets export (local only, no actual push).

import json
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from src.config import PipelineConfig
from src.hf_export import build_dataset_info, shards_to_hf_dataset


class TestBuildDatasetInfo:
    def test_contains_manifest_fields(self):
        manifest = {
            "version": 2,
            "num_cases": 100,
            "num_shards": 1,
            "federal_courts": ["ca1", "ca9"],
            "text_length_stats": {"mean": 5000},
            "run_metadata": {"git_revision": "abc123"},
        }
        info = build_dataset_info(manifest)
        assert info["dataset_name"] == "legal-rag-federal-appellate"
        assert info["num_examples"] == 100
        assert info["version"] == 2
        assert "ca1" in info["federal_courts"]
        assert info["run_metadata"]["git_revision"] == "abc123"


class TestShardsToHfDataset:
    def test_loads_shards_into_dataset(self, tmp_path):
        shard = tmp_path / "shard_0000.jsonl"
        shard.write_text(
            json.dumps({"id": 1, "text": "hello", "court_id": "ca1"})
            + "\n"
            + json.dumps({"id": 2, "text": "world", "court_id": "ca9"})
            + "\n"
        )
        config = PipelineConfig(shard_dir=tmp_path)
        ds = shards_to_hf_dataset(config)
        assert len(ds) == 2
        assert ds[0]["id"] == 1

    def test_empty_dir_returns_empty(self, tmp_path):
        config = PipelineConfig(shard_dir=tmp_path)
        ds = shards_to_hf_dataset(config)
        assert len(ds) == 0


class TestPushToHub:
    @patch("src.hf_export.shards_to_hf_dataset")
    def test_empty_dataset_warns(self, mock_ds):
        import logging

        from datasets import Dataset

        from src.hf_export import push_to_hub

        mock_ds.return_value = Dataset.from_list([])
        logger = logging.getLogger("test_push")
        msgs: list = []
        h = logging.Handler()
        h.emit = lambda r: msgs.append(r.getMessage())  # type: ignore
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
        push_to_hub(logger=logger)
        assert any("No records" in m for m in msgs)

    @patch("src.hf_export.shards_to_hf_dataset")
    def test_calls_push(self, mock_shards):
        from datasets import Dataset

        from src.hf_export import push_to_hub

        ds = Dataset.from_list([{"id": 1, "text": "hello"}])
        ds.push_to_hub = MagicMock()
        mock_shards.return_value = ds
        push_to_hub(manifest={"num_cases": 1})
        ds.push_to_hub.assert_called_once()


class TestShardsToHfDatasetBranches:
    def test_multiple_shards(self, tmp_path):
        """Loads records from multiple shard files."""
        import json

        from src.hf_export import shards_to_hf_dataset

        config = PipelineConfig(shard_dir=tmp_path)
        for i in range(3):
            (tmp_path / f"shard_{i:04d}.jsonl").write_text(json.dumps({"id": i, "text": f"record {i}"}) + "\n")
        ds = shards_to_hf_dataset(config)
        assert len(ds) == 3

    def test_push_with_manifest_sets_description(self):
        """push_to_hub attaches manifest info as description."""
        from datasets import Dataset

        from src.hf_export import push_to_hub

        ds = Dataset.from_list([{"id": 1, "text": "hello"}])
        ds.push_to_hub = MagicMock()
        with patch("src.hf_export.shards_to_hf_dataset", return_value=ds):
            import logging

            logger = logging.getLogger("test_push_desc")
            msgs: list = []
            h = logging.Handler()
            h.emit = lambda r: msgs.append(r.getMessage())  # type: ignore
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG)
            push_to_hub(manifest={"num_cases": 1}, repo_id="test-repo", logger=logger)
        ds.push_to_hub.assert_called_once()
        assert any("Pushing" in m for m in msgs)
        assert any("test-repo" in m for m in msgs)
