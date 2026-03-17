import pytest

pytestmark = pytest.mark.unit

# tests/test_config.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_config.py
# TDD RED: Config dataclass contract.

from pathlib import Path

from src.config import PipelineConfig


class TestPipelineConfig:
    def test_defaults_exist(self):
        config = PipelineConfig()
        assert config.shard_size == 10000
        assert config.min_text_length == 50
        assert len(config.federal_appellate_court_ids) == 13

    def test_manifest_path_derived(self):
        config = PipelineConfig(shard_dir=Path("/tmp/test_shards"))
        assert config.manifest_path == Path("/tmp/test_shards/manifest.json")

    def test_needed_files_has_four_entries(self):
        config = PipelineConfig()
        assert set(config.needed_files.keys()) == {"courts", "dockets", "clusters", "opinions"}

    def test_all_circuits_present(self):
        config = PipelineConfig()
        expected = {"ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10", "ca11", "cadc", "cafc"}
        assert config.federal_appellate_court_ids == frozenset(expected)

    def test_overridable(self):
        config = PipelineConfig(shard_size=500, min_text_length=100)
        assert config.shard_size == 500
        assert config.min_text_length == 100


class TestPipelineConfigBranches:
    def test_pinned_files_returns_none_when_partial(self):
        config = PipelineConfig(pinned_courts="x")
        assert config.pinned_files is None

    def test_quarantine_path_default_none(self):
        config = PipelineConfig()
        assert config.quarantine_path is None

    def test_quarantine_path_set(self, tmp_path):
        config = PipelineConfig(quarantine_path=tmp_path / "q.csv")
        assert config.quarantine_path == tmp_path / "q.csv"
