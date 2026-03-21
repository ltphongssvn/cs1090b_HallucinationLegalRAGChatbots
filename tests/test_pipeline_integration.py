# tests/test_pipeline_integration.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_pipeline_integration.py
# TDD RED: End-to-end pipeline on frozen fixtures + typed filter result.

import pytest

pytestmark = pytest.mark.integration

from src.extract import extract_opinions_to_shards
from src.filter_chain import build_federal_appellate_filter
from src.manifest import read_manifest, validate_manifest_shards, write_manifest
from src.schemas import FilterResult


class TestFilterResultTyped:
    def test_returns_filter_result(self, fixture_paths, fixture_config):
        result = build_federal_appellate_filter(fixture_paths, config=fixture_config)
        assert isinstance(result, FilterResult)
        assert len(result.fed_court_ids) > 0
        assert len(result.court_name_map) > 0
        assert len(result.docket_meta) > 0
        assert len(result.cluster_meta) > 0


class TestPipelineEndToEnd:
    def test_extract_then_manifest_then_validate(self, fixture_paths, fixture_config):
        filter_result = build_federal_appellate_filter(fixture_paths, config=fixture_config)
        stats = extract_opinions_to_shards(
            fixture_paths["opinions"],
            filter_result.cluster_meta,
            filter_result.docket_meta,
            filter_result.court_name_map,
            config=fixture_config,
        )
        assert stats["extracted_total"] > 0

        manifest = write_manifest(
            fixture_config.manifest_path,
            fixture_config.shard_dir,
            stats,
            fixture_paths,
            filter_result.fed_court_ids,
            len(filter_result.docket_meta),
            len(filter_result.cluster_meta),
            fixture_config.shard_size,
            config=fixture_config,
        )
        assert manifest["version"] == 2
        assert manifest["num_cases"] == stats["extracted_total"]

        reloaded = read_manifest(fixture_config.manifest_path)
        assert reloaded["num_cases"] == stats["extracted_total"]
        assert validate_manifest_shards(reloaded, fixture_config.shard_dir)

    def test_manifest_checksums_valid(self, fixture_paths, fixture_config):
        filter_result = build_federal_appellate_filter(fixture_paths, config=fixture_config)
        stats = extract_opinions_to_shards(
            fixture_paths["opinions"],
            filter_result.cluster_meta,
            filter_result.docket_meta,
            filter_result.court_name_map,
            config=fixture_config,
        )
        write_manifest(
            fixture_config.manifest_path,
            fixture_config.shard_dir,
            stats,
            fixture_paths,
            filter_result.fed_court_ids,
            len(filter_result.docket_meta),
            len(filter_result.cluster_meta),
            fixture_config.shard_size,
            config=fixture_config,
        )
        assert validate_manifest_shards(read_manifest(fixture_config.manifest_path), fixture_config.shard_dir)
