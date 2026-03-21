# tests/test_validation_coverage.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_validation_coverage.py
import json

import pytest

pytestmark = pytest.mark.unit

from src.config import PipelineConfig
from src.exceptions import ValidationError
from src.validation import (
    _select_shards,
    check_checksums,
    check_manifest_exists,
    check_provenance_fields,
    check_raw_and_normalized_text,
    check_schema_consistent,
    check_shard_dir_exists,
    check_shards_exist,
    check_text_present,
    check_text_source_tracked,
    check_text_substantive,
    check_total_count,
    check_valid_json,
    run_contract_tests,
)


def _make_shard(shard_dir, records, name="shard_0000.jsonl"):
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _valid_record(**overrides):
    base = {
        "id": 1,
        "cluster_id": 100,
        "docket_id": 1,
        "court_id": "ca1",
        "court_name": "First",
        "case_name": "T",
        "date_filed": "2024",
        "precedential_status": "Published",
        "opinion_type": "lead",
        "extracted_by_ocr": "False",
        "raw_text": "x" * 200,
        "text": "x" * 200,
        "text_length": 200,
        "text_source": "plain_text",
        "cleaning_flags": [],
        "source": "courtlistener_bulk",
        "token_count": 50,
        "paragraph_count": 1,
        "citation_count": 0,
        "text_hash": "abc",
        "citation_density": 0.0,
        "is_precedential": True,
        "text_entropy": 4.0,
    }
    base.update(overrides)
    return base


class TestContractTestsPassing:
    def test_shard_dir_exists_passes(self, tmp_path):
        config = PipelineConfig(shard_dir=tmp_path)
        check_shard_dir_exists(config)

    def test_shard_dir_missing_raises(self, tmp_path):
        config = PipelineConfig(shard_dir=tmp_path / "nonexistent")
        with pytest.raises(ValidationError):
            check_shard_dir_exists(config)

    def test_manifest_exists_passes(self, tmp_path):
        (tmp_path / "manifest.json").write_text("{}")
        config = PipelineConfig(shard_dir=tmp_path)
        check_manifest_exists(config)

    def test_manifest_missing_raises(self, tmp_path):
        config = PipelineConfig(shard_dir=tmp_path)
        with pytest.raises(ValidationError):
            check_manifest_exists(config)

    def test_shards_exist_passes(self, tmp_path):
        _make_shard(tmp_path, [_valid_record()])
        config = PipelineConfig(shard_dir=tmp_path)
        check_shards_exist(config)

    def test_no_shards_raises(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        config = PipelineConfig(shard_dir=tmp_path)
        with pytest.raises(ValidationError):
            check_shards_exist(config)

    def test_total_count_passes(self, tmp_path):
        records = [_valid_record(id=i) for i in range(20)]
        _make_shard(tmp_path, records)
        config = PipelineConfig(shard_dir=tmp_path, min_expected_total=10)
        check_total_count(config)

    def test_total_count_too_few_raises(self, tmp_path):
        _make_shard(tmp_path, [_valid_record()])
        config = PipelineConfig(shard_dir=tmp_path, min_expected_total=100)
        with pytest.raises(ValidationError):
            check_total_count(config)

    def test_valid_json_passes(self, tmp_path):
        _make_shard(tmp_path, [_valid_record()])
        config = PipelineConfig(shard_dir=tmp_path)
        check_valid_json(config)

    def test_text_present_passes(self, tmp_path):
        _make_shard(tmp_path, [_valid_record()])
        config = PipelineConfig(shard_dir=tmp_path)
        check_text_present(config)

    def test_text_present_empty_raises(self, tmp_path):
        _make_shard(tmp_path, [_valid_record(text="")])
        config = PipelineConfig(shard_dir=tmp_path)
        with pytest.raises(ValidationError):
            check_text_present(config)

    def test_text_substantive_passes(self, tmp_path):
        _make_shard(tmp_path, [_valid_record()])
        config = PipelineConfig(shard_dir=tmp_path)
        check_text_substantive(config)

    def test_provenance_fields_passes(self, tmp_path):
        _make_shard(tmp_path, [_valid_record()])
        config = PipelineConfig(shard_dir=tmp_path)
        check_provenance_fields(config)

    def test_provenance_missing_raises(self, tmp_path):
        rec = _valid_record()
        del rec["court_id"]
        _make_shard(tmp_path, [rec])
        config = PipelineConfig(shard_dir=tmp_path)
        with pytest.raises(ValidationError):
            check_provenance_fields(config)

    def test_raw_and_normalized_passes(self, tmp_path):
        _make_shard(tmp_path, [_valid_record()])
        config = PipelineConfig(shard_dir=tmp_path)
        check_raw_and_normalized_text(config)

    def test_text_source_tracked_passes(self, tmp_path):
        _make_shard(tmp_path, [_valid_record()])
        config = PipelineConfig(shard_dir=tmp_path)
        check_text_source_tracked(config)

    def test_bad_text_source_raises(self, tmp_path):
        _make_shard(tmp_path, [_valid_record(text_source="unknown_source")])
        config = PipelineConfig(shard_dir=tmp_path)
        with pytest.raises(ValidationError):
            check_text_source_tracked(config)

    def test_schema_consistent_passes(self, tmp_path):
        _make_shard(tmp_path, [_valid_record(), _valid_record(id=2)])
        config = PipelineConfig(shard_dir=tmp_path)
        check_schema_consistent(config)

    def test_schema_inconsistent_raises(self, tmp_path):
        rec1 = _valid_record()
        rec2 = _valid_record(id=2)
        rec2["extra_field"] = "oops"
        _make_shard(tmp_path, [rec1, rec2])
        config = PipelineConfig(shard_dir=tmp_path)
        with pytest.raises(ValidationError):
            check_schema_consistent(config)

    def test_checksums_pass(self, tmp_path):
        from src.manifest import file_checksum

        _make_shard(tmp_path, [_valid_record()])
        config = PipelineConfig(shard_dir=tmp_path)
        chk = file_checksum(tmp_path / "shard_0000.jsonl")
        check_checksums(config, {"checksum": {"shard_0000.jsonl": chk}})

    def test_checksums_mismatch_raises(self, tmp_path):
        _make_shard(tmp_path, [_valid_record()])
        config = PipelineConfig(shard_dir=tmp_path)
        with pytest.raises(ValidationError):
            check_checksums(config, {"checksum": {"shard_0000.jsonl": "badhash"}})


class TestSelectShards:
    def test_all_strategy(self, tmp_path):
        for i in range(5):
            _make_shard(tmp_path, [_valid_record()], f"shard_{i:04d}.jsonl")
        config = PipelineConfig(shard_dir=tmp_path)
        assert len(_select_shards(config, "all")) == 5

    def test_head_strategy(self, tmp_path):
        for i in range(5):
            _make_shard(tmp_path, [_valid_record()], f"shard_{i:04d}.jsonl")
        config = PipelineConfig(shard_dir=tmp_path)
        assert len(_select_shards(config, "head")) == 3

    def test_sample_strategy(self, tmp_path):
        for i in range(20):
            _make_shard(tmp_path, [_valid_record()], f"shard_{i:04d}.jsonl")
        config = PipelineConfig(shard_dir=tmp_path)
        selected = _select_shards(config, "sample")
        assert 3 <= len(selected) <= 20


class TestRunContractTests:
    def test_all_pass(self, tmp_path):
        import logging

        records = [_valid_record(id=i, court_id=f"ca{(i % 6) + 1}") for i in range(20)]
        _make_shard(tmp_path, records)
        (tmp_path / "manifest.json").write_text("{}")
        config = PipelineConfig(shard_dir=tmp_path, min_expected_total=5)
        logger = logging.getLogger("test_contract")
        assert run_contract_tests(config=config, logger=logger) is True
