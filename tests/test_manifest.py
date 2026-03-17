import pytest

pytestmark = pytest.mark.unit

# tests/test_manifest.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_manifest.py
# TDD RED: Manifest read/write/validate contract.


from src.manifest import file_checksum, read_manifest, validate_manifest_shards, write_manifest


class TestFileChecksum:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        assert file_checksum(f) == file_checksum(f)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert file_checksum(f1) != file_checksum(f2)


class TestReadManifest:
    def test_missing_returns_empty(self, tmp_path):
        assert read_manifest(tmp_path / "nope.json") == {}

    def test_reads_valid_json(self, tmp_path):
        f = tmp_path / "manifest.json"
        f.write_text('{"num_cases": 42}')
        assert read_manifest(f) == {"num_cases": 42}


class TestValidateManifestShards:
    def test_valid_shards_pass(self, tmp_path):
        shard = tmp_path / "shard_0000.jsonl"
        shard.write_text('{"id": 1}\n')
        manifest = {"checksum": {"shard_0000.jsonl": file_checksum(shard)}}
        assert validate_manifest_shards(manifest, tmp_path) is True

    def test_missing_shard_fails(self, tmp_path):
        manifest = {"checksum": {"shard_0000.jsonl": "abc123"}}
        assert validate_manifest_shards(manifest, tmp_path) is False

    def test_corrupted_shard_fails(self, tmp_path):
        shard = tmp_path / "shard_0000.jsonl"
        shard.write_text('{"id": 1}\n')
        manifest = {"checksum": {"shard_0000.jsonl": "wrong_hash"}}
        assert validate_manifest_shards(manifest, tmp_path) is False

    def test_no_checksum_key_fails(self, tmp_path):
        assert validate_manifest_shards({}, tmp_path) is False


class TestRunMetadata:
    def test_manifest_contains_run_metadata(self, tmp_path):
        from src.config import PipelineConfig

        config = PipelineConfig(shard_dir=tmp_path)
        shard = tmp_path / "shard_0000.jsonl"
        shard.write_text('{"id": 1}\n')
        stats = {
            "extracted_total": 1,
            "num_shards": 1,
            "text_source_counts": {},
            "scanned": 1,
            "skipped_empty": 0,
            "skipped_parse": 0,
        }
        result = write_manifest(
            tmp_path / "manifest.json",
            tmp_path,
            stats,
            {"courts": tmp_path / "c.csv"},
            set(),
            0,
            0,
            100,
            config=config,
        )
        assert "run_metadata" in result
        assert "timestamp" in result["run_metadata"]
        assert "git_revision" in result["run_metadata"]
        assert "python_version" in result["run_metadata"]
        assert "config" in result["run_metadata"]

    def test_manifest_contains_source_checksums(self, tmp_path):
        from src.config import PipelineConfig

        config = PipelineConfig(shard_dir=tmp_path)
        shard = tmp_path / "shard_0000.jsonl"
        shard.write_text('{"id": 1}\n')
        source = tmp_path / "courts.csv"
        source.write_text("id,name\nca1,First\n")
        stats = {
            "extracted_total": 1,
            "num_shards": 1,
            "text_source_counts": {},
            "scanned": 1,
            "skipped_empty": 0,
            "skipped_parse": 0,
        }
        result = write_manifest(
            tmp_path / "manifest.json",
            tmp_path,
            stats,
            {"courts": source},
            set(),
            0,
            0,
            100,
            config=config,
        )
        assert "source_checksums" in result
        assert "courts" in result["source_checksums"]

    def test_manifest_version_is_2(self, tmp_path):
        from src.config import PipelineConfig

        config = PipelineConfig(shard_dir=tmp_path)
        shard = tmp_path / "shard_0000.jsonl"
        shard.write_text('{"id": 1}\n')
        stats = {
            "extracted_total": 1,
            "num_shards": 1,
            "text_source_counts": {},
            "scanned": 1,
            "skipped_empty": 0,
            "skipped_parse": 0,
        }
        result = write_manifest(
            tmp_path / "manifest.json",
            tmp_path,
            stats,
            {},
            set(),
            0,
            0,
            100,
            config=config,
        )
        assert result["version"] == 2
