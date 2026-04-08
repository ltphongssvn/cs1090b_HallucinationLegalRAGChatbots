"""
Contract tests for scripts/ingest_lepard.py
RED phase — all tests must FAIL before implementation exists.
"""

from __future__ import annotations

import json


class TestIngestLepardImports:
    def test_fetch_stream_importable(self):
        from scripts.ingest_lepard import fetch_stream

        assert callable(fetch_stream)

    def test_write_jsonl_importable(self):
        from scripts.ingest_lepard import write_jsonl

        assert callable(write_jsonl)

    def test_compute_sha256_importable(self):
        from scripts.ingest_lepard import compute_sha256

        assert callable(compute_sha256)

    def test_load_lepard_config_importable(self):
        from scripts.ingest_lepard import load_lepard_config

        assert callable(load_lepard_config)


class TestWriteJsonl:
    def test_writes_correct_row_count(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i), "text": f"case {i}"} for i in range(100)]
        out = tmp_path / "out.jsonl"
        written = write_jsonl(iter(rows), out, cap=100)
        assert written == 100
        assert out.exists()

    def test_respects_cap(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(1000)]
        out = tmp_path / "out.jsonl"
        written = write_jsonl(iter(rows), out, cap=50)
        assert written == 50
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 50

    def test_output_is_valid_jsonl(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": "0", "quote": "Smith v. Jones"}]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=1)
        obj = json.loads(out.read_text().strip())
        assert obj["quote"] == "Smith v. Jones"

    def test_idempotent_skip_if_exists(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(10)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=10)
        mtime1 = out.stat().st_mtime
        write_jsonl(iter(rows), out, cap=10)
        assert out.stat().st_mtime == mtime1  # not rewritten


class TestComputeSha256:
    def test_sha256_deterministic(self, tmp_path):
        from scripts.ingest_lepard import compute_sha256

        f = tmp_path / "test.jsonl"
        f.write_text('{"id": "0"}\n')
        h1 = compute_sha256(f)
        h2 = compute_sha256(f)
        assert h1 == h2
        assert len(h1) == 64

    def test_sha256_file_written_alongside(self, tmp_path):
        from scripts.ingest_lepard import compute_sha256

        f = tmp_path / "test.jsonl"
        f.write_text('{"id": "0"}\n')
        compute_sha256(f, write_sidecar=True)
        assert (tmp_path / "test.jsonl.sha256").exists()


class TestLoadLepardConfig:
    def test_loads_yaml_config(self):
        from scripts.ingest_lepard import load_lepard_config

        cfg = load_lepard_config()
        assert cfg["dataset"] == "rmahari/LePaRD"
        assert cfg["revision"] == "0194f95c3091acceab3b887c9b09ef432cf84052"
        assert cfg["cap"] == 500000
        assert cfg["smoke_cap"] == 1000

    def test_config_has_output_dir(self):
        from scripts.ingest_lepard import load_lepard_config

        cfg = load_lepard_config()
        assert "output_dir" in cfg


# ---------------------------------------------------------------------------
# RED: CHUNK_SIZE constant, SHA256 idempotency, tqdm progress, specific exceptions
# ---------------------------------------------------------------------------


class TestChunkSizeConstant:
    def test_chunk_size_constant_importable(self):
        from scripts.ingest_lepard import CHUNK_SIZE

        assert CHUNK_SIZE == 64 * 1024


class TestSha256Idempotency:
    def test_idempotent_skips_when_sha256_sidecar_matches(self, tmp_path):
        from scripts.ingest_lepard import compute_sha256, write_jsonl

        rows = [{"id": str(i)} for i in range(10)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=10)
        compute_sha256(out, write_sidecar=True)
        mtime1 = out.stat().st_mtime
        # second call must skip based on SHA256 sidecar
        write_jsonl(iter(rows), out, cap=10)
        assert out.stat().st_mtime == mtime1


class TestTqdmProgress:
    def test_write_jsonl_uses_tqdm(self, tmp_path):
        from unittest.mock import patch

        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(10)]
        out = tmp_path / "out.jsonl"
        with patch("scripts.ingest_lepard.tqdm") as mock_tqdm:
            mock_tqdm.side_effect = lambda x, **kw: x
            write_jsonl(iter(rows), out, cap=10)
            assert mock_tqdm.called


class TestSpecificExceptions:
    def test_fetch_stream_raises_on_bad_dataset(self):
        import pytest

        from scripts.ingest_lepard import fetch_stream

        with pytest.raises((Exception,)):
            list(fetch_stream("nonexistent/dataset_xyz_123", "train", "main"))
