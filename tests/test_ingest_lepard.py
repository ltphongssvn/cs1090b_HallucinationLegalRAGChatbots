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
        written, _ = write_jsonl(iter(rows), out, cap=100)
        assert written == 100
        assert out.exists()

    def test_respects_cap(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(1000)]
        out = tmp_path / "out.jsonl"
        written, _ = write_jsonl(iter(rows), out, cap=50)
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
        assert out.stat().st_mtime == mtime1


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


class TestChunkSizeConstant:
    def test_chunk_size_importable(self):
        from scripts.ingest_lepard import CHUNK_SIZE

        assert CHUNK_SIZE == 64 * 1024


class TestSha256Idempotency:
    def test_skips_when_sha256_sidecar_matches(self, tmp_path):
        from scripts.ingest_lepard import compute_sha256, write_jsonl

        rows = [{"id": str(i)} for i in range(10)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=10)
        compute_sha256(out, write_sidecar=True)
        mtime1 = out.stat().st_mtime
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


class TestFetchStreamExceptions:
    def test_fetch_stream_raises_on_bad_dataset(self):
        import pytest

        from scripts.ingest_lepard import fetch_stream

        with pytest.raises(Exception):
            list(fetch_stream("nonexistent/dataset_xyz_123", "train", "main"))


class TestZeroCapHandling:
    def test_cap_zero_raises(self, tmp_path):
        import pytest

        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(10)]
        out = tmp_path / "out.jsonl"
        with pytest.raises(ValueError, match="cap must be positive"):
            write_jsonl(iter(rows), out, cap=0)


class TestAtomicWrite:
    def test_no_tmp_file_after_write(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5)
        assert not out.with_suffix(".jsonl.tmp").exists()


class TestShortStreamIdempotency:
    def test_short_stream_stabilizes_on_second_run(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=1000)
        r2, _ = write_jsonl(iter(rows), out, cap=1000)
        assert r2 == 0, "short stream must not be rewritten on second run"


class TestRevisionValidation:
    def test_non_sha_revision_rejected(self):
        import pytest

        from scripts.ingest_lepard import validate_revision

        with pytest.raises(ValueError):
            validate_revision("main")

    def test_valid_sha_revision_accepted(self):
        from scripts.ingest_lepard import validate_revision

        validate_revision("0194f95c3091acceab3b887c9b09ef432cf84052")


class TestAtomicWriteSafety:
    def test_tmp_file_not_present_after_successful_write(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5)
        assert not (tmp_path / "out.jsonl.tmp").exists()

    def test_output_file_present_after_write(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5)
        assert out.exists()


class TestCapValidation:
    def test_negative_cap_raises(self, tmp_path):
        import pytest

        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": "0"}]
        out = tmp_path / "out.jsonl"
        with pytest.raises(ValueError, match="cap must be positive"):
            write_jsonl(iter(rows), out, cap=-1)

    def test_zero_cap_raises_via_cap_validation(self, tmp_path):
        import pytest

        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": "0"}]
        out = tmp_path / "out.jsonl"
        with pytest.raises(ValueError, match="cap must be positive"):
            write_jsonl(iter(rows), out, cap=0)


class TestHashWhileWriting:
    def test_write_jsonl_returns_sha256(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        result = write_jsonl(iter(rows), out, cap=5)
        assert isinstance(result, tuple), "write_jsonl must return (rows_written, sha256)"
        rows_written, sha256 = result
        assert rows_written == 5
        assert len(sha256) == 64


class TestSha256SidecarIdempotency:
    def test_skips_when_sidecar_exists(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(10)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=10)
        sidecar = tmp_path / "out.jsonl.sha256"
        sidecar.write_text("dummy_hash\n")
        r2, _ = write_jsonl(iter(rows), out, cap=10)
        assert r2 == 0

    def test_no_line_count_scan_when_sidecar_present(self, tmp_path):
        from unittest.mock import patch

        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5)
        sidecar = tmp_path / "out.jsonl.sha256"
        sidecar.write_text("any_hash\n")
        with patch("builtins.sum") as mock_sum:
            write_jsonl(iter(rows), out, cap=5)
            assert not mock_sum.called, "line-count scan must not run when sidecar exists"


class TestLogException:
    def test_main_uses_log_exception_not_log_error(self):
        from pathlib import Path

        src = Path("scripts/ingest_lepard.py").read_text()
        assert "log.exception" in src, "main() must use log.exception to preserve traceback"
        assert 'log.error("Ingestion failed' not in src, "log.error hides traceback"


class TestSidecarSuffixConstant:
    def test_sidecar_suffix_constant_importable(self):
        from scripts.ingest_lepard import _SIDECAR_SUFFIX

        assert _SIDECAR_SUFFIX == ".sha256"


class TestForceFlag:
    def test_write_jsonl_force_rewrites_existing(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5)
        sidecar = tmp_path / "out.jsonl.sha256"
        sidecar.write_text("stale_hash\n")
        r2, _ = write_jsonl(iter(rows), out, cap=5, force=True)
        assert r2 == 5, "--force must rewrite even when sidecar exists"
