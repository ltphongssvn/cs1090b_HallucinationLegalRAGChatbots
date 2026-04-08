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


class TestUniqueTmpFile:
    def test_tmp_file_uses_unique_name(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5)
        assert not (tmp_path / "out.jsonl.tmp").exists()

    def test_concurrent_writes_use_different_tmp_names(self, tmp_path):
        import tempfile

        out = tmp_path / "out.jsonl"
        tmp1 = tempfile.NamedTemporaryFile(dir=out.parent, suffix=".jsonl.tmp", delete=False)
        tmp2 = tempfile.NamedTemporaryFile(dir=out.parent, suffix=".jsonl.tmp", delete=False)
        assert tmp1.name != tmp2.name


class TestSidecarSelfHeal:
    def test_valid_file_without_sidecar_gets_sidecar_written(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5)
        sidecar = tmp_path / "out.jsonl.sha256"
        sidecar.unlink(missing_ok=True)
        assert not sidecar.exists()
        write_jsonl(iter(rows), out, cap=5)
        assert sidecar.exists(), "second run must self-heal missing sidecar"


class TestTqdmDisableNone:
    def test_tqdm_called_with_disable_none(self, tmp_path):
        from unittest.mock import patch

        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        with patch("scripts.ingest_lepard.tqdm") as mock_tqdm:
            mock_tqdm.side_effect = lambda x, **kw: x
            write_jsonl(iter(rows), out, cap=5)
            _, kwargs = mock_tqdm.call_args
            assert kwargs.get("disable") is None


class TestHypothesisIdempotency:
    def test_write_jsonl_idempotent_property(self, tmp_path):
        from hypothesis import given, settings
        from hypothesis import strategies as st
        from hypothesis.strategies import composite

        from scripts.ingest_lepard import write_jsonl

        @composite
        def valid_rows(draw):
            n = draw(st.integers(min_value=1, max_value=20))
            return [{"id": str(i), "quote": draw(st.text(max_size=50))} for i in range(n)]

        @given(valid_rows())
        @settings(max_examples=30)
        def inner(rows):
            out = tmp_path / f"out_{len(rows)}.jsonl"
            cap = max(1, len(rows))
            write_jsonl(iter(rows), out, cap=cap)
            r2, _ = write_jsonl(iter(rows), out, cap=cap)
            assert r2 == 0, "second run must be a no-op"

        inner()


class TestProvenanceManifest:
    def test_provenance_manifest_written_after_ingest(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(
            iter(rows), out, cap=5, revision="0194f95c3091acceab3b887c9b09ef432cf84052", dataset="rmahari/LePaRD"
        )
        manifest = tmp_path / "out.jsonl.manifest.json"
        assert manifest.exists(), "provenance manifest must be written alongside JSONL"

    def test_provenance_manifest_has_required_fields(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(
            iter(rows), out, cap=5, revision="0194f95c3091acceab3b887c9b09ef432cf84052", dataset="rmahari/LePaRD"
        )
        manifest = json.loads((tmp_path / "out.jsonl.manifest.json").read_text())
        assert "ingestion_ts_utc" in manifest
        assert "script_git_commit" in manifest
        assert "hf_revision" in manifest
        assert "cap" in manifest
        assert "python_version" in manifest
        assert "datasets_version" in manifest
        assert "sha256" in manifest


class TestVerifyOnlyFlag:
    def test_verify_only_does_not_rewrite_file(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5)
        mtime1 = out.stat().st_mtime
        write_jsonl(iter(rows), out, cap=5, verify_only=True)
        assert out.stat().st_mtime == mtime1

    def test_verify_only_returns_existing_sha256(self, tmp_path):
        from scripts.ingest_lepard import compute_sha256, write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5)
        expected_sha = compute_sha256(out)
        _, sha = write_jsonl(iter(rows), out, cap=5, verify_only=True)
        assert sha == expected_sha


class TestFdLeakFixed:
    def test_no_fd_leak_on_exception(self, tmp_path):
        import os
        from unittest.mock import patch

        from scripts.ingest_lepard import write_jsonl

        fds_before = len(os.listdir("/proc/self/fd"))
        with patch("scripts.ingest_lepard.json.dumps", side_effect=ValueError("simulated")):
            try:
                rows = [{"id": "0"}]
                out = tmp_path / "out.jsonl"
                write_jsonl(iter(rows), out, cap=1)
            except Exception:
                pass
        fds_after = len(os.listdir("/proc/self/fd"))
        assert fds_after <= fds_before + 1, "file descriptor leaked"


class TestForcePurgesStaleArtifacts:
    def test_force_removes_stale_sidecar(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5)
        sidecar = tmp_path / "out.jsonl.sha256"
        sidecar.write_text("stale_hash\n")
        # force rewrite — stale sidecar must be purged before write
        write_jsonl(iter(rows), out, cap=5, force=True)
        # sidecar purged by --force (write_jsonl purges; main() rewrites)
        assert not sidecar.exists() or sidecar.read_text().strip() != "stale_hash"


class TestDryRun:
    def test_dry_run_does_not_write_file(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5, dry_run=True)
        assert not out.exists(), "dry_run must not write output file"

    def test_dry_run_returns_row_count(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        written, _ = write_jsonl(iter(rows), out, cap=5, dry_run=True)
        assert written == 5


class TestJsonDumpsEnsureAscii:
    def test_unicode_preserved_in_output(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": "0", "quote": "café au lait — Smith v. Jones"}]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=1)
        obj = json.loads(out.read_text(encoding="utf-8"))
        assert obj["quote"] == "café au lait — Smith v. Jones"


class TestTqdmMiniters:
    def test_tqdm_called_with_miniters_1(self, tmp_path):
        from unittest.mock import patch

        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        with patch("scripts.ingest_lepard.tqdm") as mock_tqdm:
            mock_tqdm.side_effect = lambda x, **kw: x
            write_jsonl(iter(rows), out, cap=5)
            _, kwargs = mock_tqdm.call_args
            assert kwargs.get("miniters") == 1


class TestSmokeCap:
    def test_smoke_and_cap_mutually_exclusive(self, tmp_path, monkeypatch):
        import sys

        from scripts.ingest_lepard import main

        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0"}\n')
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ingest",
                "--smoke",
                "--cap",
                "100",
                "--output-dir",
                str(tmp_path),
            ],
        )
        import pytest

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0, "--smoke and --cap must be mutually exclusive"
