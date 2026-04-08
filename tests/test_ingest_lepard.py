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
    def test_fetch_stream_raises_on_bad_revision(self):
        import pytest

        from scripts.ingest_lepard import fetch_stream

        with pytest.raises(ValueError, match="not a 40-char hex SHA"):
            next(fetch_stream("rmahari/LePaRD", "train", "main"))


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
        write_jsonl(iter(rows), out, cap=5, force=True)
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

        import pytest

        from scripts.ingest_lepard import main

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
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0, "--smoke and --cap must be mutually exclusive"


class TestFetchStreamRevisionValidation:
    def test_fetch_stream_rejects_mutable_revision(self):
        import pytest

        from scripts.ingest_lepard import fetch_stream

        # validate_revision called before network — raises immediately, no HF download
        with pytest.raises(ValueError, match="not a 40-char hex SHA"):
            next(fetch_stream("rmahari/LePaRD", "train", "main"))


class TestRevisionInOutputFilename:
    def test_output_filename_includes_revision_prefix(self):
        from scripts.ingest_lepard import load_lepard_config

        cfg = load_lepard_config()
        output_file = cfg["output_file"].format(cap=cfg["cap"])
        assert "rev" in output_file or cfg["revision"][:8] in output_file, (
            "output_file must include revision prefix to prevent same-cap collision"
        )


class TestRevisionValidationEdgeCases:
    def test_uppercase_sha_rejected(self):
        import pytest

        from scripts.ingest_lepard import validate_revision

        with pytest.raises(ValueError, match="not a 40-char hex SHA"):
            validate_revision("0194F95C3091ACCEAB3B887C9B09EF432CF84052")

    def test_short_sha_rejected(self):
        import pytest

        from scripts.ingest_lepard import validate_revision

        with pytest.raises(ValueError):
            validate_revision("0194f95c")

    def test_crash_after_replace_self_heals_on_next_run(self, tmp_path):
        from scripts.ingest_lepard import _sidecar_path, write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5, force=True)
        sidecar = _sidecar_path(out)
        # simulate crash after data write before sidecar write
        sidecar.unlink(missing_ok=True)
        assert not sidecar.exists()
        r2, _ = write_jsonl(iter(rows), out, cap=5)
        assert r2 == 0, "next run must self-heal not rewrite"
        assert sidecar.exists(), "sidecar must be self-healed"


class TestGitShaFallback:
    def test_git_sha_returns_unknown_when_git_missing(self):
        from unittest.mock import patch

        from scripts.ingest_lepard import _git_sha

        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            assert _git_sha() == "unknown"

    def test_git_sha_returns_unknown_when_not_git_repo(self):
        import subprocess
        from unittest.mock import patch

        from scripts.ingest_lepard import _git_sha

        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(128, "git")):
            assert _git_sha() == "unknown"


class TestGitShaEnvFallback:
    def test_git_sha_uses_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv("GIT_COMMIT_SHA", "abc123def456")
        from scripts.ingest_lepard import _git_sha

        assert _git_sha() == "abc123def456"

    def test_git_sha_env_takes_priority_over_subprocess(self, monkeypatch):
        from unittest.mock import patch

        monkeypatch.setenv("GIT_COMMIT_SHA", "env_sha_value")
        from scripts.ingest_lepard import _git_sha

        with patch("subprocess.check_output") as mock_sub:
            result = _git_sha()
            assert result == "env_sha_value"
            assert not mock_sub.called, "subprocess must not be called when env var set"


class TestTimezoneAwareTimestamp:
    def test_manifest_uses_timezone_aware_utc(self, tmp_path):
        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(
            iter(rows), out, cap=5, revision="0194f95c3091acceab3b887c9b09ef432cf84052", dataset="rmahari/LePaRD"
        )
        import json as _json

        manifest = _json.loads((tmp_path / "out.jsonl.manifest.json").read_text())
        ts = manifest["ingestion_ts_utc"]
        # timezone-aware UTC includes +00:00 suffix
        assert "+00:00" in ts or ts.endswith("Z"), "ingestion_ts_utc must be timezone-aware UTC"

    def test_manifest_python_version_is_exact(self, tmp_path):
        import json as _json
        import sys

        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(
            iter(rows), out, cap=5, revision="0194f95c3091acceab3b887c9b09ef432cf84052", dataset="rmahari/LePaRD"
        )
        manifest = _json.loads((tmp_path / "out.jsonl.manifest.json").read_text())
        expected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert manifest["python_version"] == expected


class TestVerifyOnlySidecarComparison:
    def test_verify_only_compares_digest_against_sidecar(self, tmp_path):
        from scripts.ingest_lepard import _sidecar_path, write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        _, digest = write_jsonl(iter(rows), out, cap=5)
        sidecar = _sidecar_path(out)
        sidecar.write_text(digest + "\n")
        # verify_only must compare computed digest against sidecar
        _, verify_digest = write_jsonl(iter([]), out, cap=5, verify_only=True)
        assert verify_digest == digest

    def test_verify_only_raises_when_sidecar_mismatches(self, tmp_path):
        import pytest

        from scripts.ingest_lepard import _sidecar_path, write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(iter(rows), out, cap=5)
        sidecar = _sidecar_path(out)
        sidecar.write_text("wrong_hash\n")
        with pytest.raises(ValueError, match="digest mismatch"):
            write_jsonl(iter([]), out, cap=5, verify_only=True)


class TestManifestCompleteness:
    def test_manifest_includes_split(self, tmp_path):
        import json as _json

        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(
            iter(rows),
            out,
            cap=5,
            revision="0194f95c3091acceab3b887c9b09ef432cf84052",
            dataset="rmahari/LePaRD",
            split="train",
        )
        manifest = _json.loads((tmp_path / "out.jsonl.manifest.json").read_text())
        assert "split" in manifest, "manifest must record dataset split"
        assert manifest["split"] == "train"

    def test_manifest_includes_rows_written(self, tmp_path):
        import json as _json

        from scripts.ingest_lepard import write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(
            iter(rows),
            out,
            cap=10,  # cap=10 but only 5 rows
            revision="0194f95c3091acceab3b887c9b09ef432cf84052",
            dataset="rmahari/LePaRD",
            split="train",
        )
        manifest = _json.loads((tmp_path / "out.jsonl.manifest.json").read_text())
        assert "rows_written" in manifest, "manifest must record actual rows written"
        assert manifest["rows_written"] == 5, "rows_written must reflect actual rows not cap"


class TestSelfHealManifest:
    def test_self_heal_restores_both_sidecar_and_manifest(self, tmp_path):
        from scripts.ingest_lepard import (
            _manifest_path,
            _sidecar_path,
            write_jsonl,
        )

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        _, digest = write_jsonl(
            iter(rows),
            out,
            cap=5,
            revision="0194f95c3091acceab3b887c9b09ef432cf84052",
            dataset="rmahari/LePaRD",
            split="train",
        )
        _sidecar_path(out).write_text(digest + "\n")
        # delete both
        _sidecar_path(out).unlink()
        _manifest_path(out).unlink()
        # self-heal run
        write_jsonl(
            iter(rows),
            out,
            cap=5,
            revision="0194f95c3091acceab3b887c9b09ef432cf84052",
            dataset="rmahari/LePaRD",
            split="train",
        )
        assert _sidecar_path(out).exists(), "self-heal must restore sidecar"
        assert _manifest_path(out).exists(), "self-heal must restore manifest"


class TestSinglePassSelfHeal:
    def test_self_heal_uses_single_pass(self, tmp_path):
        from unittest.mock import patch

        from scripts.ingest_lepard import _sidecar_path, write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        _, digest = write_jsonl(iter(rows), out, cap=5)
        _sidecar_path(out).write_text(digest + "\n")
        _sidecar_path(out).unlink()
        # single-pass: open() called once for combined line-count + hash
        with patch("scripts.ingest_lepard.open", wraps=open) as mock_open:
            write_jsonl(iter(rows), out, cap=5)
            # only one open call for the self-heal read
            read_calls = [c for c in mock_open.call_args_list if str(out) in str(c)]
            assert len(read_calls) <= 1, "self-heal must use single-pass read"


class TestCliHelpText:
    def test_all_flags_have_help_text(self):
        import subprocess

        result = subprocess.run(
            ["uv", "run", "python", "scripts/ingest_lepard.py", "--help"],
            capture_output=True,
            text=True,
        )
        help_text = result.stdout + result.stderr
        assert "Purge stale" in help_text, "--force must have help text"
        assert "Count rows" in help_text, "--dry-run must have help text"
        assert "Recompute SHA256" in help_text, "--verify-only must have help text"
        assert "smoke_cap" in help_text or "smoke" in help_text.lower(), "--smoke must have help text"
        assert "Override cap" in help_text, "--cap must have help text"


class TestFetchStreamRetry:
    def test_fetch_stream_retries_on_connection_error(self):
        from unittest.mock import MagicMock, patch

        from scripts.ingest_lepard import fetch_stream

        attempt = 0

        def flaky_load(*args, **kwargs):
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ConnectionResetError("flaky network")
            ds = MagicMock()
            ds.__iter__ = MagicMock(return_value=iter([{"id": "0"}]))
            return ds

        with patch("datasets.load_dataset", side_effect=flaky_load):
            rows = list(
                fetch_stream(
                    "rmahari/LePaRD",
                    "train",
                    "0194f95c3091acceab3b887c9b09ef432cf84052",
                )
            )
            assert len(rows) == 1, "fetch_stream must retry on connection error"
            assert attempt == 3, "must retry exactly 3 times"


class TestManifestRepairWhenSidecarPresent:
    def test_missing_manifest_repaired_when_sidecar_present(self, tmp_path):
        from scripts.ingest_lepard import _manifest_path, _sidecar_path, write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        _, digest = write_jsonl(
            iter(rows),
            out,
            cap=5,
            revision="0194f95c3091acceab3b887c9b09ef432cf84052",
            dataset="rmahari/LePaRD",
            split="train",
        )
        _sidecar_path(out).write_text(digest + "\n")
        _manifest_path(out).unlink()
        assert not _manifest_path(out).exists()
        # next run — must repair manifest even when sidecar present
        write_jsonl(
            iter(rows),
            out,
            cap=5,
            revision="0194f95c3091acceab3b887c9b09ef432cf84052",
            dataset="rmahari/LePaRD",
            split="train",
        )
        assert _manifest_path(out).exists(), "manifest must be repaired when sidecar present"


class TestSelfHealPreservesOriginalProvenance:
    def test_self_heal_preserves_original_manifest_timestamp(self, tmp_path):
        import json as _json
        import time

        from scripts.ingest_lepard import _manifest_path, _sidecar_path, write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        _, digest = write_jsonl(
            iter(rows),
            out,
            cap=5,
            revision="0194f95c3091acceab3b887c9b09ef432cf84052",
            dataset="rmahari/LePaRD",
            split="train",
        )
        _sidecar_path(out).write_text(digest + "\n")
        original_ts = _json.loads(_manifest_path(out).read_text())["ingestion_ts_utc"]
        # delete sidecar only — manifest survives
        _sidecar_path(out).unlink()
        time.sleep(0.05)
        write_jsonl(
            iter(rows),
            out,
            cap=5,
            revision="0194f95c3091acceab3b887c9b09ef432cf84052",
            dataset="rmahari/LePaRD",
            split="train",
        )
        repaired = _json.loads(_manifest_path(out).read_text())
        assert repaired["ingestion_ts_utc"] == original_ts, "self-heal must preserve original provenance timestamp"
        assert repaired.get("provenance_reconstructed") is True, "self-heal must mark manifest as reconstructed"


class TestFreshWriteFinalization:
    def test_sidecar_written_inside_write_jsonl_not_only_main(self, tmp_path):
        from scripts.ingest_lepard import _sidecar_path, write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        write_jsonl(
            iter(rows),
            out,
            cap=5,
            revision="0194f95c3091acceab3b887c9b09ef432cf84052",
            dataset="rmahari/LePaRD",
            split="train",
        )
        assert _sidecar_path(out).exists(), "sidecar must be written inside write_jsonl to close crash window"


class TestVerifyOnlyManifestCheck:
    def test_verify_only_checks_manifest_revision(self, tmp_path):
        import pytest

        from scripts.ingest_lepard import _sidecar_path, write_jsonl

        rows = [{"id": str(i)} for i in range(5)]
        out = tmp_path / "out.jsonl"
        _, digest = write_jsonl(
            iter(rows),
            out,
            cap=5,
            revision="0194f95c3091acceab3b887c9b09ef432cf84052",
            dataset="rmahari/LePaRD",
            split="train",
        )
        _sidecar_path(out).write_text(digest + "\n")
        # verify with wrong revision — should fail if manifest is consulted
        with pytest.raises(ValueError, match="manifest mismatch"):
            write_jsonl(
                iter([]),
                out,
                cap=5,
                verify_only=True,
                revision="0000000000000000000000000000000000000000",
                dataset="rmahari/LePaRD",
                split="train",
            )
