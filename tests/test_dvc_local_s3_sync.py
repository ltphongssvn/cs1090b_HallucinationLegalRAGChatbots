"""Test local DVC cache and S3 remote are in sync (no missing pushes/pulls)."""

import subprocess

import pytest


def _run_dvc(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [".venv/bin/dvc", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.contract
class TestDVCLocalS3Sync:
    def test_no_pending_pushes(self):
        """`dvc status --cloud` must show no files pending upload."""
        rc, out = _run_dvc(["status", "--cloud"])
        assert rc == 0, f"dvc status --cloud failed: {out}"
        # "Cache and remote 'storage' are in sync." OR "Data and pipelines are up to date."
        ok_markers = ["in sync", "up to date", "Everything is up to date"]
        assert any(m in out for m in ok_markers), f"local cache has unpushed files:\n{out}"

    def test_no_missing_pulls(self):
        """No DVC-tracked file should be missing from local cache."""
        rc, out = _run_dvc(["status"])
        assert rc == 0, f"dvc status failed: {out}"
        # workspace status should be clean OR only show "deleted" for files
        # we intentionally don't materialize. Allow "deleted:" but flag
        # "missing" / "not in cache" errors.
        bad_markers = ["not in cache", "missing from cache"]
        for m in bad_markers:
            assert m not in out, f"local cache has missing files:\n{out}"
