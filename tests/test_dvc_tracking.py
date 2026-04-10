# tests/test_dvc_tracking.py
# TDD RED: DVC tracking helpers for shard-directory artifact versioning.
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from src.dvc_tracking import (
    DVCTrackingError,
    add_artifact,
    is_dvc_repo,
    is_tracked,
    push_artifact,
    track_shard_directory,
)


class TestIsDvcRepo:
    def test_returns_true_when_dvc_dir_exists(self, tmp_path):
        """Detects a DVC-initialised repo by .dvc/ presence."""
        (tmp_path / ".dvc").mkdir()
        assert is_dvc_repo(tmp_path) is True

    def test_returns_false_when_dvc_dir_missing(self, tmp_path):
        """Returns False on a plain directory."""
        assert is_dvc_repo(tmp_path) is False


class TestIsTracked:
    def test_returns_true_when_pointer_exists(self, tmp_path):
        """Detects an existing <name>.dvc pointer file."""
        target = tmp_path / "data" / "shards"
        target.mkdir(parents=True)
        (tmp_path / "shards.dvc").write_text("outs:\n- md5: abc\n")
        assert is_tracked(target, repo_root=tmp_path) is True

    def test_returns_false_when_pointer_missing(self, tmp_path):
        """Returns False when no pointer file exists."""
        target = tmp_path / "data" / "shards"
        target.mkdir(parents=True)
        assert is_tracked(target, repo_root=tmp_path) is False


class TestAddArtifact:
    @patch("src.dvc_tracking.subprocess.run")
    def test_calls_dvc_add_with_path(self, mock_run, tmp_path):
        """Invokes `dvc add <path>` as a subprocess."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        target = tmp_path / "shards"
        target.mkdir()
        add_artifact(target, repo_root=tmp_path)
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "dvc"
        assert cmd[1] == "add"
        assert str(target) in cmd

    @patch("src.dvc_tracking.subprocess.run")
    def test_raises_on_dvc_failure(self, mock_run, tmp_path):
        """Raises DVCTrackingError when dvc add exits non-zero."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
        target = tmp_path / "shards"
        target.mkdir()
        with pytest.raises(DVCTrackingError, match="dvc add failed"):
            add_artifact(target, repo_root=tmp_path)


class TestPushArtifact:
    @patch("src.dvc_tracking.subprocess.run")
    def test_calls_dvc_push(self, mock_run, tmp_path):
        """Invokes `dvc push` as a subprocess."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        push_artifact(repo_root=tmp_path)
        cmd = mock_run.call_args[0][0]
        assert cmd == ["dvc", "push"]

    @patch("src.dvc_tracking.subprocess.run")
    def test_raises_on_push_failure(self, mock_run, tmp_path):
        """Raises DVCTrackingError when dvc push exits non-zero."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="net err")
        with pytest.raises(DVCTrackingError, match="dvc push failed"):
            push_artifact(repo_root=tmp_path)


class TestTrackShardDirectory:
    @patch("src.dvc_tracking.subprocess.run")
    def test_skips_if_already_tracked(self, mock_run, tmp_path):
        """Idempotent: does nothing if pointer already exists."""
        (tmp_path / ".dvc").mkdir()
        target = tmp_path / "shards"
        target.mkdir()
        (tmp_path / "shards.dvc").write_text("outs:\n- md5: abc\n")
        track_shard_directory(target, repo_root=tmp_path, push=False)
        mock_run.assert_not_called()

    @patch("src.dvc_tracking.subprocess.run")
    def test_calls_add_then_push_when_push_true(self, mock_run, tmp_path):
        """Calls dvc add then dvc push when push=True."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        (tmp_path / ".dvc").mkdir()
        target = tmp_path / "shards"
        target.mkdir()
        track_shard_directory(target, repo_root=tmp_path, push=True)
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0][0][0][:2] == ["dvc", "add"]
        assert mock_run.call_args_list[1][0][0] == ["dvc", "push"]

    @patch("src.dvc_tracking.subprocess.run")
    def test_skips_push_when_push_false(self, mock_run, tmp_path):
        """Does not push when push=False."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        (tmp_path / ".dvc").mkdir()
        target = tmp_path / "shards"
        target.mkdir()
        track_shard_directory(target, repo_root=tmp_path, push=False)
        assert mock_run.call_count == 1
        assert mock_run.call_args_list[0][0][0][:2] == ["dvc", "add"]

    def test_raises_when_not_dvc_repo(self, tmp_path):
        """Raises DVCTrackingError when .dvc/ is missing."""
        target = tmp_path / "shards"
        target.mkdir()
        with pytest.raises(DVCTrackingError, match="not a DVC repo"):
            track_shard_directory(target, repo_root=tmp_path, push=False)
