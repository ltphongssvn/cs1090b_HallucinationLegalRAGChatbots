# src/dvc_tracking.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/dvc_tracking.py
"""DVC artifact tracking for shard directories.

Thin wrapper around the ``dvc`` CLI that lets the data pipeline version
its output (CourtListener shards, LePaRD JSONL) without the notebook
having to know anything about DVC internals.

Three operations:

* :func:`is_dvc_repo` — does this directory have ``.dvc/``?
* :func:`is_tracked` — does ``<name>.dvc`` already exist for the target?
* :func:`add_artifact` / :func:`push_artifact` — single-shot wrappers
  around ``dvc add`` / ``dvc push``.

The orchestrator :func:`track_shard_directory` composes these into the
idempotent "version this directory and push it to the remote" call that
the notebook uses.

Design notes
------------
* **No business logic**: every interaction with DVC goes through the
  CLI, never via the Python API. This keeps the dependency surface tiny
  and lets us test the wrapper with ``patch("subprocess.run")``.
* **Idempotent**: re-running ``track_shard_directory`` on an
  already-tracked directory is a no-op.
* **Loud failures**: any non-zero CLI exit raises
  :class:`DVCTrackingError` with the captured stderr.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


class DVCTrackingError(RuntimeError):
    """Raised when a DVC CLI invocation fails or the workspace is misconfigured."""


def _pointer_path(target: Path, repo_root: Path) -> Path:
    """Return the ``<name>.dvc`` pointer location for ``target``.

    DVC writes pointer files at the parent directory of the tracked
    artifact (or, for top-level paths, at the repo root). For our
    notebook use case the targets are nested under ``data/raw/``, so we
    place the pointer at the repo root with the basename — matching the
    existing ``lepard_train_4000000_rev0194f95.jsonl.dvc`` convention.
    """
    return repo_root / f"{target.name}.dvc"


def is_dvc_repo(repo_root: PathLike) -> bool:
    """Return ``True`` iff ``repo_root`` contains a ``.dvc/`` directory.

    Args:
        repo_root: Project root to probe.

    Returns:
        ``True`` if DVC has been initialised in this workspace.
    """
    return (Path(repo_root) / ".dvc").is_dir()


def is_tracked(target: PathLike, repo_root: PathLike) -> bool:
    """Return ``True`` iff a ``.dvc`` pointer for ``target`` already exists.

    Args:
        target: Directory or file to check.
        repo_root: Project root that owns the pointer file.

    Returns:
        ``True`` if ``<target.name>.dvc`` exists at the repo root.
    """
    return _pointer_path(Path(target), Path(repo_root)).is_file()


def add_artifact(target: PathLike, repo_root: PathLike) -> None:
    """Run ``dvc add <target>`` from ``repo_root``.

    Args:
        target: Path to track. Must already exist on disk.
        repo_root: Working directory for the DVC invocation.

    Raises:
        DVCTrackingError: ``dvc add`` exited non-zero.
    """
    result = subprocess.run(
        ["dvc", "add", str(target)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DVCTrackingError(f"dvc add failed (exit {result.returncode}): {result.stderr.strip()}")


def push_artifact(repo_root: PathLike) -> None:
    """Run ``dvc push`` from ``repo_root`` to upload tracked artifacts.

    Args:
        repo_root: Working directory for the DVC invocation. Must
            already have a configured remote.

    Raises:
        DVCTrackingError: ``dvc push`` exited non-zero.
    """
    result = subprocess.run(
        ["dvc", "push"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DVCTrackingError(f"dvc push failed (exit {result.returncode}): {result.stderr.strip()}")


def track_shard_directory(
    target: PathLike,
    repo_root: PathLike,
    push: bool = False,
) -> None:
    """Idempotently version a directory with DVC and optionally push it.

    Decision flow:

    1. Verify ``repo_root`` is a DVC workspace; raise if not.
    2. If a pointer file already exists for ``target``, return — the
       directory is already tracked, nothing to do.
    3. Otherwise call :func:`add_artifact` to create the pointer.
    4. If ``push=True``, call :func:`push_artifact` to upload to the
       configured remote.

    Args:
        target: Directory to track.
        repo_root: Project root containing ``.dvc/``.
        push: When ``True``, also upload to the configured DVC remote
            after adding. Defaults to ``False`` so the notebook can
            opt in explicitly.

    Raises:
        DVCTrackingError: Workspace is not a DVC repo, or any CLI
            invocation fails.
    """
    target_path = Path(target)
    root_path = Path(repo_root)

    if not is_dvc_repo(root_path):
        raise DVCTrackingError(f"not a DVC repo: {root_path} (missing .dvc/)")

    if is_tracked(target_path, root_path):
        return

    add_artifact(target_path, root_path)
    if push:
        push_artifact(root_path)
