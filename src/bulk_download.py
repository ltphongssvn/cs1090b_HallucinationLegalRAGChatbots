# src/bulk_download.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/bulk_download.py
"""Bulk CSV downloader for the CourtListener public S3 bucket.

This module provides a two-tier download strategy for acquiring the four
bulk CSV exports (dockets, opinions, opinion-clusters, courts) that seed
the RAG corpus:

    1. Preferred path: the `aws` CLI with `--no-sign-request` (fastest,
       supports multipart and resume).
    2. Fallback path: a streaming `requests.get` with a tqdm progress bar,
       used when the `aws` CLI is absent or fails.

Design notes
------------
* **Idempotent**: `download_file` skips any destination path that already
  exists. Callers relying on re-download must delete the target first.
* **Atomic writes**: downloads stream to a sibling ``*.part`` file and are
  renamed on success, so an interrupted run never leaves a truncated file
  that a later idempotent rerun would mistakenly accept.
* **No credentials**: all S3 access is anonymous; the bucket is public.
* **SRP**: this module only *fetches* files. Parsing, validation, and
  schema handling live in ``src.dataset_loader`` and ``src.row_validator``.

Typical use
-----------
    >>> from src.bulk_download import download_bulk_csvs
    >>> from src.s3_discovery import discover_latest_bulk_files
    >>> latest = discover_latest_bulk_files()
    >>> paths = download_bulk_csvs(latest)
    >>> paths["opinions"]
    PosixPath('data/bulk/opinions-2024-10-01.csv.bz2')
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
from tqdm import tqdm

from src.config import PipelineConfig

_AWS_CLI_TIMEOUT_SECONDS = 3600
_HTTP_CHUNK_BYTES = 8192
_HTTP_CONNECT_READ_TIMEOUT = 60


def _download_via_aws_cli(
    s3_bucket_name: str,
    s3_key: str,
    local_path: Path,
) -> bool:
    """Attempt an anonymous download via the ``aws`` CLI.

    Runs ``aws s3 cp s3://<bucket>/<key> <local_path> --no-sign-request``
    as a subprocess and returns whether it succeeded. Both stdout and
    stderr are captured but only logged (via the caller) on failure, to
    keep the happy-path output clean.

    Args:
        s3_bucket_name: The bucket name without any scheme prefix
            (e.g. ``"com-courtlistener-storage"``).
        s3_key: The object key inside the bucket.
        local_path: Destination file path. Parent directories must
            already exist.

    Returns:
        ``True`` if the CLI exited 0, ``False`` if the CLI is missing,
        the process timed out after
        :data:`_AWS_CLI_TIMEOUT_SECONDS`, or it exited non-zero.

    Notes:
        This function does not raise on subprocess failure — it converts
        every non-success outcome into ``False`` so the caller can fall
        back to the HTTP path.
    """
    try:
        result = subprocess.run(
            [
                "aws", "s3", "cp",
                f"s3://{s3_bucket_name}/{s3_key}",
                str(local_path),
                "--no-sign-request",
            ],
            capture_output=True,
            text=True,
            timeout=_AWS_CLI_TIMEOUT_SECONDS,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _download_via_requests(
    s3_bucket_url: str,
    s3_key: str,
    local_path: Path,
) -> None:
    """Stream a single object over HTTPS with a tqdm progress bar.

    Writes to ``<local_path>.part`` and renames on success so an
    interrupted run cannot leave a truncated file that the idempotency
    check in :func:`download_file` would later treat as complete.

    Args:
        s3_bucket_url: Fully-qualified bucket URL, e.g.
            ``"https://com-courtlistener-storage.s3.amazonaws.com"``.
        s3_key: The object key inside the bucket.
        local_path: Final destination path. A sibling ``.part`` file is
            created during download and removed on any error.

    Raises:
        requests.HTTPError: The server returned a non-2xx status.
        requests.RequestException: Any lower-level transport failure
            (DNS, connection reset, read timeout, etc.).
        OSError: The local filesystem could not be written to.
    """
    url = f"{s3_bucket_url}/{s3_key}"
    part_path = local_path.with_suffix(local_path.suffix + ".part")
    try:
        resp = requests.get(url, stream=True, timeout=_HTTP_CONNECT_READ_TIMEOUT)
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))
        with open(part_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=local_path.name
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=_HTTP_CHUNK_BYTES):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
        os.replace(part_path, local_path)
    except BaseException:
        if part_path.exists():
            try:
                part_path.unlink()
            except OSError:
                pass
        raise


def download_file(
    s3_key: str,
    local_path: Union[str, Path],
    config: Optional[PipelineConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Idempotently download a single S3 object.

    The call is a no-op if ``local_path`` already exists — no HEAD
    request, no size check. Callers that need a forced re-download must
    delete the destination first.

    Download strategy:
        1. Try the ``aws`` CLI (fast, supports multipart).
        2. On CLI failure or absence, fall back to a streaming HTTP GET.

    Args:
        s3_key: The object key inside the configured bucket.
        local_path: Destination file path. Its parent directory is
            **not** created automatically; use
            :func:`download_bulk_csvs` for that.
        config: Pipeline configuration providing the bucket name and
            public HTTPS URL. A default :class:`PipelineConfig` is
            constructed when ``None``.
        logger: Optional standard-library logger. When supplied,
            progress and method-selection messages are emitted at INFO.

    Raises:
        requests.RequestException: Both the CLI path and the HTTP
            fallback failed. The CLI-path error is suppressed (only
            logged) and only the fallback exception surfaces.
        OSError: The destination is not writable.
    """
    if config is None:
        config = PipelineConfig()
    local_path = Path(local_path)

    if local_path.exists():
        if logger is not None:
            logger.info("  ✓ %s exists, skipping", local_path.name)
        return

    if _download_via_aws_cli(config.s3_bucket_name, s3_key, local_path):
        if logger is not None:
            logger.info("  ✓ %s via aws CLI", local_path.name)
        return

    if logger is not None:
        logger.info(
            "  aws CLI unavailable or failed, using requests for %s...",
            local_path.name,
        )
    _download_via_requests(config.s3_bucket_url, s3_key, local_path)
    if logger is not None:
        logger.info("  ✓ %s", local_path.name)


def download_bulk_csvs(
    latest_files: Dict[str, Dict[str, Any]],
    config: Optional[PipelineConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """Download every CourtListener bulk CSV listed in ``latest_files``.

    Ensures the configured ``bulk_dir`` exists, then calls
    :func:`download_file` for each entry. Files already present on disk
    are skipped (see :func:`download_file` for idempotency semantics).

    Args:
        latest_files: Mapping of corpus label → file metadata, as
            produced by :func:`src.s3_discovery.discover_latest_bulk_files`.
            Each value must contain at least a ``"key"`` entry giving
            the S3 object key.
        config: Pipeline configuration. A default :class:`PipelineConfig`
            is constructed when ``None``.
        logger: Optional logger forwarded to :func:`download_file`.

    Returns:
        Mapping of the same labels to the local :class:`~pathlib.Path`
        of each downloaded file.

    Raises:
        requests.RequestException: Any single file failed to download
        via both strategies. Downloads that completed before the failure
        remain on disk; the function does not roll them back.
    """
    if config is None:
        config = PipelineConfig()
    config.bulk_dir.mkdir(parents=True, exist_ok=True)

    local_paths: Dict[str, Path] = {}
    for label, file_info in latest_files.items():
        filename = Path(file_info["key"]).name
        local_path = config.bulk_dir / filename
        download_file(file_info["key"], local_path, config=config, logger=logger)
        local_paths[label] = local_path
    return local_paths
