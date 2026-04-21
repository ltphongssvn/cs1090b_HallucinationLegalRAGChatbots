# src/gcs_utils.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/gcs_utils.py
"""Google Cloud Storage streaming utilities for data pipeline.

This module provides helper functions for streaming data directly from GCS
without downloading to local disk. Designed for use in Google Colab where
ephemeral storage is limited but GCS provides persistent, scalable storage.

All functions operate on gs:// URIs and return file-like objects or metadata
that can be consumed by Polars, pandas, and other data processing libraries.

Example:
    >>> from src.gcs_utils import GCSConfig, gcs_exists, gcs_open
    >>> cfg = GCSConfig(bucket_name="my-bucket", project_id="my-project")
    >>> if gcs_exists("data/file.jsonl", cfg):
    ...     with gcs_open("data/file.jsonl", cfg) as f:
    ...         data = f.read()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GCSConfig:
    """Configuration for Google Cloud Storage access.

    Attributes:
        bucket_name: GCS bucket name (without gs:// prefix).
        project_id: GCP project ID for authentication.
        use_gcs_streaming: Flag to enable GCS streaming mode. When False,
            falls back to local file operations.
    """

    bucket_name: str = "apcomp-209b-bucket"
    project_id: str = "your-project-id"
    use_gcs_streaming: bool = False


def _get_gcs_filesystem(config: GCSConfig):
    """Initialize and return a gcsfs.GCSFileSystem instance.

    Args:
        config: GCS configuration object.

    Returns:
        gcsfs.GCSFileSystem instance for streaming operations.

    Raises:
        ImportError: If gcsfs is not installed.
    """
    try:
        import gcsfs  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError("gcsfs is required for GCS streaming. Install with: pip install gcsfs")

    return gcsfs.GCSFileSystem(project=config.project_id)  # type: ignore[import-not-found]


def gcs_exists(blob_path: str, config: GCSConfig) -> bool:
    """Check if a file exists in GCS bucket.

    Args:
        blob_path: Path within the bucket (without gs:// prefix or bucket name).
        config: GCS configuration object.

    Returns:
        True if the file exists, False otherwise.

    Example:
        >>> cfg = GCSConfig(bucket_name="my-bucket")
        >>> gcs_exists("data/file.jsonl", cfg)
        True
    """
    if not config.use_gcs_streaming:
        return False

    fs = _get_gcs_filesystem(config)
    full_path = f"gs://{config.bucket_name}/{blob_path}"
    return fs.exists(full_path)


def gcs_list_files(prefix: str, config: GCSConfig) -> List[str]:
    """List all files with given prefix in GCS bucket.

    Args:
        prefix: Prefix path within the bucket to search.
        config: GCS configuration object.

    Returns:
        List of blob paths (relative to bucket root, without gs:// prefix).
        Directories (paths ending with /) are excluded.

    Example:
        >>> cfg = GCSConfig(bucket_name="my-bucket")
        >>> files = gcs_list_files("data/shards/", cfg)
        >>> print(files)
        ['data/shards/shard_0001.jsonl', 'data/shards/shard_0002.jsonl']
    """
    if not config.use_gcs_streaming:
        return []

    fs = _get_gcs_filesystem(config)
    full_path = f"gs://{config.bucket_name}/{prefix}"
    files = fs.ls(full_path)
    return [f.replace(f"{config.bucket_name}/", "") for f in files if not f.endswith("/")]


def gcs_open(blob_path: str, config: GCSConfig, mode: str = "rb"):
    """Open a GCS file for streaming.

    Args:
        blob_path: Path within the bucket (without gs:// prefix or bucket name).
        config: GCS configuration object.
        mode: File open mode ('rb' for binary read, 'rt' for text read, etc.).

    Returns:
        File-like object that can be used with context managers or read directly.

    Example:
        >>> cfg = GCSConfig(bucket_name="my-bucket")
        >>> with gcs_open("data/file.jsonl", cfg, mode='rt') as f:
        ...     for line in f:
        ...         print(line)
    """
    if not config.use_gcs_streaming:
        raise RuntimeError("GCS streaming is not enabled in config")

    fs = _get_gcs_filesystem(config)
    full_path = f"gs://{config.bucket_name}/{blob_path}"
    return fs.open(full_path, mode)


def get_gcs_path(blob_path: str, config: GCSConfig) -> str:
    """Get full GCS path for use with libraries that support gs:// URIs.

    Many data processing libraries (Polars, pandas, etc.) can read directly
    from gs:// URIs when gcsfs is installed. This helper constructs the
    full URI from a blob path.

    Args:
        blob_path: Path within the bucket (without gs:// prefix or bucket name).
        config: GCS configuration object.

    Returns:
        Full gs:// URI string.

    Example:
        >>> cfg = GCSConfig(bucket_name="my-bucket")
        >>> get_gcs_path("data/file.jsonl", cfg)
        'gs://my-bucket/data/file.jsonl'
    """
    return f"gs://{config.bucket_name}/{blob_path}"


def gcs_get_info(blob_path: str, config: GCSConfig) -> Dict[str, Any]:
    """Get metadata about a GCS file.

    Args:
        blob_path: Path within the bucket (without gs:// prefix or bucket name).
        config: GCS configuration object.

    Returns:
        Dictionary with file metadata including 'size' in bytes.

    Example:
        >>> cfg = GCSConfig(bucket_name="my-bucket")
        >>> info = gcs_get_info("data/file.jsonl", cfg)
        >>> print(f"Size: {info['size'] / 1e9:.2f} GB")
    """
    if not config.use_gcs_streaming:
        raise RuntimeError("GCS streaming is not enabled in config")

    fs = _get_gcs_filesystem(config)
    full_path = f"gs://{config.bucket_name}/{blob_path}"
    return fs.info(full_path)


def authenticate_gcs_colab() -> None:
    """Authenticate with GCS in Google Colab environment.

    This function should be called once at the start of a Colab notebook
    to enable GCS access. It uses Colab's built-in authentication flow.

    Raises:
        ImportError: If not running in Google Colab environment.

    Example:
        >>> authenticate_gcs_colab()
        # User will be prompted to authenticate via browser
    """
    try:
        from google.colab import auth  # type: ignore[import-untyped]

        auth.authenticate_user()
    except ImportError:
        raise ImportError(
            "google.colab.auth is only available in Google Colab. "
            "For local development, use gcloud auth application-default login"
        )
