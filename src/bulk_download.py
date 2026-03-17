# src/bulk_download.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/src/bulk_download.py
# SRP: Download bulk CSV files from CourtListener S3.

import subprocess  # used: aws CLI attempt
from pathlib import Path  # used: file paths
from typing import Any, Dict, Optional, Union

import requests  # used: HTTP streaming fallback
from tqdm import tqdm  # used: progress bar

from src.config import PipelineConfig


def _download_via_aws_cli(s3_bucket_name: str, s3_key: str, local_path: Path) -> bool:
    """Try aws CLI. Returns True if successful."""
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", f"s3://{s3_bucket_name}/{s3_key}", str(local_path), "--no-sign-request"],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _download_via_requests(s3_bucket_url: str, s3_key: str, local_path: Path) -> None:
    """Streaming HTTP download with progress bar."""
    url = f"{s3_bucket_url}/{s3_key}"
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total_size = int(resp.headers.get("content-length", 0))

    with open(local_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=local_path.name) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_file(
    s3_key: str,
    local_path: Union[str, Path],
    config: Optional[PipelineConfig] = None,
    logger: Any = None,
) -> None:
    """Download a single S3 file. Skips if exists."""
    if config is None:
        config = PipelineConfig()
    local_path = Path(local_path)

    if local_path.exists():
        if logger:
            logger.info(f"  ✓ {local_path.name} exists, skipping")
        return

    if _download_via_aws_cli(config.s3_bucket_name, s3_key, local_path):
        if logger:
            logger.info(f"  ✓ {local_path.name} via aws CLI")
        return

    if logger:
        logger.info(f"  aws CLI unavailable, using requests for {local_path.name}...")
    _download_via_requests(config.s3_bucket_url, s3_key, local_path)
    if logger:
        logger.info(f"  ✓ {local_path.name}")


def download_bulk_csvs(
    latest_files: Dict[str, Dict[str, Any]],
    config: Optional[PipelineConfig] = None,
    logger: Any = None,
) -> Dict[str, Path]:
    """Download all 4 bulk CSV files.

    Returns:
        dict: {label: Path}
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
