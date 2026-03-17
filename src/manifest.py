# src/manifest.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/src/manifest.py
# SRP: Write, read, and validate manifest + checksums + run metadata.

import hashlib  # used: SHA256 checksums
import json  # used: manifest JSON
import subprocess  # used: git revision
import sys  # used: python version
from datetime import datetime, timezone  # used: timestamp
from pathlib import Path  # used: file paths
from typing import Any, Dict, Optional, Set, Union

CHECKSUM_BUFFER_SIZE: int = 8192


def file_checksum(filepath: Union[str, Path], buffer_size: int = CHECKSUM_BUFFER_SIZE) -> str:
    """Compute SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(buffer_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _get_git_revision() -> str:
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _get_run_metadata(config: Any) -> Dict[str, Any]:
    """Collect reproducibility metadata for this pipeline run."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "git_revision": _get_git_revision(),
        "config": {
            "shard_size": config.shard_size,
            "min_text_length": config.min_text_length,
            "schema_audit_per_shard": config.schema_audit_per_shard,
            "min_expected_total": config.min_expected_total,
            "federal_appellate_court_ids": sorted(config.federal_appellate_court_ids),
            "text_source_fields": list(config.text_source_fields),
            "pinned_snapshot": config.has_pinned_snapshot,
        },
    }


def write_manifest(
    manifest_path: Union[str, Path],
    shard_dir: Union[str, Path],
    extraction_stats: Dict[str, Any],
    local_paths: Dict[str, Path],
    fed_court_ids: Set[str],
    docket_count: int,
    cluster_count: int,
    shard_size: int,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Write manifest with checksums, stats, and run metadata."""
    from src.config import PipelineConfig

    if config is None:
        config = PipelineConfig()

    shard_dir_path = Path(shard_dir)
    checksums: Dict[str, str] = {}
    for shard_path in sorted(shard_dir_path.glob("shard_*.jsonl")):
        checksums[shard_path.name] = file_checksum(shard_path)

    # Source file hashes for reproducibility
    source_checksums: Dict[str, str] = {}
    for label, filepath in local_paths.items():
        if Path(filepath).exists():
            source_checksums[label] = file_checksum(filepath)

    manifest_data: Dict[str, Any] = {
        "version": 2,
        "run_metadata": _get_run_metadata(config),
        "num_cases": extraction_stats["extracted_total"],
        "num_shards": extraction_stats["num_shards"],
        "shard_size": shard_size,
        "source_files": {k: v.name for k, v in local_paths.items()},
        "source_checksums": source_checksums,
        "federal_courts": sorted(fed_court_ids),
        "filter_chain": {
            "courts": len(fed_court_ids),
            "dockets": docket_count,
            "clusters": cluster_count,
        },
        "text_source_counts": extraction_stats["text_source_counts"],
        "court_distribution": extraction_stats.get("court_distribution", {}),
        "opinion_type_distribution": extraction_stats.get("opinion_type_distribution", {}),
        "precedential_status_distribution": extraction_stats.get("precedential_status_distribution", {}),
        "text_length_stats": extraction_stats.get("text_length_stats", {}),
        "ocr_extracted_count": extraction_stats.get("ocr_extracted_count", 0),
        "schema": extraction_stats.get("schema", []),
        "skipped_empty": extraction_stats.get("skipped_empty", 0),
        "skipped_parse": extraction_stats.get("skipped_parse", 0),
        "scanned": extraction_stats.get("scanned", 0),
        "checksum": checksums,
    }

    Path(manifest_path).write_text(json.dumps(manifest_data, indent=2))
    return manifest_data


def read_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """Read manifest JSON. Returns empty dict if not found."""
    manifest_file = Path(manifest_path)
    if manifest_file.exists():
        return json.loads(manifest_file.read_text())  # type: ignore[no-any-return]
    return {}


def validate_manifest_shards(manifest_data: Dict[str, Any], shard_dir: Union[str, Path]) -> bool:
    """Verify all shards exist and checksums match."""
    if "checksum" not in manifest_data:
        return False
    shard_dir_path = Path(shard_dir)
    for shard_name, expected_hash in manifest_data["checksum"].items():
        shard_path = shard_dir_path / shard_name
        if not shard_path.exists():
            return False
        if file_checksum(shard_path) != expected_hash:
            return False
    return True
