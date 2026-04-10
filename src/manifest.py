# src/manifest.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/manifest.py
"""Run manifest writer, reader, and validator for the ingest pipeline.

Every extraction run produces a single ``manifest.json`` under
:attr:`PipelineConfig.shard_dir` that captures everything needed to
audit or reproduce the run:

* **Run metadata**: UTC timestamp, Python version, Git revision, and a
  projection of the :class:`PipelineConfig` fields that affect output.
* **Source provenance**: filenames and SHA-256 checksums of the four
  input CSVs.
* **Output provenance**: filenames and SHA-256 checksums of every
  emitted shard.
* **Filter chain sizes**: court/docket/cluster counts after each stage.
* **Extraction statistics**: totals, per-source counts, distributions,
  skip reasons, text-length percentiles, schema sample.

Design notes
------------
* **Checksums cover both ends**: source files *and* output shards are
  hashed, so a later run can prove byte-for-byte identity of both its
  inputs and its outputs.
* **Schema version**: the top-level ``"version"`` field is bumped
  whenever the manifest layout changes, letting downstream tools
  detect incompatible formats.
* **No side effects beyond the manifest file**: this module never
  touches shard files or source CSVs — only reads their bytes for
  hashing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

#: Block size for streaming SHA-256 hashing. 8 KiB balances syscall
#: overhead against per-call buffer allocation.
CHECKSUM_BUFFER_SIZE: int = 8192


def file_checksum(filepath: Union[str, Path], buffer_size: int = CHECKSUM_BUFFER_SIZE) -> str:
    """Return the hex SHA-256 of ``filepath``, streamed in fixed-size blocks.

    Streaming avoids loading multi-GB bulk CSVs and shard files into
    memory. The default block size matches :data:`CHECKSUM_BUFFER_SIZE`.

    Args:
        filepath: File to hash. Must be readable.
        buffer_size: Read-block size in bytes.

    Returns:
        64-character lowercase hex digest.
    """
    sha = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(buffer_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _get_git_revision() -> str:
    """Return the current HEAD commit hash, or ``"unknown"``.

    Falls back to ``"unknown"`` when git is not installed, the
    invocation times out, or the working directory is not a git repo —
    the manifest is still useful without this field, so a failure here
    must not abort the write.
    """
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
    """Build the ``run_metadata`` block embedded in every manifest.

    Captures the fields that affect reproducibility: timestamp, Python
    version, Git revision, and the :class:`PipelineConfig` knobs whose
    values change pipeline output. Fields that only affect performance
    (e.g. chunk sizes) are deliberately excluded to keep diffs focused
    on behaviourally-meaningful changes.
    """
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
    """Assemble and write the run manifest as indented JSON.

    Hashes every ``shard_*.jsonl`` under ``shard_dir`` and every file
    in ``local_paths`` that exists, builds the run-metadata block, and
    merges in the extraction statistics. The manifest layout is
    version-stamped (``"version": 2``) so downstream tools can detect
    format drift.

    Args:
        manifest_path: Destination JSON file path.
        shard_dir: Directory containing the emitted ``shard_*.jsonl``
            files to hash.
        extraction_stats: Statistics dict returned by
            :func:`src.extract.extract_opinions_to_shards`.
        local_paths: Mapping of corpus label → source CSV path for the
            four input files.
        fed_court_ids: Set of federal appellate court slugs used.
        docket_count: Federal-appellate docket count after filtering.
        cluster_count: Federal-appellate cluster count after filtering.
        shard_size: Rows-per-shard used by the writer.
        config: Pipeline configuration; defaults to :class:`PipelineConfig`.

    Returns:
        The fully-populated manifest dict (also written to disk).
    """
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
    """Load a previously-written manifest, or an empty dict if absent.

    Returning an empty dict rather than raising lets callers probe for
    a manifest without try/except — a missing manifest is a normal
    "first run" condition, not an error.
    """
    manifest_file = Path(manifest_path)
    if manifest_file.exists():
        return json.loads(manifest_file.read_text())  # type: ignore[no-any-return]
    return {}


def validate_manifest_shards(manifest_data: Dict[str, Any], shard_dir: Union[str, Path]) -> bool:
    """Verify every shard listed in the manifest exists with the recorded hash.

    Returns ``False`` on the first discrepancy rather than reporting
    all mismatches — the caller's expected action on any single
    failure is the same (rerun extraction), so early exit is the
    right economy.

    Args:
        manifest_data: Manifest dict returned by :func:`read_manifest`
            or :func:`write_manifest`.
        shard_dir: Directory containing the shards to verify.

    Returns:
        ``True`` iff the manifest has a ``checksum`` block and every
        listed shard is present on disk with a matching SHA-256.
    """
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
