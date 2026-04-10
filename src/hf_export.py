# src/hf_export.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/hf_export.py
"""Export sharded JSONL output to the Hugging Face ``datasets`` format.

This module bridges the pipeline's shard directory and the Hugging Face
Hub. It loads every ``shard_*.jsonl`` emitted by :mod:`src.extract`,
assembles them into a single :class:`datasets.Dataset`, attaches a
metadata summary derived from the run manifest, and optionally pushes
the result to a Hub repository.

Design notes
------------
* **In-memory assembly**: shards are loaded into a list before
  :meth:`Dataset.from_list` is called. This is fine for the federal
  appellate corpus (~100K rows); larger corpora should switch to
  :func:`datasets.load_dataset` with the ``json`` builder and streaming.
* **Metadata injection**: the run manifest is serialised into
  ``ds.info.description`` as indented JSON so the Hub displays a
  human-readable provenance block on the dataset card.
* **SRP**: this module only *exports*. It does not read the pipeline
  manifest itself — callers pass it in — so the export path can be
  tested against a synthetic manifest without touching the filesystem.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from datasets import Dataset  # type: ignore[import-untyped]

from src.config import PipelineConfig


def build_dataset_info(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Project a pipeline manifest onto the HF dataset-info schema.

    Pulls the fields that belong on a Hub dataset card (row count,
    version, per-court counts, text length statistics, and the full
    run metadata block) and leaves internal-only fields behind.

    Args:
        manifest: The pipeline manifest dict produced by
            :mod:`src.manifest`. Missing keys are tolerated and fall
            back to empty defaults.

    Returns:
        A dict suitable for JSON-serialising into
        :attr:`datasets.DatasetInfo.description`.
    """
    return {
        "dataset_name": "legal-rag-federal-appellate",
        "num_examples": manifest.get("num_cases", 0),
        "version": manifest.get("version", 0),
        "federal_courts": manifest.get("federal_courts", []),
        "text_length_stats": manifest.get("text_length_stats", {}),
        "text_source_counts": manifest.get("text_source_counts", {}),
        "run_metadata": manifest.get("run_metadata", {}),
    }


def shards_to_hf_dataset(config: Optional[PipelineConfig] = None) -> Dataset:
    """Load every ``shard_*.jsonl`` under ``config.shard_dir`` into a Dataset.

    Shards are read in lexicographic order (which matches numeric order
    because the writer uses 4-digit zero-padded indices). Empty lines
    are skipped so partially-flushed files do not break the load.

    Args:
        config: Pipeline configuration supplying ``shard_dir``. A
            default :class:`PipelineConfig` is used when ``None``.

    Returns:
        A :class:`datasets.Dataset`. An empty dataset is returned when
        no shards are found, so callers can handle the "nothing to
        export" case without branching on :class:`FileNotFoundError`.
    """
    if config is None:
        config = PipelineConfig()
    shard_files = sorted(config.shard_dir.glob("shard_*.jsonl"))
    if not shard_files:
        return Dataset.from_list([])
    records: List[Dict[str, Any]] = []
    for shard_path in shard_files:
        with open(shard_path) as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
    return Dataset.from_list(records)


def push_to_hub(
    config: Optional[PipelineConfig] = None,
    manifest: Optional[Dict[str, Any]] = None,
    repo_id: str = "legal-rag-federal-appellate-v1",
    private: bool = True,
    logger: Any = None,
) -> None:
    """Publish the sharded corpus to a Hugging Face Hub dataset repository.

    Loads the shards via :func:`shards_to_hf_dataset`, attaches the
    manifest-derived description (when a manifest is provided), and
    calls :meth:`Dataset.push_to_hub`. An empty shard directory is a
    no-op with a warning, not a failure.

    Authentication relies on the ambient ``HUGGINGFACE_HUB_TOKEN`` env
    var or a prior ``huggingface-cli login``; this function does not
    accept or store credentials itself.

    Args:
        config: Pipeline configuration supplying ``shard_dir``.
        manifest: Optional run manifest. When provided, its projection
            via :func:`build_dataset_info` is written to the dataset
            description as indented JSON.
        repo_id: Target Hub repo, either ``"name"`` or ``"org/name"``.
            Defaults to ``"legal-rag-federal-appellate-v1"``.
        private: When ``True`` (default), the repo is created as
            private. Ignored if the repo already exists.
        logger: Optional logger for push progress and the final URL.
    """
    ds = shards_to_hf_dataset(config)
    if len(ds) == 0:
        if logger:
            logger.warning("No records to push")
        return
    if manifest:
        ds.info.description = json.dumps(build_dataset_info(manifest), indent=2)
    if logger:
        logger.info(f"Pushing {len(ds):,} records to {repo_id}...")
    ds.push_to_hub(repo_id, private=private)
    if logger:
        logger.info(f"✓ Pushed to https://huggingface.co/datasets/{repo_id}")
