# src/hf_export.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/hf_export.py
# SRP: Export shards to HuggingFace Datasets format.
import json
from typing import Any, Dict, List, Optional

from datasets import Dataset  # type: ignore[import-untyped]

from src.config import PipelineConfig


def build_dataset_info(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Build dataset_info dict from manifest for HF metadata."""
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
    """Load all JSONL shards into a HuggingFace Dataset."""
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
    """Push dataset to HuggingFace Hub with dataset_info metadata."""
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
