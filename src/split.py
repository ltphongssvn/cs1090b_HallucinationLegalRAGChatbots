# src/split.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/src/split.py
# SRP: Leakage-free data splits for legal RAG evaluation.

import random  # used: deterministic shuffling
from collections import defaultdict  # used: group records
from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class SplitConfig:
    """Split parameters for reproducibility."""

    strategy: str = "docket"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    train_cutoff: str = ""
    val_cutoff: str = ""


def _group_split(
    records: List[Dict[str, Any]],
    group_key: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split by group (docket or cluster) to prevent leakage."""
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record[group_key]].append(record)

    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n = len(group_ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_groups: Set[Any] = set(group_ids[:train_end])
    val_groups: Set[Any] = set(group_ids[train_end:val_end])

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for group_id, group_records in groups.items():
        if group_id in train_groups:
            splits["train"].extend(group_records)
        elif group_id in val_groups:
            splits["val"].extend(group_records)
        else:
            splits["test"].extend(group_records)
    return splits


def split_by_docket(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split by docket_id. Opinions from same case stay together."""
    return _group_split(records, "docket_id", train_ratio, val_ratio, seed)


def split_by_cluster(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split by cluster_id. Opinions from same cluster stay together."""
    return _group_split(records, "cluster_id", train_ratio, val_ratio, seed)


def split_by_time(
    records: List[Dict[str, Any]],
    train_cutoff: str,
    val_cutoff: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Temporal split. No future leakage into training."""
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for record in records:
        date_val: str = record.get("date_filed", "")
        if date_val < train_cutoff:
            splits["train"].append(record)
        elif date_val < val_cutoff:
            splits["val"].append(record)
        else:
            splits["test"].append(record)
    return splits


def validate_no_leakage(
    splits: Dict[str, List[Dict[str, Any]]],
    group_key: str = "docket_id",
) -> Dict[str, Any]:
    """Verify no group appears in multiple splits. Returns diagnostic report."""
    split_groups: Dict[str, Set[Any]] = {}
    for split_name, records in splits.items():
        split_groups[split_name] = {r[group_key] for r in records}

    leaked: Set[Any] = set()
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for split_a, split_b in pairs:
        overlap = split_groups[split_a] & split_groups[split_b]
        leaked |= overlap

    return {
        "leaked_groups": len(leaked),
        "leaked_ids": sorted(leaked) if leaked else [],
        "train_groups": len(split_groups["train"]),
        "val_groups": len(split_groups["val"]),
        "test_groups": len(split_groups["test"]),
        "train_records": len(splits["train"]),
        "val_records": len(splits["val"]),
        "test_records": len(splits["test"]),
    }
