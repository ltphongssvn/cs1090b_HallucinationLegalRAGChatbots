# src/split.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/split.py
"""Leakage-free train/val/test splits for legal RAG evaluation.

Three strategies are provided, all returning the same
``{"train", "val", "test"}`` shape:

* **Docket split** (:func:`split_by_docket`) — the default. Every
  opinion from the same docket goes to the same fold, so the model
  never sees a sibling opinion from the same case during training
  that it is later evaluated on.
* **Cluster split** (:func:`split_by_cluster`) — tighter grouping:
  every opinion in the same opinion cluster (majority + concurrences
  + dissents) stays together.
* **Time split** (:func:`split_by_time`) — temporal cutoffs prevent
  "future leakage" where a training opinion cites a test opinion.

A diagnostic helper :func:`validate_no_leakage` audits any split
dict and reports overlap counts, so a misconfigured split is caught
before evaluation numbers become meaningless.

All random choices route through a ``seed`` argument so reruns
produce byte-identical splits.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class SplitConfig:
    """Reproducible split parameters.

    Attributes:
        strategy: One of ``"docket"``, ``"cluster"``, or ``"time"``.
        train_ratio: Fraction of groups assigned to train (group-based
            strategies only).
        val_ratio: Fraction of groups assigned to val. Test receives
            ``1 - train_ratio - val_ratio``.
        seed: RNG seed for group shuffling. Fixed default for
            publishable reproducibility.
        train_cutoff: ISO date string; rows with ``date_filed`` strictly
            earlier go to train (time strategy only).
        val_cutoff: ISO date string; rows with ``date_filed`` strictly
            earlier than this but >= ``train_cutoff`` go to val;
            the rest go to test.
    """

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
    """Split records by a grouping key so no group crosses folds.

    Bucketises records by ``group_key``, sorts the group IDs (for
    determinism across Python invocations), shuffles with a seeded
    :class:`random.Random`, and slices the ID list at the train/val
    boundaries. Every record in a given group then follows its group
    into the chosen fold.

    Args:
        records: Rows to split. Each must contain ``group_key``.
        group_key: Field name to group by (e.g. ``"docket_id"``).
        train_ratio: Fraction of groups for train.
        val_ratio: Fraction of groups for val.
        seed: RNG seed.

    Returns:
        A ``{"train", "val", "test"}`` dict of record lists.
    """
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
    """Split by ``docket_id`` — every opinion in a case stays together.

    This is the default and recommended strategy for legal RAG
    evaluation. Siblings on the same docket (main opinion plus any
    later revisions or memoranda) would otherwise create a
    near-duplicate leakage path.

    Args:
        records: Rows carrying a ``docket_id`` key.
        train_ratio: Fraction of dockets for train.
        val_ratio: Fraction of dockets for val.
        seed: RNG seed.

    Returns:
        A ``{"train", "val", "test"}`` dict of record lists.
    """
    return _group_split(records, "docket_id", train_ratio, val_ratio, seed)


def split_by_cluster(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split by ``cluster_id`` — every opinion in a cluster stays together.

    Tighter than docket split: keeps majority, concurrences, and
    dissents of the same decision in one fold. Use when the model
    might otherwise learn to quote a dissent against its own
    majority at evaluation time.

    Args:
        records: Rows carrying a ``cluster_id`` key.
        train_ratio: Fraction of clusters for train.
        val_ratio: Fraction of clusters for val.
        seed: RNG seed.

    Returns:
        A ``{"train", "val", "test"}`` dict of record lists.
    """
    return _group_split(records, "cluster_id", train_ratio, val_ratio, seed)


def split_by_time(
    records: List[Dict[str, Any]],
    train_cutoff: str,
    val_cutoff: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split by filing date with two cutoffs to prevent future leakage.

    Uses lexicographic string comparison on ``date_filed``, which is
    correct for ISO-8601 ``YYYY-MM-DD`` dates. Records with missing
    dates (empty string) fall into train because ``"" < "any date"``.

    Args:
        records: Rows carrying a ``date_filed`` key.
        train_cutoff: ISO date; rows strictly before go to train.
        val_cutoff: ISO date; rows in ``[train_cutoff, val_cutoff)``
            go to val; ``>= val_cutoff`` go to test.

    Returns:
        A ``{"train", "val", "test"}`` dict of record lists.
    """
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
    """Audit a split dict for group overlap between folds.

    Computes the intersection of ``group_key`` values across the three
    pairwise fold combinations and returns both the leak count and
    per-fold group/record counts. A healthy split has
    ``leaked_groups == 0``.

    Args:
        splits: A ``{"train", "val", "test"}`` dict from any strategy.
        group_key: The field to audit. Defaults to ``"docket_id"``;
            use ``"cluster_id"`` when checking a cluster-based split.

    Returns:
        A diagnostic dict with ``leaked_groups``, ``leaked_ids``
        (sorted; empty on success), and per-fold group/record counts.
    """
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
