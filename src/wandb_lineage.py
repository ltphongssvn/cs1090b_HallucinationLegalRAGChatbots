"""W&B lineage helpers — connect upstream artifacts to downstream runs.

The W&B lineage graph is built exclusively from ``run.use_artifact(name)``
calls (W&B docs: Create model lineage map). Without these calls, a reviewer
on the W&B run page cannot trace bm25 -> rrf -> reranker -> rag -> judge as
a connected DAG.

Offline-safe: ``use_artifact`` raises in offline mode (W&B GitHub issue
#5309). This module silently skips when the active run is offline or
missing, so cluster nodes without internet behave correctly.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def link_input_artifacts(
    artifact_names: Iterable[str],
    *,
    artifact_type: str = "dataset",
) -> list[Any]:
    """Mark each upstream artifact as an input to the active W&B run.

    Each call appears in the lineage graph as an arrow from the named
    upstream artifact into this script\'s run, regardless of whether
    this script also logs an output artifact.

    Args:
        artifact_names: Fully-qualified upstream artifact identifiers,
            e.g. ``["baseline-bm25:latest", "baseline-bge-m3:latest"]``.
        artifact_type: Optional type hint shown in the W&B UI.

    Returns:
        List of resolved Artifact objects (empty when wandb is unavailable,
        no run is active, or the run is offline). Failed individual lookups
        are logged and skipped; remaining names are still attempted.
    """
    try:
        import wandb
    except ImportError:
        logger.debug("wandb not installed — skipping use_artifact lineage")
        return []

    run = wandb.run
    if run is None:
        logger.debug("no active wandb run — skipping use_artifact lineage")
        return []
    if getattr(run, "offline", False):
        logger.debug("wandb run is offline — use_artifact unsupported (issue #5309)")
        return []

    resolved: list[Any] = []
    for name in artifact_names:
        try:
            art = run.use_artifact(name, type=artifact_type)
            resolved.append(art)
        except Exception as exc:  # noqa: BLE001 — lineage is best-effort
            logger.warning(f"use_artifact({name!r}) failed: {exc}")
    return resolved
