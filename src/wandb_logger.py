# src/wandb_logger.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/wandb_logger.py
"""Weights & Biases integration helpers for dataset provenance logging.

Thin wrappers around :mod:`wandb` that keep every call site consistent:

* :func:`setup_wandb_auth` — one-shot authentication before any
  ``wandb.init`` call.
* :func:`load_artifact` — download a named Hub artifact to a local path.
* :func:`log_run_start` — open a run and record the full dataset
  provenance from :meth:`DatasetLoader.get_provenance` into both
  ``wandb.config`` and ``wandb.summary``.
* :func:`log_dataset_stats` — compute and log a pre-training telemetry
  snapshot (token-length histogram, court distribution, sample counts).
* :func:`log_quality_signals` — run :class:`ModelQualitySignals` checks
  over a sample and log the per-signal counts.

All :mod:`wandb` imports are lazy so importing this module does not
require ``wandb`` to be installed — useful for CI paths that never log.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from src.dataset_probe import ModelQualitySignals

if TYPE_CHECKING:
    from src.dataset_loader import DatasetLoader


def setup_wandb_auth() -> None:
    """Authenticate to W&B from environment, once per process.

    Precedence:

    1. ``WANDB_API_KEY`` env var → non-interactive login.
    2. ``WANDB_MODE`` is ``"offline"`` or ``"disabled"`` → no-op.
    3. Otherwise → attempt interactive login (will use a cached
       credential file if present).

    Call once before any :func:`wandb.init`; subsequent calls are
    cheap no-ops because ``relogin=False``.
    """
    import wandb

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=False)
    elif os.environ.get("WANDB_MODE") in ("offline", "disabled"):
        pass
    else:
        wandb.login(relogin=False)


def load_artifact(
    artifact_uri: str,
    local_path: str,
    project: str = "hallucination-legal-rag",
) -> str:
    """Download a W&B artifact and return its local root directory.

    Args:
        artifact_uri: Fully-qualified artifact identifier,
            e.g. ``"entity/project/name:version"``.
        local_path: Root directory for the download.
        project: Unused; retained for backward compatibility with
            older call sites that passed a project explicitly.

    Returns:
        The path returned by :meth:`Artifact.download`, which is
        ``local_path`` unless wandb resolves it differently.
    """
    import wandb

    api = wandb.Api()
    artifact = api.artifact(artifact_uri)
    return artifact.download(root=local_path)


def log_run_start(
    loader: "DatasetLoader",
    run_name: str | None = None,
    project: str = "hallucination-legal-rag",
    tags: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Initialise a W&B run and record full dataset provenance.

    Calls :meth:`DatasetLoader.get_provenance` and passes the result
    as ``wandb.config``, then mirrors the four identity fields
    (``dataset``, ``subset``, ``revision``, ``reproducible``) into
    ``wandb.summary`` for at-a-glance visibility on the run page.

    Args:
        loader: Configured dataset loader whose provenance will be
            recorded.
        run_name: Optional W&B run name; W&B auto-generates one if
            omitted.
        project: Target W&B project.
        tags: Optional list of tags applied to the run.
        extra: Extra key/value pairs merged into ``wandb.config``
            after the provenance dict, overriding on conflict.

    Returns:
        The initialised :class:`wandb.Run` object.
    """
    import wandb

    provenance = loader.get_provenance()
    config = {**provenance, **(extra or {})}
    run = wandb.init(
        project=project,
        name=run_name,
        tags=tags or [],
        config=config,
        reinit=True,
    )
    wandb.summary["data/dataset_id"] = provenance["dataset"]
    wandb.summary["data/subset"] = provenance["subset"]
    wandb.summary["data/revision"] = provenance["revision"]
    wandb.summary["data/reproducible"] = provenance["reproducible"]
    return run


def log_dataset_stats(
    loader: "DatasetLoader",
    tokenizer: Any,
    source: Any,
    max_samples: int = 1000,
) -> dict[str, Any]:
    """Compute and log a pre-training telemetry snapshot to the active run.

    Delegates aggregation to :meth:`DatasetLoader.log_stats` (so the
    stats definition stays in one place), then emits three W&B log
    payloads:

    1. Scalar summary metrics under ``data/*``.
    2. A ``wandb.plot.bar`` of the token-length histogram.
    3. A ``wandb.plot.bar`` of the per-court distribution.

    Args:
        loader: The loader whose validator/normalizer drive stats.
        tokenizer: Tokenizer forwarded to
            :meth:`DatasetLoader.log_stats`.
        source: Row iterable to sample.
        max_samples: Hard cap on rows consumed.

    Returns:
        The raw stats dict for callers that want to log additional
        derived metrics.
    """
    import wandb

    stats = loader.log_stats(source, tokenizer, max_samples=max_samples)
    wandb.log(
        {
            "data/n_valid_samples": stats["n_valid"],
            "data/n_skipped_samples": stats["n_skipped"],
            "data/avg_token_length": stats["avg_token_length"],
            "data/min_token_length": stats["min_token_length"],
            "data/max_token_length": stats["max_token_length"],
            "data/avg_text_length_chars": stats["avg_text_length_chars"],
            "data/tokenizer": stats["tokenizer_name"],
        }
    )
    if stats["token_length_histogram"]:
        table = wandb.Table(
            columns=["token_range", "count"],
            data=[[h["range"], h["count"]] for h in stats["token_length_histogram"]],
        )
        wandb.log(
            {
                "data/token_length_distribution": wandb.plot.bar(
                    table, "token_range", "count", title="Token Length Distribution"
                )
            }
        )
    if stats["court_distribution"]:
        court_table = wandb.Table(
            columns=["court_id", "count"],
            data=[[k, v] for k, v in stats["court_distribution"].items()],
        )
        wandb.log(
            {"data/court_distribution": wandb.plot.bar(court_table, "court_id", "count", title="Court Distribution")}
        )
    return stats


def log_quality_signals(
    rows: list[dict[str, Any]],
    sample_size: int = 100,
) -> dict[str, Any]:
    """Run :class:`ModelQualitySignals` over a sample and log the counts.

    Each signal that fires on any row contributes one to its counter;
    the full tally is logged under ``data/quality/<signal_name>`` so
    W&B renders a quality-signal panel per run.

    Args:
        rows: Row list to sample from the head.
        sample_size: Maximum rows to probe.

    Returns:
        Mapping of signal name → fire count across the sample.
    """
    import wandb

    signal_counts: dict[str, int] = {}
    for row in rows[:sample_size]:
        for signal_name, _ in ModelQualitySignals.check(row):
            signal_counts[signal_name] = signal_counts.get(signal_name, 0) + 1
    if signal_counts:
        wandb.log({f"data/quality/{k}": v for k, v in signal_counts.items()})
        wandb.log({"data/quality/n_rows_sampled": min(len(rows), sample_size)})
    return signal_counts
