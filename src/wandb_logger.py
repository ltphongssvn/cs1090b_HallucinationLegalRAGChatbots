# src/wandb_logger.py
# W&B experiment logging for the legal RAG pipeline.
# Call log_run_start() once at training start — never per-row.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.dataset_loader import DatasetLoader


def log_run_start(
    loader: "DatasetLoader",
    run_name: str | None = None,
    project: str = "hallucination-legal-rag",
    tags: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Initialize a W&B run and log dataset provenance.

    Call once at training start. Returns the wandb.Run object.

    Usage:
        run = log_run_start(loader, run_name="baseline-bert", tags=["train"])
        # ... training loop ...
        run.finish()
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
    """Compute and log dataset statistics to the active W&B run.

    Call after log_run_start(). Returns the stats dict.
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

    # Log token length histogram as W&B bar chart
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

    # Log court distribution
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
    """Log ModelQualitySignals summary to the active W&B run."""
    import wandb

    from src.dataset_probe import ModelQualitySignals

    signal_counts: dict[str, int] = {}
    for row in rows[:sample_size]:
        for signal_name, _ in ModelQualitySignals.check(row):
            signal_counts[signal_name] = signal_counts.get(signal_name, 0) + 1

    if signal_counts:
        wandb.log({f"data/quality/{k}": v for k, v in signal_counts.items()})
        wandb.log({"data/quality/n_rows_sampled": min(len(rows), sample_size)})

    return signal_counts


def setup_wandb_auth() -> None:
    """Configure W&B authentication from environment.
    Call once at process start before any wandb.init().
    Supports both WANDB_API_KEY env var and wandb login --relogin.
    """
    import os

    import wandb

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=False)
    elif os.environ.get("WANDB_MODE") in ("offline", "disabled"):
        pass  # no auth needed
    else:
        # Fall back to cached credentials from wandb login
        wandb.login(relogin=False)


def load_artifact(
    artifact_uri: str,
    local_path: str,
    project: str = "hallucination-legal-rag",
) -> str:
    """Download and return local path of a W&B artifact.

    Args:
        artifact_uri: W&B artifact URI e.g. 'entity/project/artifact:version'
        local_path: local directory to download artifact into
        project: W&B project name

    Returns:
        Local path to downloaded artifact directory.
    """
    import wandb

    api = wandb.Api()
    artifact = api.artifact(artifact_uri)
    return artifact.download(root=local_path)
