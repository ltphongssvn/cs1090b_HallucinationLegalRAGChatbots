"""MS3 EDA: CourtListener federal appellate corpus distributions.

Produces visualizations and summary statistics for MS3 notebook/slides.

Artifacts (written to out_dir):
    - text_length_hist.png : linear-scale histogram with filter threshold line
    - text_length_hist_log.png : log-log histogram for tail inspection
    - circuit_distribution.png : bar chart across federal appellate circuits
    - citation_density.png : citation count histogram (if column present)
    - summary.json : SummarySchema-conformant stats + SHA256 provenance

Provenance fields in summary.json:
    - corpus_manifest_sha : SHA256 of the input manifest.json
    - figure_hashes : {fname: sha256} for every emitted PNG

W&B telemetry:
    Gated by log_to_wandb flag (matches src/dataset_probe.py pattern).
    Entity/project: phl690-harvard-extension-schol / cs1090b.
    Single wandb.log call + Artifact upload (isolation contract).

Module-level constants (repo convention; no Hydra in stack):
    FILTER_MIN_CHARS = 100 -- mirrors data_contracts.py p5 threshold
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

SCHEMA_VERSION = "1.0.0"
FILTER_MIN_CHARS = 100
DEFAULT_SHARD_GLOB = "data/raw/cl_federal_appellate_bulk/shard_*.jsonl"
DEFAULT_OUT_DIR = Path("logs/eda_ms3")
DEFAULT_MANIFEST = Path("data/raw/cl_federal_appellate_bulk/manifest.json")

plt.rcParams["savefig.dpi"] = 120
plt.rcParams["figure.autolayout"] = True
np.random.seed(0)


def is_valid_record(text_length: int) -> bool:
    """Pure predicate: record passes short-record filter iff length >= FILTER_MIN_CHARS."""
    return bool(text_length >= FILTER_MIN_CHARS)


def _sha256_file(path: Path) -> str:
    """SHA256 of a file's bytes for provenance recording."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _log_to_wandb(
    summary: dict[str, Any],
    figure_paths: list[Path],
    entity: str = "phl690-harvard-extension-schol",
    project: str = "cs1090b",
    run_name: str = "eda_ms3_corpus",
) -> None:
    """Log EDA summary + figures to W&B (isolated per repo contract).

    Matches src/dataset_probe.py::_log_report_to_wandb:
    exactly one wandb.log call + one Artifact upload. Lazy import.
    """
    try:
        import wandb
    except ImportError:
        print("[eda_ms3] wandb not installed — skipping W&B logging")
        return

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        job_type="eda",
        config={"filter_threshold": summary["filter_threshold"]},
        reinit=True,
    )
    metrics: dict[str, Any] = {
        "eda/n_total": summary["n_total"],
        "eda/text_length_mean": summary["text_length_mean"],
        "eda/text_length_median": summary["text_length_median"],
        "eda/n_short_lt_100": summary["n_short_lt_100"],
        "eda/corpus_manifest_sha": summary["corpus_manifest_sha"],
    }
    for fp in figure_paths:
        metrics[f"eda/figures/{fp.stem}"] = wandb.Image(str(fp))
    wandb.log(metrics)

    art = wandb.Artifact("ms3_eda_corpus", type="eda-report")
    for fp in figure_paths:
        art.add_file(str(fp))
    wandb.log_artifact(art)
    wandb.finish()
    if run is not None:
        print(f"[eda_ms3] W&B run complete — {entity}/{project}")


def _plot_text_length_hist(lengths: np.ndarray, out: Path, log_scale: bool = False) -> Path:
    """Text-length histogram with the filter threshold marked."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if log_scale:
        pos = lengths[lengths > 0]
        ax.hist(pos, bins=100, color="steelblue", edgecolor="white")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Text length (chars, log)")
        ax.set_ylabel("Count (log)")
        ax.set_title("Text length distribution (log-log)")
    else:
        ax.hist(lengths, bins=100, range=(0, 100_000), color="steelblue", edgecolor="white")
        ax.set_xlabel("Text length (characters)")
        ax.set_ylabel("Count")
        ax.set_title(f"CourtListener federal appellate: text length (N={len(lengths):,})")
    ax.axvline(
        FILTER_MIN_CHARS,
        color="red",
        linestyle="--",
        label=f"filter threshold = {FILTER_MIN_CHARS}",
    )
    ax.legend()
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_circuit_distribution(courts: list[str], counts: list[int], out: Path) -> Path:
    """Bar chart: opinion count per federal circuit."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(courts, counts, color="steelblue", edgecolor="white")
    ax.set_xlabel("Court (federal circuit)")
    ax.set_ylabel("Opinion count")
    ax.set_title("Corpus distribution across federal appellate circuits")
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_citation_density(counts: np.ndarray, out: Path) -> Path:
    """Histogram of citations per opinion, clipped to [0, 100]."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(counts, bins=60, range=(0, 100), color="darkorange", edgecolor="white")
    ax.set_xlabel("Citations per opinion")
    ax.set_ylabel("Count")
    ax.set_title("Citation density distribution (0-100)")
    fig.savefig(out)
    plt.close(fig)
    return out


def main(
    shard_glob: str = DEFAULT_SHARD_GLOB,
    out_dir: Path = DEFAULT_OUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    log_to_wandb: bool = False,
) -> dict[str, Any]:
    """Scan corpus, emit figures + summary.json, optionally log to W&B.

    Args:
        shard_glob: Polars glob for JSONL shards.
        out_dir: Directory for PNG + JSON artifacts (created if missing).
        manifest_path: Source manifest.json used to derive corpus_manifest_sha.
        log_to_wandb: Gate for W&B telemetry (default False for CI isolation).

    Returns:
        Summary dict conforming to SummarySchema in tests/test_eda_ms3_corpus.py.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eda_ms3] Scanning {shard_glob} via Polars lazy scan...")
    df = pl.scan_ndjson(shard_glob, low_memory=True)

    stats = df.select(
        [
            pl.len().alias("n_total"),
            pl.col("text_length").mean().alias("mean"),
            pl.col("text_length").median().alias("median"),
            (pl.col("text_length") < FILTER_MIN_CHARS).sum().alias("n_short"),
        ]
    ).collect()

    court_dist = df.group_by("court_id").agg(pl.len().alias("n")).sort("n", descending=True).collect()

    lengths = df.select("text_length").collect().to_series().to_numpy()
    courts = court_dist["court_id"].to_list()
    counts = court_dist["n"].to_list()

    figure_paths: list[Path] = []
    figure_paths.append(_plot_text_length_hist(lengths, out_dir / "text_length_hist.png"))
    figure_paths.append(_plot_text_length_hist(lengths, out_dir / "text_length_hist_log.png", log_scale=True))
    figure_paths.append(_plot_circuit_distribution(courts, counts, out_dir / "circuit_distribution.png"))

    try:
        cit = df.select("citation_count").collect().to_series().to_numpy()
        figure_paths.append(_plot_citation_density(cit, out_dir / "citation_density.png"))
    except Exception as e:
        print(f"[eda_ms3] citation_count plot skipped: {e}")

    corpus_sha = _sha256_file(manifest_path)
    figure_hashes = {fp.name: _sha256_file(fp) for fp in figure_paths}

    summary: dict[str, Any] = {
        "n_total": int(stats["n_total"][0]),
        "text_length_mean": float(stats["mean"][0]),
        "text_length_median": float(stats["median"][0]),
        "n_short_lt_100": int(stats["n_short"][0]),
        "schema_version": SCHEMA_VERSION,
        "filter_threshold": FILTER_MIN_CHARS,
        "circuit_counts": dict(zip(courts, counts)),
        "corpus_manifest_sha": corpus_sha,
        "figure_hashes": figure_hashes,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[eda_ms3] Wrote {len(figure_paths)} figures + summary.json to {out_dir}/")

    if log_to_wandb:
        _log_to_wandb(summary, figure_paths)

    return summary


if __name__ == "__main__":
    main()
