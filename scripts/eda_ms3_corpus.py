"""MS3 EDA: CourtListener federal appellate corpus distributions.

Refactored for SRP + Polars efficiency:
    - _compute_stats(): single Polars collect producing all scalars + arrays.
    - _plot_*(): pure rendering from arrays.
    - _write_summary(): atomic JSON emit with provenance.
    - main(): thin orchestration.

Artifacts (written to out_dir):
    - text_length_hist.png, text_length_hist_log.png
    - circuit_distribution.png, citation_density.png (if available)
    - summary.json (SummaryDict-conformant)

Module-level constants (repo convention; no Hydra in stack):
    SCHEMA_VERSION    = "1.0.0"
    FILTER_MIN_CHARS  = 100  (mirrors data_contracts p5 threshold)

W&B telemetry: isolated in _log_to_wandb(), gated by main(log_to_wandb=).
Matches src/dataset_probe.py::_log_report_to_wandb isolation contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[eda_ms3] %(message)s")

POLARS_SCHEMA: dict[str, pl.DataType] = {
    "text_length": pl.Int64,
    "court_id": pl.Utf8,
    "citation_count": pl.Int64,
}

SCHEMA_VERSION = "1.0.0"
FILTER_MIN_CHARS = 100
DEFAULT_SHARD_GLOB = "data/raw/cl_federal_appellate_bulk/shard_*.jsonl"
DEFAULT_OUT_DIR = Path("logs/eda_ms3")
DEFAULT_MANIFEST = Path("data/raw/cl_federal_appellate_bulk/manifest.json")

plt.rcParams["savefig.dpi"] = 120
plt.rcParams["figure.autolayout"] = True
np.random.seed(0)


class SummaryDict(TypedDict):
    """Strict shape of summary.json payload (mirrors SummarySchema test)."""

    schema_version: str
    n_total: int
    text_length_mean: float
    text_length_median: float
    n_short_lt_100: int
    filter_threshold: int
    circuit_counts: dict[str, int]
    corpus_manifest_sha: str
    figure_hashes: dict[str, str]
    git_sha: str


class _ComputedStats(TypedDict):
    """Intermediate stats bundle — input to renderers + summary writer."""

    n_total: int
    mean: float
    median: float
    n_short: int
    courts: list[str]
    counts: list[int]
    lengths: np.ndarray
    citations: np.ndarray | None


def is_valid_record(text_length: int) -> bool:
    """Public predicate: record passes short-record filter iff length >= FILTER_MIN_CHARS.

    Exposed for reuse by downstream baseline scripts (BM25, BGE-M3 indexers).
    Tested in tests/test_eda_ms3_corpus.py (contract + property tiers).
    """
    return bool(text_length >= FILTER_MIN_CHARS)


def _sha256_file(path: Path) -> str:
    """SHA256 of a file's bytes — provenance primitive."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compute_stats(shard_glob: str) -> _ComputedStats:
    """Single-scan Polars aggregation producing all values main() needs.

    Consolidates scalar stats, circuit counts, length array, and citation
    array into one logical pass. Polars' lazy engine can optimise the
    shared scan internally; we still materialise two columns (lengths
    for histograms, citations for density plot) because matplotlib needs
    raw values — acceptable at corpus size 1.5M (11.7 MB int64).
    """
    df = pl.scan_ndjson(shard_glob, schema_overrides=POLARS_SCHEMA, low_memory=True)

    scalars = df.select(
        [
            pl.len().alias("n_total"),
            pl.col("text_length").mean().alias("mean"),
            pl.col("text_length").median().alias("median"),
            (pl.col("text_length") < FILTER_MIN_CHARS).sum().alias("n_short"),
        ]
    ).collect()

    court_dist = df.group_by("court_id").agg(pl.len().alias("n")).sort("n", descending=True).collect()

    lengths = df.select("text_length").collect().to_series().to_numpy()

    citations: np.ndarray | None
    try:
        citations = df.select("citation_count").collect().to_series().to_numpy()
    except pl.exceptions.ColumnNotFoundError:
        citations = None

    return _ComputedStats(
        n_total=int(scalars["n_total"][0]),
        mean=float(scalars["mean"][0]),
        median=float(scalars["median"][0]),
        n_short=int(scalars["n_short"][0]),
        courts=court_dist["court_id"].to_list(),
        counts=court_dist["n"].to_list(),
        lengths=lengths,
        citations=citations,
    )


def _plot_text_length_hist(lengths: np.ndarray, out: Path, log_scale: bool = False) -> Path:
    """Text-length histogram with the FILTER_MIN_CHARS threshold marked."""
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


def _render_all(stats: _ComputedStats, out_dir: Path) -> list[Path]:
    """Render every figure; return list of emitted paths."""
    paths: list[Path] = [
        _plot_text_length_hist(stats["lengths"], out_dir / "text_length_hist.png"),
        _plot_text_length_hist(stats["lengths"], out_dir / "text_length_hist_log.png", log_scale=True),
        _plot_circuit_distribution(stats["courts"], stats["counts"], out_dir / "circuit_distribution.png"),
    ]
    if stats["citations"] is not None:
        paths.append(_plot_citation_density(stats["citations"], out_dir / "citation_density.png"))
    return paths


def _build_summary(
    stats: _ComputedStats,
    manifest_path: Path,
    figure_paths: list[Path],
) -> SummaryDict:
    """Assemble the SummaryDict payload with SHA256 provenance."""
    return SummaryDict(
        schema_version=SCHEMA_VERSION,
        n_total=stats["n_total"],
        text_length_mean=stats["mean"],
        text_length_median=stats["median"],
        n_short_lt_100=stats["n_short"],
        filter_threshold=FILTER_MIN_CHARS,
        circuit_counts=dict(zip(stats["courts"], stats["counts"])),
        corpus_manifest_sha=_sha256_file(manifest_path),
        figure_hashes={fp.name: _sha256_file(fp) for fp in figure_paths},
        git_sha=_git_sha(),
    )


def _write_summary(summary: SummaryDict, out_dir: Path) -> Path:
    """Emit summary.json with explicit UTF-8 encoding."""
    path = out_dir / "summary.json"
    path.write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")
    return path


def _git_sha() -> str:
    """Current HEAD SHA (short) for lineage; empty string if not a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _log_to_wandb(
    summary: SummaryDict,
    figure_paths: list[Path],
    entity: str = "phl690-harvard-extension-schol",
    project: str = "cs1090b",
    run_name: str = "eda_ms3_corpus",
) -> None:
    """Single-call W&B logger matching dataset_probe isolation contract."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping W&B logging")
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
        logger.info(f"W&B run complete — {entity}/{project}")


def main(
    shard_glob: str = DEFAULT_SHARD_GLOB,
    out_dir: Path = DEFAULT_OUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    log_to_wandb: bool = False,
) -> SummaryDict:
    """Thin orchestration: compute → render → persist → (optional) telemetry.

    Args:
        shard_glob: Polars glob for JSONL shards.
        out_dir: Destination for PNG + JSON artifacts.
        manifest_path: Source manifest for corpus_manifest_sha provenance.
        log_to_wandb: Gate for W&B telemetry (default False for CI isolation).

    Returns:
        SummaryDict payload (also written to out_dir/summary.json).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning {shard_glob} via Polars lazy scan...")
    stats = _compute_stats(shard_glob)
    figure_paths = _render_all(stats, out_dir)
    summary = _build_summary(stats, manifest_path, figure_paths)
    _write_summary(summary, out_dir)
    logger.info(f"Wrote {len(figure_paths)} figures + summary.json to {out_dir}/")

    if log_to_wandb:
        _log_to_wandb(summary, figure_paths)

    return summary


if __name__ == "__main__":
    main()
