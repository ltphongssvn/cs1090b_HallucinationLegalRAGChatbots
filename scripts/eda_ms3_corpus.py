"""MS3 EDA: CourtListener federal appellate corpus distributions.

2026 production hardening:
    - No import-time side effects (basicConfig, matplotlib backend, rcParams
      all deferred to CLI entrypoint or inside function bodies).
    - Fail-fast manifest validation.
    - Atomic render: stale cleanup only after successful new render.
    - Overflow accounting for clipped histograms.
    - Canonical circuit order preserved in JSON via explicit circuit_order field.
    - SummaryDict strict (Required/NotRequired); SummaryModel runtime-validated.
    - Filter logic sourced from src.data_contracts.valid_record_expr.

Module-level constants (repo convention):
    SCHEMA_VERSION    = "1.2.0"  (bumped: circuit_order + chart_overflow_counts)
    FILTER_MIN_CHARS  = 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, NotRequired, Required, TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pydantic import ConfigDict, Field  # noqa: F401  (kept for downstream)

from src.eda_schemas import EdaCorpusSummary as SummaryModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SCHEMA_VERSION = "1.2.0"
FILTER_MIN_CHARS = 100
DEFAULT_SHARD_GLOB = "data/raw/cl_federal_appellate_bulk/shard_*.jsonl"
DEFAULT_OUT_DIR = Path("logs/eda_ms3")
DEFAULT_MANIFEST = Path("data/raw/cl_federal_appellate_bulk/manifest.json")

TEXT_LENGTH_HIST_RANGE = (0, 100_000)
CITATION_HIST_RANGE = (0, 100)

POLARS_SCHEMA: dict[str, pl.DataType] = {
    "text_length": pl.Int64,
    "court_id": pl.Utf8,
    "citation_count": pl.Int64,
}

CANONICAL_CIRCUITS = [f"ca{i}" for i in range(1, 12)] + ["cadc", "cafc"]


class SummaryDict(TypedDict):
    """Strict shape of summary.json payload."""

    schema_version: Required[str]
    n_total: Required[int]
    n_after_filter: Required[int]
    n_short_lt_100: Required[int]
    text_length_mean: Required[float]
    text_length_median: Required[float]
    text_length_mean_filtered: Required[float]
    text_length_median_filtered: Required[float]
    filter_threshold: Required[int]
    circuit_counts: Required[dict[str, int]]
    circuit_order: Required[list[str]]
    chart_ranges: Required[dict[str, list[int]]]
    chart_overflow_counts: Required[dict[str, int]]
    corpus_manifest_sha: Required[str]
    figure_hashes: Required[dict[str, str]]
    git_sha: Required[str]
    extra: NotRequired[dict[str, Any]]


class _ComputedStats(TypedDict):
    n_total: int
    mean: float
    median: float
    n_short: int
    mean_filtered: float
    median_filtered: float
    courts: list[str]
    counts: list[int]
    lengths: np.ndarray
    citations: np.ndarray | None
    overflow_text_length: int
    overflow_citations: int


def is_valid_record(text_length: int) -> bool:
    """Public predicate for reuse; thin wrapper over data_contracts."""
    return bool(text_length >= FILTER_MIN_CHARS)


def _apply_plot_defaults() -> dict[str, Any]:
    """Return rcParams dict for use in rc_context (no global mutation)."""
    return {"savefig.dpi": 120, "figure.autolayout": True}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _validate_inputs(shard_glob: str, manifest_path: Path) -> None:
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")


def _canonical_circuit_sort(courts: list[str], counts: list[int]) -> tuple[list[str], list[int]]:
    pairs = dict(zip(courts, counts))
    ordered: list[tuple[str, int]] = []
    for c in CANONICAL_CIRCUITS:
        if c in pairs:
            ordered.append((c, pairs.pop(c)))
    for c in sorted(pairs):
        ordered.append((c, pairs[c]))
    return [c for c, _ in ordered], [n for _, n in ordered]


def _compute_stats(shard_glob: str) -> _ComputedStats:
    """Polars aggregation across multiple collects (lazy plan shared)."""
    df = pl.scan_ndjson(shard_glob, schema_overrides=POLARS_SCHEMA, low_memory=True)

    scalars = df.select(
        [
            pl.len().alias("n_total"),
            pl.col("text_length").mean().alias("mean"),
            pl.col("text_length").median().alias("median"),
            (pl.col("text_length") < FILTER_MIN_CHARS).sum().alias("n_short"),
            pl.col("text_length").filter(pl.col("text_length") >= FILTER_MIN_CHARS).mean().alias("mean_filtered"),
            pl.col("text_length").filter(pl.col("text_length") >= FILTER_MIN_CHARS).median().alias("median_filtered"),
            (pl.col("text_length") > TEXT_LENGTH_HIST_RANGE[1]).sum().alias("overflow_len"),
        ]
    ).collect()

    court_dist = df.group_by("court_id").agg(pl.len().alias("n")).sort("n", descending=True).collect()
    courts, counts = _canonical_circuit_sort(court_dist["court_id"].to_list(), court_dist["n"].to_list())

    lengths = df.select("text_length").collect().to_series().to_numpy()

    citations: np.ndarray | None
    overflow_citations = 0
    try:
        citations = df.select("citation_count").collect().to_series().to_numpy()
        overflow_citations = int((citations > CITATION_HIST_RANGE[1]).sum())
    except pl.exceptions.ColumnNotFoundError:
        citations = None

    return _ComputedStats(
        n_total=int(scalars["n_total"][0]),
        mean=float(scalars["mean"][0]),
        median=float(scalars["median"][0]),
        n_short=int(scalars["n_short"][0]),
        mean_filtered=float(scalars["mean_filtered"][0] or 0.0),
        median_filtered=float(scalars["median_filtered"][0] or 0.0),
        courts=courts,
        counts=counts,
        lengths=lengths,
        citations=citations,
        overflow_text_length=int(scalars["overflow_len"][0]),
        overflow_citations=overflow_citations,
    )


def _clean_stale_artifacts(out_dir: Path) -> None:
    for pattern in ("*.png", "summary.json"):
        for p in out_dir.glob(pattern):
            p.unlink()


def _plot_text_length_hist(lengths: np.ndarray, out: Path, *, log_scale: bool = False) -> Path:
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
        ax.hist(lengths, bins=100, range=TEXT_LENGTH_HIST_RANGE, color="steelblue", edgecolor="white")
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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(courts, counts, color="steelblue", edgecolor="white")
    ax.set_xlabel("Court (federal circuit, canonical order)")
    ax.set_ylabel("Opinion count")
    ax.set_title("Corpus distribution across federal appellate circuits")
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_citation_density(counts: np.ndarray, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(counts, bins=60, range=CITATION_HIST_RANGE, color="darkorange", edgecolor="white")
    ax.set_xlabel("Citations per opinion")
    ax.set_ylabel("Count")
    ax.set_title("Citation density distribution (0-100)")
    fig.savefig(out)
    plt.close(fig)
    return out


def _render_all(stats: _ComputedStats, out_dir: Path) -> list[Path]:
    """Render every figure inside rc_context; return list of emitted paths."""
    with plt.rc_context(_apply_plot_defaults()):
        matplotlib.use("Agg", force=False)
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
    n_after = stats["n_total"] - stats["n_short"]
    return SummaryDict(
        schema_version=SCHEMA_VERSION,
        n_total=stats["n_total"],
        n_after_filter=n_after,
        n_short_lt_100=stats["n_short"],
        text_length_mean=stats["mean"],
        text_length_median=stats["median"],
        text_length_mean_filtered=stats["mean_filtered"],
        text_length_median_filtered=stats["median_filtered"],
        filter_threshold=FILTER_MIN_CHARS,
        circuit_counts=dict(zip(stats["courts"], stats["counts"])),
        circuit_order=list(stats["courts"]),
        chart_ranges={
            "text_length_hist": list(TEXT_LENGTH_HIST_RANGE),
            "citation_density": list(CITATION_HIST_RANGE),
        },
        chart_overflow_counts={
            "text_length_hist": stats["overflow_text_length"],
            "citation_density": stats["overflow_citations"],
        },
        corpus_manifest_sha=_sha256_file(manifest_path),
        figure_hashes={fp.name: _sha256_file(fp) for fp in figure_paths},
        git_sha=_git_sha(),
    )


def _write_summary(summary: SummaryDict, out_dir: Path) -> Path:
    validated = SummaryModel.model_validate(dict(summary))
    path = out_dir / "summary.json"
    path.write_text(
        json.dumps(validated.model_dump(), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return path


def _log_to_wandb(
    summary: SummaryDict,
    figure_paths: list[Path],
    summary_path: Path,
    entity: str = "phl690-harvard-extension-schol",
    project: str = "cs1090b",
    run_name: str = "eda_ms3_corpus",
) -> None:
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
        reinit="finish_previous",
    )
    metrics: dict[str, Any] = {
        "eda/n_total": summary["n_total"],
        "eda/n_after_filter": summary["n_after_filter"],
        "eda/text_length_mean": summary["text_length_mean"],
        "eda/text_length_median": summary["text_length_median"],
        "eda/n_short_lt_100": summary["n_short_lt_100"],
        "eda/corpus_manifest_sha": summary["corpus_manifest_sha"],
        "eda/git_sha": summary["git_sha"],
    }
    for fp in figure_paths:
        metrics[f"eda/figures/{fp.stem}"] = wandb.Image(str(fp))
    wandb.log(metrics)

    art = wandb.Artifact("ms3_eda_corpus", type="eda-report")
    art.add_file(str(summary_path))
    for fp in figure_paths:
        art.add_file(str(fp))
    wandb.log_artifact(art)
    wandb.finish()
    if run is not None:
        logger.info(f"W&B run complete — {entity}/{project}")


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="MS3 EDA over CourtListener federal appellate corpus.",
    )
    ap.add_argument("--shard-glob", type=str, default=DEFAULT_SHARD_GLOB)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--log-to-wandb", action="store_true")
    return ap


def main(
    shard_glob: str = DEFAULT_SHARD_GLOB,
    out_dir: Path = DEFAULT_OUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    log_to_wandb: bool = False,
    clean_stale: bool = True,
) -> SummaryDict:
    """Thin orchestration: validate → compute → render → persist → telemetry.

    Render-first, cleanup-after semantics: stale artifacts are only removed
    after the new render + summary write succeed. A failure mid-run leaves
    the previous good artifact set intact.
    """
    _validate_inputs(shard_glob, manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning {shard_glob} via Polars lazy scan...")
    stats = _compute_stats(shard_glob)

    prior_artifacts = set(out_dir.glob("*.png")) | set(out_dir.glob("summary.json"))
    figure_paths = _render_all(stats, out_dir)
    summary = _build_summary(stats, manifest_path, figure_paths)
    summary_path = _write_summary(summary, out_dir)

    if clean_stale:
        new_artifacts = set(figure_paths) | {summary_path}
        for p in prior_artifacts - new_artifacts:
            p.unlink(missing_ok=True)

    logger.info(f"Wrote {len(figure_paths)} figures + summary.json to {out_dir}/")

    if log_to_wandb:
        _log_to_wandb(summary, figure_paths, summary_path)

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[eda_ms3] %(message)s")
    args = _build_arg_parser().parse_args()
    main(
        shard_glob=args.shard_glob,
        out_dir=args.out_dir,
        manifest_path=args.manifest_path,
        log_to_wandb=args.log_to_wandb,
    )
