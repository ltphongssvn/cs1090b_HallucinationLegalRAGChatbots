"""MS3 EDA: LePaRD × CourtListener compatibility.

Thin wrapper over src/lepard_cl_compat.run_full_analysis() that emits
MS3-specific visualizations + provenance-enriched summary.json.

Artifacts (written to out_dir):
    - pair_funnel.png         (total_rows → unique_pairs → both_in_cl)
    - court_distribution.png  (matched IDs per federal circuit)
    - id_overlap.png          (LePaRD ∩ CL id-space visualization)
    - summary.json            (LepardSummarySchema-conformant)

Module-level constants (repo convention):
    SCHEMA_VERSION = "1.0.0"

W&B telemetry gated by log_to_wandb; matches dataset_probe isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from pydantic import ConfigDict, Field  # noqa: F401

from src.eda_schemas import EdaLepardSummary as SummaryModel
from src.lepard_cl_compat import (
    DEFAULT_CL_IDS,
    DEFAULT_COURT_MAP,
    DEFAULT_LEPARD,
    run_full_analysis,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SCHEMA_VERSION = "1.0.0"
DEFAULT_OUT_DIR = Path("logs/eda_ms3_lepard")


def _apply_plot_defaults() -> dict[str, Any]:
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


def _validate_inputs(lepard_path: Path, cl_ids_path: Path, court_map_path: Path) -> None:
    for label, p in [("lepard", lepard_path), ("cl_ids", cl_ids_path), ("court_map", court_map_path)]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} path not found: {p}")


def _plot_pair_funnel(total: int, unique: int, both: int, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["total_rows", "unique_pairs", "both_in_cl\n(usable gold)"]
    values = [total, unique, both]
    colors = ["steelblue", "cornflowerblue", "seagreen"]
    ax.bar(labels, values, color=colors, edgecolor="white")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("LePaRD → CourtListener pair-usability funnel")
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_court_distribution(court_dist: dict[str, int], out: Path) -> Path:
    if not court_dist:
        court_dist = {"(none)": 0}
    items = sorted(court_dist.items(), key=lambda kv: kv[1], reverse=True)
    courts = [k for k, _ in items]
    counts = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(courts))))
    ax.barh(courts, counts, color="darkorange", edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Matched CL opinions")
    ax.set_title("Court distribution of LePaRD↔CL matched IDs")
    for i, v in enumerate(counts):
        ax.text(v, i, f" {v:,}", va="center", fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_id_overlap(lepard_ids: int, cl_ids: int, overlap: int, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["LePaRD\nonly", "Overlap", "CL only (sampled)"]
    values = [lepard_ids - overlap, overlap, max(0, cl_ids - overlap)]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    ax.bar(labels, values, color=colors, edgecolor="white")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Unique ID count")
    ax.set_title(f"ID-space overlap  ({overlap:,} shared of {lepard_ids:,} LePaRD IDs)")
    ax.set_yscale("log")
    fig.savefig(out)
    plt.close(fig)
    return out


def _render_all(report, out_dir: Path) -> list[Path]:
    with plt.rc_context(_apply_plot_defaults()):
        matplotlib.use("Agg", force=False)
        paths = [
            _plot_pair_funnel(
                report.pair_overlap.total_rows,
                report.pair_overlap.unique_pairs,
                report.pair_overlap.both_in_cl,
                out_dir / "pair_funnel.png",
            ),
            _plot_court_distribution(dict(report.court_distribution), out_dir / "court_distribution.png"),
            _plot_id_overlap(
                report.id_overlap.lepard_unique_ids,
                report.id_overlap.cl_unique_ids,
                report.id_overlap.overlap,
                out_dir / "id_overlap.png",
            ),
        ]
    return paths


def _build_summary(report, figure_paths: list[Path]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "total_rows": report.pair_overlap.total_rows,
        "unique_pairs": report.pair_overlap.unique_pairs,
        "lepard_unique_ids": report.id_overlap.lepard_unique_ids,
        "cl_unique_ids": report.id_overlap.cl_unique_ids,
        "overlap_ids": report.id_overlap.overlap,
        "both_in_cl": report.pair_overlap.both_in_cl,
        "source_only": report.pair_overlap.source_only_in_cl,
        "dest_only": report.pair_overlap.dest_only_in_cl,
        "neither": report.pair_overlap.neither_in_cl,
        "usable_pct": report.pair_overlap.usable_pct,
        "court_distribution": dict(report.court_distribution),
        "figure_hashes": {fp.name: _sha256_file(fp) for fp in figure_paths},
        "git_sha": _git_sha(),
    }


def _write_summary(summary: dict[str, Any], out_dir: Path) -> Path:
    validated = SummaryModel.model_validate(summary)
    path = out_dir / "summary.json"
    path.write_text(
        json.dumps(validated.model_dump(), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return path


def _log_to_wandb(
    summary: dict[str, Any],
    figure_paths: list[Path],
    summary_path: Path,
    entity: str = "phl690-harvard-extension-schol",
    project: str = "cs1090b",
    run_name: str = "eda_ms3_lepard",
) -> None:
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping")
        return

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        job_type="eda",
        config={"schema_version": summary["schema_version"]},
        reinit="finish_previous",
    )
    metrics: dict[str, Any] = {
        "lepard/total_rows": summary["total_rows"],
        "lepard/unique_pairs": summary["unique_pairs"],
        "lepard/both_in_cl": summary["both_in_cl"],
        "lepard/usable_pct": summary["usable_pct"],
        "lepard/overlap_ids": summary["overlap_ids"],
        "lepard/git_sha": summary["git_sha"],
    }
    for fp in figure_paths:
        metrics[f"lepard/figures/{fp.stem}"] = wandb.Image(str(fp))
    wandb.log(metrics)

    art = wandb.Artifact("ms3_eda_lepard", type="eda-report")
    art.add_file(str(summary_path))
    for fp in figure_paths:
        art.add_file(str(fp))
    wandb.log_artifact(art)
    wandb.finish()
    if run is not None:
        logger.info(f"W&B run complete — {entity}/{project}")


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MS3 LePaRD × CourtListener compatibility EDA.")
    ap.add_argument("--lepard-path", type=Path, default=DEFAULT_LEPARD)
    ap.add_argument("--cl-ids-path", type=Path, default=DEFAULT_CL_IDS)
    ap.add_argument("--court-map-path", type=Path, default=DEFAULT_COURT_MAP)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--log-to-wandb", action="store_true")
    return ap


def main(
    *,
    lepard_path: Path = DEFAULT_LEPARD,
    cl_ids_path: Path = DEFAULT_CL_IDS,
    court_map_path: Path = DEFAULT_COURT_MAP,
    out_dir: Path = DEFAULT_OUT_DIR,
    log_to_wandb: bool = False,
) -> dict[str, Any]:
    """Thin orchestration: validate → analyze → render → persist → telemetry."""
    _validate_inputs(lepard_path, cl_ids_path, court_map_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running LePaRD↔CL analysis on {lepard_path.name}")
    report = run_full_analysis(
        lepard_path=lepard_path,
        cl_ids_path=cl_ids_path,
        court_map_path=court_map_path,
    )

    figure_paths = _render_all(report, out_dir)
    summary = _build_summary(report, figure_paths)
    summary_path = _write_summary(summary, out_dir)
    logger.info(f"Wrote {len(figure_paths)} figures + summary.json to {out_dir}/")

    if log_to_wandb:
        _log_to_wandb(summary, figure_paths, summary_path)

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[eda_ms3_lepard] %(message)s")
    args = _build_arg_parser().parse_args()
    main(
        lepard_path=args.lepard_path,
        cl_ids_path=args.cl_ids_path,
        court_map_path=args.court_map_path,
        out_dir=args.out_dir,
        log_to_wandb=args.log_to_wandb,
    )
