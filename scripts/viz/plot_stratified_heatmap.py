#!/usr/bin/env python3
"""Generate stratified retrieval heatmap (6 retrievers x 3 buckets x 6 metrics).

Reads *_results.stratified.json from data/processed/baseline/cleaned/ and
emits a multi-panel heatmap PNG to artifacts/.

Usage:
    .venv/bin/python scripts/viz/plot_stratified_heatmap.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("plot_stratified_heatmap")

DEFAULT_RETRIEVERS = [
    ("bm25", "data/processed/baseline/cleaned/bm25_results.stratified.json"),
    ("bge_m3", "data/processed/baseline/cleaned/bge_m3_results.stratified.json"),
    ("rrf", "data/processed/baseline/cleaned/rrf_results.stratified.json"),
    ("reranker_concat", "data/processed/baseline/cleaned/reranker_results.stratified.json"),
    ("reranker_maxp", "data/processed/baseline/cleaned/maxp/reranker_results.stratified.json"),
    ("reranker_finetuned", "data/processed/baseline/cleaned/finetuned/reranker_results.stratified.json"),
]
BUCKETS = ["head", "torso", "tail"]
METRICS = [
    ("hit_at_k", "1", "Hit@1"),
    ("hit_at_k", "5", "Hit@5"),
    ("hit_at_k", "10", "Hit@10"),
    ("hit_at_k", "100", "Hit@100"),
    ("mrr", None, "MRR"),
    ("ndcg_at_10", None, "NDCG@10"),
]


def _get_metric(bucket_d: dict, metric_key: str, sub_key: str | None) -> float:
    val = bucket_d.get(metric_key)
    if val is None:
        return float("nan")
    if isinstance(val, dict) and sub_key is not None:
        return float(val.get(sub_key, val.get(int(sub_key), float("nan"))))
    return float(val)


def _build_matrix(
    retrievers: list[tuple[str, str]],
    metric_key: str,
    sub_key: str | None,
) -> tuple[np.ndarray, list[str]]:
    available_labels = []
    rows = []
    for label, path in retrievers:
        p = Path(path)
        if not p.is_file():
            logger.info(f"  skip {label}: {p} missing")
            continue
        d = json.loads(p.read_text())
        per_bucket = d.get("per_bucket", {})
        row = [_get_metric(per_bucket.get(b, {}), metric_key, sub_key) for b in BUCKETS]
        rows.append(row)
        available_labels.append(label)
    if not rows:
        raise RuntimeError("no stratified files found")
    return np.array(rows), available_labels


def plot_heatmap(retrievers: list[tuple[str, str]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, (metric_key, sub_key, title) in enumerate(METRICS):
        ax = axes[i]
        mat, labels = _build_matrix(retrievers, metric_key, sub_key)
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0.0)
        ax.set_xticks(range(len(BUCKETS)))
        ax.set_xticklabels([b.upper() for b in BUCKETS])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                if np.isnan(v):
                    ax.text(c, r, "-", ha="center", va="center", fontsize=9)
                else:
                    rgba = im.cmap(im.norm(v))
                    txt_color = "white" if (rgba[0] + rgba[1] + rgba[2]) / 3 < 0.5 else "black"
                    ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, color=txt_color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "MS4 Stratified Retrieval Evaluation: Retriever x Citation-Frequency Bucket\n"
        "HEAD = top tertile (most-cited); TAIL = bottom tertile (rarest)",
        fontsize=13, fontweight="bold", y=1.00,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-path", type=Path,
                    default=Path("artifacts/ms4_stratified_heatmap.png"))
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[plot_stratified_heatmap] %(message)s",
        stream=sys.stdout,
    )
    plot_heatmap(DEFAULT_RETRIEVERS, args.out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
