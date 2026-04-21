"""Pipeline DAG renderer for Cell 16 MS3 deliverable.

Dict-based DSL describes a retrieval pipeline as stages + edges. Pure
matplotlib renders to PNG for inline notebook display.

Two canonical specs:
    MS3_PIPELINE_SPEC       — conceptual research flow (corpus → retrievers → eval)
    MS3_INFRASTRUCTURE_SPEC — detailed infra overlay (DVC, 4-way sharding, checkpoint)

Stage kinds and their visual categorization:
    data    — prepared corpus / gold pairs (tan)
    model   — retrieval baseline (blue)
    eval    — metric computation (green)
    future  — MS4+ roadmap (gray, dashed border)
    infra   — storage / reproducibility layer (purple, dashed border)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REQUIRED_STAGE_KEYS = frozenset({"id", "label", "kind"})
_KIND_STYLE: dict[str, dict[str, Any]] = {
    "data": {"facecolor": "#f4e4bc", "edgecolor": "#8b7355", "linestyle": "solid"},
    "model": {"facecolor": "#c9d9ef", "edgecolor": "#3b6ba3", "linestyle": "solid"},
    "eval": {"facecolor": "#c9e6c9", "edgecolor": "#3b833b", "linestyle": "solid"},
    "future": {"facecolor": "#e8e8e8", "edgecolor": "#888888", "linestyle": "dashed"},
    "infra": {"facecolor": "#e5d4ef", "edgecolor": "#6a3b83", "linestyle": "dashed"},
}


def build_pipeline_graph(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate + normalize a pipeline spec into {nodes, edges}.

    Raises KeyError on missing required stage fields; ValueError on
    edges referencing unknown node ids.
    """
    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    for stage in spec["stages"]:
        missing = _REQUIRED_STAGE_KEYS - set(stage.keys())
        if missing:
            raise KeyError(f"stage missing required keys: {missing} in {stage}")
        if stage["kind"] not in _KIND_STYLE:
            raise ValueError(f"unknown kind {stage['kind']!r}; expected one of {list(_KIND_STYLE)}")
        if stage["id"] in node_ids:
            raise ValueError(f"duplicate stage id: {stage['id']!r}")
        nodes.append(dict(stage))
        node_ids.add(stage["id"])
    edges: list[tuple[str, str]] = []
    for src, dst in spec["edges"]:
        if src not in node_ids:
            raise ValueError(f"edge references unknown node: {src!r}")
        if dst not in node_ids:
            raise ValueError(f"edge references unknown node: {dst!r}")
        edges.append((src, dst))
    return {"nodes": nodes, "edges": edges}


def _layout_nodes(nodes: Sequence[dict[str, Any]], edges: Sequence[tuple[str, str]]) -> dict[str, tuple[float, float]]:
    """Assign (x, y) coordinates via topological-layer layout.

    Nodes with no incoming edges are layer 0; subsequent layers are 1-deeper
    than max predecessor layer. Within a layer, nodes stack vertically.
    """
    incoming: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for src, dst in edges:
        incoming[dst].append(src)
    layers: dict[str, int] = {}

    def compute_layer(node_id: str, visiting: set[str]) -> int:
        if node_id in layers:
            return layers[node_id]
        if node_id in visiting:
            # cycle — treat as layer 0
            return 0
        visiting.add(node_id)
        preds = incoming[node_id]
        layer = 0 if not preds else 1 + max(compute_layer(p, visiting) for p in preds)
        visiting.discard(node_id)
        layers[node_id] = layer
        return layer

    for n in nodes:
        compute_layer(n["id"], set())
    by_layer: dict[int, list[str]] = {}
    for nid, lyr in layers.items():
        by_layer.setdefault(lyr, []).append(nid)
    coords: dict[str, tuple[float, float]] = {}
    for lyr, ids in by_layer.items():
        for i, nid in enumerate(sorted(ids)):
            y = -(i - (len(ids) - 1) / 2.0)
            coords[nid] = (float(lyr), y)
    return coords


def render_pipeline(spec: dict[str, Any], out_path: Path) -> Path:
    """Render the pipeline spec to a PNG at out_path. Returns out_path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    graph = build_pipeline_graph(spec)
    coords = _layout_nodes(graph["nodes"], graph["edges"])
    # scale figure size with graph complexity (infrastructure spec is wider/taller)
    n_layers = int(max(c[0] for c in coords.values())) + 1
    max_per_layer = max(sum(1 for c in coords.values() if int(c[0]) == lyr) for lyr in range(n_layers))
    fig_w = max(14, 2.2 * n_layers)
    fig_h = max(7, 1.3 * max_per_layer)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    box_w, box_h = 0.8, 0.55
    for node in graph["nodes"]:
        x, y = coords[node["id"]]
        style = _KIND_STYLE[node["kind"]]
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.05",
            linewidth=1.8,
            **style,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            node["label"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            wrap=True,
        )
    for src, dst in graph["edges"]:
        sx, sy = coords[src]
        dx, dy = coords[dst]
        ax.annotate(
            "",
            xy=(dx - box_w / 2, dy),
            xytext=(sx + box_w / 2, sy),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.4),
        )
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    legend_handles = [
        mpatches.Patch(facecolor=s["facecolor"], edgecolor=s["edgecolor"], label=kind.capitalize())
        for kind, s in _KIND_STYLE.items()
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=len(_KIND_STYLE), frameon=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------- canonical MS3 conceptual pipeline spec ----------

MS3_PIPELINE_SPEC: dict[str, Any] = {
    "stages": [
        # MS3 executed stages
        {
            "id": "corpus",
            "label": "CourtListener\nCorpus Chunks\n(7.8M)",
            "kind": "data",
        },
        {
            "id": "gold",
            "label": "LePaRD Gold Pairs\n(45K test + 2K val)",
            "kind": "data",
        },
        {"id": "bm25", "label": "BM25\n(bm25s, k1=1.5, b=0.75)", "kind": "model"},
        {"id": "bge_m3", "label": "BGE-M3\n(1024-dim, FAISS IP)", "kind": "model"},
        {"id": "eval", "label": "Evaluation\nHit@k, MRR, NDCG@10", "kind": "eval"},
        # MS4+ roadmap
        {"id": "hybrid", "label": "Hybrid BM25+Dense\n(RRF Fusion)", "kind": "future"},
        {"id": "reranker", "label": "Cross-Encoder\nReranker", "kind": "future"},
        {
            "id": "rag",
            "label": "RAG Generation\n+ Hallucination Check",
            "kind": "future",
        },
    ],
    "edges": [
        # MS3 flow
        ("corpus", "bm25"),
        ("corpus", "bge_m3"),
        ("gold", "eval"),
        ("bm25", "eval"),
        ("bge_m3", "eval"),
        # MS4+ flow
        ("bm25", "hybrid"),
        ("bge_m3", "hybrid"),
        ("hybrid", "reranker"),
        ("reranker", "rag"),
    ],
}


# ---------- canonical MS3 infrastructure spec (detailed infra overlay) ----------
#
# Shows the reproducibility / distributed-compute layer that the conceptual
# MS3_PIPELINE_SPEC abstracts away:
#   - Raw bulk data provenance (cl_bulk from CourtListener)
#   - Subsampling scope decision (1.47M opinion-level subsample for MS3)
#   - 4-way BGE-M3 GPU sharding (matches Harvard ODD 4× L4 deployed reality)
#   - Checkpoint-resume for multi-day encoding jobs
#   - Merge stage that reconciles per-rank FAISS indices
#   - DVC + S3 as the single source of truth for cross-team reproducibility
#
# Sharding cardinality is pinned at 4 to match deployed infrastructure. If the
# cluster is upgraded (e.g., to 8× L4 or different GPUs), both this spec and
# the TestInfrastructureSpec tests must be updated together.

MS3_INFRASTRUCTURE_SPEC: dict[str, Any] = {
    "stages": [
        {
            "id": "cl_bulk",
            "label": "CourtListener\nBulk Dumps\n(58 GB)",
            "kind": "data",
        },
        {
            "id": "corpus_full",
            "label": "Full Corpus\nChunks (7.8M)",
            "kind": "data",
        },
        {
            "id": "subsample",
            "label": "Subsample\n(1-chunk/opinion)",
            "kind": "model",
        },
        {
            "id": "corpus_subsample",
            "label": "Subsample Corpus\n(1.47M chunks)",
            "kind": "data",
        },
        {
            "id": "bge_shard_0",
            "label": "BGE-M3 Shard 0\n(GPU rank 0)",
            "kind": "model",
        },
        {
            "id": "bge_shard_1",
            "label": "BGE-M3 Shard 1\n(GPU rank 1)",
            "kind": "model",
        },
        {
            "id": "bge_shard_2",
            "label": "BGE-M3 Shard 2\n(GPU rank 2)",
            "kind": "model",
        },
        {
            "id": "bge_shard_3",
            "label": "BGE-M3 Shard 3\n(GPU rank 3)",
            "kind": "model",
        },
        {
            "id": "checkpoint",
            "label": "Per-Rank\nFAISS Index\nCheckpoint",
            "kind": "infra",
        },
        {
            "id": "merge",
            "label": "Cross-Shard\nMaxP Merge\n(top-100)",
            "kind": "model",
        },
        {
            "id": "dvc_s3",
            "label": "DVC + S3 Remote\n(reproducibility\nsource of truth)",
            "kind": "infra",
        },
    ],
    "edges": [
        ("cl_bulk", "corpus_full"),
        ("corpus_full", "subsample"),
        ("subsample", "corpus_subsample"),
        ("corpus_subsample", "bge_shard_0"),
        ("corpus_subsample", "bge_shard_1"),
        ("corpus_subsample", "bge_shard_2"),
        ("corpus_subsample", "bge_shard_3"),
        ("bge_shard_0", "checkpoint"),
        ("bge_shard_1", "checkpoint"),
        ("bge_shard_2", "checkpoint"),
        ("bge_shard_3", "checkpoint"),
        ("checkpoint", "merge"),
        ("merge", "dvc_s3"),
    ],
}
