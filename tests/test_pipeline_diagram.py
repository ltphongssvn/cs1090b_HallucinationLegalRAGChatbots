"""Tests for src.viz.pipeline_diagram — pipeline DAG renderer for Cell 16."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def viz_module() -> Any:
    from src.viz import pipeline_diagram

    return pipeline_diagram


@pytest.mark.contract
class TestContract:
    def test_build_pipeline_graph_exists(self, viz_module: Any) -> None:
        assert callable(getattr(viz_module, "build_pipeline_graph", None))

    def test_render_pipeline_exists(self, viz_module: Any) -> None:
        assert callable(getattr(viz_module, "render_pipeline", None))

    def test_ms3_pipeline_spec_exists(self, viz_module: Any) -> None:
        spec = getattr(viz_module, "MS3_PIPELINE_SPEC", None)
        assert spec is not None
        assert "stages" in spec
        assert "edges" in spec


@pytest.mark.unit
class TestGraphBuilder:
    def test_build_returns_nodes_and_edges(self, viz_module: Any) -> None:
        spec = {
            "stages": [
                {"id": "a", "label": "Stage A", "kind": "data"},
                {"id": "b", "label": "Stage B", "kind": "model"},
            ],
            "edges": [("a", "b")],
        }
        graph = viz_module.build_pipeline_graph(spec)
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1
        assert graph["nodes"][0]["id"] == "a"

    def test_rejects_edge_to_unknown_node(self, viz_module: Any) -> None:
        spec = {
            "stages": [{"id": "a", "label": "A", "kind": "data"}],
            "edges": [("a", "nonexistent")],
        }
        with pytest.raises(ValueError, match="unknown node"):
            viz_module.build_pipeline_graph(spec)

    def test_stage_requires_id_label_kind(self, viz_module: Any) -> None:
        spec = {"stages": [{"id": "a"}], "edges": []}
        with pytest.raises((KeyError, ValueError)):
            viz_module.build_pipeline_graph(spec)

    def test_kind_controls_node_styling(self, viz_module: Any) -> None:
        spec = {
            "stages": [
                {"id": "d", "label": "Data", "kind": "data"},
                {"id": "m", "label": "Model", "kind": "model"},
                {"id": "e", "label": "Eval", "kind": "eval"},
            ],
            "edges": [],
        }
        graph = viz_module.build_pipeline_graph(spec)
        kinds = {n["id"]: n["kind"] for n in graph["nodes"]}
        assert kinds == {"d": "data", "m": "model", "e": "eval"}


@pytest.mark.unit
class TestRenderPipeline:
    def test_renders_to_file(self, viz_module: Any, tmp_path: Path) -> None:
        spec = {
            "stages": [
                {"id": "a", "label": "Corpus Prep", "kind": "data"},
                {"id": "b", "label": "BM25", "kind": "model"},
                {"id": "c", "label": "Eval", "kind": "eval"},
            ],
            "edges": [("a", "b"), ("b", "c")],
        }
        out_path = tmp_path / "pipeline.png"
        result = viz_module.render_pipeline(spec, out_path)
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_ms3_spec_renders(self, viz_module: Any, tmp_path: Path) -> None:
        out_path = tmp_path / "ms3.png"
        viz_module.render_pipeline(viz_module.MS3_PIPELINE_SPEC, out_path)
        assert out_path.exists()


@pytest.mark.unit
class TestMs3PipelineSpec:
    def test_spec_covers_all_ms3_stages(self, viz_module: Any) -> None:
        spec = viz_module.MS3_PIPELINE_SPEC
        stage_ids = {s["id"] for s in spec["stages"]}
        required = {"corpus", "gold", "bm25", "bge_m3", "eval"}
        assert required <= stage_ids, f"missing: {required - stage_ids}"

    def test_spec_has_future_stages(self, viz_module: Any) -> None:
        spec = viz_module.MS3_PIPELINE_SPEC
        stage_kinds = {s["kind"] for s in spec["stages"]}
        assert "future" in stage_kinds

    def test_all_edges_reference_existing_stages(self, viz_module: Any) -> None:
        spec = viz_module.MS3_PIPELINE_SPEC
        ids = {s["id"] for s in spec["stages"]}
        for src, dst in spec["edges"]:
            assert src in ids, f"edge src unknown: {src}"
            assert dst in ids, f"edge dst unknown: {dst}"


@pytest.mark.unit
class TestCycleHandling:
    """DAG cycles must not crash the renderer — treated as layer 0."""

    def test_cycle_does_not_crash_render(self, viz_module: Any, tmp_path: Path) -> None:
        spec = {
            "stages": [
                {"id": "a", "label": "A", "kind": "data"},
                {"id": "b", "label": "B", "kind": "model"},
            ],
            "edges": [("a", "b"), ("b", "a")],  # cycle
        }
        out = tmp_path / "cycle.png"
        viz_module.render_pipeline(spec, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_self_loop_does_not_crash(self, viz_module: Any, tmp_path: Path) -> None:
        spec = {
            "stages": [{"id": "a", "label": "A", "kind": "data"}],
            "edges": [("a", "a")],
        }
        out = tmp_path / "self_loop.png"
        viz_module.render_pipeline(spec, out)
        assert out.exists()


@pytest.mark.unit
class TestDuplicateAndUnknown:
    def test_rejects_duplicate_stage_ids(self, viz_module: Any) -> None:
        spec = {
            "stages": [
                {"id": "a", "label": "A1", "kind": "data"},
                {"id": "a", "label": "A2", "kind": "model"},
            ],
            "edges": [],
        }
        with pytest.raises(ValueError, match="duplicate stage id"):
            viz_module.build_pipeline_graph(spec)

    def test_rejects_unknown_kind(self, viz_module: Any) -> None:
        spec = {
            "stages": [{"id": "a", "label": "A", "kind": "mystery"}],
            "edges": [],
        }
        with pytest.raises(ValueError, match="unknown kind"):
            viz_module.build_pipeline_graph(spec)

    def test_preserves_label_on_nodes(self, viz_module: Any) -> None:
        spec = {
            "stages": [
                {"id": "a", "label": "Corpus Prep Step", "kind": "data"},
                {"id": "b", "label": "BM25 Retrieval", "kind": "model"},
            ],
            "edges": [("a", "b")],
        }
        graph = viz_module.build_pipeline_graph(spec)
        labels = {n["id"]: n["label"] for n in graph["nodes"]}
        assert labels == {"a": "Corpus Prep Step", "b": "BM25 Retrieval"}


@pytest.mark.unit
class TestMs3Topology:
    """Lock the scientific pipeline flow, not just node presence."""

    def test_corpus_feeds_both_retrievers(self, viz_module: Any) -> None:
        edges = set(viz_module.MS3_PIPELINE_SPEC["edges"])
        assert ("corpus", "bm25") in edges
        assert ("corpus", "bge_m3") in edges

    def test_both_retrievers_feed_eval(self, viz_module: Any) -> None:
        edges = set(viz_module.MS3_PIPELINE_SPEC["edges"])
        assert ("bm25", "eval") in edges
        assert ("bge_m3", "eval") in edges

    def test_gold_feeds_eval(self, viz_module: Any) -> None:
        edges = set(viz_module.MS3_PIPELINE_SPEC["edges"])
        assert ("gold", "eval") in edges


@pytest.mark.contract
class TestInfrastructureSpec:
    """MS3_INFRASTRUCTURE_SPEC — detailed infra view: DVC, sharding, checkpointing."""

    def test_infrastructure_spec_exists(self, viz_module: Any) -> None:
        spec = getattr(viz_module, "MS3_INFRASTRUCTURE_SPEC", None)
        assert spec is not None
        assert "stages" in spec
        assert "edges" in spec

    def test_infrastructure_spec_covers_all_infra_concerns(self, viz_module: Any) -> None:
        """Infrastructure spec must show DVC, sharding, checkpointing, subsampling."""
        spec = viz_module.MS3_INFRASTRUCTURE_SPEC
        ids = {s["id"] for s in spec["stages"]}
        required = {
            "cl_bulk",
            "corpus_full",
            "subsample",
            "corpus_subsample",
            "bge_shard_0",
            "bge_shard_1",
            "bge_shard_2",
            "bge_shard_3",
            "checkpoint",
            "merge",
            "dvc_s3",
        }
        missing = required - ids
        assert not missing, f"missing infrastructure nodes: {missing}"

    def test_infrastructure_edges_reference_existing_stages(self, viz_module: Any) -> None:
        spec = viz_module.MS3_INFRASTRUCTURE_SPEC
        ids = {s["id"] for s in spec["stages"]}
        for src, dst in spec["edges"]:
            assert src in ids, f"infra edge src unknown: {src}"
            assert dst in ids, f"infra edge dst unknown: {dst}"

    def test_infrastructure_has_dvc_tracking_node(self, viz_module: Any) -> None:
        """DVC / S3 storage layer is the key reproducibility artifact."""
        spec = viz_module.MS3_INFRASTRUCTURE_SPEC
        dvc_nodes = [s for s in spec["stages"] if s["id"] == "dvc_s3"]
        assert len(dvc_nodes) == 1
        assert "S3" in dvc_nodes[0]["label"] or "DVC" in dvc_nodes[0]["label"]

    def test_infrastructure_has_4_way_sharding(self, viz_module: Any) -> None:
        """Four per-rank BGE-M3 shards must be represented with exact IDs bge_shard_{0..3}."""
        spec = viz_module.MS3_INFRASTRUCTURE_SPEC
        shard_ids = {s["id"] for s in spec["stages"] if s["id"].startswith("bge_shard_")}
        assert shard_ids == {
            "bge_shard_0",
            "bge_shard_1",
            "bge_shard_2",
            "bge_shard_3",
        }

    def test_infrastructure_stage_ids_are_unique(self, viz_module: Any) -> None:
        """Spec must not declare duplicate stage IDs (sets would silently collapse them)."""
        spec = viz_module.MS3_INFRASTRUCTURE_SPEC
        stage_ids = [s["id"] for s in spec["stages"]]
        assert len(stage_ids) == len(set(stage_ids)), (
            f"duplicate stage ids: {[x for x in stage_ids if stage_ids.count(x) > 1]}"
        )
