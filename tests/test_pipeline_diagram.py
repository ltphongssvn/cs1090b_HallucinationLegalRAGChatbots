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
