"""Test MS3_INFRASTRUCTURE_SPEC has no stale 'transform' kind or subsample stages."""
import pytest
from src.viz.pipeline_diagram import MS3_INFRASTRUCTURE_SPEC, _KIND_STYLE, build_pipeline_graph


@pytest.mark.contract
class TestInfrastructureSpecKindValidity:
    def test_no_transform_kind(self):
        """No stage may use the legacy 'transform' kind."""
        bad = [s for s in MS3_INFRASTRUCTURE_SPEC["stages"] if s["kind"] == "transform"]
        assert not bad, f"stale 'transform' kind in stages: {[s['id'] for s in bad]}"

    def test_all_kinds_valid(self):
        """Every stage kind must be in _KIND_STYLE."""
        bad = [s for s in MS3_INFRASTRUCTURE_SPEC["stages"] if s["kind"] not in _KIND_STYLE]
        assert not bad, f"unknown kinds: {[(s['id'], s['kind']) for s in bad]}"

    def test_no_subsample_stages(self):
        """Subsample stages were removed in MS4 (full-corpus pipeline)."""
        ids = [s["id"] for s in MS3_INFRASTRUCTURE_SPEC["stages"]]
        assert "subsample" not in ids, "subsample stage must be removed"
        assert "corpus_subsample" not in ids, "corpus_subsample stage must be removed"

    def test_spec_builds(self):
        """Spec must build without errors."""
        graph = build_pipeline_graph(MS3_INFRASTRUCTURE_SPEC)
        assert len(graph["nodes"]) > 0
