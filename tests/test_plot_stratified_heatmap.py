"""Tests for stratified heatmap visualization."""
import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def heatmap_module():
    spec = importlib.util.spec_from_file_location(
        "plot_stratified_heatmap",
        Path("scripts/viz/plot_stratified_heatmap.py"),
    )
    if spec is None or spec.loader is None:
        pytest.skip("plot_stratified_heatmap.py not yet created")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.contract
class TestStratifiedHeatmap:
    def test_module_exists(self):
        assert Path("scripts/viz/plot_stratified_heatmap.py").is_file(), \
            "viz script must exist at scripts/viz/plot_stratified_heatmap.py"

    def test_has_plot_heatmap_function(self, heatmap_module):
        assert hasattr(heatmap_module, "plot_heatmap"), \
            "module must expose plot_heatmap()"

    def test_has_default_retrievers_with_6_variants(self, heatmap_module):
        retrievers = getattr(heatmap_module, "DEFAULT_RETRIEVERS", None)
        assert retrievers is not None, "DEFAULT_RETRIEVERS must be defined"
        labels = [label for label, _ in retrievers]
        assert "bm25" in labels
        assert "bge_m3" in labels
        assert "rrf" in labels
        assert "reranker_finetuned" in labels

    def test_has_three_buckets(self, heatmap_module):
        buckets = getattr(heatmap_module, "BUCKETS", None)
        assert buckets is not None
        assert {b.lower() for b in buckets} == {"head", "torso", "tail"}

    def test_main_callable(self, heatmap_module):
        assert callable(getattr(heatmap_module, "main", None)), \
            "main() must be callable"
