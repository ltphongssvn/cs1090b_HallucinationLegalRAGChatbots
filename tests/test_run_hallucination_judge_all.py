# tests/test_run_hallucination_judge_all.py
"""Test launcher script exists and is importable."""

import pytest


@pytest.fixture
def runner_module():
    return __import__("scripts.run_hallucination_judge_all", fromlist=["*"])


@pytest.mark.contract
class TestContract:
    def test_main_callable(self, runner_module):
        assert callable(getattr(runner_module, "main", None))

    def test_default_ablations(self, runner_module):
        assert "none" in runner_module.DEFAULT_ABLATIONS
        assert "bm25" in runner_module.DEFAULT_ABLATIONS
        assert "reranker" in runner_module.DEFAULT_ABLATIONS

    def test_retrieval_dir_mapping(self, runner_module):
        m = runner_module.RETRIEVAL_DIR_FOR_ABLATION
        # Reranker reads from finetuned/ subdir
        assert "finetuned" in m["reranker"]
        # Others from cleaned/
        assert m["bm25"].endswith("cleaned")

    def test_label_mapping(self, runner_module):
        assert runner_module.LABEL_FOR_ABLATION["none"] == "no_rag"
        assert runner_module.LABEL_FOR_ABLATION["reranker"] == "reranker_rag"
