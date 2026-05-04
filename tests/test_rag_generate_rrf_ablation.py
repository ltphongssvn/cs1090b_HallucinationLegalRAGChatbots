# tests/test_rag_generate_rrf_ablation.py
"""Test rag_generate supports rrf ablation."""

import pytest


@pytest.fixture
def rag_module():
    return __import__("scripts.rag_generate", fromlist=["*"])


@pytest.mark.contract
class TestRRFAblation:
    def test_rrf_in_ablation_configs(self, rag_module):
        assert "rrf" in rag_module.ABLATION_CONFIGS
        cfg = rag_module.ABLATION_CONFIGS["rrf"]
        assert cfg["results_filename"] == "rrf_results.jsonl"
        assert cfg["label"] == "rrf_rag"

    def test_sbatch_accepts_rrf(self):
        from pathlib import Path

        sbatch = Path("scripts/rag_generate_multigpu.sbatch").read_text()
        assert "rrf)" in sbatch, "sbatch case must accept rrf ablation"
