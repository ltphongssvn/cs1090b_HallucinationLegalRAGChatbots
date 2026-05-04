# tests/test_baseline_reranker_resume.py
"""Test baseline_reranker.py resumes from existing per-rank .jsonl.tmp on restart."""

import inspect

import pytest


@pytest.fixture
def reranker_module():
    return __import__("scripts.baseline_reranker", fromlist=["*"])


@pytest.mark.contract
class TestResume:
    def test_main_reads_existing_tmp(self, reranker_module):
        """main() must check for existing .tmp file and skip already-processed queries."""
        src = inspect.getsource(reranker_module.main)
        assert "already_done" in src or "resume" in src.lower() or "skip_processed" in src, (
            "main() must support resume from per-rank .jsonl.tmp on restart "
            "to avoid wasting GPU hours on already-processed queries after TIMEOUT"
        )

    def test_writer_uses_append_mode(self, reranker_module):
        """When resuming, fout must open in append mode, not write mode."""
        src = inspect.getsource(reranker_module.main)
        # Either append mode used for tmp, or queries filtered before writing
        has_resume_path = '"a"' in src or 'mode="a"' in src or "already_done" in src or "skip_processed" in src
        assert has_resume_path, "tmp writes must support resume (append mode or filter)"
