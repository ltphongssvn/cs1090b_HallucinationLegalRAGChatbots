# tests/test_baseline_reranker_resume_merged.py
"""Test resume logic also recognizes merged .jsonl (not just .tmp)."""

import inspect

import pytest


@pytest.fixture
def reranker_module():
    return __import__("scripts.baseline_reranker", fromlist=["*"])


@pytest.mark.contract
class TestResumeFromMerged:
    def test_main_checks_merged_jsonl_for_resume(self, reranker_module):
        """main() must check final results_path.jsonl as resume source.

        Yesterday's job merged .tmp -> .jsonl and deleted .tmp. Today's
        restart needs to detect the merged file and skip already-done
        queries, not re-process from scratch.
        """
        src = inspect.getsource(reranker_module.main)
        # Either checks results_path (the merged final file) for resume,
        # or has a per-rank merged-output check distinct from .tmp
        has_merged_resume = (
            ("results_path.is_file()" in src and "already_done" in src)
            or "rank_output_path" in src
            or "merged_output" in src
        )
        assert has_merged_resume, (
            "Resume logic must consider merged .jsonl in addition to .tmp; "
            "yesterdays merged outputs deleted .tmp so resume from .tmp alone "
            "wastes compute re-processing already-done ranks"
        )
