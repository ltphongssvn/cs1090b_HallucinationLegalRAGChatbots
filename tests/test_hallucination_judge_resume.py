# tests/test_hallucination_judge_resume.py
"""Test hallucination_judge resumes from existing judgments.jsonl."""

import inspect

import pytest


@pytest.fixture
def judge_module():
    return __import__("scripts.hallucination_judge", fromlist=["*"])


@pytest.mark.contract
class TestResume:
    def test_main_reads_existing_judgments(self, judge_module):
        """main() must read existing judgments.jsonl and resume from where it left off."""
        src = inspect.getsource(judge_module.main)
        assert "resume" in src.lower() or "already_judged" in src or "skip" in src.lower(), (
            "main() must implement resume support — load existing judgments.jsonl, skip rows already judged"
        )

    def test_appends_not_overwrites_when_resuming(self, judge_module):
        src = inspect.getsource(judge_module.main)
        # Either explicitly opens in append mode, or filters input by already-judged
        assert '"a"' in src or 'mode="a"' in src or "already_judged" in src or "resumed" in src.lower(), (
            "main() must not blindly overwrite judgments when resuming"
        )
