# tests/test_hallucination_judge_buffering.py
"""Test judge writes are line-buffered to prevent buffer loss on crash."""

import inspect

import pytest


@pytest.fixture
def judge_module():
    return __import__("scripts.hallucination_judge", fromlist=["*"])


@pytest.mark.contract
class TestLineBuffering:
    def test_main_uses_line_buffering_or_flush(self, judge_module):
        """main() must either open with buffering=1 (line-buffered) or call fout.flush()."""
        src = inspect.getsource(judge_module.main)
        assert "buffering=1" in src or "fout.flush()" in src or "flush()" in src, (
            "Judge must use line buffering or explicit flush so judgments persist even if process is killed mid-run"
        )
