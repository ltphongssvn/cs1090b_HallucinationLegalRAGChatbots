# tests/test_run_hallucination_judge_parallel.py
"""Test parallel launcher exists."""

import pytest


@pytest.fixture
def runner():
    return __import__("scripts.run_hallucination_judge_parallel", fromlist=["*"])


@pytest.mark.contract
class TestContract:
    def test_main_callable(self, runner):
        assert callable(getattr(runner, "main", None))

    def test_default_max_parallel(self, runner):
        assert runner.DEFAULT_MAX_PARALLEL >= 2
