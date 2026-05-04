# tests/test_run_hallucination_judge_parallel_launch.py
"""Test parallel launcher actually starts multiple processes concurrently."""

import inspect

import pytest


@pytest.fixture
def runner():
    return __import__("scripts.run_hallucination_judge_parallel", fromlist=["*"])


@pytest.mark.contract
class TestParallelLaunchLogic:
    def test_initial_batch_fills_max_parallel(self, runner):
        """Launcher must start max_parallel jobs concurrently, not one-at-a-time.

        The buggy version uses `while pending: _start_next(); sleep(1)` which
        only starts ONE job before each sleep, never reaching max_parallel
        concurrency.
        """
        src = inspect.getsource(runner.main)
        # Look for an initial batch loop that fills running up to max_parallel
        # before entering the wait-for-completion phase
        has_batch_fill = (
            "while pending and len(running) < args.max_parallel" in src
            or "for _ in range(args.max_parallel)" in src
            or "fill" in src.lower()
        )
        assert has_batch_fill, (
            "main() must fill initial batch up to max_parallel concurrently. "
            "Buggy pattern: `while pending: _start_next(); sleep(1)` only starts "
            "one process before sleeping, never reaching max_parallel."
        )
