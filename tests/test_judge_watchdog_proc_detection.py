# tests/test_judge_watchdog_proc_detection.py
"""Test watchdog uses /proc for process detection (regression: ps -ef missed children)."""

import inspect

import pytest


@pytest.fixture
def watchdog():
    return __import__("scripts.judge_watchdog", fromlist=["*"])


@pytest.mark.contract
class TestProcDetection:
    def test_count_uses_proc(self, watchdog):
        src = inspect.getsource(watchdog._count_judge_children)
        assert "/proc" in src, "_count_judge_children must use /proc/PID/cmdline; ps -ef is unreliable"
