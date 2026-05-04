# tests/test_judge_watchdog.py
"""Test watchdog module exists with required helpers."""

import pytest


@pytest.fixture
def watchdog():
    return __import__("scripts.judge_watchdog", fromlist=["*"])


@pytest.mark.contract
class TestContract:
    def test_main_callable(self, watchdog):
        assert callable(getattr(watchdog, "main", None))

    def test_count_judge_children_callable(self, watchdog):
        assert callable(getattr(watchdog, "_count_judge_children", None))

    def test_is_done_callable(self, watchdog):
        assert callable(getattr(watchdog, "_is_done", None))

    def test_default_check_interval(self, watchdog):
        assert watchdog.CHECK_INTERVAL_SEC >= 60
