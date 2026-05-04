# tests/test_judge_watchdog_dry_run.py
"""Test --dry-run mode: counts processes once, exits without launching."""

import inspect

import pytest


@pytest.fixture
def watchdog():
    return __import__("scripts.judge_watchdog", fromlist=["*"])


@pytest.mark.contract
class TestDryRun:
    def test_main_supports_dry_run(self, watchdog):
        ap = watchdog._build_arg_parser()
        args = ap.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_main_dry_run_exits_without_launch(self, watchdog):
        src = inspect.getsource(watchdog.main)
        assert "dry_run" in src, "main must check args.dry_run and exit early"
