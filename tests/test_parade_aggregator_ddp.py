# tests/test_parade_aggregator_ddp.py
"""Test DDP device binding in parade_aggregator (regression for cuda:0 collision)."""

from __future__ import annotations

import inspect
import re

import pytest


@pytest.fixture
def parade_module():
    return __import__("scripts.parade_aggregator", fromlist=["*"])


@pytest.mark.contract
class TestDDPDeviceBinding:
    def test_main_calls_set_device(self, parade_module):
        src = inspect.getsource(parade_module.main)
        assert "torch.cuda.set_device" in src

    def test_main_reads_local_rank_env(self, parade_module):
        src = inspect.getsource(parade_module.main)
        assert "LOCAL_RANK" in src

    def test_ddp_uses_local_rank_for_device_ids(self, parade_module):
        src = inspect.getsource(parade_module.main)
        m = re.search(r"DistributedDataParallel\([^)]*device_ids=\[([^\]]+)\]", src)
        assert m
        assert "local_rank" in m.group(1)
