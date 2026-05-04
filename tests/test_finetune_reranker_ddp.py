# tests/test_finetune_reranker_ddp.py
"""Test DDP device binding in finetune_reranker.

Catches the regression where rank N>0 ended up on cuda:0 (rank 0\'s GPU)
because the script used torch.cuda.current_device() without first calling
torch.cuda.set_device(local_rank).
"""

from __future__ import annotations

import inspect
import re

import pytest


@pytest.fixture
def ft_module():
    return __import__("scripts.finetune_reranker", fromlist=["*"])


@pytest.mark.contract
class TestDDPDeviceBinding:
    def test_main_calls_set_device(self, ft_module):
        """main() must call torch.cuda.set_device(local_rank) for DDP correctness."""
        src = inspect.getsource(ft_module.main)
        assert "torch.cuda.set_device" in src, (
            "main() must call torch.cuda.set_device(local_rank) before DDP init "
            "to prevent multiple ranks binding to cuda:0"
        )

    def test_main_reads_local_rank_env(self, ft_module):
        """main() must read LOCAL_RANK env var (set by torchrun) for per-process device."""
        src = inspect.getsource(ft_module.main)
        assert "LOCAL_RANK" in src, "main() must read LOCAL_RANK from env (torchrun sets this per process)"

    def test_ddp_uses_local_rank_for_device_ids(self, ft_module):
        """DDP wrap must use device_ids=[local_rank], not torch.cuda.current_device()."""
        src = inspect.getsource(ft_module.main)
        # Find the DDP wrap line
        m = re.search(r"DistributedDataParallel\([^)]*device_ids=\[([^\]]+)\]", src)
        assert m, "DistributedDataParallel(...device_ids=[...]) not found"
        device_ids_arg = m.group(1)
        assert "local_rank" in device_ids_arg, f"DDP device_ids must reference local_rank, got: {device_ids_arg!r}"
