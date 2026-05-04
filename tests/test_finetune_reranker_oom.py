# tests/test_finetune_reranker_oom.py
"""Test gradient checkpointing enabled to prevent OOM on L4 22GB."""

import inspect

import pytest


@pytest.fixture
def ft_module():
    return __import__("scripts.finetune_reranker", fromlist=["*"])


@pytest.mark.contract
class TestGradientCheckpointing:
    def test_main_enables_gradient_checkpointing(self, ft_module):
        """main() must call model.gradient_checkpointing_enable() to fit on L4 22GB."""
        src = inspect.getsource(ft_module.main)
        assert "gradient_checkpointing_enable" in src, (
            "main() must call model.gradient_checkpointing_enable() to prevent "
            "OOM at batch_size=4+ with XLM-RoBERTa-large + max_length=1024 on L4"
        )
