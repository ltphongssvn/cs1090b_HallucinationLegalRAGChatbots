# tests/test_parade_aggregator.py
"""Tests for scripts.parade_aggregator — PARADE-Transformer passage aggregation."""

from __future__ import annotations

import pytest


@pytest.fixture
def parade_module():
    return __import__("scripts.parade_aggregator", fromlist=["*"])


@pytest.mark.contract
class TestContract:
    def test_constants(self, parade_module):
        assert parade_module.SCHEMA_VERSION
        assert parade_module.DEFAULT_AGG_HIDDEN_SIZE > 0
        assert parade_module.DEFAULT_N_AGG_LAYERS >= 1
        assert parade_module.DEFAULT_MAX_PASSAGES >= 2

    def test_build_aggregator_callable(self, parade_module):
        assert callable(getattr(parade_module, "build_aggregator", None))

    def test_main_callable(self, parade_module):
        assert callable(getattr(parade_module, "main", None))


@pytest.mark.unit
class TestBuildAggregator:
    def test_returns_module_with_forward(self, parade_module):
        import torch

        agg = parade_module.build_aggregator(
            hidden_size=64,
            n_layers=2,
            max_passages=4,
            n_heads=2,
        )
        # Input: batch=2, max_passages=4, hidden=64
        x = torch.randn(2, 4, 64)
        scores = agg(x)
        assert scores.shape == (2,)

    def test_handles_max_passages_padding(self, parade_module):
        import torch

        agg = parade_module.build_aggregator(
            hidden_size=32,
            n_layers=1,
            max_passages=8,
            n_heads=2,
        )
        x = torch.randn(3, 8, 32)
        scores = agg(x)
        assert scores.shape == (3,)
