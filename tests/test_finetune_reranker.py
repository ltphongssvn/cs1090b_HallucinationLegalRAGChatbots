# tests/test_finetune_reranker.py
"""Tests for scripts.finetune_reranker — bge-reranker-v2-m3 fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def ft_module() -> Any:
    return ft_module if False else __import__("scripts.finetune_reranker", fromlist=["*"])


@pytest.mark.contract
class TestContract:
    def test_constants(self, ft_module: Any) -> None:
        assert ft_module.BASE_MODEL == "BAAI/bge-reranker-v2-m3"
        assert ft_module.DEFAULT_LR > 0
        assert ft_module.DEFAULT_EPOCHS >= 1

    def test_load_training_pairs_callable(self, ft_module: Any) -> None:
        assert callable(getattr(ft_module, "_load_training_pairs", None))

    def test_main_callable(self, ft_module: Any) -> None:
        assert callable(getattr(ft_module, "main", None))


@pytest.mark.unit
class TestLoadTrainingPairs:
    def test_yields_pos_and_neg_pairs(self, ft_module: Any, tmp_path: Path) -> None:
        train = _write_jsonl(
            tmp_path / "t.jsonl",
            [
                {"query": "q1", "pos": ["p1"], "neg": ["n1a", "n1b"]},
                {"query": "q2", "pos": ["p2"], "neg": ["n2a"]},
            ],
        )
        pairs, labels = ft_module._load_training_pairs(train)
        # Each row → 1 pos pair (label=1) + len(neg) neg pairs (label=0)
        # Row 1: 1 pos + 2 neg = 3
        # Row 2: 1 pos + 1 neg = 2
        assert len(pairs) == 5
        assert len(labels) == 5
        assert sum(labels) == 2  # 2 positives total
        # First pair = (q1, p1) with label 1
        assert pairs[0] == ["q1", "p1"]
        assert labels[0] == 1

    def test_skips_rows_missing_pos(self, ft_module: Any, tmp_path: Path) -> None:
        train = _write_jsonl(
            tmp_path / "t.jsonl",
            [
                {"query": "q1", "neg": ["n"]},  # no pos
            ],
        )
        pairs, labels = ft_module._load_training_pairs(train)
        assert pairs == []
        assert labels == []
