# tests/test_mine_hard_negatives.py
"""Tests for scripts.mine_hard_negatives — hard-negative mining for reranker fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def mine_module() -> Any:
    from scripts import mine_hard_negatives

    return mine_hard_negatives


@pytest.mark.contract
class TestContract:
    def test_mine_callable(self, mine_module: Any) -> None:
        assert callable(getattr(mine_module, "mine", None))

    def test_constants(self, mine_module: Any) -> None:
        assert mine_module.DEFAULT_N_NEGATIVES_PER_POSITIVE >= 1
        assert mine_module.DEFAULT_NEG_RANK_RANGE[0] >= 1


@pytest.mark.unit
class TestMine:
    def test_emits_one_row_per_positive(self, mine_module: Any, tmp_path: Path) -> None:
        gold = _write_jsonl(
            tmp_path / "gold.jsonl",
            [
                {"source_id": 1, "dest_id": 100, "source_cluster_id": 10, "quote": "q1"},
                {"source_id": 2, "dest_id": 200, "source_cluster_id": 20, "quote": "q2"},
            ],
        )
        # RRF: each query has 5 candidates; gold cluster appears at varied ranks
        rrf = _write_jsonl(
            tmp_path / "rrf.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [
                        {"cluster_id": 10, "score": 0.5},  # rank 1 = gold
                        {"cluster_id": 99, "score": 0.4},  # rank 2 = neg
                        {"cluster_id": 88, "score": 0.3},
                        {"cluster_id": 77, "score": 0.2},
                        {"cluster_id": 66, "score": 0.1},
                    ],
                },
                {
                    "source_id": 2,
                    "dest_id": 200,
                    "source_cluster_id": 20,
                    "retrieved": [
                        {"cluster_id": 50, "score": 0.5},  # gold not in list - skip
                        {"cluster_id": 51, "score": 0.4},
                    ],
                },
            ],
        )
        out = mine_module.mine(
            gold_path=gold,
            rrf_path=rrf,
            n_neg_per_pos=2,
            neg_rank_range=(2, 100),
            seed=0,
        )
        # Only query 1 yields training row (gold present)
        assert len(out) == 1
        row = out[0]
        assert row["query"] == "q1"
        assert row["pos_cluster_id"] == 10
        assert len(row["neg_cluster_ids"]) == 2
        # Negatives drawn from rank 2-5 (cluster 99, 88, 77, 66)
        assert all(cid in {99, 88, 77, 66} for cid in row["neg_cluster_ids"])

    def test_skips_query_when_gold_absent(self, mine_module: Any, tmp_path: Path) -> None:
        gold = _write_jsonl(
            tmp_path / "g.jsonl",
            [
                {"source_id": 1, "dest_id": 100, "source_cluster_id": 10, "quote": "q"},
            ],
        )
        rrf = _write_jsonl(
            tmp_path / "r.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [{"cluster_id": 99, "score": 0.5}],
                },  # gold not in retrieved
            ],
        )
        out = mine_module.mine(gold_path=gold, rrf_path=rrf, n_neg_per_pos=1, seed=0)
        assert out == []

    def test_seed_reproducibility(self, mine_module: Any, tmp_path: Path) -> None:
        gold = _write_jsonl(
            tmp_path / "g.jsonl",
            [
                {"source_id": 1, "dest_id": 100, "source_cluster_id": 10, "quote": "q"},
            ],
        )
        rrf = _write_jsonl(
            tmp_path / "r.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [{"cluster_id": 10, "score": 0.5}]
                    + [{"cluster_id": i, "score": 0.4 - i * 0.01} for i in range(20, 100)],
                },
            ],
        )
        out1 = mine_module.mine(gold_path=gold, rrf_path=rrf, n_neg_per_pos=3, seed=42)
        out2 = mine_module.mine(gold_path=gold, rrf_path=rrf, n_neg_per_pos=3, seed=42)
        assert out1[0]["neg_cluster_ids"] == out2[0]["neg_cluster_ids"]
