# tests/test_stratified_eval.py
"""Tests for scripts.stratified_eval — frequency-stratified retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def strat_module() -> Any:
    return strat_module if False else __import__("scripts.stratified_eval", fromlist=["*"])


@pytest.mark.contract
class TestContract:
    def test_compute_cluster_frequencies_callable(self, strat_module: Any) -> None:
        assert callable(getattr(strat_module, "_compute_cluster_frequencies", None))

    def test_assign_buckets_callable(self, strat_module: Any) -> None:
        assert callable(getattr(strat_module, "_assign_buckets", None))

    def test_evaluate_stratified_callable(self, strat_module: Any) -> None:
        assert callable(getattr(strat_module, "evaluate_stratified", None))


@pytest.mark.unit
class TestComputeClusterFrequencies:
    def test_counts_gold_occurrences(self, strat_module: Any, tmp_path: Path) -> None:
        gold = _write_jsonl(
            tmp_path / "g.jsonl",
            [
                {"source_id": 1, "dest_id": 100, "source_cluster_id": 10, "quote": "x"},
                {"source_id": 2, "dest_id": 200, "source_cluster_id": 10, "quote": "y"},
                {"source_id": 3, "dest_id": 300, "source_cluster_id": 20, "quote": "z"},
            ],
        )
        freq = strat_module._compute_cluster_frequencies(gold)
        assert freq[10] == 2
        assert freq[20] == 1


@pytest.mark.unit
class TestAssignBuckets:
    def test_three_buckets_by_quantile(self, strat_module: Any) -> None:
        # cluster freq: {10:100, 20:50, 30:10, 40:5, 50:2, 60:1}
        freq = {10: 100, 20: 50, 30: 10, 40: 5, 50: 2, 60: 1}
        buckets = strat_module._assign_buckets(freq, n_buckets=3)
        assert set(buckets.values()) <= {"head", "torso", "tail"}
        # 10 (highest) is head; 60 (lowest) is tail
        assert buckets[10] == "head"
        assert buckets[60] == "tail"

    def test_singleton_freq_all_same_bucket(self, strat_module: Any) -> None:
        freq = {1: 5, 2: 5, 3: 5}
        buckets = strat_module._assign_buckets(freq, n_buckets=3)
        # Ties — all go to same bucket
        assert len(set(buckets.values())) == 1


@pytest.mark.unit
class TestEvaluateStratified:
    def test_per_bucket_metrics_emitted(self, strat_module: Any, tmp_path: Path) -> None:
        gold = _write_jsonl(
            tmp_path / "g.jsonl",
            [
                {"source_id": 1, "dest_id": 100, "source_cluster_id": 10, "quote": "a"},
                {"source_id": 2, "dest_id": 200, "source_cluster_id": 10, "quote": "b"},
                {"source_id": 3, "dest_id": 300, "source_cluster_id": 20, "quote": "c"},
            ],
        )
        results = _write_jsonl(
            tmp_path / "r.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [{"cluster_id": 10, "score": 1.0}],
                },
                {
                    "source_id": 2,
                    "dest_id": 200,
                    "source_cluster_id": 10,
                    "retrieved": [{"cluster_id": 99, "score": 1.0}],
                },
                {
                    "source_id": 3,
                    "dest_id": 300,
                    "source_cluster_id": 20,
                    "retrieved": [{"cluster_id": 20, "score": 1.0}],
                },
            ],
        )
        out = strat_module.evaluate_stratified(
            gold_path=gold,
            results_path=results,
            n_buckets=2,
            k_values=(1,),
            gold_field="source_cluster_id",
            match_field="cluster_id",
        )
        assert "per_bucket" in out
        assert "overall" in out
        assert out["overall"]["n_queries"] == 3
        # Cluster 10 is "head" (freq=2), cluster 20 is "tail" (freq=1)
        assert "head" in out["per_bucket"]
        assert "tail" in out["per_bucket"]
