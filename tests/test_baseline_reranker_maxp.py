# tests/test_baseline_reranker_maxp.py
"""Tests for MaxP chunk-level scoring extension to baseline_reranker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def reranker_module() -> Any:
    from scripts import baseline_reranker

    return baseline_reranker


@pytest.mark.contract
class TestContract:
    def test_load_cluster_chunks_index_callable(self, reranker_module: Any) -> None:
        assert callable(getattr(reranker_module, "_load_cluster_chunks_index", None))

    def test_maxp_aggregate_callable(self, reranker_module: Any) -> None:
        assert callable(getattr(reranker_module, "_maxp_aggregate", None))


@pytest.mark.unit
class TestLoadClusterChunksIndex:
    def test_returns_list_of_chunks_per_cluster(self, reranker_module: Any, tmp_path: Path) -> None:
        corpus = _write_jsonl(
            tmp_path / "corpus.jsonl",
            [
                {"opinion_id": 1, "chunk_index": 0, "cluster_id": 100, "text": "alpha"},
                {"opinion_id": 1, "chunk_index": 1, "cluster_id": 100, "text": "beta"},
                {"opinion_id": 1, "chunk_index": 2, "cluster_id": 100, "text": "gamma"},
                {"opinion_id": 2, "chunk_index": 0, "cluster_id": 200, "text": "delta"},
            ],
        )
        idx = reranker_module._load_cluster_chunks_index(corpus, max_chunks_per_cluster=3)
        assert 100 in idx
        assert isinstance(idx[100], list)
        assert idx[100] == ["alpha", "beta", "gamma"]
        assert idx[200] == ["delta"]

    def test_max_chunks_caps(self, reranker_module: Any, tmp_path: Path) -> None:
        corpus = _write_jsonl(
            tmp_path / "corpus.jsonl",
            [{"opinion_id": 1, "chunk_index": i, "cluster_id": 1, "text": f"c{i}"} for i in range(10)],
        )
        idx = reranker_module._load_cluster_chunks_index(corpus, max_chunks_per_cluster=3)
        assert len(idx[1]) == 3
        assert idx[1] == ["c0", "c1", "c2"]

    def test_chunks_sorted_by_chunk_index(self, reranker_module: Any, tmp_path: Path) -> None:
        # Chunks written out of order
        corpus = _write_jsonl(
            tmp_path / "corpus.jsonl",
            [
                {"opinion_id": 1, "chunk_index": 2, "cluster_id": 1, "text": "third"},
                {"opinion_id": 1, "chunk_index": 0, "cluster_id": 1, "text": "first"},
                {"opinion_id": 1, "chunk_index": 1, "cluster_id": 1, "text": "second"},
            ],
        )
        idx = reranker_module._load_cluster_chunks_index(corpus, max_chunks_per_cluster=10)
        assert idx[1] == ["first", "second", "third"]


@pytest.mark.unit
class TestMaxPAggregate:
    def test_takes_max_across_chunk_scores(self, reranker_module: Any) -> None:
        # Three clusters, each with 2-3 chunk scores
        chunk_scores_by_cluster = {
            10: [0.1, 0.9, 0.5],  # max=0.9
            20: [0.7, 0.7],  # max=0.7
            30: [0.2],  # max=0.2
        }
        candidates = [
            {"cluster_id": 10, "score": 0.0},
            {"cluster_id": 20, "score": 0.0},
            {"cluster_id": 30, "score": 0.0},
        ]
        out = reranker_module._maxp_aggregate(candidates, chunk_scores_by_cluster, top_k=10)
        # Sorted by max descending
        assert [c["cluster_id"] for c in out] == [10, 20, 30]
        assert out[0]["score"] == pytest.approx(0.9)
        assert out[1]["score"] == pytest.approx(0.7)
        assert out[2]["score"] == pytest.approx(0.2)

    def test_top_k_truncation(self, reranker_module: Any) -> None:
        chunk_scores = {i: [float(i)] for i in range(10)}
        candidates = [{"cluster_id": i, "score": 0.0} for i in range(10)]
        out = reranker_module._maxp_aggregate(candidates, chunk_scores, top_k=3)
        assert len(out) == 3
        assert [c["cluster_id"] for c in out] == [9, 8, 7]

    def test_skips_clusters_with_no_chunks(self, reranker_module: Any) -> None:
        chunk_scores = {10: [0.5]}  # 20 missing
        candidates = [
            {"cluster_id": 10, "score": 0.0},
            {"cluster_id": 20, "score": 0.0},
        ]
        out = reranker_module._maxp_aggregate(candidates, chunk_scores, top_k=10)
        # Only cluster 10 emitted
        assert len(out) == 1
        assert out[0]["cluster_id"] == 10

    def test_deterministic_tie_break(self, reranker_module: Any) -> None:
        chunk_scores = {5: [0.5], 1: [0.5], 3: [0.5]}
        candidates = [{"cluster_id": cid, "score": 0.0} for cid in (5, 1, 3)]
        out = reranker_module._maxp_aggregate(candidates, chunk_scores, top_k=3)
        # Ascending cluster_id on tie
        assert [c["cluster_id"] for c in out] == [1, 3, 5]
