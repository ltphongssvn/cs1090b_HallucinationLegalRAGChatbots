# tests/test_baseline_reranker.py
"""Tests for scripts.baseline_reranker — cross-encoder reranking of fused candidates.

Heavy GPU dependencies (transformers, torch CUDA, model downloads) are mocked
or guarded with skipif so the test suite remains fast and runs on CPU-only CI.
End-to-end GPU integration is verified separately by the multigpu sbatch run.
"""

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


# ---------- contract ----------


@pytest.mark.contract
class TestContract:
    def test_load_queries_exists(self, reranker_module: Any) -> None:
        assert callable(getattr(reranker_module, "_load_queries", None))

    def test_load_cluster_text_index_exists(self, reranker_module: Any) -> None:
        assert callable(getattr(reranker_module, "_load_cluster_text_index", None))

    def test_shard_range_exists(self, reranker_module: Any) -> None:
        assert callable(getattr(reranker_module, "_shard_range", None))

    def test_merge_shard_results_exists(self, reranker_module: Any) -> None:
        assert callable(getattr(reranker_module, "_merge_shard_results", None))

    def test_main_exists(self, reranker_module: Any) -> None:
        assert callable(getattr(reranker_module, "main", None))

    def test_default_constants(self, reranker_module: Any) -> None:
        assert reranker_module.RERANKER_MODEL == "BAAI/bge-reranker-v2-m3"
        assert reranker_module.DEFAULT_TOP_K_INPUT == 100
        assert reranker_module.DEFAULT_TOP_K_OUTPUT == 100
        assert reranker_module.DEFAULT_MAX_LENGTH == 1024
        assert reranker_module.DEFAULT_BATCH_SIZE == 32

    def test_schema_version(self, reranker_module: Any) -> None:
        import re

        assert re.match(r"^\d+\.\d+\.\d+$", reranker_module.SCHEMA_VERSION)


# ---------- query loading ----------


@pytest.mark.unit
class TestLoadQueries:
    def test_loads_verified_gold_with_dest_id_dedup(self, reranker_module: Any, tmp_path: Path) -> None:
        gold = _write_jsonl(
            tmp_path / "gold.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "quote": "first quote",
                    "destination_context": "ctx1",
                },
                # duplicate (1, 100) — should dedup
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "quote": "first quote (different surrounding ctx)",
                    "destination_context": "ctx2",
                },
                {
                    "source_id": 2,
                    "dest_id": 200,
                    "source_cluster_id": 20,
                    "quote": "second quote",
                    "destination_context": "ctx3",
                },
            ],
        )
        queries = reranker_module._load_queries(gold)
        assert len(queries) == 2
        # First-occurrence-wins
        assert queries[0]["dest_id"] == 100
        assert queries[0]["query_text"] == "first quote"
        assert queries[1]["dest_id"] == 200


# ---------- cluster_id → text index ----------


@pytest.mark.unit
class TestLoadClusterTextIndex:
    def test_builds_cluster_to_text_map(self, reranker_module: Any, tmp_path: Path) -> None:
        corpus = _write_jsonl(
            tmp_path / "corpus.jsonl",
            [
                {"opinion_id": 1, "chunk_index": 0, "cluster_id": 100, "text": "alpha text"},
                {"opinion_id": 1, "chunk_index": 1, "cluster_id": 100, "text": "alpha cont"},
                {"opinion_id": 2, "chunk_index": 0, "cluster_id": 200, "text": "beta text"},
            ],
        )
        idx = reranker_module._load_cluster_text_index(corpus, max_chunks_per_cluster=2)
        assert 100 in idx
        assert 200 in idx
        # All chunks of a cluster concatenated (or list of chunks); contract: a string per cluster
        assert isinstance(idx[100], str)
        assert "alpha text" in idx[100]

    def test_max_chunks_caps(self, reranker_module: Any, tmp_path: Path) -> None:
        corpus = _write_jsonl(
            tmp_path / "corpus.jsonl",
            [{"opinion_id": 1, "chunk_index": i, "cluster_id": 1, "text": f"chunk{i}"} for i in range(10)],
        )
        idx = reranker_module._load_cluster_text_index(corpus, max_chunks_per_cluster=3)
        # Only first 3 chunks concatenated
        assert "chunk0" in idx[1]
        assert "chunk2" in idx[1]
        assert "chunk5" not in idx[1]


# ---------- shard range (mirrors baseline_bge_m3) ----------


@pytest.mark.unit
class TestShardRange:
    def test_partition_4way(self, reranker_module: Any) -> None:
        n = 100
        ranges = [reranker_module._shard_range(n, r, 4) for r in range(4)]
        assert sum(end - start for start, end in ranges) == n
        assert all(s < e or s == e for s, e in ranges)

    def test_disjoint(self, reranker_module: Any) -> None:
        ranges = [reranker_module._shard_range(50, r, 4) for r in range(4)]
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]

    def test_invalid_rank_raises(self, reranker_module: Any) -> None:
        with pytest.raises(ValueError):
            reranker_module._shard_range(10, 4, 4)
        with pytest.raises(ValueError):
            reranker_module._shard_range(10, -1, 4)

    def test_invalid_world_size_raises(self, reranker_module: Any) -> None:
        with pytest.raises(ValueError):
            reranker_module._shard_range(10, 0, 0)


# ---------- merge ----------


@pytest.mark.unit
class TestMergeShardResults:
    def test_concatenates_in_rank_order(self, reranker_module: Any, tmp_path: Path) -> None:
        s0 = _write_jsonl(
            tmp_path / "shard0.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [{"cluster_id": 10, "score": 0.9}],
                }
            ],
        )
        s1 = _write_jsonl(
            tmp_path / "shard1.jsonl",
            [
                {
                    "source_id": 2,
                    "dest_id": 200,
                    "source_cluster_id": 20,
                    "retrieved": [{"cluster_id": 20, "score": 0.8}],
                }
            ],
        )
        merged = tmp_path / "merged.jsonl"
        reranker_module._merge_shard_results([s0, s1], merged)
        rows = [json.loads(line) for line in merged.open()]
        assert len(rows) == 2
        assert rows[0]["source_id"] == 1
        assert rows[1]["source_id"] == 2

    def test_handles_empty_shard(self, reranker_module: Any, tmp_path: Path) -> None:
        s0 = _write_jsonl(tmp_path / "shard0.jsonl", [])
        s1 = _write_jsonl(
            tmp_path / "shard1.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [{"cluster_id": 10, "score": 0.9}],
                }
            ],
        )
        merged = tmp_path / "merged.jsonl"
        reranker_module._merge_shard_results([s0, s1], merged)
        rows = [json.loads(line) for line in merged.open()]
        assert len(rows) == 1


# ---------- pure scoring helper (no GPU) ----------


@pytest.mark.unit
class TestRerankCandidatesByScore:
    def test_descending_score_order(self, reranker_module: Any) -> None:
        candidates = [
            {"cluster_id": 1, "score": 0.5},
            {"cluster_id": 2, "score": 0.9},
            {"cluster_id": 3, "score": 0.7},
        ]
        scores = [0.5, 0.9, 0.7]
        out = reranker_module._rerank_candidates_by_score(candidates, scores, top_k=10)
        assert [c["cluster_id"] for c in out] == [2, 3, 1]

    def test_top_k_truncation(self, reranker_module: Any) -> None:
        candidates = [{"cluster_id": i, "score": 0.0} for i in range(10)]
        scores = [9.0 - i for i in range(10)]  # cluster 0 highest
        out = reranker_module._rerank_candidates_by_score(candidates, scores, top_k=3)
        assert len(out) == 3
        assert out[0]["cluster_id"] == 0
        assert out[1]["cluster_id"] == 1
        assert out[2]["cluster_id"] == 2

    def test_deterministic_tie_break(self, reranker_module: Any) -> None:
        candidates = [
            {"cluster_id": 5, "score": 0.0},
            {"cluster_id": 1, "score": 0.0},
            {"cluster_id": 3, "score": 0.0},
        ]
        scores = [0.5, 0.5, 0.5]
        out = reranker_module._rerank_candidates_by_score(candidates, scores, top_k=3)
        # Ascending cluster_id on tie
        assert [c["cluster_id"] for c in out] == [1, 3, 5]


# ---------- malformed inputs ----------


@pytest.mark.unit
class TestMalformedInputs:
    def test_load_queries_invalid_json_raises(self, reranker_module: Any, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not json\n")
        with pytest.raises(json.JSONDecodeError):
            reranker_module._load_queries(bad)

    def test_load_queries_missing_field_raises(self, reranker_module: Any, tmp_path: Path) -> None:
        gold = _write_jsonl(
            tmp_path / "gold.jsonl",
            # missing source_cluster_id
            [{"source_id": 1, "dest_id": 100, "quote": "x"}],
        )
        with pytest.raises((KeyError, ValueError)):
            reranker_module._load_queries(gold)
