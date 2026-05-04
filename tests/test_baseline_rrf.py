# tests/test_baseline_rrf.py
"""Tests for scripts.baseline_rrf — Reciprocal Rank Fusion of BM25 + BGE-M3 results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def rrf_module() -> Any:
    from scripts import baseline_rrf

    return baseline_rrf


# ---------- fixtures ----------


@pytest.fixture
def toy_bm25(tmp_path: Path) -> Path:
    """3 queries; cluster_id retrieval results."""
    return _write_jsonl(
        tmp_path / "bm25.jsonl",
        [
            {
                "source_id": 1,
                "dest_id": 100,
                "source_cluster_id": 10,
                "retrieved": [
                    {"cluster_id": 10, "score": 5.0},
                    {"cluster_id": 20, "score": 4.0},
                    {"cluster_id": 30, "score": 3.0},
                ],
            },
            {
                "source_id": 2,
                "dest_id": 200,
                "source_cluster_id": 20,
                "retrieved": [
                    {"cluster_id": 99, "score": 2.5},
                    {"cluster_id": 20, "score": 2.0},
                ],
            },
            {
                "source_id": 3,
                "dest_id": 300,
                "source_cluster_id": 30,
                "retrieved": [
                    {"cluster_id": 88, "score": 1.0},
                ],
            },
        ],
    )


@pytest.fixture
def toy_bge(tmp_path: Path) -> Path:
    """3 queries; cluster_id retrieval results, partly overlapping with BM25."""
    return _write_jsonl(
        tmp_path / "bge.jsonl",
        [
            {
                "source_id": 1,
                "dest_id": 100,
                "source_cluster_id": 10,
                "retrieved": [
                    {"cluster_id": 20, "score": 0.9},
                    {"cluster_id": 10, "score": 0.85},
                ],
            },
            {
                "source_id": 2,
                "dest_id": 200,
                "source_cluster_id": 20,
                "retrieved": [
                    {"cluster_id": 20, "score": 0.95},
                    {"cluster_id": 99, "score": 0.80},
                ],
            },
            {
                "source_id": 3,
                "dest_id": 300,
                "source_cluster_id": 30,
                "retrieved": [
                    {"cluster_id": 30, "score": 0.99},
                ],
            },
        ],
    )


# ---------- contract ----------


@pytest.mark.contract
class TestContract:
    def test_rrf_score_exists(self, rrf_module: Any) -> None:
        assert callable(getattr(rrf_module, "_rrf_score", None))

    def test_fuse_two_runs_exists(self, rrf_module: Any) -> None:
        assert callable(getattr(rrf_module, "fuse_two_runs", None))

    def test_main_exists(self, rrf_module: Any) -> None:
        assert callable(getattr(rrf_module, "main", None))

    def test_default_rrf_k(self, rrf_module: Any) -> None:
        # Industry-standard RRF k constant per Cormack et al. 2009
        assert rrf_module.DEFAULT_RRF_K == 60

    def test_schema_version(self, rrf_module: Any) -> None:
        import re

        assert re.match(r"^\d+\.\d+\.\d+$", rrf_module.SCHEMA_VERSION)


# ---------- pure RRF math ----------


@pytest.mark.unit
class TestRRFScore:
    def test_rank_1(self, rrf_module: Any) -> None:
        # rank=1, k=60 → 1/61
        assert rrf_module._rrf_score(rank=1, k=60) == pytest.approx(1.0 / 61)

    def test_rank_increases_score_decreases(self, rrf_module: Any) -> None:
        scores = [rrf_module._rrf_score(rank=r, k=60) for r in range(1, 11)]
        for a, b in zip(scores, scores[1:], strict=False):
            assert a > b

    def test_zero_rank_raises(self, rrf_module: Any) -> None:
        with pytest.raises(ValueError):
            rrf_module._rrf_score(rank=0, k=60)

    def test_negative_rank_raises(self, rrf_module: Any) -> None:
        with pytest.raises(ValueError):
            rrf_module._rrf_score(rank=-1, k=60)


# ---------- end-to-end fuse_two_runs ----------


@pytest.mark.unit
class TestFuseTwoRuns:
    def test_perfect_overlap_at_rank_1(self, rrf_module: Any, tmp_path: Path) -> None:
        """If both retrievers put gold at rank 1, fused rank 1 is also gold."""
        bm25 = _write_jsonl(
            tmp_path / "bm25.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [{"cluster_id": 10, "score": 5.0}, {"cluster_id": 20, "score": 4.0}],
                }
            ],
        )
        bge = _write_jsonl(
            tmp_path / "bge.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [{"cluster_id": 10, "score": 0.9}, {"cluster_id": 20, "score": 0.85}],
                }
            ],
        )
        out = tmp_path / "rrf.jsonl"
        result = rrf_module.fuse_two_runs(
            bm25_path=bm25,
            bge_m3_path=bge,
            out_path=out,
            top_k=10,
            rrf_k=60,
        )
        assert result["n_queries"] == 1
        rows = [json.loads(line) for line in out.open()]
        assert rows[0]["retrieved"][0]["cluster_id"] == 10

    def test_complementary_signal(self, rrf_module: Any, tmp_path: Path) -> None:
        """Cluster found by both retrievers ranks higher than cluster found by one."""
        bm25 = _write_jsonl(
            tmp_path / "bm25.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [
                        {"cluster_id": 10, "score": 5.0},  # rank 1 in BM25
                        {"cluster_id": 99, "score": 4.0},  # rank 2 in BM25
                    ],
                }
            ],
        )
        bge = _write_jsonl(
            tmp_path / "bge.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [
                        {"cluster_id": 88, "score": 0.9},  # rank 1 in BGE only
                        {"cluster_id": 10, "score": 0.85},  # rank 2 in BGE
                    ],
                }
            ],
        )
        out = tmp_path / "rrf.jsonl"
        rrf_module.fuse_two_runs(
            bm25_path=bm25,
            bge_m3_path=bge,
            out_path=out,
            top_k=10,
            rrf_k=60,
        )
        rows = [json.loads(line) for line in out.open()]
        ranked = [hit["cluster_id"] for hit in rows[0]["retrieved"]]
        # cluster 10: rank 1 BM25 (1/61) + rank 2 BGE (1/62) ~= 0.0325
        # cluster 99: rank 2 BM25 (1/62)                      = 0.0161
        # cluster 88: rank 1 BGE (1/61)                       = 0.0164
        assert ranked[0] == 10  # appears in both, must rank highest
        assert ranked.index(10) < ranked.index(88)
        assert ranked.index(10) < ranked.index(99)

    def test_top_k_cutoff(self, rrf_module: Any, tmp_path: Path) -> None:
        """Output truncated to top_k per query."""
        bm25 = _write_jsonl(
            tmp_path / "bm25.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [{"cluster_id": i, "score": 100 - i} for i in range(1, 21)],
                }
            ],
        )
        bge = _write_jsonl(
            tmp_path / "bge.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [{"cluster_id": i, "score": 1.0 - i * 0.01} for i in range(21, 41)],
                }
            ],
        )
        out = tmp_path / "rrf.jsonl"
        rrf_module.fuse_two_runs(
            bm25_path=bm25,
            bge_m3_path=bge,
            out_path=out,
            top_k=5,
            rrf_k=60,
        )
        row = json.loads(out.read_text().strip())
        assert len(row["retrieved"]) == 5

    def test_alignment_mismatch_raises(self, rrf_module: Any, tmp_path: Path) -> None:
        bm25 = _write_jsonl(
            tmp_path / "bm25.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "source_cluster_id": 10,
                    "retrieved": [{"cluster_id": 10, "score": 1.0}],
                }
            ],
        )
        bge = _write_jsonl(
            tmp_path / "bge.jsonl",
            [
                {
                    "source_id": 2,
                    "dest_id": 200,
                    "source_cluster_id": 20,
                    "retrieved": [{"cluster_id": 20, "score": 1.0}],
                }
            ],
        )
        out = tmp_path / "rrf.jsonl"
        with pytest.raises(ValueError, match="aligned"):
            rrf_module.fuse_two_runs(
                bm25_path=bm25,
                bge_m3_path=bge,
                out_path=out,
                top_k=5,
                rrf_k=60,
            )

    def test_output_schema(self, rrf_module: Any, toy_bm25: Path, toy_bge: Path, tmp_path: Path) -> None:
        out = tmp_path / "rrf.jsonl"
        rrf_module.fuse_two_runs(
            bm25_path=toy_bm25,
            bge_m3_path=toy_bge,
            out_path=out,
            top_k=10,
            rrf_k=60,
        )
        rows = [json.loads(line) for line in out.open()]
        assert len(rows) == 3
        for r in rows:
            assert set(r.keys()) >= {"source_id", "dest_id", "source_cluster_id", "retrieved"}
            for hit in r["retrieved"]:
                assert set(hit.keys()) == {"cluster_id", "score"}

    def test_three_query_example(self, rrf_module: Any, toy_bm25: Path, toy_bge: Path, tmp_path: Path) -> None:
        out = tmp_path / "rrf.jsonl"
        result = rrf_module.fuse_two_runs(
            bm25_path=toy_bm25,
            bge_m3_path=toy_bge,
            out_path=out,
            top_k=10,
            rrf_k=60,
        )
        assert result["n_queries"] == 3
        assert result["rrf_k"] == 60


# ---------- property tests ----------


@pytest.mark.property
class TestRRFInvariants:
    @given(rank=st.integers(min_value=1, max_value=1000), k=st.integers(min_value=1, max_value=200))
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_score_in_unit_interval(self, rrf_module: Any, rank: int, k: int) -> None:
        s = rrf_module._rrf_score(rank=rank, k=k)
        assert 0 < s <= 1.0 / (k + 1)

    @given(
        rank_a=st.integers(min_value=1, max_value=100),
        rank_b=st.integers(min_value=1, max_value=100),
        k=st.integers(min_value=1, max_value=200),
    )
    @settings(
        max_examples=80,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_lower_rank_higher_score(self, rrf_module: Any, rank_a: int, rank_b: int, k: int) -> None:
        if rank_a == rank_b:
            assert rrf_module._rrf_score(rank=rank_a, k=k) == rrf_module._rrf_score(rank=rank_b, k=k)
        elif rank_a < rank_b:
            assert rrf_module._rrf_score(rank=rank_a, k=k) > rrf_module._rrf_score(rank=rank_b, k=k)


# ---------- malformed inputs ----------


@pytest.mark.unit
class TestMalformedInputs:
    def test_invalid_json_raises(self, rrf_module: Any, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not valid json\n")
        ok = _write_jsonl(
            tmp_path / "ok.jsonl",
            [{"source_id": 1, "dest_id": 100, "source_cluster_id": 10, "retrieved": []}],
        )
        out = tmp_path / "rrf.jsonl"
        with pytest.raises(json.JSONDecodeError):
            rrf_module.fuse_two_runs(
                bm25_path=bad,
                bge_m3_path=ok,
                out_path=out,
                top_k=5,
                rrf_k=60,
            )

    def test_empty_files_zero_queries(self, rrf_module: Any, tmp_path: Path) -> None:
        a = _write_jsonl(tmp_path / "a.jsonl", [])
        b = _write_jsonl(tmp_path / "b.jsonl", [])
        out = tmp_path / "rrf.jsonl"
        result = rrf_module.fuse_two_runs(
            bm25_path=a,
            bge_m3_path=b,
            out_path=out,
            top_k=5,
            rrf_k=60,
        )
        assert result["n_queries"] == 0
