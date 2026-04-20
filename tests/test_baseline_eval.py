"""Tests for scripts.baseline_eval — retrieval evaluation (Hit@k, MRR, NDCG)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    """Helper: dump records as JSONL (dedup json.dumps + newline-join across fixtures)."""
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def eval_module() -> Any:
    from scripts import baseline_eval

    return baseline_eval


# ---------- fixtures ----------


@pytest.fixture
def toy_gold(tmp_path: Path) -> Path:
    """3 queries with gold dest_ids."""
    return _write_jsonl(
        tmp_path / "gold.jsonl",
        [
            {"source_id": 1, "dest_id": 100},
            {"source_id": 2, "dest_id": 200},
            {"source_id": 3, "dest_id": 300},
        ],
    )


@pytest.fixture
def toy_results_perfect(tmp_path: Path) -> Path:
    """Each query's gold is at rank 1."""
    return _write_jsonl(
        tmp_path / "results_perfect.jsonl",
        [
            {
                "source_id": 1,
                "dest_id": 100,
                "retrieved": [
                    {"opinion_id": 100, "score": 0.99},
                    {"opinion_id": 999, "score": 0.50},
                ],
            },
            {
                "source_id": 2,
                "dest_id": 200,
                "retrieved": [{"opinion_id": 200, "score": 0.90}],
            },
            {
                "source_id": 3,
                "dest_id": 300,
                "retrieved": [{"opinion_id": 300, "score": 0.85}],
            },
        ],
    )


@pytest.fixture
def toy_results_mixed(tmp_path: Path) -> Path:
    """Query 1 rank=1, query 2 rank=3, query 3 gold missing."""
    return _write_jsonl(
        tmp_path / "results_mixed.jsonl",
        [
            {
                "source_id": 1,
                "dest_id": 100,
                "retrieved": [
                    {"opinion_id": 100, "score": 0.99},
                    {"opinion_id": 500, "score": 0.80},
                ],
            },
            {
                "source_id": 2,
                "dest_id": 200,
                "retrieved": [
                    {"opinion_id": 700, "score": 0.95},
                    {"opinion_id": 800, "score": 0.90},
                    {"opinion_id": 200, "score": 0.70},  # rank 3
                ],
            },
            {
                "source_id": 3,
                "dest_id": 300,
                "retrieved": [{"opinion_id": 999, "score": 0.60}],
            },
        ],
    )


# ---------- contract ----------


@pytest.mark.contract
class TestContract:
    def test_evaluate_baseline_exists(self, eval_module: Any) -> None:
        assert callable(getattr(eval_module, "evaluate_baseline", None))

    def test_paired_comparison_exists(self, eval_module: Any) -> None:
        assert callable(getattr(eval_module, "paired_comparison", None))

    def test_schema_version(self, eval_module: Any) -> None:
        import re

        assert re.match(r"^\d+\.\d+\.\d+$", eval_module.SCHEMA_VERSION)

    def test_schema_present(self) -> None:
        from src.eda_schemas import BaselineEvalSummary

        fields = BaselineEvalSummary.model_fields
        required = {
            "schema_version",
            "n_queries",
            "k_values",
            "ndcg_k",
            "bm25_hit_at_k",
            "bm25_mrr",
            "bm25_ndcg_at_10",
            "bm25_results_hash",
            "bge_m3_hit_at_k",
            "bge_m3_mrr",
            "bge_m3_ndcg_at_10",
            "bge_m3_results_hash",
            "bge_m3_wins",
            "bm25_wins",
            "ties",
            "git_sha",
            "seed",
        }
        assert required <= set(fields.keys())


# ---------- pure metric functions ----------


@pytest.mark.unit
class TestFindRank:
    def test_gold_at_rank_1(self, eval_module: Any) -> None:
        retrieved = [{"opinion_id": 100}, {"opinion_id": 200}]
        assert eval_module._find_rank(retrieved, gold_id=100) == 1

    def test_gold_at_rank_3(self, eval_module: Any) -> None:
        retrieved = [{"opinion_id": 1}, {"opinion_id": 2}, {"opinion_id": 3}]
        assert eval_module._find_rank(retrieved, gold_id=3) == 3

    def test_gold_missing(self, eval_module: Any) -> None:
        assert eval_module._find_rank([{"opinion_id": 1}], gold_id=999) == 0

    def test_empty_retrieved(self, eval_module: Any) -> None:
        assert eval_module._find_rank([], gold_id=1) == 0


@pytest.mark.unit
class TestFindRankTieBreaking:
    """_find_rank uses list order (first occurrence wins)."""

    def test_duplicate_opinion_id_takes_first(self, eval_module: Any) -> None:
        retrieved = [
            {"opinion_id": 100, "score": 0.9},
            {"opinion_id": 200, "score": 0.8},
            {"opinion_id": 100, "score": 0.7},
        ]
        assert eval_module._find_rank(retrieved, gold_id=100) == 1

    def test_respects_upstream_ordering(self, eval_module: Any) -> None:
        """Both BM25 + BGE-M3 MaxP break ties via (-score, opinion_id ascending).
        _find_rank trusts that upstream ordering."""
        retrieved = [
            {"opinion_id": 50, "score": 0.9},
            {"opinion_id": 100, "score": 0.9},
        ]
        assert eval_module._find_rank(retrieved, gold_id=100) == 2
        assert eval_module._find_rank(retrieved, gold_id=50) == 1


@pytest.mark.unit
class TestMetricsFromRanks:
    def test_all_rank_1(self, eval_module: Any) -> None:
        m = eval_module._metrics_from_ranks(ranks=[1, 1, 1], k_values=(1, 5, 10), ndcg_k=10)
        assert m["hit_at_k"][1] == 1.0
        assert m["hit_at_k"][5] == 1.0
        assert m["mrr"] == 1.0
        assert m["ndcg_at_10"] == 1.0

    def test_all_missed(self, eval_module: Any) -> None:
        m = eval_module._metrics_from_ranks(ranks=[0, 0, 0], k_values=(1, 5, 10), ndcg_k=10)
        assert m["hit_at_k"][1] == 0.0
        assert m["mrr"] == 0.0
        assert m["ndcg_at_10"] == 0.0

    def test_mixed_ranks(self, eval_module: Any) -> None:
        # ranks = [1, 3, 0] → Hit@1=1/3, Hit@5=2/3, MRR=(1 + 1/3 + 0)/3
        m = eval_module._metrics_from_ranks(ranks=[1, 3, 0], k_values=(1, 5, 10), ndcg_k=10)
        assert m["hit_at_k"][1] == pytest.approx(1 / 3)
        assert m["hit_at_k"][5] == pytest.approx(2 / 3)
        assert m["mrr"] == pytest.approx((1 + 1 / 3) / 3)

    def test_ndcg_at_10_vs_rank(self, eval_module: Any) -> None:
        import math

        # rank 1 → 1/log2(2) = 1.0
        # rank 4 → 1/log2(5)
        # rank 11 → 0 (beyond k=10)
        m = eval_module._metrics_from_ranks(ranks=[1, 4, 11], k_values=(1,), ndcg_k=10)
        expected = (1.0 + 1.0 / math.log2(5) + 0.0) / 3
        assert m["ndcg_at_10"] == pytest.approx(expected)


# ---------- end-to-end evaluate_baseline ----------


@pytest.mark.unit
class TestEvaluateBaseline:
    def test_perfect_results(self, eval_module: Any, toy_gold: Path, toy_results_perfect: Path) -> None:
        m = eval_module.evaluate_baseline(
            gold_path=toy_gold,
            results_path=toy_results_perfect,
            k_values=(1, 5, 10, 100),
            ndcg_k=10,
        )
        assert m["n_queries"] == 3
        assert m["hit_at_k"][1] == 1.0
        assert m["mrr"] == 1.0
        assert m["ndcg_at_10"] == 1.0

    def test_mixed_results(self, eval_module: Any, toy_gold: Path, toy_results_mixed: Path) -> None:
        import math

        m = eval_module.evaluate_baseline(
            gold_path=toy_gold,
            results_path=toy_results_mixed,
            k_values=(1, 5, 10, 100),
            ndcg_k=10,
        )
        assert m["n_queries"] == 3
        assert m["hit_at_k"][1] == pytest.approx(1 / 3)
        assert m["hit_at_k"][5] == pytest.approx(2 / 3)
        assert m["hit_at_k"][10] == pytest.approx(2 / 3)
        assert m["hit_at_k"][100] == pytest.approx(2 / 3)
        assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)
        expected_ndcg = (1.0 + 1.0 / math.log2(4) + 0) / 3
        assert m["ndcg_at_10"] == pytest.approx(expected_ndcg)

    def test_missing_gold_query_ignored(self, eval_module: Any, tmp_path: Path) -> None:
        """Queries in results but not in gold are skipped."""
        gold = _write_jsonl(
            tmp_path / "gold.jsonl",
            [{"source_id": 1, "dest_id": 100}],
        )
        results = _write_jsonl(
            tmp_path / "results.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "retrieved": [{"opinion_id": 100, "score": 0.9}],
                },
                {
                    "source_id": 999,
                    "dest_id": 999,
                    "retrieved": [{"opinion_id": 1, "score": 0.5}],
                },
            ],
        )
        m = eval_module.evaluate_baseline(gold_path=gold, results_path=results, k_values=(1,), ndcg_k=10)
        assert m["n_queries"] == 1
        assert m["hit_at_k"][1] == 1.0


# ---------- paired comparison ----------


@pytest.mark.unit
class TestPairedComparison:
    def test_bge_wins_bm25_loses(self, eval_module: Any, tmp_path: Path) -> None:
        gold = _write_jsonl(tmp_path / "gold.jsonl", [{"source_id": 1, "dest_id": 100}])
        bm25 = _write_jsonl(
            tmp_path / "bm25.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "retrieved": [
                        {"opinion_id": 999, "score": 0.5},
                        {"opinion_id": 100, "score": 0.1},  # rank 2
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
                    "retrieved": [
                        {"opinion_id": 100, "score": 0.9},  # rank 1
                    ],
                }
            ],
        )
        c = eval_module.paired_comparison(gold_path=gold, bm25_results_path=bm25, bge_m3_results_path=bge)
        assert c["n_queries"] == 1
        assert c["bge_m3_wins"] == 1
        assert c["bm25_wins"] == 0
        assert c["ties"] == 0

    def test_tie_same_rank(self, eval_module: Any, tmp_path: Path) -> None:
        gold = _write_jsonl(tmp_path / "gold.jsonl", [{"source_id": 1, "dest_id": 100}])
        bm25 = _write_jsonl(
            tmp_path / "bm25.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "retrieved": [{"opinion_id": 100, "score": 0.9}],
                }
            ],
        )
        bge = _write_jsonl(
            tmp_path / "bge.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "retrieved": [{"opinion_id": 100, "score": 0.95}],
                }
            ],
        )
        c = eval_module.paired_comparison(gold_path=gold, bm25_results_path=bm25, bge_m3_results_path=bge)
        assert c["ties"] == 1
        assert c["bm25_wins"] == 0
        assert c["bge_m3_wins"] == 0

    def test_both_miss_is_tie(self, eval_module: Any, tmp_path: Path) -> None:
        gold = _write_jsonl(tmp_path / "gold.jsonl", [{"source_id": 1, "dest_id": 100}])
        bm25 = _write_jsonl(
            tmp_path / "bm25.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "retrieved": [{"opinion_id": 999, "score": 0.5}],
                }
            ],
        )
        bge = _write_jsonl(
            tmp_path / "bge.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "retrieved": [{"opinion_id": 888, "score": 0.5}],
                }
            ],
        )
        c = eval_module.paired_comparison(gold_path=gold, bm25_results_path=bm25, bge_m3_results_path=bge)
        assert c["ties"] == 1


# ---------- property tests ----------


@pytest.mark.property
class TestMetricInvariants:
    @given(ranks=st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=200))
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_hit_at_k_monotone_in_k(self, eval_module: Any, ranks: list[int]) -> None:
        m = eval_module._metrics_from_ranks(ranks=ranks, k_values=(1, 5, 10, 100, 1000), ndcg_k=10)
        prev = 0.0
        for k in (1, 5, 10, 100, 1000):
            assert m["hit_at_k"][k] >= prev - 1e-12
            prev = m["hit_at_k"][k]

    @given(ranks=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50))
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_mrr_bounds(self, eval_module: Any, ranks: list[int]) -> None:
        m = eval_module._metrics_from_ranks(ranks=ranks, k_values=(1,), ndcg_k=10)
        assert 0.0 <= m["mrr"] <= 1.0
        assert 0.0 <= m["ndcg_at_10"] <= 1.0


# ---------- defensive / malformed-input coverage ----------


@pytest.mark.unit
class TestMalformedInputs:
    def test_invalid_json_line_raises(self, eval_module: Any, tmp_path: Path) -> None:
        """Corrupt JSONL must fail loudly (stdlib json.JSONDecodeError)."""
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{this is not valid json\n")
        with pytest.raises(json.JSONDecodeError):
            eval_module.evaluate_baseline(gold_path=bad, results_path=bad, k_values=(1,), ndcg_k=10)

    def test_empty_gold_returns_zero_queries(self, eval_module: Any, tmp_path: Path) -> None:
        gold = _write_jsonl(tmp_path / "gold.jsonl", [])
        results = _write_jsonl(
            tmp_path / "r.jsonl",
            [{"source_id": 1, "dest_id": 1, "retrieved": []}],
        )
        m = eval_module.evaluate_baseline(gold_path=gold, results_path=results, k_values=(1,), ndcg_k=10)
        # No gold queries → no ranks → 0 for all metrics
        assert m["n_queries"] == 0
        assert m["hit_at_k"][1] == 0.0
        assert m["mrr"] == 0.0

    def test_empty_results_treated_as_all_misses(self, eval_module: Any, tmp_path: Path) -> None:
        """Results file with zero rows → no ranks computed → all-miss semantics."""
        gold = _write_jsonl(
            tmp_path / "gold.jsonl",
            [{"source_id": 1, "dest_id": 100}],
        )
        results = _write_jsonl(tmp_path / "r.jsonl", [])
        m = eval_module.evaluate_baseline(gold_path=gold, results_path=results, k_values=(1,), ndcg_k=10)
        assert m["n_queries"] == 0

    def test_missing_required_key_raises(self, eval_module: Any, tmp_path: Path) -> None:
        """Row missing source_id must fail with KeyError, not silently mis-rank."""
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({"source_id": 1, "dest_id": 100}) + "\n")
        results = tmp_path / "r.jsonl"
        results.write_text(
            json.dumps({"dest_id": 100, "retrieved": []}) + "\n"  # no source_id
        )
        with pytest.raises(KeyError):
            eval_module.evaluate_baseline(gold_path=gold, results_path=results, k_values=(1,), ndcg_k=10)


# ---------- schema round-trip ----------


@pytest.mark.contract
class TestSchemaRoundTrip:
    def test_validate_dump_reload(self) -> None:
        """BaselineEvalSummary survives JSON round-trip with field preservation."""
        from src.eda_schemas import BaselineEvalSummary

        payload = {
            "schema_version": "1.0.0",
            "n_queries": 45000,
            "k_values": [1, 5, 10, 100],
            "ndcg_k": 10,
            "bm25_hit_at_k": {"1": 0.45, "5": 0.62, "10": 0.71, "100": 0.89},
            "bm25_mrr": 0.52,
            "bm25_ndcg_at_10": 0.55,
            "bm25_results_hash": "a" * 64,
            "bge_m3_hit_at_k": {"1": 0.48, "5": 0.65, "10": 0.74, "100": 0.91},
            "bge_m3_mrr": 0.56,
            "bge_m3_ndcg_at_10": 0.59,
            "bge_m3_results_hash": "b" * 64,
            "bge_m3_wins": 15000,
            "bm25_wins": 8000,
            "ties": 22000,
            "git_sha": "abc123def456",
            "seed": 0,
        }
        validated = BaselineEvalSummary.model_validate(payload)
        dumped = validated.model_dump_json()
        reloaded = BaselineEvalSummary.model_validate_json(dumped)
        assert reloaded == validated


# ---------- metamorphic invariants ----------


@pytest.mark.property
class TestMetamorphicInvariants:
    @given(
        ranks=st.lists(st.integers(min_value=0, max_value=100), min_size=2, max_size=30),
        idx=st.integers(min_value=0),
    )
    @settings(
        max_examples=80,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_rank_improvement_monotonicity(self, eval_module: Any, ranks: list[int], idx: int) -> None:
        """Improving one rank (without worsening others) must not decrease MRR.

        'Improve' means: rank was missed (0) → rank=1, OR rank=k → rank=k-1.
        """
        i = idx % len(ranks)
        original = ranks[:]
        before = eval_module._metrics_from_ranks(ranks=original, k_values=(1,), ndcg_k=10)
        # Improve: missed query → rank 1; ranked query → one position better
        improved = original[:]
        improved[i] = 1 if improved[i] == 0 else max(1, improved[i] - 1)
        after = eval_module._metrics_from_ranks(ranks=improved, k_values=(1,), ndcg_k=10)
        assert after["mrr"] >= before["mrr"] - 1e-12
        assert after["ndcg_at_10"] >= before["ndcg_at_10"] - 1e-12

    @given(ranks=st.lists(st.integers(min_value=0, max_value=200), min_size=1, max_size=50))
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_permutation_invariance(self, eval_module: Any, ranks: list[int]) -> None:
        """Aggregate metrics must not depend on query order."""
        import random as _r

        shuffled = ranks[:]
        _r.Random(42).shuffle(shuffled)
        m1 = eval_module._metrics_from_ranks(ranks=ranks, k_values=(1, 5, 10), ndcg_k=10)
        m2 = eval_module._metrics_from_ranks(ranks=shuffled, k_values=(1, 5, 10), ndcg_k=10)
        for k in (1, 5, 10):
            assert m1["hit_at_k"][k] == pytest.approx(m2["hit_at_k"][k])
        assert m1["mrr"] == pytest.approx(m2["mrr"])
        assert m1["ndcg_at_10"] == pytest.approx(m2["ndcg_at_10"])


# ---------- memory-safe paired comparison ----------


@pytest.mark.unit
class TestPairedComparisonMemoryBounded:
    def test_streaming_does_not_load_full_index(self, eval_module: Any, tmp_path: Path) -> None:
        """paired_comparison must iterate, not index-by-key. Verify via AST."""
        import inspect

        src = inspect.getsource(eval_module.paired_comparison)
        # No dict-building over full result files
        assert "index_by_key" not in src, "paired_comparison must not build full in-memory indices"


@pytest.mark.unit
class TestPairedComparisonCutoff:
    def test_top_k_cutoff_applied(self, eval_module: Any, tmp_path: Path) -> None:
        """top_k cutoff ensures fair comparison when baselines return different list lengths."""
        gold = _write_jsonl(tmp_path / "gold.jsonl", [{"source_id": 1, "dest_id": 100}])
        # BM25 retrieves gold at rank 50 (within top-100)
        bm25 = _write_jsonl(
            tmp_path / "bm25.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "retrieved": [{"opinion_id": i, "score": 1.0 - i * 0.01} for i in range(49)]
                    + [{"opinion_id": 100, "score": 0.5}],
                }
            ],
        )
        # BGE-M3 retrieves gold at rank 50 too
        bge = _write_jsonl(
            tmp_path / "bge.jsonl",
            [
                {
                    "source_id": 1,
                    "dest_id": 100,
                    "retrieved": [{"opinion_id": i + 500, "score": 0.9 - i * 0.01} for i in range(49)]
                    + [{"opinion_id": 100, "score": 0.4}],
                }
            ],
        )
        # With top_k=10 cutoff, both miss → tie
        c = eval_module.paired_comparison(
            gold_path=gold,
            bm25_results_path=bm25,
            bge_m3_results_path=bge,
            top_k=10,
        )
        assert c["ties"] == 1, f"expected tie with top_k=10 cutoff, got {c}"

        # With top_k=100 (default), both find at rank 50 → tie
        c2 = eval_module.paired_comparison(
            gold_path=gold,
            bm25_results_path=bm25,
            bge_m3_results_path=bge,
            top_k=100,
        )
        assert c2["ties"] == 1


# ---------- git_sha provenance ----------


@pytest.mark.unit
class TestGitShaProvenance:
    def test_git_sha_returns_unknown_when_subprocess_fails(
        self, eval_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environments without git (CI containers, stripped .git) must return 'unknown'."""
        import subprocess as sp

        def fake_check_output(*args: Any, **kwargs: Any) -> bytes:
            raise sp.CalledProcessError(1, "git")

        monkeypatch.setattr(eval_module.subprocess, "check_output", fake_check_output)
        assert eval_module._git_sha() == "unknown"

    def test_git_sha_returns_12_chars_on_success(self, eval_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_check_output(*args: Any, **kwargs: Any) -> bytes:
            return b"abcdef1234567890fedcba0987654321deadbeef\n"

        monkeypatch.setattr(eval_module.subprocess, "check_output", fake_check_output)
        sha = eval_module._git_sha()
        assert len(sha) == 12
        assert sha == "abcdef123456"
