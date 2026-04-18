"""Contract + unit + property tests for scripts/baseline_bm25.py."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bm25_module() -> Any:
    return importlib.import_module("scripts.baseline_bm25")


# ---------- contract tier ----------


@pytest.mark.contract
class TestFileExists:
    def test_script_exists(self) -> None:
        assert (REPO_ROOT / "scripts" / "baseline_bm25.py").is_file()


@pytest.mark.contract
class TestModuleConstants:
    def test_constants(self, bm25_module: Any) -> None:
        m = bm25_module
        assert isinstance(m.SCHEMA_VERSION, str)
        assert len(m.SCHEMA_VERSION.split(".")) == 3
        assert isinstance(m.TOP_K, int) and m.TOP_K >= 100
        assert m.BM25_K1 == pytest.approx(1.5)  # README bm25s baseline
        assert m.BM25_B == pytest.approx(0.75)


@pytest.mark.contract
class TestMainSignature:
    def test_main_signature(self, bm25_module: Any) -> None:
        sig = inspect.signature(bm25_module.main)
        expected = {
            "corpus_path",
            "gold_pairs_path",
            "lepard_path",
            "out_dir",
            "top_k",
            "log_to_wandb",
            "seed",
        }
        assert expected.issubset(set(sig.parameters))


@pytest.mark.contract
class TestSummaryModelIsPydantic:
    def test_is_basemodel(self) -> None:
        from pydantic import BaseModel

        from src.eda_schemas import BaselineBM25Summary

        assert issubclass(BaselineBM25Summary, BaseModel)


@pytest.mark.contract
class TestNoBasicConfigAtImport:
    def test_no_logging_basicconfig(self) -> None:
        src = (REPO_ROOT / "scripts" / "baseline_bm25.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                fn = node.value.func
                if (
                    isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "logging"
                    and fn.attr == "basicConfig"
                ):
                    raise AssertionError("logging.basicConfig at module top-level")


@pytest.mark.contract
class TestArgparseCLI:
    def test_build_arg_parser(self, bm25_module: Any) -> None:
        parser = bm25_module._build_arg_parser()
        actions = {a.dest: a for a in parser._actions}
        for dest in ("corpus_path", "gold_pairs_path", "lepard_path", "out_dir", "top_k", "log_to_wandb", "seed"):
            assert dest in actions


@pytest.mark.contract
class TestUsesBM25s:
    def test_bm25s_import(self) -> None:
        """Must use repo-certified bm25s via AST (not substring match)."""
        src = (REPO_ROOT / "scripts" / "baseline_bm25.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name == "bm25s" for a in node.names):
                    found = True
            elif isinstance(node, ast.ImportFrom):
                if node.module == "bm25s":
                    found = True
        assert found, "scripts/baseline_bm25.py must import bm25s"


@pytest.mark.contract
class TestUsesQuoteFieldAsQuery:
    def test_quote_used_as_dict_key(self) -> None:
        """Query text must come from LePaRD 'quote' field, verified via AST as subscript or method-call arg."""
        src = (REPO_ROOT / "scripts" / "baseline_bm25.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            # Match row["quote"] or r.get("quote")
            if isinstance(node, ast.Subscript):
                idx = node.slice
                if isinstance(idx, ast.Constant) and idx.value == "quote":
                    return
            if isinstance(node, ast.Call):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value == "quote":
                        return
        raise AssertionError("'quote' must be used as a dict key/method arg in baseline_bm25.py")


# ---------- unit tier ----------


@pytest.fixture
def mini_corpus(tmp_path: Path) -> Path:
    p = tmp_path / "corpus_chunks.jsonl"
    chunks = [
        {"opinion_id": 1001, "chunk_index": 0, "text": "contract law breach damages consideration"},
        {"opinion_id": 1001, "chunk_index": 1, "text": "remedy specific performance injunction"},
        {"opinion_id": 1002, "chunk_index": 0, "text": "fourth amendment search seizure warrant probable cause"},
        {"opinion_id": 1003, "chunk_index": 0, "text": "antitrust monopoly sherman act market power"},
        {"opinion_id": 1004, "chunk_index": 0, "text": "copyright fair use transformative work"},
    ]
    p.write_text("\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def mini_gold(tmp_path: Path) -> Path:
    p = tmp_path / "gold_pairs_test.jsonl"
    pairs = [
        {"source_id": 9001, "dest_id": 1001, "source_court": "ca5"},
        {"source_id": 9002, "dest_id": 1002, "source_court": "ca9"},
        {"source_id": 9003, "dest_id": 1003, "source_court": "ca1"},
    ]
    p.write_text("\n".join(json.dumps(x) for x in pairs) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def mini_lepard(tmp_path: Path) -> Path:
    p = tmp_path / "lepard.jsonl"
    rows = [
        {"source_id": 9001, "dest_id": 1001, "quote": "breach of contract damages remedy"},
        {"source_id": 9002, "dest_id": 1002, "quote": "fourth amendment warrant requirement"},
        {"source_id": 9003, "dest_id": 1003, "quote": "antitrust sherman monopoly"},
        {
            "source_id": 9999,
            "dest_id": 9999,  # unrelated row
            "quote": "unrelated query",
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    return p


@pytest.mark.unit
class TestQueryJoinFromLePaRD:
    def test_joins_quote_by_source_dest(
        self,
        bm25_module: Any,
        mini_gold: Path,
        mini_lepard: Path,
    ) -> None:
        queries = bm25_module._load_queries(mini_gold, mini_lepard)
        assert len(queries) == 3
        by_pair = {(q["source_id"], q["dest_id"]): q for q in queries}
        assert "breach of contract" in by_pair[(9001, 1001)]["query_text"]
        assert "fourth amendment" in by_pair[(9002, 1002)]["query_text"]

    def test_missing_lepard_rows_dropped(
        self,
        bm25_module: Any,
        tmp_path: Path,
        mini_lepard: Path,
    ) -> None:
        gold = tmp_path / "g.jsonl"
        # gold pair with no matching LePaRD row
        gold.write_text(
            json.dumps(
                {
                    "source_id": 7777,
                    "dest_id": 8888,
                    "source_court": "ca5",
                }
            )
            + "\n"
        )
        queries = bm25_module._load_queries(gold, mini_lepard)
        assert queries == []


@pytest.mark.unit
class TestBM25Retrieval:
    def test_gold_document_retrievable(
        self,
        bm25_module: Any,
        mini_corpus: Path,
        mini_gold: Path,
        mini_lepard: Path,
        tmp_path: Path,
    ) -> None:
        """Gold dest_id should appear in top-k results when query matches."""
        out_dir = tmp_path / "out"
        bm25_module.main(
            corpus_path=mini_corpus,
            gold_pairs_path=mini_gold,
            lepard_path=mini_lepard,
            out_dir=out_dir,
            top_k=5,
            log_to_wandb=False,
            seed=0,
        )
        results_path = out_dir / "bm25_results.jsonl"
        assert results_path.exists()
        results = [json.loads(line) for line in results_path.read_text().splitlines()]
        assert len(results) == 3
        for r in results:
            assert "source_id" in r
            assert "dest_id" in r  # gold
            assert "retrieved" in r
            assert len(r["retrieved"]) <= 5
            # Each retrieved entry has opinion_id + score
            for hit in r["retrieved"]:
                assert "opinion_id" in hit
                assert "score" in hit

    def test_top_k_respected(
        self,
        bm25_module: Any,
        mini_corpus: Path,
        mini_gold: Path,
        mini_lepard: Path,
        tmp_path: Path,
    ) -> None:
        out_dir = tmp_path / "out"
        bm25_module.main(
            corpus_path=mini_corpus,
            gold_pairs_path=mini_gold,
            lepard_path=mini_lepard,
            out_dir=out_dir,
            top_k=2,
            log_to_wandb=False,
            seed=0,
        )
        results = [json.loads(line) for line in (out_dir / "bm25_results.jsonl").read_text().splitlines()]
        for r in results:
            assert len(r["retrieved"]) <= 2


@pytest.mark.unit
class TestChunkAggregationToOpinion:
    """Retrieval returns opinion_id, but BM25 scores chunks — must aggregate."""

    def test_multiple_chunks_same_opinion_aggregated(
        self,
        bm25_module: Any,
        tmp_path: Path,
    ) -> None:
        # Opinion 1001 has 2 chunks, both relevant to query
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            "\n".join(
                json.dumps(c)
                for c in [
                    {"opinion_id": 1001, "chunk_index": 0, "text": "contract breach"},
                    {"opinion_id": 1001, "chunk_index": 1, "text": "contract damages"},
                    {"opinion_id": 1002, "chunk_index": 0, "text": "unrelated topic"},
                ]
            )
            + "\n"
        )
        gold = tmp_path / "gold.jsonl"
        gold.write_text(
            json.dumps(
                {
                    "source_id": 9001,
                    "dest_id": 1001,
                    "source_court": "ca5",
                }
            )
            + "\n"
        )
        lepard = tmp_path / "lepard.jsonl"
        lepard.write_text(
            json.dumps(
                {
                    "source_id": 9001,
                    "dest_id": 1001,
                    "quote": "contract breach damages",
                }
            )
            + "\n"
        )
        out_dir = tmp_path / "out"
        bm25_module.main(
            corpus_path=corpus,
            gold_pairs_path=gold,
            lepard_path=lepard,
            out_dir=out_dir,
            top_k=10,
            log_to_wandb=False,
            seed=0,
        )
        results = [json.loads(line) for line in (out_dir / "bm25_results.jsonl").read_text().splitlines()]
        opinion_ids = [h["opinion_id"] for h in results[0]["retrieved"]]
        # Opinion 1001 must appear exactly once (aggregated from 2 chunks)
        assert opinion_ids.count(1001) == 1


@pytest.mark.unit
class TestSummarySchema:
    def test_summary_validates(self) -> None:
        from src.eda_schemas import BaselineBM25Summary

        data = {
            "schema_version": "1.0.0",
            "n_queries": 3,
            "n_corpus_chunks": 5,
            "n_unique_opinions": 4,
            "top_k": 10,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "index_build_seconds": 1.23,
            "retrieval_seconds": 2.34,
            "seed": 0,
            "git_sha": "abc123def456",
            "results_hash": "a" * 64,
        }
        mod = BaselineBM25Summary(**data)
        assert mod.n_queries == 3


@pytest.mark.unit
class TestProducedSummaryValidates:
    def test_real_summary_parses(
        self,
        bm25_module: Any,
        mini_corpus: Path,
        mini_gold: Path,
        mini_lepard: Path,
        tmp_path: Path,
    ) -> None:
        from src.eda_schemas import BaselineBM25Summary

        out_dir = tmp_path / "out"
        bm25_module.main(
            corpus_path=mini_corpus,
            gold_pairs_path=mini_gold,
            lepard_path=mini_lepard,
            out_dir=out_dir,
            top_k=5,
            log_to_wandb=False,
            seed=0,
        )
        summary_path = out_dir / "bm25_summary.json"
        validated = BaselineBM25Summary.model_validate_json(summary_path.read_bytes())
        assert validated.n_queries == 3
        assert validated.top_k == 5
        assert validated.bm25_k1 == pytest.approx(1.5)
        assert len(validated.results_hash) == 64


@pytest.mark.unit
class TestDeterminism:
    def test_same_seed_identical_results(
        self,
        bm25_module: Any,
        mini_corpus: Path,
        mini_gold: Path,
        mini_lepard: Path,
        tmp_path: Path,
    ) -> None:
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        for out in (out1, out2):
            bm25_module.main(
                corpus_path=mini_corpus,
                gold_pairs_path=mini_gold,
                lepard_path=mini_lepard,
                out_dir=out,
                top_k=5,
                log_to_wandb=False,
                seed=42,
            )
        r1 = (out1 / "bm25_results.jsonl").read_text()
        r2 = (out2 / "bm25_results.jsonl").read_text()
        assert r1 == r2


@pytest.mark.unit
class TestWandbBranchBehavior:
    def test_false_does_not_call(
        self,
        bm25_module: Any,
        mini_corpus: Path,
        mini_gold: Path,
        mini_lepard: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[Any] = []
        monkeypatch.setattr(
            bm25_module,
            "_log_to_wandb",
            lambda *a, **kw: calls.append((a, kw)),
        )
        bm25_module.main(
            corpus_path=mini_corpus,
            gold_pairs_path=mini_gold,
            lepard_path=mini_lepard,
            out_dir=tmp_path / "out",
            top_k=5,
            log_to_wandb=False,
            seed=0,
        )
        assert calls == []

    def test_true_calls_exactly_once(
        self,
        bm25_module: Any,
        mini_corpus: Path,
        mini_gold: Path,
        mini_lepard: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[Any] = []
        monkeypatch.setattr(
            bm25_module,
            "_log_to_wandb",
            lambda *a, **kw: calls.append((a, kw)),
        )
        bm25_module.main(
            corpus_path=mini_corpus,
            gold_pairs_path=mini_gold,
            lepard_path=mini_lepard,
            out_dir=tmp_path / "out",
            top_k=5,
            log_to_wandb=True,
            seed=0,
        )
        assert len(calls) == 1


# ---------- property tier ----------


@pytest.mark.property
class TestRetrievalInvariants:
    @given(
        top_k=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_results_never_exceed_top_k(
        self,
        bm25_module: Any,
        mini_corpus: Path,
        mini_gold: Path,
        mini_lepard: Path,
        tmp_path: Path,
        top_k: int,
        seed: int,
    ) -> None:
        out_dir = tmp_path / f"run_{seed}_{top_k}"
        bm25_module.main(
            corpus_path=mini_corpus,
            gold_pairs_path=mini_gold,
            lepard_path=mini_lepard,
            out_dir=out_dir,
            top_k=top_k,
            log_to_wandb=False,
            seed=seed,
        )
        results = [json.loads(line) for line in (out_dir / "bm25_results.jsonl").read_text().splitlines()]
        for r in results:
            assert len(r["retrieved"]) <= top_k
            # No duplicate opinion_ids in results (aggregation invariant)
            oids = [h["opinion_id"] for h in r["retrieved"]]
            assert len(oids) == len(set(oids))
            # Scores monotonically decreasing
            scores = [h["score"] for h in r["retrieved"]]
            assert scores == sorted(scores, reverse=True)


# ---------- second hardening round ----------


@pytest.mark.contract
class TestAggregateChunkScoresExists:
    """Pure function for chunk→opinion aggregation, separate from I/O orchestration."""

    def test_function_exists(self, bm25_module: Any) -> None:
        assert callable(getattr(bm25_module, "_aggregate_chunk_scores", None))

    def test_signature(self, bm25_module: Any) -> None:
        sig = inspect.signature(bm25_module._aggregate_chunk_scores)
        expected = {"raw_hits", "top_k"}
        assert expected.issubset(set(sig.parameters))


@pytest.mark.unit
class TestMaxPAggregation:
    """README legal-retrieval default: MaxP (max chunk score per opinion), not SumP."""

    def test_max_chunk_score_wins(self, bm25_module: Any) -> None:
        # Opinion 1001 has chunks scoring [10.5, 8.2, 7.1] — opinion score = 10.5 (MaxP)
        # Opinion 1002 has chunks scoring [9.0, 9.0]       — opinion score = 9.0
        raw_hits = [
            {"opinion_id": 1001, "chunk_index": 0, "score": 10.5},
            {"opinion_id": 1001, "chunk_index": 1, "score": 8.2},
            {"opinion_id": 1001, "chunk_index": 2, "score": 7.1},
            {"opinion_id": 1002, "chunk_index": 0, "score": 9.0},
            {"opinion_id": 1002, "chunk_index": 1, "score": 9.0},
        ]
        aggregated = bm25_module._aggregate_chunk_scores(raw_hits, top_k=10)
        assert len(aggregated) == 2
        by_oid = {h["opinion_id"]: h for h in aggregated}
        assert by_oid[1001]["score"] == pytest.approx(10.5)
        assert by_oid[1002]["score"] == pytest.approx(9.0)

    def test_rejects_sump_behavior(self, bm25_module: Any) -> None:
        """SumP would make 1001 = 25.8; MaxP keeps it at 10.5."""
        raw_hits = [
            {"opinion_id": 1001, "chunk_index": 0, "score": 10.5},
            {"opinion_id": 1001, "chunk_index": 1, "score": 8.2},
            {"opinion_id": 1001, "chunk_index": 2, "score": 7.1},
        ]
        aggregated = bm25_module._aggregate_chunk_scores(raw_hits, top_k=10)
        assert aggregated[0]["score"] != pytest.approx(25.8), "MaxP, not SumP"
        assert aggregated[0]["score"] == pytest.approx(10.5)


@pytest.mark.unit
class TestRetrievalAccuracyOnCuratedFixture:
    """Guard against fake-algorithm regression: real BM25 must rank gold doc in top-1."""

    def test_gold_doc_ranked_first_when_query_matches_only_gold(
        self,
        bm25_module: Any,
        tmp_path: Path,
    ) -> None:
        # Curated fixture: gold doc has a near-exact token overlap with query;
        # distractor docs are topically unrelated. Real BM25 must place gold @ rank 1.
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            "\n".join(
                json.dumps(c)
                for c in [
                    {
                        "opinion_id": 100,
                        "chunk_index": 0,
                        "text": "qualified immunity clearly established right constitutional violation",
                    },
                    {
                        "opinion_id": 200,
                        "chunk_index": 0,
                        "text": "maritime jurisdiction admiralty shipping vessel cargo",
                    },
                    {"opinion_id": 300, "chunk_index": 0, "text": "copyright fair use transformative parody"},
                    {"opinion_id": 400, "chunk_index": 0, "text": "bankruptcy chapter eleven reorganization creditor"},
                ]
            )
            + "\n"
        )
        gold = tmp_path / "gold.jsonl"
        gold.write_text(
            json.dumps(
                {
                    "source_id": 9001,
                    "dest_id": 100,
                    "source_court": "ca5",
                }
            )
            + "\n"
        )
        lepard = tmp_path / "lepard.jsonl"
        lepard.write_text(
            json.dumps(
                {
                    "source_id": 9001,
                    "dest_id": 100,
                    "quote": "qualified immunity clearly established constitutional",
                }
            )
            + "\n"
        )
        out_dir = tmp_path / "out"
        bm25_module.main(
            corpus_path=corpus,
            gold_pairs_path=gold,
            lepard_path=lepard,
            out_dir=out_dir,
            top_k=4,
            log_to_wandb=False,
            seed=0,
        )
        results = [json.loads(line) for line in (out_dir / "bm25_results.jsonl").read_text().splitlines()]
        assert results[0]["retrieved"][0]["opinion_id"] == 100, (
            "Gold opinion 100 must rank first for matching query — if this fails, BM25 is producing random/fake scores"
        )


# ---------- property tier on pure function ----------


@pytest.mark.property
class TestAggregateScoresInvariants:
    """Hypothesis over _aggregate_chunk_scores pure function (no I/O)."""

    @given(
        n_chunks=st.integers(min_value=1, max_value=50),
        n_opinions=st.integers(min_value=1, max_value=10),
        top_k=st.integers(min_value=1, max_value=20),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invariants(
        self,
        bm25_module: Any,
        n_chunks: int,
        n_opinions: int,
        top_k: int,
        seed: int,
    ) -> None:
        import random as _r

        rng = _r.Random(seed)
        raw_hits = [
            {
                "opinion_id": rng.randint(1, n_opinions),
                "chunk_index": i,
                "score": rng.uniform(0.0, 100.0),
            }
            for i in range(n_chunks)
        ]
        aggregated = bm25_module._aggregate_chunk_scores(raw_hits, top_k=top_k)

        # Invariant 1: output length ≤ top_k
        assert len(aggregated) <= top_k

        # Invariant 2: no duplicate opinion_ids (MaxP aggregation)
        oids = [h["opinion_id"] for h in aggregated]
        assert len(oids) == len(set(oids))

        # Invariant 3: scores monotonically decreasing
        scores = [h["score"] for h in aggregated]
        assert scores == sorted(scores, reverse=True)

        # Invariant 4: every aggregated score equals the MAX of that opinion's raw chunks
        raw_by_oid: dict[int, list[float]] = {}
        for h in raw_hits:
            raw_by_oid.setdefault(h["opinion_id"], []).append(h["score"])
        for h in aggregated:
            expected_max = max(raw_by_oid[h["opinion_id"]])
            assert h["score"] == pytest.approx(expected_max)
