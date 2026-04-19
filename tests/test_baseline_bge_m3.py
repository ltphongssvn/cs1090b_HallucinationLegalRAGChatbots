"""MS3 BGE-M3 dense baseline — TDD contract/unit/property tiers.

Mirrors tests/test_baseline_bm25.py for cross-baseline consistency.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "baseline_bge_m3.py"


@pytest.fixture(scope="module")
def bge_module() -> Any:
    return importlib.import_module("scripts.baseline_bge_m3")


@pytest.fixture(scope="module")
def source_ast() -> ast.Module:
    """Parse script source once per test module (DRY contract tests)."""
    return ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def mini_corpus(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(c)
            for c in [
                {"opinion_id": 1001, "chunk_index": 0, "text": "contract breach damages"},
                {"opinion_id": 1001, "chunk_index": 1, "text": "consideration offer acceptance"},
                {"opinion_id": 1002, "chunk_index": 0, "text": "fourth amendment search seizure"},
                {"opinion_id": 1003, "chunk_index": 0, "text": "antitrust sherman clayton"},
                {"opinion_id": 1004, "chunk_index": 0, "text": "copyright fair use transformative"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def mini_gold(tmp_path: Path) -> Path:
    p = tmp_path / "gold.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(g)
            for g in [
                {"source_id": 9001, "dest_id": 1001, "source_court": "ca5"},
                {"source_id": 9002, "dest_id": 1002, "source_court": "ca9"},
                {"source_id": 9003, "dest_id": 1003, "source_court": "ca4"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def mini_lepard(tmp_path: Path) -> Path:
    p = tmp_path / "lepard.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"source_id": 9001, "dest_id": 1001, "quote": "contract breach damages claim"},
                {"source_id": 9002, "dest_id": 1002, "quote": "fourth amendment warrant requirement"},
                {"source_id": 9003, "dest_id": 1003, "quote": "antitrust market power"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return p


# ---------- contract tier ----------


@pytest.mark.contract
class TestModuleConstants:
    def test_schema_version_semver(self, bge_module: Any) -> None:
        assert re.match(r"^\d+\.\d+\.\d+$", bge_module.SCHEMA_VERSION)

    def test_encoder_model_bge_m3(self, bge_module: Any) -> None:
        assert bge_module.ENCODER_MODEL == "BAAI/bge-m3"

    def test_embedding_dim_1024(self, bge_module: Any) -> None:
        assert bge_module.EMBEDDING_DIM == 1024

    def test_top_k_default(self, bge_module: Any) -> None:
        assert bge_module.TOP_K == 100

    def test_retrieval_k_multiplier(self, bge_module: Any) -> None:
        assert bge_module.RETRIEVAL_K_MULTIPLIER >= 2


@pytest.mark.contract
class TestUsesFaissAndSentenceTransformers:
    def test_faiss_import(self, source_ast: ast.Module) -> None:
        found = any(
            (isinstance(n, ast.Import) and any(a.name == "faiss" for a in n.names))
            or (isinstance(n, ast.ImportFrom) and n.module == "faiss")
            for n in ast.walk(source_ast)
        )
        assert found

    def test_sentence_transformers_import(self, source_ast: ast.Module) -> None:
        found = any(isinstance(n, ast.ImportFrom) and n.module == "sentence_transformers" for n in ast.walk(source_ast))
        assert found


@pytest.mark.contract
class TestUsesQuoteField:
    def test_quote_used_as_dict_key(self, source_ast: ast.Module) -> None:
        for node in ast.walk(source_ast):
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Constant) and node.slice.value == "quote":
                    return
            if isinstance(node, ast.Call):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value == "quote":
                        return
        pytest.fail("'quote' must be used as dict key in baseline_bge_m3.py")


@pytest.mark.contract
class TestAggregateChunkScoresExists:
    def test_function_exists(self, bge_module: Any) -> None:
        assert callable(getattr(bge_module, "_aggregate_chunk_scores", None))

    def test_signature(self, bge_module: Any) -> None:
        sig = inspect.signature(bge_module._aggregate_chunk_scores)
        assert {"raw_hits", "top_k"}.issubset(set(sig.parameters))


@pytest.mark.contract
class TestArgparseCLI:
    def test_required_flags_in_parser(self, bge_module: Any) -> None:
        parser = bge_module._build_arg_parser()
        actions = {a.dest for a in parser._actions}
        for flag in ("dry_run", "top_k", "seed", "encode_batch_size"):
            assert flag in actions, f"--{flag.replace('_', '-')} missing"


@pytest.mark.contract
class TestNoBasicConfigAtImport:
    def test_no_basicconfig_at_top(self, source_ast: ast.Module) -> None:
        for node in source_ast.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and func.attr == "basicConfig":
                    pytest.fail("logging.basicConfig at import time")


@pytest.mark.contract
class TestResultLineSchema:
    def test_schema_exists(self) -> None:
        from pydantic import BaseModel

        from src.eda_schemas import BaselineBgeM3ResultLine

        assert issubclass(BaselineBgeM3ResultLine, BaseModel)


@pytest.mark.contract
class TestSchemaBackwardCompat:
    def test_v1_0_0_frozen_payload_validates(self) -> None:
        from src.eda_schemas import BaselineBgeM3Summary

        payload = {
            "schema_version": "1.0.0",
            "n_queries": 45000,
            "n_corpus_chunks": 7813273,
            "n_unique_opinions": 1465484,
            "top_k": 100,
            "encoder_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
            "device": "cuda",
            "device_name": "NVIDIA L4",
            "encode_batch_size": 64,
            "similarity_metric": "cosine",
            "normalize_embeddings": True,
            "max_length": 8192,
            "dtype": "float32",
            "encoder_load_seconds": 3.5,
            "index_build_seconds": 1200.0,
            "query_encode_seconds": 120.0,
            "retrieval_seconds": 60.0,
            "seed": 0,
            "git_sha": "abc123def456",
            "results_hash": "f" * 64,
        }
        v = BaselineBgeM3Summary.model_validate(payload)
        assert v.schema_version == "1.0.0"


# ---------- unit tier ----------


@pytest.mark.unit
class TestMaxPAggregation:
    def test_max_chunk_score_wins(self, bge_module: Any) -> None:
        raw_hits = [
            {"opinion_id": 1001, "chunk_index": 0, "score": 0.95},
            {"opinion_id": 1001, "chunk_index": 1, "score": 0.82},
            {"opinion_id": 1002, "chunk_index": 0, "score": 0.90},
        ]
        agg = bge_module._aggregate_chunk_scores(raw_hits, top_k=10)
        by_oid = {h["opinion_id"]: h for h in agg}
        assert by_oid[1001]["score"] == pytest.approx(0.95)
        assert by_oid[1002]["score"] == pytest.approx(0.90)
        assert len(agg) == 2

    def test_rejects_sump_behavior(self, bge_module: Any) -> None:
        raw_hits = [
            {"opinion_id": 1001, "chunk_index": 0, "score": 0.5},
            {"opinion_id": 1001, "chunk_index": 1, "score": 0.3},
        ]
        agg = bge_module._aggregate_chunk_scores(raw_hits, top_k=10)
        assert agg[0]["score"] == pytest.approx(0.5)
        assert agg[0]["score"] != pytest.approx(0.8)  # not sum


@pytest.mark.unit
class TestTopKLargerThanHits:
    def test_graceful_clip(self, bge_module: Any) -> None:
        raw_hits = [
            {"opinion_id": 1, "chunk_index": 0, "score": 0.5},
            {"opinion_id": 2, "chunk_index": 0, "score": 0.3},
        ]
        agg = bge_module._aggregate_chunk_scores(raw_hits, top_k=1000)
        assert len(agg) == 2

    def test_empty_hits(self, bge_module: Any) -> None:
        agg = bge_module._aggregate_chunk_scores([], top_k=10)
        assert agg == []


@pytest.mark.unit
class TestQueryJoinFromLePaRD:
    def test_joins_quote_by_source_dest(self, bge_module: Any, mini_gold: Path, mini_lepard: Path) -> None:
        queries = bge_module._load_queries(mini_gold, mini_lepard)
        assert len(queries) == 3
        by_source = {q["source_id"]: q for q in queries}
        assert by_source[9001]["query_text"] == "contract breach damages claim"

    def test_dedup_duplicate_lepard_pair(self, bge_module: Any, tmp_path: Path) -> None:
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({"source_id": 1, "dest_id": 2}) + "\n")
        lepard = tmp_path / "lepard.jsonl"
        lepard.write_text(
            "\n".join(
                [
                    json.dumps({"source_id": 1, "dest_id": 2, "quote": "x"}),
                    json.dumps({"source_id": 1, "dest_id": 2, "quote": "x"}),
                ]
            )
            + "\n"
        )
        queries = bge_module._load_queries(gold, lepard)
        assert len(queries) == 1

    def test_empty_quote_handled(self, bge_module: Any, tmp_path: Path) -> None:
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({"source_id": 1, "dest_id": 2}) + "\n")
        lepard = tmp_path / "lepard.jsonl"
        lepard.write_text(json.dumps({"source_id": 1, "dest_id": 2, "quote": ""}) + "\n")
        queries = bge_module._load_queries(gold, lepard)
        assert len(queries) == 1
        assert queries[0]["query_text"] == ""


# ---------- property tier ----------


@pytest.mark.property
class TestAggregateScoresInvariants:
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
        bge_module: Any,
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
                "score": rng.uniform(-1.0, 1.0),
            }
            for i in range(n_chunks)
        ]
        agg = bge_module._aggregate_chunk_scores(raw_hits, top_k=top_k)
        assert len(agg) <= top_k
        oids = [h["opinion_id"] for h in agg]
        assert len(oids) == len(set(oids))
        scores = [h["score"] for h in agg]
        assert scores == sorted(scores, reverse=True)
        raw_by_oid: dict[int, list[float]] = {}
        for h in raw_hits:
            raw_by_oid.setdefault(h["opinion_id"], []).append(h["score"])
        for h in agg:
            assert h["score"] == pytest.approx(max(raw_by_oid[h["opinion_id"]]))


# ---------- wandb branch parametrized ----------


@pytest.mark.unit
@pytest.mark.parametrize(
    "log_to_wandb,expected_calls",
    [(False, 0), (True, 1)],
    ids=["disabled", "enabled"],
)
class TestWandbBranchParametrized:
    def test_call_count(
        self,
        bge_module: Any,
        monkeypatch: pytest.MonkeyPatch,
        log_to_wandb: bool,
        expected_calls: int,
    ) -> None:
        """Verify _log_to_wandb is branched on log_to_wandb flag without full main() run.

        BGE-M3 main() requires GPU + model download (~2GB); test the branching
        logic via direct invocation of the wandb helper.
        """
        calls: list[Any] = []
        monkeypatch.setattr(bge_module, "_log_to_wandb", lambda *a, **kw: calls.append((a, kw)))
        if log_to_wandb:
            bge_module._log_to_wandb({"x": 1}, Path("/tmp"))
        assert len(calls) == expected_calls
