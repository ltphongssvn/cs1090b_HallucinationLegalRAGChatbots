# tests/test_rag_generate.py
"""Tests for scripts.rag_generate — RAG generation pipeline.

Heavy GPU/model dependencies are guarded behind import-only contract checks;
end-to-end generation is verified separately by SLURM smoke runs.
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
def rag_module() -> Any:
    from scripts import rag_generate

    return rag_generate


# ---------- contract ----------


@pytest.mark.contract
class TestContract:
    def test_constants(self, rag_module: Any) -> None:
        assert rag_module.GENERATOR_MODEL == "Qwen/Qwen2.5-7B-Instruct"
        assert rag_module.DEFAULT_TOP_K_CONTEXT == 5
        assert rag_module.DEFAULT_MAX_NEW_TOKENS == 256
        assert rag_module.DEFAULT_BATCH_SIZE == 16
        assert rag_module.DEFAULT_MAX_LENGTH == 4096

    def test_ablation_configs(self, rag_module: Any) -> None:
        # Four canonical ablations
        cfgs = rag_module.ABLATION_CONFIGS
        assert {"none", "bm25", "bge_m3", "reranker"} <= set(cfgs.keys())
        # "none" must produce empty context
        assert cfgs["none"]["results_filename"] is None

    def test_main_callable(self, rag_module: Any) -> None:
        assert callable(getattr(rag_module, "main", None))

    def test_load_queries_callable(self, rag_module: Any) -> None:
        assert callable(getattr(rag_module, "_load_queries", None))

    def test_build_prompt_callable(self, rag_module: Any) -> None:
        assert callable(getattr(rag_module, "_build_prompt", None))

    def test_shard_range_callable(self, rag_module: Any) -> None:
        assert callable(getattr(rag_module, "_shard_range", None))

    def test_schema_version(self, rag_module: Any) -> None:
        import re

        assert re.match(r"^\d+\.\d+\.\d+$", rag_module.SCHEMA_VERSION)


# ---------- query loading ----------


@pytest.mark.unit
class TestLoadQueries:
    def test_dedup_on_source_dest(self, rag_module: Any, tmp_path: Path) -> None:
        gold = _write_jsonl(
            tmp_path / "gold.jsonl",
            [
                {"source_id": 1, "dest_id": 100, "source_cluster_id": 10, "quote": "first"},
                {"source_id": 1, "dest_id": 100, "source_cluster_id": 10, "quote": "first dup"},
                {"source_id": 2, "dest_id": 200, "source_cluster_id": 20, "quote": "second"},
            ],
        )
        queries = rag_module._load_queries(gold)
        assert len(queries) == 2
        assert queries[0]["query_text"] == "first"


# ---------- prompt building ----------


@pytest.mark.unit
class TestBuildPrompt:
    def test_no_rag_includes_only_question(self, rag_module: Any) -> None:
        prompt = rag_module._build_prompt(
            quote="What is the rule?",
            contexts=[],
        )
        assert "What is the rule?" in prompt

    def test_with_contexts_includes_them(self, rag_module: Any) -> None:
        prompt = rag_module._build_prompt(
            quote="What is the rule?",
            contexts=["Context A about X.", "Context B about Y."],
        )
        assert "Context A about X." in prompt
        assert "Context B about Y." in prompt

    def test_contexts_numbered_or_separated(self, rag_module: Any) -> None:
        prompt = rag_module._build_prompt(
            quote="q",
            contexts=["alpha", "beta", "gamma"],
        )
        # Some structural delimiter between contexts
        assert prompt.count("alpha") == 1
        assert prompt.count("beta") == 1
        assert prompt.count("gamma") == 1


# ---------- shard range ----------


@pytest.mark.unit
class TestShardRange:
    def test_4way_partition_sums_to_n(self, rag_module: Any) -> None:
        n = 100
        ranges = [rag_module._shard_range(n, r, 4) for r in range(4)]
        assert sum(end - start for start, end in ranges) == n

    def test_disjoint(self, rag_module: Any) -> None:
        ranges = [rag_module._shard_range(50, r, 4) for r in range(4)]
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]

    def test_invalid_rank_raises(self, rag_module: Any) -> None:
        with pytest.raises(ValueError):
            rag_module._shard_range(10, 4, 4)


# ---------- malformed inputs ----------


@pytest.mark.unit
class TestMalformedInputs:
    def test_invalid_json_raises(self, rag_module: Any, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not json\n")
        with pytest.raises(json.JSONDecodeError):
            rag_module._load_queries(bad)

    def test_unknown_ablation_raises(self, rag_module: Any, tmp_path: Path) -> None:
        with pytest.raises(KeyError):
            rag_module._resolve_ablation("nonexistent")
