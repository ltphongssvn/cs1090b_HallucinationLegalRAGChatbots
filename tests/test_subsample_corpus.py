"""Tests for scripts.subsample_corpus — one-chunk-per-opinion filter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def sub_module() -> Any:
    from scripts import subsample_corpus

    return subsample_corpus


@pytest.mark.contract
class TestModuleContract:
    def test_subsample_function_exists(self, sub_module: Any) -> None:
        assert callable(getattr(sub_module, "subsample_one_per_opinion", None))


@pytest.mark.unit
class TestSubsampleBehavior:
    def test_keeps_first_chunk_per_opinion(self, sub_module: Any, tmp_path: Path) -> None:
        src = tmp_path / "corpus.jsonl"
        src.write_text(
            "\n".join(
                [
                    json.dumps({"opinion_id": 1, "chunk_index": 0, "text": "a0"}),
                    json.dumps({"opinion_id": 1, "chunk_index": 1, "text": "a1"}),
                    json.dumps({"opinion_id": 1, "chunk_index": 2, "text": "a2"}),
                    json.dumps({"opinion_id": 2, "chunk_index": 0, "text": "b0"}),
                    json.dumps({"opinion_id": 2, "chunk_index": 1, "text": "b1"}),
                    json.dumps({"opinion_id": 3, "chunk_index": 0, "text": "c0"}),
                ]
            )
            + "\n"
        )
        dst = tmp_path / "out.jsonl"
        n_in, n_out = sub_module.subsample_one_per_opinion(src, dst)
        assert n_in == 6
        assert n_out == 3
        lines = dst.read_text().strip().split("\n")
        rows = [json.loads(line) for line in lines]
        oids = [r["opinion_id"] for r in rows]
        assert sorted(oids) == [1, 2, 3]
        # All kept rows must be chunk_index=0
        assert all(r["chunk_index"] == 0 for r in rows)
        # Text content preserved
        texts = {r["opinion_id"]: r["text"] for r in rows}
        assert texts == {1: "a0", 2: "b0", 3: "c0"}

    def test_empty_input(self, sub_module: Any, tmp_path: Path) -> None:
        src = tmp_path / "empty.jsonl"
        src.write_text("")
        dst = tmp_path / "out.jsonl"
        n_in, n_out = sub_module.subsample_one_per_opinion(src, dst)
        assert n_in == 0 and n_out == 0
        assert dst.exists()
        assert dst.read_text() == ""

    def test_handles_out_of_order_chunks(self, sub_module: Any, tmp_path: Path) -> None:
        """If chunk_index=0 not first in stream, still picks it."""
        src = tmp_path / "corpus.jsonl"
        src.write_text(
            "\n".join(
                [
                    json.dumps({"opinion_id": 1, "chunk_index": 2, "text": "a2"}),
                    json.dumps({"opinion_id": 1, "chunk_index": 0, "text": "a0"}),
                    json.dumps({"opinion_id": 1, "chunk_index": 1, "text": "a1"}),
                ]
            )
            + "\n"
        )
        dst = tmp_path / "out.jsonl"
        _, n_out = sub_module.subsample_one_per_opinion(src, dst)
        assert n_out == 1
        row = json.loads(dst.read_text().strip())
        assert row["chunk_index"] == 0
        assert row["text"] == "a0"

    def test_atomic_write(self, sub_module: Any, tmp_path: Path) -> None:
        """No .tmp file lingers after successful write."""
        src = tmp_path / "corpus.jsonl"
        src.write_text(json.dumps({"opinion_id": 1, "chunk_index": 0, "text": "x"}) + "\n")
        dst = tmp_path / "out.jsonl"
        sub_module.subsample_one_per_opinion(src, dst)
        assert not dst.with_suffix(".jsonl.tmp").exists()


@pytest.mark.contract
class TestCli:
    def test_cli_parses_args(self, sub_module: Any) -> None:
        parser = sub_module._build_arg_parser()
        actions = {a.dest for a in parser._actions}
        assert "corpus_path" in actions
        assert "out_path" in actions
