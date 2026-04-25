"""TDD: corpus_chunks.jsonl must include cluster_id field."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from baseline_prep import _chunk_corpus  # noqa: E402


class TestChunkCorpusIncludesClusterId:
    def _make_shard(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def _mock_tokenizer(self) -> MagicMock:
        tok = MagicMock()
        tok.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        tok.decode = MagicMock(return_value="decoded text")
        return tok

    def test_chunk_output_has_cluster_id(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        self._make_shard(
            shard_dir / "shard_0000.jsonl",
            [{"id": 100, "cluster_id": 555, "text": "opinion text here"}],
        )
        out_path = tmp_path / "corpus_chunks.jsonl"
        ckpt_path = tmp_path / "ckpt.json"

        n_chunks, n_opinions = _chunk_corpus(
            shard_dir, out_path, ckpt_path,
            resume=False, tok=self._mock_tokenizer(),
        )
        rows = [json.loads(line) for line in out_path.open()]
        assert len(rows) >= 1
        assert "cluster_id" in rows[0], "corpus chunk must carry cluster_id"
        assert rows[0]["cluster_id"] == 555
        assert rows[0]["opinion_id"] == 100

    def test_missing_cluster_id_in_shard_raises(self, tmp_path: Path) -> None:
        import pytest
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        self._make_shard(
            shard_dir / "shard_0000.jsonl",
            [{"id": 100, "text": "no cluster_id here"}],
        )
        with pytest.raises((KeyError, ValueError)):
            _chunk_corpus(
                shard_dir, tmp_path / "out.jsonl", tmp_path / "ckpt.json",
                resume=False, tok=self._mock_tokenizer(),
            )
