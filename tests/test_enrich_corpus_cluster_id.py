"""TDD: enrich existing corpus_chunks.jsonl with cluster_id via shard lookup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from baseline_prep import enrich_corpus_with_cluster_id  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestEnrichCorpusWithClusterId:
    def test_adds_cluster_id_from_shard_lookup(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "shard_0000.jsonl",
            [{"id": 100, "cluster_id": 5000, "text": "x"}],
        )
        corpus_in = tmp_path / "corpus.jsonl"
        _write_jsonl(
            corpus_in,
            [
                {"opinion_id": 100, "chunk_index": 0, "text": "first"},
                {"opinion_id": 100, "chunk_index": 1, "text": "second"},
            ],
        )
        corpus_out = tmp_path / "corpus_enriched.jsonl"
        n_total, n_enriched, n_unmatched = enrich_corpus_with_cluster_id(
            shard_dir=shard_dir,
            corpus_in_path=corpus_in,
            corpus_out_path=corpus_out,
        )
        assert n_total == 2
        assert n_enriched == 2
        assert n_unmatched == 0
        rows = [json.loads(line) for line in corpus_out.open()]
        assert all(r["cluster_id"] == 5000 for r in rows)

    def test_preserves_existing_cluster_id(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "shard_0000.jsonl",
            [{"id": 100, "cluster_id": 5000, "text": "x"}],
        )
        corpus_in = tmp_path / "corpus.jsonl"
        _write_jsonl(
            corpus_in,
            [{"opinion_id": 100, "cluster_id": 9999, "chunk_index": 0, "text": "x"}],
        )
        corpus_out = tmp_path / "out.jsonl"
        enrich_corpus_with_cluster_id(shard_dir, corpus_in, corpus_out)
        row = json.loads(corpus_out.read_text().strip())
        assert row["cluster_id"] == 9999  # preserved, not overwritten

    def test_unmatched_opinion_omits_cluster_id(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "shard_0000.jsonl",
            [{"id": 100, "cluster_id": 5000, "text": "x"}],
        )
        corpus_in = tmp_path / "corpus.jsonl"
        _write_jsonl(
            corpus_in,
            [{"opinion_id": 999, "chunk_index": 0, "text": "x"}],
        )
        corpus_out = tmp_path / "out.jsonl"
        n_total, n_enriched, n_unmatched = enrich_corpus_with_cluster_id(shard_dir, corpus_in, corpus_out)
        assert n_unmatched == 1
        row = json.loads(corpus_out.read_text().strip())
        assert "cluster_id" not in row

    def test_atomic_write(self, tmp_path: Path) -> None:
        """Output written via .tmp + rename — no partial files on crash."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(shard_dir / "shard_0000.jsonl", [{"id": 1, "cluster_id": 2, "text": "x"}])
        corpus_in = tmp_path / "in.jsonl"
        _write_jsonl(corpus_in, [{"opinion_id": 1, "chunk_index": 0, "text": "x"}])
        corpus_out = tmp_path / "out.jsonl"
        enrich_corpus_with_cluster_id(shard_dir, corpus_in, corpus_out)
        # No .tmp leftover
        assert not list(tmp_path.glob("*.tmp"))
