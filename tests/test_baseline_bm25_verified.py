"""TDD tests for baseline_bm25.py verified-subset path.

Contract:
  - corpus key = source_cluster_id (not opinion_id)
  - query text = destination_context (not quote)
  - gold matching: retrieved cluster_id must match gold source_cluster_id
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from baseline_bm25 import (  # noqa: E402
    _aggregate_chunk_scores,
    _load_queries_verified,
)


def _make_gold(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestLoadQueriesVerified:
    """Verified path: query_text = destination_context, key = source_cluster_id."""

    def test_uses_destination_context_as_query(self, tmp_path: Path) -> None:
        gold = tmp_path / "gold.jsonl"
        _make_gold(
            gold,
            [
                {
                    "source_id": 1,
                    "source_cluster_id": 100,
                    "source_court": "ca9",
                    "dest_id": 2,
                    "quote": "the cited passage",
                    "destination_context": "surrounding text from citing opinion",
                }
            ],
        )
        queries = _load_queries_verified(gold)
        assert len(queries) == 1
        assert queries[0]["query_text"] == "surrounding text from citing opinion"

    def test_includes_source_cluster_id(self, tmp_path: Path) -> None:
        gold = tmp_path / "gold.jsonl"
        _make_gold(
            gold,
            [
                {
                    "source_id": 1,
                    "source_cluster_id": 999,
                    "source_court": "ca9",
                    "dest_id": 2,
                    "quote": "q",
                    "destination_context": "ctx",
                }
            ],
        )
        queries = _load_queries_verified(gold)
        assert queries[0]["source_cluster_id"] == 999

    def test_preserves_source_dest_ids(self, tmp_path: Path) -> None:
        gold = tmp_path / "gold.jsonl"
        _make_gold(
            gold,
            [
                {
                    "source_id": 42,
                    "source_cluster_id": 100,
                    "source_court": "ca9",
                    "dest_id": 77,
                    "quote": "q",
                    "destination_context": "ctx",
                }
            ],
        )
        queries = _load_queries_verified(gold)
        assert queries[0]["source_id"] == 42
        assert queries[0]["dest_id"] == 77

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        gold = tmp_path / "gold.jsonl"
        gold.write_text(
            json.dumps(
                {
                    "source_id": 1,
                    "source_cluster_id": 1,
                    "source_court": "ca9",
                    "dest_id": 2,
                    "quote": "q",
                    "destination_context": "ctx",
                }
            )
            + "\n\n"
        )
        assert len(_load_queries_verified(gold)) == 1

    def test_raises_on_missing_destination_context(self, tmp_path: Path) -> None:
        gold = tmp_path / "gold.jsonl"
        _make_gold(
            gold,
            [
                {
                    "source_id": 1,
                    "source_cluster_id": 1,
                    "source_court": "ca9",
                    "dest_id": 2,
                    "quote": "q",
                    # no destination_context
                }
            ],
        )
        with pytest.raises((KeyError, ValueError)):
            _load_queries_verified(gold)


class TestAggregationByClusterId:
    """Verified path: chunks aggregated by cluster_id, not opinion_id."""

    def test_max_score_per_cluster(self) -> None:
        hits = [
            {"opinion_id": 100, "chunk_index": 0, "score": 0.5},
            {"opinion_id": 100, "chunk_index": 1, "score": 0.9},
            {"opinion_id": 200, "chunk_index": 0, "score": 0.7},
        ]
        agg = _aggregate_chunk_scores(hits, top_k=10)
        # MaxP: opinion 100 → 0.9, opinion 200 → 0.7
        assert agg[0]["opinion_id"] == 100
        assert agg[0]["score"] == 0.9
        assert agg[1]["opinion_id"] == 200
        assert agg[1]["score"] == 0.7

    def test_top_k_truncates(self) -> None:
        hits = [{"opinion_id": i, "chunk_index": 0, "score": float(i)} for i in range(10)]
        agg = _aggregate_chunk_scores(hits, top_k=3)
        assert len(agg) == 3
        assert [a["opinion_id"] for a in agg] == [9, 8, 7]


class TestMainVerifiedSmoke:
    """Smoke test: main_verified writes results keyed by source_cluster_id."""

    def test_writes_results_with_cluster_id_key(self, tmp_path: Path) -> None:
        from baseline_bm25 import main_verified

        # Tiny corpus: 3 chunks across 2 cluster_ids
        corpus = tmp_path / "corpus.jsonl"
        with corpus.open("w") as f:
            for c in [
                {
                    "opinion_id": 100,
                    "cluster_id": 1000,
                    "chunk_index": 0,
                    "text": "consent decree binding all parties civil rights",
                },
                {
                    "opinion_id": 200,
                    "cluster_id": 2000,
                    "chunk_index": 0,
                    "text": "antitrust monopoly merger acquisition",
                },
                {
                    "opinion_id": 300,
                    "cluster_id": 3000,
                    "chunk_index": 0,
                    "text": "fourth amendment search seizure warrant",
                },
            ]:
                f.write(json.dumps(c) + "\n")

        gold = tmp_path / "gold.jsonl"
        _make_gold(
            gold,
            [
                {
                    "source_id": 1,
                    "source_cluster_id": 1000,
                    "source_court": "ca9",
                    "dest_id": 2,
                    "quote": "irrelevant",
                    "destination_context": "consent decree binding parties",
                }
            ],
        )

        out_dir = tmp_path / "out"
        summary = main_verified(
            corpus_path=corpus,
            gold_pairs_path=gold,
            out_dir=out_dir,
            top_k=3,
        )
        assert (out_dir / "bm25_results.jsonl").exists()
        assert (out_dir / "bm25_summary.json").exists()
        assert summary["n_queries"] == 1

        # Verify result row uses cluster_id (not opinion_id) as retrieved key
        row = json.loads((out_dir / "bm25_results.jsonl").read_text().strip())
        assert "source_cluster_id" in row
        assert row["source_cluster_id"] == 1000
        # Top-1 should be cluster 1000 (its text matches query)
        assert row["retrieved"][0]["cluster_id"] == 1000
