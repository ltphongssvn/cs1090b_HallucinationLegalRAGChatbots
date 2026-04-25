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
                }
            ],
        )
        with pytest.raises((KeyError, ValueError)):
            _load_queries_verified(gold)


class TestAggregationByClusterId:
    def test_max_score_per_cluster(self) -> None:
        hits = [
            {"opinion_id": 100, "chunk_index": 0, "score": 0.5},
            {"opinion_id": 100, "chunk_index": 1, "score": 0.9},
            {"opinion_id": 200, "chunk_index": 0, "score": 0.7},
        ]
        agg = _aggregate_chunk_scores(hits, top_k=10)
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
    def test_writes_results_with_cluster_id_key(self, tmp_path: Path) -> None:
        from baseline_bm25 import main_verified

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
        row = json.loads((out_dir / "bm25_results.jsonl").read_text().strip())
        assert "source_cluster_id" in row
        assert row["source_cluster_id"] == 1000
        assert row["retrieved"][0]["cluster_id"] == 1000


class TestMainVerifiedCLI:
    def test_cli_verified_flag_exposed(self) -> None:
        import importlib

        mod = importlib.import_module("baseline_bm25")
        parser = mod._build_arg_parser()
        args = parser.parse_args(["--verified"])
        assert args.verified is True


class TestRetrievalFailureLog:
    def test_writes_failures_jsonl(self, tmp_path: Path) -> None:
        from baseline_bm25 import main_verified

        corpus = tmp_path / "corpus.jsonl"
        with corpus.open("w") as f:
            for c in [
                {"opinion_id": 100, "cluster_id": 1000, "chunk_index": 0, "text": "consent decree binding"},
                {"opinion_id": 200, "cluster_id": 2000, "chunk_index": 0, "text": "antitrust merger"},
            ]:
                f.write(json.dumps(c) + "\n")
        gold = tmp_path / "gold.jsonl"
        _make_gold(
            gold,
            [
                {
                    "source_id": 1,
                    "source_cluster_id": 9999,
                    "source_court": "ca9",
                    "dest_id": 2,
                    "quote": "irrelevant",
                    "destination_context": "antitrust merger",
                }
            ],
        )
        out_dir = tmp_path / "out"
        main_verified(
            corpus_path=corpus,
            gold_pairs_path=gold,
            out_dir=out_dir,
            top_k=2,
        )
        failures = out_dir / "bm25_failures.jsonl"
        assert failures.exists()
        rows = [json.loads(line) for line in failures.open()]
        assert len(rows) == 1
        assert rows[0]["source_cluster_id"] == 9999
        assert rows[0]["gold_in_top_k"] is False
        assert "top_retrieved" in rows[0]


class TestIndexPersistence:
    def test_save_and_load_index_skips_rebuild(self, tmp_path: Path) -> None:
        from baseline_bm25 import main_verified

        corpus = tmp_path / "corpus.jsonl"
        with corpus.open("w") as f:
            for c in [
                {"opinion_id": 100, "cluster_id": 1000, "chunk_index": 0, "text": "test text"},
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
                    "quote": "q",
                    "destination_context": "test",
                }
            ],
        )
        out_dir = tmp_path / "out"
        index_dir = tmp_path / "bm25_index"
        s1 = main_verified(
            corpus_path=corpus,
            gold_pairs_path=gold,
            out_dir=out_dir,
            top_k=1,
            index_dir=index_dir,
        )
        assert index_dir.exists()
        first_build_secs = s1["index_build_seconds"]
        s2 = main_verified(
            corpus_path=corpus,
            gold_pairs_path=gold,
            out_dir=out_dir,
            top_k=1,
            index_dir=index_dir,
        )
        assert s2["index_build_seconds"] < first_build_secs or s2["index_build_seconds"] < 0.1


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("data/processed/baseline/corpus_chunks_enriched.jsonl").exists(),
    reason="requires enriched corpus produced by enrich_corpus_with_cluster_id",
)
class TestRealDataIntegration:
    """Slow (~5-10 min) — guarded by integration marker.

    Provides regression signal that BM25 hits non-trivial Hit@100 on real pipeline output.
    Run explicitly with: pytest -m integration
    """

    def test_hit_at_100_above_threshold_on_500_query_slice(self, tmp_path: Path) -> None:
        from baseline_bm25 import main_verified

        corpus = Path("data/processed/baseline/corpus_chunks_enriched.jsonl")
        gold = Path("data/processed/baseline/gold_pairs_test.jsonl")
        slice_path = tmp_path / "gold_slice.jsonl"
        with gold.open() as fin, slice_path.open("w") as fout:
            for i, line in enumerate(fin):
                if i >= 500:
                    break
                fout.write(line)
        out_dir = tmp_path / "out"
        summary = main_verified(
            corpus_path=corpus,
            gold_pairs_path=slice_path,
            out_dir=out_dir,
            top_k=100,
        )
        assert summary["n_queries"] == 500
        results_path = out_dir / "bm25_results.jsonl"
        n_hits = 0
        n_total = 0
        with slice_path.open() as fg, results_path.open() as fr:
            for gline, rline in zip(fg, fr, strict=True):
                g = json.loads(gline)
                r = json.loads(rline)
                gold_cid = int(g["source_cluster_id"])
                top_cids = {int(x["cluster_id"]) for x in r["retrieved"]}
                if gold_cid in top_cids:
                    n_hits += 1
                n_total += 1
        hit_at_100 = n_hits / n_total
        assert hit_at_100 >= 0.10, f"BM25 Hit@100 = {hit_at_100:.4f} on 500-query slice; expected >= 0.10."
