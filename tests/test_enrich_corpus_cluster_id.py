"""TDD: enrich existing corpus_chunks.jsonl with cluster_id via shard lookup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
        assert row["cluster_id"] == 9999

    def test_unmatched_routed_to_dead_letter(self, tmp_path: Path) -> None:
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
        assert corpus_out.read_text() == ""
        dl = corpus_out.parent / "unmatched_chunks.jsonl"
        assert dl.exists()
        dl_row = json.loads(dl.read_text().strip())
        assert dl_row["opinion_id"] == 999

    def test_atomic_write(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(shard_dir / "shard_0000.jsonl", [{"id": 1, "cluster_id": 2, "text": "x"}])
        corpus_in = tmp_path / "in.jsonl"
        _write_jsonl(corpus_in, [{"opinion_id": 1, "chunk_index": 0, "text": "x"}])
        corpus_out = tmp_path / "out.jsonl"
        enrich_corpus_with_cluster_id(shard_dir, corpus_in, corpus_out)
        assert not list(tmp_path.glob("*.tmp"))


class TestDeadLetterAndThreshold:
    def test_writes_unmatched_to_dead_letter(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(shard_dir / "shard_0000.jsonl", [{"id": 100, "cluster_id": 5000, "text": "x"}])
        corpus_in = tmp_path / "corpus.jsonl"
        _write_jsonl(
            corpus_in,
            [
                {"opinion_id": 100, "chunk_index": 0, "text": "matched"},
                {"opinion_id": 999, "chunk_index": 0, "text": "unmatched"},
            ],
        )
        corpus_out = tmp_path / "out.jsonl"
        n_total, n_enriched, n_unmatched = enrich_corpus_with_cluster_id(
            shard_dir,
            corpus_in,
            corpus_out,
            max_unmatched_rate=0.6,
        )
        assert n_unmatched == 1
        dead_letter = corpus_out.parent / "unmatched_chunks.jsonl"
        assert dead_letter.exists()
        rows = [json.loads(line) for line in dead_letter.open()]
        assert len(rows) == 1
        assert rows[0]["opinion_id"] == 999

    def test_raises_when_unmatched_rate_exceeds_threshold(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(shard_dir / "shard_0000.jsonl", [{"id": 100, "cluster_id": 5000, "text": "x"}])
        corpus_in = tmp_path / "corpus.jsonl"
        _write_jsonl(
            corpus_in,
            [
                {"opinion_id": 999, "chunk_index": 0, "text": "x"},
                {"opinion_id": 998, "chunk_index": 0, "text": "x"},
            ],
        )
        corpus_out = tmp_path / "out.jsonl"
        with pytest.raises(RuntimeError, match="unmatched rate"):
            enrich_corpus_with_cluster_id(
                shard_dir,
                corpus_in,
                corpus_out,
                max_unmatched_rate=0.5,
            )

    def test_does_not_create_dead_letter_when_zero_unmatched(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(shard_dir / "shard_0000.jsonl", [{"id": 100, "cluster_id": 5000, "text": "x"}])
        corpus_in = tmp_path / "corpus.jsonl"
        _write_jsonl(corpus_in, [{"opinion_id": 100, "chunk_index": 0, "text": "x"}])
        corpus_out = tmp_path / "out.jsonl"
        enrich_corpus_with_cluster_id(shard_dir, corpus_in, corpus_out)
        assert not (corpus_out.parent / "unmatched_chunks.jsonl").exists()


class TestEnrichSummaryArtifact:
    """Enrich must write summary JSON with provenance."""

    def test_writes_summary_json(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(shard_dir / "shard_0000.jsonl", [{"id": 100, "cluster_id": 5000, "text": "x"}])
        corpus_in = tmp_path / "in.jsonl"
        _write_jsonl(corpus_in, [{"opinion_id": 100, "chunk_index": 0, "text": "x"}])
        corpus_out = tmp_path / "out.jsonl"
        enrich_corpus_with_cluster_id(shard_dir, corpus_in, corpus_out)
        summary = corpus_out.with_suffix(".summary.json")
        assert summary.exists()
        meta = json.loads(summary.read_text())
        assert meta["n_total"] == 1
        assert meta["n_enriched"] == 1
        assert meta["n_unmatched"] == 0
        assert "git_sha" in meta
        assert "corpus_in_sha256" in meta
        assert "corpus_out_sha256" in meta


class TestEnrichCLIIntegration:
    """Enrich available via CLI subcommand."""

    def test_cli_enrich_flag_exposed(self) -> None:
        import importlib

        mod = importlib.import_module("baseline_prep")
        parser = mod._build_arg_parser()
        args = parser.parse_args(
            [
                "--enrich-corpus",
                "--corpus-in",
                "/tmp/x.jsonl",
                "--corpus-out",
                "/tmp/y.jsonl",
            ]
        )
        assert args.enrich_corpus is True
        assert args.corpus_in == Path("/tmp/x.jsonl")
        assert args.corpus_out == Path("/tmp/y.jsonl")


class TestEnrichPropertyEveryClusterIdMatches:
    """Property: every chunk with mapped opinion_id gets correct cluster_id."""

    @given(
        n_chunks=st.integers(min_value=1, max_value=50),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_every_matched_opinion_gets_correct_cluster_id(self, tmp_path: Path, n_chunks: int, seed: int) -> None:
        import random

        sub = tmp_path / f"trial_{seed}_{n_chunks}"
        sub.mkdir()
        shard_dir = sub / "shards"
        shard_dir.mkdir()
        rng = random.Random(seed)
        oid_to_cid = {i + 1: 1000 + rng.randint(0, 100) for i in range(20)}
        _write_jsonl(
            shard_dir / "shard_0000.jsonl",
            [{"id": oid, "cluster_id": cid, "text": "x"} for oid, cid in oid_to_cid.items()],
        )
        corpus_in = sub / "in.jsonl"
        chunk_oids = [rng.choice(list(oid_to_cid.keys())) for _ in range(n_chunks)]
        _write_jsonl(
            corpus_in,
            [{"opinion_id": oid, "chunk_index": i, "text": "x"} for i, oid in enumerate(chunk_oids)],
        )
        corpus_out = sub / "out.jsonl"
        enrich_corpus_with_cluster_id(shard_dir, corpus_in, corpus_out)
        rows = [json.loads(line) for line in corpus_out.open()]
        for row in rows:
            assert row["cluster_id"] == oid_to_cid[row["opinion_id"]]


class TestEnrichSummaryAdditionalProvenance:
    def test_summary_includes_timestamp(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(shard_dir / "shard_0000.jsonl", [{"id": 100, "cluster_id": 5000, "text": "x"}])
        corpus_in = tmp_path / "in.jsonl"
        _write_jsonl(corpus_in, [{"opinion_id": 100, "chunk_index": 0, "text": "x"}])
        corpus_out = tmp_path / "out.jsonl"
        enrich_corpus_with_cluster_id(shard_dir, corpus_in, corpus_out)
        meta = json.loads(corpus_out.with_suffix(".summary.json").read_text())
        assert "finished_at_utc" in meta
        assert "T" in meta["finished_at_utc"]  # ISO 8601

    def test_summary_includes_sample_unmatched_ids(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(shard_dir / "shard_0000.jsonl", [{"id": 100, "cluster_id": 5000, "text": "x"}])
        corpus_in = tmp_path / "in.jsonl"
        _write_jsonl(
            corpus_in,
            [
                {"opinion_id": 999, "chunk_index": 0, "text": "x"},
                {"opinion_id": 998, "chunk_index": 0, "text": "x"},
                {"opinion_id": 997, "chunk_index": 0, "text": "x"},
            ],
        )
        corpus_out = tmp_path / "out.jsonl"
        enrich_corpus_with_cluster_id(shard_dir, corpus_in, corpus_out, max_unmatched_rate=1.0)
        meta = json.loads(corpus_out.with_suffix(".summary.json").read_text())
        assert "sample_unmatched_opinion_ids" in meta
        assert isinstance(meta["sample_unmatched_opinion_ids"], list)
        assert len(meta["sample_unmatched_opinion_ids"]) > 0
        # Each entry must be one of the unmatched ids
        for oid in meta["sample_unmatched_opinion_ids"]:
            assert oid in (999, 998, 997)

    def test_summary_omits_sample_when_no_unmatched(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(shard_dir / "shard_0000.jsonl", [{"id": 100, "cluster_id": 5000, "text": "x"}])
        corpus_in = tmp_path / "in.jsonl"
        _write_jsonl(corpus_in, [{"opinion_id": 100, "chunk_index": 0, "text": "x"}])
        corpus_out = tmp_path / "out.jsonl"
        enrich_corpus_with_cluster_id(shard_dir, corpus_in, corpus_out)
        meta = json.loads(corpus_out.with_suffix(".summary.json").read_text())
        assert meta.get("sample_unmatched_opinion_ids", []) == []
