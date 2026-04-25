"""TDD Red-first tests for scripts/build_lepard_cl_subset.py.

Adapted from Alex's lepard-eda approach: 2-stage filter (eyecite citation
match + rapidfuzz quote text overlap) producing verified LePaRD-CL pairs.

Each public function has a deterministic test using tmp_path fixtures.
No real LePaRD/CL data required — all inputs synthesized.
"""

from __future__ import annotations

import bz2
import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_lepard_cl_subset import (  # noqa: E402
    build_shard_text_index,
    build_subset,
    fuzzy_match_quote,
    is_federal_appellate,
    iter_lepard,
    load_cl_citations_index,
    normalize_reporter,
    parse_source_cite,
)

# ----- Pure helpers ----------------------------------------------------------


class TestNormalizeReporter:
    def test_lowercases_and_collapses_whitespace(self):
        assert normalize_reporter("  F. Supp.  2d ") == "f. supp. 2d"

    def test_handles_simple(self):
        assert normalize_reporter("F.2d") == "f.2d"


class TestIsFederalAppellate:
    def test_recognizes_circuits(self):
        assert is_federal_appellate("United States Court of Appeals for the Ninth Circuit")
        assert is_federal_appellate("D.C. Circuit")
        assert is_federal_appellate("Federal Circuit")

    def test_rejects_district(self):
        assert not is_federal_appellate("United States District Court for the District of Maryland")

    def test_rejects_supreme(self):
        assert not is_federal_appellate("Supreme Court of the United States")


class TestParseSourceCite:
    def test_parses_standard_citation(self):
        result = parse_source_cite("Foo v. Bar, 71 F. Supp. 2d 990 (1998)")
        assert result == ("71", "f. supp. 2d", "990")

    def test_parses_f2d(self):
        result = parse_source_cite("X v. Y, 100 F.2d 50 (1940)")
        assert result == ("100", "f.2d", "50")

    def test_returns_none_on_unparseable(self):
        assert parse_source_cite("not a citation") is None

    def test_returns_none_on_empty(self):
        assert parse_source_cite("") is None


class TestFuzzyMatchQuote:
    def test_exact_match_passes(self):
        passed, score = fuzzy_match_quote("foo bar baz", "the court held foo bar baz today", 80.0)
        assert passed is True
        assert score == 100.0

    def test_below_threshold_fails(self):
        passed, score = fuzzy_match_quote("completely different", "unrelated text here", 80.0)
        assert passed is False
        assert score < 80.0

    def test_empty_quote_fails(self):
        passed, _ = fuzzy_match_quote("", "any text", 80.0)
        assert passed is False

    def test_empty_text_fails(self):
        passed, _ = fuzzy_match_quote("any quote", "", 80.0)
        assert passed is False


# ----- Stage 1: CL citations index ------------------------------------------


class TestLoadClCitationsIndex:
    def _make_csv_bz2(self, path: Path, rows: list[dict]) -> None:
        with bz2.open(path, "wt", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["volume", "reporter", "page", "cluster_id"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def test_builds_index(self, tmp_path):
        p = tmp_path / "citations.csv.bz2"
        self._make_csv_bz2(
            p,
            [
                {"volume": "100", "reporter": "F.2d", "page": "50", "cluster_id": "1"},
                {"volume": "200", "reporter": "F.3d", "page": "60", "cluster_id": "2"},
            ],
        )
        idx = load_cl_citations_index(p)
        assert idx[("100", "f.2d", "50")] == 1
        assert idx[("200", "f.3d", "60")] == 2

    def test_first_seen_wins_collision(self, tmp_path):
        p = tmp_path / "citations.csv.bz2"
        self._make_csv_bz2(
            p,
            [
                {"volume": "1", "reporter": "F", "page": "1", "cluster_id": "111"},
                {"volume": "1", "reporter": "F", "page": "1", "cluster_id": "222"},
            ],
        )
        idx = load_cl_citations_index(p)
        assert idx[("1", "f", "1")] == 111

    def test_normalizes_reporter_in_key(self, tmp_path):
        p = tmp_path / "citations.csv.bz2"
        self._make_csv_bz2(
            p,
            [
                {"volume": "1", "reporter": "  F. Supp. 2d  ", "page": "1", "cluster_id": "10"},
            ],
        )
        idx = load_cl_citations_index(p)
        assert ("1", "f. supp. 2d", "1") in idx

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cl_citations_index(tmp_path / "missing.csv.bz2")


# ----- Stage 2: shard text index --------------------------------------------


class TestBuildShardTextIndex:
    def _make_shards(self, shard_dir: Path, shards: list[list[dict]]) -> None:
        shard_dir.mkdir(parents=True, exist_ok=True)
        for i, rows in enumerate(shards):
            p = shard_dir / f"shard_{i:04d}.jsonl"
            with p.open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

    def test_extracts_text_for_target_ids(self, tmp_path):
        shard_dir = tmp_path / "shards"
        self._make_shards(
            shard_dir,
            [
                [
                    {"cluster_id": 100, "text": "first opinion"},
                    {"cluster_id": 200, "text": "second opinion"},
                    {"cluster_id": 300, "text": "third opinion"},
                ]
            ],
        )
        idx = build_shard_text_index(shard_dir, target_ids={100, 300})
        assert idx == {100: "first opinion", 300: "third opinion"}

    def test_falls_back_to_raw_text_field(self, tmp_path):
        shard_dir = tmp_path / "shards"
        self._make_shards(
            shard_dir,
            [
                [
                    {"cluster_id": 100, "raw_text": "fallback content"},
                ]
            ],
        )
        idx = build_shard_text_index(shard_dir, target_ids={100})
        assert idx[100] == "fallback content"

    def test_raises_on_no_shards(self, tmp_path):
        with pytest.raises(RuntimeError, match="No shard"):
            build_shard_text_index(tmp_path / "empty", target_ids={1})


# ----- iter_lepard -----------------------------------------------------------


class TestIterLepard:
    def test_iterates_all_when_n_none(self, tmp_path, monkeypatch):
        import build_lepard_cl_subset as mod
        from build_lepard_cl_subset import LEPARD_JSONL  # noqa: F401

        fake = tmp_path / "lepard.jsonl"
        fake.write_text(json.dumps({"source_id": 1}) + "\n" + json.dumps({"source_id": 2}) + "\n")
        monkeypatch.setattr(mod, "LEPARD_JSONL", fake)
        rows = list(iter_lepard(n=None))
        assert [r["source_id"] for r in rows] == [1, 2]

    def test_samples_deterministically_with_seed(self, tmp_path, monkeypatch):
        import build_lepard_cl_subset as mod

        fake = tmp_path / "lepard.jsonl"
        fake.write_text("\n".join(json.dumps({"source_id": i}) for i in range(100)) + "\n")
        monkeypatch.setattr(mod, "LEPARD_JSONL", fake)

        rows1 = list(iter_lepard(n=10, seed=42))
        rows2 = list(iter_lepard(n=10, seed=42))
        assert [r["source_id"] for r in rows1] == [r["source_id"] for r in rows2]
        assert len(rows1) == 10


# ----- End-to-end build_subset ----------------------------------------------


class TestBuildSubsetEndToEnd:
    def _setup_minimal_inputs(self, tmp_path, monkeypatch):
        """Set up a self-contained mini-pipeline: 1 LePaRD row → 1 verified output."""
        import build_lepard_cl_subset as mod

        # LePaRD: one row whose quote IS in the shard text
        lepard = tmp_path / "lepard.jsonl"
        lepard.write_text(
            json.dumps(
                {
                    "source_id": 1,
                    "source_name": "Test v. Case",
                    "source_cite": "Test v. Case, 71 F. Supp. 2d 990 (1998)",
                    "source_court": "United States District Court for the District of Test",
                    "source_date": "1998-01-01",
                    "dest_id": 99,
                    "dest_name": "Citing v. Source",
                    "dest_cite": "Citing v. Source, 200 F.3d 1 (2000)",
                    "dest_court": "United States Court of Appeals for the Ninth Circuit",
                    "dest_date": "2000-01-01",
                    "passage_id": "99_1",
                    "quote": "the consent decree is binding",
                    "destination_context": "Some surrounding context.",
                }
            )
            + "\n"
        )
        monkeypatch.setattr(mod, "LEPARD_JSONL", lepard)

        # CL citations: maps (71, 'f. supp. 2d', 990) → cluster_id=42
        cit = tmp_path / "citations.csv.bz2"
        with bz2.open(cit, "wt", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["volume", "reporter", "page", "cluster_id"])
            w.writeheader()
            w.writerow({"volume": "71", "reporter": "F. Supp. 2d", "page": "990", "cluster_id": "42"})

        # CL shard: cluster_id=42 with text containing the quote
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        (shard_dir / "shard_0000.jsonl").write_text(
            json.dumps(
                {
                    "cluster_id": 42,
                    "text": "the court held that the consent decree is binding on all parties",
                }
            )
            + "\n"
        )

        return lepard, cit, shard_dir

    def test_produces_verified_pair(self, tmp_path, monkeypatch):
        _, cit, shard_dir = self._setup_minimal_inputs(tmp_path, monkeypatch)
        out = tmp_path / "out.jsonl"
        summary = build_subset(
            sample_size=10,
            full_lepard=False,
            appellate_only=False,
            text_verify=True,
            fuzzy_threshold=80.0,
            citations_path=cit,
            shard_dir=shard_dir,
            out_path=out,
        )
        assert summary["text_verified"] == 1
        assert summary["cite_matched"] == 1
        rows = [json.loads(line) for line in out.open()]
        assert len(rows) == 1
        assert rows[0]["source_cluster_id"] == 42
        assert "quote_fuzzy_score" in rows[0]
        assert rows[0]["quote_fuzzy_score"] >= 80.0

    def test_writes_summary_json(self, tmp_path, monkeypatch):
        _, cit, shard_dir = self._setup_minimal_inputs(tmp_path, monkeypatch)
        out = tmp_path / "out.jsonl"
        build_subset(
            sample_size=10,
            citations_path=cit,
            shard_dir=shard_dir,
            out_path=out,
        )
        summary_file = out.with_suffix(".summary.json")
        assert summary_file.exists()
        meta = json.loads(summary_file.read_text())
        assert meta["text_verified"] == 1

    def test_no_text_verify_falls_back_to_stage1_only(self, tmp_path, monkeypatch):
        _, cit, shard_dir = self._setup_minimal_inputs(tmp_path, monkeypatch)
        out = tmp_path / "out.jsonl"
        summary = build_subset(
            sample_size=10,
            text_verify=False,
            citations_path=cit,
            shard_dir=shard_dir,
            out_path=out,
        )
        assert summary["text_verify"] is False
        rows = [json.loads(line) for line in out.open()]
        assert len(rows) == 1
        assert "quote_fuzzy_score" not in rows[0]
