"""TDD Red-first tests for scripts/diagnose_lepard_cl_mismatch.py.

Strategy
--------
The script is a 6-stage pipeline. Each stage reads inputs, transforms, writes
outputs. To make it testable we extract pure helpers and stage functions that
accept explicit Path inputs and return a structured result dict. Tests cover:

  * pure helpers (_parse_citation, _sentinel, _stage_complete)
  * each stage (stage1..stage6) — contract: reads input path(s), writes output
    path, returns meta dict with documented keys
  * checkpoint semantics: sentinel creation, idempotent re-run skip, force flag
  * atomicity: output file appears only after full write (.tmp rename)

Fixtures are synthetic and minimal — no real CL/LePaRD data required.
"""

from __future__ import annotations

import bz2
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from diagnose_lepard_cl_mismatch import (  # noqa: E402
    _mark_done,
    _parse_citation,
    _sentinel,
    _skip_or_run,
    _stage_complete,
    stage1_parse_lepard,
    stage2_index_cl_citations,
    stage3_join,
    stage4_cluster_to_opinion,
    stage5_final_map,
    stage6_validate,
)

# ----- pure helpers -----------------------------------------------------------


class TestParseCitation:
    def test_extracts_volume_reporter_page(self):
        assert _parse_citation("Foo v. Bar, 71 F. Supp. 2d 990 (1998)") == (
            "71",
            "F. Supp. 2d",
            "990",
        )

    def test_handles_fed_appx(self):
        assert _parse_citation("X v. Y, 336 Fed.Appx. 29 (2009)") == (
            "336",
            "Fed.Appx.",
            "29",
        )

    def test_returns_none_for_unparseable(self):
        assert _parse_citation("not a citation") is None

    def test_returns_none_for_empty(self):
        assert _parse_citation("") is None

    def test_returns_none_for_none(self):
        assert _parse_citation(None) is None


class TestSentinel:
    def test_sentinel_path_has_done_suffix(self, tmp_path):
        out = tmp_path / "artifact.jsonl"
        assert _sentinel(out).name == "artifact.jsonl.done"

    def test_sentinel_alongside_output(self, tmp_path):
        out = tmp_path / "subdir" / "file.json"
        assert _sentinel(out).parent == out.parent


class TestStageComplete:
    def test_false_when_no_sentinel(self, tmp_path):
        out = tmp_path / "o.jsonl"
        out.write_text("x")
        assert _stage_complete(out) is False

    def test_false_when_output_missing(self, tmp_path):
        out = tmp_path / "o.jsonl"
        _sentinel(out).write_text("{}")
        assert _stage_complete(out) is False

    def test_false_when_output_empty(self, tmp_path):
        out = tmp_path / "o.jsonl"
        out.write_text("")
        _sentinel(out).write_text("{}")
        assert _stage_complete(out) is False

    def test_true_when_both_exist_and_nonempty(self, tmp_path):
        out = tmp_path / "o.jsonl"
        out.write_text("x")
        _sentinel(out).write_text("{}")
        assert _stage_complete(out) is True


class TestMarkDone:
    def test_writes_sentinel_with_meta(self, tmp_path):
        out = tmp_path / "o.jsonl"
        _mark_done(out, {"n": 42})
        sentinel = _sentinel(out)
        assert sentinel.exists()
        meta = json.loads(sentinel.read_text())
        assert meta["n"] == 42
        assert "finished_at" in meta


class TestSkipOrRun:
    def test_runs_when_not_complete(self, tmp_path):
        out = tmp_path / "o.jsonl"
        assert _skip_or_run(1, out, force=set(), max_stage=6) is True

    def test_skips_when_complete(self, tmp_path):
        out = tmp_path / "o.jsonl"
        out.write_text("x")
        _sentinel(out).write_text("{}")
        assert _skip_or_run(1, out, force=set(), max_stage=6) is False

    def test_runs_when_forced(self, tmp_path):
        out = tmp_path / "o.jsonl"
        out.write_text("x")
        _sentinel(out).write_text("{}")
        assert _skip_or_run(1, out, force={1}, max_stage=6) is True
        assert not _sentinel(out).exists(), "force should delete sentinel"

    def test_skips_beyond_max_stage(self, tmp_path):
        out = tmp_path / "o.jsonl"
        assert _skip_or_run(5, out, force=set(), max_stage=3) is False


# ----- Stage 1: parse LePaRD citations ---------------------------------------


class TestStage1ParseLepard:
    def _make_lepard(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_writes_jsonl_with_parsed_citations(self, tmp_path):
        lepard = tmp_path / "lepard.jsonl"
        self._make_lepard(
            lepard,
            [
                {
                    "source_id": 1,
                    "dest_id": 2,
                    "source_cite": "X v. Y, 100 F.2d 50 (1940)",
                    "dest_cite": "A v. B, 71 F. Supp. 2d 990 (1998)",
                }
            ],
        )
        out = tmp_path / "parsed.jsonl"
        meta = stage1_parse_lepard(lepard, out)

        assert out.exists()
        rows = [json.loads(line) for line in out.open()]
        ids = {r["lepard_id"]: r for r in rows}
        assert ids[1] == {
            "lepard_id": 1,
            "volume": "100",
            "reporter": "F.2d",
            "page": "50",
        }
        assert ids[2] == {
            "lepard_id": 2,
            "volume": "71",
            "reporter": "F. Supp. 2d",
            "page": "990",
        }
        assert meta["n_lepard_rows"] == 1
        assert meta["n_ids_parsed"] == 2

    def test_skips_unparseable_citations(self, tmp_path):
        lepard = tmp_path / "lepard.jsonl"
        self._make_lepard(
            lepard,
            [
                {
                    "source_id": 1,
                    "dest_id": 2,
                    "source_cite": "malformed",
                    "dest_cite": "also bad",
                }
            ],
        )
        out = tmp_path / "parsed.jsonl"
        meta = stage1_parse_lepard(lepard, out)
        assert meta["n_ids_parsed"] == 0
        assert out.stat().st_size == 0

    def test_dedups_repeated_ids(self, tmp_path):
        lepard = tmp_path / "lepard.jsonl"
        self._make_lepard(
            lepard,
            [
                {
                    "source_id": 1,
                    "dest_id": 2,
                    "source_cite": "X, 10 F.2d 1 (1940)",
                    "dest_cite": "Y, 20 F.2d 2 (1941)",
                },
                {
                    "source_id": 1,  # dup
                    "dest_id": 3,
                    "source_cite": "X, 10 F.2d 1 (1940)",
                    "dest_cite": "Z, 30 F.2d 3 (1942)",
                },
            ],
        )
        out = tmp_path / "parsed.jsonl"
        meta = stage1_parse_lepard(lepard, out)
        assert meta["n_ids_parsed"] == 3  # 1, 2, 3 unique
        rows = [json.loads(line) for line in out.open()]
        assert {r["lepard_id"] for r in rows} == {1, 2, 3}

    def test_writes_sentinel(self, tmp_path):
        lepard = tmp_path / "lepard.jsonl"
        lepard.write_text(json.dumps({"source_id": 1, "dest_id": 2}) + "\n")
        out = tmp_path / "parsed.jsonl"
        stage1_parse_lepard(lepard, out)
        assert _sentinel(out).exists()

    def test_atomic_write_no_tmp_left_behind(self, tmp_path):
        lepard = tmp_path / "lepard.jsonl"
        lepard.write_text(
            json.dumps(
                {
                    "source_id": 1,
                    "dest_id": 2,
                    "source_cite": "X, 1 F. 2 (1900)",
                    "dest_cite": "Y, 3 F. 4 (1901)",
                }
            )
            + "\n"
        )
        out = tmp_path / "parsed.jsonl"
        stage1_parse_lepard(lepard, out)
        assert not list(tmp_path.glob("*.tmp"))


# ----- Stage 2: CL citation index --------------------------------------------


class TestStage2IndexCl:
    def _make_citations_bz2(self, path: Path, rows: list[dict]) -> None:
        with bz2.open(path, "wt", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["volume", "reporter", "page", "cluster_id"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def test_writes_unique_key_jsonl(self, tmp_path):
        cl_csv = tmp_path / "citations.csv.bz2"
        self._make_citations_bz2(
            cl_csv,
            [
                {"volume": "100", "reporter": "F.2d", "page": "50", "cluster_id": "1"},
                {"volume": "200", "reporter": "F.3d", "page": "60", "cluster_id": "2"},
            ],
        )
        out = tmp_path / "idx.jsonl"
        meta = stage2_index_cl_citations(cl_csv, out)
        rows = [json.loads(line) for line in out.open()]
        assert len(rows) == 2
        assert meta["n_unique_keys"] == 2
        assert meta["n_citation_rows"] == 2

    def test_first_seen_wins_on_collision(self, tmp_path):
        cl_csv = tmp_path / "citations.csv.bz2"
        self._make_citations_bz2(
            cl_csv,
            [
                {"volume": "1", "reporter": "F", "page": "1", "cluster_id": "111"},
                {"volume": "1", "reporter": "F", "page": "1", "cluster_id": "222"},
            ],
        )
        out = tmp_path / "idx.jsonl"
        stage2_index_cl_citations(cl_csv, out)
        rows = [json.loads(line) for line in out.open()]
        assert len(rows) == 1
        assert rows[0]["cluster_id"] == 111


# ----- Stage 3: join ---------------------------------------------------------


class TestStage3Join:
    def test_joins_lepard_to_cl_cluster(self, tmp_path):
        lepard_cites = tmp_path / "lepard_cites.jsonl"
        cl_cites = tmp_path / "cl_cites.jsonl"
        lepard_cites.write_text(
            json.dumps({"lepard_id": 42, "volume": "1", "reporter": "F", "page": "1"})
            + "\n"
            + json.dumps({"lepard_id": 99, "volume": "2", "reporter": "F", "page": "2"})
            + "\n"
        )
        cl_cites.write_text(json.dumps({"volume": "1", "reporter": "F", "page": "1", "cluster_id": 555}) + "\n")
        out = tmp_path / "joined.jsonl"
        meta = stage3_join(lepard_cites, cl_cites, out)
        rows = [json.loads(line) for line in out.open()]
        assert rows == [{"lepard_id": 42, "cl_cluster_id": 555}]
        assert meta["n_lepard_ids"] == 2
        assert meta["n_matched"] == 1
        assert meta["match_rate"] == 0.5


# ----- Stage 4: cluster_id → opinion_id --------------------------------------


class TestStage4ClusterToOpinion:
    def _make_opinions_bz2(self, path: Path, rows: list[dict]) -> None:
        with bz2.open(path, "wt", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "cluster_id"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def test_filters_by_needed_clusters(self, tmp_path):
        lepard_cluster = tmp_path / "joined.jsonl"
        lepard_cluster.write_text(json.dumps({"lepard_id": 1, "cl_cluster_id": 100}) + "\n")
        opinions = tmp_path / "opinions.csv.bz2"
        self._make_opinions_bz2(
            opinions,
            [
                {"id": "1000", "cluster_id": "100"},
                {"id": "9999", "cluster_id": "999"},  # not needed
            ],
        )
        out = tmp_path / "c2o.json"
        meta = stage4_cluster_to_opinion(lepard_cluster, opinions, out)
        data = json.loads(out.read_text())
        assert data == {"100": 1000}
        assert meta["n_clusters_matched"] == 1

    def test_picks_min_opinion_id_per_cluster(self, tmp_path):
        lepard_cluster = tmp_path / "joined.jsonl"
        lepard_cluster.write_text(json.dumps({"lepard_id": 1, "cl_cluster_id": 100}) + "\n")
        opinions = tmp_path / "opinions.csv.bz2"
        self._make_opinions_bz2(
            opinions,
            [
                {"id": "5000", "cluster_id": "100"},
                {"id": "3000", "cluster_id": "100"},  # smaller → should win
            ],
        )
        out = tmp_path / "c2o.json"
        stage4_cluster_to_opinion(lepard_cluster, opinions, out)
        data = json.loads(out.read_text())
        assert data == {"100": 3000}


# ----- Stage 5: compose ------------------------------------------------------


class TestStage5Compose:
    def test_composes_lepard_to_opinion_map(self, tmp_path):
        lepard_cluster = tmp_path / "joined.jsonl"
        lepard_cluster.write_text(
            json.dumps({"lepard_id": 1, "cl_cluster_id": 100})
            + "\n"
            + json.dumps({"lepard_id": 2, "cl_cluster_id": 999})
            + "\n"
        )
        c2o = tmp_path / "c2o.json"
        c2o.write_text(json.dumps({"100": 1000}))
        out = tmp_path / "final.json"
        meta = stage5_final_map(lepard_cluster, c2o, out)
        assert json.loads(out.read_text()) == {"1": 1000}
        assert meta["n_final_mappings"] == 1
        assert meta["n_lepard_cluster_pairs"] == 2


# ----- Stage 6: validation ---------------------------------------------------


class TestStage6Validate:
    def test_reports_hit_rate(self, tmp_path):
        final = tmp_path / "final.json"
        final.write_text(json.dumps({"42": 1000}))

        corpus = tmp_path / "corpus.jsonl"
        # CL opinion 1000 contains the quote substring
        corpus.write_text(
            json.dumps(
                {
                    "opinion_id": 1000,
                    "chunk_index": 0,
                    "text": "the court held that foo bar baz",
                }
            )
            + "\n"
        )

        lepard = tmp_path / "lepard.jsonl"
        lepard.write_text(
            json.dumps(
                {
                    "dest_id": 42,
                    "source_id": 1,
                    "dest_name": "Foo v Bar",
                    "dest_cite": "Foo v Bar, 1 F 1 (1900)",
                    "quote": "foo bar baz",
                }
            )
            + "\n"
        )

        out = tmp_path / "report.json"
        meta = stage6_validate(final, corpus, lepard, out, n_sample=1)
        report = json.loads(out.read_text())
        assert meta["n_sample"] == 1
        assert meta["n_quote_found_in_cl"] == 1
        assert meta["hit_rate"] == 1.0
        assert report["details"][0]["quote_found_in_cl_text"] is True

    def test_miss_when_quote_not_in_text(self, tmp_path):
        final = tmp_path / "final.json"
        final.write_text(json.dumps({"42": 1000}))
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(json.dumps({"opinion_id": 1000, "chunk_index": 0, "text": "unrelated text"}) + "\n")
        lepard = tmp_path / "lepard.jsonl"
        lepard.write_text(
            json.dumps(
                {
                    "dest_id": 42,
                    "source_id": 1,
                    "dest_name": "X",
                    "dest_cite": "X, 1 F 1 (1900)",
                    "quote": "distinctive quote not in text",
                }
            )
            + "\n"
        )
        out = tmp_path / "report.json"
        meta = stage6_validate(final, corpus, lepard, out, n_sample=1)
        assert meta["hit_rate"] == 0.0
