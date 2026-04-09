"""Tests for src/lepard_cl_compat.py.

Written TDD-first (test file committed before source).
Covers loaders, pure analysis functions, integration, error paths,
and property-based invariants (hypothesis).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.lepard_cl_compat import (
    CompatReport,
    analyze_court_distribution,
    compute_id_overlap,
    compute_pair_overlap,
    format_report,
    load_cl_ids,
    load_court_map,
    load_lepard_pairs,
    run_full_analysis,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------- synthetic fixtures ----------


@pytest.fixture
def tmp_lepard(tmp_path: Path) -> Path:
    path = tmp_path / "lepard.jsonl"
    rows = [
        {"source_id": 100, "dest_id": 200},
        {"source_id": 100, "dest_id": 200},
        {"source_id": 100, "dest_id": 300},
        {"source_id": 999, "dest_id": 888},
        {"source_id": 100, "dest_id": 777},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


@pytest.fixture
def tmp_cl_ids_gz(tmp_path: Path) -> Path:
    path = tmp_path / "cl_ids.txt.gz"
    with gzip.open(path, "wt") as f:
        for i in [100, 200, 300, 400]:
            f.write(f"{i}\n")
    return path


@pytest.fixture
def tmp_court_map(tmp_path: Path) -> Path:
    path = tmp_path / "courts.json"
    path.write_text(json.dumps({"100": "ca9", "200": "ca5", "300": "ca9"}))
    return path


# ---------- loader contracts ----------


class TestLoaders:
    def test_load_lepard_pairs_preserves_duplicates(self, tmp_lepard):
        pairs = load_lepard_pairs(tmp_lepard)
        assert len(pairs) == 5
        assert pairs[0] == (100, 200)
        assert pairs[1] == (100, 200)

    def test_load_lepard_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_lepard_pairs(tmp_path / "nope.jsonl")

    def test_load_lepard_malformed_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"source_id": 1, "dest_id": 2}\nnot json\n')
        with pytest.raises(ValueError, match="malformed JSON"):
            load_lepard_pairs(path)

    def test_load_lepard_missing_required_key_raises_valueerror(self, tmp_path):
        path = tmp_path / "missing_key.jsonl"
        path.write_text('{"dest_id": 1}\n')
        with pytest.raises(ValueError, match=r"missing required key.*source_id.*line 0"):
            load_lepard_pairs(path)

    def test_load_cl_ids_gzipped(self, tmp_cl_ids_gz):
        ids = load_cl_ids(tmp_cl_ids_gz)
        assert ids == {100, 200, 300, 400}

    def test_load_cl_ids_plain(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("1\n2\n3\n")
        assert load_cl_ids(path) == {1, 2, 3}

    def test_load_court_map_missing_returns_empty(self, tmp_path):
        assert load_court_map(tmp_path / "nope.json") == {}

    def test_load_court_map_coerces_int_keys(self, tmp_court_map):
        cm = load_court_map(tmp_court_map)
        assert cm == {100: "ca9", 200: "ca5", 300: "ca9"}


# ---------- analysis contracts ----------


class TestIdOverlap:
    def test_basic(self):
        pairs = [(100, 200), (300, 400), (999, 888)]
        cl = {100, 200, 300, 400, 500}
        r = compute_id_overlap(pairs, cl)
        assert r.lepard_unique_ids == 6
        assert r.cl_total_ids == 5
        assert r.overlap == 4
        assert r.overlap_pct_of_lepard == pytest.approx(100 * 4 / 6)
        assert r.lepard_max == 999
        assert r.cl_max == 500
        assert r.lepard_ids_above_cl_max == 2

    def test_empty_cl(self):
        r = compute_id_overlap([(1, 2)], set())
        assert r.overlap == 0
        assert r.cl_max == 0
        assert r.lepard_ids_above_cl_max == 2


class TestPairOverlap:
    def test_deduplication(self):
        pairs = [(1, 2), (1, 2), (1, 2)]
        cl = {1, 2}
        r = compute_pair_overlap(pairs, cl)
        assert r.total_rows == 3
        assert r.unique_pairs == 1
        assert r.both_in_cl == 1
        assert r.usable_pct == 100.0

    def test_all_four_buckets(self):
        pairs = [(1, 2), (1, 99), (99, 2), (98, 97)]
        cl = {1, 2}
        r = compute_pair_overlap(pairs, cl)
        assert r.unique_pairs == 4
        assert r.both_in_cl == 1
        assert r.source_only_in_cl == 1
        assert r.dest_only_in_cl == 1
        assert r.neither_in_cl == 1
        assert r.usable_pct == 25.0

    def test_unique_sources_and_dests(self):
        pairs = [(1, 10), (1, 20), (2, 10)]
        r = compute_pair_overlap(pairs, {1, 2, 10, 20})
        assert r.unique_sources == 2
        assert r.unique_dests == 2


class TestPairOverlapProperty:
    """Property-based invariants for compute_pair_overlap (hypothesis)."""

    @given(
        pairs=st.lists(
            st.tuples(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000)),
            min_size=0,
            max_size=200,
        ),
        cl_ids=st.sets(st.integers(min_value=0, max_value=1000), max_size=500),
    )
    def test_buckets_sum_to_unique_pairs(self, pairs, cl_ids):
        """The four mutually exclusive buckets must partition unique pairs exactly."""
        r = compute_pair_overlap(pairs, cl_ids)
        assert r.both_in_cl + r.source_only_in_cl + r.dest_only_in_cl + r.neither_in_cl == r.unique_pairs

    @given(
        pairs=st.lists(
            st.tuples(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000)),
            min_size=0,
            max_size=200,
        ),
        cl_ids=st.sets(st.integers(min_value=0, max_value=1000), max_size=500),
    )
    def test_usable_pct_within_bounds(self, pairs, cl_ids):
        """usable_pct must always be in [0, 100]."""
        r = compute_pair_overlap(pairs, cl_ids)
        assert 0.0 <= r.usable_pct <= 100.0

    @given(
        pairs=st.lists(
            st.tuples(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000)),
            min_size=1,
            max_size=200,
        ),
    )
    def test_total_rows_never_less_than_unique(self, pairs):
        """Deduplication invariant: total_rows >= unique_pairs."""
        r = compute_pair_overlap(pairs, set())
        assert r.total_rows >= r.unique_pairs


class TestCourtDistribution:
    def test_counts_matched_only(self):
        pairs = [(100, 200), (100, 999)]
        cl = {100, 200}
        court_map = {100: "ca9", 200: "ca9", 999: "ca5"}
        dist = analyze_court_distribution(pairs, cl, court_map)
        assert dist == {"ca9": 2}

    def test_empty_court_map(self):
        assert analyze_court_distribution([(1, 2)], {1, 2}, {}) == {}

    def test_sorted_descending(self):
        pairs = [(1, 2), (3, 4), (5, 6)]
        cl = {1, 2, 3, 4, 5, 6}
        court_map = {1: "ca9", 2: "ca9", 3: "ca9", 4: "ca5", 5: "ca5", 6: "ca1"}
        dist = analyze_court_distribution(pairs, cl, court_map)
        assert list(dist.keys()) == ["ca9", "ca5", "ca1"]


# ---------- integration contracts ----------


class TestRunFullAnalysis:
    def test_end_to_end_synthetic(self, tmp_lepard, tmp_cl_ids_gz, tmp_court_map):
        report = run_full_analysis(tmp_lepard, tmp_cl_ids_gz, tmp_court_map)
        assert isinstance(report, CompatReport)
        assert report.pair_overlap.unique_pairs == 4
        assert report.pair_overlap.both_in_cl == 2
        assert "ca9" in report.court_distribution

    def test_report_is_serializable(self, tmp_lepard, tmp_cl_ids_gz, tmp_court_map):
        report = run_full_analysis(tmp_lepard, tmp_cl_ids_gz, tmp_court_map)
        s = json.dumps(report.to_dict())
        assert "id_overlap" in json.loads(s)

    def test_format_report_human_readable(self, tmp_lepard, tmp_cl_ids_gz, tmp_court_map):
        report = run_full_analysis(tmp_lepard, tmp_cl_ids_gz, tmp_court_map)
        text = format_report(report)
        assert "LePaRD" in text and "USABLE GOLD" in text


# ---------- real fixtures (skipped until committed) ----------


@pytest.mark.skipif(
    not (FIXTURES / "lepard_sample_1k.jsonl").exists() or not (FIXTURES / "cl_ids.txt.gz").exists(),
    reason="committed fixtures not present",
)
class TestRealFixtures:
    """Regression test against the numbers observed in the live investigation."""

    def test_matches_live_investigation(self):
        report = run_full_analysis()
        assert report.id_overlap.lepard_unique_ids == 512
        assert report.id_overlap.overlap == 70
        assert report.pair_overlap.unique_pairs == 454
        assert report.pair_overlap.both_in_cl == 13
        if report.court_distribution:
            top_court = next(iter(report.court_distribution))
            assert top_court.startswith("ca")


# ---------- CLI threshold gate ----------


class TestMinUsablePctCliGate:
    """--min-usable-pct exits non-zero when usable_pct falls below threshold."""

    def test_cli_exits_zero_when_above_threshold(self, tmp_lepard, tmp_cl_ids_gz, tmp_court_map):
        import subprocess
        import sys

        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.lepard_cl_compat",
                "--lepard",
                str(tmp_lepard),
                "--cl-ids",
                str(tmp_cl_ids_gz),
                "--court-map",
                str(tmp_court_map),
                "--min-usable-pct",
                "10.0",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_cli_exits_nonzero_when_below_threshold(self, tmp_lepard, tmp_cl_ids_gz, tmp_court_map):
        import subprocess
        import sys

        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.lepard_cl_compat",
                "--lepard",
                str(tmp_lepard),
                "--cl-ids",
                str(tmp_cl_ids_gz),
                "--court-map",
                str(tmp_court_map),
                "--min-usable-pct",
                "99.0",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert r.returncode != 0
        assert "below threshold" in r.stderr.lower() or "below threshold" in r.stdout.lower()


# ---------- --write-valid-pairs flag ----------


class TestWriteValidPairsFlag:
    """--write-valid-pairs emits usable gold pairs (both endpoints in CL) as JSONL."""

    def test_writes_only_both_in_cl_pairs(self, tmp_lepard, tmp_cl_ids_gz, tmp_court_map, tmp_path):
        import subprocess
        import sys

        out = tmp_path / "valid_pairs.jsonl"
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.lepard_cl_compat",
                "--lepard",
                str(tmp_lepard),
                "--cl-ids",
                str(tmp_cl_ids_gz),
                "--court-map",
                str(tmp_court_map),
                "--write-valid-pairs",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        pairs = [json.loads(line) for line in lines]
        # Synthetic fixture has exactly 2 unique pairs with both endpoints in cl={100,200,300,400}:
        # (100, 200) and (100, 300). (100, 777), (999, 888) excluded.
        assert len(pairs) == 2
        pair_tuples = {(p["source_id"], p["dest_id"]) for p in pairs}
        assert pair_tuples == {(100, 200), (100, 300)}

    def test_deduplicates_rows(self, tmp_lepard, tmp_cl_ids_gz, tmp_court_map, tmp_path):
        """tmp_lepard contains (100,200) twice; output must dedupe."""
        import subprocess
        import sys

        out = tmp_path / "valid_pairs.jsonl"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.lepard_cl_compat",
                "--lepard",
                str(tmp_lepard),
                "--cl-ids",
                str(tmp_cl_ids_gz),
                "--court-map",
                str(tmp_court_map),
                "--write-valid-pairs",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            check=True,
        )
        lines = [line for line in out.read_text().strip().split("\n") if line]
        assert len(lines) == 2  # deduped, not 3
