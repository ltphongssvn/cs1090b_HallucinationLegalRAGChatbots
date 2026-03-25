# tests/test_dataset_probe.py
"""
TDD contract tests for src/dataset_probe.py — CourtListener local shard probe.
Red phase: defines the contract before source implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset_probe import (
    CHUNK_OVERLAP_SUBWORDS,
    CHUNK_SIZE_SUBWORDS,
    ENCODER_MODEL,
    MIN_SENTENCE_COUNT,
    PROVISIONAL_MIN_TEXT_LENGTH,
    REQUIRED_FIELDS,
    SPACY_MODEL,
    CourtListenerDatasetProbe,
    gate_a7_text_source_breakdown,
    gate_a8_text_length_distribution,
    gate_a9_citation_count_distribution,
    gate_a11_tokenizer_chunk_count,
    gate_a12_citation_anchor_survival,
    gate_a13_sentence_density,
    gate_b6_text_entropy_distribution,
    run_probe,
    sample_records,
    validate_schema,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_RECORD: dict = {
    "id": "1",
    "cluster_id": "c1",
    "docket_id": "d1",
    "court_id": "ca9",
    "court_name": "Ninth Circuit",
    "case_name": "Smith v. Jones",
    "date_filed": "2020-01-01",
    "precedential_status": "Published",
    "opinion_type": "majority",
    "extracted_by_ocr": False,
    "raw_text": "Smith v. Jones, 123 F.3d 456 (9th Cir. 2020). AFFIRMED.",
    "text": "Smith v. Jones, 123 F.3d 456 (9th Cir. 2020). " + ("The court held. " * 60),
    "text_length": 2000,
    "text_source": "plain_text",
    "cleaning_flags": [],
    "source": "courtlistener",
    "token_count": 400,
    "paragraph_count": 5,
    "citation_count": 3,
    "text_hash": "abc123",
    "citation_density": 0.05,
    "is_precedential": True,
    "text_entropy": 4.2,
}


def _make_records(n: int, **overrides) -> list[dict]:
    records = []
    for i in range(n):
        r = dict(MINIMAL_RECORD)
        r["id"] = str(i)
        r.update(overrides)
        records.append(r)
    return records


@pytest.fixture
def sample_shard_dir(tmp_path: Path) -> Path:
    """Write a tiny .jsonl shard for sample_records() tests."""
    shard = tmp_path / "shard_000.jsonl"
    with open(shard, "w") as fh:
        for i in range(50):
            r = dict(MINIMAL_RECORD)
            r["id"] = str(i)
            fh.write(json.dumps(r) + "\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_required_fields_is_frozenset(self):
        assert isinstance(REQUIRED_FIELDS, frozenset)

    def test_required_fields_contains_core_schema(self):
        for f in (
            "id",
            "court_id",
            "text",
            "text_length",
            "text_source",
            "citation_count",
            "citation_density",
            "is_precedential",
            "text_entropy",
            "token_count",
            "paragraph_count",
        ):
            assert f in REQUIRED_FIELDS

    def test_chunk_size_is_1024(self):
        assert CHUNK_SIZE_SUBWORDS == 1024

    def test_chunk_overlap_is_128(self):
        assert CHUNK_OVERLAP_SUBWORDS == 128

    def test_encoder_model_is_bge_m3(self):
        assert ENCODER_MODEL == "BAAI/bge-m3"

    def test_spacy_model_is_en_core_web_sm(self):
        assert SPACY_MODEL == "en_core_web_sm"

    def test_min_sentence_count_is_50(self):
        assert MIN_SENTENCE_COUNT == 50

    def test_provisional_min_text_length_positive(self):
        assert PROVISIONAL_MIN_TEXT_LENGTH > 0


# ---------------------------------------------------------------------------
# sample_records
# ---------------------------------------------------------------------------


class TestSampleRecords:
    def test_returns_list(self, sample_shard_dir):
        result = sample_records(sample_shard_dir, 10)
        assert isinstance(result, list)

    def test_returns_correct_count(self, sample_shard_dir):
        result = sample_records(sample_shard_dir, 10)
        assert len(result) == 10

    def test_returns_all_when_n_exceeds_total(self, sample_shard_dir):
        result = sample_records(sample_shard_dir, 1000)
        assert len(result) == 50

    def test_deterministic_with_seed(self, sample_shard_dir):
        r1 = sample_records(sample_shard_dir, 10, seed=0)
        r2 = sample_records(sample_shard_dir, 10, seed=0)
        assert [r["id"] for r in r1] == [r["id"] for r in r2]

    def test_different_seeds_give_different_samples(self, sample_shard_dir):
        r1 = sample_records(sample_shard_dir, 10, seed=0)
        r2 = sample_records(sample_shard_dir, 10, seed=99)
        assert [r["id"] for r in r1] != [r["id"] for r in r2]

    def test_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sample_records(tmp_path / "nonexistent", 10)

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sample_records(tmp_path, 10)


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_passes_complete_records(self):
        result = validate_schema(_make_records(5))
        assert result["pass"] is True
        assert result["missing_counts"] == {}

    def test_fails_missing_field(self):
        records = _make_records(5)
        for r in records:
            del r["court_id"]
        result = validate_schema(records)
        assert result["pass"] is False
        assert "court_id" in result["missing_counts"]

    def test_reports_all_missing_fields(self):
        records = _make_records(3)
        for r in records:
            del r["text"]
            del r["text_entropy"]
        result = validate_schema(records)
        assert "text" in result["missing_counts"]
        assert "text_entropy" in result["missing_counts"]

    def test_gate_key_present(self):
        result = validate_schema(_make_records(3))
        assert result["gate"] == "schema_validation"


# ---------------------------------------------------------------------------
# Gate A7 — text_source breakdown
# ---------------------------------------------------------------------------


class TestGateA7:
    def test_gate_key(self):
        r = gate_a7_text_source_breakdown(_make_records(10))
        assert r["gate"] == "A7_text_source_breakdown"

    def test_pass_when_known_formats_dominant(self):
        records = _make_records(85, text_source="plain_text") + _make_records(15, text_source="html_with_citations")
        r = gate_a7_text_source_breakdown(records)
        assert r["pass"] is True

    def test_fail_when_unknown_formats_dominant(self):
        records = _make_records(30, text_source="plain_text") + _make_records(70, text_source="html_lawbox")
        r = gate_a7_text_source_breakdown(records)
        assert r["pass"] is False

    def test_breakdown_sums_to_100_pct(self):
        records = _make_records(60, text_source="plain_text") + _make_records(40, text_source="html_with_citations")
        r = gate_a7_text_source_breakdown(records)
        total = sum(v["pct"] for v in r["breakdown"].values())
        assert abs(total - 100.0) < 0.5

    def test_unknown_formats_pct_reported(self):
        records = (
            _make_records(50, text_source="plain_text")
            + _make_records(33, text_source="html_with_citations")
            + _make_records(17, text_source="html_lawbox")
        )
        r = gate_a7_text_source_breakdown(records)
        assert r["unknown_formats_pct"] > 0


# ---------------------------------------------------------------------------
# Gate A8 — text_length distribution
# ---------------------------------------------------------------------------


class TestGateA8:
    def test_gate_key(self):
        r = gate_a8_text_length_distribution(_make_records(10))
        assert r["gate"] == "A8_text_length_distribution"

    def test_reports_mean_median_min_max(self):
        r = gate_a8_text_length_distribution(_make_records(10))
        for key in ("mean", "median", "min", "max", "p10", "p90"):
            assert key in r

    def test_pass_when_few_below_threshold(self):
        records = _make_records(95, text_length=5000) + _make_records(5, text_length=100)
        r = gate_a8_text_length_distribution(records)
        assert r["pass"] is True

    def test_fail_when_many_below_threshold(self):
        records = _make_records(50, text_length=5000) + _make_records(50, text_length=100)
        r = gate_a8_text_length_distribution(records)
        assert r["pass"] is False

    def test_below_provisional_count_correct(self):
        records = _make_records(8, text_length=5000) + _make_records(2, text_length=100)
        r = gate_a8_text_length_distribution(records)
        assert r["below_provisional_count"] == 2


# ---------------------------------------------------------------------------
# Gate A9 — citation_count distribution
# ---------------------------------------------------------------------------


class TestGateA9:
    def test_gate_key(self):
        r = gate_a9_citation_count_distribution(_make_records(10))
        assert r["gate"] == "A9_citation_count_distribution"

    def test_pass_when_few_zero_citation(self):
        records = _make_records(90, citation_count=5) + _make_records(10, citation_count=0)
        r = gate_a9_citation_count_distribution(records)
        assert r["pass"] is True

    def test_fail_when_many_zero_citation(self):
        records = _make_records(50, citation_count=5) + _make_records(50, citation_count=0)
        r = gate_a9_citation_count_distribution(records)
        assert r["pass"] is False

    def test_above_5_count_reported(self):
        records = _make_records(70, citation_count=10) + _make_records(30, citation_count=2)
        r = gate_a9_citation_count_distribution(records)
        assert r["above_5_count"] == 70


# ---------------------------------------------------------------------------
# Gate A12 — citation anchor survival
# ---------------------------------------------------------------------------


class TestGateA12:
    def test_gate_key(self):
        r = gate_a12_citation_anchor_survival(_make_records(20))
        assert r["gate"] == "A12_citation_anchor_survival"

    def test_pass_when_most_have_anchors(self):
        records = _make_records(100)
        r = gate_a12_citation_anchor_survival(records)
        assert r["pass"] is True

    def test_fail_when_few_have_anchors(self):
        records = _make_records(100, text="No citations here at all.")
        r = gate_a12_citation_anchor_survival(records)
        assert r["pass"] is False

    def test_pct_reported(self):
        r = gate_a12_citation_anchor_survival(_make_records(50))
        assert "pct_with_citation_anchor" in r


# ---------------------------------------------------------------------------
# Gate A13 — sentence density (spaCy)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGateA13:
    def test_gate_key(self):
        long_text = "The court held this. " * 100
        records = _make_records(5, text=long_text)
        r = gate_a13_sentence_density(records)
        assert r["gate"] == "A13_sentence_density"

    def test_pass_when_dense_text(self):
        long_text = "The court held this point clearly. " * 100
        records = _make_records(10, text=long_text)
        r = gate_a13_sentence_density(records)
        assert r["pass"] is True

    def test_fail_when_sparse_text(self):
        short_text = "Affirmed."
        records = _make_records(10, text=short_text)
        r = gate_a13_sentence_density(records)
        assert r["pass"] is False

    def test_uses_spacy_not_nltk(self):
        # Verify spacy_model key is reported (confirms spaCy path used)
        long_text = "The court held. " * 100
        records = _make_records(5, text=long_text)
        r = gate_a13_sentence_density(records)
        assert r.get("spacy_model") == "en_core_web_sm"


# ---------------------------------------------------------------------------
# Gate A11 — tokenizer chunk count (marked slow, requires HF model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGateA11:
    def test_gate_key(self):
        records = _make_records(5)
        r = gate_a11_tokenizer_chunk_count(records, sample_n=5)
        assert r["gate"] == "A11_tokenizer_chunk_count"

    def test_reports_encoder_model(self):
        records = _make_records(5)
        r = gate_a11_tokenizer_chunk_count(records, sample_n=5)
        assert r.get("encoder_model") == "BAAI/bge-m3"

    def test_pass_when_median_chunks_gte_2(self):
        long_text = "The court held. " * 2000
        records = _make_records(10, text=long_text)
        r = gate_a11_tokenizer_chunk_count(records, sample_n=10)
        assert r["pass"] is True


# ---------------------------------------------------------------------------
# Gate B6 — text_entropy distribution
# ---------------------------------------------------------------------------


class TestGateB6:
    def test_gate_key(self):
        r = gate_b6_text_entropy_distribution(_make_records(10))
        assert r["gate"] == "B6_text_entropy_distribution"

    def test_always_passes_distribution_only(self):
        r = gate_b6_text_entropy_distribution(_make_records(10))
        assert r["pass"] is True

    def test_reports_percentiles(self):
        r = gate_b6_text_entropy_distribution(_make_records(20))
        for key in ("p5", "p10", "p25", "p75", "p90", "p95"):
            assert key in r

    def test_zero_entropy_count_reported(self):
        records = _make_records(8, text_entropy=4.0) + _make_records(2, text_entropy=0.0)
        r = gate_b6_text_entropy_distribution(records)
        assert r["zero_entropy_count"] == 2


# ---------------------------------------------------------------------------
# CourtListenerDatasetProbe class contract
# ---------------------------------------------------------------------------


class TestCourtListenerDatasetProbe:
    def test_probe_class_exists(self):
        probe = CourtListenerDatasetProbe()
        assert probe is not None

    def test_run_returns_report_dict(self, sample_shard_dir, tmp_path):
        probe = CourtListenerDatasetProbe()
        report = probe.run(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "report.json",
            skip_tokenizer=True,
            skip_spacy=False,
        )
        assert isinstance(report, dict)
        assert "gates" in report
        assert "summary" in report

    def test_report_written_to_disk(self, sample_shard_dir, tmp_path):
        probe = CourtListenerDatasetProbe()
        out = tmp_path / "report.json"
        probe.run(
            data_dir=sample_shard_dir,
            subset=20,
            output=out,
            skip_tokenizer=True,
        )
        assert out.exists()
        with open(out) as fh:
            data = json.load(fh)
        assert "gates" in data

    def test_summary_all_passed_key_present(self, sample_shard_dir, tmp_path):
        probe = CourtListenerDatasetProbe()
        report = probe.run(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
        )
        assert "all_passed" in report["summary"]

    def test_skipped_gates_recorded(self, sample_shard_dir, tmp_path):
        probe = CourtListenerDatasetProbe()
        report = probe.run(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "A11" in report["summary"]["skipped"]
        assert "A13" in report["summary"]["skipped"]


# ---------------------------------------------------------------------------
# run_probe (module-level function)
# ---------------------------------------------------------------------------


class TestRunProbe:
    def test_run_probe_returns_dict(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "out.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert isinstance(report, dict)

    def test_run_probe_writes_json(self, sample_shard_dir, tmp_path):
        out = tmp_path / "out.json"
        run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=out,
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert out.exists()
