# tests/test_dataset_probe.py
"""
Full contract tests for src/dataset_probe.py — CourtListener local shard probe.
Single authoritative test file covering all contracts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dataset_probe import (
    CHUNK_OVERLAP_SUBWORDS,
    CHUNK_SIZE_SUBWORDS,
    ENCODER_MODEL,
    MIN_SENTENCE_COUNT,
    PROBE_VERSION,
    PROVISIONAL_MIN_TEXT_LENGTH,
    SPACY_MODEL,
    CourtListenerDatasetProbe,
    ModelQualitySignals,
    ProbeConfig,
    _percentile,
    gate_a7_text_source_breakdown,
    gate_a8_text_length_distribution,
    gate_a9_citation_count_distribution,
    gate_a11_tokenizer_chunk_count,
    gate_a12_citation_anchor_survival,
    gate_a13_sentence_density,
    gate_b6_text_entropy_distribution,
    iter_shards,
    iter_shards_with_audit,
    run_probe,
    sample_records,
    validate_schema,
)

pytestmark = pytest.mark.unit

FIXTURE_JSONL = Path("tests/fixtures/courtlistener_sample.jsonl")

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
    shard = tmp_path / "shard_000.jsonl"
    with open(shard, "w") as fh:
        for i in range(50):
            r = dict(MINIMAL_RECORD)
            r["id"] = str(i)
            fh.write(json.dumps(r) + "\n")
    return tmp_path


# ---------------------------------------------------------------------------
# ProbeConfig
# ---------------------------------------------------------------------------


class TestProbeConfig:
    def test_is_frozen(self):
        cfg = ProbeConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.min_text_length = 999  # type: ignore[misc]

    def test_default_min_text_length(self):
        assert ProbeConfig().min_text_length == PROVISIONAL_MIN_TEXT_LENGTH

    def test_default_chunk_size(self):
        assert ProbeConfig().chunk_size_subwords == CHUNK_SIZE_SUBWORDS

    def test_default_chunk_overlap(self):
        assert ProbeConfig().chunk_overlap_subwords == CHUNK_OVERLAP_SUBWORDS

    def test_default_min_sentence_count(self):
        assert ProbeConfig().min_sentence_count == MIN_SENTENCE_COUNT

    def test_default_encoder_model(self):
        assert ProbeConfig().encoder_model == ENCODER_MODEL

    def test_custom_values_accepted(self):
        cfg = ProbeConfig(min_text_length=500)
        assert cfg.min_text_length == 500

    def test_is_json_serializable(self):
        import dataclasses

        cfg = ProbeConfig()
        d = dataclasses.asdict(cfg)
        json.dumps(d)


class TestProbeConfigMagicNumbers:
    def test_has_a13_text_cap_chars(self):
        assert hasattr(ProbeConfig(), "a13_text_cap_chars")

    def test_a13_text_cap_chars_default(self):
        assert ProbeConfig().a13_text_cap_chars == 50_000

    def test_has_a11_subsample_n(self):
        assert hasattr(ProbeConfig(), "a11_subsample_n")

    def test_a11_subsample_n_default(self):
        assert ProbeConfig().a11_subsample_n == 200

    def test_has_a12_subsample_n(self):
        assert hasattr(ProbeConfig(), "a12_subsample_n")

    def test_a12_subsample_n_default(self):
        assert ProbeConfig().a12_subsample_n == 500

    def test_has_a13_subsample_n(self):
        assert hasattr(ProbeConfig(), "a13_subsample_n")

    def test_a13_subsample_n_default(self):
        assert ProbeConfig().a13_subsample_n == 200

    def test_custom_subsample_n_accepted(self):
        cfg = ProbeConfig(a11_subsample_n=50)
        assert cfg.a11_subsample_n == 50

    def test_custom_text_cap_accepted(self):
        cfg = ProbeConfig(a13_text_cap_chars=10_000)
        assert cfg.a13_text_cap_chars == 10_000

    def test_all_new_fields_in_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        config = report["provenance"]["probe_config"]
        assert "a13_text_cap_chars" in config
        assert "a11_subsample_n" in config
        assert "a12_subsample_n" in config
        assert "a13_subsample_n" in config


class TestProbeConfigGenerativeModel:
    def test_has_a11_generative_model(self):
        assert hasattr(ProbeConfig(), "a11_generative_model")

    def test_a11_generative_model_default_is_mistral(self):
        val = ProbeConfig().a11_generative_model.lower()
        assert "mistral" in val

    def test_a11_generative_model_custom_accepted(self):
        cfg = ProbeConfig(a11_generative_model="meta-llama/Llama-2-7b")
        assert cfg.a11_generative_model == "meta-llama/Llama-2-7b"

    def test_a11_generative_model_empty_string_accepted(self):
        cfg = ProbeConfig(a11_generative_model="")
        assert cfg.a11_generative_model == ""


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_p0_returns_min(self):
        assert _percentile([1, 2, 3, 4, 5], 0) == 1

    def test_p100_returns_max(self):
        assert _percentile([1, 2, 3, 4, 5], 100) == 5

    def test_p50_returns_median(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == 3

    def test_single_element(self):
        assert _percentile([42], 50) == 42

    def test_requires_sorted_input(self):
        result = _percentile(sorted([10, 1, 5, 3]), 50)
        assert result in [3, 5]


# ---------------------------------------------------------------------------
# iter_shards
# ---------------------------------------------------------------------------


class TestIterShards:
    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(iter_shards(tmp_path))

    def test_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(iter_shards(tmp_path / "nonexistent"))

    def test_yields_valid_records(self, sample_shard_dir):
        assert len(list(iter_shards(sample_shard_dir))) == 50

    def test_skips_blank_lines(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('\n\n{"id":"1","text":"x"}\n\n')
        assert len(list(iter_shards(tmp_path))) == 1

    def test_counts_malformed_lines(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id":"1","text":"good"}\nNOT_JSON\n{"id":"2","text":"good"}\n')
        audit = iter_shards_with_audit(tmp_path)
        assert len(audit["records"]) == 2
        assert audit["total_parse_errors"] == 1

    def test_shard_level_diagnostics_available(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id":"1","text":"good"}\nBAD\n')
        audit = iter_shards_with_audit(tmp_path)
        assert audit["total_parse_errors"] == 1
        assert audit["shard_errors"]["s.jsonl"] == 1

    def test_blank_lines_counted_separately(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('\n{"id":"1","text":"good"}\n\n')
        audit = iter_shards_with_audit(tmp_path)
        assert audit["total_blank_lines"] == 2
        assert audit["total_parse_errors"] == 0


# ---------------------------------------------------------------------------
# sample_records
# ---------------------------------------------------------------------------


class TestSampleRecords:
    def test_returns_correct_count(self, sample_shard_dir):
        assert len(sample_records(sample_shard_dir, 10)) == 10

    def test_returns_all_when_n_exceeds_total(self, sample_shard_dir):
        assert len(sample_records(sample_shard_dir, 1000)) == 50

    def test_deterministic_with_seed(self, sample_shard_dir):
        r1 = sample_records(sample_shard_dir, 10, seed=0)
        r2 = sample_records(sample_shard_dir, 10, seed=0)
        assert [r["id"] for r in r1] == [r["id"] for r in r2]

    def test_different_seeds_differ(self, sample_shard_dir):
        r1 = sample_records(sample_shard_dir, 10, seed=0)
        r2 = sample_records(sample_shard_dir, 10, seed=99)
        assert [r["id"] for r in r1] != [r["id"] for r in r2]

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sample_records(tmp_path, 10)


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_passes_complete_records(self):
        assert validate_schema(_make_records(5))["pass"] is True

    def test_fails_missing_field(self):
        records = _make_records(5)
        for r in records:
            del r["court_id"]
        result = validate_schema(records)
        assert result["pass"] is False
        assert "court_id" in result["missing_counts"]

    def test_fails_non_integer_text_length(self):
        result = validate_schema(_make_records(3, text_length="not_an_int"))
        assert result["pass"] is False
        assert "type_errors" in result

    def test_fails_negative_citation_count(self):
        result = validate_schema(_make_records(3, citation_count=-1))
        assert result["pass"] is False
        assert "range_errors" in result

    def test_fails_non_bool_is_precedential(self):
        result = validate_schema(_make_records(3, is_precedential="yes"))
        assert result["pass"] is False
        assert "type_errors" in result

    def test_gate_key_present(self):
        assert validate_schema(_make_records(3))["gate"] == "schema_validation"

    def test_empty_records_handled(self):
        assert "pass" in validate_schema([])


# ---------------------------------------------------------------------------
# Gate A7
# ---------------------------------------------------------------------------


class TestGateA7:
    def test_gate_key(self):
        assert gate_a7_text_source_breakdown(_make_records(10))["gate"] == "A7_text_source_breakdown"

    def test_pass_when_known_formats_dominant(self):
        records = _make_records(85, text_source="plain_text") + _make_records(15, text_source="html_with_citations")
        assert gate_a7_text_source_breakdown(records)["pass"] is True

    def test_fail_when_unknown_formats_dominant(self):
        records = _make_records(30, text_source="plain_text") + _make_records(70, text_source="html_lawbox")
        assert gate_a7_text_source_breakdown(records)["pass"] is False

    def test_unknown_formats_pct_reported(self):
        records = _make_records(50, text_source="plain_text") + _make_records(50, text_source="html_lawbox")
        assert gate_a7_text_source_breakdown(records)["unknown_formats_pct"] > 0

    def test_empty_records_handled(self):
        assert "pass" in gate_a7_text_source_breakdown([])


# ---------------------------------------------------------------------------
# Gate A8
# ---------------------------------------------------------------------------


class TestGateA8:
    def test_passes_at_24_99_pct(self):
        records = _make_records(7501, text_length=5000) + _make_records(2499, text_length=100)
        assert gate_a8_text_length_distribution(records)["pass"] is True

    def test_fails_at_25_pct(self):
        records = _make_records(7500, text_length=5000) + _make_records(2500, text_length=100)
        assert gate_a8_text_length_distribution(records)["pass"] is False

    def test_reports_percentiles(self):
        r = gate_a8_text_length_distribution(_make_records(20))
        for key in ("p5", "p10", "p25", "p75", "p90", "p95"):
            assert key in r

    def test_empty_records_handled(self):
        assert "pass" in gate_a8_text_length_distribution([])


# ---------------------------------------------------------------------------
# Gate A9 — advisory note
# ---------------------------------------------------------------------------


class TestGateA9:
    def test_passes_at_19_99_pct(self):
        records = _make_records(8001, citation_count=5) + _make_records(1999, citation_count=0)
        assert gate_a9_citation_count_distribution(records)["pass"] is True

    def test_fails_at_20_pct(self):
        records = _make_records(8000, citation_count=5) + _make_records(2000, citation_count=0)
        assert gate_a9_citation_count_distribution(records)["pass"] is False

    def test_above_5_count_reported(self):
        records = _make_records(70, citation_count=10) + _make_records(30, citation_count=2)
        assert gate_a9_citation_count_distribution(records)["above_5_count"] == 70

    def test_empty_records_handled(self):
        assert "pass" in gate_a9_citation_count_distribution([])

    def test_note_clarifies_advisory_role(self):
        """A9 note must clarify it is advisory, not a hard corpus filter."""
        r = gate_a9_citation_count_distribution(_make_records(10))
        note = r["note"].lower()
        assert "advisory" in note or "probe" in note or "filter" in note

    def test_note_mentions_full_corpus_unfiltered(self):
        r = gate_a9_citation_count_distribution(_make_records(10))
        note = r["note"].lower()
        assert "full" in note or "corpus" in note or "final" in note


# ---------------------------------------------------------------------------
# Gate A11 — mocked tokenizer + generative tokenizer check
# ---------------------------------------------------------------------------


class TestGateA11:
    @patch("src.dataset_probe.AutoTokenizer")
    def test_pass_when_median_chunks_gte_2(self, mock_cls):
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_cls.from_pretrained.return_value = mock_tok
        assert gate_a11_tokenizer_chunk_count(_make_records(10))["pass"] is True

    @patch("src.dataset_probe.AutoTokenizer")
    def test_logs_encoder_model(self, mock_cls):
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_cls.from_pretrained.return_value = mock_tok
        assert gate_a11_tokenizer_chunk_count(_make_records(5)).get("encoder_model") == ENCODER_MODEL

    @patch("src.dataset_probe.AutoTokenizer")
    def test_logs_tokenizer_revision(self, mock_cls):
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_cls.from_pretrained.return_value = mock_tok
        assert "tokenizer_revision" in gate_a11_tokenizer_chunk_count(_make_records(5))

    @patch("src.dataset_probe.AutoTokenizer")
    def test_fail_on_tokenizer_load_error(self, mock_cls):
        mock_cls.from_pretrained.side_effect = OSError("not found")
        r = gate_a11_tokenizer_chunk_count(_make_records(5))
        assert r["pass"] is False
        assert "error" in r

    def test_empty_records_handled(self):
        assert "pass" in gate_a11_tokenizer_chunk_count([])

    def test_uses_all_records_when_fewer_than_subsample_n(self):
        """No internal subsampling — all 5 records must be processed."""
        with patch("src.dataset_probe.AutoTokenizer") as mock_cls:
            mock_tok = MagicMock()
            mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
            mock_cls.from_pretrained.return_value = mock_tok
            r = gate_a11_tokenizer_chunk_count(_make_records(5))
            assert r["subsample_n"] == 5

    @patch("src.dataset_probe.AutoTokenizer")
    def test_reports_generative_token_check_when_model_set(self, mock_cls):
        """A11 must report generative_token_check when a11_generative_model is set."""
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_cls.from_pretrained.return_value = mock_tok
        cfg = ProbeConfig(a11_generative_model="mistralai/Mistral-7B-Instruct-v0.2")
        r = gate_a11_tokenizer_chunk_count(_make_records(5), config=cfg)
        assert "generative_token_check" in r

    @patch("src.dataset_probe.AutoTokenizer")
    def test_skips_generative_check_when_model_empty(self, mock_cls):
        """A11 must skip generative check when a11_generative_model is empty."""
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_cls.from_pretrained.return_value = mock_tok
        cfg = ProbeConfig(a11_generative_model="")
        r = gate_a11_tokenizer_chunk_count(_make_records(5), config=cfg)
        assert "generative_token_check" not in r


# ---------------------------------------------------------------------------
# Gate A12 — no internal subsampling
# ---------------------------------------------------------------------------


class TestGateA12:
    def test_pass_when_most_have_anchors(self):
        assert gate_a12_citation_anchor_survival(_make_records(100))["pass"] is True

    def test_fail_when_few_have_anchors(self):
        assert gate_a12_citation_anchor_survival(_make_records(100, text="No citations here."))["pass"] is False

    def test_note_states_approximation(self):
        r = gate_a12_citation_anchor_survival(_make_records(10))
        assert "heuristic" in r["note"].lower() or "approximat" in r["note"].lower()

    def test_empty_records_handled(self):
        assert "pass" in gate_a12_citation_anchor_survival([])

    def test_uses_all_records_when_fewer_than_subsample_n(self):
        """No internal subsampling — all 5 records must be processed."""
        r = gate_a12_citation_anchor_survival(_make_records(5))
        assert r["subsample_n"] == 5


# ---------------------------------------------------------------------------
# Gate A13 — nlp injection + no internal subsampling
# ---------------------------------------------------------------------------


class TestGateA13:
    def test_excludes_short_docs_below_a8_threshold(self):
        short = _make_records(50, text_length=100, text="Short.")
        long_text = "The court held this point clearly. " * 100
        long = _make_records(50, text_length=5000, text=long_text)
        r = gate_a13_sentence_density(short + long)
        assert r["records_after_a8_filter"] == 50

    def test_pass_when_dense_text(self):
        long_text = "The court held this point clearly. " * 100
        assert gate_a13_sentence_density(_make_records(20, text_length=5000, text=long_text))["pass"] is True

    def test_fail_when_all_short_filtered_out(self):
        assert gate_a13_sentence_density(_make_records(20, text_length=100, text="Short."))["pass"] is False

    def test_spacy_model_reported(self):
        long_text = "The court held. " * 100
        r = gate_a13_sentence_density(_make_records(5, text_length=5000, text=long_text))
        assert r.get("spacy_model") == SPACY_MODEL

    def test_empty_records_handled(self):
        assert "pass" in gate_a13_sentence_density([])

    def test_uses_all_substantive_records_when_fewer_than_subsample_n(self):
        """No internal subsampling — all 5 substantive records must be processed."""
        long_text = "The court held this point clearly. " * 100
        records = _make_records(5, text_length=5000, text=long_text)
        r = gate_a13_sentence_density(records)
        assert r["subsample_n"] == 5

    def test_accepts_nlp_argument(self):
        """gate_a13_sentence_density must accept an optional nlp argument."""
        import inspect

        sig = inspect.signature(gate_a13_sentence_density)
        assert "nlp" in sig.parameters

    def test_uses_injected_nlp_without_calling_spacy_load(self):
        """When nlp is injected, gate_a13 must not call spacy.load."""
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.sents = iter([MagicMock() for _ in range(60)])
        mock_nlp.return_value = mock_doc
        records = _make_records(5, text_length=5000)
        with patch("src.dataset_probe.spacy") as mock_spacy:
            gate_a13_sentence_density(records, nlp=mock_nlp)
            mock_spacy.load.assert_not_called()


# ---------------------------------------------------------------------------
# Gate B6 + entropy spot-check
# ---------------------------------------------------------------------------


class TestGateB6:
    def test_always_passes(self):
        assert gate_b6_text_entropy_distribution(_make_records(10))["pass"] is True

    def test_reports_percentiles(self):
        r = gate_b6_text_entropy_distribution(_make_records(20))
        for key in ("p5", "p10", "p25", "p75", "p90", "p95"):
            assert key in r

    def test_zero_entropy_count_reported(self):
        records = _make_records(8, text_entropy=4.0) + _make_records(2, text_entropy=0.0)
        assert gate_b6_text_entropy_distribution(records)["zero_entropy_count"] == 2

    def test_empty_records_handled(self):
        assert "pass" in gate_b6_text_entropy_distribution([])

    def test_b6_includes_spot_check_result(self):
        assert "spot_check" in gate_b6_text_entropy_distribution(_make_records(20))

    def test_b6_spot_check_passes_when_formula_consistent(self):
        import math

        text = "the court held that the defendant was liable"
        words = text.split()
        freq: dict[str, float] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        n = len(words)
        expected = -sum((c / n) * math.log2(c / n) for c in freq.values())
        records = _make_records(5, text=text, text_entropy=round(expected, 4))
        assert gate_b6_text_entropy_distribution(records)["spot_check"]["consistent"] is True

    def test_b6_spot_check_flags_formula_drift(self):
        records = _make_records(5, text_entropy=999.0)
        assert gate_b6_text_entropy_distribution(records)["spot_check"]["consistent"] is False

    def test_b6_spot_check_reports_max_deviation(self):
        assert "max_deviation" in gate_b6_text_entropy_distribution(_make_records(10))["spot_check"]


# ---------------------------------------------------------------------------
# Fixture JSONL
# ---------------------------------------------------------------------------


def _load_fixture_records() -> list[dict]:
    return [json.loads(line) for line in FIXTURE_JSONL.read_text().splitlines() if line.strip()]


class TestFixtureJSONL:
    def test_fixture_file_exists(self):
        assert FIXTURE_JSONL.exists()

    def test_fixture_is_valid_jsonl(self):
        assert len(_load_fixture_records()) >= 5

    def test_fixture_has_all_23_schema_fields(self):
        EXPECTED = {
            "id",
            "cluster_id",
            "docket_id",
            "court_id",
            "court_name",
            "case_name",
            "date_filed",
            "precedential_status",
            "opinion_type",
            "extracted_by_ocr",
            "raw_text",
            "text",
            "text_length",
            "text_source",
            "cleaning_flags",
            "source",
            "token_count",
            "paragraph_count",
            "citation_count",
            "text_hash",
            "citation_density",
            "is_precedential",
            "text_entropy",
        }
        for r in _load_fixture_records():
            assert not (EXPECTED - set(r.keys()))

    def test_fixture_is_deterministic(self):
        assert _load_fixture_records() == _load_fixture_records()

    def test_fixture_text_length_matches_text(self):
        for r in _load_fixture_records():
            assert abs(r["text_length"] - len(r["text"])) < 10

    def test_fixture_usable_as_shard(self, tmp_path):
        import shutil

        shard_dir = tmp_path / "fixture_shards"
        shard_dir.mkdir()
        shutil.copy(FIXTURE_JSONL, shard_dir / "courtlistener_sample.jsonl")
        audit = iter_shards_with_audit(shard_dir)
        assert len(audit["records"]) >= 5
        assert audit["total_parse_errors"] == 0


# ---------------------------------------------------------------------------
# ModelQualitySignals integration
# ---------------------------------------------------------------------------


class TestModelQualitySignalsIntegration:
    def test_quality_signals_in_report(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )
        assert "quality_signals" in report

    def test_quality_signals_has_frequency_counts(self, sample_shard_dir, tmp_path):
        qs = run_probe(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )["quality_signals"]
        assert "signal_counts" in qs
        assert "pct_clean" in qs

    def test_truncated_signal_fires(self):
        assert any(s[0] == "truncated_document" for s in ModelQualitySignals.check({"text": "Motion denied."}))


# ---------------------------------------------------------------------------
# run_probe report schema
# ---------------------------------------------------------------------------


class TestRunProbeReportSchema:
    def test_report_has_required_keys(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )
        for key in ("gates", "summary", "provenance", "quality_signals"):
            assert key in report

    def test_summary_has_all_buckets(self, sample_shard_dir, tmp_path):
        summary = run_probe(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )["summary"]
        for key in ("passed", "failed", "skipped", "all_passed"):
            assert key in summary

    def test_report_is_json_serializable(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )
        json.dumps(report)

    def test_report_written_to_disk(self, sample_shard_dir, tmp_path):
        out = tmp_path / "r.json"
        run_probe(data_dir=sample_shard_dir, subset=20, output=out, skip_tokenizer=True, skip_spacy=True)
        assert out.exists()
        assert json.loads(out.read_text())

    def test_run_probe_does_subsampling_before_gates(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )
        assert report["subset_n"] == 20


# ---------------------------------------------------------------------------
# Provenance + probe_version
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_provenance_has_required_keys(self, sample_shard_dir, tmp_path):
        prov = run_probe(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )["provenance"]
        for key in ("timestamp", "spacy_model_version", "probe_config", "probe_version"):
            assert key in prov

    def test_probe_config_in_provenance(self, sample_shard_dir, tmp_path):
        prov = run_probe(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )["provenance"]
        assert "min_text_length" in prov["probe_config"]


class TestProbeVersion:
    def test_provenance_has_probe_version(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )
        assert "probe_version" in report["provenance"]

    def test_probe_version_is_string(self, sample_shard_dir, tmp_path):
        v = run_probe(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )["provenance"]["probe_version"]
        assert isinstance(v, str) and len(v) >= 3

    def test_probe_version_constant_exported(self):
        assert isinstance(PROBE_VERSION, str) and len(PROBE_VERSION) >= 3


# ---------------------------------------------------------------------------
# CourtListenerDatasetProbe — validate_row removed, orchestrator contract
# ---------------------------------------------------------------------------


class TestCourtListenerDatasetProbe:
    def test_probe_has_no_validate_row_method(self):
        """validate_row is dead code — must not exist on CourtListenerDatasetProbe."""
        assert not hasattr(CourtListenerDatasetProbe(), "validate_row")

    def test_probe_still_has_run_method(self):
        assert hasattr(CourtListenerDatasetProbe(), "run")

    def test_probe_still_has_config(self):
        assert hasattr(CourtListenerDatasetProbe(), "config")

    def test_accepts_custom_config(self):
        probe = CourtListenerDatasetProbe(config=ProbeConfig(min_text_length=500))
        assert probe.config.min_text_length == 500

    def test_default_config_used_when_none_provided(self):
        assert CourtListenerDatasetProbe().config == ProbeConfig()

    def test_run_returns_report(self, sample_shard_dir, tmp_path):
        report = CourtListenerDatasetProbe().run(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )
        assert "gates" in report and "summary" in report

    def test_shard_audit_in_report(self, sample_shard_dir, tmp_path):
        assert "shard_audit" in CourtListenerDatasetProbe().run(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )

    def test_skipped_gates_recorded(self, sample_shard_dir, tmp_path):
        report = CourtListenerDatasetProbe().run(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )
        assert "A11" in report["summary"]["skipped"]
        assert "A13" in report["summary"]["skipped"]

    def test_probe_accepts_custom_config_with_subsample(self):
        cfg = ProbeConfig(a11_subsample_n=10)
        assert CourtListenerDatasetProbe(config=cfg).config.a11_subsample_n == 10

    def test_probe_run_accepts_log_to_wandb(self, sample_shard_dir, tmp_path):
        report = CourtListenerDatasetProbe().run(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
            log_to_wandb=False,
        )
        assert report is not None

    def test_probe_run_returns_probe_version_in_provenance(self, sample_shard_dir, tmp_path):
        report = CourtListenerDatasetProbe().run(
            data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
        )
        assert "probe_version" in report["provenance"]


# ---------------------------------------------------------------------------
# W&B logging hook
# ---------------------------------------------------------------------------


class TestWandbLoggingHook:
    def test_run_probe_accepts_log_to_wandb_false(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
            log_to_wandb=False,
        )
        assert "gates" in report

    @patch("src.dataset_probe.wandb")
    def test_run_probe_calls_wandb_when_enabled(self, mock_wandb, sample_shard_dir, tmp_path):
        mock_wandb.run = MagicMock()
        run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
            log_to_wandb=True,
        )
        assert mock_wandb.log.called

    def test_default_does_not_require_wandb(self, sample_shard_dir, tmp_path):
        assert (
            run_probe(
                data_dir=sample_shard_dir, subset=20, output=tmp_path / "r.json", skip_tokenizer=True, skip_spacy=True
            )
            is not None
        )


# ---------------------------------------------------------------------------
# --ci-mode
# ---------------------------------------------------------------------------


class TestCIMode:
    def test_ci_mode_exits_1_when_gates_fail(self, tmp_path):
        shard = tmp_path / "shards" / "s.jsonl"
        shard.parent.mkdir()
        with open(shard, "w") as fh:
            for r in _make_records(100, citation_count=0):
                fh.write(json.dumps(r) + "\n")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.dataset_probe",
                "--data-dir",
                str(shard.parent),
                "--subset",
                "100",
                "--output",
                str(tmp_path / "r.json"),
                "--skip-tokenizer",
                "--skip-spacy",
                "--ci-mode",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_ci_mode_exits_0_when_all_pass(self, sample_shard_dir, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.dataset_probe",
                "--data-dir",
                str(sample_shard_dir),
                "--subset",
                "50",
                "--output",
                str(tmp_path / "r.json"),
                "--skip-tokenizer",
                "--skip-spacy",
                "--ci-mode",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_without_ci_mode_exits_0_even_on_failure(self, tmp_path):
        shard = tmp_path / "shards" / "s.jsonl"
        shard.parent.mkdir()
        with open(shard, "w") as fh:
            for r in _make_records(100, citation_count=0):
                fh.write(json.dumps(r) + "\n")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.dataset_probe",
                "--data-dir",
                str(shard.parent),
                "--subset",
                "100",
                "--output",
                str(tmp_path / "r.json"),
                "--skip-tokenizer",
                "--skip-spacy",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_runs_and_exits_zero(self, sample_shard_dir, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.dataset_probe",
                "--data-dir",
                str(sample_shard_dir),
                "--subset",
                "20",
                "--output",
                str(tmp_path / "cli_out.json"),
                "--skip-tokenizer",
                "--skip-spacy",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_cli_writes_json_output(self, sample_shard_dir, tmp_path):
        out = tmp_path / "cli_out.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.dataset_probe",
                "--data-dir",
                str(sample_shard_dir),
                "--subset",
                "20",
                "--output",
                str(out),
                "--skip-tokenizer",
                "--skip-spacy",
            ],
            capture_output=True,
        )
        assert out.exists()
        assert "gates" in json.loads(out.read_text())


# ---------------------------------------------------------------------------
# git_sha in provenance block (observation 3/5)
# ---------------------------------------------------------------------------


class TestProvenanceGitSha:
    def test_provenance_has_git_sha(self, sample_shard_dir, tmp_path):
        """Probe report provenance must include git_sha to tie report to code version."""
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "git_sha" in report["provenance"]

    def test_git_sha_is_string(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert isinstance(report["provenance"]["git_sha"], str)

    def test_git_sha_is_not_empty(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert len(report["provenance"]["git_sha"]) > 0

    def test_git_sha_in_written_json(self, sample_shard_dir, tmp_path):
        out = tmp_path / "r.json"
        run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=out,
            skip_tokenizer=True,
            skip_spacy=True,
        )
        data = json.loads(out.read_text())
        assert "git_sha" in data["provenance"]
