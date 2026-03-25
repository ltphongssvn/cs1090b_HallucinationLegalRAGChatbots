# tests/test_dataset_probe.py
"""
TDD contract tests for src/dataset_probe.py — CourtListener local shard probe.
Covers all actionable observations from critique batch 2.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dataset_probe import (
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
    PROVISIONAL_MIN_TEXT_LENGTH,
    CHUNK_SIZE_SUBWORDS,
    CHUNK_OVERLAP_SUBWORDS,
    ENCODER_MODEL,
    SPACY_MODEL,
    MIN_SENTENCE_COUNT,
)

pytestmark = pytest.mark.unit

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
    shard = tmp_path / "shard_000.jsonl"
    with open(shard, "w") as fh:
        for i in range(50):
            r = dict(MINIMAL_RECORD)
            r["id"] = str(i)
            fh.write(json.dumps(r) + "\n")
    return tmp_path


@pytest.fixture
def default_config() -> ProbeConfig:
    return ProbeConfig()


# ---------------------------------------------------------------------------
# ProbeConfig dataclass
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
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# _percentile helper
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
# iter_shards — malformed JSON audit
# ---------------------------------------------------------------------------


class TestIterShards:
    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(iter_shards(tmp_path))

    def test_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(iter_shards(tmp_path / "nonexistent"))

    def test_yields_valid_records(self, sample_shard_dir):
        records = list(iter_shards(sample_shard_dir))
        assert len(records) == 50

    def test_skips_blank_lines(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('\n\n{"id":"1","text":"x"}\n\n')
        records = list(iter_shards(tmp_path))
        assert len(records) == 1

    def test_counts_malformed_lines(self, tmp_path):
        """iter_shards_with_audit must surface malformed line count."""
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id":"1","text":"good"}\nNOT_JSON\n{"id":"2","text":"good"}\n')
        audit = iter_shards_with_audit(tmp_path)
        assert len(audit["records"]) == 2
        assert audit["total_parse_errors"] == 1

    def test_shard_level_diagnostics_available(self, tmp_path):
        """iter_shards_with_audit must return per-shard error counts."""
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
# validate_schema — type checks
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
        records = _make_records(3, text_length="not_an_int")
        result = validate_schema(records)
        assert result["pass"] is False
        assert "type_errors" in result

    def test_fails_negative_citation_count(self):
        records = _make_records(3, citation_count=-1)
        result = validate_schema(records)
        assert result["pass"] is False
        assert "range_errors" in result

    def test_fails_non_bool_is_precedential(self):
        records = _make_records(3, is_precedential="yes")
        result = validate_schema(records)
        assert result["pass"] is False
        assert "type_errors" in result

    def test_gate_key_present(self):
        assert validate_schema(_make_records(3))["gate"] == "schema_validation"

    def test_empty_records_handled(self):
        result = validate_schema([])
        assert "pass" in result


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
        r = gate_a7_text_source_breakdown(records)
        assert r["unknown_formats_pct"] > 0

    def test_empty_records_handled(self):
        result = gate_a7_text_source_breakdown([])
        assert "pass" in result


# ---------------------------------------------------------------------------
# Gate A8 — boundary tests
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
        result = gate_a8_text_length_distribution([])
        assert "pass" in result


# ---------------------------------------------------------------------------
# Gate A9 — boundary tests
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
        result = gate_a9_citation_count_distribution([])
        assert "pass" in result


# ---------------------------------------------------------------------------
# Gate A11 — mocked tokenizer (network-free)
# Patch at transformers.AutoTokenizer since it is imported at module level.
# ---------------------------------------------------------------------------


class TestGateA11:
    @patch("src.dataset_probe.AutoTokenizer")
    def test_pass_when_median_chunks_gte_2(self, mock_cls):
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_cls.from_pretrained.return_value = mock_tok
        r = gate_a11_tokenizer_chunk_count(_make_records(10), sample_n=10)
        assert r["pass"] is True

    @patch("src.dataset_probe.AutoTokenizer")
    def test_logs_encoder_model(self, mock_cls):
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_cls.from_pretrained.return_value = mock_tok
        r = gate_a11_tokenizer_chunk_count(_make_records(5), sample_n=5)
        assert r.get("encoder_model") == ENCODER_MODEL

    @patch("src.dataset_probe.AutoTokenizer")
    def test_logs_tokenizer_revision(self, mock_cls):
        """Tokenizer revision must be logged for reproducibility."""
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_cls.from_pretrained.return_value = mock_tok
        r = gate_a11_tokenizer_chunk_count(_make_records(5), sample_n=5)
        assert "tokenizer_revision" in r

    @patch("src.dataset_probe.AutoTokenizer")
    def test_fail_on_tokenizer_load_error(self, mock_cls):
        mock_cls.from_pretrained.side_effect = OSError("not found")
        r = gate_a11_tokenizer_chunk_count(_make_records(5), sample_n=5)
        assert r["pass"] is False
        assert "error" in r

    def test_empty_records_handled(self):
        r = gate_a11_tokenizer_chunk_count([])
        assert "pass" in r


# ---------------------------------------------------------------------------
# Gate A12
# ---------------------------------------------------------------------------


class TestGateA12:
    def test_pass_when_most_have_anchors(self):
        assert gate_a12_citation_anchor_survival(_make_records(100))["pass"] is True

    def test_fail_when_few_have_anchors(self):
        r = gate_a12_citation_anchor_survival(_make_records(100, text="No citations here."))
        assert r["pass"] is False

    def test_note_states_approximation(self):
        r = gate_a12_citation_anchor_survival(_make_records(10))
        assert "heuristic" in r["note"].lower() or "approximat" in r["note"].lower()

    def test_empty_records_handled(self):
        result = gate_a12_citation_anchor_survival([])
        assert "pass" in result


# ---------------------------------------------------------------------------
# Gate A13 — short docs excluded via A8 filter
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
        records = _make_records(20, text_length=5000, text=long_text)
        r = gate_a13_sentence_density(records)
        assert r["pass"] is True

    def test_fail_when_all_short_filtered_out(self):
        records = _make_records(20, text_length=100, text="Short.")
        r = gate_a13_sentence_density(records)
        assert r["pass"] is False

    def test_spacy_model_reported(self):
        long_text = "The court held. " * 100
        r = gate_a13_sentence_density(_make_records(5, text_length=5000, text=long_text))
        assert r.get("spacy_model") == SPACY_MODEL

    def test_empty_records_handled(self):
        result = gate_a13_sentence_density([])
        assert "pass" in result


# ---------------------------------------------------------------------------
# Gate B6
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
        result = gate_b6_text_entropy_distribution([])
        assert "pass" in result


# ---------------------------------------------------------------------------
# ModelQualitySignals — integrated into run_probe
# ---------------------------------------------------------------------------


class TestModelQualitySignalsIntegration:
    def test_quality_signals_in_report(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "quality_signals" in report

    def test_quality_signals_has_frequency_counts(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        qs = report["quality_signals"]
        assert "signal_counts" in qs
        assert "pct_clean" in qs

    def test_clean_signal_text_signals(self):
        signals = ModelQualitySignals.check({"text": "Motion denied."})
        assert any(s[0] == "truncated_document" for s in signals)


# ---------------------------------------------------------------------------
# run_probe — report schema
# ---------------------------------------------------------------------------


class TestRunProbeReportSchema:
    def test_report_has_required_keys(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        for key in ("gates", "summary", "provenance", "quality_signals"):
            assert key in report

    def test_summary_has_all_buckets(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        for key in ("passed", "failed", "skipped", "all_passed"):
            assert key in report["summary"]

    def test_report_is_json_serializable(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        json.dumps(report)  # must not raise

    def test_report_written_to_disk(self, sample_shard_dir, tmp_path):
        out = tmp_path / "r.json"
        run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=out,
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert out.exists()
        assert json.loads(out.read_text())


# ---------------------------------------------------------------------------
# Provenance block
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_provenance_has_required_keys(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        prov = report["provenance"]
        for key in ("timestamp", "spacy_model_version", "probe_config"):
            assert key in prov

    def test_probe_config_in_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "min_text_length" in report["provenance"]["probe_config"]


# ---------------------------------------------------------------------------
# CourtListenerDatasetProbe — config injection
# ---------------------------------------------------------------------------


class TestCourtListenerDatasetProbe:
    def test_accepts_custom_config(self, sample_shard_dir, tmp_path):
        cfg = ProbeConfig(min_text_length=500)
        probe = CourtListenerDatasetProbe(config=cfg)
        assert probe.config.min_text_length == 500

    def test_default_config_used_when_none_provided(self):
        probe = CourtListenerDatasetProbe()
        assert probe.config == ProbeConfig()

    def test_run_returns_report(self, sample_shard_dir, tmp_path):
        probe = CourtListenerDatasetProbe()
        report = probe.run(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "gates" in report
        assert "summary" in report

    def test_shard_audit_in_report(self, sample_shard_dir, tmp_path):
        probe = CourtListenerDatasetProbe()
        report = probe.run(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "shard_audit" in report


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
