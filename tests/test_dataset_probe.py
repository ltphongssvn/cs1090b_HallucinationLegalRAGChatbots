# tests/test_dataset_probe.py
"""
Full contract tests for src/dataset_probe.py — CourtListener local shard probe.
Single authoritative test file covering all contracts.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.dataset_probe import (
    CHUNK_OVERLAP_SUBWORDS,
    CHUNK_SIZE_SUBWORDS,
    DOCUMENTED_FIELDS,
    ENCODER_MODEL,
    MIN_SENTENCE_COUNT,
    PROBE_VERSION,
    PROVISIONAL_MIN_TEXT_LENGTH,
    REQUIRED_FIELDS,
    SPACY_MODEL,
    CourtListenerDatasetProbe,
    ModelQualitySignals,
    ProbeConfig,
    _LEGAL_CITATION_RE,
    _get_text,
    _percentile,
    _probe_config_to_dict,
    _safe_int,
    _shannon_entropy,
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

# "Smith v. Jones, 123 F.3d 456 (9th Cir. 2020). " = 47 chars
# "The court held. " = 16 chars; 16 * 91 = 1456; total = 1503 > 1500 (A8 threshold)
# text_length = len(_MINIMAL_TEXT) = 1503 — self-consistent with consistency check
_MINIMAL_TEXT = "Smith v. Jones, 123 F.3d 456 (9th Cir. 2020). " + ("The court held. " * 91)

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
    "text": _MINIMAL_TEXT,
    "text_length": len(_MINIMAL_TEXT),
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
        if "text" in overrides and "text_length" not in overrides:
            r["text_length"] = len(str(overrides["text"]))
        records.append(r)
    return records


@pytest.fixture
def sample_shard_dir(tmp_path: Path) -> Path:
    """Shards live in a dedicated subdirectory so output files don't pollute it."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard = shard_dir / "shard_000.jsonl"
    with open(shard, "w") as fh:
        for i in range(50):
            r = dict(MINIMAL_RECORD)
            r["id"] = str(i)
            fh.write(json.dumps(r) + "\n")
    return shard_dir


# ---------------------------------------------------------------------------
# Obs 7 — import spacy as spacy is redundant; must be import spacy
# ---------------------------------------------------------------------------


class TestImportStyle:
    def test_spacy_not_imported_as_alias(self):
        """
        'import spacy as spacy' is a redundant self-alias — the alias is
        identical to the module name. The correct form is 'import spacy'.
        This test reads the source file and asserts the redundant alias
        pattern is absent.
        """
        source = Path("src/dataset_probe.py").read_text(encoding="utf-8")
        assert "import spacy as spacy" not in source, (
            "src/dataset_probe.py must use 'import spacy' not 'import spacy as spacy'"
        )


# ---------------------------------------------------------------------------
# Obs 1/5/9 — _safe_int helper
# ---------------------------------------------------------------------------


class TestSafeInt:
    def test_safe_int_returns_int_for_valid_int(self):
        assert _safe_int(42) == 42

    def test_safe_int_returns_int_for_valid_string(self):
        assert _safe_int("123") == 123

    def test_safe_int_returns_fallback_for_na_string(self):
        assert _safe_int("N/A") == 0

    def test_safe_int_returns_fallback_for_none(self):
        assert _safe_int(None) == 0

    def test_safe_int_returns_fallback_for_bad_string(self):
        assert _safe_int("bad") == 0

    def test_safe_int_custom_fallback(self):
        assert _safe_int("bad", fallback=99) == 99

    def test_safe_int_returns_zero_for_float(self):
        assert _safe_int(3.7) == 3


# ---------------------------------------------------------------------------
# Obs 1/5/9 — A8 robust parsing of non-numeric text_length
# ---------------------------------------------------------------------------


class TestGateA8RobustParsing:
    def test_a8_does_not_crash_on_string_text_length(self):
        records = _make_records(5, text_length="N/A")
        result = gate_a8_text_length_distribution(records)
        assert "pass" in result
        assert result["count"] == 5

    def test_a8_does_not_crash_on_none_text_length(self):
        records = _make_records(5)
        for r in records:
            r["text_length"] = None
        result = gate_a8_text_length_distribution(records)
        assert "pass" in result

    def test_a8_treats_invalid_text_length_as_zero(self):
        records = _make_records(10, text_length="N/A")
        result = gate_a8_text_length_distribution(records)
        assert result["below_provisional_count"] == 10

    def test_a8_counts_valid_records_alongside_invalid(self):
        valid = _make_records(5, text_length=5000)
        invalid = _make_records(5, text_length="bad")
        result = gate_a8_text_length_distribution(valid + invalid)
        assert result["count"] == 10
        assert "pass" in result


# ---------------------------------------------------------------------------
# Obs 1/5/9 — A9 robust parsing of non-numeric citation_count
# ---------------------------------------------------------------------------


class TestGateA9RobustParsing:
    def test_a9_does_not_crash_on_string_citation_count(self):
        records = _make_records(5, citation_count="N/A")
        result = gate_a9_citation_count_distribution(records)
        assert "pass" in result
        assert result["count"] == 5

    def test_a9_does_not_crash_on_none_citation_count(self):
        records = _make_records(5)
        for r in records:
            r["citation_count"] = None
        result = gate_a9_citation_count_distribution(records)
        assert "pass" in result

    def test_a9_treats_invalid_citation_count_as_zero(self):
        records = _make_records(10, citation_count="N/A")
        result = gate_a9_citation_count_distribution(records)
        assert result["zero_citation_count"] == 10

    def test_a9_counts_valid_records_alongside_invalid(self):
        valid = _make_records(8, citation_count=5)
        invalid = _make_records(2, citation_count="bad")
        result = gate_a9_citation_count_distribution(valid + invalid)
        assert result["count"] == 10
        assert "pass" in result


# ---------------------------------------------------------------------------
# Obs 27 — warning when log_to_wandb=True but wandb.run is None
# ---------------------------------------------------------------------------


class TestWandbRunIsNoneWarning:
    def test_warning_printed_when_log_to_wandb_true_and_run_is_none(
        self, sample_shard_dir, tmp_path, capsys
    ):
        """When log_to_wandb=True but wandb.run is None, run_probe must print a warning."""
        import src.dataset_probe as dp

        mock_wandb = MagicMock()
        mock_wandb.run = None
        original_wandb = dp.wandb
        try:
            dp.wandb = mock_wandb
            run_probe(
                data_dir=sample_shard_dir,
                subset=20,
                output=tmp_path / "r.json",
                skip_tokenizer=True,
                skip_spacy=True,
                log_to_wandb=True,
            )
            captured = capsys.readouterr()
            assert "wandb" in captured.out.lower() or "log" in captured.out.lower()
        finally:
            dp.wandb = original_wandb

    def test_no_warning_when_log_to_wandb_false(
        self, sample_shard_dir, tmp_path, capsys
    ):
        """When log_to_wandb=False, no W&B-related warning must be printed."""
        import src.dataset_probe as dp

        mock_wandb = MagicMock()
        mock_wandb.run = None
        original_wandb = dp.wandb
        try:
            dp.wandb = mock_wandb
            run_probe(
                data_dir=sample_shard_dir,
                subset=20,
                output=tmp_path / "r.json",
                skip_tokenizer=True,
                skip_spacy=True,
                log_to_wandb=False,
            )
            captured = capsys.readouterr()
            assert "wandb.run" not in captured.out.lower()
        finally:
            dp.wandb = original_wandb

    def test_no_warning_when_wandb_run_is_active(
        self, sample_shard_dir, tmp_path, capsys
    ):
        """When wandb.run is not None, no 'run is None' warning should appear."""
        import src.dataset_probe as dp

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        original_wandb = dp.wandb
        try:
            dp.wandb = mock_wandb
            run_probe(
                data_dir=sample_shard_dir,
                subset=20,
                output=tmp_path / "r.json",
                skip_tokenizer=True,
                skip_spacy=True,
                log_to_wandb=True,
            )
            captured = capsys.readouterr()
            assert "no active wandb run" not in captured.out.lower()
        finally:
            dp.wandb = original_wandb


# ---------------------------------------------------------------------------
# Obs 28 — --skip-generative-tokenizer CLI flag
# ---------------------------------------------------------------------------


class TestSkipGenerativeTokenizerCLI:
    def test_cli_accepts_skip_generative_tokenizer_flag(
        self, sample_shard_dir, tmp_path
    ):
        result = subprocess.run(
            [
                sys.executable, "-m", "src.dataset_probe",
                "--data-dir", str(sample_shard_dir),
                "--subset", "20",
                "--output", str(tmp_path / "r.json"),
                "--skip-tokenizer", "--skip-spacy",
                "--skip-generative-tokenizer",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_skip_generative_tokenizer_sets_model_to_empty_string(
        self, sample_shard_dir, tmp_path
    ):
        out = tmp_path / "r.json"
        subprocess.run(
            [
                sys.executable, "-m", "src.dataset_probe",
                "--data-dir", str(sample_shard_dir),
                "--subset", "20",
                "--output", str(out),
                "--skip-tokenizer", "--skip-spacy",
                "--skip-generative-tokenizer",
            ],
            capture_output=True, text=True,
        )
        assert out.exists()
        report = json.loads(out.read_text())
        assert report["provenance"]["probe_config"]["a11_generative_model"] == ""

    def test_without_flag_generative_model_is_mistral(
        self, sample_shard_dir, tmp_path
    ):
        out = tmp_path / "r.json"
        subprocess.run(
            [
                sys.executable, "-m", "src.dataset_probe",
                "--data-dir", str(sample_shard_dir),
                "--subset", "20",
                "--output", str(out),
                "--skip-tokenizer", "--skip-spacy",
            ],
            capture_output=True, text=True,
        )
        report = json.loads(out.read_text())
        assert "mistral" in report["provenance"]["probe_config"]["a11_generative_model"].lower()


# ---------------------------------------------------------------------------
# Obs 9 — text_length vs len(text) consistency check in validate_schema
# ---------------------------------------------------------------------------


class TestValidateSchemaTextLengthConsistency:
    def test_probe_config_has_text_length_tolerance(self):
        assert hasattr(ProbeConfig(), "text_length_consistency_tolerance")

    def test_text_length_consistency_tolerance_default(self):
        tol = ProbeConfig().text_length_consistency_tolerance
        assert isinstance(tol, int)
        assert tol > 0

    def test_text_length_consistency_tolerance_custom_accepted(self):
        cfg = ProbeConfig(text_length_consistency_tolerance=500)
        assert cfg.text_length_consistency_tolerance == 500

    def test_fails_when_text_length_far_exceeds_actual_text(self):
        records = _make_records(3)
        for r in records:
            r["text"] = "Short."
            r["text_length"] = 99_999
        result = validate_schema(records)
        assert result["pass"] is False
        assert "text_length_consistency" in result.get("consistency_errors", {})

    def test_fails_when_actual_text_far_exceeds_text_length_field(self):
        records = _make_records(3)
        for r in records:
            r["text"] = "word " * 10_000
            r["text_length"] = 1
        result = validate_schema(records)
        assert result["pass"] is False
        assert "text_length_consistency" in result.get("consistency_errors", {})

    def test_passes_when_text_length_within_tolerance(self):
        text = "The court held. " * 100
        records = _make_records(5)
        for r in records:
            r["text"] = text
            r["text_length"] = len(text)
        result = validate_schema(records)
        assert result.get("consistency_errors", {}).get("text_length_consistency") is None

    def test_passes_when_text_length_slightly_off(self):
        text = "The court held. " * 100
        cfg = ProbeConfig(text_length_consistency_tolerance=200)
        records = _make_records(5)
        for r in records:
            r["text"] = text
            r["text_length"] = len(text) + 50
        result = validate_schema(records, config=cfg)
        assert result.get("consistency_errors", {}).get("text_length_consistency") is None

    def test_consistency_errors_counted_per_record(self):
        records = _make_records(4)
        for r in records:
            r["text"] = "x"
            r["text_length"] = 999_999
        result = validate_schema(records)
        assert result["consistency_errors"]["text_length_consistency"] == 4

    def test_validate_schema_accepts_config_parameter(self):
        sig = inspect.signature(validate_schema)
        assert "config" in sig.parameters

    def test_text_length_consistency_tolerance_in_provenance(
        self, sample_shard_dir, tmp_path
    ):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "text_length_consistency_tolerance" in report["provenance"]["probe_config"]


# ---------------------------------------------------------------------------
# Obs 22 — W&B logging branch tested with mock
# ---------------------------------------------------------------------------


class TestWandbLoggingBranch:
    def test_wandb_log_called_when_log_to_wandb_true_and_run_active(
        self, sample_shard_dir, tmp_path
    ):
        import src.dataset_probe as dp

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        original_wandb = dp.wandb
        try:
            dp.wandb = mock_wandb
            run_probe(
                data_dir=sample_shard_dir,
                subset=20,
                output=tmp_path / "r.json",
                skip_tokenizer=True,
                skip_spacy=True,
                log_to_wandb=True,
            )
            mock_wandb.log.assert_called()
        finally:
            dp.wandb = original_wandb

    def test_wandb_log_not_called_when_log_to_wandb_false(
        self, sample_shard_dir, tmp_path
    ):
        import src.dataset_probe as dp

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        original_wandb = dp.wandb
        try:
            dp.wandb = mock_wandb
            run_probe(
                data_dir=sample_shard_dir,
                subset=20,
                output=tmp_path / "r.json",
                skip_tokenizer=True,
                skip_spacy=True,
                log_to_wandb=False,
            )
            mock_wandb.log.assert_not_called()
        finally:
            dp.wandb = original_wandb

    def test_wandb_log_not_called_when_wandb_run_is_none(
        self, sample_shard_dir, tmp_path
    ):
        import src.dataset_probe as dp

        mock_wandb = MagicMock()
        mock_wandb.run = None
        original_wandb = dp.wandb
        try:
            dp.wandb = mock_wandb
            run_probe(
                data_dir=sample_shard_dir,
                subset=20,
                output=tmp_path / "r.json",
                skip_tokenizer=True,
                skip_spacy=True,
                log_to_wandb=True,
            )
            mock_wandb.log.assert_not_called()
        finally:
            dp.wandb = original_wandb

    def test_wandb_log_not_called_when_wandb_is_none(
        self, sample_shard_dir, tmp_path
    ):
        import src.dataset_probe as dp

        original_wandb = dp.wandb
        try:
            dp.wandb = None
            report = run_probe(
                data_dir=sample_shard_dir,
                subset=20,
                output=tmp_path / "r.json",
                skip_tokenizer=True,
                skip_spacy=True,
                log_to_wandb=True,
            )
            assert "summary" in report
        finally:
            dp.wandb = original_wandb

    def test_wandb_log_receives_all_passed_key(self, sample_shard_dir, tmp_path):
        import src.dataset_probe as dp

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        original_wandb = dp.wandb
        try:
            dp.wandb = mock_wandb
            run_probe(
                data_dir=sample_shard_dir,
                subset=20,
                output=tmp_path / "r.json",
                skip_tokenizer=True,
                skip_spacy=True,
                log_to_wandb=True,
            )
            all_logged_keys: set[str] = set()
            for call in mock_wandb.log.call_args_list:
                all_logged_keys.update(call.args[0].keys())
            assert "probe/all_passed" in all_logged_keys
        finally:
            dp.wandb = original_wandb


# ---------------------------------------------------------------------------
# Obs 1 — A12 text cap and ProbeConfig.a12_text_cap_chars
# ---------------------------------------------------------------------------


class TestA12TextCap:
    def test_probe_config_has_a12_text_cap_chars(self):
        assert hasattr(ProbeConfig(), "a12_text_cap_chars")

    def test_a12_text_cap_chars_default(self):
        assert ProbeConfig().a12_text_cap_chars == 50_000

    def test_a12_text_cap_chars_custom_accepted(self):
        cfg = ProbeConfig(a12_text_cap_chars=10_000)
        assert cfg.a12_text_cap_chars == 10_000

    def test_a12_text_cap_chars_in_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "a12_text_cap_chars" in report["provenance"]["probe_config"]

    def test_a12_uses_cap_not_full_text(self):
        cap = 100
        prefix = "x" * cap
        suffix = " Smith v. Jones, 123 F.3d 456"
        long_text = prefix + suffix
        cfg = ProbeConfig(a12_text_cap_chars=cap)
        records = [
            {**MINIMAL_RECORD, "id": str(i), "text": long_text,
             "text_length": len(long_text), "citation_count": 0}
            for i in range(5)
        ]
        r = gate_a12_citation_anchor_survival(records, config=cfg)
        assert r["records_with_citation_anchor"] == 0

    def test_a12_subsample_n_reported(self):
        r = gate_a12_citation_anchor_survival(_make_records(7))
        assert "subsample_n" in r
        assert r["subsample_n"] == 7


# ---------------------------------------------------------------------------
# Obs 1 — ModelQualitySignals text cap
# ---------------------------------------------------------------------------


class TestModelQualitySignalsTextCap:
    def test_probe_config_has_quality_signals_text_cap_chars(self):
        assert hasattr(ProbeConfig(), "quality_signals_text_cap_chars")

    def test_quality_signals_text_cap_chars_default(self):
        assert ProbeConfig().quality_signals_text_cap_chars == 50_000

    def test_quality_signals_text_cap_chars_custom_accepted(self):
        cfg = ProbeConfig(quality_signals_text_cap_chars=10_000)
        assert cfg.quality_signals_text_cap_chars == 10_000

    def test_model_quality_signals_check_accepts_config(self):
        sig = inspect.signature(ModelQualitySignals.check)
        assert "config" in sig.parameters

    def test_model_quality_signals_caps_text_for_regex(self):
        suffix = " Smith v. Jones, 123 F.3d 456"
        big_text = ("word " * 110) + suffix
        cfg_small = ProbeConfig(quality_signals_text_cap_chars=50)
        signals = ModelQualitySignals.check({"text": big_text}, config=cfg_small)
        signal_names = [s[0] for s in signals]
        assert "no_citations" in signal_names


# ---------------------------------------------------------------------------
# Obs 8 — iter_shards and sample_records docstrings mention silent parse drop
# ---------------------------------------------------------------------------


class TestIterShardsAndSampleRecordsDocstring:
    def test_iter_shards_docstring_mentions_silent_parse_error_drop(self):
        doc = (iter_shards.__doc__ or "").lower()
        assert "silent" in doc or "no count" in doc or "not count" in doc or "without count" in doc

    def test_iter_shards_docstring_directs_to_audit(self):
        doc = (iter_shards.__doc__ or "").lower()
        assert "iter_shards_with_audit" in doc or "audit" in doc

    def test_sample_records_docstring_mentions_silent_parse_error_drop(self):
        doc = (sample_records.__doc__ or "").lower()
        assert (
            "silent" in doc
            or "no count" in doc
            or "not count" in doc
            or "without count" in doc
            or "parse" in doc
        )

    def test_sample_records_docstring_directs_to_audit(self):
        doc = (sample_records.__doc__ or "").lower()
        assert "iter_shards_with_audit" in doc or "audit" in doc


# ---------------------------------------------------------------------------
# Obs 3 — frozenset[str] type hint for a7_known_formats
# ---------------------------------------------------------------------------


class TestProbeConfigA7KnownFormatsTyping:
    def test_a7_known_formats_type_hint_is_frozenset_str(self):
        hints = get_type_hints(ProbeConfig)
        a7_hint = hints.get("a7_known_formats")
        assert hasattr(a7_hint, "__args__"), f"a7_known_formats must be frozenset[str], got: {a7_hint}"
        assert a7_hint.__args__ == (str,)

    def test_a7_known_formats_default_values_are_strings(self):
        for item in ProbeConfig().a7_known_formats:
            assert isinstance(item, str)

    def test_a7_known_formats_custom_frozenset_str_accepted(self):
        cfg = ProbeConfig(a7_known_formats=frozenset({"plain_text", "xml_harvard"}))
        for item in cfg.a7_known_formats:
            assert isinstance(item, str)


# ---------------------------------------------------------------------------
# Obs 7 — _percentile docstring must not claim numpy consistency
# ---------------------------------------------------------------------------


class TestPercentileDocstring:
    def test_percentile_has_docstring(self):
        assert _percentile.__doc__ is not None
        assert len(_percentile.__doc__.strip()) > 0

    def test_percentile_docstring_does_not_claim_numpy_consistency(self):
        doc = _percentile.__doc__.lower()
        assert "numpy" not in doc

    def test_percentile_docstring_describes_ceiling_index(self):
        doc = _percentile.__doc__.lower()
        assert "ceiling" in doc or "ceil" in doc

    def test_percentile_docstring_specifies_probe_convention(self):
        doc = _percentile.__doc__.lower()
        assert "probe" in doc or "convention" in doc or "empirical" in doc or "defined" in doc


# ---------------------------------------------------------------------------
# Obs 11 — citation_density >= 0 range check in validate_schema
# ---------------------------------------------------------------------------


class TestValidateSchemaCitationDensity:
    def test_fails_negative_citation_density(self):
        result = validate_schema(_make_records(3, citation_density=-0.5))
        assert result["pass"] is False
        assert "citation_density" in result.get("range_errors", {})

    def test_fails_non_numeric_citation_density(self):
        result = validate_schema(_make_records(3, citation_density="high"))
        assert result["pass"] is False
        assert "citation_density" in result.get("type_errors", {})

    def test_passes_zero_citation_density(self):
        result = validate_schema(_make_records(5, citation_density=0.0))
        assert result.get("range_errors", {}).get("citation_density") is None

    def test_passes_valid_citation_density(self):
        result = validate_schema(_make_records(5, citation_density=0.05))
        assert result.get("type_errors", {}).get("citation_density") is None
        assert result.get("range_errors", {}).get("citation_density") is None

    def test_citation_density_none_is_skipped(self):
        records = _make_records(3)
        for r in records:
            r["citation_density"] = None
        result = validate_schema(records)
        assert result.get("type_errors", {}).get("citation_density") is None

    def test_citation_density_error_counted_per_record(self):
        result = validate_schema(_make_records(4, citation_density=-1.0))
        assert result["range_errors"]["citation_density"] == 4


# ---------------------------------------------------------------------------
# Wandb optional import guard
# ---------------------------------------------------------------------------


class TestWandbOptionalImport:
    def test_probe_runs_without_log_to_wandb_flag(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
            log_to_wandb=False,
        )
        assert report["summary"]["all_passed"] is True

    def test_wandb_none_guard_in_run_probe(self, sample_shard_dir, tmp_path):
        import src.dataset_probe as dp
        original_wandb = dp.wandb
        try:
            dp.wandb = None
            report = run_probe(
                data_dir=sample_shard_dir,
                subset=20,
                output=tmp_path / "r.json",
                skip_tokenizer=True,
                skip_spacy=True,
                log_to_wandb=True,
            )
            assert "summary" in report
        finally:
            dp.wandb = original_wandb

    def test_wandb_import_error_at_module_level_is_caught(self):
        import src.dataset_probe as dp
        assert dp.wandb is None or hasattr(dp.wandb, "__version__") or hasattr(dp.wandb, "init")


# ---------------------------------------------------------------------------
# Shannon entropy contract
# ---------------------------------------------------------------------------


class TestShannonEntropyContract:
    def test_shannon_entropy_has_docstring(self):
        assert _shannon_entropy.__doc__ is not None

    def test_shannon_entropy_docstring_specifies_whitespace_tokenization(self):
        doc = _shannon_entropy.__doc__.lower()
        assert "whitespace" in doc or "split" in doc

    def test_shannon_entropy_docstring_specifies_case_preserved(self):
        doc = _shannon_entropy.__doc__.lower()
        assert "case" in doc or "preserved" in doc

    def test_shannon_entropy_docstring_specifies_not_normalized(self):
        doc = _shannon_entropy.__doc__.lower()
        assert "normaliz" in doc or "log2" in doc or "bits" in doc

    def test_shannon_entropy_docstring_specifies_no_stopword_removal(self):
        doc = _shannon_entropy.__doc__.lower()
        assert (
            "stopword" in doc
            or "stop word" in doc
            or "all words" in doc
            or "no filter" in doc
            or "punctuation" in doc
        )

    def test_shannon_entropy_whitespace_split_behavior(self):
        import math
        text = "a a b"
        result = _shannon_entropy(text)
        expected = -(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3)
        assert abs(result - expected) < 1e-9

    def test_shannon_entropy_case_preserved(self):
        assert _shannon_entropy("Word word Word") > _shannon_entropy("word word word")

    def test_shannon_entropy_empty_string_returns_zero(self):
        assert _shannon_entropy("") == 0.0

    def test_shannon_entropy_single_word_returns_zero(self):
        assert _shannon_entropy("court") == 0.0


# ---------------------------------------------------------------------------
# MIN_REQUIRED_FIELDS rename + backward-compat alias
# ---------------------------------------------------------------------------


class TestMinRequiredFields:
    def test_min_required_fields_exported(self):
        from src.dataset_probe import MIN_REQUIRED_FIELDS
        assert isinstance(MIN_REQUIRED_FIELDS, frozenset)

    def test_min_required_fields_has_11_fields(self):
        from src.dataset_probe import MIN_REQUIRED_FIELDS
        assert len(MIN_REQUIRED_FIELDS) == 11

    def test_required_fields_alias_still_works(self):
        from src.dataset_probe import MIN_REQUIRED_FIELDS, REQUIRED_FIELDS
        assert REQUIRED_FIELDS == MIN_REQUIRED_FIELDS

    def test_min_required_fields_is_subset_of_documented_fields(self):
        from src.dataset_probe import MIN_REQUIRED_FIELDS
        assert MIN_REQUIRED_FIELDS.issubset(DOCUMENTED_FIELDS)


# ---------------------------------------------------------------------------
# One-pass streaming in run_probe
# ---------------------------------------------------------------------------


class TestShardAuditOnePassStreaming:
    def test_run_probe_report_has_total_records_decoded(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        assert "total_records_decoded" in report["shard_audit"]

    def test_subset_n_never_exceeds_total_records_decoded(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        assert report["subset_n"] <= report["shard_audit"]["total_records_decoded"]

    def test_large_subset_returns_all_records(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=10_000,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        assert report["subset_n"] == 50


# ---------------------------------------------------------------------------
# No stale sample_n / seed params in A11, A12, A13
# ---------------------------------------------------------------------------


class TestGateSignaturesNoStaleParams:
    def test_gate_a11_has_no_sample_n(self):
        assert "sample_n" not in inspect.signature(gate_a11_tokenizer_chunk_count).parameters

    def test_gate_a11_has_no_seed(self):
        assert "seed" not in inspect.signature(gate_a11_tokenizer_chunk_count).parameters

    def test_gate_a12_has_no_sample_n(self):
        assert "sample_n" not in inspect.signature(gate_a12_citation_anchor_survival).parameters

    def test_gate_a12_has_no_seed(self):
        assert "seed" not in inspect.signature(gate_a12_citation_anchor_survival).parameters

    def test_gate_a13_has_no_sample_n(self):
        assert "sample_n" not in inspect.signature(gate_a13_sentence_density).parameters

    def test_gate_a13_has_no_seed(self):
        assert "seed" not in inspect.signature(gate_a13_sentence_density).parameters


# ---------------------------------------------------------------------------
# Gate severity field
# ---------------------------------------------------------------------------


class TestGateSeverity:
    def test_gate_a7_has_severity_blocking(self):
        assert gate_a7_text_source_breakdown(_make_records(10)).get("severity") == "blocking"

    def test_gate_a8_has_severity_blocking(self):
        assert gate_a8_text_length_distribution(_make_records(10)).get("severity") == "blocking"

    def test_gate_a9_has_severity_advisory(self):
        assert gate_a9_citation_count_distribution(_make_records(10)).get("severity") == "advisory"

    def test_gate_a12_has_severity_blocking(self):
        assert gate_a12_citation_anchor_survival(_make_records(10)).get("severity") == "blocking"

    def test_gate_b6_has_severity_advisory(self):
        assert gate_b6_text_entropy_distribution(_make_records(10)).get("severity") == "advisory"

    def test_run_probe_summary_has_failed_blocking(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        assert "failed_blocking" in report["summary"]

    def test_run_probe_summary_has_failed_advisory(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        assert "failed_advisory" in report["summary"]

    def test_all_passed_uses_only_blocking_failures(self, tmp_path):
        shard = tmp_path / "shards" / "s.jsonl"
        shard.parent.mkdir()
        with open(shard, "w") as fh:
            for r in _make_records(100, citation_count=0):
                fh.write(json.dumps(r) + "\n")
        report = run_probe(
            data_dir=shard.parent, subset=100,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        assert "A9" in report["summary"]["failed_advisory"]
        assert report["summary"]["all_passed"] is True

    def test_ci_mode_exits_0_when_only_advisory_fails(self, tmp_path):
        shard = tmp_path / "shards" / "s.jsonl"
        shard.parent.mkdir()
        with open(shard, "w") as fh:
            for r in _make_records(100, citation_count=0):
                fh.write(json.dumps(r) + "\n")
        result = subprocess.run(
            [sys.executable, "-m", "src.dataset_probe",
             "--data-dir", str(shard.parent), "--subset", "100",
             "--output", str(tmp_path / "r.json"),
             "--skip-tokenizer", "--skip-spacy", "--ci-mode"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Extended type/range checks in validate_schema
# ---------------------------------------------------------------------------


class TestValidateSchemaExtendedTypes:
    def test_fails_non_numeric_text_entropy(self):
        result = validate_schema(_make_records(3, text_entropy="not_a_float"))
        assert result["pass"] is False
        assert "text_entropy" in result.get("type_errors", {})

    def test_fails_negative_text_entropy(self):
        result = validate_schema(_make_records(3, text_entropy=-1.0))
        assert result["pass"] is False
        assert "text_entropy" in result.get("range_errors", {})

    def test_fails_non_integer_paragraph_count(self):
        result = validate_schema(_make_records(3, paragraph_count="five"))
        assert result["pass"] is False
        assert "paragraph_count" in result.get("type_errors", {})

    def test_fails_negative_paragraph_count(self):
        result = validate_schema(_make_records(3, paragraph_count=-1))
        assert result["pass"] is False
        assert "paragraph_count" in result.get("range_errors", {})

    def test_fails_non_integer_token_count(self):
        result = validate_schema(_make_records(3, token_count="lots"))
        assert result["pass"] is False
        assert "token_count" in result.get("type_errors", {})

    def test_fails_negative_token_count(self):
        result = validate_schema(_make_records(3, token_count=-1))
        assert result["pass"] is False
        assert "token_count" in result.get("range_errors", {})


# ---------------------------------------------------------------------------
# iter_shards docstring accuracy
# ---------------------------------------------------------------------------


class TestIterShardsDocstring:
    def test_iter_shards_docstring_mentions_parse_errors(self):
        doc = iter_shards.__doc__ or ""
        assert "parse" in doc.lower() or "json" in doc.lower() or "error" in doc.lower()


# ---------------------------------------------------------------------------
# Single spaCy load
# ---------------------------------------------------------------------------


class TestSpaCySingleLoad:
    def test_spacy_loaded_once_in_run_probe(self, sample_shard_dir, tmp_path):
        with patch("src.dataset_probe.spacy") as mock_spacy:
            mock_nlp = MagicMock()
            mock_nlp.pipe_names = ["sentencizer"]
            mock_nlp.meta = {"version": "3.8.0"}
            mock_spacy.__version__ = "3.8.11"
            mock_spacy.load.return_value = mock_nlp
            run_probe(
                data_dir=sample_shard_dir, subset=20,
                output=tmp_path / "r.json",
                skip_tokenizer=True, skip_spacy=False,
            )
            assert mock_spacy.load.call_count <= 1


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
        d = _probe_config_to_dict(ProbeConfig())
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

    def test_all_new_fields_in_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        config = report["provenance"]["probe_config"]
        for key in ("a13_text_cap_chars", "a11_subsample_n", "a12_subsample_n", "a13_subsample_n"):
            assert key in config


class TestProbeConfigGenerativeModel:
    def test_has_a11_generative_model(self):
        assert hasattr(ProbeConfig(), "a11_generative_model")

    def test_a11_generative_model_default_is_mistral(self):
        assert "mistral" in ProbeConfig().a11_generative_model.lower()


class TestProbeConfigA7KnownFormats:
    def test_has_a7_known_formats(self):
        assert hasattr(ProbeConfig(), "a7_known_formats")

    def test_a7_known_formats_default_contains_plain_text(self):
        assert "plain_text" in ProbeConfig().a7_known_formats

    def test_a7_known_formats_is_frozenset(self):
        assert isinstance(ProbeConfig().a7_known_formats, frozenset)

    def test_a7_known_formats_custom_accepted(self):
        cfg = ProbeConfig(a7_known_formats=frozenset({"plain_text", "xml_harvard"}))
        assert "xml_harvard" in cfg.a7_known_formats

    def test_a7_uses_config_known_formats_for_pass(self):
        records = _make_records(100, text_source="xml_harvard")
        r_default = gate_a7_text_source_breakdown(records, config=ProbeConfig())
        cfg_custom = ProbeConfig(
            a7_known_formats=frozenset({"plain_text", "html_with_citations", "xml_harvard"})
        )
        r_custom = gate_a7_text_source_breakdown(records, config=cfg_custom)
        assert r_custom["known_formats_pct"] > r_default["known_formats_pct"]

    def test_a7_known_formats_in_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        assert "a7_known_formats" in report["provenance"]["probe_config"]


# ---------------------------------------------------------------------------
# DOCUMENTED_FIELDS (23 fields)
# ---------------------------------------------------------------------------


class TestValidateSchemaDocumentedFields:
    def test_documented_fields_exported(self):
        assert isinstance(DOCUMENTED_FIELDS, frozenset)

    def test_documented_fields_contains_all_23(self):
        expected = {
            "id", "cluster_id", "docket_id", "court_id", "court_name",
            "case_name", "date_filed", "precedential_status", "opinion_type",
            "extracted_by_ocr", "raw_text", "text", "text_length", "text_source",
            "cleaning_flags", "source", "token_count", "paragraph_count",
            "citation_count", "text_hash", "citation_density", "is_precedential",
            "text_entropy",
        }
        assert DOCUMENTED_FIELDS == expected

    def test_validate_schema_includes_missing_documented_fields_key(self):
        assert "missing_documented_fields" in validate_schema(_make_records(5))

    def test_missing_documented_fields_zero_for_complete_records(self):
        assert validate_schema(_make_records(5))["missing_documented_fields"] == {}

    def test_schema_pass_unaffected_by_documented_only_fields(self):
        records = _make_records(5)
        for r in records:
            del r["cluster_id"]
        result = validate_schema(records)
        assert result["pass"] is True
        assert "cluster_id" in result["missing_documented_fields"]


# ---------------------------------------------------------------------------
# No side effects on corpus shards
# ---------------------------------------------------------------------------


class TestNoSideEffectsOnShards:
    def test_run_probe_does_not_modify_shard_files(self, sample_shard_dir, tmp_path):
        shard_files = sorted(sample_shard_dir.glob("*.jsonl"))
        hashes_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shard_files}
        run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        hashes_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shard_files}
        assert hashes_before == hashes_after

    def test_run_probe_does_not_create_files_in_shard_dir(self, sample_shard_dir, tmp_path):
        files_before = set(sample_shard_dir.glob("*"))
        run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        assert files_before == set(sample_shard_dir.glob("*"))


# ---------------------------------------------------------------------------
# A12 citation field cross-validation
# ---------------------------------------------------------------------------


class TestGateA12CitationFieldCrossValidation:
    def test_a12_reports_citation_field_vs_regex(self):
        assert "citation_field_vs_regex" in gate_a12_citation_anchor_survival(_make_records(10))

    def test_a12_citation_field_vs_regex_has_required_keys(self):
        cvr = gate_a12_citation_anchor_survival(_make_records(10))["citation_field_vs_regex"]
        assert "field_nonzero_regex_zero_count" in cvr
        assert "field_nonzero_regex_zero_pct" in cvr

    def test_a12_detects_field_nonzero_but_regex_zero(self):
        records = _make_records(10, citation_count=5, text="No legal anchors here at all.")
        assert gate_a12_citation_anchor_survival(records)["citation_field_vs_regex"]["field_nonzero_regex_zero_count"] == 10

    def test_a12_no_discrepancy_when_regex_finds_citations(self):
        assert gate_a12_citation_anchor_survival(_make_records(10, citation_count=3))["citation_field_vs_regex"]["field_nonzero_regex_zero_count"] == 0


# ---------------------------------------------------------------------------
# _get_text helper
# ---------------------------------------------------------------------------


class TestGetTextHelper:
    def test_get_text_exported(self):
        assert callable(_get_text)

    def test_get_text_returns_text_field(self):
        assert _get_text({"text": "hello"}) == "hello"

    def test_get_text_returns_empty_string_when_missing(self):
        assert _get_text({}) == ""

    def test_get_text_returns_empty_string_when_none(self):
        assert _get_text({"text": None}) == ""

    def test_get_text_returns_string_when_non_string_value(self):
        assert isinstance(_get_text({"text": 42}), str)


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

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            _percentile([], 50)


class TestPercentileEdgeCases:
    def test_repeated_values(self):
        assert _percentile([5, 5, 5, 5, 5], 50) == 5

    def test_n_equals_2_p0(self):
        assert _percentile([1, 2], 0) == 1

    def test_n_equals_2_p100(self):
        assert _percentile([1, 2], 100) == 2

    def test_large_n_boundary(self):
        data = sorted(range(1000))
        assert _percentile(data, 0) == 0
        assert _percentile(data, 100) == 999


# ---------------------------------------------------------------------------
# Shared citation regex
# ---------------------------------------------------------------------------


class TestSharedCitationRegex:
    def test_legal_citation_re_exported(self):
        assert _LEGAL_CITATION_RE is not None

    def test_shared_regex_catches_federal_reporter(self):
        assert _LEGAL_CITATION_RE.search("123 F.3d 456")

    def test_shared_regex_catches_case_name_citation(self):
        assert _LEGAL_CITATION_RE.search("Smith v. Jones")

    def test_shared_regex_catches_scotus_reporter(self):
        assert _LEGAL_CITATION_RE.search("347 U.S. 483")


# ---------------------------------------------------------------------------
# iter_shards
# ---------------------------------------------------------------------------


class TestIterShards:
    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(iter_shards(tmp_path))

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
        assert validate_schema(_make_records(3, text_length="not_an_int"))["pass"] is False

    def test_fails_negative_citation_count(self):
        result = validate_schema(_make_records(3, citation_count=-1))
        assert result["pass"] is False
        assert "range_errors" in result

    def test_fails_non_bool_is_precedential(self):
        assert validate_schema(_make_records(3, is_precedential="yes"))["pass"] is False

    def test_gate_key_present(self):
        assert validate_schema(_make_records(3))["gate"] == "schema_validation"

    def test_empty_records_handled(self):
        assert "pass" in validate_schema([])


class TestValidateSchemaTextSource:
    def test_passes_known_text_sources(self):
        for src in ("plain_text", "html_with_citations", "html_lawbox", "html_columbia", "xml_harvard"):
            result = validate_schema(_make_records(3, text_source=src))
            assert result.get("vocabulary_errors", {}).get("text_source", 0) == 0

    def test_flags_unknown_text_source(self):
        result = validate_schema(_make_records(3, text_source="UNKNOWN_FORMAT_XYZ"))
        assert "text_source" in result["vocabulary_errors"]

    def test_schema_fails_with_unknown_source(self):
        assert validate_schema(_make_records(5, text_source="garbage_format"))["pass"] is False


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
# Gate A9
# ---------------------------------------------------------------------------


class TestGateA9:
    def test_passes_at_19_99_pct(self):
        records = _make_records(8001, citation_count=5) + _make_records(1999, citation_count=0)
        assert gate_a9_citation_count_distribution(records)["pass"] is True

    def test_fails_at_20_pct(self):
        records = _make_records(8000, citation_count=5) + _make_records(2000, citation_count=0)
        assert gate_a9_citation_count_distribution(records)["pass"] is False

    def test_empty_records_handled(self):
        assert "pass" in gate_a9_citation_count_distribution([])

    def test_note_clarifies_advisory_role(self):
        assert "advisory" in gate_a9_citation_count_distribution(_make_records(10))["note"].lower()


# ---------------------------------------------------------------------------
# Gate A11
# ---------------------------------------------------------------------------


class TestGateA11:
    @patch("src.dataset_probe.AutoTokenizer")
    def test_pass_when_median_chunks_gte_2(self, mock_cls):
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_cls.from_pretrained.return_value = mock_tok
        assert gate_a11_tokenizer_chunk_count(
            _make_records(10), config=ProbeConfig(a11_generative_model="")
        )["pass"] is True

    @patch("src.dataset_probe.AutoTokenizer")
    def test_fail_on_tokenizer_load_error(self, mock_cls):
        mock_cls.from_pretrained.side_effect = OSError("not found")
        r = gate_a11_tokenizer_chunk_count(_make_records(5), config=ProbeConfig(a11_generative_model=""))
        assert r["pass"] is False
        assert "error" in r

    def test_empty_records_handled(self):
        assert "pass" in gate_a11_tokenizer_chunk_count([])

    def test_gate_a11_accepts_tokenizer_argument(self):
        assert "tokenizer" in inspect.signature(gate_a11_tokenizer_chunk_count).parameters

    def test_gate_a11_uses_injected_tokenizer_without_calling_autotokenizer(self):
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        with patch("src.dataset_probe.AutoTokenizer") as mock_cls:
            gate_a11_tokenizer_chunk_count(
                _make_records(5), tokenizer=mock_tok, config=ProbeConfig(a11_generative_model="")
            )
            mock_cls.from_pretrained.assert_not_called()


# ---------------------------------------------------------------------------
# Gate A12
# ---------------------------------------------------------------------------


class TestGateA12:
    def test_pass_when_most_have_anchors(self):
        assert gate_a12_citation_anchor_survival(_make_records(100))["pass"] is True

    def test_fail_when_few_have_anchors(self):
        assert gate_a12_citation_anchor_survival(_make_records(100, text="No citations here."))["pass"] is False

    def test_empty_records_handled(self):
        assert "pass" in gate_a12_citation_anchor_survival([])


# ---------------------------------------------------------------------------
# Gate A13
# ---------------------------------------------------------------------------


class TestGateA13:
    def test_excludes_short_docs_below_a8_threshold(self):
        short = _make_records(50, text="Short.")
        long_text = "The court held this point clearly. " * 100
        long = _make_records(50, text=long_text)
        result = gate_a13_sentence_density(short + long)
        assert result["records_after_a8_filter"] == 50

    def test_pass_when_dense_text(self):
        long_text = "The court held this point clearly. " * 100
        assert gate_a13_sentence_density(_make_records(20, text=long_text))["pass"] is True

    def test_fail_when_all_short_filtered_out(self):
        assert gate_a13_sentence_density(_make_records(20, text="Short."))["pass"] is False

    def test_empty_records_handled(self):
        assert "pass" in gate_a13_sentence_density([])

    def test_accepts_nlp_argument(self):
        assert "nlp" in inspect.signature(gate_a13_sentence_density).parameters

    def test_uses_injected_nlp_without_calling_spacy_load(self):
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.sents = iter([MagicMock() for _ in range(60)])
        mock_nlp.return_value = mock_doc
        long_text = "The court held this point clearly. " * 100
        with patch("src.dataset_probe.spacy") as mock_spacy:
            gate_a13_sentence_density(_make_records(5, text=long_text), nlp=mock_nlp)
            mock_spacy.load.assert_not_called()


class TestGateA13NlpMaxLengthSideEffect:
    def test_injected_nlp_max_length_not_mutated(self):
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ["sentencizer"]
        mock_doc = MagicMock()
        mock_doc.sents = iter([MagicMock() for _ in range(60)])
        mock_nlp.return_value = mock_doc
        long_text = "The court held this point clearly. " * 100
        gate_a13_sentence_density(_make_records(5, text=long_text), nlp=mock_nlp)
        for c in mock_nlp.mock_calls:
            assert "max_length" not in str(c)

    def test_internal_nlp_max_length_is_set(self):
        long_text = "The court held this point clearly. " * 100
        with patch("src.dataset_probe.spacy") as mock_spacy:
            mock_nlp = MagicMock()
            mock_nlp.pipe_names = ["sentencizer"]
            mock_doc = MagicMock()
            mock_doc.sents = iter([MagicMock() for _ in range(60)])
            mock_nlp.return_value = mock_doc
            mock_spacy.load.return_value = mock_nlp
            mock_spacy.__version__ = "3.8.11"
            gate_a13_sentence_density(_make_records(5, text=long_text), nlp=None)
            assert mock_nlp.max_length == 2_000_000


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

    def test_empty_records_handled(self):
        assert "pass" in gate_b6_text_entropy_distribution([])

    def test_b6_spot_check_flags_formula_drift(self):
        assert gate_b6_text_entropy_distribution(_make_records(5, text_entropy=999.0))["spot_check"]["consistent"] is False

    def test_b6_spot_check_reports_max_deviation(self):
        assert "max_deviation" in gate_b6_text_entropy_distribution(_make_records(10))["spot_check"]


# ---------------------------------------------------------------------------
# Fixture JSONL
# ---------------------------------------------------------------------------


def _load_fixture_records() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_JSONL.read_text().splitlines()
        if line.strip()
    ]


class TestFixtureJSONL:
    def test_fixture_file_exists(self):
        assert FIXTURE_JSONL.exists()

    def test_fixture_is_valid_jsonl(self):
        assert len(_load_fixture_records()) >= 5

    def test_fixture_has_all_23_schema_fields(self):
        EXPECTED = {
            "id", "cluster_id", "docket_id", "court_id", "court_name",
            "case_name", "date_filed", "precedential_status", "opinion_type",
            "extracted_by_ocr", "raw_text", "text", "text_length", "text_source",
            "cleaning_flags", "source", "token_count", "paragraph_count",
            "citation_count", "text_hash", "citation_density", "is_precedential",
            "text_entropy",
        }
        for r in _load_fixture_records():
            assert not (EXPECTED - set(r.keys()))

    def test_fixture_usable_as_shard(self, tmp_path):
        import shutil
        shard_dir = tmp_path / "fixture_shards"
        shard_dir.mkdir()
        shutil.copy(FIXTURE_JSONL, shard_dir / "courtlistener_sample.jsonl")
        audit = iter_shards_with_audit(shard_dir)
        assert len(audit["records"]) >= 5
        assert audit["total_parse_errors"] == 0


# ---------------------------------------------------------------------------
# ModelQualitySignals
# ---------------------------------------------------------------------------


class TestModelQualitySignalsIntegration:
    def test_quality_signals_in_report(self, sample_shard_dir, tmp_path):
        assert "quality_signals" in run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )

    def test_truncated_signal_fires(self):
        assert any(s[0] == "truncated_document" for s in ModelQualitySignals.check({"text": "Motion denied."}))


# ---------------------------------------------------------------------------
# run_probe report schema
# ---------------------------------------------------------------------------


class TestRunProbeReportSchema:
    def test_report_has_required_keys(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        for key in ("gates", "summary", "provenance", "quality_signals"):
            assert key in report

    def test_summary_has_all_buckets(self, sample_shard_dir, tmp_path):
        summary = run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )["summary"]
        for key in ("passed", "failed_blocking", "failed_advisory", "skipped", "all_passed"):
            assert key in summary

    def test_report_is_json_serializable(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        json.dumps(report)

    def test_report_written_to_disk(self, sample_shard_dir, tmp_path):
        out = tmp_path / "r.json"
        run_probe(data_dir=sample_shard_dir, subset=20, output=out, skip_tokenizer=True, skip_spacy=True)
        assert out.exists()
        assert json.loads(out.read_text())


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_provenance_has_required_keys(self, sample_shard_dir, tmp_path):
        prov = run_probe(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )["provenance"]
        for key in ("timestamp", "spacy_model_version", "probe_config", "probe_version", "git_sha"):
            assert key in prov


class TestProbeVersion:
    def test_probe_version_constant_exported(self):
        assert isinstance(PROBE_VERSION, str) and len(PROBE_VERSION) >= 3


# ---------------------------------------------------------------------------
# CourtListenerDatasetProbe
# ---------------------------------------------------------------------------


class TestCourtListenerDatasetProbe:
    def test_probe_has_no_validate_row_method(self):
        assert not hasattr(CourtListenerDatasetProbe(), "validate_row")

    def test_accepts_custom_config(self):
        probe = CourtListenerDatasetProbe(config=ProbeConfig(min_text_length=500))
        assert probe.config.min_text_length == 500

    def test_run_returns_report(self, sample_shard_dir, tmp_path):
        report = CourtListenerDatasetProbe().run(
            data_dir=sample_shard_dir, subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True, skip_spacy=True,
        )
        assert "gates" in report and "summary" in report


# ---------------------------------------------------------------------------
# --ci-mode
# ---------------------------------------------------------------------------


class TestCIMode:
    def test_ci_mode_exits_1_when_blocking_gates_fail(self, tmp_path):
        shard = tmp_path / "shards" / "s.jsonl"
        shard.parent.mkdir()
        with open(shard, "w") as fh:
            for r in _make_records(100, text_source="garbage_unknown_format"):
                fh.write(json.dumps(r) + "\n")
        result = subprocess.run(
            [sys.executable, "-m", "src.dataset_probe",
             "--data-dir", str(shard.parent), "--subset", "100",
             "--output", str(tmp_path / "r.json"),
             "--skip-tokenizer", "--skip-spacy", "--ci-mode"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1

    def test_ci_mode_exits_0_when_all_pass(self, sample_shard_dir, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "src.dataset_probe",
             "--data-dir", str(sample_shard_dir), "--subset", "50",
             "--output", str(tmp_path / "r.json"),
             "--skip-tokenizer", "--skip-spacy", "--ci-mode"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_ci_mode_exits_0_when_only_advisory_fails(self, tmp_path):
        shard = tmp_path / "shards" / "s.jsonl"
        shard.parent.mkdir()
        with open(shard, "w") as fh:
            for r in _make_records(100, citation_count=0):
                fh.write(json.dumps(r) + "\n")
        result = subprocess.run(
            [sys.executable, "-m", "src.dataset_probe",
             "--data-dir", str(shard.parent), "--subset", "100",
             "--output", str(tmp_path / "r.json"),
             "--skip-tokenizer", "--skip-spacy", "--ci-mode"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_runs_and_exits_zero(self, sample_shard_dir, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "src.dataset_probe",
             "--data-dir", str(sample_shard_dir), "--subset", "20",
             "--output", str(tmp_path / "cli_out.json"),
             "--skip-tokenizer", "--skip-spacy"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_cli_writes_json_output(self, sample_shard_dir, tmp_path):
        out = tmp_path / "cli_out.json"
        subprocess.run(
            [sys.executable, "-m", "src.dataset_probe",
             "--data-dir", str(sample_shard_dir), "--subset", "20",
             "--output", str(out), "--skip-tokenizer", "--skip-spacy"],
            capture_output=True,
        )
        assert out.exists()
        assert "gates" in json.loads(out.read_text())


# ---------------------------------------------------------------------------
# Shard audit
# ---------------------------------------------------------------------------


class TestShardAuditTotalRecordsDecoded:
    def test_iter_shards_with_audit_has_total_records_decoded(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id":"1","text":"a"}\n{"id":"2","text":"b"}\nBAD\n\n')
        assert "total_records_decoded" in iter_shards_with_audit(tmp_path)

    def test_total_records_decoded_counts_valid_records(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id":"1","text":"a"}\n{"id":"2","text":"b"}\nBAD\n\n')
        assert iter_shards_with_audit(tmp_path)["total_records_decoded"] == 2

    def test_total_records_decoded_excludes_parse_errors(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id":"1"}\nBAD1\nBAD2\n')
        audit = iter_shards_with_audit(tmp_path)
        assert audit["total_records_decoded"] == 1
        assert audit["total_parse_errors"] == 2


# ---------------------------------------------------------------------------
# Citation regex examples
# ---------------------------------------------------------------------------


class TestLegalCitationReExamples:
    def test_regex_matches_federal_third_circuit_reporter(self):
        assert _LEGAL_CITATION_RE.search("123 F.3d 456")

    def test_regex_matches_federal_second_circuit_reporter(self):
        assert _LEGAL_CITATION_RE.search("456 F.2d 789")

    def test_regex_matches_federal_supplement(self):
        assert _LEGAL_CITATION_RE.search("123 F.Supp 456")

    def test_regex_matches_us_supreme_court_citation(self):
        assert _LEGAL_CITATION_RE.search("347 U.S. 483")

    def test_regex_matches_party_v_party(self):
        assert _LEGAL_CITATION_RE.search("Smith v. Jones")

    def test_regex_no_match_on_plain_text(self):
        assert not _LEGAL_CITATION_RE.search("The defendant argued the motion.")

    def test_regex_no_match_on_number_only(self):
        assert not _LEGAL_CITATION_RE.search("Section 42 of the statute")


# ---------------------------------------------------------------------------
# Annotation resolution
# ---------------------------------------------------------------------------


class TestAnnotationResolution:
    def test_gate_a7_annotations_resolve(self):
        assert "records" in get_type_hints(gate_a7_text_source_breakdown)

    def test_gate_a8_annotations_resolve(self):
        assert "records" in get_type_hints(gate_a8_text_length_distribution)

    def test_gate_a9_annotations_resolve(self):
        assert "records" in get_type_hints(gate_a9_citation_count_distribution)

    def test_gate_a11_annotations_resolve(self):
        assert "records" in get_type_hints(gate_a11_tokenizer_chunk_count)

    def test_gate_a12_annotations_resolve(self):
        assert "records" in get_type_hints(gate_a12_citation_anchor_survival)

    def test_gate_a13_annotations_resolve(self):
        assert "records" in get_type_hints(gate_a13_sentence_density)

    def test_gate_b6_annotations_resolve(self):
        assert "records" in get_type_hints(gate_b6_text_entropy_distribution)

    def test_validate_schema_annotations_resolve(self):
        assert "records" in get_type_hints(validate_schema)

    def test_run_probe_annotations_resolve(self):
        assert "data_dir" in get_type_hints(run_probe)

    def test_get_text_annotations_resolve(self):
        assert "row" in get_type_hints(_get_text)


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


class TestPercentileProperty:
    @given(
        values=st.lists(st.integers(min_value=0, max_value=10_000), min_size=1, max_size=500),
        p=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_result_always_within_range(self, values, p):
        sorted_vals = sorted(values)
        result = _percentile(sorted_vals, p)
        assert min(sorted_vals) <= result <= max(sorted_vals)

    @given(
        values=st.lists(st.integers(min_value=0, max_value=10_000), min_size=1, max_size=500),
    )
    @settings(max_examples=200)
    def test_p0_always_returns_min(self, values):
        assert _percentile(sorted(values), 0) == sorted(values)[0]

    @given(
        values=st.lists(st.integers(min_value=0, max_value=10_000), min_size=1, max_size=500),
    )
    @settings(max_examples=200)
    def test_p100_always_returns_max(self, values):
        assert _percentile(sorted(values), 100) == sorted(values)[-1]


class TestGateA8Property:
    @given(
        n_above=st.integers(min_value=1, max_value=500),
        n_below=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=200)
    def test_below_provisional_pct_always_in_0_100(self, n_above, n_below):
        records = _make_records(n_above, text_length=5000) + _make_records(n_below, text_length=100)
        r = gate_a8_text_length_distribution(records)
        assert 0.0 <= r["below_provisional_pct"] <= 100.0

    @given(n=st.integers(min_value=1, max_value=200))
    @settings(max_examples=100)
    def test_all_above_threshold_passes(self, n):
        assert gate_a8_text_length_distribution(_make_records(n, text_length=5000))["pass"] is True

    @given(n=st.integers(min_value=4, max_value=200))
    @settings(max_examples=100)
    def test_all_below_threshold_fails(self, n):
        assert gate_a8_text_length_distribution(_make_records(n, text_length=100))["pass"] is False
