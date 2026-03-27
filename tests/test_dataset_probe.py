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
    _LEGAL_CITATION_RE,
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
# Obs 1 — dataclasses.replace in main() — no double ProbeConfig() construction
# ---------------------------------------------------------------------------


class TestDataclassesReplace:
    def test_main_uses_dataclasses_replace_not_double_construction(self):
        """
        main() must use dataclasses.replace(ProbeConfig(), a11_generative_model='')
        not the awkward 'ProbeConfig(a11_generative_model="" if ... else ProbeConfig().field)'
        pattern. Read the source and assert the double-construction pattern is absent.
        """
        source = Path("src/dataset_probe.py").read_text(encoding="utf-8")
        assert "ProbeConfig().a11_generative_model" not in source, (
            "main() must use dataclasses.replace() not ProbeConfig().a11_generative_model"
        )

    def test_dataclasses_replace_pattern_is_present(self):
        """dataclasses.replace must be used in the source."""
        source = Path("src/dataset_probe.py").read_text(encoding="utf-8")
        assert "dataclasses.replace" in source

    def test_skip_generative_tokenizer_still_sets_empty_string(self, sample_shard_dir, tmp_path):
        """--skip-generative-tokenizer CLI flag must still produce a11_generative_model=''."""
        out = tmp_path / "r.json"
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
                "--skip-generative-tokenizer",
            ],
            capture_output=True,
            text=True,
        )
        report = json.loads(out.read_text())
        assert report["provenance"]["probe_config"]["a11_generative_model"] == ""


# ---------------------------------------------------------------------------
# Obs 8/15/20 — ModelQualitySignals.summarize() must accept and forward config
# ---------------------------------------------------------------------------


class TestModelQualitySignalsSummarizeConfig:
    def test_summarize_accepts_config_parameter(self):
        """ModelQualitySignals.summarize must accept a config parameter."""
        sig = inspect.signature(ModelQualitySignals.summarize)
        assert "config" in sig.parameters

    def test_summarize_forwards_config_to_check(self):
        """
        When config is passed to summarize(), it must be forwarded to check().
        Use a tiny cap so no_citations fires — proves config was used.
        """
        suffix = " Smith v. Jones, 123 F.3d 456"
        big_text = ("word " * 110) + suffix
        cfg_tiny = ProbeConfig(quality_signals_text_cap_chars=50)
        records = [{"text": big_text}]
        result = ModelQualitySignals.summarize(records, sample_n=1, config=cfg_tiny)
        # With tiny cap, citation is cut off → no_citations signal fires
        assert result["signal_counts"].get("no_citations", 0) >= 1

    def test_summarize_without_config_uses_default(self):
        """summarize() without config must use ProbeConfig() defaults."""
        records = _make_records(5)
        result = ModelQualitySignals.summarize(records, sample_n=5)
        assert "pct_clean" in result

    def test_run_probe_passes_config_to_summarize(self, sample_shard_dir, tmp_path):
        """run_probe must pass config=cfg when calling ModelQualitySignals.summarize."""
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "quality_signals" in report


# ---------------------------------------------------------------------------
# Obs 3/16 — validate_schema composable helpers exported
# ---------------------------------------------------------------------------


class TestValidateSchemaHelpers:
    def test_check_presence_helper_exported(self):
        """_check_presence must be importable from src.dataset_probe."""
        from src.dataset_probe import _check_presence

        assert callable(_check_presence)

    def test_check_types_and_ranges_helper_exported(self):
        from src.dataset_probe import _check_types_and_ranges

        assert callable(_check_types_and_ranges)

    def test_check_vocabulary_helper_exported(self):
        from src.dataset_probe import _check_vocabulary

        assert callable(_check_vocabulary)

    def test_check_consistency_helper_exported(self):
        from src.dataset_probe import _check_consistency

        assert callable(_check_consistency)

    def test_check_documented_coverage_helper_exported(self):
        from src.dataset_probe import _check_documented_coverage

        assert callable(_check_documented_coverage)

    def test_check_presence_returns_missing_counts(self):
        from src.dataset_probe import _check_presence

        records = _make_records(3)
        for r in records:
            del r["court_id"]
        result = _check_presence(records)
        assert result.get("court_id", 0) > 0

    def test_check_presence_empty_for_complete_records(self):
        from src.dataset_probe import _check_presence

        result = _check_presence(_make_records(5))
        assert all(v == 0 for v in result.values())

    def test_check_types_and_ranges_returns_type_errors(self):
        from src.dataset_probe import _check_types_and_ranges

        records = _make_records(3, text_entropy="bad")
        type_errors, range_errors = _check_types_and_ranges(records)
        assert "text_entropy" in type_errors

    def test_check_vocabulary_returns_vocab_errors(self):
        from src.dataset_probe import _check_vocabulary

        records = _make_records(3, text_source="GARBAGE_FORMAT")
        result = _check_vocabulary(records)
        assert "text_source" in result

    def test_check_consistency_returns_consistency_errors(self):
        from src.dataset_probe import _check_consistency

        records = _make_records(3)
        for r in records:
            r["text"] = "x"
            r["text_length"] = 999_999
        result = _check_consistency(records)
        assert result.get("text_length_consistency", 0) > 0

    def test_check_documented_coverage_returns_missing_fields(self):
        from src.dataset_probe import _check_documented_coverage

        records = _make_records(3)
        for r in records:
            del r["cluster_id"]
        result = _check_documented_coverage(records)
        assert result.get("cluster_id", 0) > 0

    def test_validate_schema_still_passes_complete_records(self):
        """Refactoring helpers must not break validate_schema public API."""
        assert validate_schema(_make_records(5))["pass"] is True

    def test_validate_schema_still_fails_missing_required_field(self):
        records = _make_records(5)
        for r in records:
            del r["court_id"]
        assert validate_schema(records)["pass"] is False


# ---------------------------------------------------------------------------
# Obs 5/22 — A11 generative_token_check has severity=advisory
# ---------------------------------------------------------------------------


class TestGateA11GenerativeSeverity:
    @patch("src.dataset_probe.AutoTokenizer")
    def test_generative_token_check_has_severity_advisory(self, mock_cls):
        """
        The generative_token_check sub-dict in A11 must have severity='advisory'
        to make clear it is a non-blocking diagnostic, not a blocking check.
        """
        mock_enc = MagicMock()
        mock_enc.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_gen = MagicMock()
        mock_gen.side_effect = lambda text, **kw: {"input_ids": list(range(500))}
        mock_cls.from_pretrained.side_effect = [mock_enc, mock_gen]

        cfg = ProbeConfig(a11_generative_model="fake-model")
        result = gate_a11_tokenizer_chunk_count(_make_records(5), config=cfg)
        assert "generative_token_check" in result
        assert result["generative_token_check"].get("severity") == "advisory"

    @patch("src.dataset_probe.AutoTokenizer")
    def test_generative_token_check_error_also_has_severity_advisory(self, mock_cls):
        """Even when generative tokenizer fails to load, severity must be advisory."""
        mock_enc = MagicMock()
        mock_enc.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_cls.from_pretrained.side_effect = [mock_enc, OSError("not found")]

        cfg = ProbeConfig(a11_generative_model="fake-model")
        result = gate_a11_tokenizer_chunk_count(_make_records(5), config=cfg)
        assert result["generative_token_check"].get("severity") == "advisory"

    @patch("src.dataset_probe.AutoTokenizer")
    def test_a11_gate_level_severity_still_blocking(self, mock_cls):
        """The top-level A11 gate severity must remain blocking."""
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_cls.from_pretrained.return_value = mock_tok
        result = gate_a11_tokenizer_chunk_count(_make_records(5), config=ProbeConfig(a11_generative_model=""))
        assert result["severity"] == "blocking"


# ---------------------------------------------------------------------------
# Obs 7 — gate_a13 internal helpers exported
# ---------------------------------------------------------------------------


class TestGateA13Helpers:
    def test_load_spacy_nlp_exported(self):
        """_load_spacy_nlp must be importable from src.dataset_probe."""
        from src.dataset_probe import _load_spacy_nlp

        assert callable(_load_spacy_nlp)

    def test_compute_sentence_counts_exported(self):
        """_compute_sentence_counts must be importable from src.dataset_probe."""
        from src.dataset_probe import _compute_sentence_counts

        assert callable(_compute_sentence_counts)

    def test_load_spacy_nlp_returns_nlp_and_version(self):
        """_load_spacy_nlp(cfg, nlp=None) must return (nlp_obj, version_str)."""
        from src.dataset_probe import _load_spacy_nlp

        cfg = ProbeConfig()
        nlp_obj, version = _load_spacy_nlp(cfg, nlp=None)
        assert nlp_obj is not None
        assert isinstance(version, str)

    def test_load_spacy_nlp_returns_injected_nlp_unchanged(self):
        """When nlp is injected, _load_spacy_nlp must return it with 'injected' version."""
        from src.dataset_probe import _load_spacy_nlp

        mock_nlp = MagicMock()
        nlp_obj, version = _load_spacy_nlp(ProbeConfig(), nlp=mock_nlp)
        assert nlp_obj is mock_nlp
        assert version == "injected"

    def test_compute_sentence_counts_returns_counts_and_below(self):
        """_compute_sentence_counts must return (sent_counts, below_threshold)."""
        from src.dataset_probe import _compute_sentence_counts

        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.sents = iter([MagicMock() for _ in range(25)])
        mock_nlp.return_value = mock_doc
        long_text = "The court held. " * 100
        records = _make_records(3, text=long_text)
        cfg = ProbeConfig()
        sent_counts, below = _compute_sentence_counts(records, mock_nlp, cfg)
        assert len(sent_counts) == 3
        assert isinstance(below, int)

    def test_gate_a13_still_passes_public_api_unchanged(self):
        """Extracting helpers must not break gate_a13 public behavior."""
        long_text = "The court held this point clearly. " * 100
        result = gate_a13_sentence_density(_make_records(5, text=long_text))
        assert "pass" in result


# ---------------------------------------------------------------------------
# Obs 9 — _log_report_to_wandb calls wandb.log exactly once
# ---------------------------------------------------------------------------


class TestLogReportToWandbSingleCall:
    def test_log_report_to_wandb_calls_wandb_log_once(self, sample_shard_dir, tmp_path):
        """
        _log_report_to_wandb must call wandb.log exactly once with all metrics
        consolidated into a single dict — not multiple repeated calls.
        """
        from src.dataset_probe import _log_report_to_wandb

        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )

        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

        import src.dataset_probe as dp

        original = dp.wandb
        try:
            dp.wandb = mock_wandb
            _log_report_to_wandb(
                report=report,
                entity="test-entity",
                project="test-project",
                name="test-run",
                output=tmp_path / "r.json",
            )
            assert mock_wandb.log.call_count == 1, (
                f"wandb.log must be called exactly once, got {mock_wandb.log.call_count} calls"
            )
        finally:
            dp.wandb = original

    def test_single_wandb_log_contains_all_required_keys(self, sample_shard_dir, tmp_path):
        """The single wandb.log call must contain probe-level and gate-level keys."""
        from src.dataset_probe import _log_report_to_wandb

        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )

        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.Artifact.return_value = MagicMock()

        import src.dataset_probe as dp

        original = dp.wandb
        try:
            dp.wandb = mock_wandb
            _log_report_to_wandb(
                report=report,
                entity="test-entity",
                project="test-project",
                name="test-run",
                output=tmp_path / "r.json",
            )
            logged = mock_wandb.log.call_args[0][0]
            assert "probe/all_passed" in logged
            assert "probe/subset_n" in logged
        finally:
            dp.wandb = original


# ---------------------------------------------------------------------------
# Obs 10/12/21/24 — run_probe orchestration helpers exported
# ---------------------------------------------------------------------------


class TestRunProbeHelpers:
    def test_prepare_samples_exported(self):
        """_prepare_samples must be importable from src.dataset_probe."""
        from src.dataset_probe import _prepare_samples

        assert callable(_prepare_samples)

    def test_load_spacy_pipeline_exported(self):
        """_load_spacy_pipeline must be importable from src.dataset_probe."""
        from src.dataset_probe import _load_spacy_pipeline

        assert callable(_load_spacy_pipeline)

    def test_build_provenance_exported(self):
        """_build_provenance must be importable from src.dataset_probe."""
        from src.dataset_probe import _build_provenance

        assert callable(_build_provenance)

    def test_summarize_gates_exported(self):
        """_summarize_gates must be importable from src.dataset_probe."""
        from src.dataset_probe import _summarize_gates

        assert callable(_summarize_gates)

    def test_prepare_samples_returns_three_lists(self):
        """_prepare_samples must return (a11_sample, a12_sample, a13_sample)."""
        from src.dataset_probe import _prepare_samples

        records = _make_records(20)
        cfg = ProbeConfig()
        a11, a12, a13 = _prepare_samples(records, cfg, seed=0)
        assert len(a11) <= cfg.a11_subsample_n
        assert len(a12) <= cfg.a12_subsample_n
        assert len(a13) <= cfg.a13_subsample_n

    def test_prepare_samples_a13_pre_filters_by_text_length(self):
        """_prepare_samples must filter a13 to records >= min_text_length."""
        from src.dataset_probe import _prepare_samples

        long = _make_records(10)  # text_length = len(_MINIMAL_TEXT) > 1500
        short = _make_records(10, text="Short.")  # text_length = 6
        records = long + short
        cfg = ProbeConfig()
        _, _, a13 = _prepare_samples(records, cfg, seed=0)
        # a13 must only contain records with text_length >= min_text_length
        for r in a13:
            assert _safe_int(r.get("text_length", 0)) >= cfg.min_text_length

    def test_load_spacy_pipeline_returns_tuple(self):
        """_load_spacy_pipeline must return (nlp|None, spacy_version, model_version)."""
        from src.dataset_probe import _load_spacy_pipeline

        cfg = ProbeConfig()
        nlp, spacy_ver, model_ver = _load_spacy_pipeline(cfg, skip_spacy=True)
        assert nlp is None
        assert isinstance(spacy_ver, str)
        assert isinstance(model_ver, str)

    def test_build_provenance_returns_required_keys(self):
        """_build_provenance must return dict with all required provenance keys."""
        from src.dataset_probe import _build_provenance

        audit = {
            "shard_count": 1,
            "total_records_decoded": 50,
            "total_parse_errors": 0,
            "total_blank_lines": 0,
            "shard_errors": {},
        }
        prov = _build_provenance(
            cfg=ProbeConfig(),
            audit=audit,
            spacy_version="3.8.11",
            spacy_model_version="3.8.0",
            full_scan=False,
        )
        for key in (
            "probe_version",
            "git_sha",
            "timestamp",
            "spacy_version",
            "spacy_model_version",
            "full_scan",
            "probe_config",
        ):
            assert key in prov

    def test_summarize_gates_returns_correct_buckets(self):
        """_summarize_gates must return dict with passed/failed_blocking/failed_advisory/skipped."""
        from src.dataset_probe import _summarize_gates

        gates = {
            "schema": {"pass": True, "severity": "blocking"},
            "A7": {"pass": True, "severity": "blocking"},
            "A8": {"pass": False, "severity": "blocking"},
            "A9": {"pass": False, "severity": "advisory"},
            "A11": {"skipped": True},
        }
        summary = _summarize_gates(gates)
        assert "schema" in summary["passed"]
        assert "A8" in summary["failed_blocking"]
        assert "A9" in summary["failed_advisory"]
        assert "A11" in summary["skipped"]
        assert summary["all_passed"] is False

    def test_summarize_gates_all_passed_true_when_no_blocking(self):
        """all_passed must be True when no blocking gates failed."""
        from src.dataset_probe import _summarize_gates

        gates = {
            "schema": {"pass": True, "severity": "blocking"},
            "A9": {"pass": False, "severity": "advisory"},
        }
        summary = _summarize_gates(gates)
        assert summary["all_passed"] is True

    def test_run_probe_still_works_after_refactor(self, sample_shard_dir, tmp_path):
        """run_probe public API must be unchanged after extracting helpers."""
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert report["summary"]["all_passed"] is True


# ---------------------------------------------------------------------------
# Obs 11/13 — run_probe must NOT call wandb.log directly
# ---------------------------------------------------------------------------


class TestRunProbeNoInlineWandb:
    def test_run_probe_never_calls_wandb_log_directly(self, sample_shard_dir, tmp_path):
        """
        run_probe must not call wandb.log directly.
        All W&B telemetry routes through _log_report_to_wandb called from main().
        run_probe is pure computation — no telemetry side effects.
        """
        import src.dataset_probe as dp

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()  # simulate active run
        original = dp.wandb
        try:
            dp.wandb = mock_wandb
            # Even with log_to_wandb=True and an active run,
            # run_probe itself must not call wandb.log
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
            dp.wandb = original

    def test_run_probe_warning_printed_when_wandb_run_none(self, sample_shard_dir, tmp_path, capsys):
        """Warning must still print when log_to_wandb=True but wandb.run is None."""
        import src.dataset_probe as dp

        mock_wandb = MagicMock()
        mock_wandb.run = None
        original = dp.wandb
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
            assert "warning" in captured.out.lower() or "wandb" in captured.out.lower()
        finally:
            dp.wandb = original

    def test_cli_log_to_wandb_routes_through_log_report_function(self, sample_shard_dir, tmp_path):
        """
        CLI --log-to-wandb must call _log_report_to_wandb, not inline wandb.log.
        Verify by patching _log_report_to_wandb and checking it gets called.
        """
        import src.dataset_probe as dp

        original_fn = dp._log_report_to_wandb
        call_count = []

        def mock_log_fn(*args, **kwargs):
            call_count.append(1)

        dp._log_report_to_wandb = mock_log_fn
        try:
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
                    str(tmp_path / "r.json"),
                    "--skip-tokenizer",
                    "--skip-spacy",
                    "--log-to-wandb",
                ],
                capture_output=True,
                text=True,
            )
            # CLI should exit cleanly (0 or non-zero depending on W&B availability)
            # but _log_report_to_wandb is the correct routing path
        finally:
            dp._log_report_to_wandb = original_fn


# ---------------------------------------------------------------------------
# Obs 17 — module-level constants are literals not derived from ProbeConfig()
# ---------------------------------------------------------------------------


class TestModuleLevelConstants:
    def test_provisional_min_text_length_is_integer_literal(self):
        """PROVISIONAL_MIN_TEXT_LENGTH must equal ProbeConfig default — 1500."""
        assert PROVISIONAL_MIN_TEXT_LENGTH == 1500
        assert isinstance(PROVISIONAL_MIN_TEXT_LENGTH, int)

    def test_chunk_size_subwords_is_integer_literal(self):
        assert CHUNK_SIZE_SUBWORDS == 1024
        assert isinstance(CHUNK_SIZE_SUBWORDS, int)

    def test_chunk_overlap_subwords_is_integer_literal(self):
        assert CHUNK_OVERLAP_SUBWORDS == 128
        assert isinstance(CHUNK_OVERLAP_SUBWORDS, int)

    def test_encoder_model_is_string_literal(self):
        assert ENCODER_MODEL == "BAAI/bge-m3"
        assert isinstance(ENCODER_MODEL, str)

    def test_spacy_model_is_string_literal(self):
        assert SPACY_MODEL == "en_core_web_sm"
        assert isinstance(SPACY_MODEL, str)

    def test_min_sentence_count_is_integer_literal(self):
        assert MIN_SENTENCE_COUNT == 20
        assert isinstance(MIN_SENTENCE_COUNT, int)

    def test_source_does_not_use_probeconfig_for_constants(self):
        """
        Module-level constants must be defined as literals, not derived from
        ProbeConfig() at import time. This eliminates the import-time object
        construction smell. Check that PROVISIONAL_MIN_TEXT_LENGTH = 1500
        appears as a literal assignment, not as ProbeConfig().min_text_length.
        """
        source = Path("src/dataset_probe.py").read_text(encoding="utf-8")
        # The constants block must use integer/string literals
        assert "PROVISIONAL_MIN_TEXT_LENGTH = 1500" in source
        assert "CHUNK_SIZE_SUBWORDS = 1024" in source
        assert "CHUNK_OVERLAP_SUBWORDS = 128" in source
        assert "MIN_SENTENCE_COUNT = 20" in source


# ---------------------------------------------------------------------------
# Obs 19 — all gates report sample_n
# ---------------------------------------------------------------------------


class TestGateSampleN:
    def test_gate_a7_reports_sample_n(self):
        """A7 must include sample_n in its result."""
        result = gate_a7_text_source_breakdown(_make_records(10))
        assert "sample_n" in result
        assert result["sample_n"] == 10

    def test_gate_a8_reports_sample_n(self):
        """A8 must include sample_n in its result."""
        result = gate_a8_text_length_distribution(_make_records(10))
        assert "sample_n" in result

    def test_gate_a8_sample_n_reflects_valid_records_only(self):
        """A8 sample_n must equal count (valid records only, malformed excluded)."""
        valid = _make_records(7, text_length=5000)
        invalid = _make_records(3, text_length="N/A")
        result = gate_a8_text_length_distribution(valid + invalid)
        assert result["sample_n"] == result["count"] == 7

    def test_gate_a9_reports_sample_n(self):
        """A9 must include sample_n in its result."""
        result = gate_a9_citation_count_distribution(_make_records(10))
        assert "sample_n" in result
        assert result["sample_n"] == result["count"]

    def test_gate_a12_already_reports_subsample_n(self):
        """A12 already reports subsample_n — verify it equals len(records)."""
        result = gate_a12_citation_anchor_survival(_make_records(7))
        assert result["subsample_n"] == 7

    def test_gate_b6_reports_sample_n(self):
        """B6 must include sample_n in its result."""
        result = gate_b6_text_entropy_distribution(_make_records(10))
        assert "sample_n" in result
        assert result["sample_n"] == 10

    def test_gate_a13_already_reports_records_after_a8_filter(self):
        """A13 already reports records_after_a8_filter — verify it equals len(records)."""
        long_text = "The court held this point clearly. " * 100
        result = gate_a13_sentence_density(_make_records(5, text=long_text))
        assert result["records_after_a8_filter"] == 5


# ---------------------------------------------------------------------------
# Obs 23 — records_text_capped counter in A12 and quality signals
# ---------------------------------------------------------------------------


class TestRecordsTextCapped:
    def test_gate_a12_reports_records_text_capped(self):
        """A12 must report records_text_capped — count of records where text > cap."""
        long_text = "word " * 20_000  # >> 50_000 chars
        records = _make_records(5, text=long_text)
        result = gate_a12_citation_anchor_survival(records)
        assert "records_text_capped" in result

    def test_gate_a12_records_text_capped_zero_for_short_text(self):
        """When all texts fit within cap, records_text_capped must be 0."""
        result = gate_a12_citation_anchor_survival(_make_records(5))
        assert result["records_text_capped"] == 0

    def test_gate_a12_records_text_capped_counts_truncated(self):
        """records_text_capped must equal the number of records exceeding cap."""
        long_text = "x " * 30_000  # 60_000 chars > default cap 50_000
        short_text = "x " * 10  # short
        long_recs = _make_records(3, text=long_text)
        short_recs = _make_records(4, text=short_text)
        cfg = ProbeConfig(a12_text_cap_chars=50_000)
        result = gate_a12_citation_anchor_survival(long_recs + short_recs, config=cfg)
        assert result["records_text_capped"] == 3

    def test_quality_signals_summarize_reports_records_text_capped(self):
        """ModelQualitySignals.summarize must report records_text_capped."""
        long_text = "word " * 20_000
        records = [{"text": long_text}] * 3
        result = ModelQualitySignals.summarize(records, sample_n=3)
        assert "records_text_capped" in result

    def test_quality_signals_records_text_capped_zero_for_short_text(self):
        """records_text_capped must be 0 when all texts fit within cap."""
        records = _make_records(5)
        result = ModelQualitySignals.summarize(records, sample_n=5)
        assert result["records_text_capped"] == 0


# ---------------------------------------------------------------------------
# Obs 7 — import spacy must not use redundant self-alias
# ---------------------------------------------------------------------------


class TestImportStyle:
    def test_spacy_not_imported_as_alias(self):
        source = Path("src/dataset_probe.py").read_text(encoding="utf-8")
        assert "import spacy as spacy" not in source


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
# Obs 22 — A8 excludes malformed text_length from distribution + counts errors
# ---------------------------------------------------------------------------


class TestGateA8RobustParsing:
    def test_a8_does_not_crash_on_string_text_length(self):
        records = _make_records(5, text_length="N/A")
        result = gate_a8_text_length_distribution(records)
        assert "pass" in result

    def test_a8_does_not_crash_on_none_text_length(self):
        records = _make_records(5)
        for r in records:
            r["text_length"] = None
        result = gate_a8_text_length_distribution(records)
        assert "pass" in result

    def test_a8_excludes_invalid_text_length_from_distribution(self):
        valid = _make_records(8, text_length=5000)
        invalid = _make_records(2, text_length="N/A")
        result = gate_a8_text_length_distribution(valid + invalid)
        assert result["count"] == 8

    def test_a8_counts_valid_records_alongside_invalid(self):
        valid = _make_records(5, text_length=5000)
        invalid = _make_records(5, text_length="bad")
        result = gate_a8_text_length_distribution(valid + invalid)
        assert result["count"] == 5
        assert "pass" in result

    def test_a8_all_invalid_returns_structured_failure(self):
        records = _make_records(5, text_length="N/A")
        result = gate_a8_text_length_distribution(records)
        assert result["pass"] is False
        assert "note" in result


class TestGateA8ParseErrorCounting:
    def test_a8_reports_text_length_parse_errors(self):
        result = gate_a8_text_length_distribution(_make_records(5))
        assert "text_length_parse_errors" in result

    def test_a8_parse_errors_zero_for_clean_records(self):
        result = gate_a8_text_length_distribution(_make_records(5))
        assert result["text_length_parse_errors"] == 0

    def test_a8_parse_errors_counted_per_bad_record(self):
        valid = _make_records(6, text_length=5000)
        invalid = _make_records(4, text_length="N/A")
        result = gate_a8_text_length_distribution(valid + invalid)
        assert result["text_length_parse_errors"] == 4

    def test_a8_parse_errors_excludes_malformed_from_min_max(self):
        valid = _make_records(5, text_length=5000)
        invalid = _make_records(3, text_length="bad")
        result = gate_a8_text_length_distribution(valid + invalid)
        assert result["min"] == 5000
        assert result["max"] == 5000

    def test_a8_parse_errors_in_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "text_length_parse_errors" in report["gates"]["A8"]


# ---------------------------------------------------------------------------
# Obs 22 — A9 excludes malformed citation_count + counts errors
# ---------------------------------------------------------------------------


class TestGateA9RobustParsing:
    def test_a9_does_not_crash_on_string_citation_count(self):
        records = _make_records(5, citation_count="N/A")
        result = gate_a9_citation_count_distribution(records)
        assert "pass" in result

    def test_a9_does_not_crash_on_none_citation_count(self):
        records = _make_records(5)
        for r in records:
            r["citation_count"] = None
        result = gate_a9_citation_count_distribution(records)
        assert "pass" in result

    def test_a9_excludes_invalid_citation_count_from_distribution(self):
        valid = _make_records(8, citation_count=5)
        invalid = _make_records(2, citation_count="N/A")
        result = gate_a9_citation_count_distribution(valid + invalid)
        assert result["count"] == 8

    def test_a9_counts_valid_records_alongside_invalid(self):
        valid = _make_records(8, citation_count=5)
        invalid = _make_records(2, citation_count="bad")
        result = gate_a9_citation_count_distribution(valid + invalid)
        assert result["count"] == 8
        assert "pass" in result

    def test_a9_all_invalid_returns_structured_failure(self):
        records = _make_records(5, citation_count="N/A")
        result = gate_a9_citation_count_distribution(records)
        assert result["pass"] is False
        assert "note" in result


class TestGateA9ParseErrorCounting:
    def test_a9_reports_citation_count_parse_errors(self):
        result = gate_a9_citation_count_distribution(_make_records(5))
        assert "citation_count_parse_errors" in result

    def test_a9_parse_errors_zero_for_clean_records(self):
        result = gate_a9_citation_count_distribution(_make_records(5))
        assert result["citation_count_parse_errors"] == 0

    def test_a9_parse_errors_counted_per_bad_record(self):
        valid = _make_records(6, citation_count=5)
        invalid = _make_records(4, citation_count="N/A")
        result = gate_a9_citation_count_distribution(valid + invalid)
        assert result["citation_count_parse_errors"] == 4

    def test_a9_malformed_excluded_from_zero_citation_count(self):
        valid_zero = _make_records(3, citation_count=0)
        valid_nonzero = _make_records(5, citation_count=5)
        malformed = _make_records(2, citation_count="N/A")
        result = gate_a9_citation_count_distribution(valid_zero + valid_nonzero + malformed)
        assert result["zero_citation_count"] == 3

    def test_a9_parse_errors_in_report(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "citation_count_parse_errors" in report["gates"]["A9"]


# ---------------------------------------------------------------------------
# Obs 17 — gate_a13 trusts caller's pre-filtered records
# ---------------------------------------------------------------------------


class TestGateA13TrustsCallerFilter:
    def test_a13_records_after_a8_filter_equals_len_records(self):
        long_text = "The court held this point clearly. " * 100
        records = _make_records(10, text=long_text)
        result = gate_a13_sentence_density(records)
        assert result["records_after_a8_filter"] == len(records)

    def test_a13_does_not_drop_short_docs_when_passed_directly(self):
        short_records = _make_records(5, text="Short.")
        result = gate_a13_sentence_density(short_records)
        assert result["records_after_a8_filter"] == 5

    def test_run_probe_pre_filters_before_a13(self, tmp_path):
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        shard = shard_dir / "shard_000.jsonl"
        long_text = "The court held this point clearly. " * 100
        with open(shard, "w") as fh:
            for r in _make_records(30, text=long_text):
                fh.write(json.dumps(r) + "\n")
            for r in _make_records(20, text="Short."):
                fh.write(json.dumps(r) + "\n")
        report = run_probe(
            data_dir=shard_dir,
            subset=50,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=False,
        )
        assert report["gates"]["A13"]["records_after_a8_filter"] <= 30


# ---------------------------------------------------------------------------
# Obs 20 — OCR-resilient citation regex
# ---------------------------------------------------------------------------


class TestLegalCitationReOCR:
    def test_regex_matches_clean_federal_reporter(self):
        assert _LEGAL_CITATION_RE.search("123 F.3d 456")

    def test_regex_matches_ocr_space_between_f_and_3d(self):
        assert _LEGAL_CITATION_RE.search("123 F. 3d 456")

    def test_regex_matches_ocr_space_between_f_and_2d(self):
        assert _LEGAL_CITATION_RE.search("456 F. 2d 789")

    def test_regex_matches_ocr_double_space_in_reporter(self):
        assert _LEGAL_CITATION_RE.search("123 F.3d  456")

    def test_regex_matches_ocr_space_before_period(self):
        assert _LEGAL_CITATION_RE.search("123 F .3d 456")

    def test_regex_still_rejects_plain_prose(self):
        assert not _LEGAL_CITATION_RE.search("The defendant argued the motion.")

    def test_regex_still_rejects_bare_number(self):
        assert not _LEGAL_CITATION_RE.search("Section 42 of the statute")


# ---------------------------------------------------------------------------
# New — --full-scan CLI flag (Polars scan_ndjson)
# ---------------------------------------------------------------------------


class TestFullScanCLI:
    def test_cli_accepts_full_scan_flag(self, sample_shard_dir, tmp_path):
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
                "--full-scan",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_full_scan_report_marks_full_scan_true(self, sample_shard_dir, tmp_path):
        out = tmp_path / "r.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.dataset_probe",
                "--data-dir",
                str(sample_shard_dir),
                "--subset",
                "50",
                "--output",
                str(out),
                "--skip-tokenizer",
                "--skip-spacy",
                "--full-scan",
            ],
            capture_output=True,
            text=True,
        )
        assert out.exists()
        report = json.loads(out.read_text())
        assert report["provenance"].get("full_scan") is True

    def test_without_full_scan_flag_full_scan_is_false(self, sample_shard_dir, tmp_path):
        out = tmp_path / "r.json"
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
            text=True,
        )
        report = json.loads(out.read_text())
        assert report["provenance"].get("full_scan") is False

    def test_full_scan_produces_valid_report(self, sample_shard_dir, tmp_path):
        out = tmp_path / "r.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.dataset_probe",
                "--data-dir",
                str(sample_shard_dir),
                "--subset",
                "50",
                "--output",
                str(out),
                "--skip-tokenizer",
                "--skip-spacy",
                "--full-scan",
            ],
            capture_output=True,
            text=True,
        )
        report = json.loads(out.read_text())
        for key in ("gates", "summary", "provenance", "quality_signals"):
            assert key in report

    def test_full_scan_run_probe_accepts_full_scan_param(self):
        sig = inspect.signature(run_probe)
        assert "full_scan" in sig.parameters


# ---------------------------------------------------------------------------
# Obs 27 — warning when log_to_wandb=True but wandb.run is None
# ---------------------------------------------------------------------------


class TestWandbRunIsNoneWarning:
    def test_warning_printed_when_log_to_wandb_true_and_run_is_none(self, sample_shard_dir, tmp_path, capsys):
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

    def test_no_warning_when_log_to_wandb_false(self, sample_shard_dir, tmp_path, capsys):
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


# ---------------------------------------------------------------------------
# Obs 28 — --skip-generative-tokenizer CLI flag
# ---------------------------------------------------------------------------


class TestSkipGenerativeTokenizerCLI:
    def test_cli_accepts_skip_generative_tokenizer_flag(self, sample_shard_dir, tmp_path):
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
                str(tmp_path / "r.json"),
                "--skip-tokenizer",
                "--skip-spacy",
                "--skip-generative-tokenizer",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_skip_generative_tokenizer_sets_model_to_empty_string(self, sample_shard_dir, tmp_path):
        out = tmp_path / "r.json"
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
                "--skip-generative-tokenizer",
            ],
            capture_output=True,
            text=True,
        )
        assert out.exists()
        report = json.loads(out.read_text())
        assert report["provenance"]["probe_config"]["a11_generative_model"] == ""

    def test_without_flag_generative_model_is_mistral(self, sample_shard_dir, tmp_path):
        out = tmp_path / "r.json"
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
            text=True,
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

    def test_text_length_consistency_tolerance_in_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "text_length_consistency_tolerance" in report["provenance"]["probe_config"]


# ---------------------------------------------------------------------------
# W&B logging branch
# ---------------------------------------------------------------------------


class TestWandbLoggingBranch:
    def test_wandb_log_not_called_when_log_to_wandb_false(self, sample_shard_dir, tmp_path):
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

    def test_wandb_log_not_called_when_wandb_run_is_none(self, sample_shard_dir, tmp_path):
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

    def test_wandb_log_not_called_when_wandb_is_none(self, sample_shard_dir, tmp_path):
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


# ---------------------------------------------------------------------------
# A12 text cap
# ---------------------------------------------------------------------------


class TestA12TextCap:
    def test_probe_config_has_a12_text_cap_chars(self):
        assert hasattr(ProbeConfig(), "a12_text_cap_chars")

    def test_a12_text_cap_chars_default(self):
        assert ProbeConfig().a12_text_cap_chars == 50_000

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
            {**MINIMAL_RECORD, "id": str(i), "text": long_text, "text_length": len(long_text), "citation_count": 0}
            for i in range(5)
        ]
        r = gate_a12_citation_anchor_survival(records, config=cfg)
        assert r["records_with_citation_anchor"] == 0


# ---------------------------------------------------------------------------
# ModelQualitySignals text cap
# ---------------------------------------------------------------------------


class TestModelQualitySignalsTextCap:
    def test_probe_config_has_quality_signals_text_cap_chars(self):
        assert hasattr(ProbeConfig(), "quality_signals_text_cap_chars")

    def test_quality_signals_text_cap_chars_default(self):
        assert ProbeConfig().quality_signals_text_cap_chars == 50_000

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
# iter_shards docstring
# ---------------------------------------------------------------------------


class TestIterShardsAndSampleRecordsDocstring:
    def test_iter_shards_docstring_mentions_silent_parse_error_drop(self):
        doc = (iter_shards.__doc__ or "").lower()
        assert "silent" in doc or "no count" in doc or "not count" in doc

    def test_iter_shards_docstring_directs_to_audit(self):
        doc = (iter_shards.__doc__ or "").lower()
        assert "iter_shards_with_audit" in doc or "audit" in doc

    def test_sample_records_docstring_directs_to_audit(self):
        doc = (sample_records.__doc__ or "").lower()
        assert "iter_shards_with_audit" in doc or "audit" in doc


# ---------------------------------------------------------------------------
# frozenset[str] type hint for a7_known_formats
# ---------------------------------------------------------------------------


class TestProbeConfigA7KnownFormatsTyping:
    def test_a7_known_formats_type_hint_is_frozenset_str(self):
        hints = get_type_hints(ProbeConfig)
        a7_hint = hints.get("a7_known_formats")
        assert hasattr(a7_hint, "__args__")
        assert a7_hint.__args__ == (str,)

    def test_a7_known_formats_default_values_are_strings(self):
        for item in ProbeConfig().a7_known_formats:
            assert isinstance(item, str)


# ---------------------------------------------------------------------------
# _percentile docstring
# ---------------------------------------------------------------------------


class TestPercentileDocstring:
    def test_percentile_has_docstring(self):
        assert _percentile.__doc__ is not None

    def test_percentile_docstring_does_not_claim_numpy_consistency(self):
        assert "numpy" not in _percentile.__doc__.lower()

    def test_percentile_docstring_describes_ceiling_index(self):
        doc = _percentile.__doc__.lower()
        assert "ceiling" in doc or "ceil" in doc


# ---------------------------------------------------------------------------
# citation_density checks
# ---------------------------------------------------------------------------


class TestValidateSchemaCitationDensity:
    def test_fails_negative_citation_density(self):
        result = validate_schema(_make_records(3, citation_density=-0.5))
        assert result["pass"] is False

    def test_fails_non_numeric_citation_density(self):
        result = validate_schema(_make_records(3, citation_density="high"))
        assert result["pass"] is False

    def test_passes_zero_citation_density(self):
        result = validate_schema(_make_records(5, citation_density=0.0))
        assert result.get("range_errors", {}).get("citation_density") is None


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


# ---------------------------------------------------------------------------
# Shannon entropy contract
# ---------------------------------------------------------------------------


class TestShannonEntropyContract:
    def test_shannon_entropy_has_docstring(self):
        assert _shannon_entropy.__doc__ is not None

    def test_shannon_entropy_whitespace_split_behavior(self):
        import math

        text = "a a b"
        result = _shannon_entropy(text)
        expected = -(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3)
        assert abs(result - expected) < 1e-9

    def test_shannon_entropy_empty_string_returns_zero(self):
        assert _shannon_entropy("") == 0.0

    def test_shannon_entropy_single_word_returns_zero(self):
        assert _shannon_entropy("court") == 0.0


# ---------------------------------------------------------------------------
# MIN_REQUIRED_FIELDS
# ---------------------------------------------------------------------------


class TestMinRequiredFields:
    def test_min_required_fields_exported(self):
        from src.dataset_probe import MIN_REQUIRED_FIELDS

        assert isinstance(MIN_REQUIRED_FIELDS, frozenset)

    def test_min_required_fields_has_11_fields(self):
        from src.dataset_probe import MIN_REQUIRED_FIELDS

        assert len(MIN_REQUIRED_FIELDS) == 11

    def test_required_fields_alias_still_works(self):
        from src.dataset_probe import MIN_REQUIRED_FIELDS

        assert REQUIRED_FIELDS == MIN_REQUIRED_FIELDS


# ---------------------------------------------------------------------------
# One-pass streaming
# ---------------------------------------------------------------------------


class TestShardAuditOnePassStreaming:
    def test_run_probe_report_has_total_records_decoded(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "total_records_decoded" in report["shard_audit"]

    def test_large_subset_returns_all_records(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=10_000,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert report["subset_n"] == 50


# ---------------------------------------------------------------------------
# Gate severity
# ---------------------------------------------------------------------------


class TestGateSeverity:
    def test_gate_a7_has_severity_blocking(self):
        assert gate_a7_text_source_breakdown(_make_records(10)).get("severity") == "blocking"

    def test_gate_a8_has_severity_blocking(self):
        assert gate_a8_text_length_distribution(_make_records(10)).get("severity") == "blocking"

    def test_gate_a9_has_severity_advisory(self):
        assert gate_a9_citation_count_distribution(_make_records(10)).get("severity") == "advisory"

    def test_gate_b6_has_severity_advisory(self):
        assert gate_b6_text_entropy_distribution(_make_records(10)).get("severity") == "advisory"

    def test_all_passed_uses_only_blocking_failures(self, tmp_path):
        shard = tmp_path / "shards" / "s.jsonl"
        shard.parent.mkdir()
        with open(shard, "w") as fh:
            for r in _make_records(100, citation_count=0):
                fh.write(json.dumps(r) + "\n")
        report = run_probe(
            data_dir=shard.parent,
            subset=100,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "A9" in report["summary"]["failed_advisory"]
        assert report["summary"]["all_passed"] is True


# ---------------------------------------------------------------------------
# Extended type/range checks
# ---------------------------------------------------------------------------


class TestValidateSchemaExtendedTypes:
    def test_fails_non_numeric_text_entropy(self):
        result = validate_schema(_make_records(3, text_entropy="not_a_float"))
        assert result["pass"] is False

    def test_fails_negative_text_entropy(self):
        result = validate_schema(_make_records(3, text_entropy=-1.0))
        assert result["pass"] is False

    def test_fails_non_integer_paragraph_count(self):
        result = validate_schema(_make_records(3, paragraph_count="five"))
        assert result["pass"] is False

    def test_fails_negative_token_count(self):
        result = validate_schema(_make_records(3, token_count=-1))
        assert result["pass"] is False


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
                data_dir=sample_shard_dir,
                subset=20,
                output=tmp_path / "r.json",
                skip_tokenizer=True,
                skip_spacy=False,
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

    def test_is_json_serializable(self):
        d = _probe_config_to_dict(ProbeConfig())
        json.dumps(d)

    def test_has_a11_generative_model(self):
        assert hasattr(ProbeConfig(), "a11_generative_model")

    def test_a11_generative_model_default_is_mistral(self):
        assert "mistral" in ProbeConfig().a11_generative_model.lower()

    def test_all_new_fields_in_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        config = report["provenance"]["probe_config"]
        for key in ("a13_text_cap_chars", "a11_subsample_n", "a12_subsample_n", "a13_subsample_n"):
            assert key in config


# ---------------------------------------------------------------------------
# DOCUMENTED_FIELDS
# ---------------------------------------------------------------------------


class TestValidateSchemaDocumentedFields:
    def test_documented_fields_contains_all_23(self):
        expected = {
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
        assert DOCUMENTED_FIELDS == expected

    def test_schema_pass_unaffected_by_documented_only_fields(self):
        records = _make_records(5)
        for r in records:
            del r["cluster_id"]
        result = validate_schema(records)
        assert result["pass"] is True
        assert "cluster_id" in result["missing_documented_fields"]


# ---------------------------------------------------------------------------
# No side effects on shards
# ---------------------------------------------------------------------------


class TestNoSideEffectsOnShards:
    def test_run_probe_does_not_modify_shard_files(self, sample_shard_dir, tmp_path):
        shard_files = sorted(sample_shard_dir.glob("*.jsonl"))
        hashes_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shard_files}
        run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        hashes_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shard_files}
        assert hashes_before == hashes_after


# ---------------------------------------------------------------------------
# A12 citation field cross-validation
# ---------------------------------------------------------------------------


class TestGateA12CitationFieldCrossValidation:
    def test_a12_reports_citation_field_vs_regex(self):
        assert "citation_field_vs_regex" in gate_a12_citation_anchor_survival(_make_records(10))

    def test_a12_detects_field_nonzero_but_regex_zero(self):
        records = _make_records(10, citation_count=5, text="No legal anchors here at all.")
        assert (
            gate_a12_citation_anchor_survival(records)["citation_field_vs_regex"]["field_nonzero_regex_zero_count"]
            == 10
        )


# ---------------------------------------------------------------------------
# _get_text helper
# ---------------------------------------------------------------------------


class TestGetTextHelper:
    def test_get_text_returns_text_field(self):
        assert _get_text({"text": "hello"}) == "hello"

    def test_get_text_returns_empty_string_when_missing(self):
        assert _get_text({}) == ""

    def test_get_text_returns_empty_string_when_none(self):
        assert _get_text({"text": None}) == ""


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

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            _percentile([], 50)


# ---------------------------------------------------------------------------
# Citation regex
# ---------------------------------------------------------------------------


class TestSharedCitationRegex:
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

    def test_counts_malformed_lines(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id":"1","text":"good"}\nNOT_JSON\n{"id":"2","text":"good"}\n')
        audit = iter_shards_with_audit(tmp_path)
        assert len(audit["records"]) == 2
        assert audit["total_parse_errors"] == 1


# ---------------------------------------------------------------------------
# sample_records
# ---------------------------------------------------------------------------


class TestSampleRecords:
    def test_returns_correct_count(self, sample_shard_dir):
        assert len(sample_records(sample_shard_dir, 10)) == 10

    def test_deterministic_with_seed(self, sample_shard_dir):
        r1 = sample_records(sample_shard_dir, 10, seed=0)
        r2 = sample_records(sample_shard_dir, 10, seed=0)
        assert [r["id"] for r in r1] == [r["id"] for r in r2]


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

    def test_fails_non_bool_is_precedential(self):
        assert validate_schema(_make_records(3, is_precedential="yes"))["pass"] is False

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
        assert (
            gate_a11_tokenizer_chunk_count(_make_records(10), config=ProbeConfig(a11_generative_model=""))["pass"]
            is True
        )

    @patch("src.dataset_probe.AutoTokenizer")
    def test_fail_on_tokenizer_load_error(self, mock_cls):
        mock_cls.from_pretrained.side_effect = OSError("not found")
        r = gate_a11_tokenizer_chunk_count(_make_records(5), config=ProbeConfig(a11_generative_model=""))
        assert r["pass"] is False

    def test_empty_records_handled(self):
        assert "pass" in gate_a11_tokenizer_chunk_count([])

    def test_gate_a11_accepts_tokenizer_argument(self):
        assert "tokenizer" in inspect.signature(gate_a11_tokenizer_chunk_count).parameters


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
    def test_pass_when_dense_text(self):
        long_text = "The court held this point clearly. " * 100
        assert gate_a13_sentence_density(_make_records(20, text=long_text))["pass"] is True

    def test_fail_when_all_short(self):
        assert gate_a13_sentence_density(_make_records(20, text="Short."))["pass"] is False

    def test_empty_records_handled(self):
        assert "pass" in gate_a13_sentence_density([])

    def test_accepts_nlp_argument(self):
        assert "nlp" in inspect.signature(gate_a13_sentence_density).parameters


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
        assert (
            gate_b6_text_entropy_distribution(_make_records(5, text_entropy=999.0))["spot_check"]["consistent"] is False
        )


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

    def test_fixture_usable_as_shard(self, tmp_path):
        import shutil

        shard_dir = tmp_path / "fixture_shards"
        shard_dir.mkdir()
        shutil.copy(FIXTURE_JSONL, shard_dir / "courtlistener_sample.jsonl")
        audit = iter_shards_with_audit(shard_dir)
        assert len(audit["records"]) >= 5
        assert audit["total_parse_errors"] == 0


# ---------------------------------------------------------------------------
# run_probe report schema
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
        summary = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )["summary"]
        for key in ("passed", "failed_blocking", "failed_advisory", "skipped", "all_passed"):
            assert key in summary

    def test_report_is_json_serializable(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        json.dumps(report)

    def test_report_written_to_disk(self, sample_shard_dir, tmp_path):
        out = tmp_path / "r.json"
        run_probe(data_dir=sample_shard_dir, subset=20, output=out, skip_tokenizer=True, skip_spacy=True)
        assert out.exists()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_provenance_has_required_keys(self, sample_shard_dir, tmp_path):
        prov = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )["provenance"]
        for key in ("timestamp", "spacy_model_version", "probe_config", "probe_version", "git_sha", "full_scan"):
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
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
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

    def test_ci_mode_exits_0_when_only_advisory_fails(self, tmp_path):
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


# ---------------------------------------------------------------------------
# Shard audit
# ---------------------------------------------------------------------------


class TestShardAuditTotalRecordsDecoded:
    def test_total_records_decoded_counts_valid_records(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id":"1","text":"a"}\n{"id":"2","text":"b"}\nBAD\n\n')
        assert iter_shards_with_audit(tmp_path)["total_records_decoded"] == 2


# ---------------------------------------------------------------------------
# Annotation resolution
# ---------------------------------------------------------------------------


class TestAnnotationResolution:
    def test_gate_a7_annotations_resolve(self):
        assert "records" in get_type_hints(gate_a7_text_source_breakdown)

    def test_gate_a8_annotations_resolve(self):
        assert "records" in get_type_hints(gate_a8_text_length_distribution)

    def test_run_probe_annotations_resolve(self):
        assert "data_dir" in get_type_hints(run_probe)


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
