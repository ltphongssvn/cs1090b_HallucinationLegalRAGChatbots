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
    GATE_REGISTRY,
    MIN_SENTENCE_COUNT,
    PROBE_VERSION,
    PROVISIONAL_MIN_TEXT_LENGTH,
    REQUIRED_FIELDS,
    SPACY_MODEL,
    STAGE3_REQUIRED_FIELDS,
    CourtListenerDatasetProbe,
    GateResult,
    ModelQualitySignals,
    ProbeConfig,
    _build_provenance,
    _check_consistency,
    _check_documented_coverage,
    _check_presence,
    _check_types_and_ranges,
    _check_vocabulary,
    _compute_sentence_counts,
    _get_text,
    _load_spacy_nlp,
    _load_spacy_pipeline,
    _log_report_to_wandb,
    _percentile,
    _prepare_samples,
    _probe_config_to_dict,
    _reservoir_sample,
    _safe_int,
    _shannon_entropy,
    _summarize_gates,
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


# ===========================================================================
# obs 16: _reservoir_sample pure function
# ===========================================================================


class TestReservoirSamplePure:
    def test_reservoir_sample_exported(self):
        assert callable(_reservoir_sample)

    def test_reservoir_sample_takes_any_iterable(self):
        records = [{"id": str(i)} for i in range(20)]
        result = _reservoir_sample(iter(records), n=5, seed=0)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_reservoir_sample_deterministic_with_same_seed(self):
        records = [{"id": str(i)} for i in range(100)]
        r1 = _reservoir_sample(iter(records), n=10, seed=42)
        r2 = _reservoir_sample(iter(records), n=10, seed=42)
        assert [r["id"] for r in r1] == [r["id"] for r in r2]

    def test_reservoir_sample_different_seeds_differ(self):
        records = [{"id": str(i)} for i in range(100)]
        r1 = _reservoir_sample(iter(records), n=10, seed=0)
        r2 = _reservoir_sample(iter(records), n=10, seed=99)
        assert [r["id"] for r in r1] != [r["id"] for r in r2]

    def test_reservoir_sample_n_larger_than_iterable_returns_all(self):
        records = [{"id": str(i)} for i in range(5)]
        result = _reservoir_sample(iter(records), n=1000, seed=0)
        assert len(result) == 5

    def test_reservoir_sample_empty_iterable_returns_empty(self):
        result = _reservoir_sample(iter([]), n=10, seed=0)
        assert result == []

    def test_reservoir_sample_no_disk_io_required(self):
        data = [{"id": str(i), "text": f"doc {i}"} for i in range(50)]
        result = _reservoir_sample(iter(data), n=10, seed=7)
        assert all("id" in r for r in result)

    def test_reservoir_sample_with_audit_uses_reservoir_sample(self, sample_shard_dir):
        from src.dataset_probe import _reservoir_sample_with_audit

        reservoir, audit = _reservoir_sample_with_audit(sample_shard_dir, n=10, seed=42)
        assert len(reservoir) == 10
        assert audit["total_records_decoded"] == 50


# ===========================================================================
# obs 18: relative text_length tolerance
# ===========================================================================


class TestTextLengthRelativeTolerance:
    def test_probeconfig_has_text_length_relative_tolerance(self):
        assert hasattr(ProbeConfig(), "text_length_relative_tolerance")

    def test_text_length_relative_tolerance_default_is_5_percent(self):
        assert ProbeConfig().text_length_relative_tolerance == 0.05

    def test_text_length_relative_tolerance_custom_accepted(self):
        cfg = ProbeConfig(text_length_relative_tolerance=0.10)
        assert cfg.text_length_relative_tolerance == 0.10

    def test_short_doc_fails_with_relative_tolerance(self):
        """500-char doc with 200-char diff = 40% > 5% relative → must fail."""
        records = _make_records(3)
        for r in records:
            r["text"] = "x" * 500
            r["text_length"] = 700  # 200 char diff → 40% of 500
        result = validate_schema(records)
        assert result["consistency_errors"].get("text_length_consistency", 0) == 3

    def test_long_doc_passes_with_relative_tolerance(self):
        """100K-char doc with 200-char diff = 0.2% < 5% relative → must pass."""
        long_text = "x" * 100_000
        records = _make_records(3)
        for r in records:
            r["text"] = long_text
            r["text_length"] = 100_200  # 200 char diff → 0.2% of 100K
        # 0.2% is well within 5% relative tolerance → must NOT flag
        result = validate_schema(records)
        assert result.get("consistency_errors", {}).get("text_length_consistency", 0) == 0

    def test_check_consistency_uses_relative_tolerance(self):
        sig = inspect.signature(_check_consistency)
        assert "relative_tolerance" in sig.parameters

    def test_custom_relative_tolerance_respected(self):
        """custom relative_tolerance=0.01 (1%) must flag a 2% discrepancy."""
        cfg = ProbeConfig(text_length_relative_tolerance=0.01)
        records = _make_records(3)
        for r in records:
            r["text"] = "x" * 1_000
            r["text_length"] = 1_020  # 20 char diff → 2% of 1000 → exceeds 1%
        result = validate_schema(records, config=cfg)
        assert result["consistency_errors"].get("text_length_consistency", 0) == 3

    def test_relative_tolerance_in_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "text_length_relative_tolerance" in report["provenance"]["probe_config"]


# ===========================================================================
# obs 23: STAGE3_REQUIRED_FIELDS + validate_schema stage3 readiness
# ===========================================================================


class TestStage3RequiredFields:
    def test_stage3_required_fields_exported(self):
        assert isinstance(STAGE3_REQUIRED_FIELDS, frozenset)

    def test_stage3_required_fields_is_frozenset(self):
        assert isinstance(STAGE3_REQUIRED_FIELDS, frozenset)

    def test_stage3_required_fields_subset_of_documented(self):
        assert STAGE3_REQUIRED_FIELDS.issubset(DOCUMENTED_FIELDS)

    def test_stage3_required_fields_include_key_stage3_fields(self):
        assert "id" in STAGE3_REQUIRED_FIELDS
        assert "court_id" in STAGE3_REQUIRED_FIELDS
        assert "text" in STAGE3_REQUIRED_FIELDS
        assert "text_length" in STAGE3_REQUIRED_FIELDS
        assert "is_precedential" in STAGE3_REQUIRED_FIELDS
        assert "citation_count" in STAGE3_REQUIRED_FIELDS
        assert "date_filed" in STAGE3_REQUIRED_FIELDS

    def test_validate_schema_reports_stage3_readiness(self):
        result = validate_schema(_make_records(5))
        assert "stage3_pass" in result

    def test_all_required_fields_present_passes_stage3(self):
        result = validate_schema(_make_records(5))
        assert result["stage3_pass"] is True

    def test_missing_stage3_field_fails_stage3_readiness(self):
        stage3_only = STAGE3_REQUIRED_FIELDS - {
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
        }
        if stage3_only:
            field_to_remove = next(iter(stage3_only))
            records = _make_records(5)
            for r in records:
                r.pop(field_to_remove, None)
            result = validate_schema(records)
            assert result["stage3_pass"] is False

    def test_validate_schema_reports_stage3_missing_counts(self):
        result = validate_schema(_make_records(5))
        assert "stage3_missing_counts" in result

    def test_stage3_pass_in_report_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "stage3_pass" in report["gates"]["schema"]


# ===========================================================================
# obs 1/9/19: lazy imports
# ===========================================================================


class TestLazyImportBehavior:
    def test_spacy_not_imported_at_module_top_level(self):
        """spacy must not appear as a top-level import statement."""
        source = Path("src/dataset_probe.py").read_text(encoding="utf-8")
        top_level_lines = [
            line
            for line in source.split("\n")
            if (line.startswith("import ") or line.startswith("from "))
            and not line.startswith("#")
            and "noqa" not in line
        ]
        spacy_top = [line for line in top_level_lines if "spacy" in line]
        assert not spacy_top, f"spacy must be lazily imported inside functions. Found: {spacy_top}"

    def test_autotokenizer_not_imported_at_module_top_level(self):
        """AutoTokenizer must not be a top-level import."""
        source = Path("src/dataset_probe.py").read_text(encoding="utf-8")
        top_level_lines = [
            line
            for line in source.split("\n")
            if (line.startswith("import ") or line.startswith("from ")) and not line.startswith("#")
        ]
        tok_top = [
            line for line in top_level_lines if "AutoTokenizer" in line or ("transformers" in line and "import" in line)
        ]
        assert not tok_top, f"AutoTokenizer must be lazily imported inside gate_a11. Found: {tok_top}"

    def test_schema_gate_works_without_spacy_available(self):
        """validate_schema must succeed even when spacy is unavailable at module level."""
        result = validate_schema(_make_records(5))
        assert "pass" in result

    def test_a8_gate_works_without_transformers_available(self):
        """gate_a8 must succeed without transformers."""
        result = gate_a8_text_length_distribution(_make_records(5))
        assert "pass" in result

    def test_a7_gate_works_without_spacy_available(self):
        """gate_a7 must not require spacy."""
        result = gate_a7_text_source_breakdown(_make_records(10))
        assert "pass" in result


# ===========================================================================
# obs 4: HTML_RE and BOILERPLATE_PHRASES in ProbeConfig
# ===========================================================================


class TestProbeConfigQualitySignalPatterns:
    def test_probeconfig_has_quality_signals_html_pattern(self):
        assert hasattr(ProbeConfig(), "quality_signals_html_pattern")

    def test_probeconfig_has_quality_signals_boilerplate_phrases(self):
        assert hasattr(ProbeConfig(), "quality_signals_boilerplate_phrases")

    def test_html_pattern_default_matches_html_tags(self):
        import re

        pattern = ProbeConfig().quality_signals_html_pattern
        assert re.search(pattern, "<div>some content</div>")

    def test_html_pattern_default_does_not_match_plain_text(self):
        import re

        pattern = ProbeConfig().quality_signals_html_pattern
        assert not re.search(pattern, "The court held this point.")

    def test_boilerplate_phrases_default_contains_known_phrases(self):
        phrases = ProbeConfig().quality_signals_boilerplate_phrases
        assert any("rights reserved" in p for p in phrases)
        assert any("not for publication" in p for p in phrases)

    def test_boilerplate_phrases_is_tuple_or_sequence(self):
        phrases = ProbeConfig().quality_signals_boilerplate_phrases
        assert hasattr(phrases, "__iter__")
        assert len(phrases) > 0

    def test_check_uses_config_html_pattern(self):
        """ModelQualitySignals.check must use cfg.quality_signals_html_pattern."""
        cfg = ProbeConfig(quality_signals_html_pattern=r"IMPOSSIBLE_PATTERN_12345")
        row = {"text": "<b>Bold legal text</b> Smith v. Jones, 123 F.3d 456."}
        signals = ModelQualitySignals.check(row, config=cfg)
        signal_names = [s[0] for s in signals]
        assert "html_remnants" not in signal_names

    def test_check_uses_config_boilerplate_phrases(self):
        """ModelQualitySignals.check must use cfg.quality_signals_boilerplate_phrases."""
        cfg = ProbeConfig(quality_signals_boilerplate_phrases=())
        row = {"text": "all rights reserved " * 5 + " Smith v. Jones, 123 F.3d 456. " * 10}
        signals = ModelQualitySignals.check(row, config=cfg)
        signal_names = [s[0] for s in signals]
        assert "boilerplate" not in signal_names

    def test_custom_html_pattern_used_by_check(self):
        cfg = ProbeConfig(quality_signals_html_pattern=r"CUSTOM_TAG")
        row = {"text": "CUSTOM_TAG present. Smith v. Jones, 123 F.3d 456. " * 5}
        signals = ModelQualitySignals.check(row, config=cfg)
        signal_names = [s[0] for s in signals]
        assert "html_remnants" in signal_names

    def test_custom_boilerplate_phrase_detected(self):
        cfg = ProbeConfig(quality_signals_boilerplate_phrases=("custom test phrase xyz",))
        row = {"text": "custom test phrase xyz " + "Smith v. Jones, 123 F.3d 456. " * 10}
        signals = ModelQualitySignals.check(row, config=cfg)
        signal_names = [s[0] for s in signals]
        assert "boilerplate" in signal_names

    def test_html_pattern_and_phrases_in_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        config = report["provenance"]["probe_config"]
        assert "quality_signals_html_pattern" in config
        assert "quality_signals_boilerplate_phrases" in config


# ===========================================================================
# obs 5/12/22: GateResult Pydantic model
# ===========================================================================


class TestGateResultModel:
    def test_gate_result_importable(self):
        assert GateResult is not None

    def test_gate_result_is_pydantic_model(self):
        from pydantic import BaseModel

        assert issubclass(GateResult, BaseModel)

    def test_gate_result_has_gate_field(self):
        r = GateResult(gate="A8_text_length_distribution", severity="blocking")
        assert r.gate == "A8_text_length_distribution"

    def test_gate_result_has_severity_field(self):
        r = GateResult(gate="A9", severity="advisory")
        assert r.severity == "advisory"

    def test_gate_result_model_dump_is_json_serializable(self):
        r = GateResult(gate="A8", severity="blocking")
        json.dumps(r.model_dump())

    def test_gate_result_frozen(self):
        from pydantic import ValidationError

        r = GateResult(gate="A8", severity="blocking")
        with pytest.raises((ValidationError, TypeError)):
            r.gate = "changed"  # type: ignore[misc]

    def test_gate_a8_output_compatible_with_gate_result(self):
        result = gate_a8_text_length_distribution(_make_records(5))
        gr = GateResult(**{k: v for k, v in result.items() if k in ("gate", "severity")})
        assert gr.gate == "A8_text_length_distribution"

    def test_gate_result_extra_fields_allowed(self):
        r = GateResult(gate="A8", severity="blocking", count=1000, mean=3200.0)
        assert r.model_dump()["gate"] == "A8"

    def test_all_gates_produce_gate_and_severity_keys(self):
        results = [
            gate_a7_text_source_breakdown(_make_records(10)),
            gate_a8_text_length_distribution(_make_records(10)),
            gate_a9_citation_count_distribution(_make_records(10)),
            gate_a12_citation_anchor_survival(_make_records(10)),
            gate_b6_text_entropy_distribution(_make_records(10)),
        ]
        for r in results:
            assert "gate" in r
            assert "severity" in r


# ===========================================================================
# obs 10: GATE_REGISTRY
# ===========================================================================


class TestGateRegistry:
    def test_gate_registry_exportable(self):
        assert GATE_REGISTRY is not None

    def test_gate_registry_is_sequence(self):
        assert hasattr(GATE_REGISTRY, "__iter__")
        assert len(GATE_REGISTRY) > 0

    def test_gate_registry_contains_all_core_gates(self):
        names = [entry["name"] for entry in GATE_REGISTRY]
        for gate_name in ("A7", "A8", "A9", "A12", "B6"):
            assert gate_name in names

    def test_gate_registry_contains_a11_and_a13(self):
        names = [entry["name"] for entry in GATE_REGISTRY]
        assert "A11" in names
        assert "A13" in names

    def test_each_entry_has_name_key(self):
        for entry in GATE_REGISTRY:
            assert "name" in entry

    def test_each_entry_has_callable_fn(self):
        for entry in GATE_REGISTRY:
            assert "fn" in entry
            assert callable(entry["fn"])

    def test_each_entry_has_severity(self):
        for entry in GATE_REGISTRY:
            assert "severity" in entry
            assert entry["severity"] in ("blocking", "advisory")

    def test_a7_entry_is_blocking(self):
        a7 = next(e for e in GATE_REGISTRY if e["name"] == "A7")
        assert a7["severity"] == "blocking"

    def test_a9_entry_is_advisory(self):
        a9 = next(e for e in GATE_REGISTRY if e["name"] == "A9")
        assert a9["severity"] == "advisory"

    def test_b6_entry_is_advisory(self):
        b6 = next(e for e in GATE_REGISTRY if e["name"] == "B6")
        assert b6["severity"] == "advisory"

    def test_registry_fn_callable_with_records_and_config(self):
        records = _make_records(5)
        cfg = ProbeConfig()
        skip_gates = {"A11", "A13"}
        for entry in GATE_REGISTRY:
            if entry["name"] in skip_gates:
                continue
            result = entry["fn"](records, cfg)
            assert "gate" in result


# ===========================================================================
# EXISTING TESTS
# ===========================================================================


class TestImportStyle:
    def test_spacy_not_imported_as_alias(self):
        source = Path("src/dataset_probe.py").read_text(encoding="utf-8")
        assert "import spacy as spacy" not in source


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


class TestDataclassesReplace:
    def test_main_uses_dataclasses_replace_not_double_construction(self):
        source = Path("src/dataset_probe.py").read_text(encoding="utf-8")
        assert "ProbeConfig().a11_generative_model" not in source

    def test_dataclasses_replace_pattern_is_present(self):
        source = Path("src/dataset_probe.py").read_text(encoding="utf-8")
        assert "dataclasses.replace" in source

    def test_skip_generative_tokenizer_still_sets_empty_string(self, sample_shard_dir, tmp_path):
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


class TestModelQualitySignalsSummarizeConfig:
    def test_summarize_accepts_config_parameter(self):
        sig = inspect.signature(ModelQualitySignals.summarize)
        assert "config" in sig.parameters

    def test_summarize_forwards_config_to_check(self):
        suffix = " Smith v. Jones, 123 F.3d 456"
        big_text = ("word " * 110) + suffix
        cfg_tiny = ProbeConfig(quality_signals_text_cap_chars=50)
        records = [{"text": big_text}]
        result = ModelQualitySignals.summarize(records, sample_n=1, config=cfg_tiny)
        assert result["signal_counts"].get("no_citations", 0) >= 1

    def test_summarize_without_config_uses_default(self):
        records = _make_records(5)
        result = ModelQualitySignals.summarize(records, sample_n=5)
        assert "pct_clean" in result

    def test_run_probe_passes_config_to_summarize(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "quality_signals" in report


class TestValidateSchemaHelpers:
    def test_check_presence_helper_exported(self):
        assert callable(_check_presence)

    def test_check_types_and_ranges_helper_exported(self):
        assert callable(_check_types_and_ranges)

    def test_check_vocabulary_helper_exported(self):
        assert callable(_check_vocabulary)

    def test_check_consistency_helper_exported(self):
        assert callable(_check_consistency)

    def test_check_documented_coverage_helper_exported(self):
        assert callable(_check_documented_coverage)

    def test_check_presence_returns_missing_counts(self):
        records = _make_records(3)
        for r in records:
            del r["court_id"]
        result = _check_presence(records)
        assert result.get("court_id", 0) > 0

    def test_check_presence_empty_for_complete_records(self):
        result = _check_presence(_make_records(5))
        assert all(v == 0 for v in result.values())

    def test_check_types_and_ranges_returns_type_errors(self):
        records = _make_records(3, text_entropy="bad")
        type_errors, range_errors = _check_types_and_ranges(records)
        assert "text_entropy" in type_errors

    def test_check_vocabulary_returns_vocab_errors(self):
        records = _make_records(3, text_source="GARBAGE_FORMAT")
        result = _check_vocabulary(records)
        assert "text_source" in result

    def test_check_consistency_returns_consistency_errors(self):
        records = _make_records(3)
        for r in records:
            r["text"] = "x"
            r["text_length"] = 999_999
        result = _check_consistency(records)
        assert result.get("text_length_consistency", 0) > 0

    def test_check_documented_coverage_returns_missing_fields(self):
        records = _make_records(3)
        for r in records:
            del r["cluster_id"]
        result = _check_documented_coverage(records)
        assert result.get("cluster_id", 0) > 0

    def test_validate_schema_still_passes_complete_records(self):
        assert validate_schema(_make_records(5))["pass"] is True

    def test_validate_schema_still_fails_missing_required_field(self):
        records = _make_records(5)
        for r in records:
            del r["court_id"]
        assert validate_schema(records)["pass"] is False


class TestGateA11GenerativeSeverity:
    """
    AutoTokenizer is now lazily imported inside gate_a11_tokenizer_chunk_count.
    Patch target is 'transformers.AutoTokenizer' not 'src.dataset_probe.AutoTokenizer'.
    Tests that inject tokenizer= directly bypass the import entirely.
    """

    def test_generative_token_check_has_severity_advisory(self):
        """Inject both encoder and generative tokenizers directly."""
        mock_enc = MagicMock()
        mock_enc.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_enc.name_or_path = "fake-encoder"

        mock_gen = MagicMock()
        mock_gen.side_effect = lambda text, **kw: {"input_ids": list(range(500))}

        with patch("transformers.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_gen
            cfg = ProbeConfig(a11_generative_model="fake-model")
            result = gate_a11_tokenizer_chunk_count(_make_records(5), config=cfg, tokenizer=mock_enc)
        assert "generative_token_check" in result
        assert result["generative_token_check"].get("severity") == "advisory"

    def test_generative_token_check_error_also_has_severity_advisory(self):
        """When generative tokenizer load fails, severity must still be advisory."""
        mock_enc = MagicMock()
        mock_enc.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_enc.name_or_path = "fake-encoder"

        with patch("transformers.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.side_effect = OSError("not found")
            cfg = ProbeConfig(a11_generative_model="fake-model")
            result = gate_a11_tokenizer_chunk_count(_make_records(5), config=cfg, tokenizer=mock_enc)
        assert result["generative_token_check"].get("severity") == "advisory"

    def test_a11_gate_level_severity_still_blocking(self):
        """Top-level A11 severity must remain blocking regardless."""
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_tok.name_or_path = "fake-model"
        result = gate_a11_tokenizer_chunk_count(
            _make_records(5),
            config=ProbeConfig(a11_generative_model=""),
            tokenizer=mock_tok,
        )
        assert result["severity"] == "blocking"


class TestGateA13Helpers:
    def test_load_spacy_nlp_exported(self):
        assert callable(_load_spacy_nlp)

    def test_compute_sentence_counts_exported(self):
        assert callable(_compute_sentence_counts)

    def test_load_spacy_nlp_returns_nlp_and_version(self):
        cfg = ProbeConfig()
        nlp_obj, version = _load_spacy_nlp(cfg, nlp=None)
        assert nlp_obj is not None
        assert isinstance(version, str)

    def test_load_spacy_nlp_returns_injected_nlp_unchanged(self):
        mock_nlp = MagicMock()
        nlp_obj, version = _load_spacy_nlp(ProbeConfig(), nlp=mock_nlp)
        assert nlp_obj is mock_nlp
        assert version == "injected"

    def test_compute_sentence_counts_returns_counts_and_below(self):
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
        long_text = "The court held this point clearly. " * 100
        result = gate_a13_sentence_density(_make_records(5, text=long_text))
        assert "pass" in result


class TestLogReportToWandbSingleCall:
    def test_log_report_to_wandb_calls_wandb_log_once(self, sample_shard_dir, tmp_path):
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
            assert mock_wandb.log.call_count == 1
        finally:
            dp.wandb = original

    def test_single_wandb_log_contains_all_required_keys(self, sample_shard_dir, tmp_path):
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


class TestRunProbeHelpers:
    def test_prepare_samples_exported(self):
        assert callable(_prepare_samples)

    def test_load_spacy_pipeline_exported(self):
        assert callable(_load_spacy_pipeline)

    def test_build_provenance_exported(self):
        assert callable(_build_provenance)

    def test_summarize_gates_exported(self):
        assert callable(_summarize_gates)

    def test_prepare_samples_returns_three_lists(self):
        records = _make_records(20)
        cfg = ProbeConfig()
        a11, a12, a13 = _prepare_samples(records, cfg, seed=0)
        assert len(a11) <= cfg.a11_subsample_n
        assert len(a12) <= cfg.a12_subsample_n
        assert len(a13) <= cfg.a13_subsample_n

    def test_prepare_samples_a13_pre_filters_by_text_length(self):
        long = _make_records(10)
        short = _make_records(10, text="Short.")
        records = long + short
        cfg = ProbeConfig()
        _, _, a13 = _prepare_samples(records, cfg, seed=0)
        for r in a13:
            assert _safe_int(r.get("text_length", 0)) >= cfg.min_text_length

    def test_load_spacy_pipeline_returns_tuple(self):
        cfg = ProbeConfig()
        nlp, spacy_ver, model_ver = _load_spacy_pipeline(cfg, skip_spacy=True)
        assert nlp is None
        assert isinstance(spacy_ver, str)
        assert isinstance(model_ver, str)

    def test_build_provenance_returns_required_keys(self):
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
        gates = {
            "schema": {"pass": True, "severity": "blocking"},
            "A9": {"pass": False, "severity": "advisory"},
        }
        summary = _summarize_gates(gates)
        assert summary["all_passed"] is True

    def test_run_probe_still_works_after_refactor(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert report["summary"]["all_passed"] is True


class TestRunProbeNoInlineWandb:
    def test_run_probe_never_calls_wandb_log_directly(self, sample_shard_dir, tmp_path):
        import src.dataset_probe as dp

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
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
            mock_wandb.log.assert_not_called()
        finally:
            dp.wandb = original

    def test_run_probe_warning_printed_when_wandb_run_none(self, sample_shard_dir, tmp_path, capsys):
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


class TestModuleLevelConstants:
    def test_provisional_min_text_length_is_integer_literal(self):
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
        source = Path("src/dataset_probe.py").read_text(encoding="utf-8")
        assert "PROVISIONAL_MIN_TEXT_LENGTH = 1500" in source
        assert "CHUNK_SIZE_SUBWORDS = 1024" in source
        assert "CHUNK_OVERLAP_SUBWORDS = 128" in source
        assert "MIN_SENTENCE_COUNT = 20" in source


class TestGateSampleN:
    def test_gate_a7_reports_sample_n(self):
        result = gate_a7_text_source_breakdown(_make_records(10))
        assert "sample_n" in result
        assert result["sample_n"] == 10

    def test_gate_a8_reports_sample_n(self):
        result = gate_a8_text_length_distribution(_make_records(10))
        assert "sample_n" in result

    def test_gate_a8_sample_n_reflects_valid_records_only(self):
        valid = _make_records(7, text_length=5000)
        invalid = _make_records(3, text_length="N/A")
        result = gate_a8_text_length_distribution(valid + invalid)
        assert result["sample_n"] == result["count"] == 7

    def test_gate_a9_reports_sample_n(self):
        result = gate_a9_citation_count_distribution(_make_records(10))
        assert "sample_n" in result
        assert result["sample_n"] == result["count"]

    def test_gate_b6_reports_sample_n(self):
        result = gate_b6_text_entropy_distribution(_make_records(10))
        assert "sample_n" in result
        assert result["sample_n"] == 10


class TestRecordsTextCapped:
    def test_gate_a12_reports_records_text_capped(self):
        long_text = "word " * 20_000
        records = _make_records(5, text=long_text)
        result = gate_a12_citation_anchor_survival(records)
        assert "records_text_capped" in result

    def test_gate_a12_records_text_capped_zero_for_short_text(self):
        result = gate_a12_citation_anchor_survival(_make_records(5))
        assert result["records_text_capped"] == 0

    def test_gate_a12_records_text_capped_counts_truncated(self):
        long_text = "x " * 30_000
        short_text = "x " * 10
        long_recs = _make_records(3, text=long_text)
        short_recs = _make_records(4, text=short_text)
        cfg = ProbeConfig(a12_text_cap_chars=50_000)
        result = gate_a12_citation_anchor_survival(long_recs + short_recs, config=cfg)
        assert result["records_text_capped"] == 3

    def test_quality_signals_summarize_reports_records_text_capped(self):
        long_text = "word " * 20_000
        records = [{"text": long_text}] * 3
        result = ModelQualitySignals.summarize(records, sample_n=3)
        assert "records_text_capped" in result

    def test_quality_signals_records_text_capped_zero_for_short_text(self):
        records = _make_records(5)
        result = ModelQualitySignals.summarize(records, sample_n=5)
        assert result["records_text_capped"] == 0


class TestGateA8RobustParsing:
    def test_a8_does_not_crash_on_string_text_length(self):
        result = gate_a8_text_length_distribution(_make_records(5, text_length="N/A"))
        assert "pass" in result

    def test_a8_excludes_invalid_text_length_from_distribution(self):
        valid = _make_records(8, text_length=5000)
        invalid = _make_records(2, text_length="N/A")
        result = gate_a8_text_length_distribution(valid + invalid)
        assert result["count"] == 8

    def test_a8_all_invalid_returns_structured_failure(self):
        result = gate_a8_text_length_distribution(_make_records(5, text_length="N/A"))
        assert result["pass"] is False

    def test_a8_parse_errors_reported(self):
        valid = _make_records(6, text_length=5000)
        invalid = _make_records(4, text_length="N/A")
        result = gate_a8_text_length_distribution(valid + invalid)
        assert result["text_length_parse_errors"] == 4


class TestGateA9RobustParsing:
    def test_a9_does_not_crash_on_string_citation_count(self):
        result = gate_a9_citation_count_distribution(_make_records(5, citation_count="N/A"))
        assert "pass" in result

    def test_a9_excludes_invalid_citation_count_from_distribution(self):
        valid = _make_records(8, citation_count=5)
        invalid = _make_records(2, citation_count="N/A")
        result = gate_a9_citation_count_distribution(valid + invalid)
        assert result["count"] == 8

    def test_a9_malformed_excluded_from_zero_citation_count(self):
        valid_zero = _make_records(3, citation_count=0)
        valid_nonzero = _make_records(5, citation_count=5)
        malformed = _make_records(2, citation_count="N/A")
        result = gate_a9_citation_count_distribution(valid_zero + valid_nonzero + malformed)
        assert result["zero_citation_count"] == 3

    def test_a9_parse_errors_reported(self):
        valid = _make_records(6, citation_count=5)
        invalid = _make_records(4, citation_count="N/A")
        result = gate_a9_citation_count_distribution(valid + invalid)
        assert result["citation_count_parse_errors"] == 4


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


class TestLegalCitationReOCR:
    def test_regex_matches_clean_federal_reporter(self):
        assert _LEGAL_CITATION_RE.search("123 F.3d 456")

    def test_regex_matches_ocr_space_between_f_and_3d(self):
        assert _LEGAL_CITATION_RE.search("123 F. 3d 456")

    def test_regex_matches_ocr_space_before_period(self):
        assert _LEGAL_CITATION_RE.search("123 F .3d 456")

    def test_regex_still_rejects_plain_prose(self):
        assert not _LEGAL_CITATION_RE.search("The defendant argued the motion.")


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
        assert json.loads(out.read_text())["provenance"].get("full_scan") is True

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
        assert json.loads(out.read_text())["provenance"].get("full_scan") is False

    def test_full_scan_run_probe_accepts_full_scan_param(self):
        sig = inspect.signature(run_probe)
        assert "full_scan" in sig.parameters


class TestWandbRunIsNoneWarning:
    def test_warning_printed_when_log_to_wandb_true_and_run_is_none(self, sample_shard_dir, tmp_path, capsys):
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
            assert "wandb" in captured.out.lower() or "log" in captured.out.lower()
        finally:
            dp.wandb = original


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


class TestValidateSchemaTextLengthConsistency:
    def test_probe_config_has_text_length_tolerance(self):
        assert hasattr(ProbeConfig(), "text_length_consistency_tolerance")

    def test_text_length_consistency_tolerance_default(self):
        tol = ProbeConfig().text_length_consistency_tolerance
        assert isinstance(tol, int) and tol > 0

    def test_fails_when_text_length_far_exceeds_actual_text(self):
        records = _make_records(3)
        for r in records:
            r["text"] = "Short."
            r["text_length"] = 99_999
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

    def test_validate_schema_accepts_config_parameter(self):
        assert "config" in inspect.signature(validate_schema).parameters


class TestWandbLoggingBranch:
    def test_wandb_log_not_called_when_log_to_wandb_false(self, sample_shard_dir, tmp_path):
        import src.dataset_probe as dp

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        original = dp.wandb
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
            dp.wandb = original

    def test_wandb_log_not_called_when_wandb_run_is_none(self, sample_shard_dir, tmp_path):
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
            mock_wandb.log.assert_not_called()
        finally:
            dp.wandb = original

    def test_wandb_log_not_called_when_wandb_is_none(self, sample_shard_dir, tmp_path):
        import src.dataset_probe as dp

        original = dp.wandb
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
            dp.wandb = original


class TestA12TextCap:
    def test_probe_config_has_a12_text_cap_chars(self):
        assert hasattr(ProbeConfig(), "a12_text_cap_chars")

    def test_a12_text_cap_chars_default(self):
        assert ProbeConfig().a12_text_cap_chars == 50_000


class TestModelQualitySignalsTextCap:
    def test_probe_config_has_quality_signals_text_cap_chars(self):
        assert hasattr(ProbeConfig(), "quality_signals_text_cap_chars")

    def test_model_quality_signals_check_accepts_config(self):
        assert "config" in inspect.signature(ModelQualitySignals.check).parameters


class TestProbeConfigA7KnownFormatsTyping:
    def test_a7_known_formats_default_values_are_strings(self):
        for item in ProbeConfig().a7_known_formats:
            assert isinstance(item, str)

    def test_a7_known_formats_is_frozenset(self):
        assert isinstance(ProbeConfig().a7_known_formats, frozenset)


class TestPercentileDocstring:
    def test_percentile_has_docstring(self):
        assert _percentile.__doc__ is not None

    def test_percentile_docstring_does_not_claim_numpy_consistency(self):
        assert "numpy" not in _percentile.__doc__.lower()


class TestValidateSchemaCitationDensity:
    def test_fails_negative_citation_density(self):
        assert validate_schema(_make_records(3, citation_density=-0.5))["pass"] is False

    def test_fails_non_numeric_citation_density(self):
        assert validate_schema(_make_records(3, citation_density="high"))["pass"] is False

    def test_passes_zero_citation_density(self):
        assert (
            validate_schema(_make_records(5, citation_density=0.0)).get("range_errors", {}).get("citation_density")
            is None
        )


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

        original = dp.wandb
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
            dp.wandb = original


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


class TestMinRequiredFields:
    def test_min_required_fields_has_11_fields(self):
        from src.dataset_probe import MIN_REQUIRED_FIELDS

        assert len(MIN_REQUIRED_FIELDS) == 11

    def test_required_fields_alias_still_works(self):
        from src.dataset_probe import MIN_REQUIRED_FIELDS

        assert REQUIRED_FIELDS == MIN_REQUIRED_FIELDS


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


class TestGateSeverity:
    def test_gate_a7_has_severity_blocking(self):
        assert gate_a7_text_source_breakdown(_make_records(10)).get("severity") == "blocking"

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


class TestValidateSchemaExtendedTypes:
    def test_fails_non_numeric_text_entropy(self):
        assert validate_schema(_make_records(3, text_entropy="bad"))["pass"] is False

    def test_fails_negative_text_entropy(self):
        assert validate_schema(_make_records(3, text_entropy=-1.0))["pass"] is False

    def test_fails_non_integer_paragraph_count(self):
        assert validate_schema(_make_records(3, paragraph_count="five"))["pass"] is False

    def test_fails_negative_token_count(self):
        assert validate_schema(_make_records(3, token_count=-1))["pass"] is False


class TestSpaCySingleLoad:
    """
    spacy is now lazily imported inside _load_spacy_pipeline.
    Patch target is 'spacy.load' directly, not 'src.dataset_probe.spacy'.
    """

    def test_spacy_loaded_once_in_run_probe(self, sample_shard_dir, tmp_path):
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ["sentencizer"]
        mock_nlp.meta = {"version": "3.8.0"}
        with patch("spacy.load", return_value=mock_nlp) as mock_load:
            run_probe(
                data_dir=sample_shard_dir,
                subset=20,
                output=tmp_path / "r.json",
                skip_tokenizer=True,
                skip_spacy=False,
            )
            assert mock_load.call_count <= 1


class TestProbeConfig:
    def test_is_frozen(self):
        cfg = ProbeConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.min_text_length = 999  # type: ignore[misc]

    def test_default_min_text_length(self):
        assert ProbeConfig().min_text_length == PROVISIONAL_MIN_TEXT_LENGTH

    def test_is_json_serializable(self):
        json.dumps(_probe_config_to_dict(ProbeConfig()))

    def test_has_a11_generative_model(self):
        assert hasattr(ProbeConfig(), "a11_generative_model")

    def test_all_new_fields_in_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        config = report["provenance"]["probe_config"]
        for key in ("a13_text_cap_chars", "a11_subsample_n", "a12_subsample_n"):
            assert key in config


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


class TestGateA12CitationFieldCrossValidation:
    def test_a12_reports_citation_field_vs_regex(self):
        assert "citation_field_vs_regex" in gate_a12_citation_anchor_survival(_make_records(10))

    def test_a12_detects_field_nonzero_but_regex_zero(self):
        records = _make_records(10, citation_count=5, text="No legal anchors here at all.")
        result = gate_a12_citation_anchor_survival(records)
        assert result["citation_field_vs_regex"]["field_nonzero_regex_zero_count"] == 10


class TestGetTextHelper:
    def test_get_text_returns_text_field(self):
        assert _get_text({"text": "hello"}) == "hello"

    def test_get_text_returns_empty_string_when_missing(self):
        assert _get_text({}) == ""

    def test_get_text_returns_empty_string_when_none(self):
        assert _get_text({"text": None}) == ""


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


class TestSharedCitationRegex:
    def test_shared_regex_catches_federal_reporter(self):
        assert _LEGAL_CITATION_RE.search("123 F.3d 456")

    def test_shared_regex_catches_case_name_citation(self):
        assert _LEGAL_CITATION_RE.search("Smith v. Jones")

    def test_shared_regex_catches_scotus_reporter(self):
        assert _LEGAL_CITATION_RE.search("347 U.S. 483")


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


class TestSampleRecords:
    def test_returns_correct_count(self, sample_shard_dir):
        assert len(sample_records(sample_shard_dir, 10)) == 10

    def test_deterministic_with_seed(self, sample_shard_dir):
        r1 = sample_records(sample_shard_dir, 10, seed=0)
        r2 = sample_records(sample_shard_dir, 10, seed=0)
        assert [r["id"] for r in r1] == [r["id"] for r in r2]


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


class TestGateA7:
    def test_gate_key(self):
        assert gate_a7_text_source_breakdown(_make_records(10))["gate"] == "A7_text_source_breakdown"

    def test_pass_when_known_formats_dominant(self):
        records = _make_records(85, text_source="plain_text") + _make_records(15, text_source="html_with_citations")
        assert gate_a7_text_source_breakdown(records)["pass"] is True

    def test_empty_records_handled(self):
        assert "pass" in gate_a7_text_source_breakdown([])


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


class TestGateA9:
    def test_passes_at_19_99_pct(self):
        records = _make_records(8001, citation_count=5) + _make_records(1999, citation_count=0)
        assert gate_a9_citation_count_distribution(records)["pass"] is True

    def test_empty_records_handled(self):
        assert "pass" in gate_a9_citation_count_distribution([])

    def test_note_clarifies_advisory_role(self):
        assert "advisory" in gate_a9_citation_count_distribution(_make_records(10))["note"].lower()


class TestGateA11:
    """
    AutoTokenizer is lazily imported. Patch via 'transformers.AutoTokenizer'
    or inject tokenizer= directly (preferred — no patch needed).
    """

    def test_pass_when_median_chunks_gte_2(self):
        mock_tok = MagicMock()
        mock_tok.side_effect = lambda text, **kw: {"input_ids": list(range(3000))}
        mock_tok.name_or_path = "fake"
        assert (
            gate_a11_tokenizer_chunk_count(
                _make_records(10),
                config=ProbeConfig(a11_generative_model=""),
                tokenizer=mock_tok,
            )["pass"]
            is True
        )

    def test_fail_on_tokenizer_load_error(self):
        with patch("transformers.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.side_effect = OSError("not found")
            r = gate_a11_tokenizer_chunk_count(_make_records(5), config=ProbeConfig(a11_generative_model=""))
        assert r["pass"] is False

    def test_empty_records_handled(self):
        assert "pass" in gate_a11_tokenizer_chunk_count([])

    def test_gate_a11_accepts_tokenizer_argument(self):
        assert "tokenizer" in inspect.signature(gate_a11_tokenizer_chunk_count).parameters


class TestGateA12:
    def test_pass_when_most_have_anchors(self):
        assert gate_a12_citation_anchor_survival(_make_records(100))["pass"] is True

    def test_fail_when_few_have_anchors(self):
        assert gate_a12_citation_anchor_survival(_make_records(100, text="No citations here."))["pass"] is False

    def test_empty_records_handled(self):
        assert "pass" in gate_a12_citation_anchor_survival([])


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
        run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=out,
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert out.exists()


class TestProvenance:
    def test_provenance_has_required_keys(self, sample_shard_dir, tmp_path):
        prov = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )["provenance"]
        for key in (
            "timestamp",
            "spacy_model_version",
            "probe_config",
            "probe_version",
            "git_sha",
            "full_scan",
        ):
            assert key in prov


class TestProbeVersion:
    def test_probe_version_constant_exported(self):
        assert isinstance(PROBE_VERSION, str) and len(PROBE_VERSION) >= 3


class TestCourtListenerDatasetProbe:
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


class TestShardAuditTotalRecordsDecoded:
    def test_total_records_decoded_counts_valid_records(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id":"1","text":"a"}\n{"id":"2","text":"b"}\nBAD\n\n')
        assert iter_shards_with_audit(tmp_path)["total_records_decoded"] == 2


class TestAnnotationResolution:
    def test_gate_a7_annotations_resolve(self):
        assert "records" in get_type_hints(gate_a7_text_source_breakdown)

    def test_gate_a8_annotations_resolve(self):
        assert "records" in get_type_hints(gate_a8_text_length_distribution)

    def test_run_probe_annotations_resolve(self):
        assert "data_dir" in get_type_hints(run_probe)


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
