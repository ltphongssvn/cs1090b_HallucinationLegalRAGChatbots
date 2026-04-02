"""
Contract tests for scripts/audit_jsonl_nan.py
RED phase — all tests must FAIL before implementation exists.
"""

from __future__ import annotations

import json

import pytest

from scripts.audit_jsonl_nan import (
    _NAN_REPAIR_PATTERN,
    _STRING_NAN_VALUES,
    DatasetHealth,
    ShardHealth,
    _has_nan,
    _nan_fields,
    audit_dataset,
    audit_shard,
    repair_shard,
)

# ---------------------------------------------------------------------------
# _has_nan — unit contracts
# ---------------------------------------------------------------------------


class TestHasNan:
    def test_float_nan_detected(self):
        assert _has_nan(float("nan")) is True

    def test_float_inf_detected(self):
        assert _has_nan(float("inf")) is True

    def test_float_neg_inf_detected(self):
        assert _has_nan(float("-inf")) is True

    def test_normal_float_not_flagged(self):
        assert _has_nan(3.14) is False

    def test_string_nan_detected(self):
        assert _has_nan("NaN") is True

    def test_string_nan_lowercase_detected(self):
        assert _has_nan("nan") is True

    def test_string_infinity_detected(self):
        assert _has_nan("Infinity") is True

    def test_string_neg_infinity_detected(self):
        assert _has_nan("-Infinity") is True

    def test_normal_string_not_flagged(self):
        assert _has_nan("Smith v. Jones") is False

    def test_none_not_flagged(self):
        assert _has_nan(None) is False

    def test_integer_not_flagged(self):
        assert _has_nan(42) is False

    def test_nested_dict_nan_detected(self):
        assert _has_nan({"a": {"b": float("nan")}}) is True

    def test_nested_list_nan_detected(self):
        assert _has_nan([1, 2, float("nan")]) is True

    def test_clean_dict_not_flagged(self):
        assert _has_nan({"case_name": "Smith v. Jones", "court_id": "ca9"}) is False


# ---------------------------------------------------------------------------
# _nan_fields — unit contracts
# ---------------------------------------------------------------------------


class TestNanFields:
    def test_returns_field_names_with_nan(self):
        obj = {"case_name": float("nan"), "court_id": "ca9"}
        assert _nan_fields(obj) == ["case_name"]

    def test_returns_empty_for_clean_record(self):
        obj = {"case_name": "Smith v. Jones", "court_id": "ca9"}
        assert _nan_fields(obj) == []

    def test_detects_multiple_nan_fields(self):
        obj = {"case_name": float("nan"), "raw_text": float("nan"), "id": "1"}
        fields = _nan_fields(obj)
        assert "case_name" in fields
        assert "raw_text" in fields


# ---------------------------------------------------------------------------
# _STRING_NAN_VALUES — constant contract
# ---------------------------------------------------------------------------


class TestStringNanValues:
    def test_nan_in_set(self):
        assert "NaN" in _STRING_NAN_VALUES

    def test_nan_lowercase_in_set(self):
        assert "nan" in _STRING_NAN_VALUES

    def test_infinity_in_set(self):
        assert "Infinity" in _STRING_NAN_VALUES

    def test_neg_infinity_in_set(self):
        assert "-Infinity" in _STRING_NAN_VALUES


# ---------------------------------------------------------------------------
# _NAN_REPAIR_PATTERN — regex contract
# ---------------------------------------------------------------------------


class TestNanRepairPattern:
    def test_matches_bare_nan(self):
        assert _NAN_REPAIR_PATTERN.search('"case_name": NaN,')

    def test_matches_bare_infinity(self):
        assert _NAN_REPAIR_PATTERN.search('"score": Infinity,')

    def test_does_not_match_quoted_nan(self):
        assert not _NAN_REPAIR_PATTERN.search('"case_name": "NaN"')

    def test_repair_replaces_nan_with_null(self):
        line = '{"case_name": NaN, "id": "1"}'
        repaired = _NAN_REPAIR_PATTERN.sub("null", line)
        assert json.loads(repaired)["case_name"] is None


# ---------------------------------------------------------------------------
# audit_shard — integration contracts (uses tmp_path fixtures)
# ---------------------------------------------------------------------------


class TestAuditShard:
    def test_clean_shard_reports_zero_nan_lines(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        records = [{"id": str(i), "case_name": "Smith v. Jones", "text_entropy": 4.2} for i in range(5)]
        shard.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        health = audit_shard(shard)
        assert health.nan_lines == 0
        assert health.total_lines == 5

    def test_shard_with_float_nan_detected(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        lines = [
            json.dumps({"id": "0", "case_name": "Smith v. Jones"}),
            '{"id": "1", "case_name": NaN}',
            json.dumps({"id": "2", "case_name": "Jones v. Smith"}),
        ]
        shard.write_text("\n".join(lines), encoding="utf-8")
        health = audit_shard(shard)
        assert health.nan_lines >= 1

    def test_shard_nan_fields_histogram(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        good = json.dumps({"id": "0", "case_name": "Smith v. Jones"})
        bad = '{"id": "1", "case_name": NaN}'
        shard.write_text(f"{good}\n{bad}\n", encoding="utf-8")
        health = audit_shard(shard)
        assert "case_name" in health.nan_fields

    def test_empty_shard_returns_zero(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text("", encoding="utf-8")
        health = audit_shard(shard)
        assert health.total_lines == 0
        assert health.nan_lines == 0

    def test_returns_shard_health_instance(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text(json.dumps({"id": "0"}) + "\n", encoding="utf-8")
        assert isinstance(audit_shard(shard), ShardHealth)


# ---------------------------------------------------------------------------
# repair_shard — contracts
# ---------------------------------------------------------------------------


class TestRepairShard:
    def test_dry_run_does_not_write(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        original = '{"id": "0", "case_name": NaN}\n'
        shard.write_text(original, encoding="utf-8")
        repair_shard(shard, dry_run=True)
        assert shard.read_text(encoding="utf-8") == original

    def test_repair_replaces_nan_with_null(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": NaN}\n', encoding="utf-8")
        _, repaired = repair_shard(shard, dry_run=False)
        assert repaired == 1
        obj = json.loads(shard.read_text(encoding="utf-8"))
        assert obj["case_name"] is None

    def test_repair_creates_backup(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": NaN}\n', encoding="utf-8")
        repair_shard(shard, dry_run=False)
        assert shard.with_suffix(".jsonl.bak").exists()

    def test_clean_shard_not_modified(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        original = '{"id": "0", "case_name": "Smith v. Jones"}\n'
        shard.write_text(original, encoding="utf-8")
        _, repaired = repair_shard(shard, dry_run=False)
        assert repaired == 0
        assert shard.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# DatasetHealth — property contracts
# ---------------------------------------------------------------------------


class TestDatasetHealth:
    def _make(self, total, nan, nan_shards, total_shards, fields, contaminated):
        return DatasetHealth(
            total_lines=total,
            nan_lines=nan,
            nan_shards=nan_shards,
            total_shards=total_shards,
            nan_fields=fields,
            contaminated_shards=contaminated,
        )

    def test_clean_pct_zero_lines_returns_zero(self):
        h = self._make(0, 0, 0, 0, {}, [])
        assert h.clean_pct == 0.0

    def test_clean_pct_all_clean(self):
        h = self._make(100, 0, 0, 5, {}, [])
        assert h.clean_pct == 100.0

    def test_clean_pct_partial(self):
        h = self._make(100, 10, 1, 5, {"case_name": 10}, ["s.jsonl"])
        assert abs(h.clean_pct - 90.0) < 0.01

    def test_gate_verdict_clean(self):
        h = self._make(100, 0, 0, 5, {}, [])
        assert h.gate_verdict() == "CLEAN"

    def test_gate_verdict_repairable(self):
        h = self._make(100, 5, 1, 5, {"case_name": 5}, ["s.jsonl"])
        assert "REPAIRABLE" in h.gate_verdict()

    def test_gate_verdict_hard_failure(self):
        h = self._make(100, 5, 1, 5, {"text": 5}, ["s.jsonl"])
        assert "HARD_FAILURE" in h.gate_verdict()


# ---------------------------------------------------------------------------
# audit_dataset — integration contracts
# ---------------------------------------------------------------------------


class TestAuditDataset:
    def test_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            audit_dataset(tmp_path / "nonexistent")

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            audit_dataset(tmp_path)

    def test_returns_dataset_health(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text(json.dumps({"id": "0", "case_name": "Smith v. Jones"}) + "\n")
        result = audit_dataset(tmp_path)
        assert isinstance(result, DatasetHealth)

    def test_clean_dataset_reports_zero_nan(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        records = [{"id": str(i), "case_name": "Smith v. Jones"} for i in range(10)]
        shard.write_text("\n".join(json.dumps(r) for r in records))
        result = audit_dataset(tmp_path)
        assert result.nan_lines == 0
        assert result.gate_verdict() == "CLEAN"


# ---------------------------------------------------------------------------
# RED: refactor contracts — must FAIL before implementation
# ---------------------------------------------------------------------------


class TestOutputFormatters:
    """#1 — main() split: output helpers must be importable as top-level functions."""

    def test_write_csv_importable(self):
        from scripts.audit_jsonl_nan import _write_csv

        assert callable(_write_csv)

    def test_emit_json_importable(self):
        from scripts.audit_jsonl_nan import _emit_json

        assert callable(_emit_json)

    def test_emit_text_importable(self):
        from scripts.audit_jsonl_nan import _emit_text

        assert callable(_emit_text)

    def test_write_csv_produces_file(self, tmp_path):
        from scripts.audit_jsonl_nan import DatasetHealth, _write_csv

        health = DatasetHealth(100, 5, 1, 5, {"case_name": 5}, ["s.jsonl"])
        out = tmp_path / "out.csv"
        _write_csv(health, out)
        assert out.exists()
        content = out.read_text()
        assert "case_name" in content

    def test_emit_json_outputs_valid_json(self, capsys):
        from scripts.audit_jsonl_nan import DatasetHealth, _emit_json

        health = DatasetHealth(100, 0, 0, 5, {}, [])
        _emit_json(health)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "gate_verdict" in parsed

    def test_emit_text_outputs_verdict(self, capsys):
        from scripts.audit_jsonl_nan import DatasetHealth, _emit_text

        health = DatasetHealth(100, 0, 0, 5, {}, [])
        _emit_text(health, emit_shard_ids=False)
        captured = capsys.readouterr()
        assert "verdict" in captured.out


class TestRepairShardStreaming:
    """#3 — tmp file must not persist after repair_shard completes."""

    def test_tmp_file_cleaned_up_after_repair(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": NaN}\n', encoding="utf-8")
        repair_shard(shard, dry_run=False)
        assert not (tmp_path / "s.jsonl.tmp").exists()

    def test_tmp_file_cleaned_up_on_dry_run(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": NaN}\n', encoding="utf-8")
        repair_shard(shard, dry_run=True)
        assert not (tmp_path / "s.jsonl.tmp").exists()


class TestLogging:
    """#4 — repair_dataset must emit log records, not print() to stdout."""

    def test_repair_dataset_uses_logging_not_stdout(self, tmp_path, capsys):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": NaN}\n', encoding="utf-8")
        from scripts.audit_jsonl_nan import repair_dataset

        repair_dataset(tmp_path, dry_run=True)
        captured = capsys.readouterr()
        # stdout must be empty — progress goes to logging/stderr, not print()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# RED: Bug B — regex corrupts quoted legal text containing spaced NaN/Infinity
# ---------------------------------------------------------------------------


class TestRepairRegexSafety:
    def test_nan_inside_quoted_string_not_modified(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        original = '{"text": "The court rejected NaN as evidence"}\n'
        shard.write_text(original, encoding="utf-8")
        repair_shard(shard, dry_run=False)
        assert shard.read_text(encoding="utf-8") == original

    def test_infinity_inside_quoted_string_not_modified(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        original = '{"text": "score was Infinity points"}\n'
        shard.write_text(original, encoding="utf-8")
        repair_shard(shard, dry_run=False)
        assert shard.read_text(encoding="utf-8") == original

    def test_bare_nan_value_still_repaired(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"case_name": NaN}\n', encoding="utf-8")
        repair_shard(shard, dry_run=False)
        obj = json.loads(shard.read_text(encoding="utf-8"))
        assert obj["case_name"] is None

    def test_repaired_output_is_valid_strict_json(self, tmp_path):
        """After repair, json.loads with parse_constant rejection must succeed."""
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"case_name": NaN, "score": Infinity}\n', encoding="utf-8")
        repair_shard(shard, dry_run=False)

        def reject(tok):
            raise ValueError(f"non-finite token: {tok}")

        obj = json.loads(shard.read_text(encoding="utf-8"), parse_constant=reject)
        assert obj["case_name"] is None
        assert obj["score"] is None


# ---------------------------------------------------------------------------
# RED: Bug C — gate_verdict misclassifies parse failures with empty nan_fields
# ---------------------------------------------------------------------------


class TestGateVerdictParseFailure:
    def test_parse_failures_with_empty_nan_fields_not_repairable(self):
        """Dataset with decode errors but no recorded fields must NOT be REPAIRABLE."""
        h = DatasetHealth(
            total_lines=100,
            nan_lines=5,  # 5 JSONDecodeError lines
            nan_shards=1,
            total_shards=5,
            nan_fields={},  # empty — no field names recorded
            contaminated_shards=["s.jsonl"],
        )
        assert "REPAIRABLE" not in h.gate_verdict()
        assert "PARSE_FAILURE" in h.gate_verdict()

    def test_zero_nan_lines_still_clean(self):
        h = DatasetHealth(100, 0, 0, 5, {}, [])
        assert h.gate_verdict() == "CLEAN"


# ---------------------------------------------------------------------------
# RED: Bug D — split nan_lines into typed counters on DatasetHealth
# ---------------------------------------------------------------------------


class TestTypedContaminationCounters:
    def test_dataset_health_has_nonfinite_lines(self):
        h = DatasetHealth(100, 0, 0, 5, {}, [])
        assert hasattr(h, "nonfinite_lines")

    def test_dataset_health_has_string_sentinel_lines(self):
        h = DatasetHealth(100, 0, 0, 5, {}, [])
        assert hasattr(h, "string_sentinel_lines")

    def test_dataset_health_has_decode_error_lines(self):
        h = DatasetHealth(100, 0, 0, 5, {}, [])
        assert hasattr(h, "decode_error_lines")

    def test_shard_health_has_typed_counters(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text(json.dumps({"id": "0"}) + "\n", encoding="utf-8")
        h = audit_shard(shard)
        assert hasattr(h, "nonfinite_lines")
        assert hasattr(h, "decode_error_lines")


# ---------------------------------------------------------------------------
# RED: Feature E — --workers configurable on audit_dataset
# ---------------------------------------------------------------------------


class TestConfigurableWorkers:
    def test_audit_dataset_accepts_workers_param(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text(json.dumps({"id": "0", "case_name": "Smith v. Jones"}) + "\n")
        result = audit_dataset(tmp_path, workers=1)
        assert isinstance(result, DatasetHealth)

    def test_repair_dataset_accepts_workers_param(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": NaN}\n', encoding="utf-8")
        from scripts.audit_jsonl_nan import repair_dataset

        repair_dataset(tmp_path, dry_run=True, workers=1)


# ---------------------------------------------------------------------------
# RED: Hypothesis property tests for _has_nan
# ---------------------------------------------------------------------------


class TestHasNanProperties:
    def test_finite_floats_never_flagged(self):
        from hypothesis import given, settings
        from hypothesis import strategies as st

        @given(st.floats(allow_nan=False, allow_infinity=False))
        @settings(max_examples=200)
        def inner(f):
            assert _has_nan(f) is False

        inner()

    def test_nan_inf_always_flagged(self):
        import math

        from hypothesis import given, settings
        from hypothesis import strategies as st

        @given(st.floats(allow_nan=True, allow_infinity=True))
        @settings(max_examples=200)
        def inner(f):
            if math.isnan(f) or math.isinf(f):
                assert _has_nan(f) is True

        inner()

    def test_arbitrary_nested_dict_consistent(self):
        import math

        from hypothesis import given, settings
        from hypothesis import strategies as st

        @given(st.dictionaries(st.text(max_size=10), st.floats(allow_nan=True, allow_infinity=True)))
        @settings(max_examples=100)
        def inner(d):
            result = _has_nan(d)
            has_bad = any(math.isnan(v) or math.isinf(v) for v in d.values())
            assert result == has_bad

        inner()


# ---------------------------------------------------------------------------
# RED: pydantic-settings AuditSettings importable and used by audit_dataset
# ---------------------------------------------------------------------------


class TestAuditSettings:
    def test_audit_settings_importable(self):
        from scripts.audit_jsonl_nan import AuditSettings

        assert callable(AuditSettings)

    def test_audit_settings_has_advisory_fields(self):
        from scripts.audit_jsonl_nan import AuditSettings

        cfg = AuditSettings()
        assert "case_name" in cfg.advisory_fields

    def test_audit_settings_has_workers(self):
        from scripts.audit_jsonl_nan import AuditSettings

        cfg = AuditSettings()
        assert isinstance(cfg.workers, int) and cfg.workers > 0

    def test_audit_settings_workers_env_override(self, monkeypatch):
        monkeypatch.setenv("AUDIT_WORKERS", "2")
        from scripts.audit_jsonl_nan import AuditSettings

        cfg = AuditSettings()
        assert cfg.workers == 2


# ---------------------------------------------------------------------------
# RED: OmegaConf — load_audit_config importable from module
# ---------------------------------------------------------------------------


class TestOmegaConf:
    def test_load_audit_config_importable(self):
        from scripts.audit_jsonl_nan import load_audit_config

        assert callable(load_audit_config)

    def test_load_audit_config_returns_advisory_fields(self, tmp_path):
        from scripts.audit_jsonl_nan import load_audit_config

        cfg_file = tmp_path / "audit.yaml"
        cfg_file.write_text("advisory_fields:\n  - case_name\n  - raw_text\n  - cleaning_flags\n")
        cfg = load_audit_config(cfg_file)
        assert "case_name" in cfg.advisory_fields


# ---------------------------------------------------------------------------
# RED: W&B — log_health_to_wandb importable and callable
# ---------------------------------------------------------------------------


class TestWandb:
    def test_log_health_to_wandb_importable(self):
        from scripts.audit_jsonl_nan import log_health_to_wandb

        assert callable(log_health_to_wandb)

    def test_log_health_to_wandb_runs_offline(self, monkeypatch):
        monkeypatch.setenv("WANDB_MODE", "offline")
        from scripts.audit_jsonl_nan import log_health_to_wandb

        health = DatasetHealth(100, 0, 0, 5, {}, [])
        # must not raise
        log_health_to_wandb(health, project="test-probe")


# ---------------------------------------------------------------------------
# RED: strict encoding mode — errors="strict" surfaces corruption
# ---------------------------------------------------------------------------


class TestStrictEncoding:
    def test_audit_shard_strict_catches_corrupt_bytes(self, tmp_path):
        from scripts.audit_jsonl_nan import audit_shard_strict

        shard = tmp_path / "s.jsonl"
        corrupt = b'{"id": "0", "case_name": "Smith \xff\xfe Jones"}\n'
        shard.write_bytes(corrupt)
        health = audit_shard_strict(shard)
        assert health.decode_error_lines >= 1

    def test_audit_shard_strict_clean_shard_passes(self, tmp_path):
        from scripts.audit_jsonl_nan import audit_shard_strict

        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": "Smith v. Jones"}\n', encoding="utf-8")
        health = audit_shard_strict(shard)
        assert health.decode_error_lines == 0
        assert health.nan_lines == 0


# ---------------------------------------------------------------------------
# RED: post-repair Polars validation
# ---------------------------------------------------------------------------


class TestPolarsValidation:
    def test_validate_shard_importable(self):
        from scripts.audit_jsonl_nan import validate_shard_polars

        assert callable(validate_shard_polars)

    def test_validate_shard_accepts_clean_shard(self, tmp_path):
        from scripts.audit_jsonl_nan import validate_shard_polars

        shard = tmp_path / "s.jsonl"
        shard.write_text(
            '{"id": "0", "case_name": "Smith v. Jones", "score": 1.0}\n',
            encoding="utf-8",
        )
        ok, err = validate_shard_polars(shard)
        assert ok is True
        assert err is None

    def test_validate_shard_rejects_bare_nan(self, tmp_path):
        from scripts.audit_jsonl_nan import validate_shard_polars

        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": NaN}\n', encoding="utf-8")
        ok, err = validate_shard_polars(shard)
        assert ok is False
        assert err is not None

    def test_validate_after_repair_passes(self, tmp_path):
        from scripts.audit_jsonl_nan import validate_shard_polars

        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": NaN}\n', encoding="utf-8")
        repair_shard(shard, dry_run=False)
        ok, err = validate_shard_polars(shard)
        assert ok is True
        assert err is None


# ---------------------------------------------------------------------------
# Coverage gap tests — push from 78% to ≥80%
# ---------------------------------------------------------------------------


class TestIsNonfiniteBranches:
    def test_list_containing_nan(self):
        from scripts.audit_jsonl_nan import _is_nonfinite

        assert _is_nonfinite([1.0, float("nan")]) is True

    def test_list_all_clean(self):
        from scripts.audit_jsonl_nan import _is_nonfinite

        assert _is_nonfinite([1.0, 2.0]) is False

    def test_dict_containing_inf(self):
        from scripts.audit_jsonl_nan import _is_nonfinite

        assert _is_nonfinite({"a": float("inf")}) is True


class TestIsStringSentinelBranches:
    def test_list_containing_sentinel(self):
        from scripts.audit_jsonl_nan import _is_string_sentinel

        assert _is_string_sentinel(["ok", "NaN"]) is True

    def test_dict_containing_sentinel(self):
        from scripts.audit_jsonl_nan import _is_string_sentinel

        assert _is_string_sentinel({"x": "Infinity"}) is True

    def test_clean_list(self):
        from scripts.audit_jsonl_nan import _is_string_sentinel

        assert _is_string_sentinel(["ok", "fine"]) is False


class TestAuditShardImplBranches:
    def test_empty_lines_skipped(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        shard.write_text("\n\n" + json.dumps({"id": "0"}) + "\n", encoding="utf-8")
        h = audit_shard(shard)
        assert h.total_lines == 1

    def test_string_sentinel_counted(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        # "NaN" as a string value — string sentinel path
        shard.write_text('{"id": "0", "case_name": "NaN"}\n', encoding="utf-8")
        h = audit_shard(shard)
        assert h.string_sentinel_lines == 1

    def test_nonfinite_and_sentinel_same_record(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        # bare NaN (nonfinite) — string sentinel is separate field
        lines = '{"id": "0", "case_name": NaN}\n'
        shard.write_text(lines, encoding="utf-8")
        h = audit_shard(shard)
        assert h.nonfinite_lines == 1

    def test_file_level_unicode_error_strict(self, tmp_path):
        from scripts.audit_jsonl_nan import audit_shard_strict

        shard = tmp_path / "s.jsonl"
        # write bytes that are invalid utf-8 at file level
        shard.write_bytes(b"\xff\xfe" + b'{"id": "0"}\n')
        h = audit_shard_strict(shard)
        assert h.decode_error_lines >= 1


class TestReplaceNonfiniteBranches:
    def test_list_with_nan(self):
        from scripts.audit_jsonl_nan import _replace_nonfinite

        result = _replace_nonfinite([1.0, float("nan"), 3.0])
        assert result == [1.0, None, 3.0]


class TestRepairShardMalformedLine:
    def test_malformed_json_line_passed_through(self, tmp_path):
        shard = tmp_path / "s.jsonl"
        malformed = "not valid json at all\n"
        good = json.dumps({"id": "1"}) + "\n"
        shard.write_text(malformed + good, encoding="utf-8")
        total, repaired = repair_shard(shard, dry_run=False)
        assert total == 2
        assert repaired == 0
        lines = shard.read_text().splitlines()
        assert lines[0] == "not valid json at all"


class TestRepairDatasetBranches:
    def test_repair_dataset_raises_on_missing_dir(self, tmp_path):
        from scripts.audit_jsonl_nan import repair_dataset

        with pytest.raises(FileNotFoundError):
            repair_dataset(tmp_path / "nonexistent")

    def test_repair_dataset_validate_flag_passes(self, tmp_path):
        from scripts.audit_jsonl_nan import repair_dataset

        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": NaN}\n', encoding="utf-8")
        repair_dataset(tmp_path, dry_run=False, validate=True)
        obj = json.loads(shard.read_text())
        assert obj["case_name"] is None

    def test_repair_dataset_validate_logs_failure(self, tmp_path, caplog):
        import logging

        from scripts.audit_jsonl_nan import repair_dataset

        shard = tmp_path / "s.jsonl"
        # write a shard that repair won't fix but polars will reject
        # inject a bare NaN that semantic repair catches — then corrupt bak
        shard.write_text('{"id": "0", "case_name": NaN}\n', encoding="utf-8")
        # mock validate_shard_polars to return failure
        import scripts.audit_jsonl_nan as m

        original = m.validate_shard_polars
        m.validate_shard_polars = lambda p: (False, "TapeError")
        try:
            with caplog.at_level(logging.ERROR, logger="scripts.audit_jsonl_nan"):
                repair_dataset(tmp_path, dry_run=False, validate=True)
            assert any("FAILED" in r.message for r in caplog.records)
        finally:
            m.validate_shard_polars = original


class TestEmitTextShardIds:
    def test_emit_text_with_shard_ids(self, capsys):
        h = DatasetHealth(
            total_lines=100,
            nan_lines=5,
            nan_shards=1,
            total_shards=5,
            nan_fields={"case_name": 5},
            contaminated_shards=["shard_0000.jsonl"],
        )
        from scripts.audit_jsonl_nan import _emit_text

        _emit_text(h, emit_shard_ids=True)
        captured = capsys.readouterr()
        assert "shard_0000.jsonl" in captured.out


class TestMain:
    def test_main_text_output(self, tmp_path, capsys, monkeypatch):
        import sys

        from scripts.audit_jsonl_nan import main

        shard = tmp_path / "s.jsonl"
        shard.write_text(json.dumps({"id": "0", "case_name": "Smith v. Jones"}) + "\n")
        monkeypatch.setattr(sys, "argv", ["audit", "--input-dir", str(tmp_path)])
        main()
        captured = capsys.readouterr()
        assert "verdict" in captured.out

    def test_main_json_flag(self, tmp_path, capsys, monkeypatch):
        import sys

        from scripts.audit_jsonl_nan import main

        shard = tmp_path / "s.jsonl"
        shard.write_text(json.dumps({"id": "0"}) + "\n")
        monkeypatch.setattr(sys, "argv", ["audit", "--input-dir", str(tmp_path), "--json"])
        main()
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "gate_verdict" in parsed

    def test_main_csv_flag(self, tmp_path, monkeypatch):
        import sys

        from scripts.audit_jsonl_nan import main

        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": NaN}\n')
        csv_out = tmp_path / "out.csv"
        monkeypatch.setattr(sys, "argv", ["audit", "--input-dir", str(tmp_path), "--csv", str(csv_out)])
        main()
        assert csv_out.exists()

    def test_main_fix_flag(self, tmp_path, monkeypatch):
        import sys

        from scripts.audit_jsonl_nan import main

        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": NaN}\n')
        monkeypatch.setattr(sys, "argv", ["audit", "--input-dir", str(tmp_path), "--fix"])
        main()
        obj = json.loads(shard.read_text())
        assert obj["case_name"] is None

    def test_main_emit_shard_ids(self, tmp_path, capsys, monkeypatch):
        import sys

        from scripts.audit_jsonl_nan import main

        shard = tmp_path / "s.jsonl"
        shard.write_text('{"id": "0", "case_name": NaN}\n')
        monkeypatch.setattr(sys, "argv", ["audit", "--input-dir", str(tmp_path), "--emit-shard-ids"])
        main()
        captured = capsys.readouterr()
        assert "s.jsonl" in captured.out


# ---------------------------------------------------------------------------
# RED: --strict-encoding CLI flag wired into main()
# ---------------------------------------------------------------------------


class TestMainStrictEncoding:
    def test_main_strict_encoding_flag_accepted(self, tmp_path, monkeypatch):
        import sys

        from scripts.audit_jsonl_nan import main

        shard = tmp_path / "s.jsonl"
        shard.write_text(json.dumps({"id": "0"}) + "\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["audit", "--input-dir", str(tmp_path), "--strict-encoding"])
        # must not raise SystemExit (unrecognised argument)
        main()

    def test_main_strict_encoding_catches_corrupt_shard(self, tmp_path, capsys, monkeypatch):
        import sys

        from scripts.audit_jsonl_nan import main

        shard = tmp_path / "s.jsonl"
        shard.write_bytes(b"\xff\xfe" + b'{"id": "0"}\n')
        monkeypatch.setattr(sys, "argv", ["audit", "--input-dir", str(tmp_path), "--strict-encoding"])
        main()
        captured = capsys.readouterr()
        assert "decode_error_lines" in captured.out
