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
