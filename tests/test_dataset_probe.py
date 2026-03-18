# tests/test_dataset_probe.py
# TDD unit tests for HF dataset schema contract — no real network calls.
import json
import re
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "courtlistener_sample.json"


class CourtListenerDatasetProbe:
    """
    Schema contract and access layer for pile-of-law/pile-of-law
    subset r_courtlistener_opinions.
    """

    DATASET_ID = "pile-of-law/pile-of-law"
    SUBSET = "r_courtlistener_opinions"
    SPLIT = "train"
    REQUIRED_FIELDS: frozenset[str] = frozenset({"text", "created_timestamp", "downloaded_timestamp", "url"})
    TEXT_FIELDS: tuple[str, ...] = ("text", "contents")
    MIN_TEXT_LENGTH = 50

    def load(self, streaming: bool = True) -> Iterable[dict[str, Any]]:
        """Load dataset. trust_remote_code is intentionally never passed.

        Returns Iterable (not Iterator) — HF IterableDataset is iterable
        but not an exhausted iterator until iter() is called on it.

        Single-pass semantics: treat the returned object as single-pass.
        Whether a second iteration re-streams or raises is HF implementation-
        dependent and not guaranteed. If single-pass must be enforced
        explicitly, wrap in an iterator: iter(probe.load()).
        """
        from datasets import load_dataset

        return load_dataset(  # type: ignore[return-value]
            self.DATASET_ID, self.SUBSET, split=self.SPLIT, streaming=streaming
        )

    def validate_row(self, row: dict[str, Any]) -> list[str]:
        """
        Return list of all validation errors for a row. Empty list = valid.

        Checks are independent — not early-exit elif chains — so all errors
        surface per row, not just the first one.
        """
        errors: list[str] = []

        missing = self.REQUIRED_FIELDS - set(row.keys())
        if missing:
            errors.append(f"Missing required fields: {sorted(missing)}")

        text_field = self.resolve_text_field(row)
        if text_field is None:
            errors.append(f"No text field found in {sorted(row.keys())}")
            return errors

        value = row[text_field]

        if not isinstance(value, str):
            errors.append(f"{text_field} must be str, got {type(value).__name__!r}: {value!r}")
            return errors

        if len(value) < self.MIN_TEXT_LENGTH:
            errors.append(f"{text_field} too short: {len(value)} < {self.MIN_TEXT_LENGTH}")

        return errors

    def normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize a validated row into a canonical form for downstream pipeline use.

        Responsibilities (single concern per step):
          - Resolve and rename the text field to the canonical key 'text'
          - Normalize timestamps to ISO-8601 date strings (YYYY-MM-DD)
          - Strip leading/trailing whitespace from text
          - Preserve all original fields; add 'source_url' as pipeline-friendly alias

        Callers should validate_row() before normalize_row(). Behavior on
        invalid rows is undefined.
        """
        text_field = self.resolve_text_field(row)
        text = str(row[text_field]).strip() if text_field else ""

        return {
            "text": text,
            "created_timestamp": self._normalize_timestamp(str(row.get("created_timestamp", ""))),
            "downloaded_timestamp": self._normalize_timestamp(str(row.get("downloaded_timestamp", ""))),
            "url": str(row.get("url", "")),
            "source_url": str(row.get("url", "")),
        }

    def _normalize_timestamp(self, ts: str) -> str:
        """Extract YYYY-MM-DD from timestamp string. Returns '' if unparseable."""
        match = re.search(r"\d{4}-\d{2}-\d{2}", ts)
        return match.group(0) if match else ""

    def resolve_text_field(self, row: dict[str, Any]) -> str | None:
        """Return the first available text field name, or None."""
        return next((k for k in self.TEXT_FIELDS if k in row), None)

    def get_text(self, row: dict[str, Any]) -> str:
        """Extract text content. Raises ValueError if no text field found."""
        field = self.resolve_text_field(row)
        if field is None:
            raise ValueError(f"No text field in row keys: {sorted(row.keys())}")
        return str(row[field])


def _mock_iterable_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    """Return MagicMock simulating HF IterableDataset — iter() returns fresh iterator."""
    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    return mock_ds


@pytest.fixture
def probe() -> CourtListenerDatasetProbe:
    return CourtListenerDatasetProbe()


@pytest.fixture
def pinned_row() -> dict[str, Any]:
    """Deterministic first-sample fixture — pins schema contract to a known-good row."""
    return json.loads(FIXTURE_PATH.read_text())


class TestCourtListenerDatasetProbeContract:
    def test_fixture_passes_validate_row(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert probe.validate_row(pinned_row) == []

    def test_fixture_text_is_non_empty_string(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        text = probe.get_text(pinned_row)
        assert isinstance(text, str)
        assert len(text) >= probe.MIN_TEXT_LENGTH

    def test_fixture_url_is_courtlistener(self, pinned_row: dict) -> None:
        assert "courtlistener.com" in pinned_row["url"]

    def test_fixture_is_deterministic(self, pinned_row: dict) -> None:
        assert pinned_row == json.loads(FIXTURE_PATH.read_text())

    def test_resolve_text_field_prefers_text_over_contents(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"text": "hello", "contents": "world", "url": "x", "created_timestamp": "", "downloaded_timestamp": ""}
        assert probe.resolve_text_field(row) == "text"

    def test_resolve_text_field_falls_back_to_contents(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"contents": "world", "url": "x", "created_timestamp": "", "downloaded_timestamp": ""}
        assert probe.resolve_text_field(row) == "contents"

    def test_resolve_text_field_returns_none_when_missing(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe.resolve_text_field({"url": "x"}) is None

    def test_get_text_raises_on_missing_field(self, probe: CourtListenerDatasetProbe) -> None:
        with pytest.raises(ValueError, match="No text field"):
            probe.get_text({"url": "x"})

    def test_validate_row_catches_missing_required_fields(self, probe: CourtListenerDatasetProbe) -> None:
        bad = {"text": "long enough text here for the test to pass validation threshold okay"}
        errors = probe.validate_row(bad)
        assert any("Missing required fields" in e for e in errors)

    def test_validate_row_catches_short_text(self, probe: CourtListenerDatasetProbe) -> None:
        short = {"text": "too short", "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        errors = probe.validate_row(short)
        assert any("too short" in e for e in errors)

    def test_validate_row_catches_non_string_text(self, probe: CourtListenerDatasetProbe) -> None:
        for bad_value in (42, None, ["a", "b"], 3.14):
            row = {"text": bad_value, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
            errors = probe.validate_row(row)
            assert any("must be str" in e for e in errors), f"Expected type error for {bad_value!r}, got: {errors}"

    def test_validate_row_reports_all_errors_not_just_first(self, probe: CourtListenerDatasetProbe) -> None:
        bad = {"text": "short", "url": "x"}
        errors = probe.validate_row(bad)
        assert any("Missing required fields" in e for e in errors)
        assert any("too short" in e for e in errors)
        assert len(errors) == 2

    def test_load_single_pass_semantics_documented(self, probe: CourtListenerDatasetProbe) -> None:
        assert "single-pass" in probe.load.__doc__.lower()


class TestNormalizeRow:
    def test_normalize_row_canonical_text_key(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        result = probe.normalize_row(pinned_row)
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0

    def test_normalize_row_strips_whitespace(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"text": "  leading and trailing  ", "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        result = probe.normalize_row(row)
        assert result["text"] == "leading and trailing"

    def test_normalize_row_renames_contents_to_text(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"contents": "A" * 60, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        result = probe.normalize_row(row)
        assert "text" in result
        assert result["text"] == "A" * 60

    def test_normalize_row_extracts_timestamp_date(self, probe: CourtListenerDatasetProbe) -> None:
        row = {
            "text": "A" * 60,
            "created_timestamp": "2022-01-15T10:30:00Z",
            "downloaded_timestamp": "2022-06-01",
            "url": "x",
        }
        result = probe.normalize_row(row)
        assert result["created_timestamp"] == "2022-01-15"
        assert result["downloaded_timestamp"] == "2022-06-01"

    def test_normalize_row_empty_timestamp_returns_empty(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"text": "A" * 60, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        result = probe.normalize_row(row)
        assert result["created_timestamp"] == ""

    def test_normalize_row_adds_source_url_alias(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        result = probe.normalize_row(pinned_row)
        assert result["source_url"] == result["url"]

    def test_normalize_row_output_has_canonical_keys(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        result = probe.normalize_row(pinned_row)
        assert set(result.keys()) == {"text", "created_timestamp", "downloaded_timestamp", "url", "source_url"}

    def test_normalize_row_idempotent(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        first = probe.normalize_row(pinned_row)
        second = probe.normalize_row(first)
        assert first["text"] == second["text"]
        assert first["created_timestamp"] == second["created_timestamp"]


class TestCourtListenerDatasetProbeLoad:
    @patch("datasets.load_dataset")
    def test_load_does_not_pass_trust_remote_code(
        self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        list(probe.load())
        assert mock_load.call_args.kwargs.get("trust_remote_code", False) is not True

    @patch("datasets.load_dataset")
    def test_load_uses_correct_dataset_id(self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        list(probe.load())
        args = mock_load.call_args.args
        assert args[0] == CourtListenerDatasetProbe.DATASET_ID
        assert args[1] == CourtListenerDatasetProbe.SUBSET

    @patch("datasets.load_dataset", side_effect=ConnectionError("network unavailable"))
    def test_network_failure_raises_clearly(self, mock_load, probe: CourtListenerDatasetProbe) -> None:
        with pytest.raises(ConnectionError, match="network unavailable"):
            list(probe.load())

    @patch("datasets.load_dataset")
    def test_first_row_passes_validation(self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        row = next(iter(probe.load()))
        assert probe.validate_row(row) == []

    @patch("datasets.load_dataset")
    def test_mock_iterable_dataset_supports_multiple_iter_calls(
        self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        ds = probe.load()
        assert list(ds) == list(ds) == [pinned_row]
