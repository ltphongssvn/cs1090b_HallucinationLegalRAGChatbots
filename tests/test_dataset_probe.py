# tests/test_dataset_probe.py
# TDD unit tests for HF dataset schema contract — no real network calls.
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "courtlistener_sample.json"

# Pinned dataset revision for reproducibility.
# IMPORTANT: replace "main" with a specific commit hash once the dataset
# version used for experiments is locked, so git-blame can identify the
# exact sample seen on any given date.
# e.g. "a1b2c3d4e5f6..." from: huggingface-cli repo info pile-of-law/pile-of-law
PINNED_REVISION = "main"


class CourtListenerDatasetProbe:
    """
    Schema contract and access layer for pile-of-law/pile-of-law
    subset r_courtlistener_opinions.

    Reproducibility contract:
      - PINNED_REVISION must be updated to a commit hash before any
        experiment whose results need to be reproducible by date.
      - trust_remote_code is never passed — remote code execution is
        a security and reproducibility violation in 2026 research standards.
    """

    DATASET_ID = "pile-of-law/pile-of-law"
    SUBSET = "r_courtlistener_opinions"
    SPLIT = "train"
    REVISION = PINNED_REVISION
    REQUIRED_FIELDS: frozenset[str] = frozenset({"text", "created_timestamp", "downloaded_timestamp", "url"})
    TEXT_FIELDS: tuple[str, ...] = ("text", "contents")
    MIN_TEXT_LENGTH = 50

    def load(self, streaming: bool = True) -> Iterable[dict[str, Any]]:
        """Load dataset at pinned revision. trust_remote_code never passed.

        Revision pinning: self.REVISION controls which dataset commit is
        loaded. Set to a specific commit hash (not "main") for experiments
        that must be reproducible by date — "main" is mutable and cannot
        be git-blamed to a specific sample.

        Single-pass semantics: treat the returned object as single-pass.
        Whether a second iteration re-streams or raises is HF implementation-
        dependent and not guaranteed. Wrap in iter() to enforce single-pass.
        """
        from datasets import load_dataset

        return load_dataset(  # type: ignore[return-value]
            self.DATASET_ID,
            self.SUBSET,
            split=self.SPLIT,
            streaming=streaming,
            revision=self.REVISION,
        )

    def iter_valid_rows(self, source: Iterable[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
        """Yield only validated, normalized rows. Invalid rows are skipped.

        Encodes the invariant: downstream code never sees invalid rows.
        """
        rows = source if source is not None else self.load()
        for row in rows:
            if not self.validate_row(row):
                yield self.normalize_row(row)

    def validate_row(self, row: dict[str, Any]) -> list[str]:
        """Return list of all validation errors. Empty list = valid."""
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
        """Normalize a validated row into canonical form for downstream use."""
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
        match = re.search(r"\d{4}-\d{2}-\d{2}", ts)
        return match.group(0) if match else ""

    def resolve_text_field(self, row: dict[str, Any]) -> str | None:
        return next((k for k in self.TEXT_FIELDS if k in row), None)

    def get_text(self, row: dict[str, Any]) -> str:
        field = self.resolve_text_field(row)
        if field is None:
            raise ValueError(f"No text field in row keys: {sorted(row.keys())}")
        return str(row[field])


def _mock_iterable_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    return mock_ds


@pytest.fixture
def probe() -> CourtListenerDatasetProbe:
    return CourtListenerDatasetProbe()


@pytest.fixture
def pinned_row() -> dict[str, Any]:
    """Deterministic fixture — sampled_at date in _fixture_meta enables git-blame traceability."""
    data = json.loads(FIXTURE_PATH.read_text())
    # Strip internal metadata before returning to tests
    return {k: v for k, v in data.items() if not k.startswith("_")}


class TestCourtListenerDatasetProbeContract:
    def test_fixture_passes_validate_row(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert probe.validate_row(pinned_row) == []

    def test_fixture_text_is_non_empty_string(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        text = probe.get_text(pinned_row)
        assert isinstance(text, str) and len(text) >= probe.MIN_TEXT_LENGTH

    def test_fixture_url_is_courtlistener(self, pinned_row: dict) -> None:
        assert "courtlistener.com" in pinned_row["url"]

    def test_fixture_has_sampled_at_metadata(self) -> None:
        """git-blame traceability: fixture must record when sample was taken."""
        data = json.loads(FIXTURE_PATH.read_text())
        assert "_fixture_meta" in data
        assert "sampled_at" in data["_fixture_meta"]
        assert "revision" in data["_fixture_meta"]

    def test_fixture_is_deterministic(self, pinned_row: dict) -> None:
        reloaded = {k: v for k, v in json.loads(FIXTURE_PATH.read_text()).items() if not k.startswith("_")}
        assert pinned_row == reloaded

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
        assert any("Missing required fields" in e for e in probe.validate_row(bad))

    def test_validate_row_catches_short_text(self, probe: CourtListenerDatasetProbe) -> None:
        short = {"text": "too short", "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        assert any("too short" in e for e in probe.validate_row(short))

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


class TestReproducibilityContract:
    def test_revision_is_set(self, probe: CourtListenerDatasetProbe) -> None:
        """Revision must be set — not None or empty — to enable reproducible loading."""
        assert probe.REVISION is not None
        assert probe.REVISION != ""

    def test_revision_constant_is_documented_in_load_docstring(self, probe: CourtListenerDatasetProbe) -> None:
        assert "revision" in probe.load.__doc__.lower()

    def test_trust_remote_code_is_never_true(self, probe: CourtListenerDatasetProbe) -> None:
        """Security invariant: trust_remote_code must never appear as True in load()."""
        import inspect

        source = inspect.getsource(probe.load)
        assert "trust_remote_code=True" not in source

    @patch("datasets.load_dataset")
    def test_load_passes_revision_to_hf(self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        list(probe.load())
        assert mock_load.call_args.kwargs.get("revision") == probe.REVISION


class TestIterValidRows:
    def test_yields_only_valid_rows(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        invalid = {"url": "x"}
        rows = list(probe.iter_valid_rows([pinned_row, invalid]))
        assert len(rows) == 1

    def test_yields_normalized_rows(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        rows = list(probe.iter_valid_rows([pinned_row]))
        assert rows[0].keys() == {"text", "created_timestamp", "downloaded_timestamp", "url", "source_url"}

    def test_empty_source_yields_nothing(self, probe: CourtListenerDatasetProbe) -> None:
        assert list(probe.iter_valid_rows([])) == []

    def test_all_invalid_yields_nothing(self, probe: CourtListenerDatasetProbe) -> None:
        assert list(probe.iter_valid_rows([{"url": "x"}, {"text": "short"}])) == []

    def test_returns_iterator_not_list(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert hasattr(probe.iter_valid_rows([pinned_row]), "__next__")

    def test_downstream_never_sees_invalid_rows(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        mixed = [pinned_row, {"url": "bad"}, pinned_row]
        for row in probe.iter_valid_rows(mixed):
            assert "text" in row and isinstance(row["text"], str)


class TestNormalizeRow:
    def test_canonical_text_key(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        result = probe.normalize_row(pinned_row)
        assert isinstance(result["text"], str) and len(result["text"]) > 0

    def test_strips_whitespace(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"text": "  leading and trailing  ", "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        assert probe.normalize_row(row)["text"] == "leading and trailing"

    def test_renames_contents_to_text(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"contents": "A" * 60, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        assert "text" in probe.normalize_row(row)

    def test_extracts_timestamp_date(self, probe: CourtListenerDatasetProbe) -> None:
        row = {
            "text": "A" * 60,
            "created_timestamp": "2022-01-15T10:30:00Z",
            "downloaded_timestamp": "2022-06-01",
            "url": "x",
        }
        result = probe.normalize_row(row)
        assert result["created_timestamp"] == "2022-01-15"
        assert result["downloaded_timestamp"] == "2022-06-01"

    def test_adds_source_url_alias(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        result = probe.normalize_row(pinned_row)
        assert result["source_url"] == result["url"]

    def test_output_has_canonical_keys(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert set(probe.normalize_row(pinned_row).keys()) == {
            "text",
            "created_timestamp",
            "downloaded_timestamp",
            "url",
            "source_url",
        }

    def test_idempotent(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        first = probe.normalize_row(pinned_row)
        second = probe.normalize_row(first)
        assert first["text"] == second["text"]


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
    def test_mock_supports_multiple_iter_calls(
        self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        ds = probe.load()
        assert list(ds) == list(ds) == [pinned_row]
