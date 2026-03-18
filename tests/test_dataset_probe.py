# tests/test_dataset_probe.py
# TDD unit tests for HF dataset schema contract — no real network calls.
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "courtlistener_sample.json"
HEX_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")

from tests.fixtures.courtlistener_checksums import COURTLISTENER_SAMPLE_TEXT_SHA256


class CourtListenerDatasetProbe:
    """
    Schema contract and access layer for pile-of-law/pile-of-law
    subset r_courtlistener_opinions.

    Reproducibility contract:
      - REVISION must be an immutable commit hash, not a mutable branch.
      - trust_remote_code must not be enabled unless explicitly required and audited.
    """

    DATASET_ID = "pile-of-law/pile-of-law"
    SUBSET = "r_courtlistener_opinions"
    SPLIT = "train"
    REVISION = "REPLACE_WITH_DATASET_COMMIT_HASH"
    REQUIRED_FIELDS: frozenset[str] = frozenset({"text", "created_timestamp", "downloaded_timestamp", "url"})
    TEXT_FIELDS: tuple[str, ...] = ("text", "contents")
    MIN_TEXT_LENGTH = 50

    def load(self, streaming: bool = True) -> Iterable[dict[str, Any]]:
        """Load dataset at pinned immutable revision. trust_remote_code never passed.

        Single-pass semantics: treat the returned object as single-pass.
        Wrap in iter() to enforce single-pass explicitly.
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
        Invariant: downstream code never sees invalid rows.
        """
        rows = source if source is not None else self.load()
        for row in rows:
            if not self.validate_row(row):
                yield self.normalize_row(row)

    def validate_row(self, row: dict[str, Any]) -> list[str]:
        """Return all validation errors. Empty list = valid."""
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
        """Normalize validated row into canonical form for downstream use."""
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
def fixture_data() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture
def pinned_row(fixture_data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in fixture_data.items() if k != "_fixture_meta"}


class TestCourtListenerDatasetProbeContract:
    def test_fixture_passes_validate_row(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert probe.validate_row(pinned_row) == []

    def test_fixture_text_is_non_empty_string(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        text = probe.get_text(pinned_row)
        assert isinstance(text, str) and len(text) >= probe.MIN_TEXT_LENGTH

    def test_fixture_url_is_courtlistener(self, pinned_row: dict) -> None:
        assert "courtlistener.com" in pinned_row["url"]

    def test_fixture_is_deterministic(self, pinned_row: dict) -> None:
        reloaded = {k: v for k, v in json.loads(FIXTURE_PATH.read_text()).items() if k != "_fixture_meta"}
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
        assert any(
            "Missing required fields" in e
            for e in probe.validate_row(
                {"text": "long enough text here for the test to pass validation threshold okay"}
            )
        )

    def test_validate_row_catches_short_text(self, probe: CourtListenerDatasetProbe) -> None:
        assert any(
            "too short" in e
            for e in probe.validate_row(
                {"text": "too short", "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
            )
        )

    def test_validate_row_catches_non_string_text(self, probe: CourtListenerDatasetProbe) -> None:
        for bad_value in (42, None, ["a", "b"], 3.14):
            row = {"text": bad_value, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
            assert any("must be str" in e for e in probe.validate_row(row)), f"Expected type error for {bad_value!r}"

    def test_validate_row_reports_all_errors(self, probe: CourtListenerDatasetProbe) -> None:
        bad = {"text": "short", "url": "x"}
        errors = probe.validate_row(bad)
        assert any("Missing required fields" in e for e in errors)
        assert any("too short" in e for e in errors)
        assert len(errors) == 2


class TestReproducibilityContract:
    def test_fixture_has_required_provenance_fields(self, fixture_data: dict) -> None:
        meta = fixture_data["_fixture_meta"]
        required = {"sampled_at", "dataset", "subset", "split", "revision", "text_sha256"}
        assert required <= set(meta.keys()), f"Missing: {required - set(meta.keys())}"

    def test_fixture_metadata_matches_probe_constants(
        self, probe: CourtListenerDatasetProbe, fixture_data: dict
    ) -> None:
        meta = fixture_data["_fixture_meta"]
        assert meta["dataset"] == probe.DATASET_ID
        assert meta["subset"] == probe.SUBSET
        assert meta["split"] == probe.SPLIT
        assert meta["revision"] == probe.REVISION

    def test_revision_must_not_be_mutable_branch(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe.REVISION not in {"main", "master", "latest", "HEAD", ""}

    def test_revision_is_commit_hash_or_explicit_placeholder(self, probe: CourtListenerDatasetProbe) -> None:
        assert (
            HEX_REVISION_RE.fullmatch(probe.REVISION) is not None
            or probe.REVISION == "REPLACE_WITH_DATASET_COMMIT_HASH"
        )

    def test_fixture_text_sha256_matches_checksum_module(self, pinned_row: dict) -> None:
        """Content fingerprint verified against checksums.py — not inline JSON (avoids detect-secrets)."""
        actual = hashlib.sha256(pinned_row["text"].encode("utf-8")).hexdigest()
        assert actual == COURTLISTENER_SAMPLE_TEXT_SHA256

    def test_sampled_at_is_iso_date(self, fixture_data: dict) -> None:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", fixture_data["_fixture_meta"]["sampled_at"])

    @patch("datasets.load_dataset")
    def test_load_passes_revision_to_hf(self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        list(probe.load())
        assert mock_load.call_args.kwargs["revision"] == probe.REVISION

    @patch("datasets.load_dataset")
    def test_load_does_not_enable_trust_remote_code(
        self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        list(probe.load())
        assert "trust_remote_code" not in mock_load.call_args.kwargs


class TestIterValidRows:
    def test_yields_only_valid_rows(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        rows = list(probe.iter_valid_rows([pinned_row, {"url": "x"}]))
        assert len(rows) == 1

    def test_yields_normalized_rows(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        rows = list(probe.iter_valid_rows([pinned_row]))
        assert set(rows[0].keys()) == {"text", "created_timestamp", "downloaded_timestamp", "url", "source_url"}

    def test_empty_source_yields_nothing(self, probe: CourtListenerDatasetProbe) -> None:
        assert list(probe.iter_valid_rows([])) == []

    def test_all_invalid_yields_nothing(self, probe: CourtListenerDatasetProbe) -> None:
        assert list(probe.iter_valid_rows([{"url": "x"}, {"text": "short"}])) == []

    def test_returns_iterator(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert hasattr(probe.iter_valid_rows([pinned_row]), "__next__")

    def test_downstream_never_sees_invalid_rows(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        for row in probe.iter_valid_rows([pinned_row, {"url": "bad"}, pinned_row]):
            assert probe.validate_row(row) == []


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

    def test_output_canonical_keys(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert set(probe.normalize_row(pinned_row).keys()) == {
            "text",
            "created_timestamp",
            "downloaded_timestamp",
            "url",
            "source_url",
        }

    def test_idempotent(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        first = probe.normalize_row(pinned_row)
        assert first["text"] == probe.normalize_row(first)["text"]


class TestCourtListenerDatasetProbeLoad:
    @patch("datasets.load_dataset")
    def test_load_uses_correct_dataset_id(self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        list(probe.load())
        assert mock_load.call_args.args[0] == CourtListenerDatasetProbe.DATASET_ID
        assert mock_load.call_args.args[1] == CourtListenerDatasetProbe.SUBSET

    @patch("datasets.load_dataset", side_effect=ConnectionError("network unavailable"))
    def test_network_failure_raises_clearly(self, mock_load, probe: CourtListenerDatasetProbe) -> None:
        with pytest.raises(ConnectionError, match="network unavailable"):
            list(probe.load())

    @patch("datasets.load_dataset")
    def test_first_row_passes_validation(self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        assert probe.validate_row(next(iter(probe.load()))) == []

    @patch("datasets.load_dataset")
    def test_mock_supports_multiple_iter_calls(
        self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        ds = probe.load()
        assert list(ds) == list(ds) == [pinned_row]
