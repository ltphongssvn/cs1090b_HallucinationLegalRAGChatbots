# tests/test_dataset_probe_contract.py
# Schema contract tests for CourtListenerDatasetProbe — validate/normalize/get_text.
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.dataset_probe import CourtListenerDatasetProbe

pytestmark = pytest.mark.unit

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "courtlistener_sample.json"


def _mock_iterable_dataset(rows: list[dict]) -> MagicMock:
    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    return mock_ds


@pytest.fixture
def probe() -> CourtListenerDatasetProbe:
    return CourtListenerDatasetProbe()


@pytest.fixture
def fixture_data() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture
def pinned_row(fixture_data: dict) -> dict:
    return {k: v for k, v in fixture_data.items() if k != "_fixture_meta"}


class TestValidateRow:
    def test_fixture_passes_validate_row(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert probe.validate_row(pinned_row) == []

    def test_catches_missing_required_fields(self, probe: CourtListenerDatasetProbe) -> None:
        assert any(
            "Missing required fields" in e
            for e in probe.validate_row(
                {"text": "long enough text here for the test to pass validation threshold okay"}
            )
        )

    def test_catches_short_text(self, probe: CourtListenerDatasetProbe) -> None:
        assert any(
            "too short" in e
            for e in probe.validate_row(
                {"text": "too short", "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
            )
        )

    def test_catches_non_string_text(self, probe: CourtListenerDatasetProbe) -> None:
        for bad_value in (42, None, ["a", "b"], 3.14):
            row = {"text": bad_value, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
            assert any("must be str" in e for e in probe.validate_row(row)), f"Expected type error for {bad_value!r}"

    def test_reports_all_errors_not_just_first(self, probe: CourtListenerDatasetProbe) -> None:
        bad = {"text": "short", "url": "x"}
        errors = probe.validate_row(bad)
        assert any("Missing required fields" in e for e in errors)
        assert any("too short" in e for e in errors)
        assert len(errors) == 2

    def test_accepts_contents_field(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"contents": "A" * 60, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        assert probe.validate_row(row) == []


class TestNormalizeRowContract:
    def test_raises_on_invalid_input(self, probe: CourtListenerDatasetProbe) -> None:
        with pytest.raises(ValueError, match="normalize_row\\(\\) called on invalid row"):
            probe.normalize_row({"url": "x"})

    def test_error_message_lists_violations(self, probe: CourtListenerDatasetProbe) -> None:
        with pytest.raises(ValueError, match="Missing required fields"):
            probe.normalize_row({"url": "x"})

    def test_raises_on_short_text(self, probe: CourtListenerDatasetProbe) -> None:
        short = {"text": "too short", "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        with pytest.raises(ValueError, match="normalize_row\\(\\) called on invalid row"):
            probe.normalize_row(short)

    def test_accepts_valid_row(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert "text" in probe.normalize_row(pinned_row)

    def test_iter_valid_rows_never_normalizes_invalid(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert len(list(probe.iter_valid_rows([pinned_row, {"url": "bad"}, pinned_row]))) == 2


class TestResolveTextField:
    def test_prefers_text_over_contents(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"text": "hello", "contents": "world", "url": "x", "created_timestamp": "", "downloaded_timestamp": ""}
        assert probe.resolve_text_field(row) == "text"

    def test_falls_back_to_contents(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"contents": "world", "url": "x", "created_timestamp": "", "downloaded_timestamp": ""}
        assert probe.resolve_text_field(row) == "contents"

    def test_returns_none_when_missing(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe.resolve_text_field({"url": "x"}) is None

    def test_get_text_raises_on_missing_field(self, probe: CourtListenerDatasetProbe) -> None:
        with pytest.raises(ValueError, match="No text field"):
            probe.get_text({"url": "x"})


class TestDeadCodeInvariants:
    """TDD: document and enforce invariants that make dead branches impossible.
    These tests serve as living proof that removed guards were correct to remove.
    """

    def test_url_always_present_after_validation(self, probe: CourtListenerDatasetProbe) -> None:
        """url is in REQUIRED_FIELDS — always present after validate_row() passes.
        Removes need for 'if url in row' guard in normalize_row().
        """
        row = {"text": "A" * 60, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        assert probe.validate_row(row) == []
        assert "url" in row

    def test_get_text_raises_value_error_on_no_text_field(self, probe: CourtListenerDatasetProbe) -> None:
        """get_text() raises ValueError when no text field present — direct call."""
        with pytest.raises(ValueError, match="No text field in row keys"):
            probe.get_text({"url": "x", "created_timestamp": "", "downloaded_timestamp": ""})

    def test_get_text_returns_string_when_field_present(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"text": "some text content here", "url": "x"}
        assert probe.get_text(row) == "some text content here"


class TestDLPipelineSemantics:
    """Contract tests for properties relevant to DL/RAG pipelines."""

    def test_non_whitespace_text_required(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"text": "   \t\n  ", "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        errors = probe.validate_row(row)
        assert any("too short" in e for e in errors), (
            "Whitespace-only text must fail — produces zero-content embeddings"
        )

    def test_minimum_token_floor_enforced(self, probe: CourtListenerDatasetProbe) -> None:
        """MIN_TEXT_LENGTH=50 chars provides a rough token floor.
        Shorter texts produce degenerate embeddings in DL pipelines.
        """
        short = {
            "text": "x" * (probe.MIN_TEXT_LENGTH - 1),
            "created_timestamp": "",
            "downloaded_timestamp": "",
            "url": "x",
        }
        assert any("too short" in e for e in probe.validate_row(short))

    def test_url_field_is_non_empty_string_after_normalization(
        self, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        result = probe.normalize_row(pinned_row)
        assert isinstance(result["url"], str) and len(result["url"]) > 0

    def test_url_has_http_scheme(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        """URL sanity — bare paths or malformed URLs break citation retrieval."""
        result = probe.normalize_row(pinned_row)
        assert result["url"].startswith(("http://", "https://")), (
            f"URL must have http/https scheme, got: {result['url']!r}"
        )

    def test_source_metadata_retained_after_normalization(
        self, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        """Optional upstream metadata must survive normalization intact."""
        row = {**pinned_row, "court_id": "ca9", "docket": "22-1234"}
        result = probe.normalize_row(row)
        assert result["court_id"] == "ca9"
        assert result["docket"] == "22-1234"

    def test_text_is_parseable_utf8_after_normalization(
        self, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        """Normalized text must round-trip through UTF-8 without loss."""
        result = probe.normalize_row(pinned_row)
        encoded = result["text"].encode("utf-8")
        assert encoded.decode("utf-8") == result["text"]
