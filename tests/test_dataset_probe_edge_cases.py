# tests/test_dataset_probe_edge_cases.py
# Property-style and edge-case tests — messy real-world legal data scenarios.
import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.dataset_probe import CourtListenerDatasetProbe

pytestmark = pytest.mark.unit

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "courtlistener_sample.json"


@pytest.fixture
def probe() -> CourtListenerDatasetProbe:
    return CourtListenerDatasetProbe()


@pytest.fixture
def pinned_row() -> dict:
    data = json.loads(FIXTURE_PATH.read_text())
    return {k: v for k, v in data.items() if k != "_fixture_meta"}


class TestWeirdTimestamps:
    def test_whitespace_only_timestamp(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("   ") == ""

    def test_timestamp_with_extra_whitespace(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe._normalize_timestamp("  2022-01-15T10:30:00Z  ")
        assert "2022-01-15" in result

    def test_timestamp_zero_time(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("2022-01-15T00:00:00Z") == "2022-01-15T00:00:00Z"

    def test_timestamp_leap_day_valid(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("2024-02-29") == "2024-02-29"

    def test_timestamp_leap_day_invalid_year(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("2023-02-29") == ""

    def test_timestamp_end_of_year(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("2022-12-31") == "2022-12-31"

    def test_multiple_dates_in_string_picks_first(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe._normalize_timestamp("Filed 2022-01-15, amended 2022-06-01")
        assert result == "2022-01-15"

    def test_timestamp_with_milliseconds(self, probe: CourtListenerDatasetProbe) -> None:
        # Milliseconds not in format patterns — falls back gracefully
        result = probe._normalize_timestamp("2022-01-15T10:30:00.123Z")
        assert "2022-01-15" in result or result == ""


class TestWhitespaceAndUnicode:
    def test_whitespace_only_text_fails_validation(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"text": "   ", "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        errors = probe.validate_row(row)
        assert any("too short" in e for e in errors)

    def test_text_with_leading_trailing_whitespace_normalized(
        self, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        row = {**pinned_row, "text": "\n\t  " + pinned_row["text"] + "  \n"}
        result = probe.normalize_row(row)
        assert result["text"] == pinned_row["text"]

    def test_unicode_text_passes_validation(self, probe: CourtListenerDatasetProbe) -> None:
        unicode_text = "法院裁定被告未能证明存在真实的争议事实" * 3  # CJK, repeated to exceed MIN_TEXT_LENGTH
        row = {"text": unicode_text, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        assert probe.validate_row(row) == []

    def test_emoji_in_text_passes_validation(self, probe: CourtListenerDatasetProbe) -> None:
        text = "The court held ⚖️ that the defendant failed to establish " * 2
        row = {"text": text, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        assert probe.validate_row(row) == []

    def test_null_bytes_in_text_passes_type_check(self, probe: CourtListenerDatasetProbe) -> None:
        text = "court held\x00null byte present in legal document text field here"
        row = {"text": text, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        # Type check passes — null byte is a str; length check may still fail
        errors = probe.validate_row(row)
        assert not any("must be str" in e for e in errors)

    def test_rtl_text_passes_validation(self, probe: CourtListenerDatasetProbe) -> None:
        rtl_text = "المحكمة قررت أن المدعى عليه فشل في إثبات وجود خلاف حقيقي"
        row = {"text": rtl_text, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        assert probe.validate_row(row) == []


class TestGiganticText:
    def test_very_large_text_passes_validation(self, probe: CourtListenerDatasetProbe) -> None:
        giant = "The court held. " * 10_000  # ~160KB
        row = {"text": giant, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        assert probe.validate_row(row) == []

    def test_very_large_text_normalized_without_error(self, probe: CourtListenerDatasetProbe) -> None:
        giant = "The court held that the defendant. " * 10_000
        row = {"text": giant, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        result = probe.normalize_row(row)
        assert result["text"] == giant.strip()


class TestUnexpectedMetadataFields:
    def test_extra_fields_preserved_through_normalization(
        self, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        row = {
            **pinned_row,
            "court_id": "ca9",
            "docket_number": "21-1234",
            "judges": ["Smith", "Jones"],
            "precedential_status": "Published",
        }
        result = probe.normalize_row(row)
        assert result["court_id"] == "ca9"
        assert result["docket_number"] == "21-1234"
        assert result["judges"] == ["Smith", "Jones"]
        assert result["precedential_status"] == "Published"

    def test_deeply_nested_metadata_preserved(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        row = {**pinned_row, "meta": {"source": "scrape", "version": 2}}
        result = probe.normalize_row(row)
        assert result["meta"] == {"source": "scrape", "version": 2}

    def test_numeric_metadata_preserved(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        row = {**pinned_row, "word_count": 1234, "page_count": 12}
        result = probe.normalize_row(row)
        assert result["word_count"] == 1234


class TestPropertyBased:
    @given(st.text(min_size=0, max_size=49))
    @settings(max_examples=50)
    def test_short_text_always_fails_validation(self, text: str) -> None:
        probe = CourtListenerDatasetProbe()
        row = {"text": text, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        errors = probe.validate_row(row)
        # Either too short or non-string errors — never valid for text < MIN_TEXT_LENGTH
        assert len(errors) > 0

    @given(st.text(min_size=50, max_size=500, alphabet=st.characters(blacklist_categories=("Cs",))))
    @settings(max_examples=50)
    def test_valid_length_text_passes_type_and_length_checks(self, text: str) -> None:
        probe = CourtListenerDatasetProbe()
        row = {"text": text, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        errors = probe.validate_row(row)
        assert not any("must be str" in e for e in errors)
        assert not any("too short" in e for e in errors)

    @given(st.one_of(st.integers(), st.floats(allow_nan=False), st.lists(st.text())))
    @settings(max_examples=30)
    def test_non_string_text_always_fails_type_check(self, bad_value: object) -> None:
        probe = CourtListenerDatasetProbe()
        row = {"text": bad_value, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        errors = probe.validate_row(row)
        assert any("must be str" in e for e in errors)
