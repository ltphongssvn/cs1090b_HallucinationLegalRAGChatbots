# tests/test_dataset_probe_fixtures.py
# Fixture-suite tests — distribution-representative coverage for modeling reliability.
# One pinned sample is good for determinism; multiple fixtures cover real data patterns.
import pytest

from src.dataset_probe import CourtListenerDatasetProbe
from tests.fixtures.courtlistener_fixtures import (
    CONTENTS_FIELD_INSTEAD_OF_TEXT,
    EDGE_CASE_URL,
    EXTRA_METADATA_FIELDS,
    INVALID_FIXTURES,
    MALFORMED_TIMESTAMPS,
    MINIMAL_VALID_TEXT,
    MISSING_TIMESTAMPS,
    VALID_FIXTURES,
    WEIRDLY_FORMATTED_TEXT,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def probe() -> CourtListenerDatasetProbe:
    return CourtListenerDatasetProbe()


class TestValidFixtures:
    @pytest.mark.parametrize(
        "row",
        VALID_FIXTURES,
        ids=[
            "standard_opinion",
            "weirdly_formatted",
            "contents_field",
            "missing_timestamps",
            "malformed_timestamps",
            "edge_case_url",
            "extra_metadata",
        ],
    )
    def test_valid_fixture_passes_validate_row(self, probe: CourtListenerDatasetProbe, row: dict) -> None:
        assert probe.validate_row(row) == []

    @pytest.mark.parametrize(
        "row",
        VALID_FIXTURES,
        ids=[
            "standard_opinion",
            "weirdly_formatted",
            "contents_field",
            "missing_timestamps",
            "malformed_timestamps",
            "edge_case_url",
            "extra_metadata",
        ],
    )
    def test_valid_fixture_normalizes_without_error(self, probe: CourtListenerDatasetProbe, row: dict) -> None:
        result = probe.normalize_row(row)
        assert "text" in result
        assert "url" in result
        assert "source_url" in result
        assert isinstance(result["text"], str)


class TestInvalidFixtures:
    def test_minimal_text_fails_validation(self, probe: CourtListenerDatasetProbe) -> None:
        """Text shorter than MIN_TEXT_LENGTH must be rejected — too little content for RAG."""
        errors = probe.validate_row(MINIMAL_VALID_TEXT)
        assert any("too short" in e for e in errors)

    @pytest.mark.parametrize("row", INVALID_FIXTURES, ids=["minimal_text"])
    def test_invalid_fixture_raises_on_normalize(self, probe: CourtListenerDatasetProbe, row: dict) -> None:
        with pytest.raises(ValueError, match="normalize_row\\(\\) called on invalid row"):
            probe.normalize_row(row)


class TestContentsFieldCoverage:
    def test_contents_field_resolved_and_renamed(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe.normalize_row(CONTENTS_FIELD_INSTEAD_OF_TEXT)
        assert "text" in result
        assert "contents" not in result
        assert len(result["text"]) >= probe.MIN_TEXT_LENGTH

    def test_contents_field_text_stripped(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe.normalize_row(CONTENTS_FIELD_INSTEAD_OF_TEXT)
        assert result["text"] == result["text"].strip()


class TestTimestampEdgeCases:
    def test_empty_timestamps_normalize_to_empty_string(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe.normalize_row(MISSING_TIMESTAMPS)
        assert result["created_timestamp"] == ""
        assert result["downloaded_timestamp"] == ""

    def test_malformed_timestamps_normalize_to_empty_string(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe.normalize_row(MALFORMED_TIMESTAMPS)
        assert result["created_timestamp"] == ""
        assert result["downloaded_timestamp"] == ""

    def test_non_utc_timestamp_preserves_offset(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe.normalize_row(EXTRA_METADATA_FIELDS)
        assert "2023-02-14" in result["created_timestamp"]
        assert "14:00:00" in result["created_timestamp"]


class TestWeirdlyFormattedText:
    def test_weirdly_formatted_text_stripped(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe.normalize_row(WEIRDLY_FORMATTED_TEXT)
        assert not result["text"].startswith("\n")
        assert not result["text"].endswith(" ")

    def test_weirdly_formatted_text_content_preserved(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe.normalize_row(WEIRDLY_FORMATTED_TEXT)
        assert "UNITED STATES DISTRICT COURT" in result["text"]


class TestEdgeCaseURL:
    def test_url_with_query_and_fragment_preserved(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe.normalize_row(EDGE_CASE_URL)
        assert result["url"] == EDGE_CASE_URL["url"]
        assert result["source_url"] == EDGE_CASE_URL["url"]


class TestExtraMetadataFields:
    def test_domain_metadata_preserved_for_rag_pipeline(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe.normalize_row(EXTRA_METADATA_FIELDS)
        assert result["court_id"] == "ca9"
        assert result["docket_number"] == "22-5678"
        assert result["judges"] == ["Smith J.", "Jones J."]
        assert result["precedential_status"] == "Published"
        assert result["citations"] == ["42 F.3d 100", "55 F.3d 200"]
