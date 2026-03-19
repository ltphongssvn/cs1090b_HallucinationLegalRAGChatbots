# tests/test_dataset_probe_normalization.py
# Normalization and timestamp tests for CourtListenerDatasetProbe.
import json
from pathlib import Path

import pytest

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


class TestNormalizeRow:
    def test_preserves_upstream_metadata(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        row = {**pinned_row, "judge": "Smith J.", "jurisdiction": "federal"}
        result = probe.normalize_row(row)
        assert result.get("judge") == "Smith J." and result.get("jurisdiction") == "federal"

    def test_canonical_text_key(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert isinstance(probe.normalize_row(pinned_row)["text"], str)

    def test_strips_whitespace(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        row = {**pinned_row, "text": "  " + pinned_row["text"] + "  "}
        assert probe.normalize_row(row)["text"] == pinned_row["text"]

    def test_renames_contents_to_text_and_removes_old_field(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"contents": "A" * 60, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        result = probe.normalize_row(row)
        assert "text" in result and "contents" not in result

    def test_preserves_full_datetime_in_timestamps(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        row = {
            **pinned_row,
            "created_timestamp": "2022-01-15T10:30:00Z",
            "downloaded_timestamp": "2022-06-01T08:00:00+05:00",
        }
        result = probe.normalize_row(row)
        assert "2022-01-15" in result["created_timestamp"] and "10:30:00" in result["created_timestamp"]

    def test_falls_back_to_date_only_when_no_time(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        row = {**pinned_row, "created_timestamp": "2022-01-15", "downloaded_timestamp": "2022-06-01"}
        assert probe.normalize_row(row)["created_timestamp"] == "2022-01-15"

    def test_source_url_is_alias_for_url(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        """source_url is an intentional pipeline alias — decouples downstream from raw field name."""
        result = probe.normalize_row(pinned_row)
        assert result["source_url"] == result["url"] == pinned_row["url"]

    def test_output_has_canonical_keys_as_superset(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        row = {**pinned_row, "dummy_judge": "Smith"}
        result = probe.normalize_row(row)
        assert {"text", "created_timestamp", "downloaded_timestamp", "url", "source_url"}.issubset(set(result.keys()))
        assert result.get("dummy_judge") == "Smith"

    def test_idempotent(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        first = probe.normalize_row(pinned_row)
        assert first["text"] == probe.normalize_row(first)["text"]

    def test_does_not_embed_provenance(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        result = probe.normalize_row(pinned_row)
        assert "_provenance" not in result and "revision" not in result


class TestNormalizeTimestamp:
    def test_preserves_full_datetime_with_utc(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("2022-01-15T10:30:00Z") == "2022-01-15T10:30:00Z"

    def test_preserves_full_datetime_with_tz_offset(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe._normalize_timestamp("2022-01-15T10:30:00+05:00")
        assert "2022-01-15" in result and "10:30:00" in result

    def test_preserves_datetime_without_tz(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("2022-01-15T10:30:00") == "2022-01-15T10:30:00"

    def test_falls_back_to_date_only(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("2022-01-15") == "2022-01-15"

    def test_extracts_from_surrounding_text(self, probe: CourtListenerDatasetProbe) -> None:
        result = probe._normalize_timestamp("Filed on 2022-01-15T10:30:00Z per docket")
        assert "2022-01-15" in result and "10:30:00" in result

    def test_prefers_more_precise_over_less(self, probe: CourtListenerDatasetProbe) -> None:
        assert "T" in probe._normalize_timestamp("2022-01-15T10:30:00Z")

    def test_rejects_impossible_month(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("2022-13-01") == ""

    def test_rejects_impossible_day(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("2022-01-99") == ""

    def test_rejects_obviously_invalid_date(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("9999-99-99") == ""

    def test_empty_string_returns_empty(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("") == ""

    def test_unparseable_returns_empty(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe._normalize_timestamp("no date here") == ""
