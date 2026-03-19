# tests/test_dataset_probe.py
# Path: cs1090b_HallucinationLegalRAGChatbots/tests/test_dataset_probe.py
# Behavior tests for CourtListenerDatasetProbe — no real network calls.
import hashlib
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dataset_probe import (
    _MUTABLE_REFS,
    HEX_REVISION_RE,
    PINNED_REVISION,
    CourtListenerDatasetProbe,
)
from tests.fixtures.courtlistener_checksums import COURTLISTENER_SAMPLE_TEXT_SHA256

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

    def test_fixture_revision_matches_probe_revision(self, fixture_data: dict) -> None:
        assert fixture_data["_fixture_meta"]["revision"] == CourtListenerDatasetProbe.REVISION

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

    def test_validate_row_accepts_contents_field(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"contents": "A" * 60, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        assert probe.validate_row(row) == []


class TestNormalizeRowContract:
    def test_normalize_row_raises_on_invalid_input(self, probe: CourtListenerDatasetProbe) -> None:
        with pytest.raises(ValueError, match="normalize_row\\(\\) called on invalid row"):
            probe.normalize_row({"url": "x"})

    def test_normalize_row_error_message_lists_violations(self, probe: CourtListenerDatasetProbe) -> None:
        with pytest.raises(ValueError, match="Missing required fields"):
            probe.normalize_row({"url": "x"})

    def test_normalize_row_raises_on_short_text(self, probe: CourtListenerDatasetProbe) -> None:
        short = {"text": "too short", "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        with pytest.raises(ValueError, match="normalize_row\\(\\) called on invalid row"):
            probe.normalize_row(short)

    def test_normalize_row_accepts_valid_row(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert "text" in probe.normalize_row(pinned_row)

    def test_iter_valid_rows_never_calls_normalize_on_invalid(
        self, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        assert len(list(probe.iter_valid_rows([pinned_row, {"url": "bad"}, pinned_row]))) == 2


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


class TestReproducibilityContract:
    def test_fixture_has_required_provenance_fields(self, fixture_data: dict) -> None:
        meta = fixture_data["_fixture_meta"]
        required = {
            "sampled_at",
            "dataset",
            "subset",
            "split",
            "revision",
            "text_sha256",
            "hf_datasets_version",
            "probe_version",
        }
        assert required <= set(meta.keys()), f"Missing: {required - set(meta.keys())}"

    def test_fixture_metadata_matches_probe_constants(
        self, probe: CourtListenerDatasetProbe, fixture_data: dict
    ) -> None:
        meta = fixture_data["_fixture_meta"]
        assert meta["dataset"] == probe.DATASET_ID
        assert meta["subset"] == probe.SUBSET
        assert meta["split"] == probe.SPLIT
        assert meta["revision"] == probe.REVISION
        assert meta["probe_version"] == probe.PROBE_VERSION

    def test_fixture_text_sha256_matches_checksum_module(self, pinned_row: dict) -> None:
        actual = hashlib.sha256(pinned_row["text"].encode("utf-8")).hexdigest()
        assert actual == COURTLISTENER_SAMPLE_TEXT_SHA256

    def test_fixture_json_sha256_matches_checksum_module(self, fixture_data: dict) -> None:
        sha_in_json = fixture_data["_fixture_meta"]["text_sha256"]
        assert sha_in_json != "REPLACE_WITH_ACTUAL_SHA256", "text_sha256 is still a placeholder"
        assert sha_in_json == COURTLISTENER_SAMPLE_TEXT_SHA256

    def test_revision_must_not_be_mutable_branch(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe.REVISION not in _MUTABLE_REFS

    def test_revision_is_40char_sha(self, probe: CourtListenerDatasetProbe) -> None:
        assert HEX_REVISION_RE.fullmatch(probe.REVISION) is not None

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


class TestReproducibilityEnforcement:
    @patch("datasets.load_dataset")
    def test_reproducible_mode_rejects_main(self, mock_load, pinned_row: dict) -> None:
        probe = CourtListenerDatasetProbe()
        probe.REPRODUCIBLE = True
        probe.REVISION = "main"
        with pytest.raises(RuntimeError, match="Reproducibility violation"):
            list(probe.load())

    @patch("datasets.load_dataset")
    def test_reproducible_mode_rejects_all_mutable_refs(self, mock_load, pinned_row: dict) -> None:
        for bad_ref in _MUTABLE_REFS:
            probe = CourtListenerDatasetProbe()
            probe.REPRODUCIBLE = True
            probe.REVISION = bad_ref
            with pytest.raises(RuntimeError, match="Reproducibility violation"):
                list(probe.load())

    @patch("datasets.load_dataset")
    def test_reproducible_mode_rejects_short_sha(self, mock_load, pinned_row: dict) -> None:
        probe = CourtListenerDatasetProbe()
        probe.REPRODUCIBLE = True
        probe.REVISION = "0dc9f2c"
        with pytest.raises(RuntimeError, match="Reproducibility violation"):
            list(probe.load())

    @patch("datasets.load_dataset")
    def test_exploration_mode_allows_mutable_ref(self, mock_load, pinned_row: dict) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        probe = CourtListenerDatasetProbe()
        probe.REPRODUCIBLE = False
        probe.REVISION = "main"
        list(probe.load())

    @patch("datasets.load_dataset")
    def test_reproducible_mode_accepts_valid_40char_sha(self, mock_load, pinned_row: dict) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        probe = CourtListenerDatasetProbe()
        probe.REPRODUCIBLE = True
        probe.REVISION = PINNED_REVISION
        list(probe.load())

    def test_default_mode_is_reproducible(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe.REPRODUCIBLE is True


class TestGetProvenance:
    def test_get_provenance_returns_dict(self, probe: CourtListenerDatasetProbe) -> None:
        assert isinstance(probe.get_provenance(), dict)

    def test_get_provenance_has_required_keys(self, probe: CourtListenerDatasetProbe) -> None:
        required = {"dataset", "subset", "split", "revision", "hf_datasets_version", "probe_version", "reproducible"}
        assert required <= set(probe.get_provenance().keys())

    def test_get_provenance_matches_probe_constants(self, probe: CourtListenerDatasetProbe) -> None:
        prov = probe.get_provenance()
        assert prov["dataset"] == probe.DATASET_ID
        assert prov["revision"] == probe.REVISION
        assert prov["probe_version"] == probe.PROBE_VERSION
        assert prov["reproducible"] == probe.REPRODUCIBLE

    def test_normalize_row_does_not_embed_provenance(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        result = probe.normalize_row(pinned_row)
        assert "_provenance" not in result and "revision" not in result


class TestIterValidRows:
    def test_yields_only_valid_rows(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert len(list(probe.iter_valid_rows([pinned_row, {"url": "x"}]))) == 1

    def test_yields_normalized_rows(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        rows = list(probe.iter_valid_rows([pinned_row]))
        assert {"text", "created_timestamp", "downloaded_timestamp", "url", "source_url"}.issubset(set(rows[0].keys()))

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
