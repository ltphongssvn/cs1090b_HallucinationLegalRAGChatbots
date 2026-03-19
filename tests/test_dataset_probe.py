# tests/test_dataset_probe.py
# TDD unit tests for HF dataset schema contract — no real network calls.
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "courtlistener_sample.json"
HEX_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

# Pinned to HEAD commit of pile-of-law/pile-of-law as of 2026-03-18.
# Update with:
#   from huggingface_hub import list_repo_commits
#   list(list_repo_commits('pile-of-law/pile-of-law', repo_type='dataset'))[0].commit_id
PINNED_REVISION = "0dc9f2c26b42af4cb6330f36d6146e82f9117a3b"  # pragma: allowlist secret

from tests.fixtures.courtlistener_checksums import COURTLISTENER_SAMPLE_TEXT_SHA256

_MUTABLE_REFS = {"main", "master", "latest", "HEAD", ""}

# Candidate formats tried in order from most to least precise.
# Each entry: (strptime format string, preserves_time, preserves_tz)
# datetime.fromisoformat() handles most ISO-8601 variants in Python 3.11+,
# but we try explicit formats first for clarity and test coverage.
_TS_FORMATS: list[tuple[str, bool, bool]] = [
    ("%Y-%m-%dT%H:%M:%S%z", True, True),  # 2022-01-15T10:30:00+05:00 or Z
    ("%Y-%m-%dT%H:%M:%S", True, False),  # 2022-01-15T10:30:00
    ("%Y-%m-%d", False, False),  # 2022-01-15
]

# Regex to extract candidate substrings before parsing — avoids feeding
# entire free-text fields to strptime.
_TS_EXTRACT_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2}|[+-]\d{4}|\.\d+Z?)?)?"
)


class CourtListenerDatasetProbe:
    """
    Schema contract and access layer for pile-of-law/pile-of-law
    subset r_courtlistener_opinions.

    Reproducibility contract:
      - REPRODUCIBLE=True (default) enforces a pinned 40-char SHA at load() time.
        Set REPRODUCIBLE=False only for fast exploration — never for training runs.
      - trust_remote_code is never passed.
      - Provenance is probe-level — call get_provenance() once at training start.

    validate_row / normalize_row contract:
      - validate_row() returns all errors; empty list means valid.
      - normalize_row() requires a pre-validated row. Raises ValueError if
        validation fails — callers cannot accidentally normalize invalid rows.
      - iter_valid_rows() is the preferred pipeline entry point.

    REQUIRED_FIELDS intentionally excludes text-variant keys ('text', 'contents').
    Text field presence and type are enforced separately via resolve_text_field().
    """

    DATASET_ID = "pile-of-law/pile-of-law"
    SUBSET = "r_courtlistener_opinions"
    SPLIT = "train"
    REVISION = PINNED_REVISION
    PROBE_VERSION = "1.0"
    REPRODUCIBLE = True
    REQUIRED_FIELDS: frozenset[str] = frozenset({"created_timestamp", "downloaded_timestamp", "url"})
    TEXT_FIELDS: tuple[str, ...] = ("text", "contents")
    MIN_TEXT_LENGTH = 50

    def load(self, streaming: bool = True) -> Iterable[dict[str, Any]]:
        """Load dataset at pinned revision. trust_remote_code never passed.

        Raises RuntimeError if REPRODUCIBLE=True and REVISION is a mutable ref.
        Single-pass semantics: wrap in iter() to enforce single-pass explicitly.
        """
        if self.REPRODUCIBLE and (self.REVISION in _MUTABLE_REFS or HEX_REVISION_RE.fullmatch(self.REVISION) is None):
            raise RuntimeError(
                f"Reproducibility violation: REVISION={self.REVISION!r} is mutable. "
                "Set REVISION to a 40-char commit SHA, or set REPRODUCIBLE=False "
                "to explicitly opt into non-deterministic exploration mode."
            )

        from datasets import load_dataset

        return load_dataset(  # type: ignore[return-value]
            self.DATASET_ID,
            self.SUBSET,
            split=self.SPLIT,
            streaming=streaming,
            revision=self.REVISION,
        )

    def get_provenance(self) -> dict[str, Any]:
        """Return full provenance dict for W&B / experiment logging.
        Provenance is probe-level — log once at training start, not per-row.
        """
        import datasets

        return {
            "dataset": self.DATASET_ID,
            "subset": self.SUBSET,
            "split": self.SPLIT,
            "revision": self.REVISION,
            "hf_datasets_version": datasets.__version__,
            "probe_version": self.PROBE_VERSION,
            "reproducible": self.REPRODUCIBLE,
        }

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
        """Normalize a pre-validated row into canonical form.

        Precondition: row must pass validate_row() with no errors.
        Raises ValueError if precondition is violated.
        Use iter_valid_rows() to enforce the contract automatically.
        """
        errors = self.validate_row(row)
        if errors:
            raise ValueError(
                f"normalize_row() called on invalid row — validate first.\n"
                f"Errors: {errors}\n"
                f"Use iter_valid_rows() to enforce validate-then-normalize automatically."
            )

        normalized = dict(row)

        text_field = self.resolve_text_field(row)
        text = str(row[text_field]).strip() if text_field else ""

        if text_field and text_field != "text":
            normalized.pop(text_field, None)

        normalized["text"] = text
        normalized["created_timestamp"] = self._normalize_timestamp(str(row.get("created_timestamp", "")))
        normalized["downloaded_timestamp"] = self._normalize_timestamp(str(row.get("downloaded_timestamp", "")))
        if "url" in row:
            # source_url is an intentional pipeline alias for url.
            # Downstream RAG components reference source_url consistently,
            # decoupling them from the raw field name which varies across
            # pile-of-law subsets (some use url, others use href or link).
            normalized["source_url"] = str(row["url"])

        return normalized

    def _normalize_timestamp(self, ts: str) -> str:
        """Parse and normalize timestamp using datetime — not just regex extraction.

        Validates actual date semantics (rejects '9999-99-99', '2022-13-45', etc.).
        Preserves the most precise valid format found:
          datetime+tz > datetime > date > '' (unparseable or invalid).

        Legal documents carry filing deadlines and appeal windows that are
        time-sensitive — silently dropping timezone or accepting invalid dates
        loses forensic value.
        """
        candidate = _TS_EXTRACT_RE.search(ts)
        if not candidate:
            return ""
        raw = candidate.group(0)

        # Normalize 'Z' suffix — Python < 3.11 fromisoformat doesn't accept it
        normalized_raw = raw.replace("Z", "+00:00")

        for fmt, preserves_time, preserves_tz in _TS_FORMATS:
            try:
                parsed = datetime.strptime(normalized_raw, fmt)
                # Re-emit in the precision level that was successfully parsed
                if preserves_tz and parsed.tzinfo is not None:
                    # Emit as ISO-8601 with original Z if UTC
                    if parsed.tzinfo == timezone.utc:
                        return parsed.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
                    return parsed.isoformat()
                if preserves_time:
                    return parsed.strftime("%Y-%m-%dT%H:%M:%S")
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return ""

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
        """datetime parsing rejects invalid dates — pure regex would accept these."""
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
        assert "_provenance" not in result
        assert "revision" not in result


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
        assert result.get("judge") == "Smith J."
        assert result.get("jurisdiction") == "federal"

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
        assert "2022-06-01" in result["downloaded_timestamp"]

    def test_falls_back_to_date_only_when_no_time(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        row = {**pinned_row, "created_timestamp": "2022-01-15", "downloaded_timestamp": "2022-06-01"}
        result = probe.normalize_row(row)
        assert result["created_timestamp"] == "2022-01-15"
        assert result["downloaded_timestamp"] == "2022-06-01"

    def test_adds_source_url_alias(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        assert probe.normalize_row(pinned_row)["source_url"] == pinned_row["url"]

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
