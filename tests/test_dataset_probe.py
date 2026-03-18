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
HEX_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

# Pinned to HEAD commit of pile-of-law/pile-of-law as of 2026-03-18.
# Update with:
#   from huggingface_hub import list_repo_commits
#   list(list_repo_commits('pile-of-law/pile-of-law', repo_type='dataset'))[0].commit_id
PINNED_REVISION = "0dc9f2c26b42af4cb6330f36d6146e82f9117a3b"  # pragma: allowlist secret

from tests.fixtures.courtlistener_checksums import COURTLISTENER_SAMPLE_TEXT_SHA256

# Mutable refs that must be rejected in REPRODUCIBLE mode.
_MUTABLE_REFS = {"main", "master", "latest", "HEAD", ""}


class CourtListenerDatasetProbe:
    """
    Schema contract and access layer for pile-of-law/pile-of-law
    subset r_courtlistener_opinions.

    Reproducibility contract:
      - REPRODUCIBLE=True (default) enforces a pinned 40-char SHA at load() time.
        Set REPRODUCIBLE=False only for fast exploration — never for training runs.
      - trust_remote_code is never passed — remote code execution is a security
        and reproducibility violation.
      - Provenance is probe-level — call get_provenance() once at training start
        and log to W&B; do not embed in data rows.
    """

    DATASET_ID = "pile-of-law/pile-of-law"
    SUBSET = "r_courtlistener_opinions"
    SPLIT = "train"
    REVISION = PINNED_REVISION
    PROBE_VERSION = "1.0"
    REPRODUCIBLE = True  # set False only for exploration — never for training
    REQUIRED_FIELDS: frozenset[str] = frozenset({"text", "created_timestamp", "downloaded_timestamp", "url"})
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
        """Normalize validated row into canonical form, preserving upstream metadata.

        Shallow copy preserves all upstream fields (jurisdiction, court level, judge).
        Old text field is removed if renamed to 'text' to avoid RAM duplication.
        Provenance is NOT embedded — call get_provenance() at training start.
        """
        normalized = dict(row)

        text_field = self.resolve_text_field(row)
        text = str(row[text_field]).strip() if text_field else ""

        if text_field and text_field != "text":
            normalized.pop(text_field, None)

        normalized["text"] = text
        normalized["created_timestamp"] = self._normalize_timestamp(str(row.get("created_timestamp", "")))
        normalized["downloaded_timestamp"] = self._normalize_timestamp(str(row.get("downloaded_timestamp", "")))
        if "url" in row:
            normalized["source_url"] = str(row["url"])

        return normalized

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

    def test_fixture_revision_matches_probe_revision(self, fixture_data: dict) -> None:
        """Fixture and probe must agree on revision — prevents silent sample drift."""
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

    def test_revision_must_not_be_mutable_branch(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe.REVISION not in _MUTABLE_REFS

    def test_revision_is_40char_sha(self, probe: CourtListenerDatasetProbe) -> None:
        assert HEX_REVISION_RE.fullmatch(probe.REVISION) is not None, (
            f"PINNED_REVISION must be a 40-char SHA, got: {probe.REVISION!r}"
        )

    def test_fixture_text_sha256_matches_checksum_module(self, pinned_row: dict) -> None:
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


class TestReproducibilityEnforcement:
    """Guardrail tests — ensure mutable refs are mechanically impossible in REPRODUCIBLE mode."""

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
        """7-char short SHAs are also mutable — require full 40-char SHA."""
        probe = CourtListenerDatasetProbe()
        probe.REPRODUCIBLE = True
        probe.REVISION = "0dc9f2c"  # short SHA — rejected
        with pytest.raises(RuntimeError, match="Reproducibility violation"):
            list(probe.load())

    @patch("datasets.load_dataset")
    def test_exploration_mode_allows_mutable_ref(self, mock_load, pinned_row: dict) -> None:
        """REPRODUCIBLE=False is the explicit opt-in for non-deterministic exploration."""
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        probe = CourtListenerDatasetProbe()
        probe.REPRODUCIBLE = False
        probe.REVISION = "main"
        list(probe.load())  # must not raise

    @patch("datasets.load_dataset")
    def test_reproducible_mode_accepts_valid_40char_sha(self, mock_load, pinned_row: dict) -> None:
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        probe = CourtListenerDatasetProbe()
        probe.REPRODUCIBLE = True
        probe.REVISION = PINNED_REVISION
        list(probe.load())  # must not raise

    def test_default_mode_is_reproducible(self, probe: CourtListenerDatasetProbe) -> None:
        """Default must be strict — exploration mode requires explicit opt-in."""
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
        assert "probe_version" not in result


class TestIterValidRows:
    def test_yields_only_valid_rows(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        rows = list(probe.iter_valid_rows([pinned_row, {"url": "x"}]))
        assert len(rows) == 1

    def test_yields_normalized_rows(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        rows = list(probe.iter_valid_rows([pinned_row]))
        expected = {"text", "created_timestamp", "downloaded_timestamp", "url", "source_url"}
        assert expected.issubset(set(rows[0].keys()))

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
        result = probe.normalize_row(pinned_row)
        assert isinstance(result["text"], str) and len(result["text"]) > 0

    def test_strips_whitespace(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"text": "  leading and trailing  ", "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        assert probe.normalize_row(row)["text"] == "leading and trailing"

    def test_renames_contents_to_text_and_removes_old_field(self, probe: CourtListenerDatasetProbe) -> None:
        row = {"contents": "A" * 60, "created_timestamp": "", "downloaded_timestamp": "", "url": "x"}
        result = probe.normalize_row(row)
        assert "text" in result
        assert "contents" not in result

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

    def test_output_has_canonical_keys_as_superset(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        row = {**pinned_row, "dummy_judge": "Smith"}
        result = probe.normalize_row(row)
        expected = {"text", "created_timestamp", "downloaded_timestamp", "url", "source_url"}
        assert expected.issubset(set(result.keys()))
        assert result.get("dummy_judge") == "Smith"

    def test_idempotent(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        first = probe.normalize_row(pinned_row)
        assert first["text"] == probe.normalize_row(first)["text"]

    def test_does_not_embed_provenance(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        result = probe.normalize_row(pinned_row)
        assert "_provenance" not in result
        assert "revision" not in result


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
