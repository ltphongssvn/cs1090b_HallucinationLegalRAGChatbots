# tests/test_dataset_probe_reproducibility.py
# Reproducibility, provenance, and revision-pinning tests.
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


class TestFixtureProvenance:
    def test_fixture_is_deterministic(self, pinned_row: dict) -> None:
        reloaded = {k: v for k, v in json.loads(FIXTURE_PATH.read_text()).items() if k != "_fixture_meta"}
        assert pinned_row == reloaded

    def test_fixture_url_is_courtlistener(self, pinned_row: dict) -> None:
        assert "courtlistener.com" in pinned_row["url"]

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

    def test_fixture_revision_matches_probe_revision(self, fixture_data: dict) -> None:
        assert fixture_data["_fixture_meta"]["revision"] == CourtListenerDatasetProbe.REVISION

    def test_fixture_text_sha256_matches_checksum_module(self, pinned_row: dict) -> None:
        actual = hashlib.sha256(pinned_row["text"].encode("utf-8")).hexdigest()
        assert actual == COURTLISTENER_SAMPLE_TEXT_SHA256

    def test_fixture_json_sha256_matches_checksum_module(self, fixture_data: dict) -> None:
        sha_in_json = fixture_data["_fixture_meta"]["text_sha256"]
        assert sha_in_json != "REPLACE_WITH_ACTUAL_SHA256", "text_sha256 is still a placeholder"
        assert sha_in_json == COURTLISTENER_SAMPLE_TEXT_SHA256

    def test_sampled_at_is_iso_date(self, fixture_data: dict) -> None:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", fixture_data["_fixture_meta"]["sampled_at"])


class TestRevisionPinning:
    def test_revision_must_not_be_mutable_branch(self, probe: CourtListenerDatasetProbe) -> None:
        assert probe.REVISION not in _MUTABLE_REFS

    def test_revision_is_40char_sha(self, probe: CourtListenerDatasetProbe) -> None:
        assert HEX_REVISION_RE.fullmatch(probe.REVISION) is not None

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
    def test_returns_dict(self, probe: CourtListenerDatasetProbe) -> None:
        assert isinstance(probe.get_provenance(), dict)

    def test_has_required_keys(self, probe: CourtListenerDatasetProbe) -> None:
        required = {"dataset", "subset", "split", "revision", "hf_datasets_version", "probe_version", "reproducible"}
        assert required <= set(probe.get_provenance().keys())

    def test_matches_probe_constants(self, probe: CourtListenerDatasetProbe) -> None:
        prov = probe.get_provenance()
        assert prov["dataset"] == probe.DATASET_ID
        assert prov["revision"] == probe.REVISION
        assert prov["probe_version"] == probe.PROBE_VERSION
        assert prov["reproducible"] == probe.REPRODUCIBLE

    def test_normalize_row_does_not_embed_provenance(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        result = probe.normalize_row(pinned_row)
        assert "_provenance" not in result and "revision" not in result
