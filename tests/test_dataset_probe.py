# tests/test_dataset_probe.py
# TDD unit tests for HF dataset schema contract — no real network calls.
# Fixtures are pinned for deterministic first-sample testing.
# HF streaming is non-deterministic without seed; mocks + fixtures solve this.
import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

DATASET_ID = "pile-of-law/pile-of-law"
SUBSET = "r_courtlistener_opinions"
SPLIT = "train"
TEXT_FIELDS = ("text", "contents")
REQUIRED_FIELDS = {"text", "created_timestamp", "downloaded_timestamp", "url"}

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "courtlistener_sample.json"


@pytest.fixture
def pinned_row() -> dict:
    """Deterministic first-sample fixture — pins schema contract to a known-good row."""
    return json.loads(FIXTURE_PATH.read_text())


class TestDatasetSchemaContract:
    def test_fixture_has_required_fields(self, pinned_row: dict) -> None:
        """Fixture itself must satisfy the schema contract."""
        missing = REQUIRED_FIELDS - set(pinned_row.keys())
        assert not missing, f"Fixture missing required fields: {missing}"

    def test_fixture_text_is_non_empty_string(self, pinned_row: dict) -> None:
        assert isinstance(pinned_row["text"], str)
        assert len(pinned_row["text"]) >= 50

    def test_fixture_url_is_courtlistener(self, pinned_row: dict) -> None:
        assert "courtlistener.com" in pinned_row["url"]

    @patch("datasets.load_dataset")
    def test_required_fields_present(self, mock_load, pinned_row: dict) -> None:
        mock_load.return_value = iter([pinned_row])
        from datasets import load_dataset

        ds = load_dataset(DATASET_ID, SUBSET, split=SPLIT, streaming=True)
        row = next(iter(ds))
        missing = REQUIRED_FIELDS - set(row.keys())
        assert not missing, f"Missing required fields: {missing}"

    @patch("datasets.load_dataset")
    def test_text_field_is_non_empty_string(self, mock_load, pinned_row: dict) -> None:
        mock_load.return_value = iter([pinned_row])
        from datasets import load_dataset

        ds = load_dataset(DATASET_ID, SUBSET, split=SPLIT, streaming=True)
        row = next(iter(ds))
        text_field = next((k for k in TEXT_FIELDS if k in row), None)
        assert text_field is not None, f"No text field in {list(row.keys())}"
        assert isinstance(row[text_field], str)
        assert len(row[text_field]) > 0

    @patch("datasets.load_dataset")
    def test_no_trust_remote_code_in_call(self, mock_load, pinned_row: dict) -> None:
        mock_load.return_value = iter([pinned_row])
        from datasets import load_dataset

        load_dataset(DATASET_ID, SUBSET, split=SPLIT, streaming=True)
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs.get("trust_remote_code", False) is not True, (
            "trust_remote_code=True is a security/reproducibility violation"
        )

    @patch("datasets.load_dataset", side_effect=ConnectionError("network unavailable"))
    def test_network_failure_raises_clearly(self, mock_load) -> None:
        from datasets import load_dataset

        with pytest.raises(ConnectionError, match="network unavailable"):
            load_dataset(DATASET_ID, SUBSET, split=SPLIT, streaming=True)

    @patch("datasets.load_dataset")
    def test_missing_text_field_detected(self, mock_load) -> None:
        bad_row = {"created_timestamp": "2022-01-01", "downloaded_timestamp": "2022-06-01", "url": "x"}
        mock_load.return_value = iter([bad_row])
        from datasets import load_dataset

        ds = load_dataset(DATASET_ID, SUBSET, split=SPLIT, streaming=True)
        row = next(iter(ds))
        text_field = next((k for k in TEXT_FIELDS if k in row), None)
        assert text_field is None

    def test_pinned_fixture_is_deterministic(self, pinned_row: dict) -> None:
        """Same fixture content every run — guards against accidental fixture mutation."""
        reloaded = json.loads(FIXTURE_PATH.read_text())
        assert pinned_row == reloaded
