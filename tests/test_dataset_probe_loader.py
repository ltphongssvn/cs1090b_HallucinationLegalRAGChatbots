# tests/test_dataset_probe_loader.py
# Load, iter_valid_rows, and single-pass semantics tests.
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dataset_probe import CourtListenerDatasetProbe

pytestmark = pytest.mark.unit

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "courtlistener_sample.json"


def _mock_iterable_dataset(rows: list[dict]) -> MagicMock:
    """Re-iterable mock for test infrastructure — not a claim about HF behavior."""
    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    return mock_ds


@pytest.fixture
def probe() -> CourtListenerDatasetProbe:
    return CourtListenerDatasetProbe()


@pytest.fixture
def pinned_row() -> dict:
    data = json.loads(FIXTURE_PATH.read_text())
    return {k: v for k, v in data.items() if k != "_fixture_meta"}


class TestLoad:
    @patch("datasets.load_dataset")
    def test_uses_correct_dataset_id(self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
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
    def test_mock_is_re_iterable_for_test_infrastructure(
        self, mock_load, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        """Mock is re-iterable for argument inspection — not a production behavior claim.
        Production use must treat load() as single-pass per the docstring contract.
        """
        mock_load.return_value = _mock_iterable_dataset([pinned_row])
        ds = probe.load()
        assert list(ds) == list(ds) == [pinned_row]


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

    def test_pipeline_consumes_source_once(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        """Single-pass contract: iter_valid_rows() exhausts a bare iterator once.
        Callers must not assume the source is re-iterable — call load() again for
        a second pass.
        """
        single_pass = iter([pinned_row, pinned_row])
        assert len(list(probe.iter_valid_rows(single_pass))) == 2
        assert list(probe.iter_valid_rows(single_pass)) == []


class TestDataLoaderCompatibility:
    """Verify normalize_row output is directly usable in a PyTorch DataLoader."""

    def test_normalized_row_text_is_str(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        result = probe.normalize_row(pinned_row)
        assert isinstance(result["text"], str)

    def test_iter_valid_rows_produces_list_of_dicts(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        rows = list(probe.iter_valid_rows([pinned_row, pinned_row]))
        assert all(isinstance(r, dict) for r in rows)

    def test_normalized_batch_text_collatable(self, probe: CourtListenerDatasetProbe, pinned_row: dict) -> None:
        """Simulate collate_fn: batch of text strings must be a list of uniform type."""
        batch = list(probe.iter_valid_rows([pinned_row] * 4))
        texts = [r["text"] for r in batch]
        assert len(texts) == 4
        assert all(isinstance(t, str) for t in texts)
        # All texts from identical rows must be identical — deterministic collation
        assert len(set(texts)) == 1

    def test_normalized_row_has_no_none_values_in_required_fields(
        self, probe: CourtListenerDatasetProbe, pinned_row: dict
    ) -> None:
        """None values in required fields cause silent DataLoader collation failures."""
        result = probe.normalize_row(pinned_row)
        for field in ("text", "url", "source_url"):
            assert result[field] is not None, f"{field} must not be None"
