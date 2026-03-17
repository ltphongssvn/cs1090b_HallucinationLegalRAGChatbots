import pytest

pytestmark = pytest.mark.integration

# tests/test_filter_logic.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_filter_logic.py
# TDD RED: Unit tests for filter chain logic with synthetic CSV data.


import pandas as pd
import pytest

from src.filter_chain import load_federal_courts


class TestLoadFederalCourts:
    """Court filtering must match exactly the 13 federal appellate IDs."""

    def _write_courts_csv(self, tmp_path, rows):
        df = pd.DataFrame(rows)
        path = tmp_path / "courts.csv"
        df.to_csv(path, index=False)
        return path

    def test_filters_only_federal_appellate(self, tmp_path):
        path = self._write_courts_csv(
            tmp_path,
            [
                {"id": "ca1", "full_name": "First Circuit"},
                {"id": "ca9", "full_name": "Ninth Circuit"},
                {"id": "nysd", "full_name": "Southern District of New York"},
                {"id": "scotus", "full_name": "Supreme Court"},
            ],
        )
        court_ids, name_map = load_federal_courts(path)
        assert court_ids == {"ca1", "ca9"}
        assert "nysd" not in court_ids
        assert "scotus" not in court_ids

    def test_returns_name_map(self, tmp_path):
        path = self._write_courts_csv(
            tmp_path,
            [
                {"id": "cadc", "full_name": "D.C. Circuit"},
            ],
        )
        _, name_map = load_federal_courts(path)
        assert name_map["cadc"] == "D.C. Circuit"

    def test_empty_csv_returns_empty(self, tmp_path):
        path = self._write_courts_csv(
            tmp_path,
            [
                {"id": "nysd", "full_name": "Not Federal Appellate"},
            ],
        )
        court_ids, _ = load_federal_courts(path)
        assert len(court_ids) == 0
