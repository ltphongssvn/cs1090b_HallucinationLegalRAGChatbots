import pytest

pytestmark = pytest.mark.unit

# tests/test_s3_discovery.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_s3_discovery.py
# TDD RED: Unit tests for S3 discovery logic (no network calls).

from datetime import date

import pytest

from src.s3_discovery import _parse_bulk_file, find_latest_file


class TestParseBulkFile:
    def test_valid_csv_bz2(self):
        result = _parse_bulk_file("bulk-data/opinions-2025-03-31.csv.bz2")
        assert result["name"] == "opinions"
        assert result["date"] == date(2025, 3, 31)

    def test_valid_csv_uncompressed(self):
        result = _parse_bulk_file("bulk-data/courts-2024-12-31.csv")
        assert result["name"] == "courts"
        assert result["date"] == date(2024, 12, 31)

    def test_hyphenated_name(self):
        result = _parse_bulk_file("bulk-data/opinion-clusters-2025-06-30.csv.bz2")
        assert result["name"] == "opinion-clusters"

    def test_rejects_non_bulk_path(self):
        assert _parse_bulk_file("embeddings/something.csv") is None

    def test_rejects_no_date(self):
        assert _parse_bulk_file("bulk-data/opinions.csv") is None

    def test_rejects_bad_date(self):
        assert _parse_bulk_file("bulk-data/opinions-9999-99-99.csv") is None


class TestFindLatestFile:
    SAMPLE_FILES = [
        {"key": "bulk-data/opinions-2024-06-30.csv.bz2", "size": 100, "size_mb": 0.1},
        {"key": "bulk-data/opinions-2025-03-31.csv.bz2", "size": 200, "size_mb": 0.2},
        {"key": "bulk-data/opinions-2024-12-31.csv.bz2", "size": 150, "size_mb": 0.15},
        {"key": "bulk-data/courts-2025-03-31.csv.bz2", "size": 10, "size_mb": 0.01},
    ]

    def test_finds_most_recent_by_date(self):
        result = find_latest_file(self.SAMPLE_FILES, "opinions-")
        assert result["date"] == "2025-03-31"

    def test_does_not_cross_match_prefixes(self):
        result = find_latest_file(self.SAMPLE_FILES, "courts-")
        assert result["name"] == "courts"

    def test_returns_none_for_missing(self):
        assert find_latest_file(self.SAMPLE_FILES, "nonexistent-") is None

    def test_date_sorting_beats_lexicographic(self):
        """2025-03-31 > 2024-12-31, even though '12' > '03' lexicographically."""
        files = [
            {"key": "bulk-data/opinions-2024-12-31.csv.bz2", "size": 1, "size_mb": 0},
            {"key": "bulk-data/opinions-2025-03-31.csv.bz2", "size": 1, "size_mb": 0},
        ]
        result = find_latest_file(files, "opinions-")
        assert result["date"] == "2025-03-31"
