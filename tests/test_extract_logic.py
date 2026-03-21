import pytest

pytestmark = pytest.mark.integration

# tests/test_extract_logic.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_extract_logic.py
# TDD RED: Unit tests for extraction logic — text source fallback, provenance resolution.

import csv
import json

import pytest

from src.config import PipelineConfig
from src.extract import extract_opinions_to_shards


class TestTextSourceFallback:
    """Text source selection must follow priority order and skip empty fields."""

    def _make_csv(self, tmp_path, rows):
        """Helper: write rows to a CSV file."""
        csv_path = tmp_path / "opinions.csv"
        if not rows:
            csv_path.write_text("")
            return csv_path
        fieldnames = rows[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def test_prefers_plain_text(self, tmp_path):
        csv_path = self._make_csv(
            tmp_path,
            [
                {
                    "id": "1",
                    "cluster_id": "100",
                    "type": "lead",
                    "plain_text": "This is the plain text opinion content here.",
                    "html_with_citations": "<p>HTML version</p>",
                    "html": "",
                    "html_lawbox": "",
                    "html_columbia": "",
                    "extracted_by_ocr": "False",
                }
            ],
        )
        config = PipelineConfig(shard_dir=tmp_path / "shards", shard_size=100, min_text_length=10)
        cluster_meta = {
            100: {"docket_id": 1, "case_name": "Test", "date_filed": "2024-01-01", "precedential_status": "Published"}
        }
        docket_meta = {1: {"court_id": "ca1", "case_name": "Test", "date_filed": "2024-01-01"}}
        court_map = {"ca1": "First Circuit"}

        stats = extract_opinions_to_shards(csv_path, cluster_meta, docket_meta, court_map, config=config)
        assert stats["text_source_counts"]["plain_text"] == 1
        assert stats["text_source_counts"]["html_with_citations"] == 0

    def test_falls_back_to_html_when_plain_empty(self, tmp_path):
        csv_path = self._make_csv(
            tmp_path,
            [
                {
                    "id": "1",
                    "cluster_id": "100",
                    "type": "lead",
                    "plain_text": "",
                    "html_with_citations": "Substantial HTML opinion content goes here.",
                    "html": "",
                    "html_lawbox": "",
                    "html_columbia": "",
                    "extracted_by_ocr": "False",
                }
            ],
        )
        config = PipelineConfig(shard_dir=tmp_path / "shards", shard_size=100, min_text_length=10)
        cluster_meta = {100: {"docket_id": 1, "case_name": "T", "date_filed": "2024", "precedential_status": "P"}}
        docket_meta = {1: {"court_id": "ca1", "case_name": "T", "date_filed": "2024"}}

        stats = extract_opinions_to_shards(csv_path, cluster_meta, docket_meta, {"ca1": "First"}, config=config)
        assert stats["text_source_counts"]["html_with_citations"] == 1

    def test_skips_when_all_text_empty(self, tmp_path):
        csv_path = self._make_csv(
            tmp_path,
            [
                {
                    "id": "1",
                    "cluster_id": "100",
                    "type": "lead",
                    "plain_text": "",
                    "html_with_citations": "",
                    "html": "",
                    "html_lawbox": "",
                    "html_columbia": "",
                    "extracted_by_ocr": "False",
                }
            ],
        )
        config = PipelineConfig(shard_dir=tmp_path / "shards", shard_size=100, min_text_length=10)
        cluster_meta = {100: {"docket_id": 1, "case_name": "T", "date_filed": "2024", "precedential_status": "P"}}
        docket_meta = {1: {"court_id": "ca1", "case_name": "T", "date_filed": "2024"}}

        stats = extract_opinions_to_shards(csv_path, cluster_meta, docket_meta, {"ca1": "First"}, config=config)
        assert stats["extracted_total"] == 0
        assert stats["skipped_empty"] == 1


class TestProvenanceResolution:
    """Each record must trace opinion → cluster → docket → court."""

    def _make_csv(self, tmp_path, rows):
        csv_path = tmp_path / "opinions.csv"
        fieldnames = rows[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def test_resolves_full_chain(self, tmp_path):
        csv_path = self._make_csv(
            tmp_path,
            [
                {
                    "id": "42",
                    "cluster_id": "100",
                    "type": "lead",
                    "plain_text": "Full opinion text content for testing provenance chain.",
                    "html_with_citations": "",
                    "html": "",
                    "html_lawbox": "",
                    "html_columbia": "",
                    "extracted_by_ocr": "False",
                }
            ],
        )
        config = PipelineConfig(shard_dir=tmp_path / "shards", shard_size=100, min_text_length=10)
        cluster_meta = {
            100: {
                "docket_id": 200,
                "case_name": "Smith v. Jones",
                "date_filed": "2024-03-15",
                "precedential_status": "Published",
            }
        }
        docket_meta = {200: {"court_id": "ca9", "case_name": "Smith v. Jones", "date_filed": "2024-03-15"}}
        court_map = {"ca9": "Ninth Circuit"}

        extract_opinions_to_shards(csv_path, cluster_meta, docket_meta, court_map, config=config)

        shard = tmp_path / "shards" / "shard_0000.jsonl"
        record = json.loads(shard.read_text().strip())
        assert record["court_id"] == "ca9"
        assert record["court_name"] == "Ninth Circuit"
        assert record["docket_id"] == 200
        assert record["case_name"] == "Smith v. Jones"
        assert record["date_filed"] == "2024-03-15"

    def test_handles_missing_docket_gracefully(self, tmp_path):
        csv_path = self._make_csv(
            tmp_path,
            [
                {
                    "id": "42",
                    "cluster_id": "100",
                    "type": "lead",
                    "plain_text": "Opinion text content here for missing docket test.",
                    "html_with_citations": "",
                    "html": "",
                    "html_lawbox": "",
                    "html_columbia": "",
                    "extracted_by_ocr": "False",
                }
            ],
        )
        config = PipelineConfig(shard_dir=tmp_path / "shards", shard_size=100, min_text_length=10)
        cluster_meta = {100: {"docket_id": 999, "case_name": "Test", "date_filed": "2024", "precedential_status": "P"}}
        docket_meta = {}  # docket 999 missing

        extract_opinions_to_shards(csv_path, cluster_meta, docket_meta, {}, config=config)

        shard = tmp_path / "shards" / "shard_0000.jsonl"
        record = json.loads(shard.read_text().strip())
        assert record["court_id"] == ""
        assert record["court_name"] == ""
