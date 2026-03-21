import pytest

pytestmark = pytest.mark.integration

from src.extract import build_record
from src.schemas import ClusterMeta, DocketMeta, OpinionRecord


class TestBuildRecordReturnsTyped:
    def test_returns_opinion_record(self):
        cluster_meta = {
            100: ClusterMeta(docket_id=200, case_name="A v B", date_filed="2024-01-01", precedential_status="Published")
        }
        docket_meta = {200: DocketMeta(court_id="ca9", case_name="A v B", date_filed="2024-01-01")}
        result = build_record(
            opinion_id=42,
            cluster_id=100,
            raw_text="raw",
            normalized_text="clean",
            text_source="plain_text",
            cleaning_flags=[],
            opinion_type="lead",
            extracted_by_ocr="False",
            cluster_meta=cluster_meta,
            docket_meta=docket_meta,
            court_name_map={"ca9": "Ninth"},
        )
        assert isinstance(result, OpinionRecord)
        assert result.id == 42
        assert result.court_id == "ca9"

    def test_to_dict_round_trips(self):
        cluster_meta = {100: ClusterMeta(docket_id=200, case_name="T", date_filed="2024", precedential_status="P")}
        docket_meta = {200: DocketMeta(court_id="ca1", case_name="T", date_filed="2024")}
        record = build_record(
            opinion_id=1,
            cluster_id=100,
            raw_text="r",
            normalized_text="n",
            text_source="plain_text",
            cleaning_flags=[],
            opinion_type="lead",
            extracted_by_ocr="False",
            cluster_meta=cluster_meta,
            docket_meta=docket_meta,
            court_name_map={"ca1": "First"},
        )
        d = record.to_dict()
        assert d["id"] == 1
        assert isinstance(d, dict)
