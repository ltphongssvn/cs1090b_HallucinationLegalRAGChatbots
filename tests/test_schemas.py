import pytest

pytestmark = pytest.mark.unit

from src.schemas import ClusterMeta, DocketMeta, ManifestData, OpinionRecord


class TestOpinionRecord:
    def test_from_dict(self):
        data = {
            "id": 1,
            "cluster_id": 100,
            "docket_id": 200,
            "court_id": "ca1",
            "court_name": "First",
            "case_name": "A v B",
            "date_filed": "2024-01-01",
            "precedential_status": "Published",
            "opinion_type": "lead",
            "extracted_by_ocr": "False",
            "raw_text": "<p>Hi</p>",
            "text": "Hi",
            "text_length": 2,
            "text_source": "html",
            "cleaning_flags": ["html_stripped"],
            "source": "courtlistener_bulk",
        }
        record = OpinionRecord(**data)
        assert record.id == 1
        assert record.court_id == "ca1"
        assert record.cleaning_flags == ["html_stripped"]

    def test_to_dict(self):
        record = OpinionRecord(
            id=1,
            cluster_id=100,
            docket_id=200,
            court_id="ca1",
            court_name="First",
            case_name="A v B",
            date_filed="2024",
            precedential_status="P",
            opinion_type="lead",
            extracted_by_ocr="False",
            raw_text="t",
            text="t",
            text_length=1,
            text_source="plain_text",
            cleaning_flags=[],
            source="courtlistener_bulk",
        )
        d = record.to_dict()
        assert d["id"] == 1
        assert isinstance(d, dict)

    def test_rejects_missing_field(self):
        with pytest.raises(TypeError):
            OpinionRecord(id=1)


class TestDocketMeta:
    def test_fields(self):
        d = DocketMeta(court_id="ca9", case_name="X v Y", date_filed="2024-01-01")
        assert d.court_id == "ca9"


class TestClusterMeta:
    def test_fields(self):
        c = ClusterMeta(docket_id=100, case_name="X v Y", date_filed="2024", precedential_status="Published")
        assert c.docket_id == 100


class TestManifestData:
    def test_version_default(self):
        m = ManifestData(num_cases=100, num_shards=1, shard_size=100)
        assert m.version == 2
