# tests/test_aggregate_judgments.py
"""Test judgment summary aggregation script."""

import json

import pytest


@pytest.fixture
def agg_module():
    return __import__("scripts.aggregate_judgments", fromlist=["*"])


@pytest.mark.contract
class TestContract:
    def test_main_callable(self, agg_module):
        assert callable(getattr(agg_module, "main", None))

    def test_aggregate_one_callable(self, agg_module):
        assert callable(getattr(agg_module, "aggregate_one", None))


@pytest.mark.unit
class TestAggregateOne:
    def test_writes_summary_with_rates(self, agg_module, tmp_path):
        jp = tmp_path / "judgments.jsonl"
        jp.write_text(
            "\n".join(
                json.dumps({"source_id": i, "dest_id": i + 100, "label": "FAITHFUL" if i % 2 else "HALLUCINATED"})
                for i in range(10)
            )
            + "\n",
            encoding="utf-8",
        )
        sp = tmp_path / "summary.json"
        agg_module.aggregate_one(
            judgments_path=jp,
            summary_path=sp,
            ablation="bm25",
            ablation_label="bm25_rag",
            judge_model="gpt-4o-mini",
            limit=10,
        )
        assert sp.is_file()
        d = json.loads(sp.read_text())
        assert d["n_total"] == 10
        assert d["n_judged"] == 10
        assert d["faithful_rate"] == pytest.approx(0.5)
        assert d["hallucinated_rate"] == pytest.approx(0.5)
