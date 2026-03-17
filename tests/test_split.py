import pytest

pytestmark = pytest.mark.integration

# tests/test_split.py
# TDD RED: Data split strategies to prevent leakage.


import pytest

from src.split import (
    split_by_cluster,
    split_by_docket,
    split_by_time,
    validate_no_leakage,
)


def _make_records():
    """10 records: 3 dockets, 3 clusters, dates 2018-2024."""
    return [
        {"id": 1, "docket_id": 100, "cluster_id": 500, "court_id": "ca1", "date_filed": "2018-03-01", "text": "a"},
        {"id": 2, "docket_id": 100, "cluster_id": 500, "court_id": "ca1", "date_filed": "2018-03-01", "text": "b"},
        {"id": 3, "docket_id": 100, "cluster_id": 501, "court_id": "ca1", "date_filed": "2019-06-15", "text": "c"},
        {"id": 4, "docket_id": 200, "cluster_id": 502, "court_id": "ca9", "date_filed": "2020-01-10", "text": "d"},
        {"id": 5, "docket_id": 200, "cluster_id": 502, "court_id": "ca9", "date_filed": "2020-01-10", "text": "e"},
        {"id": 6, "docket_id": 200, "cluster_id": 503, "court_id": "ca9", "date_filed": "2021-08-20", "text": "f"},
        {"id": 7, "docket_id": 300, "cluster_id": 504, "court_id": "cadc", "date_filed": "2022-04-01", "text": "g"},
        {"id": 8, "docket_id": 300, "cluster_id": 504, "court_id": "cadc", "date_filed": "2022-04-01", "text": "h"},
        {"id": 9, "docket_id": 300, "cluster_id": 505, "court_id": "cadc", "date_filed": "2023-12-01", "text": "i"},
        {"id": 10, "docket_id": 300, "cluster_id": 505, "court_id": "cadc", "date_filed": "2024-02-15", "text": "j"},
    ]


class TestSplitByDocket:
    def test_no_docket_leakage(self):
        records = _make_records()
        splits = split_by_docket(records, train_ratio=0.6, val_ratio=0.2, seed=42)
        train_dockets = {r["docket_id"] for r in splits["train"]}
        val_dockets = {r["docket_id"] for r in splits["val"]}
        test_dockets = {r["docket_id"] for r in splits["test"]}
        assert not (train_dockets & val_dockets), "Train/val docket leakage"
        assert not (train_dockets & test_dockets), "Train/test docket leakage"
        assert not (val_dockets & test_dockets), "Val/test docket leakage"

    def test_all_records_assigned(self):
        records = _make_records()
        splits = split_by_docket(records, train_ratio=0.6, val_ratio=0.2, seed=42)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(records)

    def test_deterministic(self):
        records = _make_records()
        s1 = split_by_docket(records, seed=42)
        s2 = split_by_docket(records, seed=42)
        assert [r["id"] for r in s1["train"]] == [r["id"] for r in s2["train"]]


class TestSplitByTime:
    def test_temporal_ordering(self):
        records = _make_records()
        splits = split_by_time(records, train_cutoff="2021-01-01", val_cutoff="2023-01-01")
        for r in splits["train"]:
            assert r["date_filed"] < "2021-01-01"
        for r in splits["val"]:
            assert "2021-01-01" <= r["date_filed"] < "2023-01-01"
        for r in splits["test"]:
            assert r["date_filed"] >= "2023-01-01"

    def test_all_records_assigned(self):
        records = _make_records()
        splits = split_by_time(records, train_cutoff="2021-01-01", val_cutoff="2023-01-01")
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(records)


class TestSplitByCluster:
    def test_no_cluster_leakage(self):
        records = _make_records()
        splits = split_by_cluster(records, train_ratio=0.6, val_ratio=0.2, seed=42)
        train_clusters = {r["cluster_id"] for r in splits["train"]}
        val_clusters = {r["cluster_id"] for r in splits["val"]}
        test_clusters = {r["cluster_id"] for r in splits["test"]}
        assert not (train_clusters & val_clusters)
        assert not (train_clusters & test_clusters)
        assert not (val_clusters & test_clusters)


class TestValidateNoLeakage:
    def test_clean_split_passes(self):
        records = _make_records()
        splits = split_by_docket(records, seed=42)
        report = validate_no_leakage(splits, group_key="docket_id")
        assert report["leaked_groups"] == 0

    def test_injected_leakage_detected(self):
        records = _make_records()
        splits = split_by_docket(records, seed=42)
        # Inject leakage: copy a train record into test
        if splits["train"]:
            splits["test"].append(splits["train"][0])
        report = validate_no_leakage(splits, group_key="docket_id")
        assert report["leaked_groups"] > 0
