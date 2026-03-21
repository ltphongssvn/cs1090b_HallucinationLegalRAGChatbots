# tests/test_split.py
# Project: HallucinationLegalRAGChatbots
# TDD unit tests for src/split.py — leakage-free data splits.

import pytest

from src.split import (
    SplitConfig,
    split_by_cluster,
    split_by_docket,
    split_by_time,
    validate_no_leakage,
)

pytestmark = pytest.mark.unit


def _make_records(n: int, docket_ids: list, cluster_ids: list, dates: list) -> list:
    return [
        {
            "id": i,
            "docket_id": docket_ids[i % len(docket_ids)],
            "cluster_id": cluster_ids[i % len(cluster_ids)],
            "date_filed": dates[i % len(dates)],
        }
        for i in range(n)
    ]


class TestSplitConfig:
    def test_defaults(self) -> None:
        cfg = SplitConfig()
        assert cfg.strategy == "docket"
        assert cfg.train_ratio == 0.7
        assert cfg.val_ratio == 0.15
        assert cfg.seed == 42

    def test_custom_values(self) -> None:
        cfg = SplitConfig(strategy="cluster", train_ratio=0.8, val_ratio=0.1, seed=0)
        assert cfg.strategy == "cluster"
        assert cfg.seed == 0


class TestSplitByDocket:
    def test_returns_three_splits(self) -> None:
        records = _make_records(30, list(range(10)), list(range(10)), ["2020-01-01"])
        splits = split_by_docket(records)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_all_records_accounted_for(self) -> None:
        records = _make_records(30, list(range(10)), list(range(10)), ["2020-01-01"])
        splits = split_by_docket(records)
        assert sum(len(v) for v in splits.values()) == 30

    def test_no_leakage_by_docket(self) -> None:
        records = _make_records(30, list(range(10)), list(range(10)), ["2020-01-01"])
        splits = split_by_docket(records)
        report = validate_no_leakage(splits, group_key="docket_id")
        assert report["leaked_groups"] == 0

    def test_reproducible_with_same_seed(self) -> None:
        records = _make_records(30, list(range(10)), list(range(10)), ["2020-01-01"])
        s1 = split_by_docket(records, seed=42)
        s2 = split_by_docket(records, seed=42)
        assert [r["id"] for r in s1["train"]] == [r["id"] for r in s2["train"]]

    def test_different_seeds_produce_different_splits(self) -> None:
        records = _make_records(100, list(range(20)), list(range(20)), ["2020-01-01"])
        s1 = split_by_docket(records, seed=1)
        s2 = split_by_docket(records, seed=2)
        assert [r["id"] for r in s1["train"]] != [r["id"] for r in s2["train"]]

    def test_empty_records(self) -> None:
        splits = split_by_docket([])
        assert splits["train"] == []
        assert splits["val"] == []
        assert splits["test"] == []

    def test_single_record(self) -> None:
        records = [{"id": 0, "docket_id": "d1", "cluster_id": "c1", "date_filed": "2020-01-01"}]
        splits = split_by_docket(records)
        assert sum(len(v) for v in splits.values()) == 1


class TestSplitByCluster:
    def test_no_leakage_by_cluster(self) -> None:
        records = _make_records(30, list(range(10)), list(range(10)), ["2020-01-01"])
        splits = split_by_cluster(records)
        report = validate_no_leakage(splits, group_key="cluster_id")
        assert report["leaked_groups"] == 0

    def test_all_records_accounted_for(self) -> None:
        records = _make_records(30, list(range(10)), list(range(10)), ["2020-01-01"])
        splits = split_by_cluster(records)
        assert sum(len(v) for v in splits.values()) == 30


class TestSplitByTime:
    def test_temporal_split_correct_buckets(self) -> None:
        records = [
            {"id": 0, "date_filed": "2018-01-01"},
            {"id": 1, "date_filed": "2020-01-01"},
            {"id": 2, "date_filed": "2022-01-01"},
        ]
        splits = split_by_time(records, train_cutoff="2019-01-01", val_cutoff="2021-01-01")
        assert splits["train"][0]["id"] == 0
        assert splits["val"][0]["id"] == 1
        assert splits["test"][0]["id"] == 2

    def test_all_records_accounted_for(self) -> None:
        records = [{"id": i, "date_filed": f"202{i}-01-01"} for i in range(5)]
        splits = split_by_time(records, train_cutoff="2022-01-01", val_cutoff="2024-01-01")
        assert sum(len(v) for v in splits.values()) == 5

    def test_empty_records(self) -> None:
        splits = split_by_time([], train_cutoff="2020-01-01", val_cutoff="2022-01-01")
        assert all(len(v) == 0 for v in splits.values())

    def test_missing_date_is_less_than_any_cutoff(self) -> None:
        # Empty string "" < any date string in Python — falls to train
        records = [{"id": 0, "date_filed": ""}]
        splits = split_by_time(records, train_cutoff="2020-01-01", val_cutoff="2022-01-01")
        assert splits["train"][0]["id"] == 0


class TestValidateNoLeakage:
    def test_clean_split_reports_zero_leakage(self) -> None:
        splits = {
            "train": [{"docket_id": "d1"}, {"docket_id": "d2"}],
            "val": [{"docket_id": "d3"}],
            "test": [{"docket_id": "d4"}],
        }
        report = validate_no_leakage(splits)
        assert report["leaked_groups"] == 0
        assert report["leaked_ids"] == []

    def test_detects_train_val_overlap(self) -> None:
        splits = {
            "train": [{"docket_id": "d1"}],
            "val": [{"docket_id": "d1"}],
            "test": [{"docket_id": "d2"}],
        }
        report = validate_no_leakage(splits)
        assert report["leaked_groups"] == 1
        assert "d1" in report["leaked_ids"]

    def test_report_contains_counts(self) -> None:
        splits = {
            "train": [{"docket_id": "d1"}, {"docket_id": "d1"}],
            "val": [{"docket_id": "d2"}],
            "test": [{"docket_id": "d3"}],
        }
        report = validate_no_leakage(splits)
        assert report["train_records"] == 2
        assert report["val_records"] == 1
        assert report["test_records"] == 1
        assert report["train_groups"] == 1
