# tests/test_filter_chain_coverage.py
import csv
import logging

import pytest

pytestmark = pytest.mark.unit

from src.config import PipelineConfig
from src.filter_chain import (
    _smoke_test_csv,
    build_federal_appellate_filter,
    load_federal_clusters,
    load_federal_courts,
    load_federal_dockets,
)


def _write_csv(tmp_path, name, headers, rows):
    path = tmp_path / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


@pytest.fixture
def logger():
    lg = logging.getLogger("test_fc")
    lg.setLevel(logging.DEBUG)
    msgs: list = []
    h = logging.Handler()
    h.emit = lambda r: msgs.append(r.getMessage())  # type: ignore
    lg.addHandler(h)
    lg.msgs = msgs  # type: ignore
    return lg


class TestSmokeTestCsv:
    def test_pass(self, tmp_path, logger):
        path = _write_csv(tmp_path, "t.csv", ["id", "name"], [{"id": "1", "name": "a"}])
        assert _smoke_test_csv(path, ["id", "name"], logger=logger) is True
        assert any("Smoke test OK" in m for m in logger.msgs)  # type: ignore

    def test_fail(self, tmp_path, logger):
        path = _write_csv(tmp_path, "t.csv", ["id"], [{"id": "1"}])
        assert _smoke_test_csv(path, ["id", "missing_col"], logger=logger) is False
        assert any("Smoke test FAIL" in m for m in logger.msgs)  # type: ignore


class TestLoadFederalCourtsWithLogger:
    def test_logs_courts(self, tmp_path, logger):
        path = _write_csv(tmp_path, "courts.csv", ["id", "full_name"], [{"id": "ca1", "full_name": "First Circuit"}])
        config = PipelineConfig(federal_appellate_court_ids=frozenset({"ca1"}))
        ids, names = load_federal_courts(path, config=config, logger=logger)
        assert "ca1" in ids
        assert any("Federal appellate courts" in m for m in logger.msgs)  # type: ignore


class TestLoadFederalDocketsWithLogger:
    def test_logs_docket_counts(self, tmp_path, logger):
        path = _write_csv(
            tmp_path,
            "dockets.csv",
            ["id", "court_id", "case_name", "date_filed"],
            [{"id": "1", "court_id": "ca1", "case_name": "T", "date_filed": "2024"}],
        )
        config = PipelineConfig()
        result = load_federal_dockets(path, {"ca1"}, config=config, logger=logger)
        assert len(result) == 1
        assert any("Federal appellate dockets" in m for m in logger.msgs)  # type: ignore


class TestLoadFederalClustersWithLogger:
    def test_logs_cluster_counts(self, tmp_path, logger):
        path = _write_csv(
            tmp_path,
            "clusters.csv",
            ["id", "docket_id", "case_name", "date_filed", "precedential_status"],
            [
                {
                    "id": "100",
                    "docket_id": "1",
                    "case_name": "T",
                    "date_filed": "2024",
                    "precedential_status": "Published",
                }
            ],
        )
        config = PipelineConfig()
        result = load_federal_clusters(path, {1}, config=config, logger=logger)
        assert len(result) == 1
        assert any("Federal appellate clusters" in m for m in logger.msgs)  # type: ignore


class TestBuildFilterWithLogger:
    def test_full_chain_logged(self, tmp_path, logger):
        courts = _write_csv(tmp_path, "courts.csv", ["id", "full_name"], [{"id": "ca1", "full_name": "First"}])
        dockets = _write_csv(
            tmp_path,
            "dockets.csv",
            ["id", "court_id", "case_name", "date_filed"],
            [{"id": "1", "court_id": "ca1", "case_name": "T", "date_filed": "2024"}],
        )
        clusters = _write_csv(
            tmp_path,
            "clusters.csv",
            ["id", "docket_id", "case_name", "date_filed", "precedential_status"],
            [{"id": "100", "docket_id": "1", "case_name": "T", "date_filed": "2024", "precedential_status": "P"}],
        )
        config = PipelineConfig(federal_appellate_court_ids=frozenset({"ca1"}))
        paths = {"courts": courts, "dockets": dockets, "clusters": clusters}
        result = build_federal_appellate_filter(paths, config=config, logger=logger)
        assert len(result.fed_court_ids) == 1
        assert any("Loading courts" in m for m in logger.msgs)  # type: ignore
