# tests/test_data_contracts.py
# TDD RED: Statistical data contracts over extraction stats.
import pytest

pytestmark = pytest.mark.unit

from src.data_contracts import (
    DataContractError,
    check_court_balance,
    check_row_count_floor,
    check_text_length_distribution,
    run_all_contracts,
)


@pytest.fixture
def healthy_manifest():
    return {
        "num_cases": 1_465_484,
        "court_distribution": {
            "ca1": 80_000,
            "ca2": 120_000,
            "ca3": 100_000,
            "ca4": 110_000,
            "ca5": 200_000,
            "ca6": 130_000,
            "ca7": 90_000,
            "ca8": 85_000,
            "ca9": 300_000,
            "ca10": 95_000,
            "ca11": 105_000,
            "cadc": 30_000,
            "cafc": 20_484,
        },
        "text_length_stats": {"mean": 5000, "median": 3500, "p5": 600, "p95": 18000, "max": 250000},
    }


class TestCheckRowCountFloor:
    def test_pass_when_above_floor(self, healthy_manifest):
        r = check_row_count_floor(healthy_manifest, min_rows=10_000)
        assert r.passed is True

    def test_fail_when_below_floor(self, healthy_manifest):
        r = check_row_count_floor(healthy_manifest, min_rows=2_000_000)
        assert r.passed is False
        assert "below floor" in r.message


class TestCheckCourtBalance:
    def test_pass_on_balanced_distribution(self, healthy_manifest):
        r = check_court_balance(healthy_manifest, max_share=0.5)
        assert r.passed is True

    def test_fail_when_one_court_dominates(self):
        manifest = {"court_distribution": {"ca9": 900, "ca1": 100}}
        r = check_court_balance(manifest, max_share=0.5)
        assert r.passed is False
        assert "ca9" in r.message


class TestCheckTextLengthDistribution:
    def test_pass_on_healthy_distribution(self, healthy_manifest):
        r = check_text_length_distribution(healthy_manifest, min_mean=1000, min_p5=100)
        assert r.passed is True

    def test_fail_on_truncated_corpus(self):
        m = {"text_length_stats": {"mean": 200, "p5": 50}}
        r = check_text_length_distribution(m, min_mean=1000, min_p5=100)
        assert r.passed is False


class TestRunAllContracts:
    def test_returns_all_results_when_healthy(self, healthy_manifest):
        results = run_all_contracts(healthy_manifest)
        assert len(results) >= 3
        assert all(r.passed for r in results)

    def test_raises_on_any_failure_when_strict(self):
        m = {"num_cases": 10, "court_distribution": {"ca9": 10}, "text_length_stats": {"mean": 10, "p5": 1}}
        with pytest.raises(DataContractError):
            run_all_contracts(m, strict=True)

    def test_returns_failing_results_when_not_strict(self):
        m = {"num_cases": 10, "court_distribution": {"ca9": 10}, "text_length_stats": {"mean": 10, "p5": 1}}
        results = run_all_contracts(m, strict=False)
        assert any(not r.passed for r in results)
