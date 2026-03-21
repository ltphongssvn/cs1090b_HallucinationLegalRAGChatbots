import pytest

pytestmark = pytest.mark.unit

# tests/test_validation_sampling.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_validation_sampling.py
# TDD RED: Unit tests for random sampling in validation.

import json

import pytest

from src.validation import _random_sample_cases


class TestRandomSampleCases:
    def _make_shard(self, tmp_path, num_records):
        shard_path = tmp_path / "shard_0000.jsonl"
        with open(shard_path, "w") as f:
            for i in range(num_records):
                f.write(json.dumps({"id": i, "text": f"record_{i}"}) + "\n")
        return shard_path

    def test_samples_correct_count(self, tmp_path):
        shard_path = self._make_shard(tmp_path, 1000)
        cases = _random_sample_cases(shard_path, sample_size=50)
        assert len(cases) == 50

    def test_samples_from_different_positions(self, tmp_path):
        shard_path = self._make_shard(tmp_path, 1000)
        cases = _random_sample_cases(shard_path, sample_size=50)
        ids = [c["id"] for c in cases]
        # Should NOT be just 0-49 (head sampling)
        assert max(ids) > 49, "Sample appears to be head-only"

    def test_deterministic_with_seed(self, tmp_path):
        shard_path = self._make_shard(tmp_path, 1000)
        run1 = _random_sample_cases(shard_path, sample_size=50, seed=42)
        run2 = _random_sample_cases(shard_path, sample_size=50, seed=42)
        assert [c["id"] for c in run1] == [c["id"] for c in run2]

    def test_different_seed_different_sample(self, tmp_path):
        shard_path = self._make_shard(tmp_path, 1000)
        run1 = _random_sample_cases(shard_path, sample_size=50, seed=42)
        run2 = _random_sample_cases(shard_path, sample_size=50, seed=99)
        assert [c["id"] for c in run1] != [c["id"] for c in run2]

    def test_small_file_returns_all(self, tmp_path):
        shard_path = self._make_shard(tmp_path, 10)
        cases = _random_sample_cases(shard_path, sample_size=50)
        assert len(cases) == 10
