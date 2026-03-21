# tests/conftest.py
# Shared fixtures for all test layers.

from pathlib import Path

import pytest

from src.config import PipelineConfig

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_dir():
    return FIXTURE_DIR


@pytest.fixture
def fixture_paths():
    return {
        "courts": FIXTURE_DIR / "courts.csv",
        "dockets": FIXTURE_DIR / "dockets.csv",
        "clusters": FIXTURE_DIR / "opinion-clusters.csv",
        "opinions": FIXTURE_DIR / "opinions.csv",
    }


@pytest.fixture
def fixture_config(tmp_path):
    return PipelineConfig(
        bulk_dir=FIXTURE_DIR,
        shard_dir=tmp_path / "shards",
        shard_size=3,
        min_text_length=50,
        min_expected_total=1,
    )
