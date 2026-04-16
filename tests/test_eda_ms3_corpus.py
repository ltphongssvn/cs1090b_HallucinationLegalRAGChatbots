"""TDD RED: contract + unit + property tests for scripts/eda_ms3_corpus.py.

Verified 2026 conventions matched to this repo:
    - Standard imports via pyproject pythonpath=["."] (no sys.path hacks).
    - tmp_path isolation for all filesystem side effects.
    - Synthetic mini-shard for deterministic math verification.
    - Pydantic SummarySchema for artifact contract (pydantic 2.12.5 pinned).
    - Hypothesis property test for filter predicate (in lockfile).
    - SHA256 provenance: input shards + output PNGs in summary.json.
    - W&B telemetry isolated to main() (matches src/dataset_probe.py pattern).
    - Module-level constants retained (repo convention; no Hydra in stack).
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, Field


class SummarySchema(BaseModel):
    """Strict contract for logs/eda_ms3/summary.json."""

    n_total: int = Field(ge=1)
    text_length_mean: float = Field(ge=0)
    text_length_median: float = Field(ge=0)
    n_short_lt_100: int = Field(ge=0)
    filter_threshold: int
    circuit_counts: dict[str, int]
    corpus_manifest_sha: str = Field(min_length=64, max_length=64)
    figure_hashes: dict[str, str]


MINI_SHARD = Path("tests/fixtures/mini_shard.jsonl")


@pytest.fixture(scope="module")
def eda_module() -> Any:
    """Import scripts.eda_ms3_corpus as a package module."""
    return importlib.import_module("scripts.eda_ms3_corpus")


@pytest.fixture
def fake_manifest(tmp_path: Path) -> Path:
    """Synthetic manifest.json with a shard checksum for SHA provenance."""
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"checksum": {"mini_shard.jsonl": "a" * 64}}))
    return m


# ---------------------------------------------------------------------------
# Contract tier
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_script_file_exists() -> None:
    assert Path("scripts/eda_ms3_corpus.py").exists()


@pytest.mark.contract
def test_module_exposes_main(eda_module) -> None:
    assert callable(getattr(eda_module, "main", None))


@pytest.mark.contract
def test_main_accepts_injected_paths(eda_module) -> None:
    """main(shard_glob, out_dir, manifest_path) enables DI."""
    sig = inspect.signature(eda_module.main)
    for param in ("shard_glob", "out_dir", "manifest_path"):
        assert param in sig.parameters


@pytest.mark.contract
def test_module_declares_filter_threshold(eda_module) -> None:
    """Module-level constant pattern — matches src/dataset_probe.py."""
    assert eda_module.FILTER_MIN_CHARS == 100


@pytest.mark.contract
def test_module_has_filter_predicate(eda_module) -> None:
    assert callable(getattr(eda_module, "is_valid_record", None))


@pytest.mark.contract
def test_module_has_wandb_logger(eda_module) -> None:
    """W&B telemetry must be a separate _log_to_wandb function.

    Matches src/dataset_probe.py::_log_report_to_wandb isolation contract:
    W&B logic lives outside main computation; main() calls it optionally.
    """
    assert callable(getattr(eda_module, "_log_to_wandb", None))


@pytest.mark.contract
def test_main_signature_has_log_to_wandb_flag(eda_module) -> None:
    """main() accepts log_to_wandb bool (default False) for CI/test isolation."""
    sig = inspect.signature(eda_module.main)
    assert "log_to_wandb" in sig.parameters
    assert sig.parameters["log_to_wandb"].default is False


# ---------------------------------------------------------------------------
# Unit tier
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mini_shard_fixture_exists() -> None:
    assert MINI_SHARD.exists()
    assert len(MINI_SHARD.read_text().strip().splitlines()) == 5


@pytest.mark.unit
def test_filter_predicate_boundary(eda_module) -> None:
    assert eda_module.is_valid_record(100) is True
    assert eda_module.is_valid_record(99) is False
    assert eda_module.is_valid_record(0) is False


@pytest.mark.unit
@given(length=st.integers(min_value=-1000, max_value=10_000_000))
def test_filter_predicate_property(length: int) -> None:
    mod = importlib.import_module("scripts.eda_ms3_corpus")
    result = mod.is_valid_record(length)
    assert isinstance(result, bool)
    assert result is (length >= mod.FILTER_MIN_CHARS)


@pytest.mark.unit
def test_main_deterministic_on_mini_shard(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    """Mini-shard composition: ids 1,2 ca9; 3,4 ca5; 5 ca2.
    Lengths: 150,110,5,108,4 → mean=75.4, median=108, short=2.
    """
    result = eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    assert result["n_total"] == 5
    assert abs(result["text_length_mean"] - 75.4) < 1e-6
    assert result["text_length_median"] == 108
    assert result["n_short_lt_100"] == 2
    assert result["circuit_counts"] == {"ca9": 2, "ca5": 2, "ca2": 1}


@pytest.mark.unit
def test_summary_validates_against_pydantic_schema(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    parsed = SummarySchema.model_validate_json((tmp_path / "summary.json").read_text())
    assert parsed.filter_threshold == 100
    assert parsed.n_total == 5
    assert len(parsed.corpus_manifest_sha) == 64
    assert "text_length_hist.png" in parsed.figure_hashes


@pytest.mark.unit
def test_figures_written_to_tmp_path(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    for fname in ("text_length_hist.png", "circuit_distribution.png"):
        assert (tmp_path / fname).exists()


@pytest.mark.unit
def test_png_hashes_match_recorded(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    import hashlib

    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    for fname, recorded in summary["figure_hashes"].items():
        actual = hashlib.sha256((tmp_path / fname).read_bytes()).hexdigest()
        assert actual == recorded


@pytest.mark.unit
def test_no_side_effects_on_real_logs_dir(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    real_dir = Path("logs/eda_ms3")
    before = set(real_dir.glob("*")) if real_dir.exists() else set()
    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    after = set(real_dir.glob("*")) if real_dir.exists() else set()
    assert before == after


@pytest.mark.unit
def test_wandb_not_called_when_flag_false(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    """log_to_wandb=False must skip W&B entirely (CI isolation)."""
    with patch.object(eda_module, "_log_to_wandb") as mock_log:
        eda_module.main(
            shard_glob=str(MINI_SHARD),
            out_dir=tmp_path,
            manifest_path=fake_manifest,
            log_to_wandb=False,
        )
        mock_log.assert_not_called()


@pytest.mark.unit
def test_wandb_called_exactly_once_when_flag_true(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    """When log_to_wandb=True, _log_to_wandb fires exactly once.

    Matches src/dataset_probe.py TestLogReportToWandbSingleCall contract.
    """
    with patch.object(eda_module, "_log_to_wandb") as mock_log:
        eda_module.main(
            shard_glob=str(MINI_SHARD),
            out_dir=tmp_path,
            manifest_path=fake_manifest,
            log_to_wandb=True,
        )
        assert mock_log.call_count == 1
