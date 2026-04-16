"""TDD contract + unit + property tests for scripts/eda_ms3_corpus.py."""

from __future__ import annotations

import hashlib
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

REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_SHARD = REPO_ROOT / "tests" / "fixtures" / "mini_shard.jsonl"
SCRIPT_PATH = REPO_ROOT / "scripts" / "eda_ms3_corpus.py"
REAL_LOGS_DIR = REPO_ROOT / "logs" / "eda_ms3"


class SummarySchema(BaseModel):
    schema_version: str = Field(min_length=1)
    n_total: int = Field(ge=1)
    text_length_mean: float = Field(ge=0)
    text_length_median: float = Field(ge=0)
    n_short_lt_100: int = Field(ge=0)
    filter_threshold: int
    circuit_counts: dict[str, int]
    corpus_manifest_sha: str = Field(min_length=64, max_length=64)
    figure_hashes: dict[str, str]


@pytest.fixture(scope="module")
def eda_module() -> Any:
    return importlib.import_module("scripts.eda_ms3_corpus")


@pytest.fixture
def fake_manifest(tmp_path: Path) -> Path:
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"checksum": {"mini_shard.jsonl": "a" * 64}}), encoding="utf-8")
    return m


# ---------------------------------------------------------------------------
# Contract tier
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_script_file_exists() -> None:
    assert SCRIPT_PATH.exists()


@pytest.mark.contract
def test_module_exposes_main(eda_module) -> None:
    assert callable(getattr(eda_module, "main", None))


@pytest.mark.contract
def test_main_accepts_injected_paths(eda_module) -> None:
    sig = inspect.signature(eda_module.main)
    for param in ("shard_glob", "out_dir", "manifest_path"):
        assert param in sig.parameters


@pytest.mark.contract
def test_module_declares_filter_threshold(eda_module) -> None:
    assert eda_module.FILTER_MIN_CHARS == 100


@pytest.mark.contract
def test_module_has_filter_predicate(eda_module) -> None:
    assert callable(getattr(eda_module, "is_valid_record", None))


@pytest.mark.contract
def test_module_has_wandb_logger(eda_module) -> None:
    assert callable(getattr(eda_module, "_log_to_wandb", None))


@pytest.mark.contract
def test_main_signature_has_log_to_wandb_flag(eda_module) -> None:
    sig = inspect.signature(eda_module.main)
    assert "log_to_wandb" in sig.parameters
    assert sig.parameters["log_to_wandb"].default is False


@pytest.mark.contract
def test_module_declares_schema_version(eda_module) -> None:
    sv = getattr(eda_module, "SCHEMA_VERSION", None)
    assert isinstance(sv, str) and len(sv) > 0


# ---------------------------------------------------------------------------
# Unit tier
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mini_shard_fixture_exists() -> None:
    assert MINI_SHARD.exists()
    assert len(MINI_SHARD.read_text(encoding="utf-8").strip().splitlines()) == 8


@pytest.mark.unit
def test_filter_predicate_boundary(eda_module) -> None:
    assert eda_module.is_valid_record(99) is False
    assert eda_module.is_valid_record(100) is True
    assert eda_module.is_valid_record(101) is True
    assert eda_module.is_valid_record(0) is False


@pytest.mark.property
@given(length=st.integers(min_value=-1000, max_value=10_000_000))
def test_filter_predicate_property(length: int) -> None:
    mod = importlib.import_module("scripts.eda_ms3_corpus")
    result = mod.is_valid_record(length)
    assert isinstance(result, bool)
    assert result is (length >= mod.FILTER_MIN_CHARS)


@pytest.mark.unit
def test_main_deterministic_on_mini_shard(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    """8 records: lengths [150,110,5,108,4,99,100,101].

    mean=(150+110+5+108+4+99+100+101)/8=84.625; short (<100): ids 3,5,6 → 3;
    circuit_counts: ca9=2, ca5=2, ca2=3, ca1=1.
    """
    result = eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    assert result["n_total"] == 8
    assert abs(result["text_length_mean"] - 84.625) < 1e-6
    assert result["n_short_lt_100"] == 3
    assert result["circuit_counts"] == {"ca1": 1, "ca2": 3, "ca5": 2, "ca9": 2}


@pytest.mark.unit
def test_summary_validates_against_pydantic_schema(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    raw = (tmp_path / "summary.json").read_text(encoding="utf-8")
    parsed = SummarySchema.model_validate_json(raw)
    assert parsed.filter_threshold == 100
    assert parsed.n_total == 8
    assert len(parsed.corpus_manifest_sha) == 64
    assert "text_length_hist.png" in parsed.figure_hashes


@pytest.mark.unit
def test_summary_invariants(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    result = eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    assert 0 <= result["n_short_lt_100"] <= result["n_total"]
    assert sum(result["circuit_counts"].values()) == result["n_total"]
    assert result["text_length_mean"] >= 0
    assert result["text_length_median"] >= 0
    assert result["filter_threshold"] == eda_module.FILTER_MIN_CHARS


@pytest.mark.unit
def test_figures_written_to_tmp_path_and_non_empty(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    for fname in ("text_length_hist.png", "circuit_distribution.png"):
        fp = tmp_path / fname
        assert fp.exists()
        assert fp.stat().st_size > 1024
        assert fp.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
def test_png_hashes_match_recorded(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    for fname, recorded in summary["figure_hashes"].items():
        actual = hashlib.sha256((tmp_path / fname).read_bytes()).hexdigest()
        assert actual == recorded


@pytest.mark.unit
def test_no_side_effects_on_real_logs_dir(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    before = set(REAL_LOGS_DIR.glob("*")) if REAL_LOGS_DIR.exists() else set()
    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    after = set(REAL_LOGS_DIR.glob("*")) if REAL_LOGS_DIR.exists() else set()
    assert before == after


@pytest.mark.unit
def test_wandb_not_called_when_flag_false(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
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
    with patch.object(eda_module, "_log_to_wandb") as mock_log:
        eda_module.main(
            shard_glob=str(MINI_SHARD),
            out_dir=tmp_path,
            manifest_path=fake_manifest,
            log_to_wandb=True,
        )
        assert mock_log.call_count == 1


# ---------------------------------------------------------------------------
# 2026 hardening — schema + logging + single-scan + git SHA + Polars schema
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_module_has_polars_schema(eda_module) -> None:
    schema = getattr(eda_module, "POLARS_SCHEMA", None)
    assert isinstance(schema, dict)
    assert "text_length" in schema
    assert "court_id" in schema


@pytest.mark.contract
def test_module_uses_logging(eda_module) -> None:
    import logging as stdlib_logging

    logger = getattr(eda_module, "logger", None)
    assert isinstance(logger, stdlib_logging.Logger)


@pytest.mark.unit
def test_summary_includes_git_sha(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    s = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "git_sha" in s
    assert isinstance(s["git_sha"], str) and len(s["git_sha"]) >= 7


@pytest.mark.unit
def test_compute_stats_single_scan(eda_module) -> None:
    import polars as pl

    original = pl.scan_ndjson
    call_count = [0]

    def counting_scan(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)

    with patch.object(pl, "scan_ndjson", side_effect=counting_scan):
        eda_module._compute_stats(str(MINI_SHARD))
    assert call_count[0] == 1


@pytest.mark.unit
def test_polars_schema_applied_at_scan(eda_module) -> None:
    import polars as pl

    original = pl.scan_ndjson
    captured_kwargs: dict = {}

    def capturing_scan(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return original(*args, **kwargs)

    with patch.object(pl, "scan_ndjson", side_effect=capturing_scan):
        eda_module._compute_stats(str(MINI_SHARD))
    assert "schema_overrides" in captured_kwargs or "schema" in captured_kwargs


# ---------------------------------------------------------------------------
# 2026 hardening tier 2 — fail-fast, idempotent, filtered stats, CLI, order
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_no_import_time_side_effects() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    baseline_dpi = plt.rcParams["savefig.dpi"]
    plt.rcParams["savefig.dpi"] = 999
    np.random.seed(42)
    sentinel = np.random.rand()

    importlib.reload(importlib.import_module("scripts.eda_ms3_corpus"))

    assert plt.rcParams["savefig.dpi"] == 999
    np.random.seed(42)
    assert np.random.rand() == sentinel
    plt.rcParams["savefig.dpi"] = baseline_dpi


@pytest.mark.unit
def test_missing_manifest_fails_fast(eda_module, tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        eda_module.main(
            shard_glob=str(MINI_SHARD),
            out_dir=tmp_path,
            manifest_path=missing,
            log_to_wandb=False,
        )


@pytest.mark.unit
def test_stale_artifacts_removed(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    stale = tmp_path / "text_length_hist.png"
    stale.write_bytes(b"stale")
    stale_sha = hashlib.sha256(b"stale").hexdigest()
    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    assert stale.exists()
    assert hashlib.sha256(stale.read_bytes()).hexdigest() != stale_sha


@pytest.mark.unit
def test_summary_includes_filtered_stats(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    result = eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    assert "n_after_filter" in result
    assert result["n_after_filter"] == 5
    assert result["n_after_filter"] + result["n_short_lt_100"] == result["n_total"]
    assert "text_length_mean_filtered" in result


@pytest.mark.unit
def test_summary_includes_chart_ranges(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    result = eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    assert "chart_ranges" in result
    cr = result["chart_ranges"]
    assert "text_length_hist" in cr
    assert cr["text_length_hist"] == [0, 100_000]


@pytest.mark.unit
def test_summary_json_deterministic(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    raw = (tmp_path / "summary.json").read_text(encoding="utf-8")
    loaded = json.loads(raw)
    resorted = json.dumps(loaded, indent=2, sort_keys=True)
    assert raw.strip() == resorted.strip()


@pytest.mark.contract
def test_module_has_cli(eda_module) -> None:
    import argparse

    fn = getattr(eda_module, "_build_arg_parser", None)
    assert callable(fn)
    parser = fn()
    assert isinstance(parser, argparse.ArgumentParser)


@pytest.mark.unit
def test_circuit_order_deterministic(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    result = eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    keys = list(result["circuit_counts"].keys())
    canonical = sorted(keys, key=lambda c: (0, int(c[2:])) if c[2:].isdigit() else (1, c))
    assert keys == canonical


# ---------------------------------------------------------------------------
# 2026 hardening tier 3 — Pydantic runtime validation + shared filter expr
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_data_contracts_exports_filter_expression() -> None:
    from src import data_contracts

    assert callable(getattr(data_contracts, "valid_record_expr", None))
    assert hasattr(data_contracts, "FILTER_MIN_CHARS")
    assert data_contracts.FILTER_MIN_CHARS == 100


@pytest.mark.unit
def test_valid_record_expr_returns_polars_expr() -> None:
    import polars as pl

    from src.data_contracts import valid_record_expr

    expr = valid_record_expr()
    assert isinstance(expr, pl.Expr)
    df = pl.DataFrame({"text_length": [50, 100, 150]})
    kept = df.filter(expr).height
    assert kept == 2


@pytest.mark.contract
def test_summary_pydantic_model_exists(eda_module) -> None:
    from pydantic import BaseModel

    model = getattr(eda_module, "SummaryModel", None)
    assert model is not None and issubclass(model, BaseModel)


@pytest.mark.unit
def test_summary_model_rejects_nan(eda_module) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        eda_module.SummaryModel(
            schema_version="1.1.0",
            n_total=10,
            n_after_filter=5,
            n_short_lt_100=5,
            text_length_mean=float("nan"),
            text_length_median=100.0,
            text_length_mean_filtered=150.0,
            text_length_median_filtered=150.0,
            filter_threshold=100,
            circuit_counts={"ca9": 10},
            chart_ranges={"text_length_hist": [0, 100000], "citation_density": [0, 100]},
            corpus_manifest_sha="a" * 64,
            figure_hashes={},
            git_sha="abc",
        )


# ---------------------------------------------------------------------------
# 2026 hardening tier 4 — rc_context + clean_stale opt-out
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rcparams_not_mutated_after_render(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    """Plotting must use rc_context; global rcParams unchanged post-run."""
    import matplotlib.pyplot as plt

    before_dpi = plt.rcParams["savefig.dpi"]
    before_autolayout = plt.rcParams["figure.autolayout"]
    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
    )
    assert plt.rcParams["savefig.dpi"] == before_dpi
    assert plt.rcParams["figure.autolayout"] == before_autolayout


@pytest.mark.contract
def test_main_accepts_clean_stale_flag(eda_module) -> None:
    sig = inspect.signature(eda_module.main)
    assert "clean_stale" in sig.parameters
    assert sig.parameters["clean_stale"].default is True


@pytest.mark.unit
def test_clean_stale_false_preserves_prior_artifacts(eda_module, tmp_path: Path, fake_manifest: Path) -> None:
    keeper = tmp_path / "stale_keep.png"
    keeper.write_bytes(b"stale")
    eda_module.main(
        shard_glob=str(MINI_SHARD),
        out_dir=tmp_path,
        manifest_path=fake_manifest,
        log_to_wandb=False,
        clean_stale=False,
    )
    assert keeper.exists()
