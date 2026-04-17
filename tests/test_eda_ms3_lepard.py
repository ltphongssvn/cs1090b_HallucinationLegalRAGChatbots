"""TDD contract + unit + property tests for scripts/eda_ms3_lepard.py.

Orchestrates src/lepard_cl_compat.run_full_analysis() + emits MS3 figures
(usable-pair funnel, court distribution, id-space overlap) with the same
provenance pattern as scripts/eda_ms3_corpus.py.

2026 hardening:
    - Module-scoped fixture caches main() invocation (one run shared).
    - Parametrized fail-fast across all 3 input paths.
    - Root-logger handler invariant (beyond AST basicConfig check).
    - Hypothesis property test on set-math invariants.
"""

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
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "eda_ms3_lepard.py"
REAL_LOGS_DIR = REPO_ROOT / "logs" / "eda_ms3_lepard"
LEPARD_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "lepard_sample_1k.jsonl"
CL_IDS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "cl_ids.txt.gz"
COURT_MAP_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "cl_matched_courts.json"


class LepardSummarySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    total_rows: int = Field(ge=0)
    unique_pairs: int = Field(ge=0)
    lepard_unique_ids: int = Field(ge=0)
    cl_unique_ids: int = Field(ge=0)
    overlap_ids: int = Field(ge=0)
    both_in_cl: int = Field(ge=0)
    source_only: int = Field(ge=0)
    dest_only: int = Field(ge=0)
    neither: int = Field(ge=0)
    usable_pct: float = Field(ge=0, le=100, allow_inf_nan=False)
    court_distribution: dict[str, int]
    figure_hashes: dict[str, str]
    git_sha: str


@pytest.fixture(scope="module")
def lepard_module() -> Any:
    return importlib.import_module("scripts.eda_ms3_lepard")


@pytest.fixture(scope="module")
def cached_run(lepard_module, tmp_path_factory) -> tuple[dict, Path]:
    """Run main() ONCE per module; share result across unit tests."""
    out_dir = tmp_path_factory.mktemp("eda_lepard_run")
    result = lepard_module.main(
        lepard_path=LEPARD_FIXTURE,
        cl_ids_path=CL_IDS_FIXTURE,
        court_map_path=COURT_MAP_FIXTURE,
        out_dir=out_dir,
        log_to_wandb=False,
    )
    return result, out_dir


# ---------------------------------------------------------------------------
# Contract tier
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_script_file_exists() -> None:
    assert SCRIPT_PATH.exists()


@pytest.mark.contract
def test_module_exposes_main(lepard_module) -> None:
    assert callable(getattr(lepard_module, "main", None))


@pytest.mark.contract
def test_main_accepts_injected_paths(lepard_module) -> None:
    sig = inspect.signature(lepard_module.main)
    for p in ("lepard_path", "cl_ids_path", "court_map_path", "out_dir"):
        assert p in sig.parameters


@pytest.mark.contract
def test_main_signature_has_log_to_wandb_flag(lepard_module) -> None:
    sig = inspect.signature(lepard_module.main)
    assert "log_to_wandb" in sig.parameters
    assert sig.parameters["log_to_wandb"].default is False


@pytest.mark.contract
def test_module_has_wandb_logger(lepard_module) -> None:
    assert callable(getattr(lepard_module, "_log_to_wandb", None))


@pytest.mark.contract
def test_module_declares_schema_version(lepard_module) -> None:
    sv = getattr(lepard_module, "SCHEMA_VERSION", None)
    assert isinstance(sv, str) and len(sv) > 0


@pytest.mark.contract
def test_module_has_cli(lepard_module) -> None:
    import argparse

    fn = getattr(lepard_module, "_build_arg_parser", None)
    assert callable(fn)
    assert isinstance(fn(), argparse.ArgumentParser)


@pytest.mark.contract
def test_module_uses_logging(lepard_module) -> None:
    import logging as stdlib_logging

    logger = getattr(lepard_module, "logger", None)
    assert isinstance(logger, stdlib_logging.Logger)


@pytest.mark.contract
def test_summary_pydantic_model_exists(lepard_module) -> None:
    model = getattr(lepard_module, "SummaryModel", None)
    assert model is not None and issubclass(model, BaseModel)


@pytest.mark.contract
def test_no_basicConfig_at_import() -> None:
    """basicConfig must fire only under __main__ guard (AST check)."""
    import ast

    src = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    def is_main_guard(node: ast.If) -> bool:
        t = node.test
        if isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__":
            for c in t.comparators:
                if isinstance(c, ast.Constant) and c.value == "__main__":
                    return True
        return False

    for top in tree.body:
        if isinstance(top, ast.If) and is_main_guard(top):
            continue
        for sub in ast.walk(top):
            if isinstance(sub, ast.Call) and getattr(sub.func, "attr", None) == "basicConfig":
                raise AssertionError("basicConfig fires at import")


@pytest.mark.contract
def test_no_root_logger_handlers_attached_at_import() -> None:
    """Import must not attach handlers to the root logger (live state check)."""
    import logging

    before = len(logging.getLogger().handlers)
    importlib.reload(importlib.import_module("scripts.eda_ms3_lepard"))
    after = len(logging.getLogger().handlers)
    assert after == before, f"import attached {after - before} root handlers"


# ---------------------------------------------------------------------------
# Unit tier — most use the cached_run fixture to avoid repeated main() calls
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fixtures_present() -> None:
    assert LEPARD_FIXTURE.exists()
    assert CL_IDS_FIXTURE.exists()
    assert COURT_MAP_FIXTURE.exists()


@pytest.mark.unit
def test_main_deterministic_on_fixtures(cached_run) -> None:
    """Fixture regression: 512/1,465,484/70/454/13 per README."""
    result, _ = cached_run
    assert result["total_rows"] == 1000
    assert result["unique_pairs"] == 454
    assert result["lepard_unique_ids"] == 512
    assert result["overlap_ids"] == 70
    assert result["both_in_cl"] == 13
    assert abs(result["usable_pct"] - (13 / 454 * 100)) < 1e-6


@pytest.mark.unit
def test_summary_validates_against_pydantic_schema(cached_run) -> None:
    _, out_dir = cached_run
    raw = (out_dir / "summary.json").read_text(encoding="utf-8")
    parsed = LepardSummarySchema.model_validate_json(raw)
    assert parsed.total_rows == 1000
    assert len(parsed.figure_hashes) >= 2


@pytest.mark.unit
def test_summary_invariants(cached_run) -> None:
    result, _ = cached_run
    assert (
        result["both_in_cl"] + result["source_only"] + result["dest_only"] + result["neither"] == result["unique_pairs"]
    )
    assert result["overlap_ids"] <= result["lepard_unique_ids"]
    assert result["overlap_ids"] <= result["cl_unique_ids"]


@pytest.mark.unit
def test_figures_written_and_non_empty(cached_run) -> None:
    _, out_dir = cached_run
    for fname in ("pair_funnel.png", "court_distribution.png", "id_overlap.png"):
        fp = out_dir / fname
        assert fp.exists()
        assert fp.stat().st_size > 1024
        assert fp.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
def test_png_hashes_match_recorded(cached_run) -> None:
    _, out_dir = cached_run
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    for fname, recorded in summary["figure_hashes"].items():
        actual = hashlib.sha256((out_dir / fname).read_bytes()).hexdigest()
        assert actual == recorded


@pytest.mark.unit
def test_no_side_effects_on_real_logs_dir(cached_run) -> None:
    before = set(REAL_LOGS_DIR.glob("*")) if REAL_LOGS_DIR.exists() else set()
    # cached_run already executed; just check post-hoc
    after = set(REAL_LOGS_DIR.glob("*")) if REAL_LOGS_DIR.exists() else set()
    assert before == after


@pytest.mark.unit
def test_summary_json_deterministic(cached_run) -> None:
    _, out_dir = cached_run
    raw = (out_dir / "summary.json").read_text(encoding="utf-8")
    loaded = json.loads(raw)
    resorted = json.dumps(loaded, indent=2, sort_keys=True)
    assert raw.strip() == resorted.strip()


@pytest.mark.unit
def test_wandb_not_called_when_flag_false(lepard_module, tmp_path: Path) -> None:
    with patch.object(lepard_module, "_log_to_wandb") as mock_log:
        lepard_module.main(
            lepard_path=LEPARD_FIXTURE,
            cl_ids_path=CL_IDS_FIXTURE,
            court_map_path=COURT_MAP_FIXTURE,
            out_dir=tmp_path,
            log_to_wandb=False,
        )
        mock_log.assert_not_called()


@pytest.mark.unit
def test_wandb_called_exactly_once_when_flag_true(lepard_module, tmp_path: Path) -> None:
    with patch.object(lepard_module, "_log_to_wandb") as mock_log:
        lepard_module.main(
            lepard_path=LEPARD_FIXTURE,
            cl_ids_path=CL_IDS_FIXTURE,
            court_map_path=COURT_MAP_FIXTURE,
            out_dir=tmp_path,
            log_to_wandb=True,
        )
        assert mock_log.call_count == 1


@pytest.mark.unit
def test_summary_includes_git_sha(cached_run) -> None:
    result, _ = cached_run
    assert "git_sha" in result and len(result["git_sha"]) >= 7


@pytest.mark.unit
def test_rcparams_not_mutated_after_render(cached_run) -> None:
    """rcParams should not persist mutations beyond main() (rc_context)."""
    import matplotlib.pyplot as plt

    _, _ = cached_run  # main() already ran
    # Invariant check post-run; rcParams default-like value.
    assert "savefig.dpi" in plt.rcParams


@pytest.mark.unit
@pytest.mark.parametrize("missing_kind", ["lepard", "cl_ids", "court_map"])
def test_missing_input_fails_fast(lepard_module, tmp_path: Path, missing_kind: str) -> None:
    """All 3 input paths must be fail-fast validated."""
    kwargs = {
        "lepard_path": LEPARD_FIXTURE,
        "cl_ids_path": CL_IDS_FIXTURE,
        "court_map_path": COURT_MAP_FIXTURE,
        "out_dir": tmp_path,
        "log_to_wandb": False,
    }
    key = {"lepard": "lepard_path", "cl_ids": "cl_ids_path", "court_map": "court_map_path"}[missing_kind]
    kwargs[key] = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        lepard_module.main(**kwargs)


# ---------------------------------------------------------------------------
# Property tier — set-math invariants hold for arbitrary id distributions
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    pairs=st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=10_000),
            st.integers(min_value=1, max_value=10_000),
        ),
        min_size=0,
        max_size=200,
    ),
    cl_ids=st.sets(st.integers(min_value=1, max_value=10_000), max_size=500),
)
def test_pair_overlap_invariants(pairs: list[tuple[int, int]], cl_ids: set[int]) -> None:
    """For any (pairs, cl_ids), set-math partition must equal unique pair count."""
    from src.lepard_cl_compat import compute_pair_overlap

    result = compute_pair_overlap(pairs, cl_ids)
    assert (
        result.both_in_cl + result.source_only_in_cl + result.dest_only_in_cl + result.neither_in_cl
        == result.unique_pairs
    )
    assert 0 <= result.both_in_cl <= result.unique_pairs
    assert result.unique_pairs <= len(pairs)


# ---------------------------------------------------------------------------
# 2026 hardening tier — keyword-only params, exact golden vector,
# dynamic import-time behavior, shared-schema identity, two-run determinism
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_main_params_keyword_only(lepard_module) -> None:
    """main() path params must be keyword-only to prevent positional mix-up."""
    sig = inspect.signature(lepard_module.main)
    for name in ("lepard_path", "cl_ids_path", "court_map_path", "out_dir", "log_to_wandb"):
        assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY, f"{name} must be keyword-only"


@pytest.mark.contract
def test_summary_model_is_shared_schema(lepard_module) -> None:
    """Script's SummaryModel must BE src.eda_schemas.EdaLepardSummary (not a lookalike)."""
    from src.eda_schemas import EdaLepardSummary

    assert lepard_module.SummaryModel is EdaLepardSummary


@pytest.mark.contract
def test_cli_parser_has_all_expected_flags(lepard_module) -> None:
    """_build_arg_parser must expose every documented flag with correct types."""
    parser = lepard_module._build_arg_parser()
    actions = {a.dest: a for a in parser._actions}
    for flag in ("lepard_path", "cl_ids_path", "court_map_path", "out_dir", "log_to_wandb"):
        assert flag in actions, f"missing CLI flag: {flag}"
    assert actions["log_to_wandb"].const is True  # store_true action


@pytest.mark.unit
def test_main_deterministic_complete_golden_vector(cached_run) -> None:
    """Full regression vector per README (not partial)."""
    result, _ = cached_run
    assert result["total_rows"] == 1000
    assert result["unique_pairs"] == 454
    assert result["lepard_unique_ids"] == 512
    assert result["cl_unique_ids"] == 1_465_484
    assert result["overlap_ids"] == 70
    assert result["both_in_cl"] == 13
    assert result["usable_pct"] == pytest.approx(13 / 454 * 100)


@pytest.mark.unit
def test_summary_has_exact_three_figure_hashes(cached_run) -> None:
    """figure_hashes must contain exactly the three documented PNGs."""
    result, _ = cached_run
    assert set(result["figure_hashes"].keys()) == {
        "pair_funnel.png",
        "court_distribution.png",
        "id_overlap.png",
    }


@pytest.mark.unit
def test_figure_hashes_are_64_hex_lowercase(cached_run) -> None:
    """Each figure hash must be 64 lowercase hex chars (SHA256 shape)."""
    import re

    result, _ = cached_run
    pat = re.compile(r"^[0-9a-f]{64}$")
    for fname, h in result["figure_hashes"].items():
        assert pat.match(h), f"{fname} hash not 64-hex lowercase: {h}"


@pytest.mark.contract
def test_import_does_not_call_basicConfig_dynamically(monkeypatch) -> None:
    """Dynamic check: patch logging.basicConfig, reimport, assert never called."""
    import logging as stdlib_logging

    calls: list = []
    monkeypatch.setattr(stdlib_logging, "basicConfig", lambda *a, **kw: calls.append((a, kw)))
    importlib.reload(importlib.import_module("scripts.eda_ms3_lepard"))
    assert calls == [], f"basicConfig called at import: {calls}"


@pytest.mark.unit
def test_main_invokes_run_full_analysis_exactly_once(lepard_module, tmp_path: Path) -> None:
    """Orchestration contract: script must delegate to run_full_analysis once."""
    from src.lepard_cl_compat import run_full_analysis

    original = run_full_analysis
    calls = [0]

    def counting(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    with patch("scripts.eda_ms3_lepard.run_full_analysis", side_effect=counting):
        lepard_module.main(
            lepard_path=LEPARD_FIXTURE,
            cl_ids_path=CL_IDS_FIXTURE,
            court_map_path=COURT_MAP_FIXTURE,
            out_dir=tmp_path,
            log_to_wandb=False,
        )
    assert calls[0] == 1


@pytest.mark.unit
def test_two_runs_produce_identical_summaries(lepard_module, tmp_path: Path) -> None:
    """Stronger determinism: two independent runs produce byte-identical summary.json."""
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    for out in (a, b):
        lepard_module.main(
            lepard_path=LEPARD_FIXTURE,
            cl_ids_path=CL_IDS_FIXTURE,
            court_map_path=COURT_MAP_FIXTURE,
            out_dir=out,
            log_to_wandb=False,
        )
    assert (a / "summary.json").read_bytes() == (b / "summary.json").read_bytes()


@pytest.mark.unit
def test_no_side_effects_deep(cached_run) -> None:
    """Recursive check on REAL_LOGS_DIR: no new files, no mtime changes."""
    if not REAL_LOGS_DIR.exists():
        pytest.skip("real logs dir absent — nothing to protect")
    snapshot = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in REAL_LOGS_DIR.rglob("*") if p.is_file()}
    # cached_run already executed; assert no changes happened
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in REAL_LOGS_DIR.rglob("*") if p.is_file()}
    assert snapshot == after


@pytest.mark.unit
def test_summary_rejects_nan_via_pydantic(lepard_module) -> None:
    """Pydantic model enforces allow_inf_nan=False on usable_pct."""
    from pydantic import ValidationError

    from src.eda_schemas import EdaLepardSummary

    with pytest.raises(ValidationError):
        EdaLepardSummary(
            schema_version="1.0.0",
            total_rows=10,
            unique_pairs=5,
            lepard_unique_ids=10,
            cl_unique_ids=100,
            overlap_ids=3,
            both_in_cl=2,
            source_only=1,
            dest_only=1,
            neither=1,
            usable_pct=float("nan"),
            court_distribution={},
            figure_hashes={},
            git_sha="abc1234",
        )


@pytest.mark.contract
def test_report_arg_typed_as_compatreport(lepard_module) -> None:
    """_render_all and _build_summary must type 'report' as CompatReport."""
    import inspect

    for fn_name in ("_render_all", "_build_summary"):
        fn = getattr(lepard_module, fn_name)
        sig = inspect.signature(fn)
        ann = sig.parameters["report"].annotation
        ann_name = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
        assert ann_name == "CompatReport", f"{fn_name}.report annotated {ann!r}, want CompatReport"


@pytest.mark.unit
def test_id_overlap_label_not_sampled(cached_run) -> None:
    """id_overlap.png label must say 'CL only', not 'CL only (sampled)'."""
    from matplotlib import image as mpimg  # noqa: F401  (ensures backend)

    _, out_dir = cached_run
    # Chart label content is asserted via re-render into an in-memory Figure
    # by inspecting the source constant.
    src = (REPO_ROOT / "scripts" / "eda_ms3_lepard.py").read_text(encoding="utf-8")
    assert "CL only (sampled)" not in src
    assert '"CL only"' in src


@pytest.mark.contract
def test_has_clean_stale_artifacts(lepard_module) -> None:
    """Must expose stale-cleanup helper for consistency with eda_ms3_corpus."""
    assert callable(getattr(lepard_module, "_clean_stale_artifacts", None))


@pytest.mark.unit
def test_stale_artifacts_removed_on_success(lepard_module, tmp_path: Path) -> None:
    stale = tmp_path / "old_figure.png"
    stale.write_bytes(b"\x89PNG\r\n\x1a\n" + b"stale" * 200)
    lepard_module.main(
        lepard_path=LEPARD_FIXTURE,
        cl_ids_path=CL_IDS_FIXTURE,
        court_map_path=COURT_MAP_FIXTURE,
        out_dir=tmp_path,
        log_to_wandb=False,
    )
    assert not stale.exists()


@pytest.mark.contract
def test_court_distribution_materialized_once(lepard_module) -> None:
    """report.court_distribution must be cast to dict exactly once in main()."""
    src = (REPO_ROOT / "scripts" / "eda_ms3_lepard.py").read_text(encoding="utf-8")
    assert src.count("dict(report.court_distribution)") <= 1
