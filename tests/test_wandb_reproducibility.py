"""RED test: assert W&B reproducibility properties for TF-reviewer reproducibility.

Each test encodes ONE empirically-verified property the codebase must satisfy
for a third-party reviewer to reproduce results from W&B run data alone.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ["src", "scripts"]
EXCLUDE_DIR_PARTS = {".venv", ".ipynb_checkpoints", "__pycache__", ".git"}


def _py_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        for p in (REPO / root).rglob("*.py"):
            if any(part in EXCLUDE_DIR_PARTS for part in p.parts):
                continue
            files.append(p)
    return files


@pytest.mark.contract
class TestWandbReproducibilityContract:
    """Reviewer-reproducibility properties for the W&B integration."""

    def test_git_sha_helper_is_centralized_not_duplicated(self) -> None:
        """_git_sha logic must live in ONE module; other defs are thin wrappers.

        A thin wrapper is a local def that simply delegates to
        ``src.repro.get_git_sha``. Re-implementations (subprocess calls,
        env-var lookups) are forbidden anywhere except src/repro.py.
        """
        reimplementations: list[str] = []
        for f in _py_files():
            text = f.read_text(errors="ignore")
            # find each def _git_sha(...) ... block
            for m in re.finditer(
                r"^def\s+_git_sha\s*\([^)]*\)[^:]*:\n((?:[ \t]+.*\n)+)",
                text,
                re.MULTILINE,
            ):
                body = m.group(1)
                rel = f.relative_to(REPO).as_posix()
                # canonical impl lives in src/repro.py — that one IS allowed
                if rel == "src/repro.py":
                    continue
                # thin wrapper: must call get_git_sha and must NOT subprocess
                calls_canonical = "get_git_sha" in body
                does_subprocess = "subprocess" in body or "rev-parse" in body
                if not calls_canonical or does_subprocess:
                    reimplementations.append(rel)
        assert not reimplementations, (
            f"{len(reimplementations)} file(s) reimplement _git_sha "
            f"instead of delegating to src.repro.get_git_sha:\n  - " + "\n  - ".join(sorted(set(reimplementations)))
        )

    def test_every_script_logging_to_wandb_records_git_sha_in_config(self) -> None:
        """Any script calling wandb.init must log git_sha in config.

        Property (not syntactic shape): the file must contain both a
        ``wandb.init`` call and a ``git_sha`` key. This accepts the common
        pattern ``wandb.init(config=summary)`` where ``summary["git_sha"]``
        was set earlier in the same module — equally reproducible as
        passing the literal dict to ``wandb.init``.
        """
        offenders: list[str] = []
        for f in _py_files():
            text = f.read_text(errors="ignore")
            if "wandb.init(" not in text:
                continue
            # Property: file must reference git_sha somewhere alongside wandb.init.
            # Accepts: config={"git_sha": ...}, config=summary (where summary has
            # git_sha), wandb.config.update({"git_sha": ...}), or any other path
            # that gets git_sha into the W&B config.
            if "git_sha" not in text:
                offenders.append(f.relative_to(REPO).as_posix())
        assert not offenders, (
            f"{len(offenders)} script(s) call wandb.init without git_sha "
            f"anywhere in module:\n  - " + "\n  - ".join(sorted(offenders))
        )

    def test_every_script_logging_to_wandb_sets_run_group(self) -> None:
        """wandb.init must pass `group=` so related runs cluster together.

        Without WANDB_RUN_GROUP, a reviewer cannot tell which BM25 / BGE-M3 /
        RRF / reranker / RAG / judge runs belong to the same experiment.
        """
        offenders: list[str] = []
        for p in _py_files():
            text = p.read_text(errors="ignore")
            if "wandb.init(" not in text:
                continue
            # Property check: file calls wandb.init AND references either
            # group= kwarg or WANDB_RUN_GROUP env var. Avoids brittle regex
            # over multi-line wandb.init blocks with nested parens
            # (e.g. group=get_run_group("ms4-baselines")).
            has_group = ("group=" in text) or ("WANDB_RUN_GROUP" in text)
            if not has_group:
                offenders.append(p.relative_to(REPO).as_posix())
        assert not offenders, f"{len(offenders)} script(s) call wandb.init without group=:\n  - " + "\n  - ".join(
            sorted(offenders)
        )

    def test_downstream_scripts_use_artifact_for_inputs(self) -> None:
        """Scripts consuming retrieval results must use_artifact, not raw I/O.

        Without use_artifact, W&B lineage breaks at every stage boundary;
        reviewer cannot trace bm25 -> rrf -> reranker -> rag -> judge.
        """
        downstream = [
            "scripts/baseline_rrf.py",
            "scripts/baseline_reranker.py",
            "scripts/rag_generate.py",
            "scripts/hallucination_judge.py",
            "scripts/stratified_eval.py",
        ]
        offenders: list[str] = []
        for rel in downstream:
            p = REPO / rel
            if not p.exists():
                continue
            text = p.read_text(errors="ignore")
            if "use_artifact" not in text:
                offenders.append(rel)
        assert not offenders, (
            f"{len(offenders)} downstream script(s) lack use_artifact "
            f"(broken lineage):\n  - " + "\n  - ".join(offenders)
        )

    def test_repro_module_exposes_canonical_helpers(self) -> None:
        """src/repro.py must export get_git_sha() and get_run_group()."""
        repro = REPO / "src" / "repro.py"
        assert repro.exists(), "src/repro.py not found"
        text = repro.read_text()
        for fn in ("get_git_sha", "get_run_group"):
            assert re.search(rf"^def\s+{fn}\s*\(", text, re.MULTILINE), f"src/repro.py must define {fn}()"

    def test_offline_runs_have_sync_instructions(self) -> None:
        """README or docs/ must document syncing offline runs to W&B cloud.

        Cluster has no internet -> runs are offline-only. Without sync
        instructions, reviewers see an empty W&B project.
        """
        candidates = [
            REPO / "README.md",
            REPO / "docs" / "wandb_sync.md",
            REPO / "docs" / "REPRODUCIBILITY.md",
        ]
        keywords = ("wandb sync", "WANDB_MODE", "offline-run")
        found = False
        for c in candidates:
            if c.exists():
                t = c.read_text(errors="ignore").lower()
                if any(k.lower() in t for k in keywords):
                    found = True
                    break
        assert found, (
            "No documentation for syncing offline W&B runs. Reviewers "
            "cannot reproduce without `wandb sync wandb/offline-run-*` instructions."
        )
