"""RED tests: remaining W&B reproducibility gaps (Oct 2025 industry checklist).

Properties (each maps to a documented W&B best practice):
  A. Seed in config — every training/retrieval/RAG script logs ``seed`` into
     wandb.init(config=...) so reviewers can reproduce stochastic outputs.
  B. Output artifact — every pipeline-stage script that consumes inputs via
     use_artifact also produces a versioned output via run.log_artifact, so
     the W&B lineage DAG is connected end-to-end.
  C. Env-overridable project/entity — no hardcoded W&B entity/project; both
     resolve via WANDB_ENTITY / WANDB_PROJECT environment variables so a
     reviewer can redirect runs to their own workspace without code edits.
  D. Reproducibility runbook — docs/REPRODUCIBILITY.md exists and covers
     env vars, offline sync, branch/commit pinning, and DVC pull.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXCLUDE_DIR_PARTS = {".venv", ".ipynb_checkpoints", "__pycache__", ".git"}


def _py_files() -> list[Path]:
    files: list[Path] = []
    for root in ("src", "scripts"):
        for p in (REPO / root).rglob("*.py"):
            if any(part in EXCLUDE_DIR_PARTS for part in p.parts):
                continue
            files.append(p)
    return files


@pytest.mark.contract
class TestWandbReproducibilityGapsRound2:
    """Round-2 reproducibility gaps remaining after the initial 6-test round."""

    def test_seed_logged_alongside_wandb_init(self) -> None:
        """Every script calling wandb.init must also reference 'seed' in module.

        Property check: presence of both ``wandb.init(`` and ``seed`` in the
        same module. Accepts ``config={"seed": ...}``, ``config=summary_data``
        where summary contains seed, or seed in module-level args.
        """
        offenders: list[str] = []
        for f in _py_files():
            text = f.read_text(errors="ignore")
            if "wandb.init(" not in text:
                continue
            if "seed" not in text:
                offenders.append(f.relative_to(REPO).as_posix())
        assert not offenders, f"{len(offenders)} script(s) call wandb.init without seed:\n  - " + "\n  - ".join(
            sorted(offenders)
        )

    def test_no_hardcoded_wandb_entity_in_scripts(self) -> None:
        """W&B entity must come from env, not hardcoded literal strings.

        Hardcoded ``entity="phl690-harvard-extension-schol"`` blocks any
        reviewer or teammate from running the scripts against their own
        W&B workspace. Industry pattern: read from WANDB_ENTITY env var.
        """
        offenders: list[str] = []
        for f in _py_files():
            text = f.read_text(errors="ignore")
            # any string literal that looks like a hardcoded entity in a wandb.init
            if re.search(
                r'wandb\.init\([^)]*entity\s*=\s*["\'][a-zA-Z0-9_-]+["\']',
                text,
                re.DOTALL,
            ):
                offenders.append(f.relative_to(REPO).as_posix())
        assert not offenders, f"{len(offenders)} script(s) hardcode wandb.init(entity=...):\n  - " + "\n  - ".join(
            sorted(offenders)
        )

    def test_pipeline_stage_scripts_log_output_artifact(self) -> None:
        """Every pipeline-stage script must produce a versioned W&B artifact.

        Without ``run.log_artifact(...)`` or a wrapper exposing one, the W&B
        lineage DAG terminates at the input edge — reviewers see what fed
        the run but not the run\'s versioned output. This is the canonical
        W&B pattern: run = init() → use_artifact(input) → log_artifact(output).
        """
        pipeline_stages = [
            "scripts/baseline_rrf.py",
            "scripts/baseline_reranker.py",
            "scripts/rag_generate.py",
            "scripts/hallucination_judge.py",
        ]
        offenders: list[str] = []
        for rel in pipeline_stages:
            p = REPO / rel
            if not p.exists():
                continue
            text = p.read_text(errors="ignore")
            if "log_artifact" not in text:
                offenders.append(rel)
        assert not offenders, f"{len(offenders)} pipeline stage(s) produce no output artifact:\n  - " + "\n  - ".join(
            sorted(offenders)
        )

    def test_reproducibility_runbook_exists_and_is_complete(self) -> None:
        """docs/REPRODUCIBILITY.md must cover the four reviewer onboarding topics."""
        runbook = REPO / "docs" / "REPRODUCIBILITY.md"
        assert runbook.exists(), "docs/REPRODUCIBILITY.md not found"
        text = runbook.read_text(errors="ignore").lower()
        required_topics = {
            "wandb_entity": "WANDB_ENTITY",
            "wandb_project": "WANDB_PROJECT",
            "wandb_sync": "wandb sync",
            "offline_mode": "WANDB_MODE",
            "dvc_pull": "dvc pull",
            "git_commit_pin": "git checkout",
        }
        missing = [name for name, kw in required_topics.items() if kw.lower() not in text]
        assert not missing, (
            f"docs/REPRODUCIBILITY.md missing topics: {missing}\n"
            "Required: WANDB_ENTITY, WANDB_PROJECT, wandb sync, WANDB_MODE, "
            "dvc pull, git checkout"
        )
