# src/repro.py — canonical reproducibility module
# Call configure() as FIRST statement in every notebook Cell 1 and CLI script.
# RANDOM_SEED injected from scripts/lib.sh. To change: update lib.sh, re-run, commit.
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
_EXPECTED_PYTHONHASHSEED = "0"
_EXPECTED_CUBLAS_CFG = ":4096:8"
_EXPECTED_TOKENIZERS_PAR = "false"
_RANDOM_SEED = 0


def _load_dotenv(project_root: Optional[Path] = None) -> None:
    root = project_root or Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f".env not found at {env_path}.\n  Fix: bash setup.sh from project root.")
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key = key.replace("export ", "").strip()
                    if key not in os.environ:
                        os.environ[key] = val.strip()


def _apply_torch_flags() -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _EXPECTED_CUBLAS_CFG)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _verify() -> dict:
    import torch

    checks: dict = {}
    for var, expected in [
        ("PYTHONHASHSEED", _EXPECTED_PYTHONHASHSEED),
        ("CUBLAS_WORKSPACE_CONFIG", _EXPECTED_CUBLAS_CFG),
        ("TOKENIZERS_PARALLELISM", _EXPECTED_TOKENIZERS_PAR),
    ]:
        actual = os.environ.get(var)
        if actual != expected:
            raise AssertionError(
                f"{var}={actual!r} — expected {expected!r}.\n"
                f"  Fix: Call configure() as the VERY FIRST statement in Cell 1."
            )
        checks[var] = actual
    if not torch.are_deterministic_algorithms_enabled():
        raise AssertionError("torch.use_deterministic_algorithms not enabled.\n  Fix: re-run Cell 1.")
    checks["deterministic_algorithms"] = True
    if torch.backends.cudnn.benchmark:
        raise AssertionError("cudnn.benchmark=True.\n  Fix: re-run Cell 1.")
    checks["cudnn_benchmark"] = False
    if not torch.backends.cudnn.deterministic:
        raise AssertionError("cudnn.deterministic=False.\n  Fix: re-run Cell 1.")
    checks["cudnn_deterministic"] = True
    checks["random_seed"] = _RANDOM_SEED
    return checks


def configure(project_root: Optional[Path] = None, verbose: bool = True) -> dict:
    """Thin orchestrator: load → apply → seed → verify. Call FIRST in every Cell 1."""
    _load_dotenv(project_root)
    _apply_torch_flags()
    _seed_all(_RANDOM_SEED)
    cfg = _verify()
    if verbose:
        import torch

        print("  [repro] Reproducibility configured:")
        for k, v in cfg.items():
            print(f"    {k}={v}")
        if torch.cuda.is_available():
            print(f"    torch.cuda.manual_seed_all({_RANDOM_SEED}) → {torch.cuda.device_count()} GPU(s)")
    return cfg


# ---------- canonical git SHA + W&B run-group helpers ----------
# Imported by every script that calls wandb.init so git_sha and run group
# stay consistent across the pipeline. Centralizes 16 duplicated _git_sha()
# definitions that drifted between short/full SHA conventions.


def get_git_sha(short: bool = False) -> str:
    """Return the current git commit SHA.

    Resolution order:
      1. ``GIT_COMMIT_SHA`` environment variable (container/CI-safe).
      2. ``git rev-parse HEAD`` from the current working tree.
      3. ``"unknown"`` if both fail.

    Args:
        short: If True, return the first 12 characters of the SHA.

    Returns:
        Full 40-char SHA, 12-char short SHA, or ``"unknown"``.
    """
    env_sha = os.environ.get("GIT_COMMIT_SHA", "").strip()
    if env_sha:
        return env_sha[:12] if short else env_sha
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return sha[:12] if short else sha


def get_run_group(prefix: Optional[str] = None) -> str:
    """Return a stable W&B run group identifier.

    Resolution order:
      1. ``WANDB_RUN_GROUP`` environment variable (operator override).
      2. ``f"{prefix}-{short_git_sha}"`` when ``prefix`` provided.
      3. ``short_git_sha`` (12 chars) as a last resort.

    Use the same prefix for related runs (e.g., ``"ms4-baselines"``) so a
    reviewer can locate every BM25/BGE-M3/RRF/reranker run from one
    experiment under a single W&B group page.
    """
    override = os.environ.get("WANDB_RUN_GROUP", "").strip()
    if override:
        return override
    short_sha = get_git_sha(short=True)
    if prefix:
        return f"{prefix}-{short_sha}"
    return short_sha


def get_wandb_entity(default: Optional[str] = None) -> Optional[str]:
    """Resolve W&B entity from ``WANDB_ENTITY`` env var.

    Returns the env value if set, else ``default``. Reviewers and teammates
    set ``WANDB_ENTITY=their-org`` to redirect runs to their workspace
    without editing source.
    """
    env = os.environ.get("WANDB_ENTITY", "").strip()
    return env or default


def get_wandb_project(default: Optional[str] = None) -> Optional[str]:
    """Resolve W&B project from ``WANDB_PROJECT`` env var.

    Returns the env value if set, else ``default``.
    """
    env = os.environ.get("WANDB_PROJECT", "").strip()
    return env or default
