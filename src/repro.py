# src/repro.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/repro.py
"""Canonical reproducibility configuration module.

This module is the single source of truth for run-time reproducibility
settings. Every notebook's Cell 1 and every CLI entry point must call
:func:`configure` as its **first executable statement**, before importing
any model, dataset, or training code.

What gets configured
--------------------
* ``.env`` is loaded so ``PYTHONHASHSEED``, ``CUBLAS_WORKSPACE_CONFIG``,
  ``TOKENIZERS_PARALLELISM``, and ``RANDOM_SEED`` are in ``os.environ``
  before any torch/transformers import reads them.
* ``torch.use_deterministic_algorithms(True, warn_only=False)`` is
  enabled, along with ``cudnn.benchmark=False`` and
  ``cudnn.deterministic=True``.
* Every RNG (Python ``random``, NumPy, torch CPU, torch CUDA) is
  seeded to ``RANDOM_SEED`` (0 by default).
* Every applied setting is re-read and asserted — a mis-ordered
  notebook cell raises :class:`AssertionError` with a specific fix hint.

Design notes
------------
* **Fails loudly**: any deviation from expected values raises rather
  than warning. Silent drift is the core failure mode this module
  exists to prevent.
* **Order matters**: ``CUBLAS_WORKSPACE_CONFIG`` must be set before
  the first CUDA kernel launches, which means before torch imports
  any CUDA module. This is why :func:`configure` must run first.
* **SRP-decomposed**: each stage (load, apply, seed, verify) is a
  private helper; :func:`configure` is a thin orchestrator so the
  individual stages can be unit-tested in isolation.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Expected value of ``PYTHONHASHSEED`` — 0 disables hash randomization.
_EXPECTED_PYTHONHASHSEED: str = "0"

#: Required ``CUBLAS_WORKSPACE_CONFIG`` for deterministic cuBLAS on
#: Ampere+. ``:4096:8`` reserves eight 4 KiB workspaces per stream.
_EXPECTED_CUBLAS_CFG: str = ":4096:8"

#: Disables HF tokenizers parallelism to avoid fork-after-thread
#: warnings and non-deterministic batch ordering.
_EXPECTED_TOKENIZERS_PAR: str = "false"

#: Global RNG seed applied to Python, NumPy, and torch (CPU + CUDA).
_RANDOM_SEED: int = 0


def _load_dotenv(project_root: Optional[Path] = None) -> None:
    """Load ``.env`` from the project root into :data:`os.environ`.

    Prefers :mod:`python-dotenv` when available and falls back to a
    minimal hand-rolled parser so the module works even in environments
    where python-dotenv has not yet been installed (e.g. the first
    bootstrap step).

    Args:
        project_root: Optional override for the project root. When
            ``None``, resolves to the parent of this file's directory.

    Raises:
        FileNotFoundError: ``.env`` does not exist — indicates setup.sh
            has not yet been run.
    """
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
    """Enable torch deterministic mode and lock cuDNN into deterministic kernels.

    Sets ``CUBLAS_WORKSPACE_CONFIG`` (via ``setdefault`` so ``.env``
    takes precedence if it already provided a value), calls
    :func:`torch.use_deterministic_algorithms` with ``warn_only=False``
    so any non-deterministic op raises rather than silently degrading,
    and disables the cuDNN autotuner.
    """
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _EXPECTED_CUBLAS_CFG)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_all(seed: int) -> None:
    """Seed every RNG the project touches: Python, NumPy, torch CPU + CUDA.

    NumPy and torch are imported lazily so a minimal environment
    without them (e.g. a pure-data smoke test) still works.
    """
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
    """Re-read every setting and assert it matches the expected value.

    Returns:
        A dict of all verified settings for caller display or logging.

    Raises:
        AssertionError: Any env var or torch flag drifted from its
            expected value. The message names the offender and the
            fix ("re-run Cell 1").
    """
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


def configure(
    project_root: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Apply every reproducibility setting and return the verified config.

    Must be called as the **first statement** of every notebook Cell 1
    and every CLI entry point, before any model, dataset, or training
    import. The four-stage orchestration (load → apply → seed → verify)
    guarantees that by the time this function returns, every RNG is
    seeded, every torch flag is deterministic, and every env var
    matches its expected value — or :class:`AssertionError` has been
    raised naming the first drift.

    Args:
        project_root: Optional override for the ``.env`` lookup root.
        verbose: When ``True`` (default), prints each verified setting
            plus the CUDA device count to stdout.

    Returns:
        The dict returned by :func:`_verify`, suitable for passing to
        :func:`src.environment.run_preflight_checks` as ``repro_cfg``.
    """
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
