# src/repro.py
# Path: cs1090b_HallucinationLegalRAGChatbots/src/repro.py
# Canonical reproducibility configuration module.
# Call configure() as FIRST statement in every notebook Cell 1 and CLI script.
import logging
import os
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_EXPECTED_PYTHONHASHSEED: str = "0"
_EXPECTED_CUBLAS_CFG: str = ":4096:8"
_EXPECTED_TOKENIZERS_PAR: str = "false"
_RANDOM_SEED: int = 0


def _load_dotenv(project_root: Optional[Path] = None) -> None:
    """Load .env. Single responsibility: env var loading only."""
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
    """Apply deterministic torch settings. Single responsibility: flag application only."""
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _EXPECTED_CUBLAS_CFG)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_all(seed: int) -> None:
    """Seed all RNGs. Single responsibility: seeding only."""
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
    """Verify all settings are active. Single responsibility: verification only."""
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
    """
    Thin orchestrator: load → apply → seed → verify.
    Call as the FIRST statement in every notebook Cell 1 and CLI script.
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
