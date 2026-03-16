# src/environment.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/environment.py
# SRP: Verify GPU environment and dependency contracts.
# No test_* functions here — those live in tests/test_environment.py.
import importlib
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

from packaging.version import Version

REQUIRED_DEPS: Dict[str, Optional[str]] = {
    "torch":        ">=2.0",
    "transformers": ">=4.35,<4.41",
    "datasets":     ">=2.16",
    "gensim":       ">=4.3",
    "spacy":        ">=3.7",
    "faiss":        None,
    "langchain":    ">=0.1",
    "sklearn":      ">=1.3",
    "numpy":        ">=1.24,<2.0",
    "pandas":       ">=2.1",
    "accelerate":   ">=0.20",
    "evaluate":     ">=0.4",
    "ragas":        ">=0.1",
    "wandb":        ">=0.16",
}

COMPAT_RULES: List[Dict[str, Any]] = [
    {
        "desc": "transformers <4.41 required for torch <2.1",
        "check": lambda mods: (
            Version(mods["torch"].__version__.split("+")[0]) < Version("2.1")
            and Version(mods["transformers"].__version__) < Version("4.41")
        ),
    },
]

MIN_GPU_MEMORY_GB: int = 10

# Preflight thresholds — must match setup.sh hardware constants
PREFLIGHT_GPU_NAME        = "L4"
PREFLIGHT_GPU_COUNT       = 4
PREFLIGHT_VRAM_GB_MIN     = 22.0
PREFLIGHT_COMPUTE_CAP_MIN = (8, 9)
PREFLIGHT_TORCH_CUDA      = "11.7"
PREFLIGHT_MIN_DISK_GB     = 50.0

# Reproducibility expectations — must match src/repro.py and .env exactly
EXPECTED_PYTHONHASHSEED        = "0"
EXPECTED_CUBLAS_CFG            = ":4096:8"
EXPECTED_TOKENIZERS_PARALLELISM = "false"


class PreflightError(RuntimeError):
    """Raised when a preflight check fails — hard stop before expensive training."""


def _get_version(module: Any) -> Optional[str]:
    for attr in ("__version__", "VERSION"):
        if hasattr(module, attr):
            return str(getattr(module, attr)).split("+")[0]
    return None


def _check_constraint(actual_str: str, constraint: str) -> Tuple[bool, str]:
    actual = Version(actual_str)
    for part in constraint.split(","):
        part = part.strip()
        if part.startswith(">="):
            if actual < Version(part[2:]):
                return False, f"{actual_str} < {part[2:]} (need {part})"
        elif part.startswith("<"):
            if actual >= Version(part[1:]):
                return False, f"{actual_str} >= {part[1:]} (need {part})"
    return True, ""


def run_environment_checks(logger: Any = None) -> bool:
    checks: List[Tuple[str, Callable[[], None]]] = [
        ("Every required dependency must be importable and meet version constraints",
         _check_deps),
        ("CUDA GPU must be detected for training",
         _check_gpu_available),
        ("GPU must have at least 10GB VRAM for transformer fine-tuning",
         _check_gpu_memory),
        ("PyTorch must be compiled with CUDA support",
         _check_pytorch_cuda),
        ("Cross-dependency version constraints must be satisfied",
         _check_compat),
    ]
    all_passed = True
    for description, check_fn in checks:
        try:
            check_fn()
            if logger:
                logger.info(f"✓ PASS: {description}")
        except AssertionError as error:
            if logger:
                logger.error(f"✗ FAIL: {description} — {error}")
            all_passed = False
    return all_passed


def run_preflight_checks(
    logger: Any = None,
    repro_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Hard gate before expensive GPU training. Raises PreflightError on failure."""
    import torch

    failures: List[str] = []

    # Check 1: GPU count
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n < PREFLIGHT_GPU_COUNT:
        failures.append(
            f"GPU count: expected >={PREFLIGHT_GPU_COUNT}x NVIDIA {PREFLIGHT_GPU_NAME}, "
            f"got {n}. Check CUDA_VISIBLE_DEVICES or cluster allocation."
        )
    else:
        if logger: logger.info(f"✓ PASS: GPU count {n} >= {PREFLIGHT_GPU_COUNT}")

    # Check 2: GPU name, compute cap, VRAM
    if torch.cuda.is_available():
        for i in range(n):
            name    = torch.cuda.get_device_name(i)
            cap     = torch.cuda.get_device_capability(i)
            vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            if PREFLIGHT_GPU_NAME not in name:
                failures.append(f"GPU[{i}] name: expected {PREFLIGHT_GPU_NAME}, got '{name}'.")
            elif cap < PREFLIGHT_COMPUTE_CAP_MIN:
                failures.append(f"GPU[{i}] compute capability: expected >={PREFLIGHT_COMPUTE_CAP_MIN}, got {cap}.")
            elif vram_gb < PREFLIGHT_VRAM_GB_MIN:
                failures.append(f"GPU[{i}] VRAM: expected >={PREFLIGHT_VRAM_GB_MIN}GB, got {vram_gb:.1f}GB.")
            else:
                if logger: logger.info(f"✓ PASS: GPU[{i}] {name} | cap {cap} | {vram_gb:.1f}GB")

    # Check 3: torch CUDA runtime
    if torch.cuda.is_available():
        actual_cuda = torch.version.cuda or "unknown"
        if not actual_cuda.startswith(PREFLIGHT_TORCH_CUDA):
            failures.append(
                f"torch CUDA runtime: expected {PREFLIGHT_TORCH_CUDA} (cu117 wheel), "
                f"got {actual_cuda}. Run: bash setup.sh"
            )
        else:
            if logger: logger.info(f"✓ PASS: torch CUDA runtime {actual_cuda}")

    # Check 4: Disk space
    project_root = Path(__file__).resolve().parent.parent
    disk = shutil.disk_usage(project_root)
    free_gb = disk.free / 1e9
    if free_gb < PREFLIGHT_MIN_DISK_GB:
        failures.append(f"Disk space: expected >={PREFLIGHT_MIN_DISK_GB}GB free, got {free_gb:.1f}GB.")
    else:
        if logger: logger.info(f"✓ PASS: Disk {free_gb:.1f}GB free")

    # Check 5: src/repro.py
    repro_path = project_root / "src" / "repro.py"
    if not repro_path.exists():
        failures.append("src/repro.py not found. Run: bash setup.sh")
    else:
        try:
            importlib.import_module("src.repro")
            if logger: logger.info("✓ PASS: src/repro.py importable")
        except ImportError as e:
            failures.append(f"src/repro.py import failed: {e}")

    # Check 6: repro_cfg integrity
    if repro_cfg is not None:
        required_repro_keys = {
            "PYTHONHASHSEED":           EXPECTED_PYTHONHASHSEED,
            "CUBLAS_WORKSPACE_CONFIG":  EXPECTED_CUBLAS_CFG,
            "TOKENIZERS_PARALLELISM":   EXPECTED_TOKENIZERS_PARALLELISM,
            "deterministic_algorithms": True,
            "cudnn_benchmark":          False,
            "cudnn_deterministic":      True,
        }
        for key, expected in required_repro_keys.items():
            actual = repro_cfg.get(key)
            if actual != expected:
                failures.append(f"repro_cfg['{key}']: expected {expected!r}, got {actual!r}.")
            else:
                if logger: logger.info(f"✓ PASS: repro_cfg['{key}'] = {actual!r}")
    else:
        failures.append("repro_cfg not provided to run_preflight_checks().")

    # Check 7: torch runtime state — independent of repro_cfg
    torch_state_failures: List[str] = []
    if not torch.are_deterministic_algorithms_enabled():
        torch_state_failures.append(
            "torch.use_deterministic_algorithms is NOT enabled. Re-run Cell 1."
        )
    if torch.backends.cudnn.benchmark:
        torch_state_failures.append(
            "torch.backends.cudnn.benchmark=True. Re-run Cell 1."
        )
    if not torch.backends.cudnn.deterministic:
        torch_state_failures.append(
            "torch.backends.cudnn.deterministic=False. Re-run Cell 1."
        )
    if torch_state_failures:
        failures.extend(torch_state_failures)
    else:
        if logger:
            logger.info("✓ PASS: torch runtime state — deterministic_algorithms=True | cudnn.benchmark=False | cudnn.deterministic=True")

    # Check 8: OS env vars — independent of repro_cfg
    for var, expected in [
        ("PYTHONHASHSEED",          EXPECTED_PYTHONHASHSEED),
        ("CUBLAS_WORKSPACE_CONFIG", EXPECTED_CUBLAS_CFG),
        ("TOKENIZERS_PARALLELISM",  EXPECTED_TOKENIZERS_PARALLELISM),
    ]:
        actual = os.environ.get(var)
        if actual != expected:
            failures.append(f"os.environ['{var}']={actual!r} — expected {expected!r}. Re-run Cell 1.")
        else:
            if logger: logger.info(f"✓ PASS: os.environ['{var}'] = {actual!r}")

    # Check 9: uv.lock present
    uvlock_path = project_root / "uv.lock"
    if not uvlock_path.exists():
        failures.append("uv.lock not found. Run: uv lock && git add uv.lock && git commit")
    else:
        if logger: logger.info("✓ PASS: uv.lock present")

    if failures:
        msg = (
            f"\n{'=' * 60}\n"
            f"  PREFLIGHT FAILED — {len(failures)} issue(s) detected.\n"
            f"{'=' * 60}\n"
            + "\n".join(f"  [{i+1}] {f}" for i, f in enumerate(failures))
        )
        raise PreflightError(msg)

    if logger:
        logger.info("\n✓ All preflight checks passed — safe to proceed to training cells.")


def _check_deps() -> None:
    failures: List[str] = []
    for pkg, constraint in REQUIRED_DEPS.items():
        try:
            mod = importlib.import_module(pkg)
        except ImportError:
            failures.append(f"{pkg}: not installed")
            continue
        if constraint is not None:
            ver = _get_version(mod)
            if ver is None:
                failures.append(f"{pkg}: version undetectable")
            else:
                ok, reason = _check_constraint(ver, constraint)
                if not ok:
                    failures.append(f"{pkg}: {reason}")
    if failures:
        raise AssertionError("Dependency failures:\n  " + "\n  ".join(failures))


def _check_gpu_available() -> None:
    import torch
    if not torch.cuda.is_available():
        raise AssertionError("No CUDA GPU detected")


def _check_gpu_memory() -> None:
    import torch
    props: Any = torch.cuda.get_device_properties(0)
    gigabytes: float = props.total_memory / 1e9
    if gigabytes < MIN_GPU_MEMORY_GB:
        raise AssertionError(f"{gigabytes:.1f}GB < {MIN_GPU_MEMORY_GB}GB")


def _check_pytorch_cuda() -> None:
    import torch
    if torch.version.cuda is None:
        raise AssertionError("PyTorch built without CUDA")


def _check_compat() -> None:
    mods: Dict[str, ModuleType] = {}
    for pkg in REQUIRED_DEPS:
        try:
            mods[pkg] = importlib.import_module(pkg)
        except ImportError:
            pass
    failures: List[str] = []
    for rule in COMPAT_RULES:
        try:
            if not rule["check"](mods):
                failures.append(str(rule["desc"]))
        except (KeyError, AttributeError):
            failures.append(f"{rule['desc']} — could not verify")
    if failures:
        raise AssertionError("Compat failures:\n  " + "\n  ".join(failures))


def get_environment_summary() -> Dict[str, Any]:
    """Return dict of verified environment details including reproducibility state."""
    import torch
    summary: Dict[str, Any] = {"python": sys.version.split()[0]}
    for pkg in REQUIRED_DEPS:
        try:
            mod = importlib.import_module(pkg)
            summary[pkg] = _get_version(mod) or "installed"
        except ImportError:
            summary[pkg] = "MISSING"
    summary["gpu"]           = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    summary["gpu_count"]     = torch.cuda.device_count() if torch.cuda.is_available() else 0
    props: Any               = torch.cuda.get_device_properties(0)
    summary["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
    summary["cuda"]          = torch.version.cuda
    # Reproducibility runtime state snapshot
    summary["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    summary["cudnn_benchmark"]          = torch.backends.cudnn.benchmark
    summary["cudnn_deterministic"]      = torch.backends.cudnn.deterministic
    summary["PYTHONHASHSEED"]           = os.environ.get("PYTHONHASHSEED", "NOT SET")
    summary["CUBLAS_WORKSPACE_CONFIG"]  = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "NOT SET")
    summary["TOKENIZERS_PARALLELISM"]   = os.environ.get("TOKENIZERS_PARALLELISM", "NOT SET")
    return summary
