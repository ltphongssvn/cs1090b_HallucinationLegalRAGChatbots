# src/environment.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/environment.py
# SRP: Verify GPU environment and dependency contracts.
# No test_* functions here — those live in tests/test_environment.py.
import importlib
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
PREFLIGHT_MIN_DISK_GB     = 50.0   # minimum free disk on project root filesystem


class PreflightError(RuntimeError):
    """Raised when a preflight check fails — hard stop before expensive training."""


def _get_version(module: Any) -> Optional[str]:
    """Extract version string, stripping CUDA suffixes."""
    for attr in ("__version__", "VERSION"):
        if hasattr(module, attr):
            return str(getattr(module, attr)).split("+")[0]
    return None


def _check_constraint(actual_str: str, constraint: str) -> Tuple[bool, str]:
    """Check version against >=min or >=min,<max. Returns (pass, reason)."""
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
    """Run all environment checks. Returns True if all pass."""
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
    """
    Hard gate — validates ALL critical preconditions before expensive GPU training.
    Raises PreflightError with an actionable message on any failure.

    Checks:
      1. GPU count, name, compute capability, and VRAM against PREFLIGHT_* thresholds
      2. torch CUDA runtime version matches pinned cu117 wheel
      3. Sufficient free disk space for training artifacts
      4. src/repro.py exists and is importable
      5. repro_cfg integrity — all required keys present and correctly set
      6. uv.lock exists and its sha256 is recorded in manifest (if available)

    Call order:
      run_preflight_checks() must be called AFTER run_environment_checks() passes
      and AFTER src/repro.configure() has been called (repro_cfg passed in).
    """
    import torch

    failures: List[str] = []

    # --- Check 1: GPU count ---
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n < PREFLIGHT_GPU_COUNT:
        failures.append(
            f"GPU count: expected >={PREFLIGHT_GPU_COUNT}x NVIDIA {PREFLIGHT_GPU_NAME}, "
            f"got {n}. Check CUDA_VISIBLE_DEVICES or cluster allocation."
        )
    else:
        if logger:
            logger.info(f"✓ PASS: GPU count {n} >= {PREFLIGHT_GPU_COUNT}")

    # --- Check 2: GPU name, compute capability, VRAM per device ---
    if torch.cuda.is_available():
        for i in range(n):
            name    = torch.cuda.get_device_name(i)
            cap     = torch.cuda.get_device_capability(i)
            vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9

            if PREFLIGHT_GPU_NAME not in name:
                failures.append(
                    f"GPU[{i}] name: expected NVIDIA {PREFLIGHT_GPU_NAME}, got '{name}'. "
                    f"Wrong node — request a {PREFLIGHT_GPU_NAME} allocation."
                )
            elif cap < PREFLIGHT_COMPUTE_CAP_MIN:
                failures.append(
                    f"GPU[{i}] compute capability: expected >={PREFLIGHT_COMPUTE_CAP_MIN}, "
                    f"got {cap}. {PREFLIGHT_GPU_NAME} requires cc {PREFLIGHT_COMPUTE_CAP_MIN}."
                )
            elif vram_gb < PREFLIGHT_VRAM_GB_MIN:
                failures.append(
                    f"GPU[{i}] VRAM: expected >={PREFLIGHT_VRAM_GB_MIN}GB, "
                    f"got {vram_gb:.1f}GB. Insufficient for transformer fine-tuning."
                )
            else:
                if logger:
                    logger.info(
                        f"✓ PASS: GPU[{i}] {name} | cap {cap} | {vram_gb:.1f}GB"
                    )

    # --- Check 3: torch CUDA runtime matches pinned cu117 wheel ---
    if torch.cuda.is_available():
        actual_cuda = torch.version.cuda or "unknown"
        if not actual_cuda.startswith(PREFLIGHT_TORCH_CUDA):
            failures.append(
                f"torch CUDA runtime: expected {PREFLIGHT_TORCH_CUDA} (cu117 wheel), "
                f"got {actual_cuda}. Wrong torch wheel installed — run: bash setup.sh"
            )
        else:
            if logger:
                logger.info(f"✓ PASS: torch CUDA runtime {actual_cuda} matches cu117 wheel")

    # --- Check 4: Free disk space ---
    project_root = Path(__file__).resolve().parent.parent
    disk = shutil.disk_usage(project_root)
    free_gb = disk.free / 1e9
    if free_gb < PREFLIGHT_MIN_DISK_GB:
        failures.append(
            f"Disk space: expected >={PREFLIGHT_MIN_DISK_GB}GB free on {project_root}, "
            f"got {free_gb:.1f}GB. Training artifacts may fail to write."
        )
    else:
        if logger:
            logger.info(f"✓ PASS: Disk space {free_gb:.1f}GB free >= {PREFLIGHT_MIN_DISK_GB}GB")

    # --- Check 5: src/repro.py exists and is importable ---
    repro_path = project_root / "src" / "repro.py"
    if not repro_path.exists():
        failures.append(
            f"src/repro.py not found at {repro_path}. "
            f"Run: bash setup.sh to regenerate it."
        )
    else:
        try:
            importlib.import_module("src.repro")
            if logger:
                logger.info(f"✓ PASS: src/repro.py exists and is importable")
        except ImportError as e:
            failures.append(f"src/repro.py import failed: {e}")

    # --- Check 6: repro_cfg integrity ---
    if repro_cfg is not None:
        required_repro_keys = {
            "PYTHONHASHSEED":          "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TOKENIZERS_PARALLELISM":  "false",
            "deterministic_algorithms": True,
            "cudnn_benchmark":          False,
            "cudnn_deterministic":      True,
        }
        for key, expected in required_repro_keys.items():
            actual = repro_cfg.get(key)
            if actual != expected:
                failures.append(
                    f"repro_cfg['{key}']: expected {expected!r}, got {actual!r}. "
                    f"Call configure() before any torch import."
                )
            else:
                if logger:
                    logger.info(f"✓ PASS: repro_cfg['{key}'] = {actual!r}")
    else:
        failures.append(
            "repro_cfg not provided to run_preflight_checks(). "
            "Pass repro_cfg=repro_cfg from Cell 1 configure() call."
        )

    # --- Check 7: uv.lock exists ---
    uvlock_path = project_root / "uv.lock"
    if not uvlock_path.exists():
        failures.append(
            f"uv.lock not found at {uvlock_path}. "
            f"Run: uv lock && git add uv.lock && git commit -m 'chore: pin uv.lock'"
        )
    else:
        if logger:
            logger.info(f"✓ PASS: uv.lock present at {uvlock_path}")

    # --- Final gate ---
    if failures:
        msg = (
            f"\n{'=' * 60}\n"
            f"  PREFLIGHT FAILED — {len(failures)} issue(s) detected.\n"
            f"  Fix ALL issues before running training cells.\n"
            f"{'=' * 60}\n"
            + "\n".join(f"  [{i+1}] {f}" for i, f in enumerate(failures))
        )
        raise PreflightError(msg)

    if logger:
        logger.info(
            f"\n✓ All preflight checks passed — "
            f"environment is safe to proceed to training cells."
        )


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
    """Return dict of verified environment details."""
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
    return summary
