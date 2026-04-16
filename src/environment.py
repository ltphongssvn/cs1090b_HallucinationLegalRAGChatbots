# src/environment.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/environment.py
"""GPU environment verification and dependency contract enforcement.
This module is the runtime gate that training and evaluation code calls
before doing anything expensive. It answers three questions, each with
its own public entry point:
1. **Are my dependencies sane?** — :func:`run_environment_checks` imports
   every package in :data:`REQUIRED_DEPS` and verifies each one meets its
   declared version constraint. Soft cross-package compatibility warnings
   are emitted via :func:`_check_compat`.
2. **Is the hardware + reproducibility state correct?** —
   :func:`run_preflight_checks` is a hard gate that verifies GPU count,
   compute capability, VRAM, torch CUDA runtime, disk space, the presence
   of the generated ``src/repro.py``, the integrity of the injected
   ``repro_cfg`` dict, live torch determinism flags, OS env vars, and the
   presence of ``uv.lock``. Any failure raises :class:`PreflightError`
   with a numbered list of issues — no training run ever proceeds past
   a partial failure.
3. **What is my environment fingerprint?** —
   :func:`get_environment_summary` returns a dict suitable for logging
   to W&B or embedding in a run manifest.
Design notes
------------
* **Hardware-agnostic preflight**: GPU family names are informational
  (:data:`SUPPORTED_GPU_FAMILIES`); the hard check is on compute
  capability (Ampere minimum, ``(8, 0)``) and VRAM. This lets the same
  code work on A10G/L4 Harvard nodes and on A100 Colab runtimes.
* **Env-overridable thresholds**: ``TARGET_GPU_COUNT`` and
  ``TARGET_VRAM_GB_MIN`` are read from the environment so the Harvard
  setup scripts and ad-hoc Colab sessions can both drive the same code.
  ``TARGET_GPU_COUNT`` is resolved at setup time from SLURM_GPUS_ON_NODE
  (or nvidia-smi count) and written into ``.env`` by setup.sh. When set
  to a positive integer, Check 1 enforces **exact** match — both
  over-allocation and under-allocation fail, catching silent SLURM
  reallocations. ``0`` (default) disables the check for Colab sessions.
* **Two error types**: :class:`AssertionError` for dependency / basic
  environment failures (recoverable — user fixes and retries),
  :class:`PreflightError` for the pre-training hard gate.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from packaging.version import Version

#: Mapping of importable package name → PEP 440 version constraint
#: (``None`` means "importable only, version not checked"). ``sentence_transformers``
#: is ``None`` because its ``__version__`` attribute is unreliable under
#: :mod:`importlib`; ``faiss`` is ``None`` because the wheel publishes no
#: version attribute on the module object.
REQUIRED_DEPS: Dict[str, Optional[str]] = {
    "torch": ">=2.0",
    "transformers": ">=4.35,<4.42",
    "datasets": ">=2.16",
    "sentence_transformers": None,  # v3.1.1 installed; __version__ unreliable via importlib
    "bm25s": None,  # replaces rank_bm25 — memory-mapped sparse retrieval
    "spacy": ">=3.7",
    "faiss": None,
    "langchain": ">=0.1",
    "sklearn": ">=1.3",
    "numpy": ">=1.24,<2.0",
    "pandas": ">=2.1",
    "accelerate": ">=0.20",
    "evaluate": ">=0.4",
    "wandb": ">=0.16",
}
#: Informational allow-list of GPU families known to work with the
#: cu117 torch wheel. Preflight does **not** fail on an unlisted name —
#: it only emits a warning — because new datacenter cards (B200, etc.)
#: should not be blocked by a stale string list.
SUPPORTED_GPU_FAMILIES = ("A10G", "A10", "L4", "L40", "A100", "H100", "V100")


@dataclass
class CompatRule:
    """Declarative cross-package compatibility rule.
    Attributes:
        name: Stable short identifier used in log lines and test assertions.
        check: Zero-arg callable that returns ``True`` when the rule
            **fires** (i.e. the risky condition is present).
        message: Human-readable explanation shown when the rule fires.
        severity: ``"error"`` → raise :class:`AssertionError` and block
            the run; ``"warn"`` → log only, do not block.
    """

    name: str
    check: Callable[[], bool]
    message: str
    severity: Literal["error", "warn"]


def _build_compat_rules() -> List[CompatRule]:
    """Construct the live list of compatibility rules after importing torch + transformers.
    Rules are built lazily (rather than at module import) because they
    need the actual installed versions, and importing torch at module
    load time would make :mod:`src.environment` itself non-importable
    on a broken environment — defeating the purpose of the checker.
    """
    import torch  # type: ignore[import]
    import transformers  # type: ignore[import]

    torch_ver = Version(torch.__version__.split("+")[0])
    tf_ver = Version(transformers.__version__)
    return [
        CompatRule(
            name="torch_transformers_pre_2_1",
            check=lambda: torch_ver < Version("2.1") and tf_ver < Version("4.41"),
            message=(
                f"torch {torch_ver} + transformers {tf_ver}: "
                "this combination may be unstable on some systems. "
                "Upgrade torch>=2.1 or transformers>=4.41 when cluster driver supports it."
            ),
            severity="warn",
        ),
    ]


#: Minimum VRAM (GB) required by :func:`run_environment_checks`. The
#: stricter preflight path uses :data:`PREFLIGHT_VRAM_GB_MIN` instead.
MIN_GPU_MEMORY_GB: int = 10
#: Preflight VRAM minimum, overridable via ``TARGET_VRAM_GB_MIN`` env var.
PREFLIGHT_VRAM_GB_MIN = float(os.environ.get("TARGET_VRAM_GB_MIN", "20.0"))
#: Minimum CUDA compute capability. ``(8, 0)`` = Ampere — covers A100
#: (8.0), A10G (8.6), L4 (8.9), H100 (9.0), etc.
PREFLIGHT_COMPUTE_CAP_MIN = (8, 0)
#: Expected torch CUDA runtime prefix. Must match the wheel locked in
#: ``uv.lock``; drift here indicates a silent wheel substitution.
PREFLIGHT_TORCH_CUDA = "11.7"
#: Free disk space required under the project root (GB).
PREFLIGHT_MIN_DISK_GB = 50.0
#: Reproducibility env-var expectations. Must match the values written
#: by ``scripts/setup_notebook.sh`` into ``.env``.
EXPECTED_PYTHONHASHSEED = "0"
EXPECTED_CUBLAS_CFG = ":4096:8"
EXPECTED_TOKENIZERS_PARALLELISM = "false"


class PreflightError(RuntimeError):
    """Raised when :func:`run_preflight_checks` finds any failure.
    Distinct from :class:`AssertionError` so callers (notebooks,
    training scripts) can catch preflight failures specifically and
    print an actionable remediation banner without catching unrelated
    assertions.
    """


def _get_version(module: Any) -> Optional[str]:
    """Return a cleaned version string for an imported module, or ``None``.
    Tries ``__version__`` then ``VERSION``, strips any ``+local``
    segment (e.g. ``2.0.1+cu117`` → ``2.0.1``) so PEP 440 comparisons
    work.
    """
    for attr in ("__version__", "VERSION"):
        if hasattr(module, attr):
            return str(getattr(module, attr)).split("+")[0]
    return None


def _check_constraint(actual_str: str, constraint: str) -> Tuple[bool, str]:
    """Evaluate a comma-separated PEP 440 constraint against an installed version.
    Args:
        actual_str: The installed version as a string (already stripped
            of any local segment).
        constraint: Comma-separated clause, e.g. ``">=4.35,<4.42"``.
            Only ``>=`` and ``<`` operators are recognised — sufficient
            for every entry in :data:`REQUIRED_DEPS`.
    Returns:
        ``(ok, reason)`` where ``reason`` is empty on success and a
        human-readable diagnostic on failure.
    """
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
    """Run the five basic environment checks, logging per-check pass/fail.
    Unlike :func:`run_preflight_checks`, this function never raises —
    it returns a boolean so callers (e.g. ``make check-env``) can
    aggregate results without try/except.
    Args:
        logger: Optional standard-library logger. When supplied,
            each check emits a ``PASS``/``FAIL`` line at INFO/ERROR.
    Returns:
        ``True`` iff all five checks passed.
    """
    checks: List[Tuple[str, Callable[[], None]]] = [
        ("Every required dependency must be importable and meet version constraints", _check_deps),
        ("CUDA GPU must be detected for training", _check_gpu_available),
        ("GPU must have at least 10GB VRAM for transformer fine-tuning", _check_gpu_memory),
        ("PyTorch must be compiled with CUDA support", _check_pytorch_cuda),
        ("Cross-dependency version constraints must be satisfied", lambda: _check_compat(logger=logger)),
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
    """Hard-gate the environment before any expensive GPU training.
    Runs nine grouped checks and raises :class:`PreflightError` with a
    numbered failure list if *any* of them fail. Partial success is not
    a partial pass — the function either completes silently (with
    optional INFO logging) or aborts the run.
    The nine checks:
    1. **GPU count** — when ``TARGET_GPU_COUNT`` env var is a positive
       integer, ``torch.cuda.device_count()`` must **exactly** equal it
       (both over- and under-allocation fail). ``0`` (default) disables
       the check for Colab/interactive sessions.
    2. **GPU properties** — each device meets compute-cap and VRAM
       minima; unknown family name is a warning, not a failure.
    3. **torch CUDA runtime** — matches :data:`PREFLIGHT_TORCH_CUDA`
       prefix (catches silent wheel substitution).
    4. **Disk space** — at least :data:`PREFLIGHT_MIN_DISK_GB` GB free.
    5. **src/repro.py** — file exists and imports cleanly.
    6. **repro_cfg integrity** — every required key present with the
       expected value (caller must pass the dict).
    7. **torch runtime determinism** — ``use_deterministic_algorithms``
       on, ``cudnn.benchmark`` off, ``cudnn.deterministic`` on.
    8. **OS env vars** — ``PYTHONHASHSEED``, ``CUBLAS_WORKSPACE_CONFIG``,
       ``TOKENIZERS_PARALLELISM`` match expected values.
    9. **uv.lock** — present at the project root.
    Args:
        logger: Optional logger for per-check PASS lines.
        repro_cfg: The dict returned by ``src.repro.configure()``.
            ``None`` is treated as a failure (check 6) because
            preflight without a reproducibility config is meaningless.
    Raises:
        PreflightError: One or more checks failed. The exception
            message is a fully-formatted multi-line report suitable
            for direct display to the user.
    """
    import torch

    expected_gpu_count = int(os.environ.get("TARGET_GPU_COUNT", "0"))
    vram_min = float(os.environ.get("TARGET_VRAM_GB_MIN", str(PREFLIGHT_VRAM_GB_MIN)))
    failures: List[str] = []
    # Check 1: GPU count — exact match (catches over- AND under-allocation)
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if expected_gpu_count > 0 and n != expected_gpu_count:
        failures.append(
            f"GPU count: expected exactly {expected_gpu_count}, got {n}. "
            f"Check CUDA_VISIBLE_DEVICES or cluster allocation (SLURM_GPUS_ON_NODE)."
        )
    else:
        if logger:
            if expected_gpu_count > 0:
                logger.info(f"✓ PASS: GPU count {n} == TARGET_GPU_COUNT={expected_gpu_count} (exact)")
            else:
                logger.info(f"✓ PASS: GPU count {n} (TARGET_GPU_COUNT unset — check skipped)")
    # Check 2: GPU properties — detected at runtime, no hardcoded GPU family name
    if torch.cuda.is_available():
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            if not any(f in name for f in SUPPORTED_GPU_FAMILIES) and logger:
                logger.warning(f"  ⚠ GPU[{i}] '{name}' not in known families {SUPPORTED_GPU_FAMILIES}")
            if cap < PREFLIGHT_COMPUTE_CAP_MIN:
                failures.append(f"GPU[{i}] '{name}' compute cap {cap} < {PREFLIGHT_COMPUTE_CAP_MIN} (Ampere minimum).")
            elif vram_gb < vram_min:
                failures.append(f"GPU[{i}] '{name}' VRAM {vram_gb:.1f}GB < {vram_min}GB minimum.")
            else:
                if logger:
                    logger.info(f"✓ PASS: GPU[{i}] {name} | cap {cap} | {vram_gb:.1f}GB")
    # Check 3: torch CUDA runtime
    if torch.cuda.is_available():
        actual_cuda = torch.version.cuda or "unknown"
        if not actual_cuda.startswith(PREFLIGHT_TORCH_CUDA):
            failures.append(
                f"torch CUDA runtime: expected {PREFLIGHT_TORCH_CUDA} (cu117 wheel), "
                f"got {actual_cuda}. Run: bash setup.sh"
            )
        else:
            if logger:
                logger.info(f"✓ PASS: torch CUDA runtime {actual_cuda}")
    # Check 4: Disk space
    project_root = Path(__file__).resolve().parent.parent
    disk = shutil.disk_usage(project_root)
    free_gb = disk.free / 1e9
    if free_gb < PREFLIGHT_MIN_DISK_GB:
        failures.append(f"Disk space: expected >={PREFLIGHT_MIN_DISK_GB}GB free, got {free_gb:.1f}GB.")
    else:
        if logger:
            logger.info(f"✓ PASS: Disk {free_gb:.1f}GB free")
    # Check 5: src/repro.py
    repro_path = project_root / "src" / "repro.py"
    if not repro_path.exists():
        failures.append("src/repro.py not found. Run: bash setup.sh")
    else:
        try:
            importlib.import_module("src.repro")
            if logger:
                logger.info("✓ PASS: src/repro.py importable")
        except ImportError as e:
            failures.append(f"src/repro.py import failed: {e}")
    # Check 6: repro_cfg integrity
    if repro_cfg is not None:
        required_repro_keys = {
            "PYTHONHASHSEED": EXPECTED_PYTHONHASHSEED,
            "CUBLAS_WORKSPACE_CONFIG": EXPECTED_CUBLAS_CFG,
            "TOKENIZERS_PARALLELISM": EXPECTED_TOKENIZERS_PARALLELISM,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        }
        for key, expected in required_repro_keys.items():
            actual = repro_cfg.get(key)
            if actual != expected:
                failures.append(f"repro_cfg['{key}']: expected {expected!r}, got {actual!r}.")
            else:
                if logger:
                    logger.info(f"✓ PASS: repro_cfg['{key}'] = {actual!r}")
    else:
        failures.append("repro_cfg not provided to run_preflight_checks().")
    # Check 7: torch runtime state
    torch_state_failures: List[str] = []
    if not torch.are_deterministic_algorithms_enabled():
        torch_state_failures.append("torch.use_deterministic_algorithms is NOT enabled. Re-run Cell 1.")
    if torch.backends.cudnn.benchmark:
        torch_state_failures.append("torch.backends.cudnn.benchmark=True. Re-run Cell 1.")
    if not torch.backends.cudnn.deterministic:
        torch_state_failures.append("torch.backends.cudnn.deterministic=False. Re-run Cell 1.")
    if torch_state_failures:
        failures.extend(torch_state_failures)
    else:
        if logger:
            logger.info("✓ PASS: torch runtime state — deterministic")
    # Check 8: OS env vars
    for var, expected in [
        ("PYTHONHASHSEED", EXPECTED_PYTHONHASHSEED),
        ("CUBLAS_WORKSPACE_CONFIG", EXPECTED_CUBLAS_CFG),
        ("TOKENIZERS_PARALLELISM", EXPECTED_TOKENIZERS_PARALLELISM),
    ]:
        actual = os.environ.get(var)
        if actual != expected:
            failures.append(f"os.environ['{var}']={actual!r} — expected {expected!r}. Re-run Cell 1.")
        else:
            if logger:
                logger.info(f"✓ PASS: os.environ['{var}'] = {actual!r}")
    # Check 9: uv.lock present
    uvlock_path = project_root / "uv.lock"
    if not uvlock_path.exists():
        failures.append("uv.lock not found. Run: uv lock && git add uv.lock && git commit")
    else:
        if logger:
            logger.info("✓ PASS: uv.lock present")
    if failures:
        msg = f"\n{'=' * 60}\n  PREFLIGHT FAILED — {len(failures)} issue(s) detected.\n{'=' * 60}\n" + "\n".join(
            f"  [{i + 1}] {f}" for i, f in enumerate(failures)
        )
        raise PreflightError(msg)
    if logger:
        logger.info("\n✓ All preflight checks passed — safe to proceed to training cells.")


def _check_deps() -> None:
    """Import every entry in :data:`REQUIRED_DEPS` and enforce its constraint.
    Raises:
        AssertionError: One or more packages are missing, have
            undetectable versions, or fail their constraint. The
            message lists every failure, not just the first.
    """
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
    """Raise :class:`AssertionError` if no CUDA device is visible to torch."""
    import torch

    if not torch.cuda.is_available():
        raise AssertionError("No CUDA GPU detected")


def _check_gpu_memory() -> None:
    """Raise :class:`AssertionError` if device 0 has <:data:`MIN_GPU_MEMORY_GB` GB VRAM."""
    import torch

    props: Any = torch.cuda.get_device_properties(0)
    gigabytes: float = props.total_memory / 1e9
    if gigabytes < MIN_GPU_MEMORY_GB:
        raise AssertionError(f"{gigabytes:.1f}GB < {MIN_GPU_MEMORY_GB}GB")


def _check_pytorch_cuda() -> None:
    """Raise :class:`AssertionError` if the installed torch wheel lacks CUDA.
    A CPU-only wheel has ``torch.version.cuda is None``; this catches
    the common failure where a fallback install silently replaced the
    cu117 wheel with the CPU build.
    """
    import torch

    if torch.version.cuda is None:
        raise AssertionError("PyTorch built without CUDA")


def _check_compat(logger: Any = None) -> None:
    """Evaluate all :class:`CompatRule` instances, surfacing warnings and errors.
    Non-firing and exception-raising rules are skipped silently
    (exceptions are a sign the rule's precondition is not met, not a
    bug). Firing rules are partitioned by severity: warnings go to the
    logger (or stderr as a fallback), errors are aggregated and raised
    as a single :class:`AssertionError` at the end.
    """
    rules = _build_compat_rules()
    errors: List[str] = []
    warnings: List[str] = []
    for rule in rules:
        try:
            fired = rule.check()
        except (KeyError, AttributeError, Exception):
            continue
        if not fired:
            continue
        if rule.severity == "error":
            errors.append(f"[{rule.name}] {rule.message}")
        else:
            warnings.append(f"[{rule.name}] {rule.message}")
    if warnings and logger:
        logger.warning("Compat warnings (non-blocking):\n  " + "\n  ".join(warnings))
    elif warnings:
        print("WARNING — Compat (non-blocking):\n  " + "\n  ".join(warnings), file=sys.stderr)
    if errors:
        raise AssertionError("Compat failures:\n  " + "\n  ".join(errors))


def get_environment_summary() -> Dict[str, Any]:
    """Return a dict snapshot of the verified environment for logging.
    Captures Python version, every :data:`REQUIRED_DEPS` package version,
    GPU name/count/memory, torch CUDA runtime, determinism flags, and
    the three reproducibility env vars. Safe to serialise to JSON.
    Returns:
        A flat dict ready for ``wandb.config.update`` or inclusion in
        an experiment manifest.
    """
    import torch

    summary: Dict[str, Any] = {"python": sys.version.split()[0]}
    for pkg in REQUIRED_DEPS:
        try:
            mod = importlib.import_module(pkg)
            summary[pkg] = _get_version(mod) or "installed"
        except ImportError:
            summary[pkg] = "MISSING"
    summary["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    summary["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        props: Any = torch.cuda.get_device_properties(0)
        summary["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
    else:
        summary["gpu_memory_gb"] = 0.0
    summary["cuda"] = torch.version.cuda
    summary["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    summary["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    summary["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    summary["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", "NOT SET")
    summary["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "NOT SET")
    summary["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "NOT SET")
    return summary
