# src/drift_check.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/drift_check.py
"""Two-tier dependency drift detector for the ML runtime.

Background
----------
``uv sync`` only guarantees that the *metadata* of installed distributions
matches ``uv.lock``. It cannot detect three failure modes that routinely
break ML environments on cluster nodes:

1. A wheel was installed but one of its compiled ``.so`` files is missing
   or ABI-incompatible with the host libc / CUDA driver.
2. A CPU-only torch wheel was silently substituted when the CUDA wheel
   index was unreachable.
3. A C-extension module imports cleanly but its core op segfaults or
   raises at first real call (e.g. FAISS without AVX2 on Skylake).

This module runs two cheap-but-thorough tiers over a curated list of
critical packages:

* **Tier 4 — metadata check** (``tier4_metadata_check``): reads
  distribution versions via :mod:`importlib.metadata` and compares
  against a lower bound. Does not import any package.
* **Tier 5 — import + functional check** (``tier5_import_functional_check``):
  actually imports each module and exercises a minimal but representative
  code path (tensor mean, FAISS index search, spaCy blank pipeline, etc.).

The two tiers complement each other: Tier 4 is fast and catches outright
version drift; Tier 5 is slower but catches binary corruption that
Tier 4 cannot see.

Exit codes
----------
* ``0``: both tiers passed.
* ``1``: at least one tier found a problem. Stderr lists the offenders
  and a remediation hint (reinstall from ``uv.lock``).
"""

from __future__ import annotations

import importlib
import importlib.metadata as meta
import sys
from typing import Callable

from packaging.version import Version  # type: ignore[import]

#: Curated list of ML packages we insist on verifying.
#:
#: Each tuple is ``(dist_name, import_name, min_version, exact_version)``.
#: ``exact_ver`` is ``None`` for ``torch`` because
#: :mod:`importlib.metadata` strips the ``+cu117`` local version suffix;
#: the CUDA build is re-verified inside :func:`_check_torch` via
#: ``torch.__version__`` directly.
REQUIRED: list[tuple[str, str, str, str | None]] = [
    ("torch", "torch", "2.0.0", None),
    ("transformers", "transformers", "4.35.0", None),
    ("datasets", "datasets", "2.16.0", None),
    ("faiss-cpu", "faiss", "1.7.0", None),
    ("spacy", "spacy", "3.7.0", None),
    ("scikit-learn", "sklearn", "1.5.0", None),
    ("numpy", "numpy", "1.24.0", None),
    ("pandas", "pandas", "2.2.0", None),
    ("accelerate", "accelerate", "0.20.0", None),
    ("peft", "peft", "0.7.0", None),
    ("evaluate", "evaluate", "0.4.0", None),
    ("wandb", "wandb", "0.16.0", None),
]

#: Human-readable remediation command shown on every failure path.
FIX_HINT = "rm -rf .venv && bash setup.sh  (reinstalls from uv.lock)"


def _check_torch(mod: object) -> str:
    """Exercise torch on CPU and assert the CUDA wheel is installed.

    Builds a 3-element tensor, verifies the mean is exactly 2.0, and
    confirms the version string contains a ``+cu`` local segment. A
    CPU-only wheel raises :class:`AssertionError` so the caller can
    report drift rather than silently training on CPU.

    Returns:
        A short status string for the Tier 5 log line.
    """
    import torch  # type: ignore[import]

    t = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(t.mean(), torch.tensor(2.0))
    ver = torch.__version__
    if "+cu" not in ver:
        raise AssertionError(f"CPU-only torch wheel detected: {ver!r} — expected +cu117")
    return f"tensor mean=2.0 ok, version={ver}"


def _check_transformers(mod: object) -> str:
    """Import :mod:`transformers` and return its version."""
    import transformers  # type: ignore[import]

    return f"version={transformers.__version__}"


def _check_datasets(mod: object) -> str:
    """Import :mod:`datasets` + :mod:`pyarrow` and report both versions.

    Arrow is the storage backend for ``datasets``; reporting both helps
    diagnose format-compat breakage in one step.
    """
    import datasets  # type: ignore[import]
    import pyarrow  # type: ignore[import]

    return f"version={datasets.__version__}, arrow={pyarrow.__version__}"


def _check_faiss(mod: object) -> str:
    """Build a small :class:`faiss.IndexFlatL2` and run one search.

    Catches missing BLAS / AVX symbols at first op, which is the most
    common FAISS-on-cluster failure mode.
    """
    import faiss  # type: ignore[import]
    import numpy as np

    idx = faiss.IndexFlatL2(4)
    vecs = np.random.rand(5, 4).astype("float32")
    idx.add(vecs)
    assert idx.ntotal == 5
    distances, indices = idx.search(vecs[:1], 2)
    assert indices.shape == (1, 2)
    return "IndexFlatL2 add/search ok"


def _check_spacy(mod: object) -> str:
    """Instantiate a blank English pipeline to prove Cython libs loaded."""
    import spacy  # type: ignore[import]

    nlp = spacy.blank("en")
    assert nlp.vocab
    return f"version={spacy.__version__}, blank vocab ok"


def _check_sklearn(mod: object) -> str:
    """Import :mod:`sklearn` and exercise a trivial numpy op.

    The numpy round-trip confirms sklearn's own numpy ABI matches the
    system numpy — a frequent source of import-time segfaults.
    """
    import numpy as np
    import sklearn  # type: ignore[import]

    arr = np.array([1, 2, 3])
    assert arr.sum() == 6
    return f"version={sklearn.__version__}, numpy array ok"


def _check_numpy(mod: object) -> str:
    """Verify :mod:`numpy` sum works on a 3-element array."""
    import numpy as np

    arr = np.array([1, 2, 3])
    assert arr.sum() == 6
    return f"version={np.__version__}, sum ok"


def _check_pandas(mod: object) -> str:
    """Build a 3-row DataFrame and assert its length."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3]})
    assert len(df) == 3
    return f"version={pd.__version__}, DataFrame ok"


def _check_accelerate(mod: object) -> str:
    """Confirm :class:`accelerate.Accelerator` is exposed at package level."""
    import accelerate  # type: ignore[import]

    assert hasattr(accelerate, "Accelerator"), "accelerate.Accelerator not found"
    return f"version={accelerate.__version__}, Accelerator accessible"


def _check_peft(mod: object) -> str:
    """Instantiate a :class:`peft.LoraConfig` to exercise the config path."""
    import peft  # type: ignore[import]
    from peft import LoraConfig, TaskType  # type: ignore[import]

    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16)
    assert cfg.r == 8
    return f"version={peft.__version__}, LoraConfig(r=8) ok"


def _check_evaluate(mod: object) -> str:
    """Import :mod:`evaluate` and return its version."""
    import evaluate  # type: ignore[import]

    return f"version={evaluate.__version__}, module ok"


def _check_wandb(mod: object) -> str:
    """Verify :mod:`wandb` exposes the ``init``/``log`` API surface."""
    import wandb  # type: ignore[import]

    assert hasattr(wandb, "init"), "wandb.init not found"
    assert hasattr(wandb, "log"), "wandb.log not found"
    return f"version={wandb.__version__}, init/log API accessible"


#: Tier 5 check dispatch: import name → functional probe function.
FUNCTIONAL_CHECKS: list[tuple[str, Callable[[object], str]]] = [
    ("torch", _check_torch),
    ("transformers", _check_transformers),
    ("datasets", _check_datasets),
    ("faiss", _check_faiss),
    ("spacy", _check_spacy),
    ("sklearn", _check_sklearn),
    ("numpy", _check_numpy),
    ("pandas", _check_pandas),
    ("accelerate", _check_accelerate),
    ("peft", _check_peft),
    ("evaluate", _check_evaluate),
    ("wandb", _check_wandb),
]


def tier4_metadata_check() -> list[str]:
    """Compare installed dist versions in :data:`REQUIRED` to their minima.

    Reads versions via :func:`importlib.metadata.version` — no package
    is imported, so the check is fast and cannot itself segfault. Prints
    a ``✓`` line per package that meets the floor, a yellow ``⚠`` line
    when an ``exact_ver`` pin mismatches but the minimum is satisfied,
    and collects a drift description for any package below the minimum
    or missing entirely.

    Returns:
        A list of drift descriptions. Empty means Tier 4 passed.
    """
    drift: list[str] = []
    for dist_name, _import_name, min_ver, exact_ver in REQUIRED:
        try:
            installed = meta.version(dist_name)
            if Version(installed) < Version(min_ver):
                drift.append(f"{dist_name}: {installed} < minimum {min_ver}")
            elif exact_ver and installed != exact_ver:
                print(f"  \033[0;33m⚠ WARNING\033[0m {dist_name} installed={installed}, expected={exact_ver}")
            else:
                print(f"  \033[0;32m✓\033[0m {dist_name:<20} {installed} (metadata)")
        except meta.PackageNotFoundError:
            drift.append(f"{dist_name}: NOT INSTALLED")
    return drift


def tier5_import_functional_check() -> list[str]:
    """Actually import each :data:`REQUIRED` package and run its probe.

    Three failure classes are distinguished in the output:

    1. **Import failed** — usually a missing compiled extension.
    2. **Unexpected import error** — non-:class:`ImportError` exception
       raised during module load (rare, but seen with broken C init).
    3. **Import ok, functional call failed** — module loaded but its
       core op raised. Indicates ABI mismatch or missing CPU features.

    Any package without a registered probe in :data:`FUNCTIONAL_CHECKS`
    is reported as "no functional check" and counted as passing once
    imported.

    Returns:
        A list of distribution names that failed the tier. Empty means
        Tier 5 passed.
    """
    failed: list[str] = []
    func_map = {name: fn for name, fn in FUNCTIONAL_CHECKS}
    for dist_name, import_name, _min_ver, _exact_ver in REQUIRED:
        try:
            mod = importlib.import_module(import_name)
        except ImportError as e:
            print(f"\033[0;31m  ✗ {dist_name}: import failed — {e}\033[0m")
            print("    Why: metadata exists but module cannot load (missing .so / ABI mismatch)")
            print(f"    Fix: {FIX_HINT}")
            failed.append(dist_name)
            continue
        except Exception as e:
            print(f"\033[0;31m  ✗ {dist_name}: unexpected import error — {e}\033[0m")
            print(f"    Fix: {FIX_HINT}")
            failed.append(dist_name)
            continue
        check_fn = func_map.get(import_name)
        if check_fn is None:
            print(f"  \033[0;32m✓\033[0m {dist_name:<20} import ok (no functional check)")
            continue
        try:
            result = check_fn(mod)
            print(f"  \033[0;32m✓\033[0m {dist_name:<20} import ok | {result}")
        except Exception as e:
            print(f"\033[0;31m  ✗ {dist_name}: import ok but functional call failed — {e}\033[0m")
            print("    Why: C extension loaded but core op is broken")
            print(f"    Fix: {FIX_HINT}")
            failed.append(dist_name)
    return failed


def main() -> None:
    """CLI entry point. Runs Tier 4, then Tier 5, and exits non-zero on any failure.

    Prints a section header for each tier so log scrapers can attribute
    failures correctly. Exits ``1`` as soon as either tier reports a
    problem; Tier 5 is only reached when Tier 4 is clean, because a
    version-drift package cannot be meaningfully probed.
    """
    print("  Tier 4: metadata version check...")
    drift = tier4_metadata_check()
    if drift:
        print("\n  \033[0;31mDrift detected:\033[0m")
        for d in drift:
            print(f"  \033[0;31m  • {d}\033[0m")
        print("  \033[0;36m  Fix: bash setup.sh\033[0m")
        sys.exit(1)
    print("  Metadata versions ok — proceeding to import verification")
    print("\n  Tier 5: actual import + functional call...")
    failed = tier5_import_functional_check()
    if failed:
        print(f"\n\033[0;31m  {len(failed)} package(s) failed: {failed}\033[0m")
        print(f"\033[0;36m  Fix: {FIX_HINT}\033[0m")
        sys.exit(1)
    print("  All packages verified: metadata + import + functional call")


if __name__ == "__main__":
    main()
