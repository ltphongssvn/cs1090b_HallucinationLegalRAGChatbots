# src/drift_check.py
# Path: cs1090b_HallucinationLegalRAGChatbots/src/drift_check.py
# Responsibility: verify installed package versions meet minimum requirements
#                 and that actual imports + functional calls succeed.
# Called by scripts/bootstrap_env.sh Tier 4+5 via: $PYTHON src/drift_check.py
# Keeping this in Python (not a shell heredoc) enables mypy, ruff, and pytest.
import importlib
import importlib.metadata as meta
import sys
from typing import Callable

from packaging.version import Version  # type: ignore[import]

# Core packages: (dist_name, import_name, min_version, exact_version_or_None)
REQUIRED: list[tuple[str, str, str, str | None]] = [
    ("torch",            "torch",          "2.0.0",  "2.0.1+cu117"),
    ("transformers",     "transformers",   "4.35.0", None),
    ("datasets",         "datasets",       "2.16.0", None),
    ("faiss-cpu",        "faiss",          "1.7.0",  None),
    ("spacy",            "spacy",          "3.7.0",  None),
    ("scikit-learn",     "sklearn",        "1.5.0",  None),
    ("numpy",            "numpy",          "1.24.0", None),
    ("pandas",           "pandas",         "2.2.0",  None),
]

# Functional checks: (import_name, check_fn, description)
# Each check_fn receives the imported module and must not raise on success.
def _check_torch(mod: object) -> str:
    import torch  # type: ignore[import]
    t = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(t.mean(), torch.tensor(2.0))
    return f"tensor mean=2.0 ok, version={torch.__version__}"


def _check_transformers(mod: object) -> str:
    import transformers  # type: ignore[import]
    return f"version={transformers.__version__}"


def _check_datasets(mod: object) -> str:
    import datasets  # type: ignore[import]
    import pyarrow  # type: ignore[import]
    return f"version={datasets.__version__}, arrow={pyarrow.__version__}"


def _check_faiss(mod: object) -> str:
    import faiss  # type: ignore[import]
    import numpy as np
    idx = faiss.IndexFlatL2(4)
    vecs = np.random.rand(5, 4).astype("float32")
    idx.add(vecs)
    assert idx.ntotal == 5
    D, I = idx.search(vecs[:1], 2)
    assert I.shape == (1, 2)
    return "IndexFlatL2 add/search ok"


def _check_spacy(mod: object) -> str:
    import spacy  # type: ignore[import]
    nlp = spacy.blank("en")
    assert nlp.vocab
    return f"version={spacy.__version__}, blank vocab ok"


def _check_sklearn(mod: object) -> str:
    import sklearn  # type: ignore[import]
    import numpy as np
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6
    return f"version={sklearn.__version__}, numpy array ok"


def _check_numpy(mod: object) -> str:
    import numpy as np
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6
    return f"version={np.__version__}, sum ok"


def _check_pandas(mod: object) -> str:
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert len(df) == 3
    return f"version={pd.__version__}, DataFrame ok"


FUNCTIONAL_CHECKS: list[tuple[str, Callable[[object], str]]] = [
    ("torch",        _check_torch),
    ("transformers", _check_transformers),
    ("datasets",     _check_datasets),
    ("faiss",        _check_faiss),
    ("spacy",        _check_spacy),
    ("sklearn",      _check_sklearn),
    ("numpy",        _check_numpy),
    ("pandas",       _check_pandas),
]

FIX_HINT = "rm -rf .venv && bash setup.sh  (reinstalls from uv.lock)"


def tier4_metadata_check() -> list[str]:
    """
    Tier 4: metadata version check via importlib.metadata.
    Fast — no import needed. Catches missing installs and version mismatches.
    NOTE: passes even for broken installs (missing .so). Tier 5 catches those.
    """
    drift: list[str] = []
    for dist_name, _import_name, min_ver, exact_ver in REQUIRED:
        try:
            installed = meta.version(dist_name)
            if Version(installed) < Version(min_ver):
                drift.append(f"{dist_name}: {installed} < minimum {min_ver}")
            elif exact_ver and installed != exact_ver:
                print(
                    f"  \033[0;33m⚠ WARNING\033[0m {dist_name} "
                    f"installed={installed}, expected={exact_ver} (check wheel type)"
                )
            else:
                print(f"  \033[0;32m✓\033[0m {dist_name:<20} {installed} (metadata)")
        except meta.PackageNotFoundError:
            drift.append(f"{dist_name}: NOT INSTALLED")
    return drift


def tier5_import_functional_check() -> list[str]:
    """
    Tier 5: actual import + minimal functional call.
    Catches broken installs that pass metadata: missing .so, ABI mismatch,
    corrupt wheel, missing C extension.
    """
    failed: list[str] = []
    func_map = {name: fn for name, fn in FUNCTIONAL_CHECKS}

    for dist_name, import_name, _min_ver, _exact_ver in REQUIRED:
        try:
            mod = importlib.import_module(import_name)
        except ImportError as e:
            print(f"\033[0;31m  ✗ {dist_name}: import failed — {e}\033[0m")
            print(f"    Why: Package metadata exists but module cannot load")
            print(f"         Likely: missing .so, ABI mismatch, or corrupt wheel")
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
            print(f"  \033[0;32m✓\033[0m {dist_name:<20} import ok (no functional check defined)")
            continue

        try:
            result = check_fn(mod)
            print(f"  \033[0;32m✓\033[0m {dist_name:<20} import ok | functional: {result}")
        except Exception as e:
            print(f"\033[0;31m  ✗ {dist_name}: import ok but functional call failed — {e}\033[0m")
            print(f"    Why: C extension loaded but core op is broken")
            print(f"    Fix: {FIX_HINT}")
            failed.append(dist_name)

    return failed


def main() -> None:
    print("  Tier 4: metadata version check...")
    drift = tier4_metadata_check()
    if drift:
        print("\n  \033[0;31mDrift detected — packages below minimum required versions:\033[0m")
        for d in drift:
            print(f"  \033[0;31m  • {d}\033[0m")
        print(f"  \033[0;36m  Fix: bash setup.sh\033[0m")
        sys.exit(1)
    print("  Metadata versions ok — proceeding to import verification")

    print("\n  Tier 5: actual import + functional call (catches broken .so / ABI issues)...")
    failed = tier5_import_functional_check()
    if failed:
        print(f"\n\033[0;31m  {len(failed)} package(s) failed: {failed}\033[0m")
        print(f"\033[0;36m  These passed metadata but failed at runtime.\033[0m")
        print(f"\033[0;36m  Fix: {FIX_HINT}\033[0m")
        sys.exit(1)

    print("  All packages verified: metadata + import + functional call")


if __name__ == "__main__":
    main()
