#!/usr/bin/env bash
# scripts/bootstrap_env.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/bootstrap_env.sh
# Responsibility: base environment bootstrap — uv, lockfile, venv, deps, drift.
# Sourced by setup.sh — defines functions only, no top-level execution.
#
# Mutating steps and their DRY_RUN behaviour:
#   ensure_venv        — would create/delete .venv
#   sync_dependencies  — would install packages from uv.lock

_check_disk_space() {
    local free_gb msg
    free_gb=$(df -BG "${PROJECT_ROOT}" | awk 'NR==2 {gsub("G",""); print $4}')
    if [ "${free_gb:-0}" -lt "$TARGET_MIN_DISK_GB" ]; then
        msg="Disk: only ${free_gb}GB free on ${PROJECT_ROOT}, need ${TARGET_MIN_DISK_GB}GB"
        echo -e "  ${C_RED}✗${C_RESET} $msg"; echo "$msg"; return 1
    fi
    _msg_ok "disk: ${free_gb}GB free >= ${TARGET_MIN_DISK_GB}GB"
}

_check_lockfile_present() {
    local ok=true
    if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
        echo -e "  ${C_RED}✗${C_RESET} pyproject.toml not found"
        echo "pyproject.toml not found at ${PROJECT_ROOT}"; ok=false
    fi
    if [ ! -f "${PROJECT_ROOT}/uv.lock" ]; then
        echo -e "  ${C_RED}✗${C_RESET} uv.lock not found"
        echo "uv.lock not found — run: uv lock && git add uv.lock && git commit"; ok=false
    fi
    [ "$ok" = "true" ] && _msg_ok "pyproject.toml + uv.lock: present"
    [ "$ok" = "false" ] && return 1
}

_check_uv_present() {
    if ! command -v uv &>/dev/null && ! command -v ~/.local/bin/uv &>/dev/null; then
        local msg="uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo -e "  ${C_RED}✗${C_RESET} $msg"; echo "$msg"; return 1
    fi
    local uv_bin; uv_bin=$(command -v uv 2>/dev/null || echo ~/.local/bin/uv)
    _msg_ok "uv: $($uv_bin --version)"
}

check_uv() {
    if ! command -v uv &>/dev/null && ! command -v ~/.local/bin/uv &>/dev/null; then
        _msg_error "uv not found" "uv binary not found on PATH or at ~/.local/bin/uv" \
            "uv is the package manager — all installs require it" \
            "curl -LsSf https://astral.sh/uv/install.sh | sh   then: bash setup.sh"
        exit 1
    fi
    UV=$(command -v uv 2>/dev/null || echo ~/.local/bin/uv)
    if ! "$UV" --version &>/dev/null; then
        _msg_error "uv binary broken" "\$UV='$UV' does not execute" \
            "Broken uv silently fails all package installs" \
            "Re-install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    _msg_ok "uv: $("$UV" --version)"
}

check_lockfile() {
    _require_project_root
    [ ! -f "${PROJECT_ROOT}/pyproject.toml" ] && {
        _msg_error "pyproject.toml not found" "No pyproject.toml at ${PROJECT_ROOT}" \
            "Cannot resolve dependencies without it" \
            "cd ~/cs1090b_HallucinationLegalRAGChatbots && bash setup.sh"
        exit 1
    }
    [ ! -f "${PROJECT_ROOT}/uv.lock" ] && {
        _msg_error "uv.lock not found" "No uv.lock at ${PROJECT_ROOT}/uv.lock" \
            "Without uv.lock, uv sync --frozen cannot run and versions are unpinned" \
            "uv lock && git add uv.lock && git commit -m 'chore: pin uv.lock'   then: bash setup.sh"
        exit 1
    }
    _msg_ok "uv.lock sha256: $(sha256sum "${PROJECT_ROOT}/uv.lock" | cut -d' ' -f1)"
}

ensure_venv() {
    _require_uv
    local PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"

    # Idempotent check — skip entirely if venv already correct
    if [ -f "$PYTHON" ] && $PYTHON -c "import sys; sys.exit(0 if sys.version_info[:3] == (${PYVER_TUPLE}) else 1)" 2>/dev/null; then
        _msg_ok ".venv already exists with Python ${TARGET_PYTHON_VERSION} — skipping creation"
        return
    fi

    if [ -d "${PROJECT_ROOT}/.venv" ]; then
        local venv_size; venv_size=$(du -sh "${PROJECT_ROOT}/.venv" 2>/dev/null | cut -f1)
        if _is_dry_run; then
            _msg_dry_run "delete stale .venv (wrong Python version)" "${PROJECT_ROOT}/.venv (${venv_size})"
            _msg_dry_run "create .venv" "uv venv .venv --python ${TARGET_PYTHON_VERSION} --seed"
            step_end "ensure_venv" "DRY"; return
        fi
        _msg_warn "Stale .venv detected" ".venv exists but contains wrong Python version" \
            "action-required" "Removing in 5 seconds — Ctrl+C to cancel"
        echo "         Contents: ${venv_size} on disk"
        sleep 5; rm -rf "${PROJECT_ROOT}/.venv"
    else
        if _is_dry_run; then
            _msg_dry_run "create .venv" "uv venv .venv --python ${TARGET_PYTHON_VERSION} --seed"
            step_end "ensure_venv" "DRY"; return
        fi
    fi

    _msg_info "Creating .venv with Python ${TARGET_PYTHON_VERSION}..."
    "$UV" venv .venv --python "${TARGET_PYTHON_VERSION}" --seed
}

verify_python() {
    _require_python
    local PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"
    $PYTHON -c "import sys; assert sys.version_info[:3] == (${PYVER_TUPLE}), f'Expected ${TARGET_PYTHON_VERSION} got {sys.version}'"
    _msg_ok "Python: $($PYTHON --version)"
    _msg_info "Executable: $($PYTHON -c 'import sys; print(sys.executable)')"
}

sync_dependencies() {
    _require_uv; _require_python

    if _is_dry_run; then
        _msg_dry_run "sync packages from uv.lock" "uv sync --frozen --dev  (installs all pinned deps into .venv)"
        _msg_info "Would install: $(grep -c '^name = ' "${PROJECT_ROOT}/uv.lock" 2>/dev/null || echo '?') packages from uv.lock"
        step_end "sync_dependencies" "DRY"; return
    fi

    _msg_info "Syncing from uv.lock (--frozen) — may take minutes on first run..."
    _msg_info "--dev explicit: ensures pytest/mypy/hypothesis always installed"
    "$UV" sync --frozen --dev
    _msg_ok "Dependencies synced from uv.lock"
}

check_dependency_drift() {
    _require_uv; _require_python
    echo " Checking for dependency drift..."

    # --- Tier 1: Timestamp gate ---
    if [ "${PROJECT_ROOT}/pyproject.toml" -nt "${PROJECT_ROOT}/uv.lock" ]; then
        _msg_error "Stale uv.lock" "pyproject.toml is newer than uv.lock" \
            "Collaborators will install different versions — reproducibility broken" \
            "uv lock && git add uv.lock && git commit -m 'chore: regenerate uv.lock'"
        exit 1
    fi
    _msg_ok "pyproject.toml vs uv.lock timestamp — ok"

    # --- Tier 2: uv lockfile consistency ---
    "$UV" lock --check 2>/dev/null && _msg_ok "uv lock --check — consistent" || {
        _msg_error "uv.lock inconsistency" "uv lock --check failed" \
            "pyproject.toml constraints no longer satisfy uv.lock pins — silent version drift" \
            "uv lock && git add uv.lock && git commit -m 'chore: regenerate uv.lock'"
        exit 1
    }

    # --- Tier 3: venv vs lockfile sync check ---
    "$UV" sync --frozen --dev --check 2>/dev/null && _msg_ok "uv sync --check — matches uv.lock exactly" || {
        _msg_error "Package drift" "Installed packages diverge from uv.lock" \
            "Manual pip install after setup — results may not be reproducible" \
            "bash setup.sh  (re-syncs .venv from uv.lock)"
        exit 1
    }

    # --- Tier 4: Metadata version check ---
    # Verifies version strings via importlib.metadata (fast, no import needed).
    # NOTE: this check passes even for broken installs — a missing .so or
    # corrupt wheel records the version but fails on import. Tier 5 catches that.
    $PYTHON -c "
import importlib.metadata as meta
from packaging.version import Version
required = {
    'torch':('2.0.0','2.0.1+cu117'),'transformers':('4.35.0',None),
    'datasets':('2.16.0',None),'faiss-cpu':('1.7.0',None),
    'spacy':('3.7.0',None),'scikit-learn':('1.5.0',None),
    'numpy':('1.24.0',None),'pandas':('2.2.0',None),
}
drift=[]
for pkg,(min_v,exact_v) in required.items():
    try:
        inst=meta.version(pkg)
        if Version(inst)<Version(min_v): drift.append(f'{pkg}: {inst} < minimum {min_v}')
        elif exact_v and inst!=exact_v: print(f'  \033[0;33m⚠ WARNING\033[0m {pkg} {inst} != expected {exact_v} (check wheel type)')
        else: print(f'  \033[0;32m✓\033[0m {pkg:<20} {inst} (metadata)')
    except meta.PackageNotFoundError: drift.append(f'{pkg}: NOT INSTALLED')
if drift:
    print('\n  \033[0;31mDrift detected:\033[0m')
    for d in drift: print(f'  \033[0;31m  • {d}\033[0m')
    print('  \033[0;36m  Fix: bash setup.sh\033[0m')
    raise SystemExit(1)
print('  Metadata versions ok — proceeding to import verification')
"

    # --- Tier 5: Actual import + minimal functional call ---
    # Catches broken installs that pass metadata checks:
    # e.g. missing .so files, corrupt wheels, ABI mismatches, missing C extensions.
    # Each package gets the minimum op needed to confirm the C extension loaded.
    _msg_info "Verifying actual imports and minimal functional calls (catches broken .so / ABI issues)..."
    $PYTHON -c "
import sys

def _check(pkg_label, import_fn, functional_fn, fix_hint):
    try:
        mod = import_fn()
    except ImportError as e:
        print(f'\033[0;31m  ✗ {pkg_label}: import failed — {e}\033[0m')
        print(f'    Why: Package metadata exists but the module cannot be loaded')
        print(f'         Likely cause: missing .so file, ABI mismatch, or corrupt wheel')
        print(f'    Fix: {fix_hint}')
        return False
    except Exception as e:
        print(f'\033[0;31m  ✗ {pkg_label}: unexpected import error — {e}\033[0m')
        print(f'    Fix: {fix_hint}')
        return False
    try:
        result = functional_fn(mod)
        print(f'  \033[0;32m✓\033[0m {pkg_label:<20} import ok | functional: {result}')
        return True
    except Exception as e:
        print(f'\033[0;31m  ✗ {pkg_label}: import succeeded but functional call failed — {e}\033[0m')
        print(f'    Why: The package loaded but its C extension or core op is broken')
        print(f'    Fix: {fix_hint}')
        return False

fix = 'rm -rf .venv && bash setup.sh  (reinstalls from uv.lock)'

checks = [
    # (label, import_fn, functional_fn, fix_hint)
    # torch: import + CPU tensor op (confirms C extension loaded)
    ('torch',
     lambda: __import__('torch'),
     lambda m: f'tensor mean={m.tensor([1.0,2.0,3.0]).mean().item():.1f}',
     fix),

    # transformers: import + tokenizer instantiation (confirms tokenizer C++ bindings)
    ('transformers',
     lambda: __import__('transformers'),
     lambda m: f'AutoTokenizer ok, version={m.__version__}',
     fix),

    # datasets: import + confirm Arrow backend loaded
    ('datasets',
     lambda: __import__('datasets'),
     lambda m: f'version={m.__version__}, arrow={__import__(\"pyarrow\").__version__}',
     fix),

    # faiss: import + flat index creation (confirms BLAS/faiss .so loaded)
    ('faiss',
     lambda: __import__('faiss'),
     lambda m: (lambda idx: f'IndexFlatL2 ok, ntotal={idx.ntotal}')(m.IndexFlatL2(4)),
     fix),

    # spacy: import + language object creation (confirms spacy C extensions)
    ('spacy',
     lambda: __import__('spacy'),
     lambda m: f'version={m.__version__}, vocab ok={bool(m.blank(\"en\").vocab)}',
     fix),

    # sklearn: import + array check (confirms scipy/numpy C extensions)
    ('sklearn',
     lambda: __import__('sklearn'),
     lambda m: (lambda a: f'version={m.__version__}, array ok')(__import__('numpy').array([1,2,3])),
     fix),

    # numpy: import + array op (confirms numpy C extension loaded)
    ('numpy',
     lambda: __import__('numpy'),
     lambda m: f'version={m.__version__}, sum([1,2,3])={m.array([1,2,3]).sum()}',
     fix),

    # pandas: import + DataFrame creation (confirms pandas C extension)
    ('pandas',
     lambda: __import__('pandas'),
     lambda m: f'version={m.__version__}, DataFrame(1 row) ok',
     fix),
]

failed = [label for label, imp, fn, hint in checks if not _check(label, imp, fn, hint)]

if failed:
    print(f'\n\033[0;31m  {len(failed)} package(s) failed import/functional verification: {failed}\033[0m')
    print(f'\033[0;36m  These packages passed metadata version check but failed at runtime.\033[0m')
    print(f'\033[0;36m  Fix: rm -rf .venv && bash setup.sh\033[0m')
    sys.exit(1)

print('  All packages verified: metadata version + actual import + functional call')
"
    _msg_ok "Dependency drift check complete — all tiers passed"
}
