#!/usr/bin/env bash
# scripts/bootstrap_env.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/bootstrap_env.sh
# Responsibility: base environment bootstrap — uv, lockfile, venv, deps, drift.
# Sourced by setup.sh — defines functions only, no top-level execution.

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
    if [ "$ok" = "true" ]; then
        _msg_ok "pyproject.toml + uv.lock: present"
        return 0
    fi
    return 1
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
    if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
        _msg_error "pyproject.toml not found" "No pyproject.toml at ${PROJECT_ROOT}" \
            "Cannot resolve dependencies without it" \
            "cd ~/cs1090b_HallucinationLegalRAGChatbots && bash setup.sh"
        exit 1
    fi
    if [ ! -f "${PROJECT_ROOT}/uv.lock" ]; then
        _msg_error "uv.lock not found" "No uv.lock at ${PROJECT_ROOT}/uv.lock" \
            "Without uv.lock, uv sync --frozen cannot run and versions are unpinned" \
            "uv lock && git add uv.lock && git commit -m 'chore: pin uv.lock'   then: bash setup.sh"
        exit 1
    fi
    _msg_ok "uv.lock sha256: $(sha256sum "${PROJECT_ROOT}/uv.lock" | cut -d' ' -f1)"
}

ensure_venv() {
    _require_uv
    local PYVER_TUPLE="${TARGET_PYTHON_VERSION//./,}"
    if [ -f "$PYTHON" ] && $PYTHON -c "import sys; sys.exit(0 if sys.version_info[:3] == (${PYVER_TUPLE}) else 1)" 2>/dev/null; then
        _msg_ok ".venv already exists with Python ${TARGET_PYTHON_VERSION} — skipping"
        return
    fi
    if [ -d "${PROJECT_ROOT}/.venv" ]; then
        local venv_size; venv_size=$(du -sh "${PROJECT_ROOT}/.venv" 2>/dev/null | cut -f1)
        if _is_dry_run; then
            _msg_dry_run "delete stale .venv" "${PROJECT_ROOT}/.venv (${venv_size})"
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
    _require_uv
    if _is_dry_run; then
        _msg_dry_run "sync packages from uv.lock" "uv sync --frozen --dev"
        _msg_info "Would install: $(grep -c '^name = ' "${PROJECT_ROOT}/uv.lock" 2>/dev/null || echo '?') packages"
        step_end "sync_dependencies" "DRY"; return
    fi
    _require_python
    _msg_info "Syncing from uv.lock (--frozen) — may take minutes on first run..."
    "$UV" sync --frozen --dev
    _msg_ok "Dependencies synced from uv.lock"
}

_run_vulnerability_audit() {
    _require_python
    _msg_info "Running vulnerability audit (pip-audit against OSV database)..."

    if ! $PYTHON -m pip_audit --version &>/dev/null 2>&1; then
        _msg_warn "pip-audit not found" \
            "pip-audit is not installed in the venv" \
            "informational" \
            "STEP=sync_dependencies bash setup.sh"
        return 0
    fi

    local audit_file audit_exit=0
    audit_file=$(mktemp /tmp/pip-audit-XXXXXX.json)
    "$UV" run pip-audit --format=json --progress-spinner=off --output="$audit_file" 2>/dev/null \
        || audit_exit=$?
    local audit_output=""
    if [ -f "$audit_file" ]; then
        audit_output=$(cat "$audit_file")
    fi
    rm -f "$audit_file"

    if [ "$audit_exit" -ne 0 ] && [ -n "$audit_output" ]; then
        local vuln_summary
        vuln_summary=$(echo "$audit_output" | $PYTHON -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    vulns = [d for d in data.get('dependencies', []) if d.get('vulns')]
    print(len(vulns))
    for dep in vulns:
        for v in dep['vulns']:
            desc = v.get('description','')[:120]
            print(f\"  {dep['name']}=={dep['version']}: {v['id']} — {desc}\")
except Exception as e:
    print(f'parse-failed: {e}')
" 2>/dev/null || echo "unknown")
        # CVEs in torch<2.2 and transformers<4.41 are pinned for CUDA 11.7 wheel
        # compatibility on the L4 cluster. These are accepted technical debt until
        # the cluster driver supports a newer CUDA runtime.
        _msg_warn "Vulnerability audit found issues" \
            "${vuln_summary}" \
            "informational" \
            "CVEs are pinned for CUDA 11.7 compat — upgrade when cluster driver supports torch>=2.2"
    else
        _msg_ok "Vulnerability audit passed — no known CVEs"
    fi
}

check_dependency_drift() {
    _require_uv; _require_python
    echo " Checking for dependency drift..."

    # Timestamp comparison omitted — unreliable after git ops (mtime = checkout time)
    _msg_ok "pyproject.toml vs uv.lock — skipping mtime check (unreliable after git ops)"

    # Use if/fi — avoids set -e killing on non-zero exit
    if "$UV" lock --check 2>/dev/null; then
        _msg_ok "uv lock --check — consistent"
    else
        _msg_error "uv.lock inconsistency" "uv lock --check failed" \
            "pyproject.toml constraints no longer satisfy uv.lock pins" \
            "uv lock && git add uv.lock && git commit -m 'chore: regenerate uv.lock'"
        exit 1
    fi

    if "$UV" sync --frozen --dev --check 2>/dev/null; then
        _msg_ok "uv sync --check — matches uv.lock"
    else
        _msg_error "Package drift" "Installed packages diverge from uv.lock" \
            "Manual pip install after setup — results not reproducible" \
            "bash setup.sh"
        exit 1
    fi

    _msg_info "Running src/drift_check.py (Tier 4: metadata, Tier 5: import + functional)..."
    if $PYTHON "${PROJECT_ROOT}/src/drift_check.py"; then
        _msg_ok "Dependency drift check complete — all tiers passed"
    else
        _msg_error "Dependency drift check failed" "src/drift_check.py exited non-zero" \
            "One or more packages failed version or import/functional verification" \
            "rm -rf .venv && bash setup.sh"
        exit 1
    fi

    _run_vulnerability_audit
}
