#!/usr/bin/env bash
# scripts/validate_tests.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/validate_tests.sh
# Responsibility: test discovery, unit test gate, environment smoke tests,
#                 and shell script tests via bats-core.
# Sourced by setup.sh — defines functions only, no top-level execution.

run_env_smoke_tests() {
    _require_python
    echo " Running environment smoke tests..."
    $PYTHON -c "
import torch, sys
ver=torch.__version__
if not (ver.startswith('2.') and 'cu' in ver):
    print(f'\033[0;31m  ✗ Wrong torch build: {ver!r} — expected 2.x+cuXXX\033[0m')
    print('    Fix: rm -rf .venv && bash setup.sh  (uv.lock pins cu117 wheel)')
    sys.exit(1)
t=torch.tensor([1.0,2.0,3.0])
assert torch.allclose(t.mean(),torch.tensor(2.0))
print(f'  \033[0;32m✓\033[0m torch {ver} — tensor op ok')
"
    $PYTHON -c "
import transformers, sys
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('bert-base-uncased',local_files_only=False)
ids=tok('hello world',return_tensors='pt')
if ids['input_ids'].shape[1]==0:
    print('\033[0;31m  ✗ tokenizer produced empty output\033[0m')
    print('    Fix: STEP=sync_dependencies bash setup.sh')
    sys.exit(1)
print(f'  \033[0;32m✓\033[0m transformers {transformers.__version__} — tokenizer ok')
"
    $PYTHON -c "
import faiss,numpy as np,sys
idx=faiss.IndexFlatL2(64); vecs=np.random.rand(10,64).astype('float32')
idx.add(vecs); D,I=idx.search(vecs[:1],3)
if I.shape!=(1,3):
    print(f'\033[0;31m  ✗ faiss search shape {I.shape} != (1,3)\033[0m')
    print('    Fix: STEP=sync_dependencies bash setup.sh')
    sys.exit(1)
print('  \033[0;32m✓\033[0m faiss — index add/search ok')
"
    $PYTHON -c "
import spacy, sys
nlp=spacy.load('${SPACY_MODEL}')
v=nlp.meta.get('version')
if v!='${SPACY_MODEL_VERSION}':
    print(f'\033[0;31m  ✗ spaCy model {v} != ${SPACY_MODEL_VERSION}\033[0m')
    print('    Fix: STEP=download_nlp_models bash setup.sh')
    sys.exit(1)
doc=nlp('The Supreme Court ruled in favor of the plaintiff.')
ents=[e.label_ for e in doc.ents]
print(f'  \033[0;32m✓\033[0m spacy {spacy.__version__} | model {v} | entities: {ents}')
"
}

run_shell_tests() {
    # Run bats-core tests for shell scripts.
    # bats-core must be installed — install via: npm install -g bats
    # or: brew install bats-core
    # or: git clone bats-core into tests/shell/bats-core and use local install.
    echo " Running shell script tests (bats-core)..."

    local bats_bin=""
    # Prefer project-local bats install, then PATH
    if [ -x "${PROJECT_ROOT}/tests/shell/bats-core/bin/bats" ]; then
        bats_bin="${PROJECT_ROOT}/tests/shell/bats-core/bin/bats"
    elif command -v bats &>/dev/null; then
        bats_bin="$(command -v bats)"
    fi

    if [ -z "$bats_bin" ]; then
        _msg_warn "bats-core not found" \
            "bats binary not found on PATH or at tests/shell/bats-core/bin/bats" \
            "action-required" \
            "Install: npm install -g bats   or: git clone https://github.com/bats-core/bats-core tests/shell/bats-core"
        echo " Shell tests SKIPPED — bats-core not installed."
        return 0
    fi

    _msg_info "bats: $("$bats_bin" --version)"
    _msg_info "Running: tests/shell/test_lib.bats tests/shell/test_bootstrap_env.bats tests/shell/test_preflight.bats"

    local shell_test_files=(
        "${PROJECT_ROOT}/tests/shell/test_lib.bats"
        "${PROJECT_ROOT}/tests/shell/test_bootstrap_env.bats"
        "${PROJECT_ROOT}/tests/shell/test_preflight.bats"
    )

    # Verify test files exist before running
    local missing=()
    for f in "${shell_test_files[@]}"; do
        [ ! -f "$f" ] && missing+=("$f")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        _msg_warn "Shell test files missing" \
            "Not found: ${missing[*]}" \
            "informational" \
            "Re-run bash setup.sh to regenerate, or check git status"
        return 0
    fi

    if "$bats_bin" --tap "${shell_test_files[@]}"; then
        _msg_ok "All shell tests passed"
    else
        _msg_error "Shell tests failed" \
            "One or more bats tests in tests/shell/ failed" \
            "A broken shell helper (guard, messaging, dry-run) will cause silent failures in setup.sh" \
            "Run manually: bats tests/shell/   or: STEP=run_shell_tests bash setup.sh"
        exit 1
    fi
}

verify_tests() {
    _require_uv; _require_python
    echo " Verifying test suite..."

    # --- Shell tests (bats-core) ---
    run_shell_tests

    # --- Python unit tests ---
    local UNIT_COUNT
    UNIT_COUNT=$("$UV" run pytest tests/ --co -q -m unit 2>/dev/null | grep -c "^tests/" || true)
    if [ "${UNIT_COUNT}" -gt 0 ]; then
        _msg_info "Found ${UNIT_COUNT} Python unit tests — running as environment verification gate..."
        "$UV" run pytest tests/ -m unit -q --tb=short && \
            _msg_ok "Python unit tests passed — environment verified end-to-end" || {
            _msg_error "Python unit tests failed" "${UNIT_COUNT} unit tests but one or more failed" \
                "Failing unit tests indicate broken environment — do not proceed to training" \
                ".venv/bin/pytest tests/ -m unit -v --tb=long"
            exit 1
        }
    else
        _msg_info "No Python unit tests yet — collection check..."
        "$UV" run pytest tests/ --co -q 2>/dev/null && \
            _msg_ok "Python test collection ok — no unit tests to run yet" || \
            _msg_warn "Python test collection failed" "pytest --co could not collect tests" \
                "informational" ".venv/bin/python -c 'import src.environment; import src.repro'"
    fi
}
