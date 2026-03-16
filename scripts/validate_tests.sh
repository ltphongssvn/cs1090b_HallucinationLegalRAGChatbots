#!/usr/bin/env bash
# scripts/validate_tests.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/validate_tests.sh
# Responsibility: test discovery, tiered test execution, environment smoke tests,
#                 and shell script tests via bats-core.
#
# verify_tests() runs four tiers:
#   Tier 1 — Shell tests (bats-core): setup.sh helpers + artifact verification
#   Tier 2 — Python unit tests (marker=unit): fast, no I/O, always executed
#   Tier 3 — CPU inference smoke test: real forward pass on CPU
#   Tier 4 — GPU smoke subset (marker=gpu): only when SKIP_GPU != 1

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
    echo " Running shell script tests (bats-core)..."

    local bats_bin=""
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

    local shell_test_files=(
        "${PROJECT_ROOT}/tests/shell/test_lib.bats"
        "${PROJECT_ROOT}/tests/shell/test_bootstrap_env.bats"
        "${PROJECT_ROOT}/tests/shell/test_preflight.bats"
        "${PROJECT_ROOT}/tests/shell/test_failure_paths.bats"
        "${PROJECT_ROOT}/tests/shell/test_artifact_verification.bats"
    )

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
        _msg_ok "All shell tests passed (helpers + failure paths + artifact verification)"
    else
        _msg_error "Shell tests failed" \
            "One or more bats tests in tests/shell/ failed" \
            "A broken helper or incorrect artifact causes silent failures downstream" \
            "Run manually: bats tests/shell/   or: STEP=run_shell_tests bash setup.sh"
        exit 1
    fi
}

_run_cpu_inference_smoke_test() {
    _msg_info "Running CPU inference smoke test (real forward pass — not just import)..."
    $PYTHON -c "
import sys, torch, torch.nn as nn
from transformers import AutoTokenizer

try:
    tok = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
    ids = tok('The court ruled in favor of the plaintiff.', return_tensors='pt')
    assert ids['input_ids'].shape[1] > 0, 'tokenizer produced empty output'
    seq_len = ids['input_ids'].shape[1]
    print(f'  \033[0;32m✓\033[0m tokenizer: seq_len={seq_len} tokens')
except Exception as e:
    print(f'\033[0;31m  ✗ tokenizer forward pass failed: {e}\033[0m')
    print('    Fix: rm -rf ~/.cache/huggingface && STEP=run_env_smoke_tests bash setup.sh')
    sys.exit(1)

try:
    vocab_size = tok.vocab_size
    embed_dim  = 32
    num_labels = 2

    class TinyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.fc1   = nn.Linear(embed_dim, 16)
            self.fc2   = nn.Linear(16, num_labels)
            self.relu  = nn.ReLU()
        def forward(self, input_ids):
            x = self.embed(input_ids).mean(dim=1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    model = TinyClassifier().eval()
    with torch.no_grad():
        logits = model(ids['input_ids'])
        probs  = torch.softmax(logits, dim=-1)

    assert logits.shape == (1, num_labels)
    assert abs(probs.sum().item() - 1.0) < 1e-5
    print(f'  \033[0;32m✓\033[0m CPU forward pass: probs={[round(p,3) for p in probs[0].tolist()]}')
    print(f'  \033[0;32m✓\033[0m CPU inference smoke test passed')
except Exception as e:
    print(f'\033[0;31m  ✗ CPU forward pass failed: {e}\033[0m')
    print('    Fix: rm -rf .venv && bash setup.sh')
    sys.exit(1)

try:
    import faiss, numpy as np
    dim=64; n_vecs=100
    index=faiss.IndexFlatL2(dim)
    vecs=np.random.rand(n_vecs,dim).astype('float32')
    index.add(vecs)
    D,I=index.search(vecs[:5],3)
    assert index.ntotal==n_vecs
    assert I.shape==(5,3)
    assert (D[:,0]<1e-5).all()
    print(f'  \033[0;32m✓\033[0m FAISS: {n_vecs}-vector index, self-query L2 ≈ 0 confirmed')
except Exception as e:
    print(f'\033[0;31m  ✗ FAISS forward pass failed: {e}\033[0m')
    print('    Fix: rm -rf .venv && bash setup.sh')
    sys.exit(1)
" || {
        _msg_error "CPU inference smoke test failed" \
            "A real forward pass failed despite import success" \
            "Broken C extension, ABI mismatch, or corrupt wheel" \
            "rm -rf .venv && bash setup.sh"
        exit 1
    }
    _msg_ok "CPU inference smoke test passed"
}

_run_gpu_smoke_subset() {
    _require_uv; _require_python
    _msg_info "Running GPU test subset (marker=gpu)..."

    local GPU_COUNT
    GPU_COUNT=$("$UV" run pytest tests/ --co -q -m gpu 2>/dev/null | grep -c "^tests/" || true)

    if [ "${GPU_COUNT}" -gt 0 ]; then
        _msg_info "Found ${GPU_COUNT} GPU-marked tests — running..."
        "$UV" run pytest tests/ -m gpu -q --tb=short && \
            _msg_ok "GPU test subset passed (${GPU_COUNT} tests)" || {
            _msg_error "GPU tests failed" \
                "${GPU_COUNT} gpu-marked tests but one or more failed" \
                "GPU kernel dispatch broken — do not proceed to training" \
                ".venv/bin/pytest tests/ -m gpu -v --tb=long"
            exit 1
        }
    else
        _msg_info "No gpu-marked tests found yet — GPU subset skipped"
        _msg_info "To add: decorate a test with @pytest.mark.gpu"
    fi
}

verify_tests() {
    _require_uv; _require_python
    echo " Verifying test suite (tiered execution)..."

    # Tier 1: Shell tests (helpers + failure paths + artifact verification)
    run_shell_tests

    # Tier 2: Python unit tests — executed, not just collected
    local UNIT_COUNT
    UNIT_COUNT=$("$UV" run pytest tests/ --co -q -m unit 2>/dev/null | grep -c "^tests/" || true)

    if [ "${UNIT_COUNT}" -gt 0 ]; then
        _msg_info "Tier 2: ${UNIT_COUNT} unit tests — executing..."
        "$UV" run pytest tests/ -m unit -q --tb=short && \
            _msg_ok "Tier 2 passed: ${UNIT_COUNT} unit tests executed" || {
            _msg_error "Unit tests failed" \
                "${UNIT_COUNT} unit tests collected but one or more FAILED execution" \
                "Failing unit tests = broken environment. Do not proceed to training." \
                ".venv/bin/pytest tests/ -m unit -v --tb=long"
            exit 1
        }
    else
        _msg_info "No unit tests yet — collection check as diagnostic only..."
        if "$UV" run pytest tests/ --co -q 2>/dev/null; then
            _msg_warn "No unit tests to execute" \
                "pytest --co succeeded but no unit-marked tests found" \
                "informational" \
                "Collection-only is NOT a gate. Add unit tests to enable Tier 2."
        else
            _msg_warn "Test collection failed" "pytest --co could not import tests" \
                "informational" ".venv/bin/python -c 'import src.environment; import src.repro'"
        fi
    fi

    # Tier 3: CPU inference smoke test (always runs)
    _msg_info "Tier 3: CPU inference smoke test..."
    _run_cpu_inference_smoke_test

    # Tier 4: GPU test subset (skipped when SKIP_GPU=1)
    if [ "${SKIP_GPU:-0}" = "1" ]; then
        _msg_skip "Tier 4: GPU test subset — SKIP_GPU=1, skipped"
    else
        _msg_info "Tier 4: GPU test subset (marker=gpu)..."
        _run_gpu_smoke_subset
    fi

    _msg_ok "All verification tiers passed"
}
