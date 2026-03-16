#!/usr/bin/env bash
# scripts/validate_tests.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/validate_tests.sh
# Responsibility: test discovery, unit test gate, environment smoke tests.
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

verify_tests() {
    _require_uv; _require_python
    echo " Verifying test suite..."
    local UNIT_COUNT
    UNIT_COUNT=$("$UV" run pytest tests/ --co -q -m unit 2>/dev/null | grep -c "^tests/" || true)
    if [ "${UNIT_COUNT}" -gt 0 ]; then
        _msg_info "Found ${UNIT_COUNT} unit tests — running as environment verification gate..."
        "$UV" run pytest tests/ -m unit -q --tb=short && \
            _msg_ok "Unit tests passed — environment verified end-to-end" || {
            _msg_error "Unit tests failed" "${UNIT_COUNT} unit tests but one or more failed" \
                "Failing unit tests indicate broken environment — do not proceed to training" \
                ".venv/bin/pytest tests/ -m unit -v --tb=long"
            exit 1
        }
    else
        _msg_info "No unit tests yet — collection check..."
        "$UV" run pytest tests/ --co -q 2>/dev/null && \
            _msg_ok "Test collection ok — no unit tests to run yet" || \
            _msg_warn "Test collection failed" "pytest --co could not collect tests" \
                "informational" ".venv/bin/python -c 'import src.environment; import src.repro'"
    fi
}
