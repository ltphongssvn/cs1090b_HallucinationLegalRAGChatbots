#!/usr/bin/env bash
# scripts/setup_nlp.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/setup_nlp.sh
# Responsibility: NLP asset management — pinned spaCy model download, checksum, install.
# Sourced by setup.sh — defines functions only, no top-level execution.

download_nlp_models() {
    _require_python
    local SPACY_CACHE_DIR="${PROJECT_ROOT}/.cache/spacy"
    local SPACY_WHEEL="${SPACY_CACHE_DIR}/${SPACY_MODEL}-${SPACY_MODEL_VERSION}-py3-none-any.whl"
    echo " Installing spaCy ${SPACY_MODEL} ${SPACY_MODEL_VERSION} (pinned)..."
    mkdir -p "$SPACY_CACHE_DIR"

    if $PYTHON -c "
import spacy,sys
try:
    nlp=spacy.load('${SPACY_MODEL}')
    sys.exit(0 if nlp.meta.get('version')=='${SPACY_MODEL_VERSION}' else 1)
except OSError: sys.exit(1)
" 2>/dev/null; then
        _msg_ok "${SPACY_MODEL} ${SPACY_MODEL_VERSION} already installed — skipping download"
        return
    fi

    if [ ! -f "$SPACY_WHEEL" ]; then
        if [ "${OFFLINE:-0}" = "1" ]; then
            _msg_error "Offline mode: wheel not cached" \
                "OFFLINE=1 but wheel not found at $SPACY_WHEEL" \
                "Cannot download in offline mode" \
                "Pre-cache: mkdir -p .cache/spacy && wget -O $SPACY_WHEEL ${SPACY_MODEL_URL}   then: OFFLINE=1 bash setup.sh"
            exit 1
        fi
        _msg_info "Downloading ${SPACY_MODEL} ${SPACY_MODEL_VERSION} (will be cached at $SPACY_WHEEL)..."
        curl -fsSL -o "$SPACY_WHEEL" "${SPACY_MODEL_URL}"
        [ ! -f "$SPACY_WHEEL" ] && {
            _msg_error "Download failed" "curl succeeded but wheel not present" \
                "spaCy NER required for entity extraction in the RAG pipeline" \
                "Check network: curl -I ${SPACY_MODEL_URL}"
            exit 1
        }
    else
        _msg_info "Using cached wheel: $SPACY_WHEEL"
    fi

    local ACTUAL_SHA; ACTUAL_SHA=$(sha256sum "$SPACY_WHEEL" | cut -d' ' -f1)
    if [ "$ACTUAL_SHA" != "$SPACY_MODEL_SHA256" ]; then
        _msg_error "spaCy wheel checksum mismatch" \
            "expected=${SPACY_MODEL_SHA256}  actual=${ACTUAL_SHA}" \
            "Corrupted wheel produces wrong NER results, silently degrading RAG quality" \
            "Cached wheel deleted. Re-run: bash setup.sh  (will re-download and re-verify)"
        rm -f "$SPACY_WHEEL"; exit 1
    fi
    _msg_ok "Checksum verified: ${ACTUAL_SHA}"
    $PYTHON -m pip install --quiet "$SPACY_WHEEL"

    $PYTHON -c "
import spacy, sys
nlp = spacy.load('${SPACY_MODEL}')
v = nlp.meta.get('version')
if v != '${SPACY_MODEL_VERSION}':
    print(f'\033[0;31m  ✗ Post-install version mismatch: {v} != ${SPACY_MODEL_VERSION}\033[0m')
    print('    Fix: rm -rf .venv && bash setup.sh')
    sys.exit(1)
print(f'  \033[0;32m✓\033[0m ${SPACY_MODEL} {v} installed and verified')
" || {
        _msg_error "spaCy post-install check failed" "Wrong version loaded after install" \
            "Entity extraction will silently use wrong model" "rm -rf .venv && bash setup.sh"
        exit 1
    }
}
