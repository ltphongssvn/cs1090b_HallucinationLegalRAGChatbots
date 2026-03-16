#!/usr/bin/env bash
# scripts/setup_nlp.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/setup_nlp.sh
# Responsibility: NLP asset management — pinned spaCy model download, checksum, install.
# Sourced by setup.sh — defines functions only, no top-level execution.
#
# Dependency control: spaCy model wheel is installed via `$UV pip install`, NOT
# `python -m pip install`. This keeps the install within uv's awareness so that
# subsequent `uv sync --frozen --check` drift detection does not raise a false
# positive or silently overwrite the model on the next sync.
#
# Mutating steps and their mode behaviour:
#   download_nlp_models:
#     DRY_RUN=1      — preview what would be downloaded/installed, no changes
#     NO_DOWNLOAD=1  — skip download entirely
#     OFFLINE=1      — require cached wheel; hard-fail if not present

download_nlp_models() {
    _require_python
    _require_uv
    local SPACY_CACHE_DIR="${PROJECT_ROOT}/.cache/spacy"
    local SPACY_WHEEL="${SPACY_CACHE_DIR}/${SPACY_MODEL}-${SPACY_MODEL_VERSION}-py3-none-any.whl"
    echo " Installing spaCy ${SPACY_MODEL} ${SPACY_MODEL_VERSION} (pinned)..."

    # NO_DOWNLOAD=1 — skip entirely
    if [ "${NO_DOWNLOAD:-0}" = "1" ]; then
        _msg_skip "NO_DOWNLOAD=1 — skipping spaCy model download/install"
        _msg_info "Assuming ${SPACY_MODEL} ${SPACY_MODEL_VERSION} is already installed."
        _msg_info "If not installed: STEP=download_nlp_models bash setup.sh"
        step_end "download_nlp_models" "SKIP"; return
    fi

    mkdir -p "$SPACY_CACHE_DIR"

    # Idempotent check
    if $PYTHON -c "
import spacy,sys
try:
    nlp=spacy.load('${SPACY_MODEL}')
    sys.exit(0 if nlp.meta.get('version')=='${SPACY_MODEL_VERSION}' else 1)
except OSError: sys.exit(1)
" 2>/dev/null; then
        _msg_ok "${SPACY_MODEL} ${SPACY_MODEL_VERSION} already installed — skipping"
        return
    fi

    # DRY_RUN preview
    if _is_dry_run; then
        if [ -f "$SPACY_WHEEL" ]; then
            _msg_dry_run "install spaCy model from cached wheel (via uv pip install)" "$SPACY_WHEEL"
        else
            [ "${OFFLINE:-0}" = "1" ] && {
                _msg_error "Offline mode: wheel not cached" \
                    "OFFLINE=1 but wheel not found at $SPACY_WHEEL" \
                    "Cannot download in offline mode" \
                    "Pre-cache: mkdir -p .cache/spacy && wget -O $SPACY_WHEEL ${SPACY_MODEL_URL}"
                exit 1
            }
            _msg_dry_run "download spaCy wheel" "${SPACY_MODEL_URL}"
            _msg_dry_run "verify checksum" "sha256=${SPACY_MODEL_SHA256}"
            _msg_dry_run "uv pip install wheel" "$SPACY_WHEEL"
        fi
        step_end "download_nlp_models" "DRY"; return
    fi

    if [ ! -f "$SPACY_WHEEL" ]; then
        if [ "${OFFLINE:-0}" = "1" ]; then
            _msg_error "Offline mode: wheel not cached" \
                "OFFLINE=1 but wheel not found at $SPACY_WHEEL" \
                "Cannot download in offline mode" \
                "Pre-cache: mkdir -p .cache/spacy && wget -O $SPACY_WHEEL ${SPACY_MODEL_URL}   then: OFFLINE=1 bash setup.sh"
            exit 1
        fi
        _msg_info "Downloading ${SPACY_MODEL} ${SPACY_MODEL_VERSION} (cached at $SPACY_WHEEL)..."
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

    # Checksum verification — always, regardless of source
    local ACTUAL_SHA; ACTUAL_SHA=$(sha256sum "$SPACY_WHEEL" | cut -d' ' -f1)
    if [ "$ACTUAL_SHA" != "$SPACY_MODEL_SHA256" ]; then
        _msg_error "spaCy wheel checksum mismatch" \
            "expected=${SPACY_MODEL_SHA256}  actual=${ACTUAL_SHA}" \
            "Corrupted wheel produces wrong NER results, silently degrading RAG quality" \
            "Cached wheel deleted. Re-run: bash setup.sh"
        rm -f "$SPACY_WHEEL"; exit 1
    fi
    _msg_ok "Checksum verified: ${ACTUAL_SHA}"

    # Install via uv pip install, NOT python -m pip install.
    # Reason: python -m pip bypasses uv's awareness of the venv state.
    # A subsequent `uv sync --frozen --check` would flag the spaCy model
    # as an unexpected package, causing drift detection to hard-fail or
    # silently overwrite it. Using uv pip install keeps the install
    # within uv's managed context.
    "$UV" pip install --quiet "$SPACY_WHEEL" || {
        _msg_error "uv pip install failed" \
            "uv pip install $SPACY_WHEEL exited non-zero" \
            "spaCy model not installed — NER pipeline will fail at runtime" \
            "Check uv output above, then: STEP=download_nlp_models bash setup.sh"
        exit 1
    }

    # Post-install verification
    $PYTHON -c "
import spacy, sys
nlp = spacy.load('${SPACY_MODEL}')
v = nlp.meta.get('version')
if v != '${SPACY_MODEL_VERSION}':
    print(f'\033[0;31m  ✗ Post-install version mismatch: {v} != ${SPACY_MODEL_VERSION}\033[0m')
    print('    Fix: rm -rf .venv && bash setup.sh')
    sys.exit(1)
print(f'  \033[0;32m✓\033[0m ${SPACY_MODEL} {v} installed and verified via uv pip install')
" || {
        _msg_error "spaCy post-install check failed" "Wrong version loaded after install" \
            "Entity extraction will silently use wrong model" "rm -rf .venv && bash setup.sh"
        exit 1
    }
}
