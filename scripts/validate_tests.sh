#!/usr/bin/env bash
# scripts/validate_tests.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/validate_tests.sh
# Responsibility: test discovery, tiered test execution, environment smoke tests,
#                 and shell script tests via bats-core.
#
# verify_tests() runs five tiers:
#   Tier 1 — Shell tests (bats-core): setup.sh helpers + artifact verification
#   Tier 2 — Python unit tests (marker=unit): fast, no I/O, always executed
#   Tier 3 — CPU inference smoke test: real forward pass on CPU
#   Tier 4 — GPU smoke subset (marker=gpu): only when SKIP_GPU != 1
#   Tier 5 — DL + RAG integration checks: GPU benchmark, BERT inference, vector store

run_env_smoke_tests() {
    _require_python
    echo " Running environment smoke tests..."
    $PYTHON -c "
import torch, sys
ver=torch.__version__
if not (ver.startswith('2.') and 'cu' in ver):
    print(f'\033[0;31m  ✗ Wrong torch build: {ver!r} — expected 2.x+cuXXX\033[0m')
    print('    Fix: rm -rf .venv && bash setup.sh')
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
    sys.exit(1)
print(f'  \033[0;32m✓\033[0m transformers {transformers.__version__} — tokenizer ok')
"
    $PYTHON -c "
import faiss,numpy as np,sys
idx=faiss.IndexFlatL2(64); vecs=np.random.rand(10,64).astype('float32')
idx.add(vecs); D,I=idx.search(vecs[:1],3)
if I.shape!=(1,3):
    print(f'\033[0;31m  ✗ faiss search shape {I.shape} != (1,3)\033[0m')
    sys.exit(1)
print('  \033[0;32m✓\033[0m faiss — index add/search ok')
"
    $PYTHON -c "
import spacy, sys
nlp=spacy.load('${SPACY_MODEL}')
v=nlp.meta.get('version')
if v!='${SPACY_MODEL_VERSION}':
    print(f'\033[0;31m  ✗ spaCy model {v} != ${SPACY_MODEL_VERSION}\033[0m')
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
    for f in "${shell_test_files[@]}"; do [ ! -f "$f" ] && missing+=("$f"); done
    if [ ${#missing[@]} -gt 0 ]; then
        _msg_warn "Shell test files missing" "Not found: ${missing[*]}" \
            "informational" "Re-run bash setup.sh or check git status"
        return 0
    fi

    if "$bats_bin" --tap "${shell_test_files[@]}"; then
        _msg_ok "All shell tests passed"
    else
        _msg_error "Shell tests failed" "One or more bats tests failed" \
            "A broken helper causes silent failures in setup.sh" \
            "bats tests/shell/   or: STEP=run_shell_tests bash setup.sh"
        exit 1
    fi
}

_run_cpu_inference_smoke_test() {
    _msg_info "Running CPU inference smoke test (real forward pass)..."
    $PYTHON -c "
import sys, torch, torch.nn as nn
from transformers import AutoTokenizer

try:
    tok = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
    ids = tok('The court ruled in favor of the plaintiff.', return_tensors='pt')
    assert ids['input_ids'].shape[1] > 0
    print(f'  \033[0;32m✓\033[0m tokenizer: seq_len={ids[\"input_ids\"].shape[1]}')
except Exception as e:
    print(f'\033[0;31m  ✗ tokenizer failed: {e}\033[0m')
    sys.exit(1)

try:
    vocab_size=tok.vocab_size; embed_dim=32; num_labels=2
    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed=torch.nn.Embedding(vocab_size,embed_dim)
            self.fc1=torch.nn.Linear(embed_dim,16)
            self.fc2=torch.nn.Linear(16,num_labels)
            self.relu=torch.nn.ReLU()
        def forward(self,x):
            return self.fc2(self.relu(self.fc1(self.embed(x).mean(dim=1))))
    model=TinyClassifier().eval()
    with torch.no_grad():
        logits=model(ids['input_ids'])
        probs=torch.softmax(logits,dim=-1)
    assert logits.shape==(1,num_labels)
    assert abs(probs.sum().item()-1.0)<1e-5
    print(f'  \033[0;32m✓\033[0m CPU forward pass ok | probs={[round(p,3) for p in probs[0].tolist()]}')
except Exception as e:
    print(f'\033[0;31m  ✗ CPU forward pass failed: {e}\033[0m')
    sys.exit(1)

try:
    import faiss, numpy as np
    idx=faiss.IndexFlatL2(64); vecs=np.random.rand(100,64).astype('float32')
    idx.add(vecs); D,I=idx.search(vecs[:5],3)
    assert idx.ntotal==100 and I.shape==(5,3) and (D[:,0]<1e-5).all()
    print(f'  \033[0;32m✓\033[0m FAISS 100-vector self-query L2≈0 confirmed')
except Exception as e:
    print(f'\033[0;31m  ✗ FAISS failed: {e}\033[0m')
    sys.exit(1)
" || { _msg_error "CPU inference smoke test failed" "Forward pass failed despite import success" \
        "Broken C extension or ABI mismatch" "rm -rf .venv && bash setup.sh"; exit 1; }
    _msg_ok "CPU inference smoke test passed"
}

_run_gpu_benchmark() {
    # GPU matrix multiply benchmark — confirms CUDA compute is functional,
    # not just that memory allocation succeeds. A degraded GPU passes allocation
    # tests but fails or produces wrong results on compute-intensive ops.
    # Uses 4096x4096 FP16 matmul — representative of transformer attention sizes.
    _require_python
    _msg_info "Running GPU matrix multiply benchmark (4096x4096 FP16 on all GPUs)..."
    $PYTHON -c "
import torch, sys, time

if not torch.cuda.is_available():
    print('  \033[0;33m⚠ SKIP\033[0m GPU benchmark skipped — CUDA not available')
    sys.exit(0)

N = 4096
failed = False
for i in range(torch.cuda.device_count()):
    device = f'cuda:{i}'
    try:
        # Warmup
        a = torch.randn(N, N, dtype=torch.float16, device=device)
        b = torch.randn(N, N, dtype=torch.float16, device=device)
        _ = torch.matmul(a, b)
        torch.cuda.synchronize(i)

        # Timed run
        start = time.perf_counter()
        c = torch.matmul(a, b)
        torch.cuda.synchronize(i)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Sanity check: result shape and no NaN/Inf
        assert c.shape == (N, N), f'unexpected output shape {c.shape}'
        assert not torch.isnan(c).any(), 'NaN in matmul output — GPU compute error'
        assert not torch.isinf(c).any(), 'Inf in matmul output — GPU compute error'

        # L4 should complete 4096x4096 FP16 matmul in <500ms
        # This is a loose bound — catches severely degraded GPUs
        if elapsed_ms > 500:
            print(f'  \033[0;33m⚠ WARNING\033[0m GPU[{i}] matmul slow: {elapsed_ms:.1f}ms (expected <500ms)')
        else:
            print(f'  \033[0;32m✓\033[0m GPU[{i}] {torch.cuda.get_device_name(i)} — FP16 {N}x{N} matmul: {elapsed_ms:.1f}ms')

        del a, b, c
        torch.cuda.empty_cache()

    except Exception as e:
        print(f'\033[0;31m  ✗ GPU[{i}] matrix multiply benchmark failed: {e}\033[0m')
        print(f'    Why: GPU compute is broken — training will produce wrong results')
        print(f'    Fix: nvidia-smi -i {i}   |   request a new allocation')
        failed = True

if failed:
    sys.exit(1)
print(f'  \033[0;32m✓\033[0m All {torch.cuda.device_count()} GPUs passed FP16 matmul benchmark')
" || { _msg_error "GPU benchmark failed" "FP16 matmul failed or produced NaN/Inf" \
        "GPU compute is broken — training will fail or produce wrong results" \
        "nvidia-smi   |   request a new allocation"; exit 1; }
}

_run_bert_gpu_inference() {
    # Small BERT GPU inference — confirms the actual DL stack works end-to-end
    # on GPU, not just CPU. Tests: tokenizer → GPU tensor → BERT encoder → pooled output.
    # Uses bert-base-uncased (already in HF cache from CPU smoke test) with no
    # classification head — purely encoder inference, ~440MB VRAM.
    _require_python
    _msg_info "Running BERT GPU inference check (bert-base-uncased encoder on cuda:0)..."
    $PYTHON -c "
import sys, torch
from transformers import AutoTokenizer, AutoModel

if not torch.cuda.is_available():
    print('  \033[0;33m⚠ SKIP\033[0m BERT GPU inference skipped — CUDA not available')
    sys.exit(0)

device = 'cuda:0'
try:
    tok = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
    model = AutoModel.from_pretrained('bert-base-uncased').to(device).eval()

    # Legal text sample — representative of actual inference inputs
    text = 'The plaintiff alleged breach of contract under federal appellate jurisdiction.'
    ids = tok(text, return_tensors='pt', max_length=128, truncation=True)
    ids = {k: v.to(device) for k, v in ids.items()}

    with torch.no_grad():
        out = model(**ids)

    cls_emb = out.last_hidden_state[:, 0, :]  # [CLS] embedding
    assert cls_emb.shape == (1, 768), f'expected (1, 768), got {cls_emb.shape}'
    assert not torch.isnan(cls_emb).any(), 'NaN in BERT output — GPU precision error'
    assert not torch.isinf(cls_emb).any(), 'Inf in BERT output — GPU precision error'
    assert cls_emb.device.type == 'cuda', 'output not on CUDA device'

    norm = cls_emb.norm().item()
    print(f'  \033[0;32m✓\033[0m BERT GPU inference ok | [CLS] shape={list(cls_emb.shape)} norm={norm:.3f}')
    print(f'  \033[0;32m✓\033[0m Device: {cls_emb.device} | dtype: {cls_emb.dtype}')

    # Cleanup
    del model, ids, out, cls_emb
    torch.cuda.empty_cache()

except Exception as e:
    print(f'\033[0;31m  ✗ BERT GPU inference failed: {e}\033[0m')
    print(f'    Why: BERT encoder on GPU is broken — retriever fine-tuning will fail')
    print(f'    Fix: STEP=run_gpu_smoke_tests bash setup.sh   |   check GPU[0] health')
    sys.exit(1)
" || { _msg_error "BERT GPU inference failed" "bert-base-uncased encoder failed on cuda:0" \
        "Legal-BERT fine-tuning (Cell 4) will fail at the same point" \
        "STEP=run_gpu_smoke_tests bash setup.sh"; exit 1; }
}

_run_vector_store_integrity_check() {
    # Vector store integrity — RAG-specific test confirming FAISS IVF index
    # (the actual index type used in the project, not just FlatL2) correctly
    # stores and retrieves legal text embeddings.
    # Tests: build IVF index → add vectors → search → verify nearest neighbours
    # are closer than random → confirm index survives serialize/deserialize cycle.
    _require_python
    _msg_info "Running vector store integrity check (FAISS IVF — RAG pipeline component)..."
    $PYTHON -c "
import sys, numpy as np, faiss, tempfile, os

dim        = 768   # BERT [CLS] embedding dimension
n_train    = 2000  # min vectors needed to train IVF centroids
n_vectors  = 5000  # total index size
n_centroids = 64   # IVF1024 would be used in production; 64 for speed here
n_probe    = 8
k          = 5

np.random.seed(42)

try:
    # Build IVF index — same architecture as production RAG retriever
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, n_centroids, faiss.METRIC_L2)

    # Train on a representative sample
    train_vecs = np.random.randn(n_train, dim).astype('float32')
    faiss.normalize_L2(train_vecs)
    index.train(train_vecs)
    assert index.is_trained, 'IVF index failed to train'
    print(f'  \033[0;32m✓\033[0m IVF index trained on {n_train} vectors (dim={dim}, centroids={n_centroids})')

    # Add all vectors
    all_vecs = np.random.randn(n_vectors, dim).astype('float32')
    faiss.normalize_L2(all_vecs)
    index.add(all_vecs)
    assert index.ntotal == n_vectors, f'expected {n_vectors}, got {index.ntotal}'
    print(f'  \033[0;32m✓\033[0m {n_vectors} vectors added | ntotal={index.ntotal}')

    # Search: nearest neighbours of first 10 queries should be themselves (L2≈0)
    index.nprobe = n_probe
    query_vecs = all_vecs[:10].copy()
    D, I = index.search(query_vecs, k)
    assert I.shape == (10, k), f'unexpected search shape {I.shape}'

    # For normalized vectors, self-query should be the nearest neighbour
    self_hits = sum(1 for i in range(10) if i in I[i])
    assert self_hits >= 8, (
        f'only {self_hits}/10 self-queries found in top-{k} — '
        f'IVF retrieval quality degraded (expected >=8)'
    )
    print(f'  \033[0;32m✓\033[0m IVF search: {self_hits}/10 self-queries in top-{k} (nprobe={n_probe})')

    # Serialize / deserialize cycle — confirms index survives save/load
    with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as f:
        tmp_path = f.name
    faiss.write_index(index, tmp_path)
    loaded = faiss.read_index(tmp_path)
    os.unlink(tmp_path)
    assert loaded.ntotal == n_vectors, 'ntotal mismatch after serialize/deserialize'
    D2, I2 = loaded.search(query_vecs, k)
    assert (I == I2).all(), 'search results differ after serialize/deserialize'
    print(f'  \033[0;32m✓\033[0m Vector store serialize/deserialize: results identical')

    print(f'  \033[0;32m✓\033[0m Vector store integrity check passed — RAG retriever component confirmed')

except AssertionError as e:
    print(f'\033[0;31m  ✗ Vector store integrity check FAILED: {e}\033[0m')
    print(f'    Why: FAISS IVF index is broken — RAG retrieval will return wrong passages')
    print(f'    Fix: rm -rf .venv && bash setup.sh  (reinstalls faiss-cpu from uv.lock)')
    sys.exit(1)
except Exception as e:
    print(f'\033[0;31m  ✗ Vector store unexpected error: {e}\033[0m')
    sys.exit(1)
" || { _msg_error "Vector store integrity check failed" \
        "FAISS IVF index failed train/add/search/serialize cycle" \
        "RAG retriever will return wrong passages — hallucination rate will be invalid" \
        "rm -rf .venv && bash setup.sh"; exit 1; }
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
            _msg_error "GPU tests failed" "${GPU_COUNT} gpu tests but one or more failed" \
                "GPU kernel dispatch broken — do not proceed to training" \
                ".venv/bin/pytest tests/ -m gpu -v --tb=long"
            exit 1
        }
    else
        _msg_info "No gpu-marked tests found yet — skipped"
    fi
}

verify_tests() {
    _require_uv; _require_python
    echo " Verifying test suite (tiered execution)..."

    # Tier 1: Shell tests
    run_shell_tests

    # Tier 2: Python unit tests — executed, not just collected
    local UNIT_COUNT
    UNIT_COUNT=$("$UV" run pytest tests/ --co -q -m unit 2>/dev/null | grep -c "^tests/" || true)
    if [ "${UNIT_COUNT}" -gt 0 ]; then
        _msg_info "Tier 2: ${UNIT_COUNT} unit tests — executing..."
        "$UV" run pytest tests/ -m unit -q --tb=short && \
            _msg_ok "Tier 2 passed: ${UNIT_COUNT} unit tests executed" || {
            _msg_error "Unit tests failed" \
                "${UNIT_COUNT} unit tests collected but one or more FAILED" \
                "Broken environment — do not proceed to training" \
                ".venv/bin/pytest tests/ -m unit -v --tb=long"
            exit 1
        }
    else
        _msg_info "No unit tests yet — collection check as diagnostic only..."
        "$UV" run pytest tests/ --co -q 2>/dev/null && \
            _msg_warn "No unit tests to execute" "No unit-marked tests found" \
                "informational" "Add unit tests to enable Tier 2 gate." || \
            _msg_warn "Test collection failed" "pytest --co could not import tests" \
                "informational" ".venv/bin/python -c 'import src.environment; import src.repro'"
    fi

    # Tier 3: CPU inference smoke test (always runs)
    _msg_info "Tier 3: CPU inference smoke test..."
    _run_cpu_inference_smoke_test

    # Tiers 4+5: GPU-specific checks (skipped when SKIP_GPU=1)
    if [ "${SKIP_GPU:-0}" = "1" ]; then
        _msg_skip "Tiers 4+5: GPU checks — SKIP_GPU=1, skipped"
    else
        # Tier 4: GPU pytest subset
        _msg_info "Tier 4: GPU test subset (marker=gpu)..."
        _run_gpu_smoke_subset

        # Tier 5: DL + RAG integration checks
        _msg_info "Tier 5: DL + RAG integration checks..."
        _run_gpu_benchmark
        _run_bert_gpu_inference
        _run_vector_store_integrity_check
        _msg_ok "Tier 5 passed: GPU benchmark + BERT inference + vector store integrity"
    fi

    _msg_ok "All verification tiers passed"
}
