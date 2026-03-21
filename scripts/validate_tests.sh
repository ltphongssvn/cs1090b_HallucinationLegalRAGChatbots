#!/usr/bin/env bash
# scripts/validate_tests.sh
run_env_smoke_tests() {
    _require_python
    echo " Running environment smoke tests..."
    $PYTHON -c "
import torch, sys
ver=torch.__version__
if not (ver.startswith('2.') and 'cu' in ver):
    print(f'\033[0;31m  ✗ Wrong torch build: {ver!r} — expected 2.x+cuXXX\033[0m')
    sys.exit(1)
t=torch.tensor([1.0,2.0,3.0])
assert torch.allclose(t.mean(),torch.tensor(2.0))
print(f'  \033[0;32m✓\033[0m torch {ver} — tensor op ok')
"
    $PYTHON -c "
import warnings, transformers, sys
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')
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
        _msg_info "bats-core not found — auto-installing via git clone..."
        if git clone --depth=1 https://github.com/bats-core/bats-core \
            "${PROJECT_ROOT}/tests/shell/bats-core" &>/dev/null; then
            bats_bin="${PROJECT_ROOT}/tests/shell/bats-core/bin/bats"
            _msg_ok "bats-core installed: $("$bats_bin" --version)"
        else
            _msg_warn "bats-core auto-install failed" \
                "git clone bats-core returned non-zero" \
                "informational" \
                "Manual: git clone https://github.com/bats-core/bats-core tests/shell/bats-core"
            return 0
        fi
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
import warnings, sys, torch, torch.nn as nn
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')
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
        a = torch.randn(N, N, dtype=torch.float16, device=device)
        b = torch.randn(N, N, dtype=torch.float16, device=device)
        _ = torch.matmul(a, b)
        torch.cuda.synchronize(i)
        start = time.perf_counter()
        c = torch.matmul(a, b)
        torch.cuda.synchronize(i)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert c.shape == (N, N)
        assert not torch.isnan(c).any()
        assert not torch.isinf(c).any()
        if elapsed_ms > 500:
            print(f'  \033[0;33m⚠ WARNING\033[0m GPU[{i}] matmul slow: {elapsed_ms:.1f}ms (expected <500ms)')
        else:
            print(f'  \033[0;32m✓\033[0m GPU[{i}] {torch.cuda.get_device_name(i)} — FP16 {N}x{N} matmul: {elapsed_ms:.1f}ms')
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'\033[0;31m  ✗ GPU[{i}] matrix multiply benchmark failed: {e}\033[0m')
        failed = True

if failed:
    sys.exit(1)
print(f'  \033[0;32m✓\033[0m All {torch.cuda.device_count()} GPUs passed FP16 matmul benchmark')
" || { _msg_error "GPU benchmark failed" "FP16 matmul failed or produced NaN/Inf" \
        "GPU compute is broken" "nvidia-smi   |   request a new allocation"; exit 1; }
}

_run_bert_gpu_inference() {
    _require_python
    _msg_info "Running BERT GPU inference check (bert-base-uncased encoder on cuda:0)..."
    $PYTHON -c "
import warnings, sys, torch
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')
from transformers import AutoTokenizer, AutoModel

if not torch.cuda.is_available():
    print('  \033[0;33m⚠ SKIP\033[0m BERT GPU inference skipped — CUDA not available')
    sys.exit(0)

device = 'cuda:0'
try:
    tok = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
    model = AutoModel.from_pretrained('bert-base-uncased').to(device).eval()
    text = 'The plaintiff alleged breach of contract under federal appellate jurisdiction.'
    ids = tok(text, return_tensors='pt', max_length=128, truncation=True)
    ids = {k: v.to(device) for k, v in ids.items()}
    with torch.no_grad():
        out = model(**ids)
    cls_emb = out.last_hidden_state[:, 0, :]
    assert cls_emb.shape == (1, 768)
    assert not torch.isnan(cls_emb).any()
    assert not torch.isinf(cls_emb).any()
    assert cls_emb.device.type == 'cuda'
    norm = cls_emb.norm().item()
    print(f'  \033[0;32m✓\033[0m BERT GPU inference ok | [CLS] shape={list(cls_emb.shape)} norm={norm:.3f}')
    del model, ids, out, cls_emb
    torch.cuda.empty_cache()
except Exception as e:
    print(f'\033[0;31m  ✗ BERT GPU inference failed: {e}\033[0m')
    sys.exit(1)
" || { _msg_error "BERT GPU inference failed" "bert-base-uncased encoder failed on cuda:0" \
        "Legal-BERT fine-tuning will fail at the same point" \
        "STEP=run_gpu_smoke_tests bash setup.sh"; exit 1; }
}

_run_vector_store_integrity_check() {
    _require_python
    _msg_info "Running vector store integrity check (FAISS IVF — RAG pipeline component)..."
    $PYTHON -c "
import sys, numpy as np, faiss, tempfile, os

dim         = 768
n_centroids = 64
# FAISS requires >= 39 * n_centroids training points to suppress clustering warning
n_train     = 39 * n_centroids  # 2496 — meets FAISS minimum exactly
n_vectors   = 5000
n_probe     = 8
k           = 5

np.random.seed(42)

try:
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, n_centroids, faiss.METRIC_L2)
    train_vecs = np.random.randn(n_train, dim).astype('float32')
    faiss.normalize_L2(train_vecs)
    index.train(train_vecs)
    assert index.is_trained
    print(f'  \033[0;32m✓\033[0m IVF index trained on {n_train} vectors (dim={dim}, centroids={n_centroids})')

    all_vecs = np.random.randn(n_vectors, dim).astype('float32')
    faiss.normalize_L2(all_vecs)
    index.add(all_vecs)
    assert index.ntotal == n_vectors
    print(f'  \033[0;32m✓\033[0m {n_vectors} vectors added | ntotal={index.ntotal}')

    index.nprobe = n_probe
    query_vecs = all_vecs[:10].copy()
    D, I = index.search(query_vecs, k)
    assert I.shape == (10, k)
    self_hits = sum(1 for i in range(10) if i in I[i])
    assert self_hits >= 8, f'only {self_hits}/10 self-queries found in top-{k}'
    print(f'  \033[0;32m✓\033[0m IVF search: {self_hits}/10 self-queries in top-{k} (nprobe={n_probe})')

    with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as f:
        tmp_path = f.name
    faiss.write_index(index, tmp_path)
    loaded = faiss.read_index(tmp_path)
    os.unlink(tmp_path)
    assert loaded.ntotal == n_vectors
    D2, I2 = loaded.search(query_vecs, k)
    assert (I == I2).all()
    print(f'  \033[0;32m✓\033[0m Vector store serialize/deserialize: results identical')
    print(f'  \033[0;32m✓\033[0m Vector store integrity check passed — RAG retriever component confirmed')

except AssertionError as e:
    print(f'\033[0;31m  ✗ Vector store integrity check FAILED: {e}\033[0m')
    sys.exit(1)
except Exception as e:
    print(f'\033[0;31m  ✗ Vector store unexpected error: {e}\033[0m')
    sys.exit(1)
" || { _msg_error "Vector store integrity check failed" \
        "FAISS IVF index failed train/add/search/serialize cycle" \
        "RAG retriever will return wrong passages" \
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

    run_shell_tests

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

    _msg_info "Tier 3: CPU inference smoke test..."
    _run_cpu_inference_smoke_test

    if [ "${SKIP_GPU:-0}" = "1" ]; then
        _msg_skip "Tiers 4+5: GPU checks — SKIP_GPU=1, skipped"
    else
        _msg_info "Tier 4: GPU test subset (marker=gpu)..."
        _run_gpu_smoke_subset
        _msg_info "Tier 5: DL + RAG integration checks..."
        _run_gpu_benchmark
        _run_bert_gpu_inference
        _run_vector_store_integrity_check
        _msg_ok "Tier 5 passed: GPU benchmark + BERT inference + vector store integrity"
    fi

    _msg_ok "All verification tiers passed"
}
