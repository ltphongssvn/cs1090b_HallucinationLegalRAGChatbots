#!/usr/bin/env bats
# tests/shell/test_artifact_verification.bats
# Path: cs1090b_HallucinationLegalRAGChatbots/tests/shell/test_artifact_verification.bats
# Artifact verification tests — deeply verify contents of artifacts written by setup.sh:
#   1. .venv — correct Python version, expected packages importable, no CUDA wheel drift
#   2. environment_manifest.json — required keys present, no sentinel values, valid JSON
#   3. Jupyter kernelspec — correct display name, kernel.json points to venv Python

load helpers

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
MANIFEST="$PROJECT_ROOT/logs/environment_manifest.json"
KERNELSPEC_DIR="$HOME/.local/share/jupyter/kernels/hallucination-legal-rag"

# ===========================================================================
# Skip guard — skip artifact tests if setup has not been run yet
# ===========================================================================
_skip_if_no_venv() {
    [ ! -x "$VENV_PYTHON" ] && skip ".venv not built — run bash setup.sh first"
}

_skip_if_no_manifest() {
    [ ! -f "$MANIFEST" ] && skip "manifest not written — run bash setup.sh first"
}

_skip_if_no_kernelspec() {
    [ ! -d "$KERNELSPEC_DIR" ] && skip "kernelspec not registered — run bash setup.sh first"
}

# ===========================================================================
# 1. .venv artifact verification
# ===========================================================================

@test ".venv/bin/python exists and is executable" {
    _skip_if_no_venv
    [ -x "$VENV_PYTHON" ]
}

@test ".venv Python is exactly 3.11.9" {
    _skip_if_no_venv
    run "$VENV_PYTHON" -c "import sys; print('.'.join(map(str,sys.version_info[:3])))"
    [ "$status" -eq 0 ]
    [ "$output" = "3.11.9" ]
}

@test ".venv Python executable path is inside PROJECT_ROOT" {
    _skip_if_no_venv
    run "$VENV_PYTHON" -c "import sys; print(sys.executable)"
    [ "$status" -eq 0 ]
    assert_contains "$output" "$PROJECT_ROOT/.venv"
}

@test ".venv torch version is 2.0.1+cu117 (not CPU wheel)" {
    _skip_if_no_venv
    run "$VENV_PYTHON" -c "import torch; print(torch.__version__)"
    [ "$status" -eq 0 ]
    assert_contains "$output" "2.0.1"
    assert_contains "$output" "cu117"
    # Explicit guard: CPU-only wheel must not be present
    assert_not_contains "$output" "2.0.1+cpu"
}

@test ".venv torch reports CUDA runtime 11.7" {
    _skip_if_no_venv
    run "$VENV_PYTHON" -c "import torch; print(torch.version.cuda)"
    [ "$status" -eq 0 ]
    assert_contains "$output" "11.7"
}

@test ".venv transformers version satisfies >=4.35,<4.41" {
    _skip_if_no_venv
    run "$VENV_PYTHON" -c "
from packaging.version import Version
import transformers
v = Version(transformers.__version__)
assert Version('4.35') <= v < Version('4.41'), f'out of range: {v}'
print(transformers.__version__)
"
    [ "$status" -eq 0 ]
}

@test ".venv faiss C extension loads and IndexFlatL2 is functional" {
    _skip_if_no_venv
    run "$VENV_PYTHON" -c "
import faiss, numpy as np
idx = faiss.IndexFlatL2(8)
vecs = np.random.rand(5, 8).astype('float32')
idx.add(vecs)
assert idx.ntotal == 5
D, I = idx.search(vecs[:1], 2)
assert I.shape == (1, 2)
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test ".venv spaCy en_core_web_sm is pinned version 3.8.0" {
    _skip_if_no_venv
    run "$VENV_PYTHON" -c "
import spacy
nlp = spacy.load('en_core_web_sm')
v = nlp.meta.get('version')
assert v == '3.8.0', f'expected 3.8.0, got {v}'
print(v)
"
    [ "$status" -eq 0 ]
    [ "$output" = "3.8.0" ]
}

@test ".venv spaCy NER pipeline processes legal text correctly" {
    _skip_if_no_venv
    run "$VENV_PYTHON" -c "
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp('The Supreme Court ruled in favor of the plaintiff in Smith v. Jones.')
ents = [(e.text, e.label_) for e in doc.ents]
assert len(ents) > 0, f'no entities found in legal test sentence — NER pipeline broken'
print(ents)
"
    [ "$status" -eq 0 ]
    assert_contains "$output" "Supreme Court"
}

@test ".venv dev tools are installed (pytest, mypy, hypothesis)" {
    _skip_if_no_venv
    run "$VENV_PYTHON" -c "
import pytest, mypy, hypothesis
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test ".venv ipykernel is installed" {
    _skip_if_no_venv
    run "$VENV_PYTHON" -c "import ipykernel; print(ipykernel.__version__)"
    [ "$status" -eq 0 ]
    [ -n "$output" ]
}

# ===========================================================================
# 2. environment_manifest.json artifact verification
# ===========================================================================

@test "manifest exists at logs/environment_manifest.json" {
    _skip_if_no_manifest
    [ -f "$MANIFEST" ]
}

@test "manifest is valid JSON" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest contains all required top-level keys" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json, sys
with open('$MANIFEST') as f:
    data = json.load(f)
required = [
    'timestamp','git_sha','git_branch','git_dirty_files','uv_lock_sha256',
    'python','platform','repro_env','numerical_stability','parity_module',
    'hardware_target','hardware_detected','torch','torch_cuda_runtime',
    'cuda_available','gpu_count','gpus','transformers','spacy',
    'spacy_model','spacy_model_version','faiss','installed_packages',
]
missing = [k for k in required if k not in data]
if missing:
    print(f'MISSING KEYS: {missing}')
    sys.exit(1)
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest contains no UNDETECTED sentinel values" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json, sys
with open('$MANIFEST') as f:
    raw = f.read()
if 'UNDETECTED' in raw:
    import re
    hits = re.findall(r'\"[^\"]+\":\s*\"UNDETECTED\"', raw)
    print(f'SENTINEL VALUES FOUND: {hits}')
    sys.exit(1)
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest contains no NOT SET sentinel values for repro_env keys" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json, sys
with open('$MANIFEST') as f:
    data = json.load(f)
repro = data.get('repro_env', {})
not_set = {k: v for k, v in repro.items() if v == 'NOT SET'}
if not_set:
    print(f'NOT SET repro_env keys: {list(not_set.keys())}')
    sys.exit(1)
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest repro_env.PYTHONHASHSEED is '0'" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
v = data['repro_env']['PYTHONHASHSEED']
assert v == '0', f'expected 0, got {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest repro_env.RANDOM_SEED is '0'" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
v = data['repro_env']['RANDOM_SEED']
assert v == '0', f'expected 0, got {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest numerical_stability.deterministic_algorithms_enabled is true" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
v = data['numerical_stability']['deterministic_algorithms_enabled']
assert v is True, f'expected True, got {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest numerical_stability.cudnn_benchmark is false" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
v = data['numerical_stability']['cudnn_benchmark']
assert v is False, f'expected False, got {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest hardware_detected.hardware_match is 'true'" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
v = data['hardware_detected']['hardware_match']
assert v == 'true', f'expected true, got {v!r} — hardware mismatch was recorded'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest hardware_detected.gpu_count matches TARGET_GPU_COUNT (4)" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
v = data['hardware_detected']['gpu_count']
assert str(v) == '4', f'expected 4, got {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest gpu_count matches len(gpus) array" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
count = data.get('gpu_count', -1)
gpus  = data.get('gpus', [])
assert count == len(gpus), f'gpu_count={count} but len(gpus)={len(gpus)}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest torch version is 2.0.1+cu117" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
v = data['torch']
assert v == '2.0.1+cu117', f'expected 2.0.1+cu117, got {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest uv_lock_sha256 matches actual uv.lock on disk" {
    _skip_if_no_manifest
    [ ! -f "$PROJECT_ROOT/uv.lock" ] && skip "uv.lock not present"
    run bash -c "
        actual=\$(sha256sum '$PROJECT_ROOT/uv.lock' | cut -d' ' -f1)
        recorded=\$('$VENV_PYTHON' -c \"
import json
with open('$MANIFEST') as f:
    data = json.load(f)
print(data['uv_lock_sha256'])
\")
        if [ \"\$actual\" != \"\$recorded\" ]; then
            echo \"MISMATCH: actual=\$actual recorded=\$recorded\"
            exit 1
        fi
        echo ok
    "
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest git_sha is not 'not-a-git-repo'" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
v = data['git_sha']
assert v != 'not-a-git-repo', 'git SHA is placeholder — git repo not initialized'
assert len(v) == 40, f'expected 40-char SHA, got {len(v)} chars: {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest spacy_model_version is 3.8.0" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
v = data['spacy_model_version']
assert v == '3.8.0', f'expected 3.8.0, got {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest installed_packages has no 'not installed' entries for core packages" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json, sys
with open('$MANIFEST') as f:
    data = json.load(f)
pkgs = data.get('installed_packages', {})
core = ['torch','transformers','datasets','faiss-cpu','spacy','scikit-learn','numpy','pandas']
missing = {k: pkgs.get(k) for k in core if pkgs.get(k) == 'not installed'}
if missing:
    print(f'Core packages missing: {missing}')
    sys.exit(1)
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest parity_module is src/repro.py and parity_usage is correct" {
    _skip_if_no_manifest
    run "$VENV_PYTHON" -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
assert data['parity_module'] == 'src/repro.py', f'got {data[\"parity_module\"]!r}'
assert 'configure' in data['parity_usage'], f'got {data[\"parity_usage\"]!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

# ===========================================================================
# 3. Jupyter kernelspec artifact verification
# ===========================================================================

@test "kernelspec directory exists at expected path" {
    _skip_if_no_kernelspec
    [ -d "$KERNELSPEC_DIR" ]
}

@test "kernelspec kernel.json exists" {
    _skip_if_no_kernelspec
    [ -f "$KERNELSPEC_DIR/kernel.json" ]
}

@test "kernelspec kernel.json is valid JSON" {
    _skip_if_no_kernelspec
    run "$VENV_PYTHON" -c "
import json
with open('$KERNELSPEC_DIR/kernel.json') as f:
    data = json.load(f)
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "kernelspec display_name is HallucinationLegalRAG (3.11.9)" {
    _skip_if_no_kernelspec
    run "$VENV_PYTHON" -c "
import json
with open('$KERNELSPEC_DIR/kernel.json') as f:
    data = json.load(f)
name = data.get('display_name', '')
assert name == 'HallucinationLegalRAG (3.11.9)', f'got {name!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "kernelspec argv[0] points to venv Python (not system Python)" {
    _skip_if_no_kernelspec
    run "$VENV_PYTHON" -c "
import json
with open('$KERNELSPEC_DIR/kernel.json') as f:
    data = json.load(f)
argv = data.get('argv', [])
assert len(argv) > 0, 'argv is empty'
python_path = argv[0]
assert '.venv' in python_path, \
    f'kernelspec argv[0]={python_path!r} does not point to .venv Python — wrong kernel'
print(python_path)
"
    [ "$status" -eq 0 ]
    assert_contains "$output" ".venv"
}

@test "kernelspec argv[0] Python binary exists and is executable" {
    _skip_if_no_kernelspec
    run "$VENV_PYTHON" -c "
import json, os, sys
with open('$KERNELSPEC_DIR/kernel.json') as f:
    data = json.load(f)
python_path = data['argv'][0]
if not os.path.isfile(python_path):
    print(f'NOT FOUND: {python_path}')
    sys.exit(1)
if not os.access(python_path, os.X_OK):
    print(f'NOT EXECUTABLE: {python_path}')
    sys.exit(1)
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "kernelspec argv[0] Python reports version 3.11.9" {
    _skip_if_no_kernelspec
    run "$VENV_PYTHON" -c "
import json, subprocess
with open('$KERNELSPEC_DIR/kernel.json') as f:
    data = json.load(f)
python_path = data['argv'][0]
result = subprocess.run([python_path, '--version'], capture_output=True, text=True)
version_str = result.stdout.strip() or result.stderr.strip()
assert '3.11.9' in version_str, \
    f'kernelspec Python version is {version_str!r}, expected 3.11.9'
print(version_str)
"
    [ "$status" -eq 0 ]
    assert_contains "$output" "3.11.9"
}

@test "kernelspec name is hallucination-legal-rag" {
    _skip_if_no_kernelspec
    # The directory name IS the kernel name
    [ "$(basename "$KERNELSPEC_DIR")" = "hallucination-legal-rag" ]
}

@test "kernelspec language is python" {
    _skip_if_no_kernelspec
    run "$VENV_PYTHON" -c "
import json
with open('$KERNELSPEC_DIR/kernel.json') as f:
    data = json.load(f)
lang = data.get('language', '')
assert lang == 'python', f'expected python, got {lang!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}
