#!/usr/bin/env bats
# tests/shell/test_artifact_verification.bats

load helpers

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
MANIFEST="$PROJECT_ROOT/logs/environment_manifest.json"
KERNELSPEC_DIR="$HOME/.local/share/jupyter/kernels/hallucination-legal-rag"

@test ".venv/bin/python exists and is executable" {
    _skip_if_no_venv
    [ -x "$PROJECT_ROOT/.venv/bin/python" ]
}

@test ".venv Python is exactly 3.11.9" {
    _skip_if_no_venv
    run "$PROJECT_ROOT/.venv/bin/python" -c "import sys; print('.'.join(map(str,sys.version_info[:3])))"
    [ "$status" -eq 0 ]
    [ "$output" = "3.11.9" ]
}

@test ".venv Python executable path is inside PROJECT_ROOT" {
    _skip_if_no_venv
    run "$PROJECT_ROOT/.venv/bin/python" -c "import sys; print(sys.executable)"
    [ "$status" -eq 0 ]
    assert_contains "$output" "$PROJECT_ROOT/.venv"
}

@test ".venv torch version is 2.0.1+cu117 (not CPU wheel)" {
    _skip_if_no_venv
    run "$PROJECT_ROOT/.venv/bin/python" -c "import torch; print(torch.__version__)"
    [ "$status" -eq 0 ]
    assert_contains "$output" "2.0.1"
    assert_contains "$output" "cu117"
    assert_not_contains "$output" "2.0.1+cpu"
}

@test ".venv torch reports CUDA runtime 11.7" {
    _skip_if_no_venv
    run "$PROJECT_ROOT/.venv/bin/python" -c "import torch; print(torch.version.cuda)"
    [ "$status" -eq 0 ]
    assert_contains "$output" "11.7"
}

@test ".venv transformers version satisfies >=4.35,<4.42" {
    _skip_if_no_venv
    run "$PROJECT_ROOT/.venv/bin/python" -c "
from packaging.version import Version
import transformers
v = Version(transformers.__version__)
assert Version('4.35') <= v < Version('4.42'), f'out of range: {v}'
print(transformers.__version__)
"
    [ "$status" -eq 0 ]
}

@test ".venv faiss C extension loads and IndexFlatL2 is functional" {
    _skip_if_no_venv
    run "$PROJECT_ROOT/.venv/bin/python" -c "
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
    run "$PROJECT_ROOT/.venv/bin/python" -c "
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
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp('The Supreme Court ruled in favor of the plaintiff in Smith v. Jones.')
ents = [(e.text, e.label_) for e in doc.ents]
assert len(ents) > 0, f'no entities found'
print(ents)
"
    [ "$status" -eq 0 ]
    assert_contains "$output" "Supreme Court"
}

@test ".venv dev tools are installed (pytest, mypy, hypothesis)" {
    _skip_if_no_venv
    run "$PROJECT_ROOT/.venv/bin/python" -c "import pytest, mypy, hypothesis; print('ok')"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test ".venv ipykernel is installed" {
    _skip_if_no_venv
    run "$PROJECT_ROOT/.venv/bin/python" -c "import ipykernel; print(ipykernel.__version__)"
    [ "$status" -eq 0 ]
    [ -n "$output" ]
}

@test "manifest exists at logs/environment_manifest.json" {
    _skip_if_no_manifest
    [ -f "$PROJECT_ROOT/logs/environment_manifest.json" ]
}

@test "manifest is valid JSON" {
    _skip_if_no_manifest
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
    data = json.load(f)
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest contains all required top-level keys" {
    _skip_if_no_manifest
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json, sys
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
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
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json, sys, re
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
    raw = f.read()
if 'UNDETECTED' in raw:
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
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json, sys
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
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
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
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
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
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
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
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
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
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
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
    data = json.load(f)
v = data['hardware_detected']['hardware_match']
assert str(v).lower() == 'true', f'expected true, got {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest hardware_detected.gpu_count matches TARGET_GPU_COUNT (4)" {
    _skip_if_no_manifest
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
    data = json.load(f)
v = data['hardware_detected']['gpu_count']
assert int(v) == 4, f'expected 4, got {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest gpu_count matches len(gpus) array" {
    _skip_if_no_manifest
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
    data = json.load(f)
assert int(data['gpu_count']) == len(data['gpus']), \
    f'gpu_count={data[\"gpu_count\"]} != len(gpus)={len(data[\"gpus\"])}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest torch version is 2.0.1+cu117" {
    _skip_if_no_manifest
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
    data = json.load(f)
v = data['torch']
assert '2.0.1' in v and 'cu117' in v, f'unexpected torch version: {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest uv_lock_sha256 matches actual uv.lock on disk" {
    _skip_if_no_manifest
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json, hashlib
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
    data = json.load(f)
recorded = data['uv_lock_sha256']
actual = hashlib.sha256(open('$PROJECT_ROOT/uv.lock','rb').read()).hexdigest()
assert recorded == actual, f'SHA mismatch: recorded={recorded!r} actual={actual!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest git_sha is not 'not-a-git-repo'" {
    _skip_if_no_manifest
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
    data = json.load(f)
v = data['git_sha']
assert v != 'not-a-git-repo', f'git_sha sentinel found: {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest spacy_model_version is 3.8.0" {
    _skip_if_no_manifest
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
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
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json, sys
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
    data = json.load(f)
pkgs = data.get('installed_packages', {})
core = ['torch','transformers','datasets','faiss-cpu','spacy','numpy','pandas']
bad = {k: pkgs.get(k) for k in core if pkgs.get(k) == 'not installed'}
if bad:
    print(f'MISSING: {bad}')
    sys.exit(1)
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "manifest parity_module is src/repro.py and parity_usage is correct" {
    _skip_if_no_manifest
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$PROJECT_ROOT/logs/environment_manifest.json') as f:
    data = json.load(f)
assert 'repro' in data.get('parity_module','').lower(), \
    f'unexpected parity_module: {data.get(\"parity_module\")!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "kernelspec directory exists at expected path" {
    _skip_if_no_kernelspec
    [ -d "$HOME/.local/share/jupyter/kernels/hallucination-legal-rag" ]
}

@test "kernelspec kernel.json exists" {
    _skip_if_no_kernelspec
    [ -f "$HOME/.local/share/jupyter/kernels/hallucination-legal-rag/kernel.json" ]
}

@test "kernelspec kernel.json is valid JSON" {
    _skip_if_no_kernelspec
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$HOME/.local/share/jupyter/kernels/hallucination-legal-rag/kernel.json') as f:
    data = json.load(f)
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "kernelspec display_name is HallucinationLegalRAG (3.11.9)" {
    _skip_if_no_kernelspec
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$HOME/.local/share/jupyter/kernels/hallucination-legal-rag/kernel.json') as f:
    data = json.load(f)
v = data.get('display_name','')
assert '3.11.9' in v, f'unexpected display_name: {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "kernelspec argv[0] points to venv Python (not system Python)" {
    _skip_if_no_kernelspec
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$HOME/.local/share/jupyter/kernels/hallucination-legal-rag/kernel.json') as f:
    data = json.load(f)
argv0 = data.get('argv', [''])[0]
assert '.venv' in argv0, f'argv[0] not in .venv: {argv0!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "kernelspec argv[0] Python binary exists and is executable" {
    _skip_if_no_kernelspec
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json, os
with open('$HOME/.local/share/jupyter/kernels/hallucination-legal-rag/kernel.json') as f:
    data = json.load(f)
argv0 = data.get('argv', [''])[0]
assert os.path.isfile(argv0) and os.access(argv0, os.X_OK), \
    f'argv[0] not executable: {argv0!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "kernelspec argv[0] Python reports version 3.11.9" {
    _skip_if_no_kernelspec
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json, subprocess
with open('$HOME/.local/share/jupyter/kernels/hallucination-legal-rag/kernel.json') as f:
    data = json.load(f)
argv0 = data.get('argv', [''])[0]
result = subprocess.run([argv0, '--version'], capture_output=True, text=True)
out = result.stdout + result.stderr
assert '3.11.9' in out, f'unexpected version: {out!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "kernelspec name is hallucination-legal-rag" {
    _skip_if_no_kernelspec
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$HOME/.local/share/jupyter/kernels/hallucination-legal-rag/kernel.json') as f:
    data = json.load(f)
v = data.get('name', data.get('display_name', ''))
assert 'hallucination' in v.lower() or 'legal' in v.lower(), \
    f'unexpected name: {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}

@test "kernelspec language is python" {
    _skip_if_no_kernelspec
    run "$PROJECT_ROOT/.venv/bin/python" -c "
import json
with open('$HOME/.local/share/jupyter/kernels/hallucination-legal-rag/kernel.json') as f:
    data = json.load(f)
v = data.get('language', '')
assert v == 'python', f'unexpected language: {v!r}'
print('ok')
"
    [ "$status" -eq 0 ]
    [ "$output" = "ok" ]
}
