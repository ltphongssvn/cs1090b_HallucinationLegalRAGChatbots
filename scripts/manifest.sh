#!/usr/bin/env bash
# scripts/manifest.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/manifest.sh
# Responsibility: environment manifest — collect data, validate JSON, write to disk.
# Sourced by setup.sh — defines functions only, no top-level execution.
#
# Mutating steps and their DRY_RUN behaviour:
#   write_manifest — would write logs/environment_manifest.json

_collect_manifest_data() {
    _require_python; _require_hardware_detected
    local git_sha="$1" git_branch="$2" git_dirty="$3" uvlock_sha256="$4"
    for arg_name in git_sha git_branch git_dirty uvlock_sha256; do
        local val="${!arg_name}"
        if [ -z "$val" ]; then
            _msg_warn "Manifest arg '${arg_name}' empty" \
                "_collect_manifest_data received empty '${arg_name}'" \
                "informational" "Manifest will record 'unknown' — non-blocking"
            eval "${arg_name}=unknown"
        fi
    done
    $PYTHON -c "
import json, torch, transformers, spacy, sys, platform, subprocess, os
import importlib.metadata as meta
from datetime import datetime

def _get_nvcc():
    try:
        r=subprocess.run(['nvcc','--version'],capture_output=True,text=True)
        for l in r.stdout.splitlines():
            if 'release' in l: return l.strip()
    except FileNotFoundError: return 'nvcc not found'
    return 'unknown'

def _get_driver_cuda():
    try:
        r=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
        for l in r.stdout.splitlines():
            if 'CUDA Version' in l: return l.strip().split()[-1]
    except FileNotFoundError: return 'nvidia-smi not found'
    return 'unknown'

def _get_driver_version():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                         capture_output=True,text=True)
        return r.stdout.strip().splitlines()[0] if r.stdout.strip() else 'unknown'
    except FileNotFoundError: return 'nvidia-smi not found'

def _get_faiss():
    try:
        import faiss; return getattr(faiss,'__version__','installed — no version attr')
    except ImportError: return 'not installed'

def _get_pkgs(pkgs):
    out={}
    for p in pkgs:
        try: out[p]=meta.version(p)
        except meta.PackageNotFoundError: out[p]='not installed'
    return out

gpus=[]
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props=torch.cuda.get_device_properties(i)
        gpus.append({'index':i,'name':props.name,
                     'vram_gb':round(props.total_memory/1e9,2),
                     'compute_capability':list(torch.cuda.get_device_capability(i))})

nlp=spacy.load('${SPACY_MODEL}')
data={
    'timestamp':      datetime.utcnow().isoformat()+'Z',
    'git_sha':        '${git_sha}',
    'git_branch':     '${git_branch}',
    'git_dirty_files': int('${git_dirty}') if '${git_dirty}'.isdigit() else -1,
    'uv_lock_sha256': '${uvlock_sha256}',
    'python':         sys.version,
    'platform':       platform.platform(),
    'repro_env':{
        'PYTHONHASHSEED':         os.environ.get('PYTHONHASHSEED','NOT SET'),
        'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG','NOT SET'),
        'TOKENIZERS_PARALLELISM': os.environ.get('TOKENIZERS_PARALLELISM','NOT SET'),
        'RANDOM_SEED':            os.environ.get('RANDOM_SEED','NOT SET'),
    },
    'numerical_stability':{
        'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark':                 torch.backends.cudnn.benchmark,
        'cudnn_deterministic':             torch.backends.cudnn.deterministic,
    },
    'parity_module':'src/repro.py',
    'parity_usage': 'from src.repro import configure; configure()',
    'hardware_target':{
        'gpu_name':          '${TARGET_GPU_NAME}',
        'gpu_count':          ${TARGET_GPU_COUNT},
        'compute_cap_min':   [${TARGET_COMPUTE_CAP_MAJOR},${TARGET_COMPUTE_CAP_MINOR}],
        'vram_gb_min':        ${TARGET_VRAM_GB_MIN},
        'torch_cuda_runtime':'${TARGET_TORCH_CUDA_RUNTIME}',
        'driver_cuda':       '${TARGET_DRIVER_CUDA}',
        'python_version':    '${TARGET_PYTHON_VERSION}',
        'min_disk_gb':        ${TARGET_MIN_DISK_GB},
    },
    'hardware_detected':{
        'gpu_name':          '${DETECTED_GPU_NAME}',
        'gpu_count':         '${DETECTED_GPU_COUNT}',
        'torch_cuda_runtime':'${DETECTED_TORCH_CUDA}',
        'driver_cuda':       '${DETECTED_DRIVER_CUDA}',
        'cudnn':             '${DETECTED_CUDNN}',
        'hardware_match':    '${HARDWARE_MATCH}',
    },
    'torch':              torch.__version__,
    'torch_cuda_runtime': torch.version.cuda,
    'driver_cuda':        _get_driver_cuda(),
    'driver_version':     _get_driver_version(),
    'cudnn':              str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
    'cuda_toolkit_nvcc':  _get_nvcc(),
    'cuda_available':     torch.cuda.is_available(),
    'gpu_count':          torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'gpus':               gpus,
    'transformers':       transformers.__version__,
    'spacy':              spacy.__version__,
    'spacy_model':        '${SPACY_MODEL}',
    'spacy_model_version':nlp.meta.get('version'),
    'spacy_model_sha256': '${SPACY_MODEL_SHA256}',
    'faiss':              _get_faiss(),
    'installed_packages': _get_pkgs([
        'torch','transformers','datasets','faiss-cpu','spacy',
        'scikit-learn','numpy','pandas','langchain','gensim',
        'sentence-transformers','networkx','pytest','mypy','hypothesis',
    ]),
}
print(json.dumps(data))
"
}

_write_manifest_file() {
    local manifest_json="$1"
    if [ -z "$manifest_json" ]; then
        _msg_error "Empty manifest JSON" "_write_manifest_file received empty input" \
            "Empty manifest cannot be used for reproducibility auditing" \
            "STEP=write_manifest bash setup.sh to retry"
        exit 1
    fi
    if ! echo "$manifest_json" | $PYTHON -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
        _msg_error "Malformed manifest JSON" "JSON failed to parse — bash interpolation issue" \
            "Corrupt manifest silently records wrong reproducibility state" \
            "DEBUG=1 bash setup.sh and inspect _collect_manifest_data output"
        echo "  First 300 chars: ${manifest_json:0:300}"
        exit 1
    fi
    mkdir -p "${PROJECT_ROOT}/logs"
    echo "$manifest_json" | $PYTHON -c "
import json,sys
data=json.load(sys.stdin)
with open('logs/environment_manifest.json','w') as f:
    json.dump(data,f,indent=2)
print('  \033[0;32m✓\033[0m manifest → logs/environment_manifest.json')
"
}

write_manifest() {
    _require_python; _require_hardware_detected

    if _is_dry_run; then
        local git_sha; git_sha=$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo "not-a-git-repo")
        _msg_dry_run "write environment manifest" "${PROJECT_ROOT}/logs/environment_manifest.json"
        _msg_info "Would record: git=${git_sha} | hardware_match=${HARDWARE_MATCH} | detected=${DETECTED_GPU_COUNT}x ${DETECTED_GPU_NAME}"
        step_end "write_manifest" "DRY"; return
    fi

    echo " Writing environment manifest..."
    local git_sha git_branch git_dirty uvlock_sha256 manifest_json
    git_sha=$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo "not-a-git-repo")
    git_branch=$(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    git_dirty=$(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null | wc -l | xargs)
    [ ! -f "${PROJECT_ROOT}/uv.lock" ] && {
        _msg_error "uv.lock missing for manifest" "Cannot compute sha256 — file not found" \
            "Manifest will not record exact dependency snapshot" \
            "uv lock && git add uv.lock && git commit -m 'chore: pin uv.lock'"
        exit 1
    }
    uvlock_sha256=$(sha256sum "${PROJECT_ROOT}/uv.lock" | cut -d' ' -f1)
    manifest_json=$(_collect_manifest_data "$git_sha" "$git_branch" "$git_dirty" "$uvlock_sha256")
    _write_manifest_file "$manifest_json"
    _msg_info "git: ${git_sha} | branch: ${git_branch} | dirty: ${git_dirty}"
    _msg_info "uv.lock sha256: ${uvlock_sha256}"
    _msg_info "hardware_match: ${HARDWARE_MATCH} | detected: ${DETECTED_GPU_COUNT}x ${DETECTED_GPU_NAME}"
}
