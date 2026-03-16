#!/usr/bin/env bash
# scripts/manifest.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/manifest.sh
# Responsibility: environment manifest — collect data, validate JSON, write to disk.
# Python logic lives in src/manifest_collector.py (lintable, type-checkable, testable).
# Sourced by setup.sh — defines functions only, no top-level execution.
#
# Mutating steps and their DRY_RUN behaviour:
#   write_manifest — would write logs/environment_manifest.json

_write_manifest_file() {
    local manifest_json="$1"
    if [ -z "$manifest_json" ]; then
        _msg_error "Empty manifest JSON" "_write_manifest_file received empty input" \
            "Empty manifest cannot be used for reproducibility auditing" \
            "STEP=write_manifest bash setup.sh to retry"
        exit 1
    fi
    if ! echo "$manifest_json" | $PYTHON -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
        _msg_error "Malformed manifest JSON" "JSON failed to parse" \
            "Corrupt manifest silently records wrong reproducibility state" \
            "DEBUG=1 bash setup.sh and inspect src/manifest_collector.py output"
        echo "  First 300 chars: ${manifest_json:0:300}"
        exit 1
    fi
    mkdir -p "${PROJECT_ROOT}/logs"
    echo "$manifest_json" | $PYTHON -c "
import json,sys
data=json.load(sys.stdin)
with open('logs/environment_manifest.json','w') as f:
    json.dump(data,f,indent=2)
pkg_count = len(data.get('freeze_snapshot', {}))
cpu = data.get('cpu', {})
print(f'  \033[0;32m✓\033[0m manifest → logs/environment_manifest.json')
print(f'    freeze snapshot: {pkg_count} packages')
print(f'    cpu: {cpu.get(\"cpu_model\",\"unknown\")} | cores: {cpu.get(\"logical_cores\",\"?\")} | RAM: {cpu.get(\"ram_total_gb\",\"?\")}GB')
"
}

write_manifest() {
    _require_python; _require_hardware_detected

    if _is_dry_run; then
        local git_sha; git_sha=$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo "not-a-git-repo")
        _msg_dry_run "write environment manifest" "${PROJECT_ROOT}/logs/environment_manifest.json"
        _msg_info "Would record: git=${git_sha} | host=$(hostname 2>/dev/null) | slurm_job=${SLURM_JOB_ID:-none}"
        _msg_info "Would record: hardware_match=${HARDWARE_MATCH} | detected=${DETECTED_GPU_COUNT}x ${DETECTED_GPU_NAME}"
        _msg_info "Would record: CUDA_HOME=${CUDA_HOME:-NOT SET} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NOT SET}"
        step_end "write_manifest" "DRY"; return
    fi

    echo " Writing environment manifest (via src/manifest_collector.py)..."
    local git_sha git_branch git_dirty uvlock_sha256
    git_sha=$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo "not-a-git-repo")
    git_branch=$(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    git_dirty=$(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null | wc -l | xargs)

    [ ! -f "${PROJECT_ROOT}/uv.lock" ] && {
        _msg_error "uv.lock missing for manifest" "Cannot compute sha256" \
            "Manifest will not record exact dependency snapshot" \
            "uv lock && git add uv.lock && git commit -m 'chore: pin uv.lock'"
        exit 1
    }
    uvlock_sha256=$(sha256sum "${PROJECT_ROOT}/uv.lock" | cut -d' ' -f1)

    # Capture uv version
    local uv_version="unknown"
    if [ -n "${UV:-}" ] && "$UV" --version &>/dev/null; then
        uv_version=$("$UV" --version 2>/dev/null || echo "unknown")
    fi

    # Full freeze snapshot
    local freeze_output="unavailable"
    if [ -n "${UV:-}" ] && "$UV" pip list --format=freeze &>/dev/null 2>&1; then
        freeze_output=$("$UV" pip list --format=freeze 2>/dev/null || echo "unavailable")
    elif [ -x "$PYTHON" ]; then
        freeze_output=$("$PYTHON" -m pip list --format=freeze 2>/dev/null || echo "unavailable")
    fi

    # Write freeze to a temp file to avoid heredoc quoting issues with long strings
    local freeze_tmp; freeze_tmp=$(mktemp)
    echo "$freeze_output" > "$freeze_tmp"

    local manifest_json
    manifest_json=$($PYTHON "${PROJECT_ROOT}/src/manifest_collector.py" \
        --git-sha            "$git_sha" \
        --git-branch         "$git_branch" \
        --git-dirty          "$git_dirty" \
        --uvlock-sha256      "$uvlock_sha256" \
        --uv-version         "$uv_version" \
        --hostname           "$(hostname -f 2>/dev/null || hostname)" \
        --slurm-job-id       "${SLURM_JOB_ID:-none}" \
        --slurm-job-name     "${SLURM_JOB_NAME:-none}" \
        --slurm-nodelist     "${SLURM_JOB_NODELIST:-none}" \
        --target-gpu-name       "$TARGET_GPU_NAME" \
        --target-gpu-count      "$TARGET_GPU_COUNT" \
        --target-cap-major      "$TARGET_COMPUTE_CAP_MAJOR" \
        --target-cap-minor      "$TARGET_COMPUTE_CAP_MINOR" \
        --target-vram-gb-min    "$TARGET_VRAM_GB_MIN" \
        --target-torch-cuda     "$TARGET_TORCH_CUDA_RUNTIME" \
        --target-driver-cuda    "$TARGET_DRIVER_CUDA" \
        --target-python-version "$TARGET_PYTHON_VERSION" \
        --target-min-disk-gb    "$TARGET_MIN_DISK_GB" \
        --detected-gpu-name     "$DETECTED_GPU_NAME" \
        --detected-gpu-count    "$DETECTED_GPU_COUNT" \
        --detected-torch-cuda   "$DETECTED_TORCH_CUDA" \
        --detected-driver-cuda  "$DETECTED_DRIVER_CUDA" \
        --detected-cudnn         "$DETECTED_CUDNN" \
        --hardware-match         "$HARDWARE_MATCH" \
        --spacy-model            "$SPACY_MODEL" \
        --spacy-model-sha256     "$SPACY_MODEL_SHA256" \
        --freeze                 "$(cat "$freeze_tmp")"
    ) || {
        rm -f "$freeze_tmp"
        _msg_error "manifest_collector.py failed" \
            "src/manifest_collector.py exited non-zero" \
            "Manifest not written — reproducibility state unrecorded" \
            "DEBUG=1 STEP=write_manifest bash setup.sh"
        exit 1
    }
    rm -f "$freeze_tmp"

    _write_manifest_file "$manifest_json"
    _msg_info "git: ${git_sha} | branch: ${git_branch} | dirty: ${git_dirty}"
    _msg_info "uv: ${uv_version} | host: $(hostname 2>/dev/null) | slurm_job: ${SLURM_JOB_ID:-none}"
    _msg_info "uv.lock sha256: ${uvlock_sha256}"
    _msg_info "CUDA_HOME: ${CUDA_HOME:-NOT SET} | CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-NOT SET}"
    _msg_info "hardware_match: ${HARDWARE_MATCH} | detected: ${DETECTED_GPU_COUNT}x ${DETECTED_GPU_NAME}"
}
