#!/usr/bin/env bash
# scripts/validate_gpu.sh
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/validate_gpu.sh
# Responsibility: GPU hardware policy — detection, comparison, smoke tests.
# Sourced by setup.sh — defines functions only, no top-level execution.

_check_nvidia_smi_present() {
    if ! command -v nvidia-smi &>/dev/null; then
        local msg="nvidia-smi not found — not a GPU node"
        echo -e "  ${C_RED}✗${C_RESET} $msg"; echo "$msg"; return 1
    fi
    _msg_ok "nvidia-smi: present"
}

_check_gpu_count_smi() {
    local count
    count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | xargs)
    if [ "${count:-0}" -lt "$TARGET_GPU_COUNT" ]; then
        local msg="GPU count (nvidia-smi): detected ${count}, need ${TARGET_GPU_COUNT}"
        echo -e "  ${C_RED}✗${C_RESET} $msg"; echo "$msg"; return 1
    fi
    _msg_ok "nvidia-smi GPU count: ${count} >= ${TARGET_GPU_COUNT}"
}

_check_gpu_name_smi() {
    local names
    names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')
    if ! echo "$names" | grep -q "$TARGET_GPU_NAME"; then
        local msg="GPU name (nvidia-smi): detected '${names}', expected '${TARGET_GPU_NAME}'"
        echo -e "  ${C_RED}✗${C_RESET} $msg"; echo "$msg"; return 1
    fi
    _msg_ok "nvidia-smi GPU name: contains '${TARGET_GPU_NAME}'"
}

_check_driver_cuda_smi() {
    local driver_cuda
    driver_cuda=$(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}')
    if [ "${driver_cuda}" != "${TARGET_DRIVER_CUDA}" ]; then
        _msg_warn "Driver CUDA version mismatch" \
            "Detected driver CUDA ${driver_cuda}, target is ${TARGET_DRIVER_CUDA}" \
            "informational" \
            "Update TARGET_DRIVER_CUDA in scripts/lib.sh if this node is intentionally different"
    else
        _msg_ok "driver CUDA: ${driver_cuda}"
    fi
}

_query_torch_hardware() {
    _require_python
    $PYTHON -c "
import torch, json
result = {
    'cuda_available': torch.cuda.is_available(),
    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'torch_cuda': torch.version.cuda or 'unknown',
    'cudnn': str(torch.backends.cudnn.version()) if torch.cuda.is_available() else 'N/A',
    'gpus': [],
}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        result['gpus'].append({
            'index': i, 'name': props.name,
            'vram_gb': round(props.total_memory / 1e9, 2),
            'compute_capability': list(torch.cuda.get_device_capability(i)),
        })
print(json.dumps(result))
"
}

_parse_detected_hardware() {
    local hw_json="$1"
    if [ -z "$hw_json" ]; then
        _msg_error "Hardware query empty" "_query_torch_hardware produced no JSON" \
            "torch may not be installed or venv is broken" \
            "STEP=sync_dependencies bash setup.sh   then: STEP=detect_hardware bash setup.sh"
        exit 1
    fi
    DETECTED_GPU_COUNT=$(echo "$hw_json" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['gpu_count'])")
    DETECTED_TORCH_CUDA=$(echo "$hw_json" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['torch_cuda'])")
    DETECTED_CUDNN=$(echo "$hw_json" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['cudnn'])")
    DETECTED_GPU_NAME=$(echo "$hw_json" | $PYTHON -c "
import json,sys
d=json.load(sys.stdin)
names=list({g['name'] for g in d['gpus']})
print(', '.join(names) if names else 'N/A')
")
    for var_name in DETECTED_GPU_COUNT DETECTED_TORCH_CUDA DETECTED_CUDNN DETECTED_GPU_NAME; do
        local val="${!var_name}"
        if [ -z "$val" ] || [ "$val" = "UNDETECTED" ]; then
            _msg_error "Hardware parse failure" "${var_name} empty after parse" \
                "Manifest and GPU smoke tests will use incorrect values" \
                "DEBUG=1 bash setup.sh to inspect raw output"
            exit 1
        fi
    done
    if command -v nvidia-smi &>/dev/null; then
        DETECTED_DRIVER_CUDA=$(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}')
        [ -z "$DETECTED_DRIVER_CUDA" ] && DETECTED_DRIVER_CUDA="parse-failed"
    else
        DETECTED_DRIVER_CUDA="nvidia-smi-not-found"
    fi
}

_print_hardware_table() {
    local hw_json="$1"
    _msg_info "Detected hardware vs targets:"
    printf "  %-20s %-30s %s\n" "Property" "Detected" "Target"
    printf "  %-20s %-30s %s\n" "--------" "--------" "------"
    printf "  %-20s %-30s %s\n" "GPU name"    "'${DETECTED_GPU_NAME}'"  "contains '${TARGET_GPU_NAME}'"
    printf "  %-20s %-30s %s\n" "GPU count"   "${DETECTED_GPU_COUNT}"   "${TARGET_GPU_COUNT}"
    printf "  %-20s %-30s %s\n" "torch CUDA"  "${DETECTED_TORCH_CUDA}"  "${TARGET_TORCH_CUDA_RUNTIME}"
    printf "  %-20s %-30s %s\n" "driver CUDA" "${DETECTED_DRIVER_CUDA}" "${TARGET_DRIVER_CUDA}"
    printf "  %-20s %-30s\n"    "cuDNN"        "${DETECTED_CUDNN}"
    echo "$hw_json" | $PYTHON -c "
import json,sys
for g in json.load(sys.stdin)['gpus']:
    print(f\"  GPU[{g['index']}]: {g['name']} | {g['vram_gb']}GB | cap {g['compute_capability']}\")
"
}

_compare_hardware_to_targets() {
    if [ "${TARGET_GPU_NAME}" != "" ] && ! echo "${DETECTED_GPU_NAME}" | grep -q "${TARGET_GPU_NAME}"; then
        _msg_warn "GPU name mismatch" \
            "Detected '${DETECTED_GPU_NAME}', target '${TARGET_GPU_NAME}'" \
            "action-required" \
            "Wrong cluster node. run_gpu_smoke_tests() will hard-fail with details."
        HARDWARE_MATCH="false"
    fi
    if [ "${DETECTED_GPU_COUNT}" != "${TARGET_GPU_COUNT}" ]; then
        _msg_warn "GPU count mismatch" \
            "Detected ${DETECTED_GPU_COUNT}, target ${TARGET_GPU_COUNT}" \
            "action-required" \
            "Check: echo \$CUDA_VISIBLE_DEVICES   or request ${TARGET_GPU_COUNT}x GPU allocation"
        HARDWARE_MATCH="false"
    fi
    if ! echo "${DETECTED_TORCH_CUDA}" | grep -q "^${TARGET_TORCH_CUDA_RUNTIME}"; then
        _msg_warn "torch CUDA runtime mismatch" \
            "Detected ${DETECTED_TORCH_CUDA}, target ${TARGET_TORCH_CUDA_RUNTIME} (cu117 wheel)" \
            "action-required" \
            "Wrong torch wheel. Fix: rm -rf .venv && bash setup.sh"
        HARDWARE_MATCH="false"
    fi
    if [ "$HARDWARE_MATCH" = "true" ]; then
        _msg_ok "Hardware detection complete — all values match targets."
    else
        _msg_warn "Hardware mismatches detected" \
            "One or more DETECTED_* values do not match TARGET_* constants" \
            "action-required" \
            "See warnings above. run_gpu_smoke_tests() will hard-fail."
    fi
}

_assert_cuda_available() {
    _require_python
    $PYTHON -c "
import torch, sys
if not torch.cuda.is_available():
    print('\033[0;31m  ✗ ERROR — CUDA not available\033[0m')
    print('    What:  torch.cuda.is_available() returned False')
    print('    Why:   Without CUDA, all GPU training will fail at runtime')
    print('    Fix:   Verify torch wheel: .venv/bin/python -c \"import torch; print(torch.__version__)\"')
    print('           Expected 2.0.1+cu117. If CPU wheel: rm -rf .venv && bash setup.sh')
    sys.exit(1)
actual = torch.version.cuda or 'unknown'
if not actual.startswith('${TARGET_TORCH_CUDA_RUNTIME}'):
    print(f'\033[0;31m  ✗ ERROR — torch CUDA runtime mismatch\033[0m')
    print(f'    What:  torch.version.cuda={actual!r}, expected starts with ${TARGET_TORCH_CUDA_RUNTIME}')
    print(f'    Why:   cu117 wheel required for this cluster driver')
    print(f'    Fix:   rm -rf .venv && bash setup.sh')
    sys.exit(1)
print(f'  \033[0;32m✓\033[0m CUDA available | torch runtime {actual}')
"
}

_assert_gpu_count() {
    _require_python
    $PYTHON -c "
import torch, sys
n = torch.cuda.device_count()
if n < ${TARGET_GPU_COUNT}:
    print(f'\033[0;31m  ✗ ERROR — Insufficient GPU count\033[0m')
    print(f'    What:  torch sees {n} GPU(s), need ${TARGET_GPU_COUNT}x ${TARGET_GPU_NAME}')
    print(f'    Why:   Multi-GPU training configured for ${TARGET_GPU_COUNT} GPUs')
    print(f'    Fix:   Check CUDA_VISIBLE_DEVICES or request ${TARGET_GPU_COUNT}x GPU allocation')
    sys.exit(1)
print(f'  \033[0;32m✓\033[0m GPU count: {n} >= ${TARGET_GPU_COUNT}')
"
}

_assert_per_gpu_specs() {
    _require_python
    $PYTHON -c "
import torch, sys
TARGET_GPU_NAME    = '${TARGET_GPU_NAME}'
TARGET_CAP         = (${TARGET_COMPUTE_CAP_MAJOR}, ${TARGET_COMPUTE_CAP_MINOR})
TARGET_VRAM_GB_MIN = ${TARGET_VRAM_GB_MIN}
failed = False
for i in range(torch.cuda.device_count()):
    name    = torch.cuda.get_device_name(i)
    cap     = torch.cuda.get_device_capability(i)
    vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
    if TARGET_GPU_NAME not in name:
        print(f'\033[0;31m  ✗ ERROR — GPU[{i}] wrong hardware: {name}\033[0m')
        print(f'    Fix: Request {TARGET_GPU_COUNT}x NVIDIA {TARGET_GPU_NAME}')
        failed = True
    elif cap < TARGET_CAP:
        print(f'\033[0;31m  ✗ ERROR — GPU[{i}] compute cap {cap} < {TARGET_CAP}\033[0m')
        print(f'    Fix: Request NVIDIA {TARGET_GPU_NAME} (compute cap {TARGET_CAP})')
        failed = True
    elif vram_gb < TARGET_VRAM_GB_MIN:
        print(f'\033[0;31m  ✗ ERROR — GPU[{i}] {vram_gb:.1f}GB < {TARGET_VRAM_GB_MIN}GB\033[0m')
        print(f'    Fix: Request {TARGET_GPU_COUNT}x NVIDIA {TARGET_GPU_NAME}')
        failed = True
    else:
        print(f'  \033[0;32m✓\033[0m GPU[{i}] {name} | cap {cap} | {vram_gb:.1f}GB')
if failed: sys.exit(1)
"
}

_assert_gpu_tensor_op() {
    _require_python
    $PYTHON -c "
import torch, sys
try:
    t = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
    assert torch.allclose(t.mean().cpu(), torch.tensor(2.0))
    print(f'  \033[0;32m✓\033[0m CUDA ${TARGET_TORCH_CUDA_RUNTIME} — tensor round-trip on cuda:0 ok')
except Exception as e:
    print(f'\033[0;31m  ✗ ERROR — GPU functional test failed: {e}\033[0m')
    print(f'    Fix: nvidia-smi to check GPU health   |   STEP=run_gpu_smoke_tests bash setup.sh')
    sys.exit(1)
"
}

log_gpu() {
    _msg_info "Hardware target: ${TARGET_GPU_COUNT}x NVIDIA ${TARGET_GPU_NAME} | CUDA runtime ${TARGET_TORCH_CUDA_RUNTIME} | driver CUDA ${TARGET_DRIVER_CUDA}"
    _msg_info "torch wheel 2.0.1+cu117 compiled against CUDA ${TARGET_TORCH_CUDA_RUNTIME} — driver CUDA ${TARGET_DRIVER_CUDA} is forward-compatible (expected)"
    if command -v nvidia-smi &>/dev/null; then
        echo " --- nvidia-smi per-GPU summary (pre-venv) ---"
        nvidia-smi --query-gpu=index,name,memory.total,driver_version \
            --format=csv,noheader | while IFS=',' read -r idx name mem drv; do
            echo "  GPU $idx:$(echo "$name"|xargs) | VRAM:$(echo "$mem"|xargs) | Driver:$(echo "$drv"|xargs)"
        done
    else
        _msg_warn "nvidia-smi not found" "Cannot log pre-venv GPU details" \
            "informational" "Check module loads if this is a GPU node"
    fi
    command -v nvcc &>/dev/null && \
        _msg_ok "CUDA toolkit (nvcc): $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')" || \
        _msg_warn "nvcc not on PATH" "Toolkit version unverifiable" "informational" "Not required for runtime"
}

detect_hardware() {
    _require_python
    echo " Detecting hardware (post-venv, torch-level)..."
    local hw_json; hw_json=$(_query_torch_hardware)
    _parse_detected_hardware "$hw_json"
    _print_hardware_table "$hw_json"
    _compare_hardware_to_targets
    [ "$HARDWARE_MATCH" = "false" ] && step_end "detect_hardware" "WARN" && return
}

run_gpu_smoke_tests() {
    if [ "${SKIP_GPU:-0}" = "1" ]; then
        _msg_skip "SKIP_GPU=1 — GPU smoke tests skipped (CPU-only mode)"
        step_end "run_gpu_smoke_tests" "SKIP"; return
    fi
    _require_python; _require_hardware_detected
    echo " Running GPU smoke tests — enforcing TARGET_* constraints..."
    [ "$HARDWARE_MATCH" = "false" ] && _msg_warn "Hardware mismatches were flagged" \
        "DETECTED_* values do not match TARGET_* constants" "action-required" \
        "Assertions below will hard-fail with specific details."
    _assert_cuda_available
    _assert_gpu_count
    _assert_per_gpu_specs
    _assert_gpu_tensor_op
}
