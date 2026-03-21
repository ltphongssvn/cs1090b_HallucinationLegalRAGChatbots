# tests/shell/helpers.bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT

load_lib() {
    source "$PROJECT_ROOT/scripts/lib.sh"
}
load_bootstrap_env() {
    load_lib
    source "$PROJECT_ROOT/scripts/bootstrap_env.sh"
}
make_valid_project_root() {
    local dir="$1"
    mkdir -p "$dir"
    touch "$dir/pyproject.toml"
    touch "$dir/uv.lock"
}
assert_contains() {
    local haystack="$1" needle="$2"
    if [[ "$haystack" != *"$needle"* ]]; then
        echo "ASSERT FAILED: expected output to contain: '$needle'"
        echo "Actual output: '$haystack'"
        return 1
    fi
}
assert_not_contains() {
    local haystack="$1" needle="$2"
    if [[ "$haystack" == *"$needle"* ]]; then
        echo "ASSERT FAILED: expected output NOT to contain: '$needle'"
        echo "Actual output: '$haystack'"
        return 1
    fi
}
_skip_if_no_venv() {
    [ ! -x "$PROJECT_ROOT/.venv/bin/python" ] && skip ".venv not built — run bash setup.sh first"
    return 0
}
_skip_if_no_manifest() {
    [ ! -f "$PROJECT_ROOT/logs/environment_manifest.json" ] && skip "manifest not written — run bash setup.sh first"
    return 0
}
_skip_if_no_kernelspec() {
    [ ! -d "$HOME/.local/share/jupyter/kernels/hallucination-legal-rag" ] && skip "kernelspec not registered — run bash setup.sh first"
    return 0
}
