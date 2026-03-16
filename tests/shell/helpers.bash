# tests/shell/helpers.bash
# Path: cs1090b_HallucinationLegalRAGChatbots/tests/shell/helpers.bash
# Shared test helpers — loaded by every .bats file via: load helpers

# Resolve project root from this file's location (tests/shell/ → project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT

# Source lib.sh into the test environment so helper functions are available.
# Colors are suppressed (no TTY in test runner) — lib.sh handles this automatically.
load_lib() {
    source "$PROJECT_ROOT/scripts/lib.sh"
}

load_bootstrap_env() {
    load_lib
    source "$PROJECT_ROOT/scripts/bootstrap_env.sh"
}

# Create a minimal temp project root with pyproject.toml and uv.lock
make_valid_project_root() {
    local dir="$1"
    mkdir -p "$dir"
    touch "$dir/pyproject.toml"
    touch "$dir/uv.lock"
}

# Assert output contains a substring
assert_contains() {
    local haystack="$1" needle="$2"
    if [[ "$haystack" != *"$needle"* ]]; then
        echo "ASSERT FAILED: expected output to contain: '$needle'"
        echo "Actual output: '$haystack'"
        return 1
    fi
}

# Assert output does NOT contain a substring
assert_not_contains() {
    local haystack="$1" needle="$2"
    if [[ "$haystack" == *"$needle"* ]]; then
        echo "ASSERT FAILED: expected output NOT to contain: '$needle'"
        echo "Actual output: '$haystack'"
        return 1
    fi
}
