# tests/shell/setup_suite.bash
# Ensures PROJECT_ROOT is always the repo root regardless of bats working directory.
setup() {
    export PROJECT_ROOT
    PROJECT_ROOT="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
}
