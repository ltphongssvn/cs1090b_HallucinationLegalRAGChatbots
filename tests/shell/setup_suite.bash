# tests/shell/setup_suite.bash
# bats-core: setup_suite() runs once before all tests in the suite.
setup_suite() {
    export PROJECT_ROOT
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
}
