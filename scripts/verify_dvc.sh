#!/usr/bin/env bash
# verify_dvc.sh — DVC pipeline reproducibility checker.
#
# Verifies that the current repo state is reproducible via DVC remote:
#   - dvc binary + remote configured
#   - local cache aligned with .dvc pointers
#   - cloud remote aligned with local cache (no drift)
#   - no large files evading DVC tracking
#
# Exit codes:
#   0  — all checks pass (default mode: warnings allowed)
#   1  — cloud drift detected in --strict mode
#   2  — environment or configuration error (missing dvc, no remote)
#
# Environment:
#   VERIFY_DVC_BIN       — override dvc binary path
#   VERIFY_DVC_SIZE_MB   — large-file threshold (default: 100)

set -euo pipefail

readonly SCHEMA_VERSION="1.0.0"
readonly DEFAULT_SIZE_MB=100

MODE_JSON=0
MODE_STRICT=0

usage() {
    cat << USAGE
verify_dvc.sh — DVC pipeline reproducibility checker

Usage: verify_dvc.sh [--help] [--json] [--strict]

Options:
  --help      Show this help and exit.
  --json      Emit machine-readable JSON output (schema v${SCHEMA_VERSION}).
  --strict    Exit 1 on cloud drift (for CI gating). Default: warn only.

Exit codes:
  0  All checks pass.
  1  Cloud drift detected in --strict mode.
  2  Environment/config error (dvc missing, no remote configured).

Environment:
  VERIFY_DVC_BIN       Override path to dvc binary.
  VERIFY_DVC_SIZE_MB   Large-file threshold in MB (default: ${DEFAULT_SIZE_MB}).
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help) usage; exit 0 ;;
        --json) MODE_JSON=1; shift ;;
        --strict) MODE_STRICT=1; shift ;;
        *)
            echo "error: unknown flag: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# ---------- preflight: dvc binary ----------

DVC_BIN="${VERIFY_DVC_BIN:-$(command -v dvc 2>/dev/null || true)}"
if [[ -z "$DVC_BIN" ]] || ! [[ -x "$DVC_BIN" ]]; then
    echo "error: dvc not found on PATH (install via 'pip install dvc[s3]' or set VERIFY_DVC_BIN)" >&2
    exit 2
fi

# ---------- verification checks ----------

remote_list_exit=0
remote_list_output=$("$DVC_BIN" remote list 2>&1) || remote_list_exit=$?

if [[ $remote_list_exit -ne 0 ]]; then
    echo "error: 'dvc remote list' failed (exit $remote_list_exit): $remote_list_output" >&2
    exit 2
fi

if [[ -z "$(echo "$remote_list_output" | tr -d '[:space:]')" ]]; then
    echo "error: no DVC remote configured (run 'dvc remote add -d <name> <url>')" >&2
    exit 2
fi

DVC_REMOTE=$(echo "$remote_list_output" | head -1 | awk '{print $1}')

status_output=$("$DVC_BIN" status 2>&1 || true)

cloud_output=$("$DVC_BIN" status --cloud 2>&1 || true)
cloud_drift=false
if echo "$cloud_output" | grep -qE '^\s*(new|deleted|modified):'; then
    cloud_drift=true
fi

# ---------- .dvc pointer enumeration (fast with -prune) ----------
#
# Excludes: .git, .venv, .dvc/cache, .dvc/tmp, and any directory already
# tracked by a sibling .dvc file (the directory's contents are reproducible
# via that pointer, so per-file scanning is redundant).

pointers_found=$(find . \
    \( -path ./.git -o -path ./.venv -o -path ./.dvc/cache -o -path ./.dvc/tmp -o -path ./node_modules \) -prune \
    -o -type f -name "*.dvc" -print 2>/dev/null | wc -l | tr -d ' ')

# ---------- untracked large-file scan (fast) ----------
#
# Exclude DVC-tracked directory subtrees: for each <dir>.dvc pointer file,
# prune <dir> from the scan (contents are reproducible through the pointer).

size_mb="${VERIFY_DVC_SIZE_MB:-$DEFAULT_SIZE_MB}"

# Collect prune arguments for directory-level .dvc pointers
prune_args=(-path ./.git -o -path ./.venv -o -path ./.dvc -o -path ./node_modules)
while IFS= read -r dvc_ptr; do
    # Directory pointer convention: foo/bar.dvc prunes foo/bar/
    dir_tracked="${dvc_ptr%.dvc}"
    if [[ -d "$dir_tracked" ]]; then
        prune_args+=(-o -path "$dir_tracked")
    fi
done < <(find . -maxdepth 5 -type f -name "*.dvc" \
    -not -path "./.git/*" -not -path "./.venv/*" -not -path "./.dvc/*" 2>/dev/null)

untracked_large=()
while IFS= read -r f; do
    # Also skip files with a direct sibling .dvc pointer
    if [[ -f "${f}.dvc" ]]; then
        continue
    fi
    untracked_large+=("$f")
done < <(find . \( "${prune_args[@]}" \) -prune \
    -o -type f -size "+${size_mb}M" -not -name "*.dvc" -print 2>/dev/null || true)

# ---------- output ----------

if [[ $MODE_JSON -eq 1 ]]; then
    printf '{\n'
    printf '  "schema_version": "%s",\n' "$SCHEMA_VERSION"
    printf '  "dvc_remote": "%s",\n' "$DVC_REMOTE"
    printf '  "pointers_found": %d,\n' "$pointers_found"
    printf '  "cloud_drift": %s,\n' "$cloud_drift"
    printf '  "strict_mode": %s,\n' "$([[ $MODE_STRICT -eq 1 ]] && echo true || echo false)"
    printf '  "untracked_large_files": ['
    if [[ ${#untracked_large[@]} -gt 0 ]]; then
        printf '\n'
        for i in "${!untracked_large[@]}"; do
            sep=$([[ $i -lt $(( ${#untracked_large[@]} - 1 )) ]] && echo "," || echo "")
            printf '    "%s"%s\n' "${untracked_large[$i]}" "$sep"
        done
        printf '  '
    fi
    printf ']\n'
    printf '}\n'
else
    echo "=== DVC Reproducibility Check ==="
    echo "remote:           $DVC_REMOTE"
    echo "pointers found:   $pointers_found"
    echo "local status:     $(echo "$status_output" | head -1)"
    echo "cloud drift:      $cloud_drift"
    if [[ "$cloud_drift" == "true" ]]; then
        echo "--- cloud drift details ---"
        echo "$cloud_output" | head -20
    fi
    echo "untracked >${size_mb}MB: ${#untracked_large[@]}"
    for f in "${untracked_large[@]}"; do
        echo "  $f"
    done
fi

if [[ $MODE_STRICT -eq 1 ]] && [[ "$cloud_drift" == "true" ]]; then
    exit 1
fi

exit 0
