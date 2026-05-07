"""Untrack PNG artifacts from Git so they can be DVC-tracked instead.

Idempotent: safe to re-run.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ARTIFACTS_TO_UNTRACK = [
    "artifacts/ms3_pipeline.png",
    "artifacts/ms3_infrastructure.png",
    "artifacts/ms4_stratified_heatmap.png",
]


def _git_tracked(path: str) -> bool:
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        capture_output=True, check=False,
    )
    return proc.returncode == 0


def main() -> int:
    for path in ARTIFACTS_TO_UNTRACK:
        if not Path(path).exists():
            print(f"SKIP {path} (not on disk)")
            continue
        if not _git_tracked(path):
            print(f"SKIP {path} (not in Git)")
            continue
        print(f">>> git rm --cached {path}")
        proc = subprocess.run(
            ["git", "rm", "--cached", path], check=False
        )
        if proc.returncode != 0:
            return 1

    # Ensure artifacts/*.png is gitignored going forward
    gitignore = Path(".gitignore")
    needed = "artifacts/*.png"
    existing = gitignore.read_text() if gitignore.exists() else ""
    if needed not in existing:
        with gitignore.open("a") as f:
            f.write(f"\n# DVC-tracked figure artifacts\n{needed}\n")
        print(f"appended '{needed}' to .gitignore")
    else:
        print(f".gitignore already contains '{needed}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
