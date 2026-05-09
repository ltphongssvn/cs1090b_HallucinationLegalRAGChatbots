"""Cell 0 logic: notebook bootstrap (repo-root resolution + accumulator reset).

Extracted from the notebook so the cell contains only orchestration.
Entry point: :func:`run`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` until ``src/__init__.py`` is found."""
    repo = (start or Path.cwd()).resolve()
    while repo != repo.parent and not (repo / "src" / "__init__.py").is_file():
        repo = repo.parent
    if not (repo / "src" / "__init__.py").is_file():
        raise RuntimeError(f"could not locate repo root from {start or Path.cwd()}")
    return repo


def run() -> dict[str, Any]:
    """Resolve repo root, prepend to sys.path, chdir, reset cell-time accumulator."""
    repo = find_repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(repo)

    # Reset accumulated cell time on kernel restart
    from src.timer import reset_accumulated

    reset_accumulated()

    return {
        "repo_root": str(repo),
        "cwd": os.getcwd(),
        "sys_path_0": sys.path[0],
    }
