"""TDD for scripts/run_baseline_bm25_verified.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_baseline_bm25_verified.py"

pytestmark = pytest.mark.unit


class TestRunBaselineBm25Verified:
    def test_script_exists(self) -> None:
        assert SCRIPT.exists()

    def test_dry_run_prints_resolved_config(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--corpus-path",
                str(tmp_path / "c.jsonl"),
                "--gold-pairs-path",
                str(tmp_path / "g.jsonl"),
                "--out-dir",
                str(tmp_path),
                "--index-dir",
                str(tmp_path / "idx"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "DRY RUN" in result.stdout
        assert "corpus_path" in result.stdout
        assert "gold_pairs_path" in result.stdout
        assert "index_dir" in result.stdout
        assert "top_k" in result.stdout

    def test_default_top_k_is_100(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--corpus-path",
                "/tmp/c.jsonl",
                "--gold-pairs-path",
                "/tmp/g.jsonl",
                "--out-dir",
                str(tmp_path),
                "--index-dir",
                str(tmp_path / "idx"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "top_k            : 100" in result.stdout

    def test_overrides_via_cli(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--corpus-path",
                "/tmp/c.jsonl",
                "--gold-pairs-path",
                "/tmp/g.jsonl",
                "--out-dir",
                str(tmp_path),
                "--index-dir",
                str(tmp_path / "idx"),
                "--top-k",
                "50",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "top_k            : 50" in result.stdout
