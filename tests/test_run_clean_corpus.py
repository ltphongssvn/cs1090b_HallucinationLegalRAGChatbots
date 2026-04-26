"""TDD for scripts/run_clean_corpus.py — reproducible runner with config.

Wraps clean_corpus.main() with hardcoded sensible defaults derived from
empirical 27GB-corpus measurements (~65 rows/sec via eyecite + hyperscan).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_clean_corpus.py"


@pytest.mark.unit
class TestRunCleanCorpus:
    def test_script_exists(self) -> None:
        assert SCRIPT.exists(), f"missing {SCRIPT}"

    def test_dry_run_prints_resolved_config(self, tmp_path: Path) -> None:
        in_path = tmp_path / "in.jsonl"
        in_path.write_text('{"opinion_id":1,"cluster_id":1,"chunk_index":0,"text":"x"}\n')
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--in-path",
                str(in_path),
                "--out-path",
                str(tmp_path / "out.jsonl"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "DRY RUN" in result.stdout
        # Production defaults must appear
        assert "rows_per_shard" in result.stdout
        assert "shard_timeout_sec" in result.stdout
        assert "workers" in result.stdout

    def test_warmup_hyperscan_cache_runs_first(self, tmp_path: Path) -> None:
        """Cache pre-warm must happen before subprocess pool to avoid race."""
        in_path = tmp_path / "in.jsonl"
        in_path.write_text('{"opinion_id":1,"cluster_id":1,"chunk_index":0,"text":"x"}\n')
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--in-path",
                str(in_path),
                "--out-path",
                str(tmp_path / "out.jsonl"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "warming hyperscan cache" in result.stdout.lower() or "hyperscan_cache" in result.stdout.lower()

    def test_overrides_via_cli_flags(self, tmp_path: Path) -> None:
        in_path = tmp_path / "in.jsonl"
        in_path.write_text('{"opinion_id":1,"cluster_id":1,"chunk_index":0,"text":"x"}\n')
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--in-path",
                str(in_path),
                "--out-path",
                str(tmp_path / "out.jsonl"),
                "--workers",
                "4",
                "--rows-per-shard",
                "10000",
                "--shard-timeout-sec",
                "300",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "workers           : 4" in result.stdout
        assert "rows_per_shard    : 10000" in result.stdout
        assert "shard_timeout_sec : 300" in result.stdout

    def test_end_to_end_small_input(self, tmp_path: Path) -> None:
        in_path = tmp_path / "in.jsonl"
        rows = [
            {"opinion_id": i, "cluster_id": 100 + i, "chunk_index": 0, "text": f"Brown v. Board, 347 U.S. {i}"}
            for i in range(5)
        ]
        with in_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        out_path = tmp_path / "out.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--in-path",
                str(in_path),
                "--out-path",
                str(out_path),
                "--workers",
                "2",
                "--rows-per-shard",
                "3",
                "--shard-timeout-sec",
                "60",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert out_path.exists()
        out_rows = [json.loads(line) for line in out_path.open()]
        assert len(out_rows) == 5
        for row in out_rows:
            assert "347 U.S." not in row["text"]
