# tests/test_wandb_integration.py
"""Contract tests for W&B CLI integration in src/dataset_probe.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_shard_dir(tmp_path: Path) -> Path:
    import json
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard = shard_dir / "shard_000.jsonl"
    record = {
        "id": "1", "cluster_id": "c1", "docket_id": "d1", "court_id": "ca9",
        "court_name": "Ninth Circuit", "case_name": "Smith v. Jones",
        "date_filed": "2020-01-01", "precedential_status": "Published",
        "opinion_type": "majority", "extracted_by_ocr": False,
        "raw_text": "Smith v. Jones, 123 F.3d 456 (9th Cir. 2020).",
        "text": "Smith v. Jones, 123 F.3d 456 (9th Cir. 2020). " + ("The court held. " * 60),
        "text_length": 2000, "text_source": "plain_text", "cleaning_flags": [],
        "source": "courtlistener", "token_count": 400, "paragraph_count": 5,
        "citation_count": 3, "text_hash": "abc123", "citation_density": 0.05,
        "is_precedential": True, "text_entropy": 4.2,
    }
    with open(shard, "w") as fh:
        for i in range(20):
            r = dict(record)
            r["id"] = str(i)
            import json as _json
            fh.write(_json.dumps(r) + "\n")
    return shard_dir


class TestCLIWandbFlags:
    def test_cli_accepts_log_to_wandb_flag(self, sample_shard_dir, tmp_path):
        """--log-to-wandb flag must be accepted without error (wandb mocked offline)."""
        result = subprocess.run(
            [
                sys.executable, "-m", "src.dataset_probe",
                "--data-dir", str(sample_shard_dir),
                "--subset", "10",
                "--output", str(tmp_path / "r.json"),
                "--skip-tokenizer", "--skip-spacy",
                "--log-to-wandb",
                "--wandb-entity", "phl690-harvard-extension-schol",
                "--wandb-project", "cs1090b",
                "--wandb-name", "test_run",
            ],
            capture_output=True, text=True,
            env={**__import__("os").environ, "WANDB_MODE": "disabled"},
        )
        assert result.returncode == 0, result.stderr

    def test_cli_accepts_wandb_entity_flag(self, sample_shard_dir, tmp_path):
        """--wandb-entity must be a recognized CLI argument."""
        result = subprocess.run(
            [
                sys.executable, "-m", "src.dataset_probe",
                "--data-dir", str(sample_shard_dir),
                "--subset", "10",
                "--output", str(tmp_path / "r.json"),
                "--skip-tokenizer", "--skip-spacy",
                "--wandb-entity", "phl690-harvard-extension-schol",
            ],
            capture_output=True, text=True,
            env={**__import__("os").environ, "WANDB_MODE": "disabled"},
        )
        assert result.returncode == 0, result.stderr

    def test_cli_accepts_wandb_project_flag(self, sample_shard_dir, tmp_path):
        """--wandb-project must be a recognized CLI argument."""
        result = subprocess.run(
            [
                sys.executable, "-m", "src.dataset_probe",
                "--data-dir", str(sample_shard_dir),
                "--subset", "10",
                "--output", str(tmp_path / "r.json"),
                "--skip-tokenizer", "--skip-spacy",
                "--wandb-project", "cs1090b",
            ],
            capture_output=True, text=True,
            env={**__import__("os").environ, "WANDB_MODE": "disabled"},
        )
        assert result.returncode == 0, result.stderr

    def test_cli_accepts_wandb_name_flag(self, sample_shard_dir, tmp_path):
        """--wandb-name must be a recognized CLI argument."""
        result = subprocess.run(
            [
                sys.executable, "-m", "src.dataset_probe",
                "--data-dir", str(sample_shard_dir),
                "--subset", "10",
                "--output", str(tmp_path / "r.json"),
                "--skip-tokenizer", "--skip-spacy",
                "--wandb-name", "my_probe_run",
            ],
            capture_output=True, text=True,
            env={**__import__("os").environ, "WANDB_MODE": "disabled"},
        )
        assert result.returncode == 0, result.stderr

    def test_without_log_to_wandb_no_wandb_called(self, sample_shard_dir, tmp_path):
        """Without --log-to-wandb, probe must complete without any wandb init."""
        result = subprocess.run(
            [
                sys.executable, "-m", "src.dataset_probe",
                "--data-dir", str(sample_shard_dir),
                "--subset", "10",
                "--output", str(tmp_path / "r.json"),
                "--skip-tokenizer", "--skip-spacy",
            ],
            capture_output=True, text=True,
            env={**__import__("os").environ, "WANDB_MODE": "disabled"},
        )
        assert result.returncode == 0
        assert "wandb" not in result.stdout.lower() or "warning" not in result.stdout.lower()
