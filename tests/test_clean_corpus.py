"""TDD for scripts/clean_corpus.py — clean citation leakage from corpus chunk text.

Mirror of clean_gold_pairs.py but for corpus_chunks_enriched.jsonl.
Cleans the `text` field on each chunk row, preserves opinion_id/cluster_id/chunk_index.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "clean_corpus.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clean_corpus import main  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _chunk(oid: int, cid: int, idx: int, text: str) -> dict:
    return {
        "opinion_id": oid,
        "cluster_id": cid,
        "chunk_index": idx,
        "text": text,
    }


@pytest.fixture
def make_corpus(tmp_path: Path):
    def _build(rows: list[dict]) -> tuple[Path, Path]:
        in_path = tmp_path / "corpus.jsonl"
        out_path = tmp_path / "corpus_cleaned.jsonl"
        _write_jsonl(in_path, rows)
        return in_path, out_path

    return _build


@pytest.mark.unit
class TestCleanCorpusInProcess:
    def test_script_exists(self) -> None:
        assert SCRIPT.exists(), f"missing {SCRIPT}"

    def test_cleans_text_field(self, make_corpus) -> None:
        in_path, out_path = make_corpus(
            [
                _chunk(100, 1000, 0, "The Court held in Brown v. Board, 347 U.S. 483 (1954) that..."),
                _chunk(200, 2000, 0, "Plain corpus text without citations."),
            ]
        )
        main(in_path=in_path, out_path=out_path)
        rows = [json.loads(line) for line in out_path.open()]
        assert len(rows) == 2
        assert "347 U.S." not in rows[0]["text"]
        assert "Brown" not in rows[0]["text"]
        assert "Plain corpus text" in rows[1]["text"]

    def test_preserves_id_fields(self, make_corpus) -> None:
        in_path, out_path = make_corpus(
            [
                _chunk(42, 777, 5, "See Roe v. Wade, 410 U.S. 113."),
            ]
        )
        main(in_path=in_path, out_path=out_path)
        row = json.loads(out_path.read_text().strip())
        assert row["opinion_id"] == 42
        assert row["cluster_id"] == 777
        assert row["chunk_index"] == 5
        assert "410 U.S." not in row["text"]

    def test_writes_summary(self, make_corpus) -> None:
        in_path, out_path = make_corpus([_chunk(1, 1, 0, "Brown v. Board, 347 U.S. 483")])
        main(in_path=in_path, out_path=out_path)
        summary = out_path.with_suffix(".summary.json")
        assert summary.exists()
        meta = json.loads(summary.read_text())
        assert meta["total_rows"] == 1
        assert "git_sha" in meta
        assert "input_sha256" in meta
        assert "output_sha256" in meta

    def test_idempotent(self, make_corpus) -> None:
        import hashlib

        in_path, out_path = make_corpus(
            [
                _chunk(1, 1, 0, "Brown v. Board, 347 U.S. 483 (1954)"),
                _chunk(2, 2, 0, "Plain text."),
            ]
        )
        main(in_path=in_path, out_path=out_path)
        first = out_path.read_bytes()

        out2 = out_path.parent / "out2.jsonl"
        main(in_path=out_path, out_path=out2)
        second = out2.read_bytes()
        assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()

    def test_raises_on_missing_input(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            main(in_path=tmp_path / "missing.jsonl", out_path=tmp_path / "out.jsonl")


@pytest.mark.integration
class TestCleanCorpusCLI:
    def test_dry_run_via_subprocess(self, make_corpus) -> None:
        in_path, out_path = make_corpus([_chunk(1, 1, 0, "Brown v. Board, 347 U.S. 483")])
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--in-path", str(in_path), "--out-path", str(out_path), "--dry-run"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "DRY RUN" in result.stdout
        assert not out_path.exists()
