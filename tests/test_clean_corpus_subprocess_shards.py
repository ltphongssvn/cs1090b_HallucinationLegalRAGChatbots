"""TDD for subprocess-per-shard cleaning (replaces mp.Pool).

Industry pattern (per cpython#96062, #66587): mp.Pool deadlocks when a
worker hangs. Solution: each shard runs as an independent subprocess with
subprocess.run(timeout=N). Hung shards get killed cleanly, others continue.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clean_corpus import main  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _chunk(oid: int, cid: int, idx: int, text: str) -> dict:
    return {"opinion_id": oid, "cluster_id": cid, "chunk_index": idx, "text": text}


@pytest.fixture
def make_corpus(tmp_path: Path):
    def _build(rows: list[dict]) -> tuple[Path, Path]:
        in_path = tmp_path / "corpus.jsonl"
        out_path = tmp_path / "corpus_cleaned.jsonl"
        _write_jsonl(in_path, rows)
        return in_path, out_path

    return _build


@pytest.mark.integration
class TestSubprocessShardMode:
    """mode='subprocess' runs each shard as independent subprocess.

    Marked integration because subprocess.run spawns real Python interpreters
    and writes real shard files (~50-100ms per shard).
    """

    def test_subprocess_mode_cleans_correctly(self, make_corpus) -> None:
        rows = [_chunk(i, 100 + i, 0, f"Brown v. Board, 347 U.S. {i}") for i in range(20)]
        in_path, out_path = make_corpus(rows)
        main(in_path=in_path, out_path=out_path, workers=4, mode="subprocess")
        rows_out = [json.loads(line) for line in out_path.open()]
        assert len(rows_out) == 20
        for row in rows_out:
            assert "347 U.S." not in row["text"]

    def test_subprocess_mode_preserves_input_order(self, make_corpus) -> None:
        rows = [_chunk(i, 100 + i, 0, f"Plain text {i}.") for i in range(30)]
        in_path, out_path = make_corpus(rows)
        main(in_path=in_path, out_path=out_path, workers=3, mode="subprocess")
        rows_out = [json.loads(line) for line in out_path.open()]
        assert [r["opinion_id"] for r in rows_out] == list(range(30))

    def test_subprocess_mode_summary_includes_dlq_field(self, make_corpus) -> None:
        rows = [_chunk(i, i, 0, f"text {i}") for i in range(10)]
        in_path, out_path = make_corpus(rows)
        main(in_path=in_path, out_path=out_path, workers=2, mode="subprocess")
        meta = json.loads(out_path.with_suffix(".summary.json").read_text())
        assert "shards_failed" in meta
        assert meta["shards_failed"] == 0
        assert "shard_timeout_sec" in meta

    def test_subprocess_mode_handles_shard_timeout(self, make_corpus, tmp_path: Path) -> None:
        """Critical: when a shard worker times out, pipeline survives.

        Sets a 0.001s timeout — guaranteed to fire even on a no-op subprocess
        (Python interpreter startup alone takes ~30-50ms). All shards will
        be killed; pipeline must NOT hang and must report shards_failed > 0.
        """
        rows = [_chunk(i, i, 0, f"text {i}") for i in range(10)]
        in_path, out_path = make_corpus(rows)
        main(
            in_path=in_path,
            out_path=out_path,
            workers=2,
            mode="subprocess",
            shard_timeout_sec=0.001,
            rows_per_shard=5,  # force 2 shards from 10 rows
        )
        meta = json.loads(out_path.with_suffix(".summary.json").read_text())
        # All shards timed out; rows lost but pipeline survived
        assert meta["shards_failed"] == 2
        assert "failed_shard_paths" in meta
        # Output file exists (possibly empty) — atomic write completed
        assert out_path.exists()

    def test_subshard_size_limits_blast_radius(self, make_corpus, tmp_path: Path) -> None:
        """A single hung shard must not lose more than --rows-per-shard rows.

        With 100 rows + workers=2 + rows_per_shard=10, we get 10 shards.
        Killing 1 shard loses ≤10 rows (not all 50 a worker would otherwise own).
        """
        rows = [_chunk(i, i, 0, f"text {i}") for i in range(100)]
        in_path, out_path = make_corpus(rows)
        main(
            in_path=in_path,
            out_path=out_path,
            workers=2,
            mode="subprocess",
            rows_per_shard=10,
        )
        meta = json.loads(out_path.with_suffix(".summary.json").read_text())
        assert meta["shards_failed"] == 0
        assert meta["n_shards"] == 10


@pytest.mark.integration
class TestSubprocessShardMixedFailure:
    """The production failure mode: one shard hangs, others must complete.

    Uses a sentinel string `__HANG_SENTINEL__` that the worker recognizes
    and sleeps on (longer than shard_timeout_sec). All other rows process
    normally. Verifies:
      - the hung shard times out (shards_failed == 1)
      - successful shards write their output (rows_out > 0)
      - pipeline does not deadlock (test completes within reasonable time)
      - failed shard's input path appears in summary["failed_shard_paths"]
    """

    def test_one_hung_shard_does_not_block_others(self, make_corpus) -> None:
        # Construct rows where row 0 will be in shard 0 with the sentinel,
        # remaining rows distributed across other shards.
        rows = [_chunk(0, 0, 0, "__HANG_SENTINEL__")]
        rows.extend(_chunk(i, i, 0, f"normal text {i}") for i in range(1, 20))
        in_path, out_path = make_corpus(rows)
        main(
            in_path=in_path,
            out_path=out_path,
            workers=4,
            mode="subprocess",
            shard_timeout_sec=2,  # generous for normal shards, kills hung one
            rows_per_shard=5,  # 4 shards: shard 0 hangs, shards 1-3 succeed
        )
        meta = json.loads(out_path.with_suffix(".summary.json").read_text())
        # Shard 0 hung; shards 1-3 succeeded
        assert meta["shards_failed"] == 1
        assert "failed_shard_paths" in meta
        assert len(meta["failed_shard_paths"]) == 1
        # Successful shards wrote their rows (15 normal rows in shards 1-3)
        rows_out = [json.loads(line) for line in out_path.open()]
        assert len(rows_out) >= 14  # at least most non-hung rows survived
        # No sentinel in output (lost with hung shard)
        for row in rows_out:
            assert "__HANG_SENTINEL__" not in row.get("text", "")
