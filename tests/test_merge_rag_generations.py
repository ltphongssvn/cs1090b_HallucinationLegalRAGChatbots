# tests/test_merge_rag_generations.py
"""Tests for scripts.merge_rag_generations — concatenate per-rank RAG outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def merge_module() -> Any:
    from scripts import merge_rag_generations

    return merge_rag_generations


# ---------- contract ----------


@pytest.mark.contract
class TestContract:
    def test_main_callable(self, merge_module: Any) -> None:
        assert callable(getattr(merge_module, "main", None))

    def test_merge_callable(self, merge_module: Any) -> None:
        assert callable(getattr(merge_module, "merge", None))


# ---------- end-to-end merge ----------


@pytest.mark.unit
class TestMerge:
    def _make_rank_outputs(self, tmp_path: Path, ablation_label: str = "no_rag") -> Path:
        """Stage 4 rank shards + 4 rank summaries that merge() expects."""
        out_dir = tmp_path / "rag" / ablation_label
        out_dir.mkdir(parents=True)
        # 4 rank result files (2 generations each, 8 total)
        for r in range(4):
            _write_jsonl(
                out_dir / f"generations.rank{r:03d}.jsonl",
                [
                    {
                        "source_id": r * 2,
                        "dest_id": r * 2 + 100,
                        "source_cluster_id": r * 2 + 1000,
                        "ablation": "none",
                        "generation": f"answer_{r}_a",
                    },
                    {
                        "source_id": r * 2 + 1,
                        "dest_id": r * 2 + 101,
                        "source_cluster_id": r * 2 + 1001,
                        "ablation": "none",
                        "generation": f"answer_{r}_b",
                    },
                ],
            )
            (out_dir / f"generation_summary.rank{r:03d}.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "ablation": "none",
                        "ablation_label": ablation_label,
                        "n_queries_total": 8,
                        "n_queries_this_rank": 2,
                        "n_generated": 2,
                        "n_total_tokens_out": 100,
                        "top_k_context": 5,
                        "max_new_tokens": 256,
                        "batch_size": 16,
                        "max_length": 4096,
                        "max_chunks_per_cluster": 2,
                        "generator_model": "Qwen/Qwen2.5-7B-Instruct",
                        "device": "cuda",
                        "device_name": "NVIDIA L4",
                        "world_size": 4,
                        "shard_rank": r,
                        "shard_start": r * 2,
                        "shard_end": (r + 1) * 2,
                        "encoder_load_seconds": 30.0,
                        "text_index_seconds": 10.0,
                        "generation_seconds": 100.0 + r,  # vary to exercise max
                        "seed": 0,
                        "git_sha": "abc",
                        "results_hash": "x" * 64,
                    }
                ),
                encoding="utf-8",
            )
        return tmp_path / "rag"

    def test_merges_all_shards_in_rank_order(self, merge_module: Any, tmp_path: Path) -> None:
        rag_root = self._make_rank_outputs(tmp_path)
        result = merge_module.merge(
            ablation="none",
            out_root=rag_root,
            world_size=4,
            seed=0,
        )
        merged = rag_root / "no_rag" / "generations.jsonl"
        assert merged.exists()
        rows = [json.loads(line) for line in merged.open()]
        assert len(rows) == 8
        # Source ids should be in rank order: 0,1, 2,3, 4,5, 6,7
        assert [r["source_id"] for r in rows] == list(range(8))
        assert result["n_queries_total"] == 8
        assert result["n_generated_total"] == 8

    def test_summary_aggregates_max_walltime(self, merge_module: Any, tmp_path: Path) -> None:
        rag_root = self._make_rank_outputs(tmp_path)
        result = merge_module.merge(
            ablation="none",
            out_root=rag_root,
            world_size=4,
            seed=0,
        )
        # Per-rank generation_seconds = 100, 101, 102, 103 — max should be 103
        assert result["generation_seconds"] == pytest.approx(103.0)

    def test_summary_sums_tokens(self, merge_module: Any, tmp_path: Path) -> None:
        rag_root = self._make_rank_outputs(tmp_path)
        result = merge_module.merge(
            ablation="none",
            out_root=rag_root,
            world_size=4,
            seed=0,
        )
        # 4 ranks × 100 tokens each
        assert result["n_total_tokens_out"] == 400

    def test_unknown_ablation_raises(self, merge_module: Any, tmp_path: Path) -> None:
        with pytest.raises(KeyError):
            merge_module.merge(
                ablation="bogus",
                out_root=tmp_path,
                world_size=4,
                seed=0,
            )

    def test_missing_shard_raises(self, merge_module: Any, tmp_path: Path) -> None:
        rag_root = self._make_rank_outputs(tmp_path)
        # Delete one shard
        (rag_root / "no_rag" / "generations.rank002.jsonl").unlink()
        with pytest.raises(RuntimeError, match="expected 4"):
            merge_module.merge(
                ablation="none",
                out_root=rag_root,
                world_size=4,
                seed=0,
            )
