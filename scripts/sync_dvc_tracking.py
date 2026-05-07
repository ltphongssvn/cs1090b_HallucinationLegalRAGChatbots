"""Add missing DVC tracking for all pipeline artifacts on disk.

Usage:
    .venv/bin/python scripts/sync_dvc_tracking.py [--dry-run]

Idempotent: safe to re-run. Only `dvc add`s files lacking .dvc pointers
and removes stale .dvc pointers whose target file no longer exists.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

EXPECTED_ARTIFACTS: dict[str, str] = {
    "data/raw/cl_bulk": "directory",
    "data/raw/cl_federal_appellate_bulk": "directory",
    "lepard_train_4000000_rev0194f95.jsonl": "file",
    "data/processed/baseline/corpus_chunks.jsonl": "file",
    "data/processed/baseline/corpus_chunks_enriched.jsonl": "file",
    "data/processed/baseline/corpus_chunks_cleaned.jsonl": "file",
    "data/processed/baseline/corpus_chunks_opinion_sample.jsonl": "file",
    "data/processed/baseline/cleaned/gold_pairs_test.jsonl": "file",
    "data/processed/baseline/cleaned/gold_pairs_val.jsonl": "file",
    "data/processed/baseline/cleaned/bm25_results.jsonl": "file",
    "data/processed/baseline/cleaned/bge_m3_results.jsonl": "file",
    "data/processed/baseline/cleaned/rrf_results.jsonl": "file",
    "data/processed/baseline/cleaned/reranker_results.jsonl": "file",
    "data/processed/baseline/cleaned/maxp/reranker_results.jsonl": "file",
    "data/processed/baseline/cleaned/finetuned/reranker_results.jsonl": "file",
    "data/processed/finetune/hard_negatives.jsonl": "file",
    "data/processed/finetune/hard_negatives.val.jsonl": "file",
    "data/processed/finetune/bge_reranker_legal/model.safetensors": "file",
    "data/processed/finetune/parade/parade_aggregator.pt": "file",
    "data/processed/lepard_cl_verified_subset.jsonl": "file",
    "data/processed/rag/no_rag/generations.jsonl": "file",
    "data/processed/rag/bm25_rag/generations.jsonl": "file",
    "data/processed/rag/bge_m3_rag/generations.jsonl": "file",
    "data/processed/rag/rrf_rag/generations.jsonl": "file",
    "data/processed/rag/reranker_rag/generations.jsonl": "file",
    "data/processed/hallucination/no_rag/judgments.jsonl": "file",
    "data/processed/hallucination/bm25_rag/judgments.jsonl": "file",
    "data/processed/hallucination/bge_m3_rag/judgments.jsonl": "file",
    "data/processed/hallucination/rrf_rag/judgments.jsonl": "file",
    "data/processed/hallucination/reranker_rag/judgments.jsonl": "file",
    "artifacts/ms3_pipeline.png": "file",
    "artifacts/ms3_infrastructure.png": "file",
    "artifacts/ms4_stratified_heatmap.png": "file",
}


def _dvc_tracked_paths() -> set[str]:
    tracked: set[str] = set()
    for p in Path(".").rglob("*.dvc"):
        s = str(p)
        if ".venv" in s or ".git" in s or s == "./.dvc":
            continue
        n = s[2:] if s.startswith("./") else s
        if n.endswith(".dvc"):
            n = n[:-4]
        tracked.add(n)
    return tracked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tracked = _dvc_tracked_paths()
    to_add = [p for p in EXPECTED_ARTIFACTS if Path(p).exists() and p not in tracked]
    stale = [p for p in EXPECTED_ARTIFACTS if p in tracked and not Path(p).exists()]

    print(f"Missing DVC tracking ({len(to_add)}):")
    for p in to_add:
        print(f"  + {p}")
    print(f"\nStale DVC pointers ({len(stale)}):")
    for p in stale:
        print(f"  - {p}.dvc")

    if args.dry_run:
        print("\n[dry-run] no changes made")
        return 0

    rc = 0
    for p in to_add:
        print(f"\n>>> dvc add {p}")
        proc = subprocess.run([".venv/bin/dvc", "add", p], check=False)
        if proc.returncode != 0:
            print(f"FAILED: rc={proc.returncode}")
            rc = 1

    for p in stale:
        dvc_pointer = Path(f"{p}.dvc")
        if dvc_pointer.exists():
            print(f"\n>>> removing stale {dvc_pointer}")
            dvc_pointer.unlink()

    return rc


if __name__ == "__main__":
    sys.exit(main())
