"""Audit all artifacts produced by notebook/scripts vs DVC tracking.

Usage:
    .venv/bin/python scripts/audit_dvc_tracking.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# All known artifact paths the pipeline produces (from cell interpretations)
EXPECTED_ARTIFACTS: dict[str, str] = {
    # Raw data
    "data/raw/cl_bulk": "directory",
    "data/raw/cl_federal_appellate_bulk": "directory",
    "lepard_train_4000000_rev0194f95.jsonl": "file",
    # Processed corpus (Cells 5, 12, 12c, 12d)
    "data/processed/baseline/corpus_chunks.jsonl": "file",
    "data/processed/baseline/corpus_chunks_enriched.jsonl": "file",
    "data/processed/baseline/corpus_chunks_cleaned.jsonl": "file",
    "data/processed/baseline/corpus_chunks_opinion_sample.jsonl": "file",
    # Cleaned gold (Cell 12b)
    "data/processed/baseline/cleaned/gold_pairs_test.jsonl": "file",
    "data/processed/baseline/cleaned/gold_pairs_val.jsonl": "file",
    # Retrieval results (Cells 13, 14, 17, 18)
    "data/processed/baseline/cleaned/bm25_results.jsonl": "file",
    "data/processed/baseline/cleaned/bge_m3_results.jsonl": "file",
    "data/processed/baseline/cleaned/rrf_results.jsonl": "file",
    "data/processed/baseline/cleaned/reranker_results.jsonl": "file",
    "data/processed/baseline/cleaned/maxp/reranker_results.jsonl": "file",
    "data/processed/baseline/cleaned/finetuned/reranker_results.jsonl": "file",
    # Finetune (Cells 17b, 17c, 17d)
    "data/processed/finetune/hard_negatives.jsonl": "file",
    "data/processed/finetune/hard_negatives.val.jsonl": "file",
    "data/processed/finetune/bge_reranker_legal/model.safetensors": "file",
    "data/processed/finetune/parade/parade_aggregator.pt": "file",
    # LePaRD verified subset (Cell 11 semantic bridge)
    "data/processed/lepard_cl_verified_subset.jsonl": "file",
    # RAG generations (Cell 20)
    "data/processed/rag/no_rag/generations.jsonl": "file",
    "data/processed/rag/bm25_rag/generations.jsonl": "file",
    "data/processed/rag/bge_m3_rag/generations.jsonl": "file",
    "data/processed/rag/rrf_rag/generations.jsonl": "file",
    "data/processed/rag/reranker_rag/generations.jsonl": "file",
    # Hallucination judgments (Cell 21) — large JSONL files
    "data/processed/hallucination/no_rag/judgments.jsonl": "file",
    "data/processed/hallucination/bm25_rag/judgments.jsonl": "file",
    "data/processed/hallucination/bge_m3_rag/judgments.jsonl": "file",
    "data/processed/hallucination/rrf_rag/judgments.jsonl": "file",
    "data/processed/hallucination/reranker_rag/judgments.jsonl": "file",
    # Figures (Cells 16a, 16b, 18b)
    "artifacts/ms3_pipeline.png": "file",
    "artifacts/ms3_infrastructure.png": "file",
    "artifacts/ms4_stratified_heatmap.png": "file",
    "artifacts/eda_ms3_lepard": "directory",
}


def collect_dvc_tracked() -> set[str]:
    """Return set of paths tracked by DVC (.dvc files stripped of suffix)."""
    tracked: set[str] = set()
    for p in Path(".").rglob("*.dvc"):
        s = str(p)
        if ".venv" in s or ".git" in s or s == "./.dvc":
            continue
        # Strip leading ./ and trailing .dvc
        normalized = s[2:] if s.startswith("./") else s
        if normalized.endswith(".dvc"):
            normalized = normalized[:-4]
        tracked.add(normalized)
    return tracked


def main() -> int:
    dvc_tracked = collect_dvc_tracked()

    print("=" * 78)
    print(f"{'ON DISK':<10}{'DVC TRACKED':<14}PATH")
    print("=" * 78)

    missing_dvc: list[tuple[str, str]] = []
    stale_dvc: list[str] = []

    for path_str, kind in EXPECTED_ARTIFACTS.items():
        p = Path(path_str)
        on_disk = p.exists()
        in_dvc = path_str in dvc_tracked
        d = "yes" if on_disk else "NO"
        v = "yes" if in_dvc else "NO"
        print(f"{d:<10}{v:<14}{path_str}")
        if on_disk and not in_dvc:
            missing_dvc.append((path_str, kind))
        if in_dvc and not on_disk:
            stale_dvc.append(path_str)

    print("\n" + "=" * 78)
    print(f"ON DISK BUT NOT DVC-TRACKED ({len(missing_dvc)}):")
    for path_str, kind in missing_dvc:
        print(f"  [{kind:9}] {path_str}")

    print(f"\nDVC-TRACKED BUT MISSING ON DISK ({len(stale_dvc)}):")
    for path_str in stale_dvc:
        print(f"  {path_str}")

    # DVC-tracked paths NOT in the expected list (orphans)
    expected = set(EXPECTED_ARTIFACTS.keys())
    orphans = sorted(dvc_tracked - expected)
    print(f"\nDVC-TRACKED BUT NOT IN EXPECTED LIST (orphans, {len(orphans)}):")
    for path_str in orphans:
        print(f"  {path_str}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Expected artifacts:            {len(EXPECTED_ARTIFACTS)}")
    print(f"  On disk:                       {sum(1 for p in EXPECTED_ARTIFACTS if Path(p).exists())}")
    print(f"  DVC-tracked (expected):        {sum(1 for p in EXPECTED_ARTIFACTS if p in dvc_tracked)}")
    print(f"  Missing DVC tracking:          {len(missing_dvc)}")
    print(f"  Stale DVC pointers (no file):  {len(stale_dvc)}")

    return 0 if not missing_dvc and not stale_dvc else 1


if __name__ == "__main__":
    sys.exit(main())
