"""Test that all expected pipeline artifacts are tracked by DVC."""

from pathlib import Path

import pytest
import yaml

# All artifacts produced by notebook cells / scripts in src/ and scripts/
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
    # Finetune (Cells 17b/c/d)
    "data/processed/finetune/hard_negatives.jsonl": "file",
    "data/processed/finetune/hard_negatives.val.jsonl": "file",
    "data/processed/finetune/bge_reranker_legal/model.safetensors": "file",
    "data/processed/finetune/parade/parade_aggregator.pt": "file",
    # LePaRD verified subset (Cell 11)
    "data/processed/lepard_cl_verified_subset.jsonl": "file",
    # RAG generations (Cell 20)
    "data/processed/rag/no_rag/generations.jsonl": "file",
    "data/processed/rag/bm25_rag/generations.jsonl": "file",
    "data/processed/rag/bge_m3_rag/generations.jsonl": "file",
    "data/processed/rag/rrf_rag/generations.jsonl": "file",
    "data/processed/rag/reranker_rag/generations.jsonl": "file",
    # Hallucination judgments (Cell 21)
    "data/processed/hallucination/no_rag/judgments.jsonl": "file",
    "data/processed/hallucination/bm25_rag/judgments.jsonl": "file",
    "data/processed/hallucination/bge_m3_rag/judgments.jsonl": "file",
    "data/processed/hallucination/rrf_rag/judgments.jsonl": "file",
    "data/processed/hallucination/reranker_rag/judgments.jsonl": "file",
    # Figures (Cells 16a, 16b, 18b)
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
        normalized = s[2:] if s.startswith("./") else s
        if normalized.endswith(".dvc"):
            normalized = normalized[:-4]
        tracked.add(normalized)
    return tracked


def _dvc_pointer_path(artifact_path: str) -> Path:
    """Return the .dvc pointer file path for an artifact."""
    return Path(artifact_path + ".dvc")


@pytest.mark.contract
class TestDVCTrackingCompleteness:
    def test_all_on_disk_artifacts_are_dvc_tracked(self):
        """Every artifact present on disk must have a .dvc pointer."""
        tracked = _dvc_tracked_paths()
        missing = [p for p in EXPECTED_ARTIFACTS if Path(p).exists() and p not in tracked]
        assert not missing, f"{len(missing)} artifacts on disk lack DVC tracking:\n  - " + "\n  - ".join(missing)

    def test_no_stale_dvc_pointers(self):
        """Every DVC pointer must be well-formed (parseable YAML with outs[].md5 and outs[].path).

        Note: the referenced artifact does NOT need to be on the working tree —
        DVC stores artifacts in the cache and materializes them only via
        `dvc pull` or `dvc checkout`. A pointer is stale only if it is
        malformed or missing required fields, not if the working-tree copy
        is absent.
        """
        tracked = _dvc_tracked_paths()
        malformed: list[str] = []
        for artifact in EXPECTED_ARTIFACTS:
            if artifact not in tracked:
                continue
            pointer = _dvc_pointer_path(artifact)
            if not pointer.exists():
                malformed.append(f"{artifact} (pointer file missing: {pointer})")
                continue
            try:
                data = yaml.safe_load(pointer.read_text())
            except yaml.YAMLError as exc:
                malformed.append(f"{artifact} (unparseable YAML: {exc})")
                continue
            outs = (data or {}).get("outs") or []
            if not outs:
                malformed.append(f"{artifact} (no outs entry in pointer)")
                continue
            entry = outs[0]
            if "md5" not in entry:
                malformed.append(f"{artifact} (outs entry missing md5)")
            if "path" not in entry:
                malformed.append(f"{artifact} (outs entry missing path)")
        assert not malformed, f"{len(malformed)} malformed DVC pointers:\n  - " + "\n  - ".join(malformed)
