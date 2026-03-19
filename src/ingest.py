# src/ingest.py
# Ingestion pipeline entrypoint — run once to materialize an immutable artifact.
# Usage: python -m src.ingest --artifact-path ./data/courtlistener --max-samples 100000
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from datetime import datetime, timezone
from typing import Any, Literal

from src.dataset_config import ArtifactManifest, DatasetConfig
from src.dataset_loader import DatasetLoader
from src.dataset_probe import ModelQualitySignals


def _sha256_dir(path: pathlib.Path) -> str:
    """SHA256 of sorted file contents in artifact directory."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.name != "artifact_manifest.json":
            h.update(f.read_bytes())
    return h.hexdigest()


def run_ingestion(
    artifact_path: str,
    max_samples: int = 100_000,
    project: str = "hallucination-legal-rag",
    wandb_mode: Literal["online", "offline", "disabled"] = "online",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Full ingestion pipeline:
    1. Open W&B run
    2. Load from HF Hub (trust_remote_code allowed in ingestion-only mode)
    3. Validate, normalize, collect quality signals
    4. Save to disk as immutable artifact
    5. Write artifact_manifest.json
    6. Log provenance + metrics + tables to W&B
    7. Attach manifest as W&B artifact
    8. Finish run

    Returns summary dict.
    """
    import wandb

    config = DatasetConfig(
        data_source="hf",
        reproducible=False,  # ingestion: HF Hub allowed
        artifact_path=artifact_path,
    )
    loader = DatasetLoader(config)
    provenance = loader.get_provenance()

    run = wandb.init(
        project=project,
        job_type="ingestion",
        config={
            **provenance,
            "artifact_path": artifact_path,
            "max_samples": max_samples,
            "dry_run": dry_run,
        },
        mode=wandb_mode,
    )

    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    quality_signals: dict[str, int] = {}

    raw_source = loader.load()
    for i, row in enumerate(raw_source):
        if i >= max_samples:
            break
        errors = loader._validator.validate(row)
        if errors:
            rejected_rows.append({"url": row.get("url", ""), "errors": str(errors)})
        else:
            normalized = loader._normalizer.normalize(row)
            valid_rows.append(normalized)
            for sig_name, _ in ModelQualitySignals.check(normalized):
                quality_signals[sig_name] = quality_signals.get(sig_name, 0) + 1

    # Log summary metrics
    wandb.log(
        {
            "ingestion/n_valid": len(valid_rows),
            "ingestion/n_rejected": len(rejected_rows),
            "ingestion/n_total": len(valid_rows) + len(rejected_rows),
            **{f"ingestion/quality/{k}": v for k, v in quality_signals.items()},
        }
    )
    wandb.summary["ingestion/revision"] = provenance["revision"]
    wandb.summary["ingestion/dataset"] = provenance["dataset"]

    # Log example tables
    if valid_rows:
        valid_table = wandb.Table(
            columns=["text_preview", "url", "created_timestamp"],
            data=[[r["text"][:200], r["url"], r["created_timestamp"]] for r in valid_rows[:50]],
        )
        wandb.log({"ingestion/accepted_samples": valid_table})

    if rejected_rows:
        rejected_table = wandb.Table(
            columns=["url", "errors"],
            data=[[r["url"], r["errors"]] for r in rejected_rows[:50]],
        )
        wandb.log({"ingestion/rejected_samples": rejected_table})

    if dry_run:
        wandb.finish()
        return {"dry_run": True, "n_valid": len(valid_rows), "n_rejected": len(rejected_rows)}

    # Save artifact to disk
    out_path = pathlib.Path(artifact_path)
    out_path.mkdir(parents=True, exist_ok=True)
    import datasets as hf_datasets

    hf_datasets.Dataset.from_list(valid_rows).save_to_disk(str(out_path))

    checksum = _sha256_dir(out_path)
    manifest = ArtifactManifest(
        source_dataset_id=provenance["dataset"],
        subset=config.subset,
        revision=provenance["revision"],
        ingestion_timestamp=datetime.now(timezone.utc).isoformat(),
        loader_version="1.0",
        schema_version="1.0",
        row_count=len(valid_rows),
        artifact_checksum=checksum,
        hf_datasets_version=provenance["hf_datasets_version"],
    )
    manifest_path = out_path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest.__dict__, indent=2))

    # Log as W&B artifact with manifest attached
    artifact = wandb.Artifact(
        name=f"{config.subset.replace('/', '-')}-dataset",
        type="dataset",
        metadata={**provenance, "row_count": len(valid_rows), "checksum": checksum},
    )
    artifact.add_file(str(manifest_path), name="artifact_manifest.json")
    run.log_artifact(artifact)
    wandb.finish()

    return {
        "n_valid": len(valid_rows),
        "n_rejected": len(rejected_rows),
        "artifact_path": str(out_path),
        "checksum": checksum,
        "manifest": manifest.__dict__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest pile-of-law dataset")
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--project", default="hallucination-legal-rag")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_ingestion(
        artifact_path=args.artifact_path,
        max_samples=args.max_samples,
        project=args.project,
        wandb_mode=args.wandb_mode,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
