"""Regenerate LePaRD sidecar (.sha256) and manifest (.manifest.json) from disk bytes.

Avoids re-downloading 5.78 GB from HuggingFace when artifacts already exist locally
but sidecar/manifest were lost. Idempotent.

Usage:
    .venv/bin/python scripts/regenerate_lepard_sidecar.py
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


JSONL_NAME = "lepard_train_4000000_rev0194f95.jsonl"
HF_DATASET = "rmahari/LePaRD"
HF_REVISION = "0194f95c3091acceab3b887c9b09ef432cf84052"  # pragma: allowlist secret
CAP = 4_000_000


def main() -> None:
    jsonl = Path(JSONL_NAME)
    if not jsonl.is_file():
        raise SystemExit(f"missing {jsonl} — re-run scripts/ingest_lepard.py to download")

    print(f"Computing SHA256 of {jsonl} ({jsonl.stat().st_size / 1e9:.2f} GB) ...")
    h = hashlib.sha256()
    with jsonl.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    print(f"  sha256: {digest}")

    sidecar = jsonl.with_suffix(".jsonl.sha256")
    sidecar.write_text(f"{digest}  {jsonl.name}\n", encoding="utf-8")
    print(f"  Wrote {sidecar}")

    print(f"Counting lines in {jsonl} ...")
    n_lines = 0
    with jsonl.open("rb") as f:
        for _ in f:
            n_lines += 1
    print(f"  n_lines: {n_lines:,}")

    manifest = {
        "dataset": HF_DATASET,
        "revision": HF_REVISION,
        "cap": CAP,
        "path": jsonl.name,
        "size_bytes": jsonl.stat().st_size,
        "sha256": digest,
        "n_lines": n_lines,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_reconstructed": True,
    }
    manifest_path = jsonl.with_suffix(".jsonl.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  Wrote {manifest_path}")

    print("\nOK sidecar + manifest regenerated")


if __name__ == "__main__":
    main()
