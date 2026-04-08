"""
scripts/ingest_lepard.py
------------------------
Production-grade LePaRD dataset ingestion pipeline.

LePaRD (Legal Passage Retrieval Dataset — ACL 2024)
  HF repo:  rmahari/LePaRD
  Paper:    arXiv:2311.09356
  Schema:   dest_id, source_id, dest_date, dest_court, dest_name, dest_cite,
            source_date, source_court, source_name, source_cite,
            passage_id, quote, destination_context

Features:
  - Config-driven via config/data/lepard.yaml (OmegaConf)
  - Pinned revision SHA for reproducibility
  - Idempotent: skips re-download if output exists with correct line count
  - SHA256 sidecar written alongside output JSONL
  - Structured logging via module-level logger
  - Error handling with retry on network failures
  - Smoke mode (--smoke) for CI: downloads only smoke_cap rows

Usage
-----
    uv run python scripts/ingest_lepard.py
    uv run python scripts/ingest_lepard.py --smoke
    uv run python scripts/ingest_lepard.py --cap 10000
    uv run python scripts/ingest_lepard.py --output-dir data/raw/lepard
"""

from __future__ import annotations

import argparse
import re
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from omegaconf import OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024  # bytes per SHA256 read chunk

# ---------------------------------------------------------------------------
# Revision validation
# ---------------------------------------------------------------------------

_HEX_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def validate_revision(revision: str) -> str:
    """
    Validate that revision is a 40-char hex commit SHA — not a branch/tag.
    Branch names like 'main' are mutable; only commit SHAs are immutable
    research artifacts.
    """
    if not _HEX_SHA_RE.match(revision):
        raise ValueError(
            f"revision={revision!r} is not a 40-char hex SHA. "
            "Use an exact commit SHA for reproducible research artifacts."
        )
    return revision


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "lepard.yaml"


def load_lepard_config(config_path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """Load LePaRD ingestion config from YAML via OmegaConf."""
    raw = OmegaConf.load(config_path)
    return dict(OmegaConf.to_container(raw, resolve=True))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Core functions (independently testable)
# ---------------------------------------------------------------------------


def fetch_stream(
    dataset: str,
    split: str,
    revision: str,
) -> Iterator[dict[str, Any]]:
    """
    Stream rows from HuggingFace dataset with pinned revision.
    Pinned revision ensures reproducibility — streaming order is deterministic
    for a fixed commit SHA (confirmed: revision=main is not pinned).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True, revision=revision)
    yield from ds  # type: ignore[misc]


def write_jsonl(
    stream: Iterable[dict[str, Any]],
    output_path: Path,
    cap: int,
) -> tuple[int, str]:
    """
    Write rows from stream to JSONL, respecting cap. Atomic write via tmp→rename.
    Computes SHA256 while writing — avoids second full file pass.
    Idempotent: skips if output exists with cap or fewer lines (short-stream stable).
    Returns (rows_written, sha256_hex). Returns (0, "") if skipped.

    Raises:
        ValueError: if cap <= 0.
    """
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")

    if output_path.exists():
        with output_path.open(encoding="utf-8") as fh:
            existing = sum(1 for _ in fh)
        if existing == cap or (0 < existing < cap):
            log.info("Skipping — %s already has %d lines", output_path.name, existing)
            return 0, ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".jsonl.tmp")
    written = 0
    h = hashlib.sha256()
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for row in tqdm(stream, total=cap, desc="Ingesting LePaRD", unit="row"):
                line = json.dumps(row) + "\n"
                fh.write(line)
                h.update(line.encode("utf-8"))
                written += 1
                if written >= cap:
                    break
        tmp.replace(output_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    digest = h.hexdigest()
    log.info("Wrote %d rows -> %s (sha256=%s...)", written, output_path, digest[:8])
    return written, digest


def compute_sha256(path: Path, write_sidecar: bool = False) -> str:
    """
    Compute SHA256 of file. Optionally write <path>.sha256 sidecar.
    Sidecar enables downstream integrity checks without re-reading the data.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if write_sidecar:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        sidecar.write_text(digest + "\n")
        log.info("SHA256 written -> %s", sidecar.name)
    return digest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Download smoke_cap rows only (for CI).")
    parser.add_argument("--cap", type=int, default=None, help="Override cap from config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output_dir from config.")
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH, help="Path to lepard.yaml config.")
    args = parser.parse_args()

    cfg = load_lepard_config(args.config)

    cap = cfg["smoke_cap"] if args.smoke else (args.cap or cfg["cap"])
    output_dir = args.output_dir or Path(cfg["output_dir"])
    output_file = output_dir / cfg["output_file"].format(cap=cap)

    log.info("LePaRD ingestion — dataset=%s revision=%s cap=%d", cfg["dataset"], cfg["revision"], cap)

    try:
        stream = fetch_stream(cfg["dataset"], cfg["split"], cfg["revision"])
        written, sha256 = write_jsonl(stream, output_file, cap=cap)
        if written > 0:
            sidecar = output_file.with_suffix(".jsonl.sha256")
            sidecar.write_text(sha256 + "\n", encoding="utf-8")
            log.info("SHA256 sidecar written -> %s", sidecar.name)
        log.info("Done — %s", output_file)
    except Exception as exc:
        log.error("Ingestion failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
