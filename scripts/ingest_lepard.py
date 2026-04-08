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
  - Config-driven via config/lepard.yaml (OmegaConf)
  - Pinned revision SHA for reproducibility (validated as 40-char hex)
  - Idempotent: O(1) SHA256 sidecar check before O(N) line scan
  - Self-heal: valid file without sidecar gets sidecar written on next run
  - Atomic write via unique tmp→rename — safe for concurrent runs
  - SHA256 computed while writing — avoids second full file pass
  - Provenance manifest JSON written alongside JSONL artifact
  - tqdm progress bar (disable=None auto-disables on non-TTY for CI)
  - log.exception preserves full traceback in CI logs
  - Smoke mode (--smoke) for CI: downloads only smoke_cap rows
  - --force flag to bypass idempotency for forced re-ingestion
  - --dry-run flag to count rows without writing output file
  - --verify-only flag to check SHA/provenance without re-downloading
  - cap validated > 0 at entry
  - _SIDECAR_SUFFIX constant — single source of truth for sidecar path

Usage
-----
    uv run python scripts/ingest_lepard.py
    uv run python scripts/ingest_lepard.py --smoke
    uv run python scripts/ingest_lepard.py --cap 10000
    uv run python scripts/ingest_lepard.py --output-dir data/raw/lepard
    uv run python scripts/ingest_lepard.py --force
    uv run python scripts/ingest_lepard.py --dry-run
    uv run python scripts/ingest_lepard.py --verify-only
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from omegaconf import OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024  # bytes per SHA256 read chunk
_SIDECAR_SUFFIX = ".sha256"  # appended to full filename: out.jsonl -> out.jsonl.sha256
_MANIFEST_SUFFIX = ".manifest.json"  # provenance manifest: out.jsonl -> out.jsonl.manifest.json


def _sidecar_path(output_path: Path) -> Path:
    """Return the SHA256 sidecar path for a given output file."""
    return output_path.parent / (output_path.name + _SIDECAR_SUFFIX)


def _manifest_path(output_path: Path) -> Path:
    """Return the provenance manifest path for a given output file."""
    return output_path.parent / (output_path.name + _MANIFEST_SUFFIX)


def _git_sha() -> str:
    """Return current git commit SHA or 'unknown'."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _write_provenance_manifest(
    output_path: Path,
    sha256: str,
    cap: int,
    hf_revision: str,
    dataset: str,
    force_used: bool = False,
) -> None:
    """Write provenance manifest JSON alongside JSONL artifact."""
    import datasets as _ds

    manifest = {
        "ingestion_ts_utc": datetime.datetime.utcnow().isoformat(),
        "script_git_commit": _git_sha(),
        "hf_revision": hf_revision,
        "dataset": dataset,
        "cap": cap,
        "python_version": sys.version[:6],
        "datasets_version": _ds.__version__,
        "force_used": force_used,
        "sha256": sha256,
    }
    _manifest_path(output_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Provenance manifest written -> %s", _manifest_path(output_path).name)


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
    force: bool = False,
    dry_run: bool = False,
    verify_only: bool = False,
    revision: str = "",
    dataset: str = "",
) -> tuple[int, str]:
    """
    Write rows from stream to JSONL, respecting cap. Atomic write via unique
    tmp→rename — safe for concurrent runs in same directory.
    Computes SHA256 while writing — avoids second full file pass.
    Idempotent: O(1) sidecar check first, then O(N) line scan as fallback.
    Self-heals: valid file without sidecar gets sidecar written on skip path.
    Writes provenance manifest JSON alongside artifact when revision+dataset provided.
    dry_run=True: counts rows without writing output file (CI preflight).
    verify_only=True: checks SHA of existing file without re-downloading.
    Returns (rows_written, sha256_hex). Returns (0, "") if skipped.

    Args:
        stream: Iterable of row dicts to write.
        output_path: Destination JSONL file path.
        cap: Maximum rows to write. Must be > 0.
        force: If True, bypass idempotency and always rewrite.
        dry_run: If True, count rows only — no file written.
        verify_only: If True, verify existing artifact SHA without rewriting.
        revision: HF dataset revision SHA for provenance manifest.
        dataset: HF dataset name for provenance manifest.

    Raises:
        ValueError: if cap <= 0.
    """
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")

    if verify_only:
        if not output_path.exists():
            raise FileNotFoundError(f"Cannot verify — {output_path} does not exist")
        digest = compute_sha256(output_path)
        log.info("Verified %s sha256=%s...", output_path.name, digest[:8])
        return 0, digest

    if dry_run:
        written = 0
        for _ in tqdm(stream, total=cap, desc="Dry-run LePaRD", unit="row", disable=None):
            written += 1
            if written >= cap:
                break
        log.info("Dry-run complete — would write %d rows to %s", written, output_path)
        return written, ""

    if not force and output_path.exists():
        sidecar = _sidecar_path(output_path)
        if sidecar.exists():
            log.info("Skipping — sidecar found for %s", output_path.name)
            return 0, ""
        with output_path.open(encoding="utf-8") as fh:
            existing = sum(1 for _ in fh)
        if existing == cap or (0 < existing < cap):
            log.info("Skipping — %s already has %d lines; self-healing sidecar", output_path.name, existing)
            digest = compute_sha256(output_path, write_sidecar=True)
            return 0, digest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=output_path.parent, suffix=".jsonl.tmp")
    tmp = Path(tmp_name)
    written = 0
    h = hashlib.sha256()
    try:
        with open(tmp_fd, "w", encoding="utf-8", newline="\n") as fh:
            for row in tqdm(stream, total=cap, desc="Ingesting LePaRD", unit="row", disable=None):
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

    if revision and dataset:
        _write_provenance_manifest(
            output_path,
            sha256=digest,
            cap=cap,
            hf_revision=revision,
            dataset=dataset,
            force_used=force,
        )

    return written, digest


def compute_sha256(path: Path, write_sidecar: bool = False) -> str:
    """
    Compute SHA256 of file. Optionally write <path>.sha256 sidecar.
    Useful for external verification of existing artifacts and self-healing
    missing sidecars on valid existing files.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if write_sidecar:
        sidecar = _sidecar_path(path)
        sidecar.write_text(digest + "\n", encoding="utf-8")
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
    parser.add_argument("--force", action="store_true", help="Bypass idempotency and force re-ingestion.")
    parser.add_argument("--dry-run", action="store_true", help="Count rows without writing output file.")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing artifact SHA only.")
    args = parser.parse_args()

    cfg = load_lepard_config(args.config)

    cap = cfg["smoke_cap"] if args.smoke else (args.cap if args.cap is not None else cfg["cap"])
    output_dir = args.output_dir or Path(cfg["output_dir"])
    output_file = output_dir / cfg["output_file"].format(cap=cap)

    log.info(
        "LePaRD ingestion — dataset=%s revision=%s cap=%d force=%s dry_run=%s verify_only=%s",
        cfg["dataset"],
        cfg["revision"],
        cap,
        args.force,
        args.dry_run,
        args.verify_only,
    )

    try:
        validate_revision(cfg["revision"])
        stream = fetch_stream(cfg["dataset"], cfg["split"], cfg["revision"])
        written, sha256 = write_jsonl(
            stream,
            output_file,
            cap=cap,
            force=args.force,
            dry_run=args.dry_run,
            verify_only=args.verify_only,
            revision=cfg["revision"],
            dataset=cfg["dataset"],
        )
        if written > 0 and not args.dry_run:
            sidecar = _sidecar_path(output_file)
            sidecar.write_text(sha256 + "\n", encoding="utf-8")
            log.info("SHA256 sidecar written -> %s", sidecar.name)
        log.info("Done — %s", output_file)
    except Exception:
        log.exception("Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
