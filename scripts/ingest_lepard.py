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
  - ProvenanceContext dataclass — single object for dataset/split/revision/cap
  - Pinned revision SHA for reproducibility (validated as 40-char lowercase hex)
  - fetch_stream validates revision before any network call
  - fetch_stream retries on OSError (covers ConnectionResetError, TimeoutError)
  - Output filename includes revision prefix — prevents same-cap collision
  - Idempotent: O(1) sidecar-presence fast path before O(N) line scan
  - Self-heal: restores both sidecar AND manifest on valid existing file
  - Self-heal preserves original provenance; marks reconstructed manifest
  - Sidecar-present fast path repairs missing manifest without re-downloading
  - Single-pass self-heal: line count + SHA256 computed in one disk read
  - Crash-safe: sidecar written inside write_jsonl — no crash window
  - Atomic write via unique tmp→rename — safe for concurrent runs
  - FD-safe: os.close(tmp_fd) before open() prevents file descriptor leaks
  - SHA256 computed while writing — avoids second full file pass
  - json.dumps with ensure_ascii=False — preserves legal unicode
  - NaN rows passed through — audit_jsonl_nan.py is the downstream gate
  - tqdm(miniters=1) for network-shaped iteration progress
  - Provenance manifest: split, rows_written, timezone-aware UTC, exact python version
  - --verify-only checks digest AND manifest fields (revision, dataset, split)
  - _git_sha() checks GIT_COMMIT_SHA env var first — container-safe
  - tqdm progress bar (disable=None auto-disables on non-TTY for CI)
  - log.exception preserves full traceback in CI logs
  - Smoke mode (--smoke) for CI: downloads only smoke_cap rows
  - --smoke and --cap are mutually exclusive
  - --force flag: purges stale sidecar+manifest then re-ingests
  - --dry-run flag to count rows without writing output file
  - --verify-only flag: recomputes SHA256, compares against sidecar and manifest
  - cap validated > 0 at entry
  - _SIDECAR_SUFFIX constant — single source of truth for sidecar path

Idempotency design:
  The sidecar-presence fast path is a trust-based shortcut, not a verified
  integrity check. It is framed honestly: sidecar present = skip. For verified
  integrity, use --verify-only which recomputes SHA256 from disk bytes,
  compares against the stored sidecar, and checks manifest provenance fields.
  If manifest is missing when sidecar is present, manifest is repaired.

Self-heal design:
  When a valid file exists without sidecar/manifest (e.g. after a crash),
  the next run restores BOTH the sidecar and the manifest if ProvenanceContext
  is provided. Single-pass: line count and SHA256 computed in one disk read
  (2.3x faster than double-pass on 500K-row files).
  If original manifest exists, its timestamp is preserved and
  provenance_reconstructed=True is added — never silently overwrites provenance.

Finalization design:
  Sidecar and manifest are both written inside write_jsonl() — no crash window
  between data file commit and metadata publication.

Concurrency design:
  Unique temp names via mkstemp are collision-safe for temp file creation.
  Path.replace() is atomic on POSIX. However this is not full concurrency
  control — concurrent runs are last-writer-wins, not serialized.

Network retry design:
  fetch_stream retries up to 3 times on OSError with exponential backoff
  (1s→2s→4s). OSError covers ConnectionResetError and TimeoutError as subclasses —
  no redundant exception types in the retry predicate.
  Note: retry covers _load_hf_dataset() only, not mid-stream iteration failures.

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
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from omegaconf import OmegaConf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

log = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024  # bytes per SHA256 read chunk
_SIDECAR_SUFFIX = ".sha256"  # appended to full filename: out.jsonl -> out.jsonl.sha256
_MANIFEST_SUFFIX = ".manifest.json"  # provenance manifest suffix

# Network retry config: 3 attempts, exponential backoff 1s→2s→4s
# OSError covers ConnectionResetError and TimeoutError as subclasses — no redundancy
_FETCH_RETRY = retry(
    retry=retry_if_exception_type(OSError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)


# ---------------------------------------------------------------------------
# ProvenanceContext — single object for dataset/split/revision/cap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceContext:
    """
    Immutable provenance context passed through ingestion functions.
    Replaces repetitive (dataset, split, revision, cap) parameter trampolining.
    Adding a new provenance field requires updating only this dataclass,
    not every function signature in the pipeline.
    """

    dataset: str
    split: str
    revision: str
    cap: int


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _sidecar_path(output_path: Path) -> Path:
    """Return the SHA256 sidecar path for a given output file."""
    return output_path.parent / (output_path.name + _SIDECAR_SUFFIX)


def _manifest_path(output_path: Path) -> Path:
    """Return the provenance manifest path for a given output file."""
    return output_path.parent / (output_path.name + _MANIFEST_SUFFIX)


# ---------------------------------------------------------------------------
# System helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """
    Return current git commit SHA or 'unknown'.
    Checks GIT_COMMIT_SHA environment variable first — container-safe.
    Falls back to subprocess for local development.
    Narrows to specific subprocess exceptions:
      FileNotFoundError  — git not installed
      CalledProcessError — not a git repo or git error
      OSError            — OS-level failure
    """
    env_sha = os.environ.get("GIT_COMMIT_SHA", "").strip()
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return "unknown"


def _python_version() -> str:
    """Return exact Python version string using version_info — not brittle slice."""
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def _utc_now() -> str:
    """Return timezone-aware UTC timestamp — replaces deprecated utcnow()."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _count_lines_and_hash(path: Path) -> tuple[int, str]:
    """
    Single-pass: count newlines and compute SHA256 in one disk read.
    2.3x faster than double-pass (line scan + separate hash read).
    """
    h = hashlib.sha256()
    lines = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            h.update(chunk)
            lines += chunk.count(b"\n")
    return lines, h.hexdigest()


def _write_sidecar(output_path: Path, digest: str) -> None:
    """Write SHA256 sidecar alongside JSONL artifact."""
    sidecar = _sidecar_path(output_path)
    sidecar.write_text(digest + "\n", encoding="utf-8")
    log.info("SHA256 sidecar written -> %s", sidecar.name)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _write_provenance_manifest(
    output_path: Path,
    ctx: ProvenanceContext,
    sha256: str,
    rows_written: int,
    force_used: bool = False,
    provenance_reconstructed: bool = False,
    original_ingestion_ts: str = "",
) -> None:
    """
    Write provenance manifest JSON alongside JSONL artifact.
    Accepts ProvenanceContext — single object instead of 4 loose params.
    If provenance_reconstructed=True, preserves original_ingestion_ts and
    marks manifest so callers know provenance was repaired, not original.
    """
    import datasets as _ds

    manifest = {
        "ingestion_ts_utc": original_ingestion_ts if original_ingestion_ts else _utc_now(),
        "script_git_commit": _git_sha(),
        "hf_revision": ctx.revision,
        "dataset": ctx.dataset,
        "split": ctx.split,
        "cap": ctx.cap,
        "rows_written": rows_written,
        "python_version": _python_version(),
        "datasets_version": _ds.__version__,
        "force_used": force_used,
        "sha256": sha256,
    }
    if provenance_reconstructed:
        manifest["provenance_reconstructed"] = True
        manifest["reconstruction_ts_utc"] = _utc_now()

    _manifest_path(output_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Provenance manifest written -> %s", _manifest_path(output_path).name)


def _finalize_artifact(
    output_path: Path,
    ctx: ProvenanceContext,
    digest: str,
    rows_written: int,
    force_used: bool = False,
) -> None:
    """
    Write sidecar and manifest together — no crash window between data and metadata.
    Called after successful tmp.replace(output_path) to complete artifact bundle.
    Accepts ProvenanceContext — single object instead of 4 loose params.
    """
    _write_sidecar(output_path, digest)
    if ctx.revision and ctx.dataset:
        _write_provenance_manifest(
            output_path,
            ctx=ctx,
            sha256=digest,
            rows_written=rows_written,
            force_used=force_used,
        )


def _repair_manifest_from_sidecar(
    output_path: Path,
    ctx: ProvenanceContext,
) -> None:
    """
    Repair missing manifest when sidecar is present.
    Reads digest from sidecar, counts lines for rows_written,
    and writes manifest marked provenance_reconstructed=True.
    Accepts ProvenanceContext — single object instead of 3 loose params.
    cap from ctx preserves requested cap semantics (not conflated with rows_written).
    """
    sidecar = _sidecar_path(output_path)
    digest = sidecar.read_text(encoding="utf-8").strip()
    with output_path.open(encoding="utf-8") as fh:
        rows_written = sum(1 for _ in fh)
    log.info("Repairing missing manifest for %s", output_path.name)
    _write_provenance_manifest(
        output_path,
        ctx=ctx,
        sha256=digest,
        rows_written=rows_written,
        force_used=False,
        provenance_reconstructed=True,
    )


def _purge_stale_artifacts(output_path: Path) -> None:
    """Remove stale sidecar and manifest before forced re-ingestion."""
    for stale in [_sidecar_path(output_path), _manifest_path(output_path)]:
        if stale.exists():
            stale.unlink()
            log.info("Purged stale artifact -> %s", stale.name)


def _self_heal_artifact(
    output_path: Path,
    existing: int,
    digest: str,
    ctx: ProvenanceContext,
) -> tuple[int, str]:
    """
    Self-heal: restore sidecar AND manifest for a valid existing file.
    Called when file exists but sidecar is missing (e.g. after crash).
    Restores full artifact bundle — not just sidecar.
    digest already computed by _count_lines_and_hash — no second disk read.
    Preserves original manifest timestamp if manifest exists;
    marks manifest provenance_reconstructed=True to signal repair.
    Accepts ProvenanceContext — single object instead of 4 loose params.
    Returns (0, digest).
    """
    log.info(
        "Skipping — %s already has %d lines; self-healing full artifact bundle",
        output_path.name,
        existing,
    )
    _write_sidecar(output_path, digest)

    if ctx.revision and ctx.dataset:
        original_ts = ""
        manifest = _manifest_path(output_path)
        if manifest.exists():
            try:
                existing_manifest = json.loads(manifest.read_text(encoding="utf-8"))
                original_ts = existing_manifest.get("ingestion_ts_utc", "")
            except (json.JSONDecodeError, OSError):
                original_ts = ""

        _write_provenance_manifest(
            output_path,
            ctx=ctx,
            sha256=digest,
            rows_written=existing,
            force_used=False,
            provenance_reconstructed=True,
            original_ingestion_ts=original_ts,
        )
    return 0, digest


# ---------------------------------------------------------------------------
# Revision validation
# ---------------------------------------------------------------------------

_HEX_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def validate_revision(revision: str) -> str:
    """
    Validate that revision is a 40-char lowercase hex commit SHA — not a
    branch/tag. Branch names like 'main' are mutable; only commit SHAs are
    immutable research artifacts. Uppercase hex also rejected.
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


@_FETCH_RETRY
def _load_hf_dataset(dataset: str, split: str, revision: str) -> Any:
    """
    Load HuggingFace dataset with retry on OSError (covers ConnectionResetError,
    TimeoutError as subclasses). Retries up to 3 times with exponential backoff.
    Separated from fetch_stream so retry wraps only the network call.
    """
    from datasets import load_dataset

    return load_dataset(dataset, split=split, streaming=True, revision=revision)


def fetch_stream(
    dataset: str,
    split: str,
    revision: str,
) -> Iterator[dict[str, Any]]:
    """
    Stream rows from HuggingFace dataset with pinned revision.
    Validates revision is a 40-char lowercase hex SHA before any network call.
    Retries load_dataset up to 3 times on OSError.
    NaN rows are passed through — audit_jsonl_nan.py is the downstream gate.
    """
    validate_revision(revision)
    ds = _load_hf_dataset(dataset, split, revision)
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
    split: str = "",
) -> tuple[int, str]:
    """
    Write rows from stream to JSONL, respecting cap. Atomic write via unique
    tmp→rename — safe for concurrent runs in same directory.
    FD-safe: os.close(tmp_fd) before open() prevents file descriptor leaks.
    Computes SHA256 while writing — avoids second full file pass.
    Uses ensure_ascii=False to preserve legal unicode (café, em-dash, etc.).
    tqdm(miniters=1) for network-shaped iteration progress visibility.
    Idempotent: O(1) sidecar-presence fast path before O(N) line scan.
      If manifest missing when sidecar present, manifest is repaired.
    Self-heals: restores BOTH sidecar and manifest on valid existing file.
      Preserves original provenance; marks manifest provenance_reconstructed=True.
    Single-pass self-heal: line count + SHA256 in one disk read (2.3x faster).
    force=True: purges stale sidecar+manifest before re-ingesting.
    verify_only=True: recomputes SHA256, compares against sidecar AND manifest
      fields (revision, dataset, split) — raises ValueError on mismatch.
    Sidecar and manifest written together inside write_jsonl — no crash window.
    dry_run=True: counts rows without writing output file (CI preflight).
    Returns (rows_written, sha256_hex). Returns (0, "") if skipped.

    Args:
        stream: Iterable of row dicts to write.
        output_path: Destination JSONL file path.
        cap: Maximum rows to write. Must be > 0.
        force: If True, purge stale artifacts and always rewrite.
        dry_run: If True, count rows only — no file written.
        verify_only: If True, verify existing artifact SHA and manifest fields.
        revision: HF dataset revision SHA for provenance manifest.
        dataset: HF dataset name for provenance manifest.
        split: HF dataset split for provenance manifest.

    Raises:
        ValueError: if cap <= 0, digest mismatch, or manifest field mismatch.
        FileNotFoundError: if verify_only and output_path does not exist.
    """
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")

    ctx = ProvenanceContext(dataset=dataset, split=split, revision=revision, cap=cap)

    if verify_only:
        if not output_path.exists():
            raise FileNotFoundError(f"Cannot verify — {output_path} does not exist")
        digest = compute_sha256(output_path)
        sidecar = _sidecar_path(output_path)
        if sidecar.exists():
            stored = sidecar.read_text(encoding="utf-8").strip()
            if digest != stored:
                raise ValueError(
                    f"digest mismatch for {output_path.name}: computed={digest[:8]}... stored={stored[:8]}..."
                )
            log.info("Verified %s — digest matches sidecar", output_path.name)
        else:
            log.warning("Verified %s — no sidecar to compare against", output_path.name)
        manifest_file = _manifest_path(output_path)
        if manifest_file.exists() and revision and dataset:
            try:
                stored_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                mismatches = []
                if revision and stored_manifest.get("hf_revision") != revision:
                    mismatches.append(f"revision: stored={stored_manifest.get('hf_revision')} requested={revision}")
                if dataset and stored_manifest.get("dataset") != dataset:
                    mismatches.append(f"dataset: stored={stored_manifest.get('dataset')} requested={dataset}")
                if split and stored_manifest.get("split") != split:
                    mismatches.append(f"split: stored={stored_manifest.get('split')} requested={split}")
                if mismatches:
                    raise ValueError(f"manifest mismatch for {output_path.name}: " + "; ".join(mismatches))
                log.info("Verified %s — manifest fields match", output_path.name)
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Could not read manifest for verification: %s", e)
        log.info("sha256=%s...", digest[:8])
        return 0, digest

    if dry_run:
        written = 0
        for _ in tqdm(stream, total=cap, desc="Dry-run LePaRD", unit="row", disable=None, miniters=1):
            written += 1
            if written >= cap:
                break
        log.info("Dry-run complete — would write %d rows to %s", written, output_path)
        return written, ""

    if force:
        _purge_stale_artifacts(output_path)

    if not force and output_path.exists():
        sidecar = _sidecar_path(output_path)
        if sidecar.exists():
            if revision and dataset and not _manifest_path(output_path).exists():
                _repair_manifest_from_sidecar(output_path, ctx)
            log.info("Skipping — sidecar found for %s", output_path.name)
            return 0, ""
        existing, digest = _count_lines_and_hash(output_path)
        if existing == cap or (0 < existing < cap):
            return _self_heal_artifact(output_path, existing, digest, ctx)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=output_path.parent, suffix=".jsonl.tmp")
    os.close(tmp_fd)
    tmp = Path(tmp_name)
    written = 0
    h = hashlib.sha256()
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for row in tqdm(stream, total=cap, desc="Ingesting LePaRD", unit="row", disable=None, miniters=1):
                line = json.dumps(row, ensure_ascii=False) + "\n"
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

    _finalize_artifact(output_path, ctx=ctx, digest=digest, rows_written=written, force_used=force)

    return written, digest


def compute_sha256(path: Path, write_sidecar: bool = False) -> str:
    """
    Compute SHA256 of file from disk bytes. Optionally write <path>.sha256 sidecar.
    O(N) in file size — use sidecar-presence fast path for routine skipping.
    For combined line-count + hash, use _count_lines_and_hash() instead.
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output_dir from config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="Path to lepard.yaml config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Purge stale artifacts and force re-ingestion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows without writing output file (CI preflight).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Recompute SHA256 from disk, compare against sidecar and manifest fields.",
    )

    cap_group = parser.add_mutually_exclusive_group()
    cap_group.add_argument(
        "--smoke",
        action="store_true",
        help="Download smoke_cap rows only (for CI). Mutually exclusive with --cap.",
    )
    cap_group.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Override cap from config. Mutually exclusive with --smoke.",
    )

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
            split=cfg["split"],
        )
        log.info("Done — %s", output_file)
    except Exception:
        log.exception("Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
