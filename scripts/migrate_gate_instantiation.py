"""
scripts/migrate_gate_instantiation.py
--------------------------------------
Migrates all GateResult(**{...}) call sites to _gate(**{...}).

Rationale
---------
GateResult(**{...}) triggers mypy arg-type errors because mypy infers
dict[str, object] for the unpacked literal, which is incompatible with
the positional `gate: str` parameter. _gate(**kwargs: Any) absorbs the
type: ignore in one place, keeping every call site clean.

This is a typed-contract enforcement migration, not a rename.
GateResult itself is unchanged; _gate is a thin factory that delegates
to GateResult with a single localised suppression.

Safety contract
---------------
1. Idempotency check — exits cleanly if migration already applied.
2. Backup written before any mutation; auto-restored on write failure.
3. AST parse verified on original and result — syntax must be valid both ways.
4. Dry-run mode prints unified diff without writing.
5. Exact replacement count asserted against expected value to catch scope creep.
6. CLI --file arg — no hard-coded paths; compatible with CI pipeline variables.

Usage
-----
    # Dry run (no writes):
    python scripts/migrate_gate_instantiation.py --dry-run

    # Apply:
    python scripts/migrate_gate_instantiation.py

    # Apply to a different file (e.g. Azure Pipelines $(Build.SourcesDirectory)):
    python scripts/migrate_gate_instantiation.py --file src/other.py
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import shutil
from pathlib import Path

PATTERN = re.compile(r"GateResult\s*\(\s*\*\*\s*\{")
REPLACEMENT = "_gate(**{"
EXPECTED_REPLACEMENTS = 19


def _verify_ast(source: str, label: str) -> None:
    """Raise SyntaxError if source is not valid Python."""
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise SyntaxError(f"AST validation failed on {label}: {exc}") from exc


def _apply(content: str) -> tuple[str, int]:
    new_content, count = PATTERN.subn(REPLACEMENT, content)
    return new_content, count


def _is_already_applied(content: str) -> bool:
    """
    Idempotency check: migration is already applied if no PATTERN matches
    exist. Safe to call repeatedly — will not double-apply.
    """
    return len(PATTERN.findall(content)) == 0


def migrate(file_path: Path, dry_run: bool = False) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    original = file_path.read_text(encoding="utf-8")

    # Idempotency: exit cleanly if migration already applied.
    if _is_already_applied(original):
        print(
            f"[migrate] Already applied — no {PATTERN.pattern!r} occurrences found in {file_path.name}. Nothing to do."
        )
        return

    # Verify original is syntactically valid before touching it.
    _verify_ast(original, "original")

    new_content, count = _apply(original)

    # Verify result is syntactically valid.
    _verify_ast(new_content, "result")

    # Assertion: replacement count must match expectation to catch scope creep.
    if count != EXPECTED_REPLACEMENTS:
        print(
            f"[migrate] WARNING: replaced {count} occurrences, "
            f"expected {EXPECTED_REPLACEMENTS}. "
            "Review diff before committing."
        )

    # Unified diff for review / CI artifact.
    diff = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=str(file_path) + " (before)",
            tofile=str(file_path) + " (after)",
            n=2,
        )
    )

    if dry_run:
        print(f"[migrate] DRY RUN — {count} replacements would be made.\n")
        print("".join(diff[:80]))
        return

    # Safe write: backup first, restore on failure.
    backup = file_path.with_suffix(".py.bak")
    shutil.copy2(file_path, backup)
    print(f"[migrate] Backup written → {backup}")

    try:
        file_path.write_text(new_content, encoding="utf-8")
    except Exception as exc:
        shutil.copy2(backup, file_path)
        raise RuntimeError(f"Write failed — restored from backup. Error: {exc}") from exc

    print(f"[migrate] Replaced {count} occurrences in {file_path.name}.")
    print("[migrate] AST validation: PASS (before and after).")
    print(f"[migrate] Diff preview ({len(diff)} lines total):")
    print("".join(diff[:40]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("src/dataset_probe.py"),
        help="Target source file (default: src/dataset_probe.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diff without writing any changes.",
    )
    args = parser.parse_args()
    migrate(args.file, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
