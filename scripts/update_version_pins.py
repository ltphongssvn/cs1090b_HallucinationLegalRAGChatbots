"""
scripts/update_version_pins.py
------------------------------
Surgically replaces old transformers/tokenizers version references
with verified pinned versions across documentation files.

Pinned (verified 2026-04):
  transformers==4.41.2  — minimum for Mistral/LLaMA tokenizer schema
  tokenizers==0.19.1    — adds PyPreTokenizerTypeWrapper support

Usage
-----
    python scripts/update_version_pins.py --dry-run
    python scripts/update_version_pins.py
"""

from __future__ import annotations

import argparse
import difflib
import shutil
from pathlib import Path

REPLACEMENTS: list[tuple[str, str]] = [
    ("transformers>=4.35,<4.41", "transformers==4.41.2"),
    ("transformers>=4.35,<4.42", "transformers==4.41.2"),
    ("transformers==4.39.3", "transformers==4.41.2"),
    ("transformers==4.37.0", "transformers==4.41.2"),
    ("tokenizers==0.15.2", "tokenizers==0.19.1"),
    ("tokenizers>=0.19.0", "tokenizers==0.19.1"),
]

TARGET_FILES: list[str] = ["README.md"]


def patch_file(path: Path, dry_run: bool) -> int:
    original = path.read_text(encoding="utf-8")
    content = original
    for old, new in REPLACEMENTS:
        content = content.replace(old, new)
    if content == original:
        return 0
    diff = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=str(path) + " (before)",
            tofile=str(path) + " (after)",
            n=2,
        )
    )
    print("".join(diff[:40]))
    if not dry_run:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text(content, encoding="utf-8")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--files", nargs="+", default=TARGET_FILES)
    args = parser.parse_args()

    mode = "DRY RUN" if args.dry_run else "PATCHING"
    print(f"[update_version_pins] {mode} ...")
    total = 0
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"  SKIP (not found): {f}")
            continue
        n = patch_file(path, dry_run=args.dry_run)
        if n:
            print(f"  patched: {f}")
            total += n
    print(f"[update_version_pins] Total files patched: {total}")


if __name__ == "__main__":
    main()
