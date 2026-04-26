"""Worker: clean one JSONL shard. Invoked as `python scripts/clean_corpus_worker.py IN OUT`.

Standalone subprocess design (per cpython#96062): mp.Pool deadlocks when
workers hang. Each shard runs as independent OS process; orchestrator
uses subprocess.run(timeout=N) for clean kill semantics.

Test-only: rows containing "__HANG_SENTINEL__" trigger time.sleep(60) so
TDD can simulate a hung worker.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clean_query import clean_destination_context  # noqa: E402

HANG_SENTINEL = "__HANG_SENTINEL__"


def _clean_line(line: str) -> str:
    line = line.rstrip("\n")
    if not line:
        return ""
    row = json.loads(line)
    text = row.get("text") or ""
    # TDD hook: simulate hung worker
    if HANG_SENTINEL in text:
        time.sleep(60)
    if "text" in row:
        row["text"] = clean_destination_context(text)
    return json.dumps(row)


def main(in_path: Path, out_path: Path) -> int:
    n = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            cleaned = _clean_line(line)
            if cleaned:
                fout.write(cleaned + "\n")
                n += 1
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: clean_corpus_worker.py IN.jsonl OUT.jsonl", file=sys.stderr)
        sys.exit(2)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
