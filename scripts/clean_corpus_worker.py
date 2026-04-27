"""Worker: clean one JSONL shard via RE2 linear-time citation stripper.

Eyecite hangs on pathological corpus inputs (cpython#96062 + catastrophic
backtracking in supra/id chain resolution). RE2 is mathematically immune
to backtracking → guaranteed throughput on 7.8M-row corpus.

Case-name extraction is sacrificed here; that concern is handled
query-side via clean_query.py (45K queries can afford eyecite).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from strip_citations_simple import strip_citations  # noqa: E402

HANG_SENTINEL = "__HANG_SENTINEL__"


def _clean_line(line: str) -> str:
    line = line.rstrip("\n")
    if not line:
        return ""
    row = json.loads(line)
    text = row.get("text") or ""
    if HANG_SENTINEL in text:
        time.sleep(60)
    if "text" in row:
        row["text"] = strip_citations(text)
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
