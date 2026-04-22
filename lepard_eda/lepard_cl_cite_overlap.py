"""
LePaRD <-> CourtListener citation-string overlap analysis
==========================================================

Checks document-level overlap WITHOUT downloading the full opinion text corpus.

Method
------
1. Sample LePaRD pairs and parse source_cite strings with eyecite
   → extract (volume, reporter, page) tuples
2. Load CourtListener citations.csv.bz2 (~127 MB download, not the full corpus)
   → build a (volume, reporter, page) → cluster_id lookup
3. Join: what % of LePaRD source citations can be resolved to a CL cluster_id?

This answers: "do the cases LePaRD references actually exist in CourtListener?"
at the citation-string level, not the raw-ID level.

Usage
-----
  python lepard_eda/lepard_cl_cite_overlap.py
  python lepard_eda/lepard_cl_cite_overlap.py --sample 50000
  python lepard_eda/lepard_cl_cite_overlap.py --full-lepard   # all 22M rows
  python lepard_eda/lepard_cl_cite_overlap.py --out lepard_eda/cite_overlap_results.jsonl
"""

from __future__ import annotations

import argparse
import bz2
import csv
import io
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = REPO_ROOT / "lepard_eda"
LEPARD_DISK = EDA_DIR / "lepard_data"
CL_BULK_DIR = EDA_DIR / "cl_bulk"
DEFAULT_CITATIONS_BZ2 = CL_BULK_DIR / "citations-2026-03-31.csv.bz2"
CL_CITATIONS_URL = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/citations-2026-03-31.csv.bz2"

DEFAULT_SAMPLE = 10_000

# Federal appellate court keywords for filtering analysis
FEDERAL_APPELLATE_KEYWORDS = [
    "first circuit", "second circuit", "third circuit", "fourth circuit",
    "fifth circuit", "sixth circuit", "seventh circuit", "eighth circuit",
    "ninth circuit", "tenth circuit", "eleventh circuit",
    "district of columbia circuit", "d.c. circuit", "federal circuit",
    "court of appeals",
]

# ---------------------------------------------------------------------------
# Step 1: Download citations.csv.bz2 if not cached
# ---------------------------------------------------------------------------

def ensure_citations_file(path: Path, url: str) -> Path:
    if path.exists():
        print(f"  [cache] Using existing CL citations file: {path} ({path.stat().st_size/1e6:.1f} MB)")
        return path
    print(f"  Downloading CL citations CSV from S3...")
    print(f"  URL: {url}")
    print(f"  Destination: {path}")
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    last_pct = -1
    def reporthook(count, block_size, total_size):
        nonlocal last_pct
        if total_size <= 0:
            return
        pct = int(count * block_size * 100 / total_size)
        if pct != last_pct and pct % 10 == 0:
            elapsed = time.time() - t0
            print(f"    {pct}% ({elapsed:.0f}s)", file=sys.stderr)
            last_pct = pct
    urllib.request.urlretrieve(url, path, reporthook)
    print(f"  Downloaded in {time.time()-t0:.1f}s ({path.stat().st_size/1e6:.1f} MB)")
    return path


# ---------------------------------------------------------------------------
# Step 2: Load CL citations index
# ---------------------------------------------------------------------------

def load_cl_citations_index(path: Path) -> dict[tuple[str, str, str], int]:
    """
    Load citations.csv.bz2 into a (volume, reporter, page) -> cluster_id dict.
    Normalizes reporter strings (strip spaces, lowercase) for robust matching.
    Returns ~20M entries in ~2-3 GB RAM for full file; use with care.
    """
    print(f"\nLoading CL citations index from: {path}")
    t0 = time.time()
    index: dict[tuple[str, str, str], int] = {}
    n_rows = 0
    with bz2.open(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            vol = row.get("volume", "").strip()
            rep = normalize_reporter(row.get("reporter", ""))
            page = row.get("page", "").strip()
            cluster_id_str = row.get("cluster_id", "").strip()
            if vol and rep and page and cluster_id_str:
                key = (vol, rep, page)
                try:
                    index[key] = int(cluster_id_str)
                except ValueError:
                    pass
            if n_rows % 1_000_000 == 0:
                print(f"  [cl-index] loaded {n_rows:,} rows, {len(index):,} keys ({time.time()-t0:.0f}s)",
                      file=sys.stderr)
    print(f"  [cl-index] done: {n_rows:,} rows, {len(index):,} unique (vol,rep,page) keys in {time.time()-t0:.1f}s")
    return index


def normalize_reporter(rep: str) -> str:
    """Normalize reporter string for matching: lowercase, collapse spaces."""
    return re.sub(r"\s+", " ", rep.strip()).lower()


# ---------------------------------------------------------------------------
# Step 3: Parse LePaRD cite strings with eyecite
# ---------------------------------------------------------------------------

def parse_cite(cite_str: str) -> tuple[str, str, str] | None:
    """
    Parse a legal citation string into (volume, reporter, page).
    Returns None if no FullCaseCitation is found.
    """
    if not cite_str:
        return None
    try:
        from eyecite import get_citations
        from eyecite.models import FullCaseCitation
    except ImportError:
        raise RuntimeError("eyecite not installed — run: uv add eyecite")

    cites = get_citations(cite_str)
    for c in cites:
        if isinstance(c, FullCaseCitation):
            g = c.groups
            vol = g.get("volume", "").strip()
            rep = normalize_reporter(g.get("reporter", ""))
            page = g.get("page", "").strip()
            if vol and rep and page:
                return (vol, rep, page)
    return None


# ---------------------------------------------------------------------------
# Step 4: Sample LePaRD
# ---------------------------------------------------------------------------

def sample_lepard(n: int | None, seed: int = 42) -> list[dict]:
    """
    Load LePaRD pairs. n=None means load all rows (slow for 22M).
    Returns list of dicts with source_cite, dest_cite, source_court, etc.
    """
    try:
        from datasets import load_from_disk, load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed.", file=sys.stderr)
        sys.exit(1)

    if LEPARD_DISK.exists():
        print(f"  Loading LePaRD from disk: {LEPARD_DISK}")
        ds = load_from_disk(str(LEPARD_DISK))
    else:
        print("  LePaRD not on disk — streaming from HuggingFace...")
        ds = load_dataset("rmahari/LePaRD", split="train")

    total = len(ds)
    print(f"  Total LePaRD rows: {total:,}")

    if n is None or n >= total:
        indices = list(range(total))
        print(f"  Using all {total:,} rows")
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(total), n))
        print(f"  Sampled {len(indices):,} rows (seed={seed})")

    rows = []
    for idx in indices:
        row = ds[idx]
        rows.append({
            "source_id": int(row["source_id"]),
            "dest_id": int(row["dest_id"]),
            "source_cite": row.get("source_cite") or "",
            "dest_cite": row.get("dest_cite") or "",
            "source_name": row.get("source_name") or "",
            "source_court": row.get("source_court") or "",
            "dest_court": row.get("dest_court") or "",
            "source_date": row.get("source_date") or "",
        })
    return rows


# ---------------------------------------------------------------------------
# Step 5: Run the join analysis
# ---------------------------------------------------------------------------

def is_federal_appellate(court_name: str) -> bool:
    low = court_name.lower()
    return any(kw in low for kw in FEDERAL_APPELLATE_KEYWORDS)


def run_cite_overlap(
    sample_size: int = DEFAULT_SAMPLE,
    full_lepard: bool = False,
    seed: int = 42,
    citations_path: Path = DEFAULT_CITATIONS_BZ2,
    out_path: Path | None = None,
) -> dict:
    t_start = time.time()

    # 1. Ensure CL citations file
    citations_path = ensure_citations_file(citations_path, CL_CITATIONS_URL)

    # 2. Load CL citations index
    cl_index = load_cl_citations_index(citations_path)

    # 3. Sample LePaRD
    print(f"\nSampling LePaRD...")
    n = None if full_lepard else sample_size
    pairs = sample_lepard(n, seed)

    # 4. Parse + join
    print(f"\nParsing citation strings and joining with CL index...")
    t0 = time.time()

    results = []
    counters = Counter()
    parse_failures = Counter()
    reporter_misses: Counter[str] = Counter()

    unique_source_cites: dict[str, tuple[str, str, str] | None] = {}  # cache parses

    for i, pair in enumerate(pairs):
        src_cite = pair["source_cite"]
        src_court = pair["source_court"]
        is_appellate = is_federal_appellate(src_court)

        # Parse cite string (cached)
        if src_cite not in unique_source_cites:
            unique_source_cites[src_cite] = parse_cite(src_cite)
        parsed = unique_source_cites[src_cite]

        parse_ok = parsed is not None
        cl_cluster_id = None
        in_cl = False

        if parse_ok:
            cl_cluster_id = cl_index.get(parsed)
            in_cl = cl_cluster_id is not None
            if not in_cl:
                reporter_misses[parsed[1]] += 1

        counters["total"] += 1
        counters["has_source_cite"] += int(bool(src_cite))
        counters["parse_ok"] += int(parse_ok)
        counters["parse_fail"] += int(bool(src_cite) and not parse_ok)
        counters["in_cl"] += int(in_cl)
        counters["appellate"] += int(is_appellate)
        counters["appellate_in_cl"] += int(is_appellate and in_cl)
        counters["appellate_parse_ok"] += int(is_appellate and parse_ok)

        results.append({
            **pair,
            "parsed_volume": parsed[0] if parsed else None,
            "parsed_reporter": parsed[1] if parsed else None,
            "parsed_page": parsed[2] if parsed else None,
            "parse_ok": parse_ok,
            "cl_cluster_id": cl_cluster_id,
            "in_cl": in_cl,
            "is_federal_appellate_source": is_appellate,
        })

        if (i + 1) % 10_000 == 0:
            print(f"  [{i+1:,}/{len(pairs):,}] parsed {counters['parse_ok']:,}, "
                  f"in_cl {counters['in_cl']:,} ({time.time()-t0:.0f}s)", file=sys.stderr)

    # 5. Summary
    total = counters["total"]
    has_cite = counters["has_source_cite"]
    parse_ok = counters["parse_ok"]
    in_cl = counters["in_cl"]
    appellate = counters["appellate"]

    # Unique source cite stats
    unique_parsed = {c: p for c, p in unique_source_cites.items() if p is not None}
    unique_in_cl = sum(1 for p in unique_parsed.values() if cl_index.get(p) is not None)

    summary = {
        "sample_size": total,
        "full_lepard_run": full_lepard,
        "citations_file": str(citations_path.name),
        "cl_index_size": len(cl_index),
        "elapsed_sec": round(time.time() - t_start, 1),
        "row_level": {
            "total_rows": total,
            "has_source_cite": has_cite,
            "parse_ok": parse_ok,
            "parse_fail": counters["parse_fail"],
            "in_cl": in_cl,
            "parse_rate_pct": round(100 * parse_ok / max(1, has_cite), 2),
            "cl_match_of_parsed_pct": round(100 * in_cl / max(1, parse_ok), 2),
            "cl_match_of_total_pct": round(100 * in_cl / max(1, total), 2),
        },
        "unique_cite_level": {
            "unique_source_cites": len(unique_source_cites),
            "unique_parsed": len(unique_parsed),
            "unique_in_cl": unique_in_cl,
            "unique_match_of_parsed_pct": round(100 * unique_in_cl / max(1, len(unique_parsed)), 2),
        },
        "federal_appellate_subset": {
            "appellate_rows": appellate,
            "appellate_parse_ok": counters["appellate_parse_ok"],
            "appellate_in_cl": counters["appellate_in_cl"],
            "appellate_match_of_parsed_pct": round(
                100 * counters["appellate_in_cl"] / max(1, counters["appellate_parse_ok"]), 2
            ),
        },
        "top_missed_reporters": [
            {"reporter": r, "count": c}
            for r, c in reporter_misses.most_common(20)
        ],
    }

    print_report(summary)

    # 6. Save outputs
    summary_path = EDA_DIR / "cite_overlap_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[out] Summary -> {summary_path}")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[out] Pair-level results -> {out_path} ({len(results):,} rows)")

    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(s: dict) -> None:
    r = s["row_level"]
    u = s["unique_cite_level"]
    a = s["federal_appellate_subset"]
    print("\n" + "=" * 65)
    print("LePaRD <-> CourtListener CITATION-STRING OVERLAP REPORT")
    print("=" * 65)
    print(f"\nSample size:         {s['sample_size']:,} rows {'(full dataset)' if s['full_lepard_run'] else ''}")
    print(f"CL citations file:   {s['citations_file']}  ({s['cl_index_size']:,} entries)")
    print(f"Elapsed:             {s['elapsed_sec']}s")
    print()
    print("[1] Row-level (each LePaRD citation pair)")
    print(f"  Has source_cite:   {r['has_source_cite']:,}")
    print(f"  eyecite parsed:    {r['parse_ok']:,}  ({r['parse_rate_pct']}% of rows with cite)")
    print(f"  Parse failed:      {r['parse_fail']:,}")
    print(f"  Matched in CL:     {r['in_cl']:,}  ({r['cl_match_of_parsed_pct']}% of parsed, {r['cl_match_of_total_pct']}% of total)")
    print()
    print("[2] Unique citation level (deduplicated source_cite strings)")
    print(f"  Unique source cites:   {u['unique_source_cites']:,}")
    print(f"  Successfully parsed:   {u['unique_parsed']:,}")
    print(f"  Found in CL:           {u['unique_in_cl']:,}  ({u['unique_match_of_parsed_pct']}% of parsed)")
    print()
    print("[3] Federal appellate subset (source court is circuit court)")
    print(f"  Appellate rows:        {a['appellate_rows']:,}")
    print(f"  Parsed:                {a['appellate_parse_ok']:,}")
    print(f"  Matched in CL:         {a['appellate_in_cl']:,}  ({a['appellate_match_of_parsed_pct']}% of parsed)")
    print()
    if s["top_missed_reporters"]:
        print("[4] Top reporters in CL miss (parsed OK but not found in CL)")
        for entry in s["top_missed_reporters"][:10]:
            print(f"  {entry['reporter']:30s}  {entry['count']:,}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--sample", type=int, default=DEFAULT_SAMPLE,
        help=f"Number of LePaRD rows to sample (default: {DEFAULT_SAMPLE})",
    )
    ap.add_argument(
        "--full-lepard", action="store_true",
        help="Run on all 22M LePaRD rows (slow, ~1-2 hours)",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    ap.add_argument(
        "--citations-file", type=Path, default=DEFAULT_CITATIONS_BZ2,
        help=f"Path to CL citations CSV bz2 (default: {DEFAULT_CITATIONS_BZ2})",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Optional path to write pair-level results as JSONL",
    )
    args = ap.parse_args()

    run_cite_overlap(
        sample_size=args.sample,
        full_lepard=args.full_lepard,
        seed=args.seed,
        citations_path=args.citations_file,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
