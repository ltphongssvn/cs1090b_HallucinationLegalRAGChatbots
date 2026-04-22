"""
LePaRD <-> CourtListener TEXT overlap analysis
===============================================

Goes beyond ID matching: for sampled LePaRD pairs, loads the corresponding
CourtListener opinion text and checks whether the LePaRD `quote` field
actually appears in that opinion's text.

Three levels of match:
  exact       -- quote is a verbatim substring of CL text
  normalized  -- after lowercasing + collapsing whitespace
  fuzzy       -- difflib SequenceMatcher ratio >= threshold (default 0.85)
                 on a sliding window of text chunks

Usage
-----
  # Full run on cluster (CL shards in data/raw/cl_federal_appellate_bulk/)
  python lepard_eda/lepard_text_overlap.py

  # Point at a different CL shard directory or fixture
  python lepard_eda/lepard_text_overlap.py --cl-dir data/raw/cl_federal_appellate_bulk
  python lepard_eda/lepard_text_overlap.py --cl-dir tests/fixtures --cl-glob "*.jsonl"

  # Change sample size
  python lepard_eda/lepard_text_overlap.py --sample 500

  # Save pair-level results as JSONL
  python lepard_eda/lepard_text_overlap.py --out lepard_eda/text_overlap_results.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = REPO_ROOT / "lepard_eda"
LEPARD_DISK = EDA_DIR / "lepard_data"
DEFAULT_CL_DIR = REPO_ROOT / "data" / "raw" / "cl_federal_appellate_bulk"
DEFAULT_SAMPLE = 1_000
DEFAULT_FUZZY_THRESHOLD = 0.85
# Chunk size used for fuzzy windowing (chars)
FUZZY_WINDOW = 500
FUZZY_STEP = 250

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase + collapse all whitespace to single spaces."""
    return re.sub(r"\s+", " ", text).strip().lower()


def fuzzy_match(quote: str, opinion_text: str, threshold: float, window: int, step: int) -> float:
    """
    Slide a window over opinion_text and return the max SequenceMatcher ratio
    between quote and any window chunk. Returns 0.0 if opinion_text is empty.
    """
    if not opinion_text:
        return 0.0
    q_norm = normalize(quote)
    t_norm = normalize(opinion_text)
    if not q_norm:
        return 0.0
    # Exact match short-circuit
    if q_norm in t_norm:
        return 1.0
    best = 0.0
    for start in range(0, max(1, len(t_norm) - window + 1), step):
        chunk = t_norm[start : start + window]
        ratio = SequenceMatcher(None, q_norm, chunk, autojunk=False).ratio()
        if ratio > best:
            best = ratio
        if best >= threshold:
            return best
    return best


# ---------------------------------------------------------------------------
# CourtListener index builder
# ---------------------------------------------------------------------------

def iter_cl_shards(cl_dir: Path, glob: str = "*.jsonl*") -> "Iterator[dict]":
    """Yield parsed opinion records from all JSONL (or .jsonl.gz) shards."""
    shards = sorted(cl_dir.glob(glob))
    if not shards:
        print(f"  [warn] No CL shards matched {cl_dir / glob}", file=sys.stderr)
        return
    for shard in shards:
        opener = gzip.open if shard.suffix == ".gz" else open
        with opener(shard, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def build_cl_index(cl_dir: Path, target_ids: set[int], glob: str = "*.jsonl*") -> dict[int, str]:
    """
    Stream CL shards and collect {opinion_id: text} for ids in target_ids.
    Returns only matched ids — stops loading once all targets are found.
    """
    index: dict[int, str] = {}
    remaining = set(target_ids)
    t0 = time.time()
    n_records = 0
    for record in iter_cl_shards(cl_dir, glob):
        n_records += 1
        try:
            oid = int(record.get("id", -1))
        except (TypeError, ValueError):
            continue
        if oid in remaining:
            text = record.get("text") or record.get("raw_text") or ""
            index[oid] = text
            remaining.discard(oid)
        if not remaining:
            break
        if n_records % 100_000 == 0:
            elapsed = time.time() - t0
            print(
                f"  [cl-index] scanned {n_records:,} records, "
                f"found {len(index):,}/{len(target_ids):,} targets "
                f"({elapsed:.0f}s)",
                file=sys.stderr,
            )
    elapsed = time.time() - t0
    print(
        f"  [cl-index] done: scanned {n_records:,} records, "
        f"found {len(index):,}/{len(target_ids):,} targets in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return index


# ---------------------------------------------------------------------------
# LePaRD sampler
# ---------------------------------------------------------------------------

def sample_lepard_pairs(
    n: int, seed: int = 42
) -> list[dict]:
    """
    Load a random sample of LePaRD pairs from disk, returning dicts with
    source_id, dest_id, quote, source_name, dest_name, source_court, dest_court.
    Falls back to HuggingFace streaming if local Arrow files are absent.
    """
    print(f"\nLoading LePaRD sample (n={n:,}, seed={seed})...")
    try:
        from datasets import load_from_disk, load_dataset  # type: ignore
    except ImportError:
        print("ERROR: `datasets` package not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    if LEPARD_DISK.exists():
        ds = load_from_disk(str(LEPARD_DISK))
    else:
        print("  LePaRD not on disk — streaming from HuggingFace (slow)...")
        ds = load_dataset("rmahari/LePaRD", split="train")

    total = len(ds)
    print(f"  Total LePaRD rows: {total:,}")

    rng = random.Random(seed)
    indices = rng.sample(range(total), min(n, total))
    indices_sorted = sorted(indices)

    sample = []
    for idx in indices_sorted:
        row = ds[idx]
        sample.append({
            "source_id": int(row["source_id"]),
            "dest_id": int(row["dest_id"]),
            "quote": row.get("quote") or "",
            "source_name": row.get("source_name") or "",
            "dest_name": row.get("dest_name") or "",
            "source_court": row.get("source_court") or "",
            "dest_court": row.get("dest_court") or "",
            "source_cite": row.get("source_cite") or "",
        })
    print(f"  Sampled {len(sample):,} pairs.")
    return sample


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_text_overlap(
    cl_dir: Path,
    sample_size: int = DEFAULT_SAMPLE,
    seed: int = 42,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    cl_glob: str = "*.jsonl*",
    out_path: Path | None = None,
) -> dict:
    """
    Full pipeline: sample LePaRD -> build CL index -> check text overlap.
    Returns summary dict.
    """
    t_start = time.time()

    # 1. Sample LePaRD
    pairs = sample_lepard_pairs(sample_size, seed)

    # 2. Collect unique SOURCE ids we need from CL
    #    (source_id = the cited opinion — the one whose text should contain the quote)
    source_ids = {p["source_id"] for p in pairs}
    print(f"\n  Unique source IDs to look up in CL: {len(source_ids):,}")

    # 3. Build CL text index
    print(f"\nBuilding CL text index from: {cl_dir}")
    cl_index = build_cl_index(cl_dir, source_ids, glob=cl_glob)
    found_ids = set(cl_index.keys())
    missing_ids = source_ids - found_ids
    print(f"  Source IDs found in CL: {len(found_ids):,} / {len(source_ids):,} "
          f"({100*len(found_ids)/max(1,len(source_ids)):.1f}%)")
    print(f"  Missing (not in CL corpus): {len(missing_ids):,}")

    # 4. Text overlap checks
    print(f"\nRunning text overlap checks on {len(pairs):,} pairs...")
    results = []
    counters = Counter()

    for pair in pairs:
        sid = pair["source_id"]
        quote = pair["quote"]
        cl_text = cl_index.get(sid, "")

        id_found = sid in found_ids
        has_quote = bool(quote.strip())
        has_cl_text = bool(cl_text.strip())

        exact_match = False
        norm_match = False
        fuzzy_ratio = 0.0
        fuzzy_match_hit = False

        if id_found and has_quote and has_cl_text:
            exact_match = quote in cl_text
            norm_match = normalize(quote) in normalize(cl_text)
            fuzzy_ratio = fuzzy_match(quote, cl_text, fuzzy_threshold, FUZZY_WINDOW, FUZZY_STEP)
            fuzzy_match_hit = fuzzy_ratio >= fuzzy_threshold

        counters["total"] += 1
        counters["id_found"] += int(id_found)
        counters["id_missing"] += int(not id_found)
        counters["has_quote"] += int(has_quote)
        counters["has_cl_text"] += int(has_cl_text)
        counters["exact_match"] += int(exact_match)
        counters["norm_match"] += int(norm_match)
        counters["fuzzy_match"] += int(fuzzy_match_hit)
        counters["checkable"] += int(id_found and has_quote and has_cl_text)

        results.append({
            **pair,
            "id_found_in_cl": id_found,
            "cl_text_length": len(cl_text),
            "quote_length": len(quote),
            "exact_match": exact_match,
            "norm_match": norm_match,
            "fuzzy_ratio": round(fuzzy_ratio, 4),
            "fuzzy_match": fuzzy_match_hit,
        })

    # 5. Summary
    total = counters["total"]
    checkable = counters["checkable"]

    summary = {
        "sample_size": total,
        "cl_dir": str(cl_dir),
        "fuzzy_threshold": fuzzy_threshold,
        "id_level": {
            "source_ids_sampled": len(source_ids),
            "found_in_cl": len(found_ids),
            "missing_from_cl": len(missing_ids),
            "id_match_pct": round(100 * len(found_ids) / max(1, len(source_ids)), 2),
        },
        "text_level": {
            "checkable_pairs": checkable,
            "exact_match": counters["exact_match"],
            "norm_match": counters["norm_match"],
            "fuzzy_match": counters["fuzzy_match"],
            "exact_match_pct_of_checkable": round(100 * counters["exact_match"] / max(1, checkable), 2),
            "norm_match_pct_of_checkable": round(100 * counters["norm_match"] / max(1, checkable), 2),
            "fuzzy_match_pct_of_checkable": round(100 * counters["fuzzy_match"] / max(1, checkable), 2),
            "exact_match_pct_of_total": round(100 * counters["exact_match"] / max(1, total), 2),
            "norm_match_pct_of_total": round(100 * counters["norm_match"] / max(1, total), 2),
        },
        "elapsed_sec": round(time.time() - t_start, 1),
    }

    # 6. Print report
    print_report(summary)

    # 7. Optionally write pair-level JSONL
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[out] Wrote {len(results):,} pair records -> {out_path}")

    # Save summary JSON alongside script
    summary_path = EDA_DIR / "text_overlap_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[out] Summary -> {summary_path}")

    return summary


def print_report(s: dict) -> None:
    il = s["id_level"]
    tl = s["text_level"]
    print("\n" + "=" * 65)
    print("LePaRD <-> CourtListener TEXT OVERLAP REPORT")
    print("=" * 65)
    print(f"\nSample size:            {s['sample_size']:,} pairs")
    print(f"CL corpus dir:          {s['cl_dir']}")
    print(f"Fuzzy threshold:        {s['fuzzy_threshold']}")
    print(f"Elapsed:                {s['elapsed_sec']}s")
    print()
    print("[1] ID-level (source_id found in CL)")
    print(f"  Unique source IDs:    {il['source_ids_sampled']:,}")
    print(f"  Found in CL:          {il['found_in_cl']:,}  ({il['id_match_pct']}%)")
    print(f"  Missing from CL:      {il['missing_from_cl']:,}")
    print()
    print("[2] Text-level (quote appears in CL opinion text)")
    print(f"  Checkable pairs:      {tl['checkable_pairs']:,}  (id found + has quote + has CL text)")
    print(f"  Exact match:          {tl['exact_match']:,}  ({tl['exact_match_pct_of_checkable']}% of checkable, {tl['exact_match_pct_of_total']}% of total)")
    print(f"  Normalized match:     {tl['norm_match']:,}  ({tl['norm_match_pct_of_checkable']}% of checkable, {tl['norm_match_pct_of_total']}% of total)")
    print(f"  Fuzzy match:          {tl['fuzzy_match']:,}  ({tl['fuzzy_match_pct_of_checkable']}% of checkable)")
    print()
    if tl["checkable_pairs"] == 0:
        print("  NOTE: 0 checkable pairs — likely the CL corpus is not present locally.")
        print("  Run on the cluster with --cl-dir data/raw/cl_federal_appellate_bulk")
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
        "--cl-dir", type=Path, default=DEFAULT_CL_DIR,
        help=f"Directory containing CourtListener JSONL shards (default: {DEFAULT_CL_DIR})",
    )
    ap.add_argument(
        "--cl-glob", default="*.jsonl*",
        help="Glob pattern for shard files inside --cl-dir (default: *.jsonl*)",
    )
    ap.add_argument(
        "--sample", type=int, default=DEFAULT_SAMPLE,
        help=f"Number of LePaRD pairs to sample (default: {DEFAULT_SAMPLE})",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    ap.add_argument(
        "--fuzzy-threshold", type=float, default=DEFAULT_FUZZY_THRESHOLD,
        help=f"SequenceMatcher ratio threshold for fuzzy match (default: {DEFAULT_FUZZY_THRESHOLD})",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Optional path to write pair-level results as JSONL",
    )
    args = ap.parse_args()

    run_text_overlap(
        cl_dir=args.cl_dir,
        sample_size=args.sample,
        seed=args.seed,
        fuzzy_threshold=args.fuzzy_threshold,
        cl_glob=args.cl_glob,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
