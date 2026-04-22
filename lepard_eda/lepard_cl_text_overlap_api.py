"""
LePaRD <-> CourtListener TEXT overlap via CL REST API
======================================================

For a sample of LePaRD pairs:
  1. Parse source_cite → (volume, reporter, page) via eyecite
  2. Join citations.csv → cluster_id
  3. Fetch opinion text from CL API for that cluster_id
  4. Check whether the LePaRD `quote` appears in the opinion text

Three match levels:
  exact      -- verbatim substring
  normalized -- after lowercasing + collapsing whitespace
  fuzzy      -- difflib ratio >= threshold on a sliding window

Usage
-----
  python lepard_eda/lepard_cl_text_overlap_api.py
  python lepard_eda/lepard_cl_text_overlap_api.py --sample 200
  python lepard_eda/lepard_cl_text_overlap_api.py --out lepard_eda/api_overlap_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz as rf_fuzz

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = REPO_ROOT / "lepard_eda"
LEPARD_DISK = EDA_DIR / "lepard_data"
DEFAULT_CITATIONS_BZ2 = EDA_DIR / "cl_bulk" / "citations-2026-03-31.csv.bz2"
DEFAULT_CITATIONS_DB  = EDA_DIR / "cl_bulk" / "citations.db"
CLUSTER_CACHE_PATH    = EDA_DIR / "cl_bulk" / "cluster_text_cache.pkl"

CL_API_BASE = "https://www.courtlistener.com/api/rest/v4"
DEFAULT_SAMPLE = 100
# rapidfuzz partial_ratio is on 0-100; 75 means the quote aligns to a CL substring
# with ~75% character similarity. Chosen to recover OCR/bracket drift without
# accepting pure paraphrase.
DEFAULT_FUZZY_THRESHOLD = 75.0
# Polite delay between API requests (seconds)
REQUEST_DELAY = 0.3

# Opinion type priority: prefer majority/combined over dissent/concurrence
OPINION_TYPE_PRIORITY = {"010combined": 0, "020lead": 1, "030concurrence": 2, "040dissent": 3}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env(path: Path = REPO_ROOT / ".env") -> None:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()  # always overwrite stale values


def get_token() -> str:
    load_env()
    token = os.environ.get("COURTLISTENER_API_TOKEN", "")
    if not token:
        print("ERROR: COURTLISTENER_API_TOKEN not set. Add it to .env", file=sys.stderr)
        sys.exit(1)
    return token


def strip_html(html: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# Patterns that appear in CL opinion text (and sometimes inside LePaRD quotes)
# but carry no semantic content for a substring match.
_STAR_PAGING    = re.compile(r"\*+\d+")              # *432, **3
_PAGE_BRACKETS  = re.compile(r"\[\s*\d+\s*\]")       # [432]
_BRACKET_INSERT = re.compile(r"\[([A-Za-z])\]")      # [I]n  -> In
_BRACKET_SIC    = re.compile(r"\[sic\]", re.IGNORECASE)
_BRACKET_ELLIP  = re.compile(r"\[\s*\.\.\.\s*\]")    # [...]
_SMART_QUOTES   = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
})


def clean_legal_text(text: str) -> str:
    """
    Normalize text for fuzzy matching: remove legal-pagination artifacts and
    bracket substitutions that appear on only one side (LePaRD vs CL).
    Does NOT lowercase — caller does that.
    """
    if not text:
        return ""
    text = text.translate(_SMART_QUOTES)
    text = _BRACKET_INSERT.sub(r"\1", text)   # [I]n -> In
    text = _BRACKET_SIC.sub("", text)
    text = _BRACKET_ELLIP.sub("...", text)
    text = _STAR_PAGING.sub(" ", text)
    text = _PAGE_BRACKETS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_and_normalize(text: str) -> str:
    return clean_legal_text(text).lower()


def fuzzy_match(quote: str, opinion_text: str, threshold: float) -> float:
    """
    Return rapidfuzz partial_ratio on a 0-100 scale.

    Semantics: "does `quote` appear approximately as a substring of `opinion_text`?"
    partial_ratio aligns the shorter string to the best-matching substring of the
    longer one — exactly the operation we want.

    If the quote contains literal ellipses (LePaRD sometimes elides middle
    portions with `...`), score each non-ellipsis segment independently and
    return the MIN. This prevents a long quote from scoring 100 on just the
    first half when the second half is absent.
    """
    if not opinion_text or not quote:
        return 0.0
    q = clean_and_normalize(quote)
    t = clean_and_normalize(opinion_text)
    if not q:
        return 0.0
    if q in t:
        return 100.0

    # Split on literal ellipsis if present. Only split on segments of meaningful
    # length; stray "..." next to punctuation isn't real elision.
    segments = [s.strip() for s in re.split(r"\s*\.{3,}\s*", q) if len(s.strip()) >= 20]
    if len(segments) >= 2:
        # Each segment must independently match somewhere in t; take the worst.
        return min(rf_fuzz.partial_ratio(seg, t) for seg in segments)

    return rf_fuzz.partial_ratio(q, t)


# ---------------------------------------------------------------------------
# CL API
# ---------------------------------------------------------------------------

OPINION_TYPE_LABELS = {
    "010combined":   "combined",
    "020lead":       "majority",
    "030concurrence":"concurrence",
    "040dissent":    "dissent",
}

def fetch_opinions_for_cluster(cluster_id: int, token: str, session) -> list[dict]:
    """Return list of opinion dicts for a cluster, sorted by type priority."""
    url = f"{CL_API_BASE}/opinions/"
    resp = session.get(
        url,
        params={"cluster": cluster_id, "format": "json"},
        headers={"Authorization": f"Token {token}"},
        timeout=20,
    )
    if resp.status_code == 429:
        print("  [rate-limit] sleeping 60s...", file=sys.stderr)
        time.sleep(60)
        resp = session.get(url, params={"cluster": cluster_id, "format": "json"},
                           headers={"Authorization": f"Token {token}"}, timeout=20)
    if resp.status_code != 200:
        return []
    results = resp.json().get("results", [])
    return sorted(results, key=lambda op: OPINION_TYPE_PRIORITY.get(op.get("type", ""), 99))


def extract_text(opinion: dict) -> str:
    """Extract plain text from an opinion, trying fields in order."""
    for field in ["plain_text", "html_with_citations", "html", "html_lawbox",
                  "html_columbia", "xml_harvard"]:
        raw = opinion.get(field, "")
        if raw:
            return strip_html(raw) if field != "plain_text" else raw
    return ""


# ---------------------------------------------------------------------------
# LePaRD sampler
# ---------------------------------------------------------------------------

def sample_lepard(n: int, seed: int = 42) -> list[dict]:
    """Sample n rows from LePaRD by reading one Arrow shard directly — fast, low RAM."""
    import random
    import pyarrow as pa

    # Each shard has ~380-410K rows; one shard is plenty for small samples
    shards = sorted(LEPARD_DISK.glob("data-*.arrow")) if LEPARD_DISK.exists() else []
    if not shards:
        print("ERROR: LePaRD Arrow files not found at", LEPARD_DISK, file=sys.stderr)
        sys.exit(1)

    # Read just the first shard (380K rows) — memory-mapped, fast
    shard = shards[0]
    print(f"  Reading shard: {shard.name} ({shard.stat().st_size/1e6:.0f} MB)")
    with pa.memory_map(str(shard), "r") as mmap:
        try:
            table = pa.ipc.open_file(mmap).read_all()
        except pa.lib.ArrowInvalid:
            table = pa.ipc.open_stream(mmap).read_all()

    total = len(table)
    print(f"  Shard rows: {total:,}  — sampling {min(n, total)}")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(total), min(n, total)))

    cols = {name: table.column(name).to_pylist() for name in
            ["source_id", "dest_id", "source_cite", "source_name",
             "source_court", "dest_name", "dest_court", "quote", "destination_context"]}

    return [{
        "lepard_source_id":    int(cols["source_id"][i]),
        "lepard_dest_id":      int(cols["dest_id"][i]),
        "source_cite":         cols["source_cite"][i] or "",
        "source_name":         cols["source_name"][i] or "",
        "source_court":        cols["source_court"][i] or "",
        "dest_name":           cols["dest_name"][i] or "",
        "dest_court":          cols["dest_court"][i] or "",
        "quote":               cols["quote"][i] or "",
        "destination_context": cols["destination_context"][i] or "",
    } for i in indices]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def lookup_cluster_id(db: sqlite3.Connection, parsed: tuple[str, str, str] | None) -> int | None:
    if not parsed:
        return None
    row = db.execute(
        "SELECT cluster_id FROM citations WHERE volume=? AND reporter=? AND page=? LIMIT 1",
        parsed,
    ).fetchone()
    return row[0] if row else None


def run(
    sample_size: int = DEFAULT_SAMPLE,
    seed: int = 42,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    out_path: Path | None = None,
) -> dict:
    import requests
    sys.path.insert(0, str(Path(__file__).parent))
    from lepard_cl_cite_overlap import parse_cite

    token = get_token()
    t_start = time.time()

    # 1. Open SQLite citations index (fast, no RAM spike)
    db_path = DEFAULT_CITATIONS_DB
    if not db_path.exists():
        print(f"ERROR: SQLite index not found at {db_path}", file=sys.stderr)
        print("Run the one-time build first (see script header)", file=sys.stderr)
        sys.exit(1)
    print(f"Opening CL citations index: {db_path} ({db_path.stat().st_size/1e6:.0f} MB)")
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # 2. Sample LePaRD
    print(f"\nSampling {sample_size} LePaRD rows (seed={seed})...")
    pairs = sample_lepard(sample_size, seed)
    print(f"  Loaded {len(pairs)} pairs")

    # 3. Resolve cluster_ids
    print("\nResolving cluster_ids via citation string join...")
    resolved = []
    for p in pairs:
        parsed = parse_cite(p["source_cite"])
        cluster_id = lookup_cluster_id(db, parsed)
        resolved.append({**p, "parsed_cite": parsed, "cl_cluster_id": cluster_id})
    db.close()

    n_resolved = sum(1 for r in resolved if r["cl_cluster_id"])
    print(f"  Resolved: {n_resolved}/{len(resolved)} pairs have a cluster_id")

    # 4. Fetch opinion texts from CL API (with on-disk cache so re-runs are fast)
    unique_clusters = {r["cl_cluster_id"] for r in resolved if r["cl_cluster_id"]}
    cl_opinion_cache: dict[int, dict[str, str]] = {}
    if CLUSTER_CACHE_PATH.exists():
        with CLUSTER_CACHE_PATH.open("rb") as f:
            cl_opinion_cache = pickle.load(f)
        print(f"\n  Loaded {len(cl_opinion_cache)} cluster entries from cache "
              f"({CLUSTER_CACHE_PATH.stat().st_size/1e6:.1f} MB)")

    to_fetch = [cid for cid in unique_clusters if cid not in cl_opinion_cache]
    print(f"\nFetching opinion texts from CL API "
          f"({len(to_fetch)} new / {len(unique_clusters)} total clusters)...")
    session = requests.Session()
    for i, cid in enumerate(to_fetch, 1):
        opinions = fetch_opinions_for_cluster(cid, token, session)
        by_type: dict[str, str] = {}
        for op in opinions:
            label = OPINION_TYPE_LABELS.get(op.get("type", ""), op.get("type", "unknown"))
            text = extract_text(op)
            if text:
                by_type[label] = by_type.get(label, "") + " " + text
        cl_opinion_cache[cid] = by_type
        if i % 25 == 0:
            print(f"  [{i}/{len(to_fetch)}] fetched...", file=sys.stderr)
            # Periodic cache flush so we don't lose progress on interrupt
            CLUSTER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CLUSTER_CACHE_PATH.open("wb") as f:
                pickle.dump(cl_opinion_cache, f)
        time.sleep(REQUEST_DELAY)

    CLUSTER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLUSTER_CACHE_PATH.open("wb") as f:
        pickle.dump(cl_opinion_cache, f)
    n_with_text = sum(1 for cid in unique_clusters if cl_opinion_cache.get(cid))
    print(f"  Cache now holds {len(cl_opinion_cache)} clusters; "
          f"{n_with_text}/{len(unique_clusters)} in this sample have non-empty text")

    # 5. Text overlap checks — per opinion type
    print("\nRunning quote match checks...")
    results = []
    counters = Counter()

    for r in resolved:
        cid = r["cl_cluster_id"]
        quote = r["quote"]
        by_type = cl_opinion_cache.get(cid, {}) if cid else {}
        all_text = " ".join(by_type.values())

        has_quote   = bool(quote.strip())
        has_cl_text = bool(all_text.strip())
        checkable   = bool(cid and has_quote and has_cl_text)

        exact = norm = clean = False
        fuzz = 0.0
        matched_in: list[str] = []

        if checkable:
            exact = quote in all_text
            norm  = normalize(quote) in normalize(all_text)
            clean = clean_and_normalize(quote) in clean_and_normalize(all_text)
            fuzz  = fuzzy_match(quote, all_text, fuzzy_threshold)
            for op_type, op_text in by_type.items():
                if (quote in op_text
                        or normalize(quote) in normalize(op_text)
                        or clean_and_normalize(quote) in clean_and_normalize(op_text)):
                    matched_in.append(op_type)

        fuzzy_hit = fuzz >= fuzzy_threshold
        counters["total"] += 1
        counters["cluster_resolved"] += int(bool(cid))
        counters["checkable"] += int(checkable)
        counters["exact"] += int(exact)
        counters["norm"]  += int(norm)
        counters["clean"] += int(clean)
        counters["fuzzy"] += int(fuzzy_hit)
        for op_type in matched_in:
            counters[f"match_in_{op_type}"] += 1

        results.append({
            **r,
            "opinion_types_available": list(by_type.keys()),
            "cl_text_length": len(all_text),
            "quote_length":   len(quote),
            "checkable":      checkable,
            "exact_match":    exact,
            "norm_match":     norm,
            "clean_match":    clean,
            "fuzzy_ratio":    round(fuzz, 2),
            "fuzzy_match":    fuzzy_hit,
            "matched_in_opinion_types": matched_in,
        })

    # 6. Print report
    total     = counters["total"]
    checkable = counters["checkable"]
    type_breakdown = {k.replace("match_in_", ""): v
                      for k, v in counters.items() if k.startswith("match_in_")}
    # Fuzzy-ratio distribution (only over checkable pairs)
    ratios = [r["fuzzy_ratio"] for r in results if r["checkable"]]
    bins = [(100, 100, "exact(=100)"),
            (90, 99.999, "near(90-100)"),
            (75, 89.999, "fuzzy(75-90)"),
            (50, 74.999, "low(50-75)"),
            (0, 49.999,  "miss(<50)")]
    dist = {label: sum(1 for x in ratios if lo <= x <= hi) for lo, hi, label in bins}

    summary = {
        "sample_size":      total,
        "fuzzy_threshold":  fuzzy_threshold,
        "elapsed_sec":      round(time.time() - t_start, 1),
        "cluster_resolved": counters["cluster_resolved"],
        "checkable_pairs":  checkable,
        "exact_match":      counters["exact"],
        "norm_match":       counters["norm"],
        "clean_match":      counters["clean"],
        "fuzzy_match":      counters["fuzzy"],
        "exact_pct_of_checkable": round(100 * counters["exact"] / max(1, checkable), 2),
        "norm_pct_of_checkable":  round(100 * counters["norm"]  / max(1, checkable), 2),
        "clean_pct_of_checkable": round(100 * counters["clean"] / max(1, checkable), 2),
        "fuzzy_pct_of_checkable": round(100 * counters["fuzzy"] / max(1, checkable), 2),
        "ratio_distribution":     dist,
        "match_by_opinion_type":  type_breakdown,
    }

    print("\n" + "=" * 60)
    print("LePaRD <-> CourtListener TEXT OVERLAP (via API)")
    print("=" * 60)
    print(f"Sample size:          {total}")
    print(f"Cluster resolved:     {counters['cluster_resolved']}")
    print(f"Checkable pairs:      {checkable}  (resolved + has quote + has CL text)")
    print(f"Exact match:          {counters['exact']}  ({summary['exact_pct_of_checkable']}% of checkable)")
    print(f"Normalized match:     {counters['norm']}  ({summary['norm_pct_of_checkable']}% of checkable)")
    print(f"Clean match:          {counters['clean']}  ({summary['clean_pct_of_checkable']}% of checkable)  [strips brackets/star-paging]")
    print(f"Fuzzy match (≥{fuzzy_threshold}): {counters['fuzzy']}  ({summary['fuzzy_pct_of_checkable']}% of checkable)  [rapidfuzz partial_ratio]")
    print(f"\nFuzzy-ratio distribution (checkable pairs):")
    for lo, hi, label in bins:
        print(f"  {label:18s} {dist[label]:4d}")
    if type_breakdown:
        print(f"\nMatch by opinion type (exact/norm/clean):")
        for op_type, cnt in sorted(type_breakdown.items(), key=lambda x: -x[1]):
            print(f"  {op_type:15s}  {cnt:4d}  ({100*cnt/max(1,checkable):.1f}% of checkable)")
    print(f"\nElapsed:              {summary['elapsed_sec']}s")
    print("=" * 60)

    # Print a few illustrative examples
    print("\n--- Sample matches ---")
    shown = 0
    for r in results:
        if not r["checkable"]:
            continue
        match_type = ("EXACT" if r["exact_match"]
                      else "NORM"  if r["norm_match"]
                      else f"FUZZY({r['fuzzy_ratio']:.2f})" if r["fuzzy_match"]
                      else "MISS")
        if shown < 5:
            print(f"\n[{match_type}] {r['source_name']} ({r['source_cite']})")
            print(f"  cluster_id : {r['cl_cluster_id']}")
            print(f"  CL URL     : https://www.courtlistener.com/opinion/{r['cl_cluster_id']}/")
            print(f"  quote      : \"{r['quote'][:120]}\"")
            if r["exact_match"] or r["norm_match"]:
                # Show surrounding context in CL text
                idx = r["cl_text_length"] and (
                    (r.get("_cl_text","") or "").find(r["quote"])
                )
            shown += 1

    # Save outputs
    summary_path = EDA_DIR / "api_overlap_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[out] Summary -> {summary_path}")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                # Don't dump full CL text into JSONL — just a snippet around the quote
                f.write(json.dumps({k: v for k, v in r.items() if k != "_cl_text"},
                                   ensure_ascii=False) + "\n")
        print(f"[out] Pair results -> {out_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                    help=f"LePaRD rows to sample (default: {DEFAULT_SAMPLE})")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fuzzy-threshold", type=float, default=DEFAULT_FUZZY_THRESHOLD)
    ap.add_argument("--out", type=Path, default=None,
                    help="Write pair-level results to JSONL")
    args = ap.parse_args()
    run(sample_size=args.sample, seed=args.seed,
        fuzzy_threshold=args.fuzzy_threshold, out_path=args.out)


if __name__ == "__main__":
    main()
