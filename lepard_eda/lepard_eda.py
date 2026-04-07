"""
LePaRD EDA — Exploring the dataset and its overlap with CourtListener
=====================================================================
Run: python lepard_eda/lepard_eda.py

Outputs analysis to stdout and saves figures to lepard_eda/figures/
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = REPO_ROOT / "lepard_eda"
FIG_DIR = EDA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

LEPARD_DISK = EDA_DIR / "lepard_data"
CL_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "courtlistener_sample.jsonl"

# CourtListener federal appellate court IDs (from src/config.py)
CL_FEDERAL_APPELLATE_COURTS = frozenset({
    "ca1", "ca2", "ca3", "ca4", "ca5", "ca6",
    "ca7", "ca8", "ca9", "ca10", "ca11", "cadc", "cafc",
})

# ---------------------------------------------------------------------------
# 1. Load LePaRD
# ---------------------------------------------------------------------------
print("=" * 70)
print("LePaRD EDA")
print("=" * 70)

if LEPARD_DISK.exists():
    print(f"\nLoading LePaRD from disk: {LEPARD_DISK}")
    from datasets import load_from_disk
    ds = load_from_disk(str(LEPARD_DISK))
    print(f"  Rows: {len(ds):,}")
    print(f"  Features: {list(ds.features.keys())}")
else:
    print(f"\nLePaRD not found at {LEPARD_DISK}")
    print("Loading via streaming from HuggingFace (will be slower)...")
    from datasets import load_dataset
    ds = load_dataset("rmahari/LePaRD", split="train")
    print(f"  Rows: {len(ds):,}")
    print(f"  Features: {list(ds.features.keys())}")

# Convert to Arrow table for fast analytics
print("\nConverting to Arrow table...")
table = ds.data if hasattr(ds, 'data') and isinstance(ds.data, pa.Table) else ds.to_arrow() if hasattr(ds, 'to_arrow') else None

if table is None:
    print("WARNING: Could not get Arrow table, falling back to slower path")

# ---------------------------------------------------------------------------
# 2. Basic Statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 1: Basic Dataset Statistics")
print("=" * 70)

n_rows = len(ds)
print(f"\nTotal citation pairs: {n_rows:,}")

# Unique cases
dest_ids = set(ds["dest_id"])
source_ids = set(ds["source_id"])
all_opinion_ids = dest_ids | source_ids
print(f"\nUnique destination (citing) cases:  {len(dest_ids):,}")
print(f"Unique source (cited) cases:       {len(source_ids):,}")
print(f"Total unique opinion IDs:          {len(all_opinion_ids):,}")
print(f"Overlap (IDs appearing as both):   {len(dest_ids & source_ids):,}")

# Unique passage IDs
passage_ids = set(ds["passage_id"])
print(f"\nUnique passage IDs: {len(passage_ids):,}")
print(f"Avg citations per passage: {n_rows / len(passage_ids):.2f}")

# ---------------------------------------------------------------------------
# 3. Court Analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 2: Court Distribution")
print("=" * 70)

dest_courts = Counter(ds["dest_court"])
source_courts = Counter(ds["source_court"])

print(f"\nUnique destination courts: {len(dest_courts):,}")
print(f"Unique source courts:      {len(source_courts):,}")

print("\n--- Top 20 Destination Courts (citing cases) ---")
for court, count in dest_courts.most_common(20):
    pct = count / n_rows * 100
    print(f"  {court[:80]:80s}  {count:>10,}  ({pct:5.2f}%)")

print("\n--- Top 20 Source Courts (cited cases) ---")
for court, count in source_courts.most_common(20):
    pct = count / n_rows * 100
    print(f"  {court[:80]:80s}  {count:>10,}  ({pct:5.2f}%)")

# ---------------------------------------------------------------------------
# 4. Federal Appellate Court overlap
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 3: Federal Appellate Court Overlap with CourtListener")
print("=" * 70)

# CourtListener uses short codes like "ca9", LePaRD uses full names like
# "United States Court of Appeals for the Ninth Circuit"
# We need to map between them.

# Build a mapping from LePaRD court names to CL court IDs
COURT_NAME_TO_CL_ID = {}
circuit_keywords = {
    "ca1": ["first circuit"],
    "ca2": ["second circuit"],
    "ca3": ["third circuit"],
    "ca4": ["fourth circuit"],
    "ca5": ["fifth circuit"],
    "ca6": ["sixth circuit"],
    "ca7": ["seventh circuit"],
    "ca8": ["eighth circuit"],
    "ca9": ["ninth circuit"],
    "ca10": ["tenth circuit"],
    "ca11": ["eleventh circuit"],
    "cadc": ["district of columbia circuit", "d.c. circuit"],
    "cafc": ["federal circuit"],
}

all_lepard_courts = set(dest_courts.keys()) | set(source_courts.keys())
for court_name in all_lepard_courts:
    name_lower = court_name.lower()
    for cl_id, keywords in circuit_keywords.items():
        if any(kw in name_lower for kw in keywords):
            COURT_NAME_TO_CL_ID[court_name] = cl_id
            break

# Also check for "court of appeals" without specific circuit
appellate_courts_lepard = {c for c in all_lepard_courts if "court of appeals" in c.lower() or "circuit" in c.lower()}
district_courts_lepard = {c for c in all_lepard_courts if "district court" in c.lower()}
supreme_court_lepard = {c for c in all_lepard_courts if "supreme court" in c.lower()}

print(f"\nAll unique courts in LePaRD: {len(all_lepard_courts):,}")
print(f"  Appellate courts (Court of Appeals / Circuit): {len(appellate_courts_lepard):,}")
print(f"  District courts: {len(district_courts_lepard):,}")
print(f"  Supreme court entries: {len(supreme_court_lepard):,}")
print(f"  Other: {len(all_lepard_courts - appellate_courts_lepard - district_courts_lepard - supreme_court_lepard):,}")

print(f"\nMapped to CourtListener federal appellate IDs: {len(COURT_NAME_TO_CL_ID):,} court names")

# Count rows where BOTH dest and source are federal appellate
dest_court_list = ds["dest_court"]
source_court_list = ds["source_court"]

n_dest_appellate = 0
n_source_appellate = 0
n_both_appellate = 0
dest_appellate_ids = set()
source_appellate_ids = set()

dest_id_list = ds["dest_id"]
source_id_list = ds["source_id"]

for i in range(n_rows):
    d_app = dest_court_list[i] in COURT_NAME_TO_CL_ID
    s_app = source_court_list[i] in COURT_NAME_TO_CL_ID
    if d_app:
        n_dest_appellate += 1
        dest_appellate_ids.add(dest_id_list[i])
    if s_app:
        n_source_appellate += 1
        source_appellate_ids.add(source_id_list[i])
    if d_app and s_app:
        n_both_appellate += 1

all_appellate_ids = dest_appellate_ids | source_appellate_ids

print(f"\nRows where citing case is federal appellate:    {n_dest_appellate:>12,}  ({n_dest_appellate/n_rows*100:.1f}%)")
print(f"Rows where cited case is federal appellate:     {n_source_appellate:>12,}  ({n_source_appellate/n_rows*100:.1f}%)")
print(f"Rows where BOTH are federal appellate:          {n_both_appellate:>12,}  ({n_both_appellate/n_rows*100:.1f}%)")
print(f"\nUnique federal appellate opinion IDs in LePaRD: {len(all_appellate_ids):,}")
print(f"  (These should directly match CourtListener opinion IDs)")

# ---------------------------------------------------------------------------
# 5. Date Analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 4: Date Distribution")
print("=" * 70)

dest_dates = ds["dest_date"]
source_dates = ds["source_date"]

# Extract years
from collections import defaultdict
dest_year_counts = Counter()
source_year_counts = Counter()

for d in dest_dates:
    if d and len(d) >= 4:
        try:
            dest_year_counts[int(d[:4])] += 1
        except (ValueError, TypeError):
            pass

for d in source_dates:
    if d and len(d) >= 4:
        try:
            source_year_counts[int(d[:4])] += 1
        except (ValueError, TypeError):
            pass

dest_years = sorted(dest_year_counts.keys())
source_years = sorted(source_year_counts.keys())

if dest_years:
    print(f"\nDestination (citing) case date range: {dest_years[0]} - {dest_years[-1]}")
if source_years:
    print(f"Source (cited) case date range:        {source_years[0]} - {source_years[-1]}")

# Print decade distribution
print("\n--- Destination cases by decade ---")
decade_counts = Counter()
for yr, cnt in dest_year_counts.items():
    decade_counts[(yr // 10) * 10] += cnt
for decade in sorted(decade_counts.keys()):
    cnt = decade_counts[decade]
    print(f"  {decade}s: {cnt:>12,}  ({cnt/n_rows*100:5.2f}%)")

print("\n--- Source cases by decade ---")
decade_counts = Counter()
for yr, cnt in source_year_counts.items():
    decade_counts[(yr // 10) * 10] += cnt
for decade in sorted(decade_counts.keys()):
    cnt = decade_counts[decade]
    print(f"  {decade}s: {cnt:>12,}  ({cnt/n_rows*100:5.2f}%)")

# ---------------------------------------------------------------------------
# 6. Citation Pattern Analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 5: Citation Patterns")
print("=" * 70)

# How many times is each source case cited?
source_citation_counts = Counter(ds["source_id"])
most_cited = source_citation_counts.most_common(20)

print("\n--- Top 20 Most-Cited Cases ---")
source_names = {}
source_cites_map = {}
# Build lookup from source_id -> source_name (take first occurrence)
for i in range(min(n_rows, 5_000_000)):  # cap to avoid excessive iteration
    sid = source_id_list[i]
    if sid not in source_names:
        source_names[sid] = ds["source_name"][i]
        source_cites_map[sid] = ds["source_cite"][i]

for sid, count in most_cited:
    name = source_names.get(sid, "?")
    cite = source_cites_map.get(sid, "?")
    print(f"  [{count:>7,}x] {name[:60]:60s} ({cite[:40]})")

# Distribution of citation counts
print("\n--- Citation count distribution (how many times each source is cited) ---")
cite_vals = list(source_citation_counts.values())
cite_vals.sort()
n_sources = len(cite_vals)
print(f"  Total unique source cases: {n_sources:,}")
print(f"  Mean citations per source: {sum(cite_vals)/n_sources:.1f}")
print(f"  Median: {cite_vals[n_sources//2]:,}")
print(f"  P90: {cite_vals[int(n_sources*0.9)]:,}")
print(f"  P99: {cite_vals[int(n_sources*0.99)]:,}")
print(f"  Max: {cite_vals[-1]:,}")

cited_once = sum(1 for v in cite_vals if v == 1)
print(f"  Cited only once: {cited_once:,} ({cited_once/n_sources*100:.1f}%)")

# How many dest cases cite how many sources?
dest_citation_counts = Counter(ds["dest_id"])
dest_vals = sorted(dest_citation_counts.values())
n_dests = len(dest_vals)
print(f"\n--- Citing behavior (citations per destination case) ---")
print(f"  Total unique dest cases: {n_dests:,}")
print(f"  Mean citations per dest: {sum(dest_vals)/n_dests:.1f}")
print(f"  Median: {dest_vals[n_dests//2]:,}")
print(f"  P90: {dest_vals[int(n_dests*0.9)]:,}")
print(f"  P99: {dest_vals[int(n_dests*0.99)]:,}")
print(f"  Max: {dest_vals[-1]:,}")

# ---------------------------------------------------------------------------
# 7. Quote/Context Length Analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 6: Text Length Analysis (sampled)")
print("=" * 70)

import random
random.seed(42)
sample_size = min(100_000, n_rows)
sample_indices = random.sample(range(n_rows), sample_size)

quote_lengths = []
context_lengths = []

for idx in sample_indices:
    q = ds["quote"][idx]
    c = ds["destination_context"][idx]
    if q:
        quote_lengths.append(len(q))
    if c:
        context_lengths.append(len(c))

def print_stats(name, vals):
    vals.sort()
    n = len(vals)
    if n == 0:
        print(f"  {name}: no data")
        return
    print(f"  {name} (n={n:,}):")
    print(f"    Mean:   {sum(vals)/n:,.0f} chars")
    print(f"    Median: {vals[n//2]:,} chars")
    print(f"    P10:    {vals[int(n*0.1)]:,} chars")
    print(f"    P90:    {vals[int(n*0.9)]:,} chars")
    print(f"    Min:    {vals[0]:,} chars")
    print(f"    Max:    {vals[-1]:,} chars")

print(f"\nSampled {sample_size:,} rows:")
print_stats("Quote length", quote_lengths)
print_stats("Context length", context_lengths)

# ---------------------------------------------------------------------------
# 8. ID Overlap Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 7: CourtListener Overlap Summary")
print("=" * 70)

print(f"""
KEY FINDING: LePaRD uses CourtListener opinion IDs as dest_id / source_id.

LePaRD opinion IDs:
  - Total unique opinion IDs:           {len(all_opinion_ids):>12,}
  - Federal appellate opinion IDs:      {len(all_appellate_ids):>12,}

CourtListener corpus (from README):
  - Federal appellate opinions:         1,465,484

Overlap estimate:
  The LePaRD dest_id/source_id fields ARE CourtListener opinion IDs.
  LePaRD has {len(all_appellate_ids):,} unique federal appellate IDs.
  CourtListener has 1,465,484 federal appellate opinions.

  These datasets share the same ID space — overlap is structural, not coincidental.
  The actual overlap ratio depends on which CourtListener opinions contain
  citations that LePaRD extracted (quotations to precedent).

  To compute exact overlap, you would need to:
  1. Extract all opinion IDs from the CourtListener JSONL shards (DVC/S3)
  2. Intersect with LePaRD's all_opinion_ids set

  However, since LePaRD was BUILT FROM CourtListener data, the overlap
  for federal appellate cases is expected to be very high.
""")

# Save ID sets for later use
print("Saving ID sets for downstream use...")
id_data = {
    "all_opinion_ids": sorted(all_opinion_ids),
    "dest_ids": sorted(dest_ids),
    "source_ids": sorted(source_ids),
    "federal_appellate_ids": sorted(all_appellate_ids),
}
# Save as compact JSON
for key, ids in id_data.items():
    outpath = EDA_DIR / f"{key}.json"
    with open(outpath, "w") as f:
        json.dump(ids, f)
    print(f"  Saved {key}: {len(ids):,} IDs -> {outpath.name}")

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
