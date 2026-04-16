#!/usr/bin/env python3
"""Rebuild court_distribution, text_length_stats, opinion_type_distribution,
precedential_status_distribution, text_source_counts, ocr_extracted_count
from existing shards. Rewrites manifest.json in place."""

import json
from collections import Counter
from pathlib import Path

SHARD_DIR = Path("data/raw/cl_federal_appellate_bulk")
MANIFEST = SHARD_DIR / "manifest.json"

m = json.loads(MANIFEST.read_text())

court_dist: Counter[str] = Counter()
op_type_dist: Counter[str] = Counter()
prec_dist: Counter[str] = Counter()
text_src: Counter[str] = Counter()
ocr_count = 0
lengths: list[int] = []

shards = sorted(SHARD_DIR.glob("shard_*.jsonl"))
print(f"scanning {len(shards)} shards...")
for i, sp in enumerate(shards):
    with open(sp) as f:
        for line in f:
            r = json.loads(line)
            court_dist[r.get("court_id", "")] += 1
            op_type_dist[r.get("opinion_type", "")] += 1
            prec_dist[r.get("precedential_status", "")] += 1
            text_src[r.get("text_source", "")] += 1
            if str(r.get("extracted_by_ocr", "")).lower() == "true":
                ocr_count += 1
            lengths.append(r.get("text_length", 0))
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(shards)}")

lengths.sort()
n = len(lengths)
tls = {
    "count": n,
    "mean": int(sum(lengths) / n),
    "min": lengths[0],
    "max": lengths[-1],
    "median": lengths[n // 2],
    "p25": lengths[n // 4],
    "p75": lengths[3 * n // 4],
    "p90": lengths[int(n * 0.9)],
    "p95": lengths[int(n * 0.95)],
    "p99": lengths[int(n * 0.99)],
}

m["court_distribution"] = dict(court_dist)
m["opinion_type_distribution"] = dict(op_type_dist)
m["precedential_status_distribution"] = dict(prec_dist)
m["text_source_counts"] = dict(text_src)
m["text_length_stats"] = tls
m["ocr_extracted_count"] = ocr_count

MANIFEST.write_text(json.dumps(m, indent=2))
print(f"\n✓ rebuilt stats: {n:,} records, {len(court_dist)} courts")
print(f"  text len mean={tls['mean']:,} median={tls['median']:,} p95={tls['p95']:,}")
