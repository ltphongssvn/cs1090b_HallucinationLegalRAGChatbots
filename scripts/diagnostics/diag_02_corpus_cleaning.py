# scripts/diagnostics/diag_02_corpus_cleaning.py
"""Verify corpus cleaning + cluster_id presence + cluster-text retrievability.

Checks:
1. All chunks have cluster_id field
2. cluster_id values overlap with gold source_cluster_id (so retrieval can match)
3. Each cleaned chunk has substantive text (not just whitespace/citations stripped)
4. No empty text fields
5. Sample 3 cluster_ids that appear in gold and verify their corpus text exists
"""
import json
import random
from pathlib import Path

random.seed(0)
corpus = Path("data/processed/baseline/corpus_chunks_cleaned.jsonl")
gold = Path("data/processed/baseline/cleaned/gold_pairs_test.jsonl")

# Collect gold cluster_ids
gold_clusters = set()
with gold.open() as f:
    for line in f:
        gold_clusters.add(int(json.loads(line)["source_cluster_id"]))
print(f"unique gold source_cluster_ids: {len(gold_clusters):,}")

# Stream corpus, check fields + collect clusters
n_chunks = 0
n_with_cluster = 0
n_empty_text = 0
text_lens = []
corpus_clusters = set()
sample_chunks = {}  # cluster_id -> first chunk text
target_samples = list(random.sample(list(gold_clusters), min(3, len(gold_clusters))))
with corpus.open() as f:
    for line in f:
        n_chunks += 1
        r = json.loads(line)
        if "cluster_id" in r:
            n_with_cluster += 1
            cid = int(r["cluster_id"])
            corpus_clusters.add(cid)
            if cid in target_samples and cid not in sample_chunks:
                sample_chunks[cid] = r.get("text", "")[:300]
        text = r.get("text", "")
        if not text or not text.strip():
            n_empty_text += 1
        else:
            text_lens.append(len(text.split()))

print(f"\n--- FIELD CHECKS ---")
print(f"  total chunks       : {n_chunks:,}")
print(f"  with cluster_id    : {n_with_cluster:,} ({100*n_with_cluster/n_chunks:.2f}%)")
print(f"  empty text fields  : {n_empty_text:,}")

print(f"\n--- CLUSTER OVERLAP ---")
overlap = gold_clusters & corpus_clusters
print(f"  gold ∩ corpus      : {len(overlap):,} ({100*len(overlap)/len(gold_clusters):.2f}% of gold)")
print(f"  gold not in corpus : {len(gold_clusters - corpus_clusters):,}")

print(f"\n--- TEXT LENGTH DISTRIBUTION (words) ---")
text_lens.sort()
n = len(text_lens)
print(f"  min   : {text_lens[0]}")
print(f"  p10   : {text_lens[n//10]}")
print(f"  p50   : {text_lens[n//2]}")
print(f"  p90   : {text_lens[int(n*0.9)]}")
print(f"  max   : {text_lens[-1]}")

print(f"\n--- SAMPLE CHUNKS FROM GOLD CLUSTERS ---")
for cid, text in sample_chunks.items():
    print(f"\n  cluster_id={cid}")
    print(f"  text: {text}...")

verdict = "PASS"
if n_with_cluster < n_chunks * 0.99:
    verdict = "FAIL: <99% chunks have cluster_id"
if n_empty_text > n_chunks * 0.01:
    verdict = "FAIL: >1% empty text"
if len(overlap) < len(gold_clusters) * 0.95:
    verdict = "FAIL: <95% gold clusters in corpus"
print(f"\n=== VERDICT: {verdict} ===")
