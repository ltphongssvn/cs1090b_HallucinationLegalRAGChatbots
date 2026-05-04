# scripts/diagnostics/diag_04_reranker_truncation.py
"""Measure how much cluster text the reranker actually saw.

bge-reranker-v2-m3 truncates [query, doc] pair at max_length=512 tokens.
If clusters concatenate 3 chunks × 1024 tokens, most of the document is
discarded BEFORE scoring. This diagnostic quantifies that loss.
"""
import json
from collections import defaultdict
from pathlib import Path

corpus = Path("data/processed/baseline/corpus_chunks_cleaned.jsonl")
reranker_results = Path("data/processed/baseline/cleaned/reranker_results.jsonl")
gold = Path("data/processed/baseline/cleaned/gold_pairs_test.jsonl")

# Build cluster -> 3-chunk concat (matches reranker default)
chunks_by_cluster = defaultdict(list)
with corpus.open() as f:
    for line in f:
        r = json.loads(line)
        cid = int(r["cluster_id"])
        if len(chunks_by_cluster[cid]) >= 3:
            continue
        chunks_by_cluster[cid].append(r.get("text", ""))

cluster_text_lens = {}
for cid, chunks in chunks_by_cluster.items():
    concat = " ".join(chunks)
    cluster_text_lens[cid] = len(concat.split())

# Quote lengths
quote_lens = {}
with gold.open() as f:
    for line in f:
        r = json.loads(line)
        quote_lens[(int(r["source_id"]), int(r["dest_id"]))] = len(r["quote"].split())

# For reranker results, compute estimated truncation
print("=== Estimated reranker pair-token usage ===")
print("(rough estimate: words × 1.3 ≈ tokens for English)")
truncated_count = 0
total_pairs = 0
sample_pairs = []
with reranker_results.open() as f:
    for line in f:
        r = json.loads(line)
        key = (int(r["source_id"]), int(r["dest_id"]))
        q_words = quote_lens.get(key, 0)
        for hit in r["retrieved"][:10]:  # sample top-10 only for speed
            cid = int(hit["cluster_id"])
            doc_words = cluster_text_lens.get(cid, 0)
            est_tokens = int((q_words + doc_words) * 1.3)
            total_pairs += 1
            if est_tokens > 512:
                truncated_count += 1
            if len(sample_pairs) < 5:
                sample_pairs.append((q_words, doc_words, est_tokens))

print(f"  total top-10 pairs sampled : {total_pairs:,}")
print(f"  estimated > 512 tokens     : {truncated_count:,} ({100*truncated_count/total_pairs:.1f}%)")
print(f"\n  sample (q_words, doc_words, est_tokens):")
for q, d, t in sample_pairs:
    print(f"    q={q:>4}  d={d:>5}  est_total={t:>5}  truncated={t > 512}")

# Doc length percentiles
doc_lens = list(cluster_text_lens.values())
doc_lens.sort()
n = len(doc_lens)
print(f"\n=== Cluster doc-length distribution (words) ===")
print(f"  p10  : {doc_lens[n//10]}")
print(f"  p50  : {doc_lens[n//2]}")
print(f"  p90  : {doc_lens[int(n*0.9)]}")
print(f"  p99  : {doc_lens[int(n*0.99)]}")
print(f"\n=== If most docs >> 512 tokens, reranker only saw the FIRST 512 ===")
print("=== This explains the underperformance vs BM25/BGE-M3 (which see full doc) ===")
