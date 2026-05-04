# scripts/diagnostics/diag_03_retrieval_quality.py
"""Compare BM25 + BGE-M3 candidate sets — focus on whether they include gold cluster.

Key insight: if neither retriever puts gold in top-100 for ~67% of queries
(Hit@100 ~0.33), then reranking can't help — it only reorders within the
existing candidate set. This diagnostic verifies the input to reranking
contains the gold cluster.
"""
import json
from pathlib import Path

cleaned = Path("data/processed/baseline/cleaned")

def gold_recall_at_100(results_path: Path) -> dict:
    rows = [json.loads(l) for l in results_path.open()]
    n = len(rows)
    hits = 0
    rank_dist = []
    for r in rows:
        gold = int(r["source_cluster_id"])
        retrieved_ids = [int(h["cluster_id"]) for h in r["retrieved"]]
        if gold in retrieved_ids:
            hits += 1
            rank_dist.append(retrieved_ids.index(gold) + 1)
    return {"n": n, "recall_100": hits / n, "ranks_when_found": rank_dist}

print("=== Gold-cluster recall in top-100 ===")
for label in ("bm25", "bge_m3", "rrf", "reranker"):
    p = cleaned / f"{label}_results.jsonl"
    if not p.exists():
        print(f"  {label:<10} MISSING")
        continue
    d = gold_recall_at_100(p)
    ranks = sorted(d["ranks_when_found"])
    n = len(ranks)
    p10 = ranks[n//10] if n else 0
    p50 = ranks[n//2] if n else 0
    p90 = ranks[int(n*0.9)] if n else 0
    print(f"  {label:<10} recall@100={d['recall_100']:.4f}  "
          f"rank-when-found p10={p10}  p50={p50}  p90={p90}")

# Critical: how often is gold in BOTH bm25 and bge_m3 top-100?
print("\n=== Joint recall (RRF input quality) ===")
bm25 = {(r["source_id"], r["dest_id"]): {int(h["cluster_id"]) for h in r["retrieved"]}
        for r in (json.loads(l) for l in (cleaned / "bm25_results.jsonl").open())}
bge = {(r["source_id"], r["dest_id"]): {int(h["cluster_id"]) for h in r["retrieved"]}
       for r in (json.loads(l) for l in (cleaned / "bge_m3_results.jsonl").open())}
gold = {(r["source_id"], r["dest_id"]): int(r["source_cluster_id"])
        for r in (json.loads(l) for l in (cleaned / "gold_pairs_test.jsonl").open())}

n = 0
both = 0; bm25_only = 0; bge_only = 0; neither = 0
for k, gid in gold.items():
    if k not in bm25 or k not in bge:
        continue
    n += 1
    in_bm25 = gid in bm25[k]
    in_bge = gid in bge[k]
    if in_bm25 and in_bge: both += 1
    elif in_bm25: bm25_only += 1
    elif in_bge: bge_only += 1
    else: neither += 1
print(f"  n={n:,}  both={both:,} ({100*both/n:.1f}%)  "
      f"bm25-only={bm25_only:,} ({100*bm25_only/n:.1f}%)  "
      f"bge-only={bge_only:,} ({100*bge_only/n:.1f}%)  "
      f"neither={neither:,} ({100*neither/n:.1f}%)")
print(f"\n  union recall@100 (gold in EITHER): {(both+bm25_only+bge_only)/n:.4f}")
print(f"  this is the upper bound on RRF/reranker Hit@100")
