# scripts/diagnostics/diag_05_reranker_vs_rrf.py
"""Confirm reranker did its job (changed orders) but degraded quality.

Compares reranker output to RRF input row-by-row to see:
- How often the reranker moved the gold cluster up vs down
- Whether the reranker is a no-op (preserves RRF order)
- How often candidate set is identical (only reorder, no replace)
"""
import json
from pathlib import Path

cleaned = Path("data/processed/baseline/cleaned")
rrf_path = cleaned / "rrf_results.jsonl"
rer_path = cleaned / "reranker_results.jsonl"

rrf_rows = {(r["source_id"], r["dest_id"]): r for r in (json.loads(l) for l in rrf_path.open())}
rer_rows = {(r["source_id"], r["dest_id"]): r for r in (json.loads(l) for l in rer_path.open())}

shared_keys = set(rrf_rows) & set(rer_rows)
print(f"shared keys: {len(shared_keys):,}")

n_same_set = 0
n_same_order = 0
gold_moved_up = 0
gold_moved_down = 0
gold_unchanged = 0
gold_appeared = 0
gold_disappeared = 0

for k in shared_keys:
    rrf_ids = [int(h["cluster_id"]) for h in rrf_rows[k]["retrieved"]]
    rer_ids = [int(h["cluster_id"]) for h in rer_rows[k]["retrieved"]]
    gold_id = int(rrf_rows[k]["source_cluster_id"])

    if set(rrf_ids) == set(rer_ids):
        n_same_set += 1
    if rrf_ids == rer_ids:
        n_same_order += 1

    rrf_rank = rrf_ids.index(gold_id) + 1 if gold_id in rrf_ids else 0
    rer_rank = rer_ids.index(gold_id) + 1 if gold_id in rer_ids else 0
    if rrf_rank == 0 and rer_rank > 0:
        gold_appeared += 1
    elif rrf_rank > 0 and rer_rank == 0:
        gold_disappeared += 1
    elif rrf_rank > 0 and rer_rank > 0:
        if rer_rank < rrf_rank:
            gold_moved_up += 1
        elif rer_rank > rrf_rank:
            gold_moved_down += 1
        else:
            gold_unchanged += 1

n = len(shared_keys)
print(f"\n--- CANDIDATE SET COMPARISON ---")
print(f"  same candidate set      : {n_same_set:,} ({100*n_same_set/n:.1f}%)")
print(f"  same order (no-op)      : {n_same_order:,} ({100*n_same_order/n:.1f}%)")

print(f"\n--- GOLD POSITION CHANGES ---")
print(f"  gold moved UP   (better): {gold_moved_up:,} ({100*gold_moved_up/n:.1f}%)")
print(f"  gold moved DOWN (worse) : {gold_moved_down:,} ({100*gold_moved_down/n:.1f}%)")
print(f"  gold unchanged          : {gold_unchanged:,} ({100*gold_unchanged/n:.1f}%)")
print(f"  gold appeared (new)     : {gold_appeared:,}")
print(f"  gold disappeared        : {gold_disappeared:,}")

print(f"\n--- VERDICT ---")
if gold_moved_down > gold_moved_up:
    diff = gold_moved_down - gold_moved_up
    print(f"  CONFIRMED: reranker moved gold DOWN more than UP by {diff:,} queries")
    print(f"  net rank degradation explains the Hit@k drop")
else:
    print(f"  reranker moved gold up more than down — drop must be elsewhere")
