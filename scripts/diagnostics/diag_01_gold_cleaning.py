# scripts/diagnostics/diag_01_gold_cleaning.py
"""Verify gold pair cleaning removed leakage but preserved query semantics.

Checks:
1. Cleaned quote has no obvious case names (X v. Y patterns)
2. Cleaned quote has no bluebook citations (\d+ F.\d+d \d+)
3. Cleaned quote retains substantive legal content (avg word count > 10)
4. Quote-length distribution: not trivially short or empty
5. Sample 5 random pairs to inspect manually
"""
import json
import random
import re
from pathlib import Path

random.seed(0)
gold = Path("data/processed/baseline/cleaned/gold_pairs_test.jsonl")
rows = [json.loads(l) for l in gold.open()]
print(f"n_rows: {len(rows):,}")

case_pat = re.compile(r"\b[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+", re.IGNORECASE)
cite_pat = re.compile(r"\b\d+\s+F\.?\s*\d?\s*d?\s+\d+", re.IGNORECASE)
us_pat = re.compile(r"\b\d+\s+U\.\s*S\.\s*\d+", re.IGNORECASE)

n_case_leak = sum(1 for r in rows if case_pat.search(r["quote"]))
n_cite_leak = sum(1 for r in rows if cite_pat.search(r["quote"]) or us_pat.search(r["quote"]))
word_counts = [len(r["quote"].split()) for r in rows]
empty_count = sum(1 for w in word_counts if w == 0)

print(f"\n--- LEAKAGE CHECKS ---")
print(f"  case-name pattern hits  : {n_case_leak:,} ({100*n_case_leak/len(rows):.2f}%)")
print(f"  citation pattern hits   : {n_cite_leak:,} ({100*n_cite_leak/len(rows):.2f}%)")
print(f"\n--- LENGTH DISTRIBUTION ---")
word_counts.sort()
n = len(word_counts)
print(f"  min words      : {word_counts[0]}")
print(f"  p10            : {word_counts[n//10]}")
print(f"  p50            : {word_counts[n//2]}")
print(f"  p90            : {word_counts[int(n*0.9)]}")
print(f"  max words      : {word_counts[-1]}")
print(f"  empty quotes   : {empty_count}")

print(f"\n--- 5 RANDOM SAMPLES ---")
for r in random.sample(rows, 5):
    print(f"\n  source_id={r['source_id']}  dest_id={r['dest_id']}  cluster={r['source_cluster_id']}")
    print(f"  quote: {r['quote'][:200]}...")

# Verdict
verdict = "PASS"
if n_case_leak > len(rows) * 0.05:
    verdict = "FAIL: >5% case-name leakage"
if n_cite_leak > len(rows) * 0.05:
    verdict = "FAIL: >5% citation leakage"
if word_counts[n//2] < 10:
    verdict = "FAIL: median <10 words (over-cleaned)"
if empty_count > 0:
    verdict = "FAIL: empty quotes present"
print(f"\n=== VERDICT: {verdict} ===")
