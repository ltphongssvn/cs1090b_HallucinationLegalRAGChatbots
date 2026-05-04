# scripts/baseline_eval.py
"""MS3 retrieval evaluation — Hit@k, MRR, NDCG@10 over BM25 + BGE-M3 results.

Pure-Python streaming evaluator. Reads per-query retrieval output (produced by
scripts/baseline_bm25.py and scripts/baseline_bge_m3.py) alongside
gold_pairs_test.jsonl and computes standard IR metrics. Enables a paired
per-query comparison between the two baselines.

Metrics
-------
Hit@k       — fraction of queries where gold id appears in top-k retrieved
MRR         — mean reciprocal rank of gold across all queries (0 if missed)
NDCG@10     — normalized DCG with binary relevance at k=10 (pinned by MS3 contract)

Two evaluation modes (selected via gold_field / match_field):
  - LEGACY  (default): gold_field="dest_id",            match_field="opinion_id"
  - VERIFIED         : gold_field="source_cluster_id",  match_field="cluster_id"

The verified mode supports the cleaned-corpus pipeline where retrieval is keyed
by cluster_id (deduplicated opinion clusters) instead of raw opinion_id.
"""
from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
NDCG_K = 10  # Pinned by MS3 contract — output key is always "ndcg_at_10"


# ---------- provenance ----------
def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()[:12]
    except Exception:
        return "unknown"


# ---------- I/O ----------
def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_gold(
    gold_path: Path,
    *,
    gold_field: str = "dest_id",
) -> dict[tuple[int, int], int]:
    """Returns {(source_id, dest_id): gold_field_value}.

    The composite key (source_id, dest_id) uniquely identifies each LePaRD
    citation — one source opinion typically cites many dest opinions, each
    a distinct query. The mapped value is the *match target* used by the
    evaluator to score retrieved hits:
      - gold_field="dest_id"           → opinion-level matching (legacy)
      - gold_field="source_cluster_id" → cluster-level matching (verified)

    Decoupling the join key (always dest_id) from the match target lets the
    verified pipeline aggregate retrieval at cluster granularity without
    collapsing the 45K distinct queries into ~1.5K source_ids.
    """
    gold: dict[tuple[int, int], int] = {}
    for row in _iter_jsonl(gold_path):
        key = (int(row["source_id"]), int(row["dest_id"]))
        gold[key] = int(row[gold_field])
    return gold


# ---------- pure metric functions (unit-testable) ----------
def _find_rank(
    retrieved: list[dict[str, Any]],
    *,
    gold_id: int,
    match_field: str = "opinion_id",
) -> int:
    """Return 1-indexed rank of gold_id in retrieved, or 0 if not present.

    Trusts upstream ordering (MaxP aggregation already applied deterministic
    tie-break via (-score, match_field ascending)).

    match_field selects which retrieved attribute to compare against gold_id:
      - "opinion_id"  (legacy)
      - "cluster_id"  (verified)
    """
    for i, hit in enumerate(retrieved, start=1):
        if int(hit[match_field]) == int(gold_id):
            return i
    return 0


def _metrics_from_ranks(
    *,
    ranks: list[int],
    k_values: tuple[int, ...],
    ndcg_k: int = NDCG_K,
) -> dict[str, Any]:
    """Compute Hit@k, MRR, NDCG@10 from a list of per-query ranks.

    ndcg_k parameter is retained for test flexibility but the MS3 contract
    pins the output key to "ndcg_at_10". Callers that pass ndcg_k != 10 still
    get the metric computed against that cutoff, keyed as "ndcg_at_10".

    rank=0 means gold was not retrieved (contributes 0 to all metrics).
    """
    n = len(ranks)
    if n == 0:
        return {
            "hit_at_k": {k: 0.0 for k in k_values},
            "mrr": 0.0,
            "ndcg_at_10": 0.0,
        }

    hit_at_k = {k: 0 for k in k_values}
    reciprocal_sum = 0.0
    ndcg_sum = 0.0

    for r in ranks:
        if r == 0:
            continue
        reciprocal_sum += 1.0 / r
        for k in k_values:
            if r <= k:
                hit_at_k[k] += 1
        if r <= ndcg_k:
            ndcg_sum += 1.0 / math.log2(1 + r)

    return {
        "hit_at_k": {k: hit_at_k[k] / n for k in k_values},
        "mrr": reciprocal_sum / n,
        "ndcg_at_10": ndcg_sum / n,
    }


# ---------- public API ----------
def evaluate_baseline(
    *,
    gold_path: Path,
    results_path: Path,
    k_values: tuple[int, ...] = (1, 5, 10, 100),
    ndcg_k: int = NDCG_K,
    gold_field: str = "dest_id",
    match_field: str = "opinion_id",
) -> dict[str, Any]:
    """Compute Hit@k, MRR, NDCG@10 for one baseline's results file.

    Queries in results but not in gold are silently skipped (recorded as
    n_skipped in the returned dict) — possible only if the gold set was
    filtered after retrieval ran.

    For the verified pipeline (cleaned corpus + cluster_id matching), pass:
        gold_field="source_cluster_id", match_field="cluster_id"
    """
    gold = _load_gold(Path(gold_path), gold_field=gold_field)
    ranks: list[int] = []
    n_skipped = 0

    for row in _iter_jsonl(Path(results_path)):
        key = (int(row["source_id"]), int(row["dest_id"]))
        if key not in gold:
            n_skipped += 1
            continue
        gold_id = gold[key]
        r = _find_rank(row["retrieved"], gold_id=gold_id, match_field=match_field)
        ranks.append(r)

    m = _metrics_from_ranks(ranks=ranks, k_values=k_values, ndcg_k=ndcg_k)
    m["n_queries"] = len(ranks)
    m["n_skipped"] = n_skipped
    return m


def paired_comparison(
    *,
    gold_path: Path,
    bm25_results_path: Path,
    bge_m3_results_path: Path,
    top_k: int = 100,
    gold_field: str = "dest_id",
    match_field: str = "opinion_id",
) -> dict[str, Any]:
    """Per-query rank comparison between BM25 and BGE-M3 at a fixed top_k cutoff.

    Streams both result files row-by-row (no full in-memory indexing) on the
    assumption that upstream baselines emit queries in gold order. If any row
    pair disagrees on (source_id, gold_field), raises ValueError.

    A "win" means strictly better rank within top_k (smaller rank = better).
    Ranks beyond top_k count as misses (rank=inf). Both-miss → tie.
    """
    gold = _load_gold(Path(gold_path), gold_field=gold_field)

    bge_wins = 0
    bm25_wins = 0
    ties = 0
    n_queries = 0

    with (
        Path(bm25_results_path).open(encoding="utf-8") as bm25_f,
        Path(bge_m3_results_path).open(encoding="utf-8") as bge_f,
    ):
        for bm25_line, bge_line in zip(bm25_f, bge_f, strict=True):
            bm25_row = json.loads(bm25_line)
            bge_row = json.loads(bge_line)

            key_bm25 = (int(bm25_row["source_id"]), int(bm25_row["dest_id"]))
            key_bge = (int(bge_row["source_id"]), int(bge_row["dest_id"]))

            if key_bm25 != key_bge:
                raise ValueError(
                    f"paired_comparison requires aligned result files; BM25 row={key_bm25} vs BGE-M3 row={key_bge}"
                )

            if key_bm25 not in gold:
                continue

            gold_id = gold[key_bm25]
            bm25_rank = _find_rank(bm25_row["retrieved"], gold_id=gold_id, match_field=match_field)
            bge_rank = _find_rank(bge_row["retrieved"], gold_id=gold_id, match_field=match_field)

            n_queries += 1

            def effective(r: int) -> float:
                """Rank beyond top_k cutoff (or 0=missed) counts as worst."""
                return float("inf") if r == 0 or r > top_k else float(r)

            b = effective(bm25_rank)
            g = effective(bge_rank)

            if g < b:
                bge_wins += 1
            elif b < g:
                bm25_wins += 1
            else:
                ties += 1

    return {
        "n_queries": n_queries,
        "bge_m3_wins": bge_wins,
        "bm25_wins": bm25_wins,
        "ties": ties,
        "top_k": top_k,
    }
