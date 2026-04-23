"""MS3 retrieval evaluation — Hit@k, MRR, NDCG@10 over BM25 + BGE-M3 results.

Pure-Python streaming evaluator. Reads per-query retrieval output (produced by
scripts/baseline_bm25.py and scripts/baseline_bge_m3.py) alongside
gold_pairs_test.jsonl and computes standard IR metrics. Enables a paired
per-query comparison between the two baselines.

Metrics
-------
Hit@k       — fraction of queries where gold source_id appears in top-k retrieved
MRR         — mean reciprocal rank of gold across all queries (0 if missed)
NDCG@10     — normalized DCG with binary relevance at k=10 (pinned by MS3 contract)

All metrics operate on opinion-level retrieved lists (upstream MaxP already
aggregated chunk-level scores). Both baselines evaluated on the same 45K test
queries from gold_pairs_test.jsonl.

This module returns plain dicts; the downstream caller (notebook Cell 15)
wraps the output in BaselineEvalSummary (defined in src.eda_schemas) and
performs Pydantic validation before writing eval_summary.json.

No external dependencies beyond stdlib. Metric correctness is locked by the
test suite (contract + unit + Hypothesis property invariants).
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


def _load_gold(gold_path: Path) -> dict[tuple[int, int], int]:
    """Returns {(source_id, dest_id): source_id}.

    In LePaRD: source = cited precedent (what we retrieve), dest = citing case
    (where the destination_context query comes from). The gold document is
    source_id (the cited opinion whose chunks should rank first).
    """
    gold: dict[tuple[int, int], int] = {}
    for row in _iter_jsonl(gold_path):
        key = (int(row["source_id"]), int(row["dest_id"]))
        gold[key] = int(row["source_id"])
    return gold


# ---------- pure metric functions (unit-testable) ----------


def _find_rank(retrieved: list[dict[str, Any]], *, gold_id: int) -> int:
    """Return 1-indexed rank of gold_id in retrieved, or 0 if not present.

    Trusts upstream ordering (MaxP aggregation already applied deterministic
    tie-break via (-score, opinion_id ascending)).
    """
    for i, hit in enumerate(retrieved, start=1):
        if int(hit["opinion_id"]) == int(gold_id):
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
) -> dict[str, Any]:
    """Compute Hit@k, MRR, NDCG@10 for one baseline's results file.

    Queries in results but not in gold are silently skipped (recorded as
    n_skipped in the returned dict) — possible only if the gold set was
    filtered after retrieval ran.
    """
    gold = _load_gold(Path(gold_path))
    ranks: list[int] = []
    n_skipped = 0
    for row in _iter_jsonl(Path(results_path)):
        key = (int(row["source_id"]), int(row["dest_id"]))
        if key not in gold:
            n_skipped += 1
            continue
        gold_id = gold[key]
        r = _find_rank(row["retrieved"], gold_id=gold_id)
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
) -> dict[str, Any]:
    """Per-query rank comparison between BM25 and BGE-M3 at a fixed top_k cutoff.

    Streams both result files row-by-row (no full in-memory indexing) on the
    assumption that upstream baselines emit queries in gold order. If any row
    pair disagrees on (source_id, dest_id), raises ValueError.

    A "win" means strictly better rank within top_k (smaller rank = better).
    Ranks beyond top_k count as misses (rank=inf). Both-miss → tie.
    """
    gold = _load_gold(Path(gold_path))

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

            bm25_rank = _find_rank(bm25_row["retrieved"], gold_id=gold_id)
            bge_rank = _find_rank(bge_row["retrieved"], gold_id=gold_id)
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
