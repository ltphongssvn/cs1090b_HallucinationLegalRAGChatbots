# scripts/stratified_eval.py
"""Frequency-stratified retrieval evaluation — addresses long-tail/popularity bias.

Per Steck (2011) popularity-stratified recall and the LePaRD paper's own
finding that retrievers exhibit "an inverse relationship between model
performance and passage frequency," report Hit@k / MRR / NDCG separately
for HEAD / TORSO / TAIL gold-cluster popularity buckets. This isolates
whether a retriever's aggregate score reflects genuine retrieval ability
or just lexical alignment with frequently-cited "popular" precedents.

Bucketing
---------
- HEAD  : top n_buckets-quantile most-cited gold cluster_ids
- TAIL  : bottom n_buckets-quantile least-cited
- TORSO : middle (when n_buckets >= 3)

Frequency = count of (source_id, dest_id) gold rows mapping to that cluster.

Outputs
-------
{
  "n_buckets": 3,
  "overall": {n_queries, hit_at_k, mrr, ndcg_at_10},
  "per_bucket": {
    "head":  {n_queries, gold_freq_range, hit_at_k, mrr, ndcg_at_10},
    "torso": {...},
    "tail":  {...}
  }
}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Ensure repo root on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.baseline_eval import (
    NDCG_K,
    SCHEMA_VERSION,
    _find_rank,
    _git_sha,
    _iter_jsonl,
    _load_gold,
    _metrics_from_ranks,
)


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("stratified_eval")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[stratified_eval] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


def _compute_cluster_frequencies(gold_path: Path) -> dict[int, int]:
    """Count (source_id, dest_id) gold rows per source_cluster_id."""
    counts: Counter[int] = Counter()
    for row in _iter_jsonl(Path(gold_path)):
        counts[int(row["source_cluster_id"])] += 1
    return dict(counts)


def _assign_buckets(
    freq: dict[int, int],
    *,
    n_buckets: int = 3,
) -> dict[int, str]:
    """Quantile-bucket cluster_ids by frequency. n_buckets in {2, 3}.

    Ties at quantile boundaries collapse to the same bucket (no cluster
    is split across buckets just because of a frequency tie).
    """
    if n_buckets not in (2, 3):
        raise ValueError(f"n_buckets must be 2 or 3, got {n_buckets}")
    if not freq:
        return {}
    sorted_items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    n = len(sorted_items)

    # All ties → single bucket
    unique_freqs = {f for _, f in sorted_items}
    if len(unique_freqs) == 1:
        only = "head" if n_buckets >= 2 else "head"
        return {cid: only for cid, _ in sorted_items}

    if n_buckets == 2:
        cut = n // 2
        head_set = {cid for cid, _ in sorted_items[:cut]}
        return {cid: ("head" if cid in head_set else "tail") for cid, _ in sorted_items}
    else:  # 3
        cut1 = n // 3
        cut2 = (2 * n) // 3
        out: dict[int, str] = {}
        for i, (cid, _) in enumerate(sorted_items):
            if i < cut1:
                out[cid] = "head"
            elif i < cut2:
                out[cid] = "torso"
            else:
                out[cid] = "tail"
        return out


def evaluate_stratified(
    *,
    gold_path: Path,
    results_path: Path,
    n_buckets: int = 3,
    k_values: tuple[int, ...] = (1, 5, 10, 100),
    ndcg_k: int = NDCG_K,
    gold_field: str = "source_cluster_id",
    match_field: str = "cluster_id",
) -> dict[str, Any]:
    """Compute Hit@k / MRR / NDCG@10 overall AND per popularity bucket."""
    gold = _load_gold(Path(gold_path), gold_field=gold_field)
    freq = _compute_cluster_frequencies(Path(gold_path))
    buckets = _assign_buckets(freq, n_buckets=n_buckets)

    # Collect ranks per bucket + overall
    ranks_by_bucket: dict[str, list[int]] = {b: [] for b in set(buckets.values())}
    ranks_overall: list[int] = []
    n_skipped = 0

    for row in _iter_jsonl(Path(results_path)):
        key = (int(row["source_id"]), int(row["dest_id"]))
        if key not in gold:
            n_skipped += 1
            continue
        gold_id = gold[key]
        bucket = buckets.get(int(row[gold_field]))
        r = _find_rank(row["retrieved"], gold_id=gold_id, match_field=match_field)
        ranks_overall.append(r)
        if bucket is not None:
            ranks_by_bucket[bucket].append(r)

    overall = _metrics_from_ranks(ranks=ranks_overall, k_values=k_values, ndcg_k=ndcg_k)
    overall["n_queries"] = len(ranks_overall)
    overall["n_skipped"] = n_skipped

    per_bucket: dict[str, dict[str, Any]] = {}
    for bname, ranks in ranks_by_bucket.items():
        m = _metrics_from_ranks(ranks=ranks, k_values=k_values, ndcg_k=ndcg_k)
        m["n_queries"] = len(ranks)
        # Frequency range of clusters in this bucket
        cluster_ids_in_bucket = [c for c, b in buckets.items() if b == bname]
        if cluster_ids_in_bucket:
            freqs = [freq[c] for c in cluster_ids_in_bucket]
            m["gold_freq_min"] = min(freqs)
            m["gold_freq_max"] = max(freqs)
            m["n_clusters_in_bucket"] = len(cluster_ids_in_bucket)
        per_bucket[bname] = m

    return {
        "schema_version": SCHEMA_VERSION,
        "n_buckets": n_buckets,
        "overall": overall,
        "per_bucket": per_bucket,
        "gold_field": gold_field,
        "match_field": match_field,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Frequency-stratified retrieval evaluation (HEAD/TORSO/TAIL).",
    )
    ap.add_argument("--gold-path", type=Path, required=True)
    ap.add_argument("--results-path", type=Path, required=True)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--n-buckets", type=int, default=3, choices=(2, 3))
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 100])
    ap.add_argument("--ndcg-k", type=int, default=10)
    ap.add_argument("--gold-field", type=str, default="source_cluster_id")
    ap.add_argument("--match-field", type=str, default="cluster_id")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    label = args.label or Path(args.results_path).stem
    out = evaluate_stratified(
        gold_path=args.gold_path,
        results_path=args.results_path,
        n_buckets=args.n_buckets,
        k_values=tuple(args.k_values),
        ndcg_k=args.ndcg_k,
        gold_field=args.gold_field,
        match_field=args.match_field,
    )
    out["label"] = label
    out["results_hash"] = hashlib.sha256(Path(args.results_path).read_bytes()).hexdigest()
    out["git_sha"] = _git_sha()

    eval_path = Path(args.results_path).with_suffix(".stratified.json")
    eval_path.write_text(
        json.dumps(out, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    logger.info("=" * 70)
    logger.info(f"Stratified evaluation: {label}  (n_buckets={args.n_buckets})")
    logger.info("=" * 70)
    logger.info(f"  OVERALL n_queries={out['overall']['n_queries']:,}")
    for k in args.k_values:
        logger.info(f"    Hit@{k:<3}: {out['overall']['hit_at_k'][k]:.4f}")
    logger.info(f"    MRR    : {out['overall']['mrr']:.4f}")
    logger.info(f"    NDCG@10: {out['overall']['ndcg_at_10']:.4f}")
    logger.info("")
    for bname in ("head", "torso", "tail"):
        if bname not in out["per_bucket"]:
            continue
        b = out["per_bucket"][bname]
        logger.info(
            f"  {bname.upper():<5} n_queries={b['n_queries']:,}  "
            f"clusters={b.get('n_clusters_in_bucket', 0):,}  "
            f"gold_freq=[{b.get('gold_freq_min', 0)}-{b.get('gold_freq_max', 0)}]"
        )
        for k in args.k_values:
            logger.info(f"    Hit@{k:<3}: {b['hit_at_k'][k]:.4f}")
        logger.info(f"    MRR    : {b['mrr']:.4f}")
        logger.info(f"    NDCG@10: {b['ndcg_at_10']:.4f}")
    logger.info(f"\n  wrote -> {eval_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
