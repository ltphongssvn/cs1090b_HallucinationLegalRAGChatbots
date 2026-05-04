# scripts/run_baseline_eval.py
"""Reproducible CLI runner for baseline retrieval evaluation.

Wraps scripts/baseline_eval.py::evaluate_baseline + paired_comparison so any
retrieval result file (BM25, BGE-M3, RRF, reranker) can be evaluated against
gold pairs from the command line — no notebook cell or one-off Python -c
needed.

Two modes
---------
  single  : evaluate ONE results file → Hit@k / MRR / NDCG@10
  paired  : compare TWO results files → per-query win/tie counts

Both modes default to the verified-pipeline field selectors
(gold_field=source_cluster_id, match_field=cluster_id) but accept overrides
for legacy-mode evaluation.

Usage
-----
    # Single retriever evaluation (e.g. after RRF)
    .venv/bin/python scripts/run_baseline_eval.py single \
        --gold-path data/processed/baseline/cleaned/gold_pairs_test.jsonl \
        --results-path data/processed/baseline/cleaned/rrf_results.jsonl \
        --label "RRF (k=60)"

    # Paired comparison (for slide deck head-to-head)
    .venv/bin/python scripts/run_baseline_eval.py paired \
        --gold-path data/processed/baseline/cleaned/gold_pairs_test.jsonl \
        --bm25-path data/processed/baseline/cleaned/bm25_results.jsonl \
        --bge-m3-path data/processed/baseline/cleaned/bge_m3_results.jsonl

Outputs
-------
Mode `single` writes `<results_path>.eval.json` next to the results file with
schema {label, n_queries, n_skipped, hit_at_k, mrr, ndcg_at_10, gold_field,
match_field, results_hash}.

Mode `paired` writes `<out_dir>/paired_<labelA>_vs_<labelB>.json` with
schema {n_queries, A_wins, B_wins, ties, top_k, ...}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

# Ensure repo root on sys.path so `scripts.baseline_eval` resolves regardless
# of how this script is invoked (python scripts/run_baseline_eval.py ... vs
# python -m scripts.run_baseline_eval).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.baseline_eval import evaluate_baseline, paired_comparison


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("run_baseline_eval")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[run_baseline_eval] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


def _run_single(args: argparse.Namespace) -> int:
    gold = Path(args.gold_path)
    results = Path(args.results_path)
    label = args.label or results.stem

    logger.info("=" * 60)
    logger.info(f"Single-baseline evaluation: {label}")
    logger.info("=" * 60)
    logger.info(f"  gold_path    : {gold}")
    logger.info(f"  results_path : {results}")
    logger.info(f"  gold_field   : {args.gold_field}")
    logger.info(f"  match_field  : {args.match_field}")

    if not gold.is_file():
        raise FileNotFoundError(f"gold_path missing: {gold}")
    if not results.is_file():
        raise FileNotFoundError(f"results_path missing: {results}")

    m = evaluate_baseline(
        gold_path=gold,
        results_path=results,
        k_values=tuple(args.k_values),
        ndcg_k=args.ndcg_k,
        gold_field=args.gold_field,
        match_field=args.match_field,
    )

    logger.info(f"  n_queries    : {m['n_queries']:,}")
    logger.info(f"  n_skipped    : {m['n_skipped']:,}")
    for k in args.k_values:
        logger.info(f"  Hit@{k:<8}: {m['hit_at_k'][k]:.4f}")
    logger.info(f"  MRR          : {m['mrr']:.4f}")
    logger.info(f"  NDCG@{args.ndcg_k:<6} : {m['ndcg_at_10']:.4f}")

    # Persist eval artifact alongside results file
    eval_out = results.with_suffix(".eval.json")
    payload = {
        "label": label,
        "n_queries": m["n_queries"],
        "n_skipped": m["n_skipped"],
        "k_values": list(args.k_values),
        "ndcg_k": args.ndcg_k,
        "hit_at_k": {str(k): v for k, v in m["hit_at_k"].items()},
        "mrr": m["mrr"],
        "ndcg_at_10": m["ndcg_at_10"],
        "gold_field": args.gold_field,
        "match_field": args.match_field,
        "results_hash": hashlib.sha256(results.read_bytes()).hexdigest(),
    }
    eval_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(f"  wrote -> {eval_out}")
    return 0


def _run_paired(args: argparse.Namespace) -> int:
    gold = Path(args.gold_path)
    a = Path(args.bm25_path)
    b = Path(args.bge_m3_path)
    label_a = args.label_a or a.stem
    label_b = args.label_b or b.stem

    logger.info("=" * 60)
    logger.info(f"Paired comparison: {label_a} vs {label_b}")
    logger.info("=" * 60)
    logger.info(f"  gold_path  : {gold}")
    logger.info(f"  A path     : {a}  (label={label_a})")
    logger.info(f"  B path     : {b}  (label={label_b})")
    logger.info(f"  top_k      : {args.top_k}")

    for p in (gold, a, b):
        if not p.is_file():
            raise FileNotFoundError(f"input missing: {p}")

    c = paired_comparison(
        gold_path=gold,
        bm25_results_path=a,
        bge_m3_results_path=b,
        top_k=args.top_k,
        gold_field=args.gold_field,
        match_field=args.match_field,
    )

    n = c["n_queries"]
    if n == 0:
        logger.info("  no queries — nothing to compare")
        return 0

    a_wins = c["bm25_wins"]
    b_wins = c["bge_m3_wins"]
    ties = c["ties"]
    logger.info(f"  n_queries    : {n:,}")
    logger.info(f"  A wins ({label_a:<10}) : {a_wins:,} ({100*a_wins/n:.1f}%)")
    logger.info(f"  B wins ({label_b:<10}) : {b_wins:,} ({100*b_wins/n:.1f}%)")
    logger.info(f"  ties (incl. both-missed): {ties:,} ({100*ties/n:.1f}%)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_out = out_dir / f"paired_{label_a}_vs_{label_b}.json"
    payload = {
        "label_a": label_a,
        "label_b": label_b,
        "n_queries": n,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "top_k": c["top_k"],
        "gold_field": args.gold_field,
        "match_field": args.match_field,
    }
    eval_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(f"  wrote -> {eval_out}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reproducible CLI for baseline retrieval evaluation.",
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    # single
    s = sub.add_parser("single", help="Evaluate one retrieval results file.")
    s.add_argument("--gold-path", type=Path, required=True)
    s.add_argument("--results-path", type=Path, required=True)
    s.add_argument("--label", type=str, default=None,
                   help="Display label (defaults to results filename stem).")
    s.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 100])
    s.add_argument("--ndcg-k", type=int, default=10)
    s.add_argument("--gold-field", type=str, default="source_cluster_id")
    s.add_argument("--match-field", type=str, default="cluster_id")
    s.set_defaults(func=_run_single)

    # paired
    p = sub.add_parser("paired", help="Compare two retrieval results files head-to-head.")
    p.add_argument("--gold-path", type=Path, required=True)
    p.add_argument("--bm25-path", type=Path, required=True,
                   help="(arg name kept for API compat with paired_comparison) — first results file.")
    p.add_argument("--bge-m3-path", type=Path, required=True,
                   help="(arg name kept for API compat) — second results file.")
    p.add_argument("--label-a", type=str, default=None)
    p.add_argument("--label-b", type=str, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--gold-field", type=str, default="source_cluster_id")
    p.add_argument("--match-field", type=str, default="cluster_id")
    p.set_defaults(func=_run_paired)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
