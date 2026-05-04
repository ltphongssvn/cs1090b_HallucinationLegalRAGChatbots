# scripts/mine_hard_negatives.py
"""Hard-negative mining for bge-reranker fine-tuning on cleaned LePaRD.

For each gold (source_id, dest_id) pair where the gold cluster_id appears
in the RRF top-100, sample N hard negatives from the same retrieval list
(rank 2-100, excluding the gold cluster). Output JSONL of training rows
in BAAI FlagEmbedding format: {query, pos: [pos_text], neg: [neg_text, ...]}.

Hard-negative range
-------------------
Per Karpukhin et al. 2020 (DPR) + 4Huiter 2024 (legal): negatives drawn
from rank 2-100 of the first-stage retriever are "hard" (semantically close
to query but wrong). True random negatives are too easy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Repo root on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = "1.0.0"
DEFAULT_N_NEGATIVES_PER_POSITIVE = 7
DEFAULT_NEG_RANK_RANGE = (2, 100)
DEFAULT_MAX_CHUNKS_PER_CLUSTER = 2


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("mine_hard_negatives")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[mine_hard_negatives] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def mine(
    *,
    gold_path: Path,
    rrf_path: Path,
    n_neg_per_pos: int = DEFAULT_N_NEGATIVES_PER_POSITIVE,
    neg_rank_range: tuple[int, int] = DEFAULT_NEG_RANK_RANGE,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Mine hard negatives. Returns list of training rows.

    Each row: {query, source_id, dest_id, pos_cluster_id, neg_cluster_ids}.
    Caller materializes pos/neg text from corpus separately (memory-friendly).
    """
    gold_path = Path(gold_path)
    rrf_path = Path(rrf_path)
    rng = random.Random(seed)

    # Index gold by (source_id, dest_id) → (source_cluster_id, quote)
    gold: dict[tuple[int, int], dict[str, Any]] = {}
    for r in _iter_jsonl(gold_path):
        key = (int(r["source_id"]), int(r["dest_id"]))
        if key in gold:
            continue
        gold[key] = {
            "source_cluster_id": int(r["source_cluster_id"]),
            "quote": r["quote"],
        }

    rmin, rmax = neg_rank_range
    rows: list[dict[str, Any]] = []
    n_skipped_no_gold = 0

    for r in _iter_jsonl(rrf_path):
        key = (int(r["source_id"]), int(r["dest_id"]))
        if key not in gold:
            continue
        gold_cid = gold[key]["source_cluster_id"]
        retrieved_ids = [int(h["cluster_id"]) for h in r["retrieved"]]
        if gold_cid not in retrieved_ids:
            n_skipped_no_gold += 1
            continue
        # Negative pool: ranks rmin..rmax, excluding gold cluster
        neg_pool: list[int] = []
        for rank, cid in enumerate(retrieved_ids, start=1):
            if rank < rmin or rank > rmax:
                continue
            if cid == gold_cid:
                continue
            neg_pool.append(cid)
        if len(neg_pool) < n_neg_per_pos:
            negs = neg_pool[:]
        else:
            negs = rng.sample(neg_pool, n_neg_per_pos)
        if not negs:
            continue
        rows.append({
            "query": gold[key]["quote"],
            "source_id": key[0],
            "dest_id": key[1],
            "pos_cluster_id": gold_cid,
            "neg_cluster_ids": negs,
        })

    if n_skipped_no_gold:
        logger.info(
            f"  skipped {n_skipped_no_gold:,} queries where gold not in RRF top-100"
        )
    return rows


def _load_cluster_text_index(
    corpus_path: Path,
    *,
    max_chunks_per_cluster: int = DEFAULT_MAX_CHUNKS_PER_CLUSTER,
    cluster_filter: set[int] | None = None,
) -> dict[int, str]:
    """Build cluster_id -> first-N-chunks-concatenated index."""
    chunks_by_cluster: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for row in _iter_jsonl(Path(corpus_path)):
        cid = int(row["cluster_id"])
        if cluster_filter is not None and cid not in cluster_filter:
            continue
        if len(chunks_by_cluster[cid]) >= max_chunks_per_cluster:
            continue
        chunks_by_cluster[cid].append((int(row["chunk_index"]), str(row["text"])))
    out: dict[int, str] = {}
    for cid, chunks in chunks_by_cluster.items():
        chunks.sort(key=lambda t: t[0])
        out[cid] = " ".join(t for _, t in chunks)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Mine hard negatives for reranker fine-tuning.")
    ap.add_argument("--gold-path", type=Path, required=True)
    ap.add_argument("--rrf-path", type=Path, required=True)
    ap.add_argument("--corpus-path", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument("--n-neg-per-pos", type=int, default=DEFAULT_N_NEGATIVES_PER_POSITIVE)
    ap.add_argument("--neg-rank-min", type=int, default=DEFAULT_NEG_RANK_RANGE[0])
    ap.add_argument("--neg-rank-max", type=int, default=DEFAULT_NEG_RANK_RANGE[1])
    ap.add_argument("--max-chunks-per-cluster", type=int, default=DEFAULT_MAX_CHUNKS_PER_CLUSTER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.05,
                    help="Fraction held out as validation set")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    logger.info("=" * 60)
    logger.info("MS5 hard-negative mining for reranker fine-tuning")
    logger.info("=" * 60)
    logger.info(f"  gold_path     : {args.gold_path}")
    logger.info(f"  rrf_path      : {args.rrf_path}")
    logger.info(f"  corpus_path   : {args.corpus_path}")
    logger.info(f"  out_path      : {args.out_path}")
    logger.info(f"  n_neg_per_pos : {args.n_neg_per_pos}")
    logger.info(f"  neg_rank_range: [{args.neg_rank_min}, {args.neg_rank_max}]")

    rows = mine(
        gold_path=args.gold_path,
        rrf_path=args.rrf_path,
        n_neg_per_pos=args.n_neg_per_pos,
        neg_rank_range=(args.neg_rank_min, args.neg_rank_max),
        seed=args.seed,
    )
    logger.info(f"  mined {len(rows):,} training rows")

    # Materialize cluster text — only for clusters we need
    needed: set[int] = set()
    for r in rows:
        needed.add(r["pos_cluster_id"])
        needed.update(r["neg_cluster_ids"])
    logger.info(f"  resolving text for {len(needed):,} unique clusters from corpus")
    cluster_text = _load_cluster_text_index(
        args.corpus_path,
        max_chunks_per_cluster=args.max_chunks_per_cluster,
        cluster_filter=needed,
    )
    logger.info(f"  resolved {len(cluster_text):,} clusters")

    # Train/val split (deterministic by source_id hash)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    train_path = args.out_path
    val_path = args.out_path.with_suffix(".val.jsonl")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = int(len(rows) * args.val_fraction)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    def _emit(path: Path, rs: list[dict[str, Any]]) -> int:
        n_written = 0
        with path.open("w", encoding="utf-8") as fout:
            for r in rs:
                pos_text = cluster_text.get(r["pos_cluster_id"])
                if pos_text is None:
                    continue
                neg_texts = [
                    cluster_text[cid]
                    for cid in r["neg_cluster_ids"]
                    if cid in cluster_text
                ]
                if not neg_texts:
                    continue
                fout.write(json.dumps({
                    "query": r["query"],
                    "pos": [pos_text],
                    "neg": neg_texts,
                    "source_id": r["source_id"],
                    "dest_id": r["dest_id"],
                    "pos_cluster_id": r["pos_cluster_id"],
                    "neg_cluster_ids": r["neg_cluster_ids"],
                }, allow_nan=False) + "\n")
                n_written += 1
        return n_written

    n_train = _emit(train_path, train_rows)
    n_val_emitted = _emit(val_path, val_rows)
    logger.info(f"  wrote train: {n_train:,} -> {train_path}")
    logger.info(f"  wrote val  : {n_val_emitted:,} -> {val_path}")

    # Provenance
    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_train": n_train,
        "n_val": n_val_emitted,
        "n_neg_per_pos": args.n_neg_per_pos,
        "neg_rank_range": [args.neg_rank_min, args.neg_rank_max],
        "max_chunks_per_cluster": args.max_chunks_per_cluster,
        "seed": args.seed,
        "train_hash": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "val_hash": hashlib.sha256(val_path.read_bytes()).hexdigest() if val_path.exists() else "",
    }
    summary_path = args.out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(f"  wrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
