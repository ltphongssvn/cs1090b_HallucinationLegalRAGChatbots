"""MS3 BM25 baseline retrieval.

Indexes data/processed/baseline/corpus_chunks.jsonl under bm25s (k1=1.5, b=0.75
per README Certified Baseline Stack), retrieves top-k chunks per query, and
aggregates chunk scores to opinion-level via MaxP (max chunk score per opinion).

Query text: LePaRD 'quote' field (citing context), joined by (source_id, dest_id)
from gold_pairs_test.jsonl → lepard_train_*.jsonl.

Outputs (data/processed/baseline/):
    bm25_results.jsonl    — one row per query: {source_id, dest_id, retrieved: [{opinion_id, score}]}
    bm25_summary.json     — BaselineBM25Summary (Pydantic-validated)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import bm25s

SCHEMA_VERSION = "1.0.0"
TOP_K = 100
BM25_K1 = 1.5
BM25_B = 0.75
RETRIEVAL_K_MULTIPLIER = 3  # over-retrieve chunks before MaxP aggregation

DEFAULT_CORPUS = Path("data/processed/baseline/corpus_chunks.jsonl")
DEFAULT_GOLD = Path("data/processed/baseline/gold_pairs_test.jsonl")
DEFAULT_LEPARD = Path("lepard_train_4000000_rev0194f95.jsonl")
DEFAULT_OUT_DIR = Path("data/processed/baseline")


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("baseline_bm25")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[baseline_bm25] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


# ---------- I/O ----------


def _iter_corpus(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _load_queries(
    gold_path: Path,
    lepard_path: Path,
) -> list[dict[str, Any]]:
    """Join gold pairs with LePaRD rows by (source_id, dest_id) to get quote text."""
    gold_keys: set[tuple[int, int]] = set()
    with gold_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold_keys.add((int(r["source_id"]), int(r["dest_id"])))

    queries: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int]] = set()
    with lepard_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (int(r["source_id"]), int(r["dest_id"]))
            if key in gold_keys and key not in seen_keys:
                seen_keys.add(key)
                queries.append(
                    {
                        "source_id": key[0],
                        "dest_id": key[1],
                        "query_text": r.get("quote", ""),
                    }
                )
    return queries


# ---------- aggregation (pure function for property testing) ----------


def _aggregate_chunk_scores(
    raw_hits: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Aggregate chunk-level hits to opinion-level via MaxP (max per opinion).

    Input: [{opinion_id, chunk_index, score}, ...]
    Output: [{opinion_id, score}, ...] sorted by score desc, len <= top_k.
    """
    best: dict[int, float] = {}
    for h in raw_hits:
        oid = h["opinion_id"]
        s = h["score"]
        if oid not in best or s > best[oid]:
            best[oid] = s
    ranked = sorted(
        ({"opinion_id": oid, "score": sc} for oid, sc in best.items()),
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[:top_k]


# ---------- provenance ----------


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()[:12]
        )
    except Exception:
        return "unknown"


# ---------- W&B ----------


def _log_to_wandb(summary: dict[str, Any], out_dir: Path) -> None:
    try:
        import wandb
    except ImportError:
        logger.info("  wandb unavailable — skipping telemetry")
        return
    run = wandb.init(
        entity="phl690-harvard-extension-schol",
        project="cs1090b",
        job_type="baseline-bm25",
        config=summary,
        reinit=True,
    )
    wandb.log(summary)
    art = wandb.Artifact("baseline-bm25", type="dataset")
    art.add_dir(str(out_dir))
    run.log_artifact(art)
    run.finish()


# ---------- main ----------


def main(
    corpus_path: Path = DEFAULT_CORPUS,
    gold_pairs_path: Path = DEFAULT_GOLD,
    lepard_path: Path = DEFAULT_LEPARD,
    out_dir: Path = DEFAULT_OUT_DIR,
    top_k: int = TOP_K,
    log_to_wandb: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    from src.eda_schemas import BaselineBM25Summary

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"MS3 BM25 baseline  (k1={BM25_K1}, b={BM25_B}, top_k={top_k}, seed={seed})")
    logger.info("=" * 60)
    logger.info(f"  corpus_path    : {corpus_path}")
    logger.info(f"  gold_pairs_path: {gold_pairs_path}")
    logger.info(f"  lepard_path    : {lepard_path}")
    logger.info(f"  out_dir        : {out_dir}")

    # --- Load corpus chunks ---
    logger.info("\n[1/4] Loading corpus chunks")
    chunks: list[dict[str, Any]] = list(_iter_corpus(corpus_path))
    chunk_texts = [c["text"] for c in chunks]
    chunk_meta = [(c["opinion_id"], c["chunk_index"]) for c in chunks]
    unique_opinions = len({m[0] for m in chunk_meta})
    logger.info(f"  chunks           : {len(chunks):,}")
    logger.info(f"  unique opinions  : {unique_opinions:,}")

    # --- Build BM25 index ---
    logger.info("\n[2/4] Building BM25 index (k1=%.2f, b=%.2f)" % (BM25_K1, BM25_B))
    t0 = time.perf_counter()
    tokenized_corpus = bm25s.tokenize(chunk_texts, stopwords="en")
    retriever = bm25s.BM25(k1=BM25_K1, b=BM25_B)
    retriever.index(tokenized_corpus)
    index_build_seconds = time.perf_counter() - t0
    logger.info(f"  index built in   : {index_build_seconds:.2f}s")

    # --- Load queries ---
    logger.info("\n[3/4] Loading queries")
    queries = _load_queries(gold_pairs_path, lepard_path)
    logger.info(f"  queries          : {len(queries):,}")

    # --- Retrieval ---
    logger.info("\n[4/4] Retrieving top-%d per query" % top_k)
    t0 = time.perf_counter()
    results_path = out_dir / "bm25_results.jsonl"
    # Request extra chunks (top_k * 3) so aggregation to opinion-level yields >= top_k
    retrieval_k = min(top_k * RETRIEVAL_K_MULTIPLIER, len(chunks))
    # Batch-tokenize all queries once, then retrieve with multi-threading.
    # bm25s 0.3.2 n_threads parameter parallelizes per-query scoring across
    # the batch; empirically ~10x faster than a Python per-query loop on 48-core AMD EPYC.
    query_texts = [q["query_text"] for q in queries]
    logger.info(f"  tokenizing {len(query_texts):,} queries")
    batched_tokens = bm25s.tokenize(query_texts, stopwords="en")
    logger.info("  retrieving (n_threads=48, batched)")
    all_indices, all_scores = retriever.retrieve(
        batched_tokens,
        k=retrieval_k,
        n_threads=16,
        show_progress=True,
    )
    # all_indices shape: (n_queries, retrieval_k)
    with results_path.open("w", encoding="utf-8") as fout:
        for qi, q in enumerate(queries):
            raw_hits = [
                {
                    "opinion_id": chunk_meta[int(idx)][0],
                    "chunk_index": chunk_meta[int(idx)][1],
                    "score": float(score),
                }
                for idx, score in zip(
                    all_indices[qi],
                    all_scores[qi],
                    strict=False,
                )
            ]
            aggregated = _aggregate_chunk_scores(raw_hits, top_k=top_k)
            fout.write(
                json.dumps(
                    {
                        "source_id": q["source_id"],
                        "dest_id": q["dest_id"],
                        "retrieved": aggregated,
                    }
                )
                + "\n"
            )
    retrieval_seconds = time.perf_counter() - t0
    logger.info(f"  retrieval done in: {retrieval_seconds:.2f}s")

    # --- Summary ---
    results_hash = hashlib.sha256(results_path.read_bytes()).hexdigest()
    summary_data = {
        "schema_version": SCHEMA_VERSION,
        "n_queries": len(queries),
        "n_corpus_chunks": len(chunks),
        "n_unique_opinions": unique_opinions,
        "top_k": top_k,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "index_build_seconds": round(index_build_seconds, 3),
        "retrieval_seconds": round(retrieval_seconds, 3),
        "seed": seed,
        "git_sha": _git_sha(),
        "results_hash": results_hash,
    }
    validated = BaselineBM25Summary.model_validate(summary_data)
    summary_path = out_dir / "bm25_summary.json"
    summary_path.write_text(
        json.dumps(validated.model_dump(), sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(f"\nWrote bm25_summary.json -> {summary_path}")

    if log_to_wandb:
        _log_to_wandb(summary_data, out_dir)

    return summary_data


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MS3 BM25 baseline retrieval.")
    ap.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--gold-pairs-path", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--lepard-path", type=Path, default=DEFAULT_LEPARD)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--log-to-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print(
            f"[baseline_bm25] DRY RUN  schema={SCHEMA_VERSION}  top_k={args.top_k}  "
            f"k1={BM25_K1}  b={BM25_B}  git_sha={_git_sha()}  "
            f"python={sys.version.split()[0]}  args={vars(args)}"
        )
        sys.exit(0)
    main(
        corpus_path=args.corpus_path,
        gold_pairs_path=args.gold_pairs_path,
        lepard_path=args.lepard_path,
        out_dir=args.out_dir,
        top_k=args.top_k,
        log_to_wandb=args.log_to_wandb,
        seed=args.seed,
    )


# ---------- verified-subset path ----------


REQUIRED_VERIFIED_QUERY_FIELDS = (
    "source_id",
    "source_cluster_id",
    "dest_id",
    "destination_context",
)


def _load_queries_verified(gold_path: Path) -> list[dict[str, Any]]:
    """Load queries directly from verified gold pairs.

    Verified contract:
      - query_text = destination_context (citing context, not the cited quote)
      - corpus key = source_cluster_id (cluster-aware retrieval)

    Unlike _load_queries, no LePaRD join needed — the verified gold file
    already contains all required fields in-line.
    """
    queries: list[dict[str, Any]] = []
    with gold_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            missing = [k for k in REQUIRED_VERIFIED_QUERY_FIELDS if k not in r]
            if missing:
                raise ValueError(
                    f"verified gold line {line_no} missing required fields: {missing}"
                )
            queries.append({
                "source_id": int(r["source_id"]),
                "source_cluster_id": int(r["source_cluster_id"]),
                "dest_id": int(r["dest_id"]),
                "query_text": r["destination_context"],
            })
    return queries


def main_verified(
    corpus_path: Path = DEFAULT_CORPUS,
    gold_pairs_path: Path = DEFAULT_GOLD,
    out_dir: Path = DEFAULT_OUT_DIR,
    top_k: int = TOP_K,
    log_to_wandb: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """BM25 retrieval keyed by source_cluster_id with destination_context queries.

    Verified contract:
      - corpus chunks must have cluster_id field
      - queries are loaded directly from verified gold (no LePaRD join)
      - aggregation produces cluster_id keys (not opinion_id)
    """
    from src.eda_schemas import BaselineBM25Summary

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(
        f"MS3 BM25 baseline VERIFIED (k1={BM25_K1}, b={BM25_B}, top_k={top_k}, seed={seed})"
    )
    logger.info("=" * 60)

    # --- Load corpus chunks ---
    chunks: list[dict[str, Any]] = list(_iter_corpus(corpus_path))
    chunk_texts = [c["text"] for c in chunks]
    # Verified path: store cluster_id as primary key, opinion_id as secondary
    chunk_meta = [
        (
            int(c["cluster_id"]),
            int(c.get("opinion_id", 0)),
            int(c["chunk_index"]),
        )
        for c in chunks
    ]
    unique_clusters = len({m[0] for m in chunk_meta})
    logger.info(f"  chunks: {len(chunks):,}  unique clusters: {unique_clusters:,}")

    # --- Build BM25 index ---
    t0 = time.perf_counter()
    tokenized_corpus = bm25s.tokenize(chunk_texts, stopwords="en")
    retriever = bm25s.BM25(k1=BM25_K1, b=BM25_B)
    retriever.index(tokenized_corpus)
    index_build_seconds = time.perf_counter() - t0
    logger.info(f"  index built in {index_build_seconds:.2f}s")

    # --- Load queries from verified gold ---
    queries = _load_queries_verified(gold_pairs_path)
    logger.info(f"  queries: {len(queries):,}")

    # --- Retrieval ---
    t0 = time.perf_counter()
    retrieval_k = min(top_k * RETRIEVAL_K_MULTIPLIER, len(chunks))
    query_texts = [q["query_text"] for q in queries]
    batched_tokens = bm25s.tokenize(query_texts, stopwords="en")
    all_indices, all_scores = retriever.retrieve(
        batched_tokens, k=retrieval_k, n_threads=16, show_progress=False
    )

    # Aggregate by cluster_id (not opinion_id)
    results_path = out_dir / "bm25_results.jsonl"
    with results_path.open("w", encoding="utf-8") as fout:
        for qi, q in enumerate(queries):
            best_per_cluster: dict[int, float] = {}
            for idx, score in zip(all_indices[qi], all_scores[qi], strict=False):
                cid = chunk_meta[int(idx)][0]
                s = float(score)
                if cid not in best_per_cluster or s > best_per_cluster[cid]:
                    best_per_cluster[cid] = s
            ranked = sorted(
                ({"cluster_id": cid, "score": sc} for cid, sc in best_per_cluster.items()),
                key=lambda x: x["score"],
                reverse=True,
            )[:top_k]
            fout.write(
                json.dumps(
                    {
                        "source_id": q["source_id"],
                        "source_cluster_id": q["source_cluster_id"],
                        "dest_id": q["dest_id"],
                        "retrieved": ranked,
                    }
                )
                + "\n"
            )
    retrieval_seconds = time.perf_counter() - t0
    logger.info(f"  retrieval done in {retrieval_seconds:.2f}s")

    results_hash = hashlib.sha256(results_path.read_bytes()).hexdigest()
    summary_data = {
        "schema_version": SCHEMA_VERSION,
        "n_queries": len(queries),
        "n_corpus_chunks": len(chunks),
        "n_unique_opinions": unique_clusters,
        "top_k": top_k,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "index_build_seconds": round(index_build_seconds, 3),
        "retrieval_seconds": round(retrieval_seconds, 3),
        "seed": seed,
        "git_sha": _git_sha(),
        "results_hash": results_hash,
    }
    validated = BaselineBM25Summary.model_validate(summary_data)
    summary_path = out_dir / "bm25_summary.json"
    summary_path.write_text(
        json.dumps(validated.model_dump(), sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if log_to_wandb:
        _log_to_wandb(summary_data, out_dir)
    return summary_data
