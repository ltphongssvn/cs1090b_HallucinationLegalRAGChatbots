"""MS3 BM25 baseline retrieval.

Two paths:
  - main(): legacy path (joins gold pairs with LePaRD by source_id/dest_id, query=quote)
  - main_verified(): cluster-aware path (corpus key=cluster_id, query=destination_context)

Outputs (data/processed/baseline/):
    bm25_results.jsonl    — per-query top-k retrieved
    bm25_summary.json     — BaselineBM25Summary (Pydantic-validated)
    bm25_failures.jsonl   — per-query miss debug (verified path only)
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
RETRIEVAL_K_MULTIPLIER = 3

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


def _iter_corpus(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _load_queries(
    gold_path: Path,
    lepard_path: Path,
) -> list[dict[str, Any]]:
    """Legacy path: join gold pairs with LePaRD rows by (source_id, dest_id)."""
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
                queries.append({
                    "source_id": key[0],
                    "dest_id": key[1],
                    "query_text": r.get("quote", ""),
                })
    return queries


def _aggregate_chunk_scores(
    raw_hits: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Legacy path: aggregate by opinion_id."""
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


def _log_to_wandb(summary: dict[str, Any], out_dir: Path) -> None:
    try:
        import wandb
    except ImportError:
        logger.info("  wandb unavailable")
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


# ---------- legacy main ----------


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

    chunks: list[dict[str, Any]] = list(_iter_corpus(corpus_path))
    chunk_texts = [c["text"] for c in chunks]
    chunk_meta = [(c["opinion_id"], c["chunk_index"]) for c in chunks]
    unique_opinions = len({m[0] for m in chunk_meta})

    t0 = time.perf_counter()
    tokenized_corpus = bm25s.tokenize(chunk_texts, stopwords="en")
    retriever = bm25s.BM25(k1=BM25_K1, b=BM25_B)
    retriever.index(tokenized_corpus)
    index_build_seconds = time.perf_counter() - t0

    queries = _load_queries(gold_pairs_path, lepard_path)
    t0 = time.perf_counter()
    results_path = out_dir / "bm25_results.jsonl"
    retrieval_k = min(top_k * RETRIEVAL_K_MULTIPLIER, len(chunks))
    query_texts = [q["query_text"] for q in queries]
    batched_tokens = bm25s.tokenize(query_texts, stopwords="en")
    all_indices, all_scores = retriever.retrieve(
        batched_tokens, k=retrieval_k, n_threads=16, show_progress=True
    )
    with results_path.open("w", encoding="utf-8") as fout:
        for qi, q in enumerate(queries):
            raw_hits = [
                {
                    "opinion_id": chunk_meta[int(idx)][0],
                    "chunk_index": chunk_meta[int(idx)][1],
                    "score": float(score),
                }
                for idx, score in zip(all_indices[qi], all_scores[qi], strict=False)
            ]
            aggregated = _aggregate_chunk_scores(raw_hits, top_k=top_k)
            fout.write(
                json.dumps({
                    "source_id": q["source_id"],
                    "dest_id": q["dest_id"],
                    "retrieved": aggregated,
                }) + "\n"
            )
    retrieval_seconds = time.perf_counter() - t0
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
    if log_to_wandb:
        _log_to_wandb(summary_data, out_dir)
    return summary_data


# ---------- verified-subset path ----------


REQUIRED_VERIFIED_QUERY_FIELDS = (
    "source_id",
    "source_cluster_id",
    "dest_id",
    "destination_context",
)


def _load_queries_verified(gold_path: Path) -> list[dict[str, Any]]:
    """Load queries directly from verified gold (no LePaRD join)."""
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
    index_dir: Path | None = None,
) -> dict[str, Any]:
    """BM25 retrieval keyed by source_cluster_id with destination_context queries.

    If index_dir is provided and contains a saved bm25s index + chunk_meta sidecar,
    skip the index rebuild. Otherwise build and (if index_dir set) save for reuse.
    """
    from src.eda_schemas import BaselineBM25Summary
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(
        f"MS3 BM25 baseline VERIFIED (k1={BM25_K1}, b={BM25_B}, top_k={top_k}, seed={seed})"
    )
    logger.info("=" * 60)

    # Try to reload pre-built index (issue #9)
    retriever: Any = None
    chunk_meta: list[tuple[int, int, int]] = []
    n_chunks = 0
    n_unique_clusters = 0
    index_meta_path = (
        index_dir / "chunk_meta.jsonl" if index_dir is not None else None
    )
    bm25_index_path = index_dir / "bm25_index" if index_dir is not None else None
    can_reload = (
        index_dir is not None
        and bm25_index_path is not None
        and index_meta_path is not None
        and bm25_index_path.exists()
        and index_meta_path.exists()
    )

    t0 = time.perf_counter()
    if can_reload:
        logger.info(f"  reloading on-disk index from {index_dir}")
        retriever = bm25s.BM25.load(str(bm25_index_path), load_corpus=False)
        with index_meta_path.open() as f:
            for line in f:
                m = json.loads(line)
                chunk_meta.append(
                    (int(m["cluster_id"]), int(m["opinion_id"]), int(m["chunk_index"]))
                )
        n_chunks = len(chunk_meta)
        n_unique_clusters = len({m[0] for m in chunk_meta})
    else:
        chunks: list[dict[str, Any]] = list(_iter_corpus(corpus_path))
        chunk_texts = [c["text"] for c in chunks]
        chunk_meta = [
            (
                int(c["cluster_id"]),
                int(c.get("opinion_id", 0)),
                int(c["chunk_index"]),
            )
            for c in chunks
        ]
        n_chunks = len(chunks)
        n_unique_clusters = len({m[0] for m in chunk_meta})
        logger.info(f"  chunks: {n_chunks:,}  unique clusters: {n_unique_clusters:,}")

        tokenized_corpus = bm25s.tokenize(chunk_texts, stopwords="en")
        retriever = bm25s.BM25(k1=BM25_K1, b=BM25_B)
        retriever.index(tokenized_corpus)

        if index_dir is not None:
            index_dir.mkdir(parents=True, exist_ok=True)
            retriever.save(str(bm25_index_path))
            with index_meta_path.open("w", encoding="utf-8") as f:
                for cid, oid, cidx in chunk_meta:
                    f.write(
                        json.dumps(
                            {"cluster_id": cid, "opinion_id": oid, "chunk_index": cidx}
                        )
                        + "\n"
                    )
            logger.info(f"  saved index to {index_dir}")
    index_build_seconds = time.perf_counter() - t0
    logger.info(f"  index ready in {index_build_seconds:.2f}s")

    queries = _load_queries_verified(gold_pairs_path)
    logger.info(f"  queries: {len(queries):,}")

    t0 = time.perf_counter()
    retrieval_k = min(top_k * RETRIEVAL_K_MULTIPLIER, n_chunks)
    query_texts = [q["query_text"] for q in queries]
    batched_tokens = bm25s.tokenize(query_texts, stopwords="en")
    all_indices, all_scores = retriever.retrieve(
        batched_tokens, k=retrieval_k, n_threads=16, show_progress=False
    )

    results_path = out_dir / "bm25_results.jsonl"
    failures_path = out_dir / "bm25_failures.jsonl"
    n_failures = 0
    with results_path.open("w", encoding="utf-8") as fout, failures_path.open(
        "w", encoding="utf-8"
    ) as ffail:
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
                json.dumps({
                    "source_id": q["source_id"],
                    "source_cluster_id": q["source_cluster_id"],
                    "dest_id": q["dest_id"],
                    "retrieved": ranked,
                }) + "\n"
            )
            # Failure log (#11): gold cluster_id not in top-k
            top_ids = [r["cluster_id"] for r in ranked]
            gold_cid = q["source_cluster_id"]
            gold_in_top = gold_cid in top_ids
            if not gold_in_top:
                n_failures += 1
                ffail.write(
                    json.dumps({
                        "source_id": q["source_id"],
                        "source_cluster_id": q["source_cluster_id"],
                        "dest_id": q["dest_id"],
                        "gold_in_top_k": False,
                        "top_retrieved": ranked[:5],
                    }) + "\n"
                )
    retrieval_seconds = time.perf_counter() - t0
    logger.info(f"  retrieval done in {retrieval_seconds:.2f}s")
    logger.info(f"  misses: {n_failures}/{len(queries):,} -> {failures_path}")

    results_hash = hashlib.sha256(results_path.read_bytes()).hexdigest()
    summary_data = {
        "schema_version": SCHEMA_VERSION,
        "n_queries": len(queries),
        "n_corpus_chunks": n_chunks,
        "n_unique_opinions": n_unique_clusters,
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


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MS3 BM25 baseline retrieval.")
    ap.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--gold-pairs-path", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--lepard-path", type=Path, default=DEFAULT_LEPARD)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--log-to-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--verified",
        action="store_true",
        help="Use cluster-aware verified path (corpus needs cluster_id, queries use destination_context)",
    )
    ap.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Directory to save/load pre-built BM25 index (verified path only)",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print(
            f"[baseline_bm25] DRY RUN  schema={SCHEMA_VERSION}  top_k={args.top_k}  "
            f"k1={BM25_K1}  b={BM25_B}  git_sha={_git_sha()}  args={vars(args)}"
        )
        sys.exit(0)
    if args.verified:
        main_verified(
            corpus_path=args.corpus_path,
            gold_pairs_path=args.gold_pairs_path,
            out_dir=args.out_dir,
            top_k=args.top_k,
            log_to_wandb=args.log_to_wandb,
            seed=args.seed,
            index_dir=args.index_dir,
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
