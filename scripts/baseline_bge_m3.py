"""MS3 BGE-M3 dense baseline retrieval.

Embeds data/processed/baseline/corpus_chunks.jsonl with BAAI/bge-m3 (1024-dim
dense vectors), retrieves top-k chunks per query via FAISS cosine similarity
(IndexFlatIP on normalized vectors), and aggregates chunk scores to opinion-
level via MaxP. Mirrors the BM25 baseline contract so Cell 15 can compare.

Query text: LePaRD 'quote' field, joined by (source_id, dest_id) from
gold_pairs_test.jsonl → lepard_train_*.jsonl.

Outputs (data/processed/baseline/):
    bge_m3_results.jsonl    — per-query: {source_id, dest_id, retrieved: [...]}
    bge_m3_summary.json     — BaselineBgeM3Summary (Pydantic-validated)
    bge_m3_index.faiss      — FAISS IndexFlatIP (inner product on normalized)
    bge_m3_index_meta.jsonl — parallel chunk_meta per index row

Lazy loading: faiss, torch, sentence_transformers imported only inside main(),
never at module top. This keeps --dry-run fast and test-friendly.

Production hardening applied (2026 review):
    - encoder.max_seq_length enforced to MAX_LENGTH to prevent CUDA OOM from
      pathological long chunks (would otherwise allocate n² attention matrix).
    - Streaming corpus encoding: never materialize full (N, 1024) float32 array
      in RAM. Batches flow disk → encoder → index.add(), discarding embeddings
      after FAISS absorbs them. Keeps RAM flat regardless of corpus size.
    - Dynamic retrieval k-expansion: if one opinion dominates initial top-k
      chunk retrieval (e.g. 300 chunks from opinion 1001), expand k per-query
      until MaxP aggregation yields ≥ top_k unique opinions or index exhausted.
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

SCHEMA_VERSION = "1.0.0"
TOP_K = 100
ENCODER_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
ENCODE_BATCH_SIZE = 64
QUERY_BATCH_SIZE = 256
RETRIEVAL_K_MULTIPLIER = 3
MAX_LENGTH = 8192
SIMILARITY_METRIC = "cosine"
NORMALIZE_EMBEDDINGS = True
DTYPE = "float32"

DEFAULT_CORPUS = Path("data/processed/baseline/corpus_chunks.jsonl")
DEFAULT_GOLD = Path("data/processed/baseline/gold_pairs_test.jsonl")
DEFAULT_LEPARD = Path("lepard_train_4000000_rev0194f95.jsonl")
DEFAULT_OUT_DIR = Path("data/processed/baseline")


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("baseline_bge_m3")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[baseline_bge_m3] %(message)s"))
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
    """Join gold pairs with LePaRD by (source_id, dest_id) to get quote text."""
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
    """MaxP aggregation: max chunk score per opinion, sorted desc, clipped to top_k."""
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


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _device_name(device: str) -> str:
    if device != "cuda":
        return device
    try:
        import torch

        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda-unknown"


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
        job_type="baseline-bge-m3",
        config=summary,
        reinit=True,
    )
    wandb.log(summary)
    art = wandb.Artifact("baseline-bge-m3", type="dataset")
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
    encode_batch_size: int = ENCODE_BATCH_SIZE,
    query_batch_size: int = QUERY_BATCH_SIZE,
) -> dict[str, Any]:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from src.eda_schemas import BaselineBgeM3Summary

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _detect_device()
    device_name = _device_name(device)

    logger.info("=" * 60)
    logger.info(f"MS3 BGE-M3 dense baseline  (top_k={top_k}, seed={seed}, device={device_name})")
    logger.info("=" * 60)
    logger.info(f"  corpus_path    : {corpus_path}")
    logger.info(f"  gold_pairs_path: {gold_pairs_path}")
    logger.info(f"  lepard_path    : {lepard_path}")
    logger.info(f"  out_dir        : {out_dir}")
    logger.info(f"  encoder        : {ENCODER_MODEL} (dim={EMBEDDING_DIM})")

    # --- Load encoder with enforced max length (prevents OOM from long chunks) ---
    logger.info("\n[1/5] Loading encoder")
    t0 = time.perf_counter()
    encoder = SentenceTransformer(ENCODER_MODEL, device=device)
    encoder.max_seq_length = MAX_LENGTH
    encoder_load_seconds = time.perf_counter() - t0
    logger.info(f"  encoder loaded in: {encoder_load_seconds:.2f}s")
    logger.info(f"  max_seq_length   : {encoder.max_seq_length}")

    # --- Stream corpus: collect meta, then encode in batches direct to FAISS ---
    # Never materialize full (N, 1024) float32 array in RAM; at 7.8M chunks that
    # would be 30GB of contiguous memory.
    logger.info("\n[2/5] Streaming corpus metadata")
    chunk_meta: list[tuple[int, int]] = []
    for c in _iter_corpus(corpus_path):
        chunk_meta.append((c["opinion_id"], c["chunk_index"]))
    n_chunks = len(chunk_meta)
    unique_opinions = len({m[0] for m in chunk_meta})
    logger.info(f"  chunks           : {n_chunks:,}")
    logger.info(f"  unique opinions  : {unique_opinions:,}")

    # --- Build FAISS index via streaming batches ---
    logger.info(f"\n[3/5] Encoding corpus + building FAISS index (batch={encode_batch_size})")
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    def _iter_batches() -> Iterator[list[str]]:
        batch: list[str] = []
        for c in _iter_corpus(corpus_path):
            batch.append(c["text"])
            if len(batch) >= encode_batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    n_encoded = 0
    for batch_texts in _iter_batches():
        embs = encoder.encode(
            batch_texts,
            batch_size=encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        ).astype(np.float32)
        index.add(embs)
        n_encoded += len(batch_texts)
        if n_encoded % (encode_batch_size * 100) == 0:
            logger.info(f"    encoded {n_encoded:,} / {n_chunks:,}")

    index_build_seconds = time.perf_counter() - t0
    logger.info(f"  index size       : {index.ntotal:,}")
    logger.info(f"  index build      : {index_build_seconds:.2f}s")

    index_path = out_dir / "bge_m3_index.faiss"
    faiss.write_index(index, str(index_path))
    meta_path = out_dir / "bge_m3_index_meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for oid, ci in chunk_meta:
            f.write(json.dumps({"opinion_id": oid, "chunk_index": ci}) + "\n")
    logger.info(f"  wrote index -> {index_path}")
    logger.info(f"  wrote meta  -> {meta_path}")

    # --- Load + encode queries ---
    logger.info("\n[4/5] Loading + encoding queries")
    queries = _load_queries(gold_pairs_path, lepard_path)
    logger.info(f"  queries          : {len(queries):,}")
    query_texts = [q["query_text"] for q in queries]
    t0 = time.perf_counter()
    query_embeddings = encoder.encode(
        query_texts,
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    ).astype(np.float32)
    query_encode_seconds = time.perf_counter() - t0
    logger.info(f"  query embeddings : {query_embeddings.shape}")
    logger.info(f"  encode time      : {query_encode_seconds:.2f}s")

    # --- Retrieval with dynamic k-expansion ---
    logger.info(f"\n[5/5] FAISS search top-{top_k} per query")
    t0 = time.perf_counter()
    initial_k = min(top_k * RETRIEVAL_K_MULTIPLIER, n_chunks)
    results_path = out_dir / "bge_m3_results.jsonl"

    def _retrieve_and_aggregate(q_emb: np.ndarray) -> list[dict[str, Any]]:
        """Retrieve with k-expansion until MaxP yields >= top_k unique opinions."""
        current_k = initial_k
        while True:
            scores, indices = index.search(q_emb, current_k)
            raw_hits = [
                {
                    "opinion_id": chunk_meta[int(idx)][0],
                    "chunk_index": chunk_meta[int(idx)][1],
                    "score": float(score),
                }
                for idx, score in zip(indices[0], scores[0], strict=False)
                if idx != -1
            ]
            aggregated = _aggregate_chunk_scores(raw_hits, top_k=top_k)
            if len(aggregated) >= top_k or current_k >= n_chunks:
                return aggregated
            current_k = min(current_k * 2, n_chunks)

    with results_path.open("w", encoding="utf-8") as fout:
        for batch_start in range(0, len(queries), query_batch_size):
            batch_end = min(batch_start + query_batch_size, len(queries))
            # Batched search for common case (no k-expansion needed)
            batch_q_emb = query_embeddings[batch_start:batch_end]
            scores_batch, indices_batch = index.search(batch_q_emb, initial_k)
            for qi_local, qi_global in enumerate(range(batch_start, batch_end)):
                q = queries[qi_global]
                raw_hits = [
                    {
                        "opinion_id": chunk_meta[int(idx)][0],
                        "chunk_index": chunk_meta[int(idx)][1],
                        "score": float(score),
                    }
                    for idx, score in zip(
                        indices_batch[qi_local],
                        scores_batch[qi_local],
                        strict=False,
                    )
                    if idx != -1
                ]
                aggregated = _aggregate_chunk_scores(raw_hits, top_k=top_k)
                # k-expansion fallback only when batch retrieval under-filled top_k
                if len(aggregated) < top_k and initial_k < n_chunks:
                    aggregated = _retrieve_and_aggregate(query_embeddings[qi_global : qi_global + 1])
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
        "n_corpus_chunks": n_chunks,
        "n_unique_opinions": unique_opinions,
        "top_k": top_k,
        "encoder_model": ENCODER_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "device": device,
        "device_name": device_name,
        "encode_batch_size": encode_batch_size,
        "similarity_metric": SIMILARITY_METRIC,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "max_length": MAX_LENGTH,
        "dtype": DTYPE,
        "encoder_load_seconds": round(encoder_load_seconds, 3),
        "index_build_seconds": round(index_build_seconds, 3),
        "query_encode_seconds": round(query_encode_seconds, 3),
        "retrieval_seconds": round(retrieval_seconds, 3),
        "seed": seed,
        "git_sha": _git_sha(),
        "results_hash": results_hash,
    }
    validated = BaselineBgeM3Summary.model_validate(summary_data)
    summary_path = out_dir / "bge_m3_summary.json"
    summary_path.write_text(
        json.dumps(validated.model_dump(), sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(f"\nWrote bge_m3_summary.json -> {summary_path}")

    if log_to_wandb:
        _log_to_wandb(summary_data, out_dir)

    return summary_data


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MS3 BGE-M3 dense baseline retrieval.")
    ap.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--gold-pairs-path", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--lepard-path", type=Path, default=DEFAULT_LEPARD)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--encode-batch-size", type=int, default=ENCODE_BATCH_SIZE)
    ap.add_argument("--query-batch-size", type=int, default=QUERY_BATCH_SIZE)
    ap.add_argument("--log-to-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print(
            f"[baseline_bge_m3] DRY RUN  schema={SCHEMA_VERSION}  "
            f"top_k={args.top_k}  encoder={ENCODER_MODEL}  dim={EMBEDDING_DIM}  "
            f"device={_detect_device()}  git_sha={_git_sha()}  "
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
        encode_batch_size=args.encode_batch_size,
        query_batch_size=args.query_batch_size,
    )
