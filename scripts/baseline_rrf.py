# scripts/baseline_rrf.py
"""MS4 Reciprocal Rank Fusion (RRF) of BM25 + BGE-M3 retrieval results.

Combines two retrieval runs into a single fused ranking via Cormack et al. 2009
RRF: each candidate's fused score = sum over runs of 1/(k + rank). Runs that
disagree on the gold-pair identity (source_id, dest_id) raise ValueError.

Why RRF
-------
- Parameter-light: single hyperparameter k (default 60 per Cormack 2009).
- Score-scale agnostic: normalizes BM25 (unbounded positive) and BGE-M3
  (cosine in [-1,1]) to a comparable rank-based signal.
- Deterministic and stable across re-runs.
- Per Cell 15 verified-pipeline result, BM25 strictly dominates BGE-M3 on
  every metric, but BGE-M3 wins 16.2% of paired queries — RRF captures that
  complementary signal.

I/O contract (verified pipeline)
--------------------------------
Input rows  : {source_id, dest_id, source_cluster_id, retrieved: [{cluster_id, score}]}
Output rows : same schema, with retrieved re-ranked by RRF score
Both inputs MUST be aligned row-for-row on (source_id, dest_id) — produced by
the verified runners with deduplicated query loaders.

Outputs (data/processed/baseline/cleaned/)
------------------------------------------
rrf_results.jsonl   — per-query top-k fused candidates
rrf_summary.json    — provenance + hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
DEFAULT_RRF_K = 60  # Cormack et al. 2009 standard
DEFAULT_TOP_K = 100


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("baseline_rrf")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[baseline_rrf] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------- pure RRF math ----------


def _rrf_score(*, rank: int, k: int) -> float:
    """Reciprocal Rank Fusion score: 1 / (k + rank).

    rank is 1-indexed; rank=0 is reserved for "not retrieved" and is invalid here.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1 (1-indexed), got {rank}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    return 1.0 / (k + rank)


# ---------- public API ----------


def fuse_two_runs(
    *,
    bm25_path: Path,
    bge_m3_path: Path,
    out_path: Path,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[str, Any]:
    """Fuse aligned BM25 + BGE-M3 result files into a single RRF ranking.

    Streams both inputs row-by-row (assumes upstream verified runners emit
    queries in identical order). For each query, computes per-cluster fused
    score = RRF(bm25_rank) + RRF(bge_rank), keeps top_k by descending score.
    Tie-break: ascending cluster_id (deterministic).

    Returns a dict with n_queries, top_k, rrf_k, results_hash for downstream
    summary writing.
    """
    bm25_path = Path(bm25_path)
    bge_m3_path = Path(bge_m3_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_queries = 0
    tmp_path = out_path.with_suffix(".jsonl.tmp")

    with (
        bm25_path.open(encoding="utf-8") as bm25_f,
        bge_m3_path.open(encoding="utf-8") as bge_f,
        tmp_path.open("w", encoding="utf-8") as fout,
    ):
        for bm25_line, bge_line in zip(bm25_f, bge_f, strict=True):
            bm25_line = bm25_line.strip()
            bge_line = bge_line.strip()
            if not bm25_line and not bge_line:
                continue
            if not bm25_line or not bge_line:
                raise ValueError(
                    f"baseline files have unequal row counts at query #{n_queries + 1}"
                )

            bm25_row = json.loads(bm25_line)
            bge_row = json.loads(bge_line)

            key_bm25 = (int(bm25_row["source_id"]), int(bm25_row["dest_id"]))
            key_bge = (int(bge_row["source_id"]), int(bge_row["dest_id"]))
            if key_bm25 != key_bge:
                raise ValueError(
                    f"fuse_two_runs requires aligned result files; "
                    f"BM25 row={key_bm25} vs BGE-M3 row={key_bge}"
                )

            # Compute fused scores per cluster_id
            fused: dict[int, float] = {}
            for r, hit in enumerate(bm25_row["retrieved"], start=1):
                cid = int(hit["cluster_id"])
                fused[cid] = fused.get(cid, 0.0) + _rrf_score(rank=r, k=rrf_k)
            for r, hit in enumerate(bge_row["retrieved"], start=1):
                cid = int(hit["cluster_id"])
                fused[cid] = fused.get(cid, 0.0) + _rrf_score(rank=r, k=rrf_k)

            # Sort by (-score, cluster_id) for deterministic tie-break
            ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
            ranked_records = [{"cluster_id": cid, "score": sc} for cid, sc in ranked]

            out_row = {
                "source_id": bm25_row["source_id"],
                "dest_id": bm25_row["dest_id"],
                "source_cluster_id": bm25_row["source_cluster_id"],
                "retrieved": ranked_records,
            }
            fout.write(json.dumps(out_row, allow_nan=False) + "\n")
            n_queries += 1

    import os
    os.replace(tmp_path, out_path)

    results_hash = hashlib.sha256(out_path.read_bytes()).hexdigest() if n_queries > 0 else ""

    return {
        "n_queries": n_queries,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "results_hash": results_hash,
    }


# ---------- CLI ----------


def main(
    bm25_path: Path,
    bge_m3_path: Path,
    out_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    seed: int = 0,
) -> dict[str, Any]:
    """CLI wrapper: fuse + write summary alongside results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "rrf_results.jsonl"
    summary_path = out_dir / "rrf_summary.json"

    logger.info("=" * 60)
    logger.info(f"MS4 RRF fusion  (rrf_k={rrf_k}, top_k={top_k})")
    logger.info("=" * 60)
    logger.info(f"  bm25_path   : {bm25_path}")
    logger.info(f"  bge_m3_path : {bge_m3_path}")
    logger.info(f"  out_path    : {out_path}")

    result = fuse_two_runs(
        bm25_path=Path(bm25_path),
        bge_m3_path=Path(bge_m3_path),
        out_path=out_path,
        top_k=top_k,
        rrf_k=rrf_k,
    )

    bm25_hash = hashlib.sha256(Path(bm25_path).read_bytes()).hexdigest()
    bge_hash = hashlib.sha256(Path(bge_m3_path).read_bytes()).hexdigest()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_queries": result["n_queries"],
        "top_k": result["top_k"],
        "rrf_k": result["rrf_k"],
        "results_hash": result["results_hash"],
        "bm25_input_hash": bm25_hash,
        "bge_m3_input_hash": bge_hash,
        "git_sha": _git_sha(),
        "seed": seed,
    }
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(f"  n_queries   : {result['n_queries']:,}")
    logger.info(f"  results_hash: {result['results_hash'][:16]}...")
    logger.info(f"  wrote summary -> {summary_path}")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MS4 RRF fusion of BM25 + BGE-M3 results.")
    ap.add_argument("--bm25-path", type=Path, required=True)
    ap.add_argument("--bge-m3-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print(
            f"[baseline_rrf] DRY RUN  schema={SCHEMA_VERSION}  rrf_k={args.rrf_k}  "
            f"top_k={args.top_k}  git_sha={_git_sha()}  args={vars(args)}"
        )
        sys.exit(0)
    main(
        bm25_path=args.bm25_path,
        bge_m3_path=args.bge_m3_path,
        out_dir=args.out_dir,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        seed=args.seed,
    )
