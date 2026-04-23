"""MS3 Baseline dataset prep: chunk CL corpus + extract usable gold pairs.

Thin orchestration over:
    - Verified LePaRD-CL subset (from scripts/build_lepard_cl_subset.py)
    - transformers.AutoTokenizer for 1024-subword, 128-overlap chunking

Outputs (data/processed/baseline/):
    corpus_chunks.jsonl        — one row per chunk (opinion_id, chunk_index, text)
    gold_pairs_val.jsonl       — 2K val pairs (stratified by source_court)
    gold_pairs_test.jsonl      — 45K test pairs (stratified by source_court)
    chunking_checkpoint.json   — per-shard resume state (atomic via os.replace)
    summary.json               — BaselinePrepSummary (Pydantic-validated)

Gold pairs are drawn exclusively from the verified LePaRD-CL subset produced
by scripts/build_lepard_cl_subset.py (eyecite citation match + fuzzy text
overlap). Each pair uses source_cluster_id (verified CL opinion ID) as
source_id; dest_id is the LePaRD destination opinion ID. The corpus is
restricted to opinions referenced in the verified pairs (source_cluster_id
union dest_id), keeping it tractable while preserving all gold targets.

Checkpoint semantics: per-shard completion recorded atomically via
tempfile.mkstemp + os.replace (matches scripts/ingest_lepard.py:578 +
src/bulk_download.py:141 repo convention). Crashed runs resume from
last completed shard. Checkpoints are invalidated when the verified subset
SHA changes (new subset → fresh chunking run).

Stratification: source_court key (present in verified subset rows),
minority-group-preserving largest-remainder allocation. Guarantees every
stratum with >=2 members appears in val or test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
CHUNK_SIZE_SUBWORDS = 1024
CHUNK_OVERLAP_SUBWORDS = 128
ENCODER_MODEL = "BAAI/bge-m3"
VAL_SIZE = 2000
TEST_SIZE = 45000

DEFAULT_SHARD_DIR = Path("data/raw/cl_federal_appellate_bulk")
DEFAULT_VERIFIED_SUBSET = Path("data/processed/lepard_cl_verified_subset.jsonl")
DEFAULT_OUT_DIR = Path("data/processed/baseline")


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("baseline_prep")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[baseline_prep] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


# ---------- I/O primitives ----------


def _load_verified_subset(path: Path) -> list[dict[str, Any]]:
    """Load all rows from the verified LePaRD-CL subset JSONL."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------- stratified split ----------


def _largest_remainder(
    budget: int,
    weights: dict[str, float],
    caps: dict[str, int],
) -> dict[str, int]:
    """Largest-remainder apportionment respecting per-stratum caps."""
    if budget == 0 or not weights:
        return dict.fromkeys(weights, 0)
    wsum = sum(weights.values()) or 1.0
    raw = {c: budget * (w / wsum) for c, w in weights.items()}
    floor = {c: int(r) for c, r in raw.items()}
    leftover = budget - sum(floor.values())
    remainder = sorted(
        ((raw[c] - floor[c], c) for c in raw),
        reverse=True,
    )
    for i in range(leftover):
        floor[remainder[i % len(remainder)][1]] += 1
    for c in floor:
        floor[c] = min(floor[c], caps[c])
    return floor


def _stratified_split(
    pairs: list[dict[str, Any]],
    *,
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Minority-preserving proportional stratified split by source_court.

    Guarantees every stratum with >=2 members appears in val or test.
    Algorithm: reserve a minority floor (1 slot in test for each stratum
    with >=2 members), then distribute remaining val/test budget
    proportionally using largest-remainder apportionment. Final shuffle
    is seed-deterministic.
    """
    rng = random.Random(seed)
    by_court: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_court[pair["source_court"]].append(pair)
    for court in by_court:
        rng.shuffle(by_court[court])

    total = sum(len(v) for v in by_court.values())
    if total == 0:
        return [], []

    strata = sorted(by_court.items())
    val_alloc: dict[str, int] = dict.fromkeys(by_court, 0)
    test_alloc: dict[str, int] = dict.fromkeys(by_court, 0)

    # Minority floor: 1 slot in test for each stratum with >=2 members
    for court, members in strata:
        if len(members) >= 2 and test_size > 0:
            test_alloc[court] = 1

    weights = {c: float(len(by_court[c])) for c in by_court}

    # Proportional val allocation, capped at stratum size
    val_caps = {c: len(by_court[c]) for c in by_court}
    add_val = _largest_remainder(val_size, weights, val_caps)
    for c in val_alloc:
        val_alloc[c] = add_val[c]

    # Test caps respect val allocation; clamp minority floor to cap
    test_caps = {c: len(by_court[c]) - val_alloc[c] for c in by_court}
    for c in test_alloc:
        test_alloc[c] = min(test_alloc[c], test_caps[c])

    # Remaining test budget distributed proportionally
    remaining_test = max(0, test_size - sum(test_alloc.values()))
    test_caps_after_floor = {c: test_caps[c] - test_alloc[c] for c in test_alloc}
    add_test = _largest_remainder(
        remaining_test,
        weights,
        test_caps_after_floor,
    )
    for c in test_alloc:
        test_alloc[c] += add_test[c]

    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for court, members in strata:
        v_n = val_alloc[court]
        t_n = test_alloc[court]
        val.extend(members[:v_n])
        test.extend(members[v_n : v_n + t_n])

    rng.shuffle(val)
    rng.shuffle(test)
    return val, test


# ---------- chunking ----------


def _get_tokenizer() -> Any:
    """Load BAAI/bge-m3 tokenizer. Network call — mock in unit tests."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(ENCODER_MODEL)


def _chunk_text(
    text: str,
    *,
    opinion_id: int,
    tok: Any,
) -> list[dict[str, Any]]:
    """Chunk text into 1024-subword windows with paragraph-boundary awareness.

    Algorithm:
      1. Split on blank lines (\\n\\n+) so cuts land at natural paragraph
         boundaries wherever possible.
      2. Paragraphs that exceed CHUNK_SIZE on their own are pre-split at
         token level (fallback), producing sub-paragraph slices.
      3. The main loop greedily fills a buffer with whole paragraph token
         sequences. When the next paragraph would overflow, the buffer is
         flushed as a chunk and the last CHUNK_OVERLAP_SUBWORDS tokens are
         carried forward so no inter-chunk context is hard-lost.
      4. Flat text (no blank lines) degrades gracefully to the old token-
         level sliding-window behaviour via the pre-split fallback.
    """
    raw_paras = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in raw_paras if p.strip()]
    if not paragraphs:
        return []

    stride = CHUNK_SIZE_SUBWORDS - CHUNK_OVERLAP_SUBWORDS  # 896

    # Encode each paragraph; pre-split oversized ones at token level so the
    # main loop can assume every element fits within CHUNK_SIZE.
    para_seqs: list[list[int]] = []
    for para in paragraphs:
        tokens = tok.encode(para, add_special_tokens=False)
        if not tokens:
            continue
        if len(tokens) <= CHUNK_SIZE_SUBWORDS:
            para_seqs.append(tokens)
        else:
            start = 0
            while start < len(tokens):
                end = min(start + CHUNK_SIZE_SUBWORDS, len(tokens))
                para_seqs.append(tokens[start:end])
                if end >= len(tokens):
                    break
                start += stride

    if not para_seqs:
        return []

    chunks: list[dict[str, Any]] = []
    chunk_idx = 0
    current: list[int] = []

    for para_tokens in para_seqs:
        if len(current) + len(para_tokens) <= CHUNK_SIZE_SUBWORDS:
            current.extend(para_tokens)
        else:
            if current:
                chunks.append(
                    {
                        "opinion_id": opinion_id,
                        "chunk_index": chunk_idx,
                        "text": tok.decode(current, skip_special_tokens=True),
                    }
                )
                chunk_idx += 1
                current = current[-CHUNK_OVERLAP_SUBWORDS:]
            # para_tokens is guaranteed <= CHUNK_SIZE (pre-split above).
            # If overlap + para still overflows (edge case where overlap itself
            # is near-full), drop the overlap rather than truncating the para.
            if len(current) + len(para_tokens) <= CHUNK_SIZE_SUBWORDS:
                current.extend(para_tokens)
            else:
                current = list(para_tokens)

    if current:
        chunks.append(
            {
                "opinion_id": opinion_id,
                "chunk_index": chunk_idx,
                "text": tok.decode(current, skip_special_tokens=True),
            }
        )

    return chunks


# ---------- checkpoint (atomic) ----------


def _load_checkpoint(path: Path) -> tuple[set[str], str]:
    """Return (completed_shard_names, run_key).

    Missing run_key in legacy checkpoints defaults to "" so that old
    checkpoint files trigger a fresh run (run_key mismatch with any real key).
    """
    if not path.exists():
        return set(), ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data["completed"]), str(data.get("run_key", ""))
    except (KeyError, json.JSONDecodeError, ValueError):
        return set(), ""


def _save_checkpoint(path: Path, completed: set[str], run_key: str = "") -> None:
    """Atomic write: tempfile.mkstemp + os.replace (repo convention)."""
    data = json.dumps(
        {"completed": sorted(completed), "run_key": run_key},
        indent=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


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


def _corpus_manifest_sha(shard_dir: Path) -> str:
    manifest = shard_dir / "manifest.json"
    if not manifest.exists():
        return "0" * 64
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


# ---------- corpus chunking ----------


def _chunk_corpus(
    shard_dir: Path,
    out_path: Path,
    ckpt_path: Path,
    *,
    resume: bool,
    tok: Any,
    opinion_ids: set[int],
    run_key: str = "",
) -> tuple[int, int]:
    """Stream CL shards, chunk opinions in opinion_ids, write corpus_chunks.jsonl.

    Returns (n_chunks, n_opinions). Only opinions whose id is in opinion_ids
    are chunked; others are skipped. Checkpoints are invalidated when run_key
    changes (i.e. when the verified subset is updated).

    Resume semantics: if the existing checkpoint recorded a different run_key,
    the checkpoint is discarded and chunking restarts from scratch.
    """
    completed, ckpt_run_key = _load_checkpoint(ckpt_path) if resume else (set(), run_key)
    if ckpt_run_key != run_key:
        logger.info(
            f"  run_key changed ({ckpt_run_key[:8] or 'none'!r} → {run_key[:8]!r}), discarding checkpoint"
        )
        completed = set()

    shards = sorted(shard_dir.glob("shard_*.jsonl"))
    mode = "a" if completed else "w"
    n_chunks = 0
    n_opinions = 0

    # Idempotency: count pre-existing chunks from prior resumed runs.
    if completed and out_path.exists():
        seen_opinions: set[int] = set()
        with out_path.open(encoding="utf-8") as fin:
            for line in fin:
                r = json.loads(line)
                n_chunks += 1
                seen_opinions.add(r["opinion_id"])
        n_opinions = len(seen_opinions)

    with out_path.open(mode, encoding="utf-8") as fout:
        for shard in shards:
            if shard.name in completed:
                logger.info(f"  skip {shard.name} (checkpointed)")
                continue
            logger.info(f"  chunking {shard.name}")
            with shard.open(encoding="utf-8") as fin:
                for line in fin:
                    r = json.loads(line)
                    oid = int(r["id"])
                    if oid not in opinion_ids:
                        continue
                    text = r.get("text", "")
                    if not text:
                        continue
                    chunks = _chunk_text(text, opinion_id=oid, tok=tok)
                    for c in chunks:
                        fout.write(json.dumps(c) + "\n")
                    n_chunks += len(chunks)
                    if chunks:
                        n_opinions += 1
            completed.add(shard.name)
            _save_checkpoint(ckpt_path, completed, run_key=run_key)
    return n_chunks, n_opinions


# ---------- W&B ----------


def _log_to_wandb(summary: dict[str, Any], out_dir: Path) -> None:
    """Single consolidated wandb.log call. Repo isolation contract."""
    try:
        import wandb
    except ImportError:
        logger.info("  wandb unavailable — skipping telemetry")
        return
    run = wandb.init(
        entity="phl690-harvard-extension-schol",
        project="cs1090b",
        job_type="baseline-prep",
        config=summary,
        reinit=True,
    )
    wandb.log(summary)
    art = wandb.Artifact("baseline-prep", type="dataset")
    art.add_dir(str(out_dir))
    run.log_artifact(art)
    run.finish()


# ---------- main ----------


def main(
    shard_dir: Path = DEFAULT_SHARD_DIR,
    verified_subset_path: Path = DEFAULT_VERIFIED_SUBSET,
    out_dir: Path = DEFAULT_OUT_DIR,
    log_to_wandb: bool = False,
    resume: bool = True,
    seed: int = 0,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
) -> dict[str, Any]:
    """Run baseline prep pipeline. Returns summary dict.

    Reads gold pairs from the verified LePaRD-CL subset (eyecite + fuzzy
    match verified). Corpus chunking is restricted to the opinion IDs
    referenced in those pairs (source_cluster_id ∪ dest_id), keeping the
    corpus tractable while ensuring all gold targets are retrievable.
    """
    from src.eda_schemas import BaselinePrepSummary

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"MS3 baseline prep  (seed={seed})")
    logger.info("=" * 60)
    logger.info(f"  shard_dir            : {shard_dir}")
    logger.info(f"  verified_subset_path : {verified_subset_path}")
    logger.info(f"  out_dir              : {out_dir}")

    logger.info("\n[1/3] Loading verified LePaRD-CL subset")
    verified_rows = _load_verified_subset(verified_subset_path)
    logger.info(f"  verified pairs       : {len(verified_rows):,}")

    verified_subset_sha = hashlib.sha256(verified_subset_path.read_bytes()).hexdigest()
    logger.info(f"  verified_subset_sha  : {verified_subset_sha[:16]}...")

    logger.info("\n[2/3] Building gold pairs + corpus opinion ID set")

    # Deduplicate by (source_cluster_id, dest_id)
    seen: set[tuple[int, int]] = set()
    deduped: list[dict[str, Any]] = []
    corpus_opinion_ids: set[int] = set()

    for row in verified_rows:
        src_cid = int(row["source_cluster_id"])
        dst_id = int(row["dest_id"])
        corpus_opinion_ids.add(src_cid)
        corpus_opinion_ids.add(dst_id)
        k = (src_cid, dst_id)
        if k not in seen:
            seen.add(k)
            deduped.append(
                {
                    "source_id": src_cid,
                    "dest_id": dst_id,
                    "source_court": row.get("source_court", "unknown"),
                    "quote": row.get("quote", ""),
                    "destination_context": row.get("destination_context", ""),
                }
            )

    logger.info(f"  raw pairs            : {len(verified_rows):,}")
    logger.info(f"  unique pairs         : {len(deduped):,}")
    logger.info(f"  corpus opinion IDs   : {len(corpus_opinion_ids):,} unique")

    val, test = _stratified_split(
        deduped,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    logger.info(f"  val split            : {len(val):,}")
    logger.info(f"  test split           : {len(test):,}")

    (out_dir / "gold_pairs_val.jsonl").write_text(
        "\n".join(json.dumps(p) for p in val) + "\n",
        encoding="utf-8",
    )
    (out_dir / "gold_pairs_test.jsonl").write_text(
        "\n".join(json.dumps(p) for p in test) + "\n",
        encoding="utf-8",
    )

    val_dist: dict[str, int] = defaultdict(int)
    test_dist: dict[str, int] = defaultdict(int)
    for p in val:
        val_dist[p["source_court"]] += 1
    for p in test:
        test_dist[p["source_court"]] += 1

    logger.info("\n[3/3] Chunking CL corpus (verified opinions only)")
    tok = _get_tokenizer()
    corpus_path = out_dir / "corpus_chunks.jsonl"
    ckpt_path = out_dir / "chunking_checkpoint.json"
    n_chunks, n_opinions = _chunk_corpus(
        shard_dir,
        corpus_path,
        ckpt_path,
        resume=resume,
        tok=tok,
        opinion_ids=corpus_opinion_ids,
        run_key=verified_subset_sha,
    )
    logger.info(f"  chunks written       : {n_chunks:,}")
    logger.info(f"  opinions             : {n_opinions:,}")

    gold_pair_hashes = {
        fname: hashlib.sha256((out_dir / fname).read_bytes()).hexdigest()
        for fname in ("gold_pairs_val.jsonl", "gold_pairs_test.jsonl")
    }

    summary_data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "corpus_chunks": n_chunks,
        "n_opinions_chunked": n_opinions,
        "gold_pairs_total": len(deduped),
        "gold_pairs_train": 0,
        "gold_pairs_val": len(val),
        "gold_pairs_test": len(test),
        "val_court_distribution": dict(val_dist),
        "test_court_distribution": dict(test_dist),
        "seed": seed,
        "git_sha": _git_sha(),
        "corpus_manifest_sha": _corpus_manifest_sha(shard_dir),
        "gold_pair_hashes": gold_pair_hashes,
        "verified_subset_sha": verified_subset_sha,
        "n_verified_pairs": len(verified_rows),
    }
    validated = BaselinePrepSummary.model_validate(summary_data)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(validated.model_dump(), sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(f"\nWrote summary.json -> {summary_path}")

    if log_to_wandb:
        _log_to_wandb(summary_data, out_dir)

    return summary_data


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MS3 baseline dataset prep.")
    ap.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    ap.add_argument("--verified-subset-path", type=Path, default=DEFAULT_VERIFIED_SUBSET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--log-to-wandb", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-size", type=int, default=VAL_SIZE)
    ap.add_argument("--test-size", type=int, default=TEST_SIZE)
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs + print fingerprint without running pipeline",
    )
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        import sys as _sys

        print("[baseline_prep] DRY RUN")
        print(f"  schema_version        : {SCHEMA_VERSION}")
        print(f"  CHUNK_SIZE            : {CHUNK_SIZE_SUBWORDS}")
        print(f"  CHUNK_OVERLAP         : {CHUNK_OVERLAP_SUBWORDS}")
        print(f"  ENCODER_MODEL         : {ENCODER_MODEL}")
        print(f"  VAL_SIZE              : {VAL_SIZE}")
        print(f"  TEST_SIZE             : {TEST_SIZE}")
        print(f"  python                : {_sys.version.split()[0]}")
        print(f"  git_sha               : {_git_sha()}")
        print(f"  args                  : {vars(args)}")
        _sys.exit(0)
    main(
        shard_dir=args.shard_dir,
        verified_subset_path=args.verified_subset_path,
        out_dir=args.out_dir,
        log_to_wandb=args.log_to_wandb,
        resume=args.resume,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
    )
