"""MS3 Baseline dataset prep: chunk CL corpus + extract usable gold pairs.

Two split paths:
  - main(): legacy cl_ids-based path (splits by source_id)
  - main_verified(): cluster-aware path for verified subset
    (splits by source_cluster_id to prevent retrieval-eval leakage)

Corpus chunks include cluster_id field so downstream BM25/BGE-M3 can
retrieve directly by cluster_id (the verified gold key).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
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
DEFAULT_LEPARD = Path("lepard_train_4000000_rev0194f95.jsonl")
DEFAULT_CL_IDS = Path("data/processed/cl_ids.txt.gz")
DEFAULT_COURT_MAP = Path("data/processed/cl_matched_courts.json")
DEFAULT_OUT_DIR = Path("data/processed/baseline")

REQUIRED_VERIFIED_FIELDS = (
    "source_id",
    "source_cluster_id",
    "source_court",
    "dest_id",
    "quote",
    "destination_context",
)


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


def _load_cl_ids(path: Path) -> set[int]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def _load_court_map(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def _iter_usable_gold(
    lepard_path: Path,
    cl_ids: set[int],
) -> Iterator[dict[str, Any]]:
    with lepard_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src = int(r["source_id"])
            dst = int(r["dest_id"])
            if src in cl_ids and dst in cl_ids:
                yield {"source_id": src, "dest_id": dst}


def _annotate_source_court(
    pairs: list[dict[str, Any]],
    court_map: dict[int, str],
) -> list[dict[str, Any]]:
    return [
        {**p, "source_court": court_map.get(p["source_id"], "unknown")}
        for p in pairs
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


# ---------- stratified split ----------


def _largest_remainder(
    budget: int,
    weights: dict[str, float],
    caps: dict[str, int],
) -> dict[str, int]:
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

    for court, members in strata:
        if len(members) >= 2 and test_size > 0:
            test_alloc[court] = 1

    weights = {c: float(len(by_court[c])) for c in by_court}
    val_caps = {c: len(by_court[c]) for c in by_court}
    add_val = _largest_remainder(val_size, weights, val_caps)
    for c in val_alloc:
        val_alloc[c] = add_val[c]

    test_caps = {c: len(by_court[c]) - val_alloc[c] for c in by_court}
    for c in test_alloc:
        test_alloc[c] = min(test_alloc[c], test_caps[c])

    remaining_test = max(0, test_size - sum(test_alloc.values()))
    test_caps_after_floor = {c: test_caps[c] - test_alloc[c] for c in test_alloc}
    add_test = _largest_remainder(
        remaining_test, weights, test_caps_after_floor
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
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ENCODER_MODEL)


def _chunk_text(
    text: str,
    *,
    opinion_id: int,
    cluster_id: int | None = None,
    tok: Any,
) -> list[dict[str, Any]]:
    """Chunk text. Output includes opinion_id and (if provided) cluster_id."""
    tokens = tok.encode(text, add_special_tokens=False)
    if not tokens:
        return []
    stride = CHUNK_SIZE_SUBWORDS - CHUNK_OVERLAP_SUBWORDS
    chunks: list[dict[str, Any]] = []
    idx = 0
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE_SUBWORDS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tok.decode(chunk_tokens, skip_special_tokens=True)
        chunk_dict: dict[str, Any] = {
            "opinion_id": opinion_id,
            "chunk_index": idx,
            "text": chunk_text,
        }
        if cluster_id is not None:
            chunk_dict["cluster_id"] = cluster_id
        chunks.append(chunk_dict)
        idx += 1
        if end >= len(tokens):
            break
        start += stride
    return chunks


# ---------- checkpoint ----------


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8"))["completed"])


def _save_checkpoint(path: Path, completed: set[str]) -> None:
    data = json.dumps({"completed": sorted(completed)}, indent=2)
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
) -> tuple[int, int]:
    """Chunk CL shards. Each chunk row carries opinion_id and cluster_id (if shard provides it)."""
    completed = _load_checkpoint(ckpt_path) if resume else set()
    shards = sorted(shard_dir.glob("shard_*.jsonl"))
    mode = "a" if completed else "w"
    n_chunks = 0
    n_opinions = 0

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
                    cid = int(r["cluster_id"]) if "cluster_id" in r else None
                    text = r.get("text", "")
                    if not text:
                        continue
                    chunks = _chunk_text(
                        text, opinion_id=oid, cluster_id=cid, tok=tok
                    )
                    for c in chunks:
                        fout.write(json.dumps(c) + "\n")
                    n_chunks += len(chunks)
                    if chunks:
                        n_opinions += 1
            completed.add(shard.name)
            _save_checkpoint(ckpt_path, completed)
    return n_chunks, n_opinions


# ---------- W&B ----------


def _log_to_wandb(summary: dict[str, Any], out_dir: Path) -> None:
    try:
        import wandb
    except ImportError:
        logger.info("  wandb unavailable")
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


# ---------- verified-subset consumer ----------


def _iter_verified_subset(path: Path) -> Iterator[dict[str, Any]]:
    """Stream rows from lepard_cl_verified_subset.jsonl. Fail-fast on malformed JSON."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _validate_verified_row(row: dict[str, Any], line_no: int) -> None:
    """Raise ValueError if required keys missing."""
    missing = [k for k in REQUIRED_VERIFIED_FIELDS if k not in row]
    if missing:
        raise ValueError(
            f"missing required key(s) {missing} in verified subset line {line_no}"
        )


def _stratified_split_verified(
    pairs: list[dict[str, Any]],
    *,
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cluster-aware split: same source_cluster_id never in both val and test."""
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[int(pair["source_cluster_id"])].append(pair)

    group_records: list[dict[str, Any]] = []
    for cluster_id, members in groups.items():
        court_counts = Counter(m["source_court"] for m in members)
        majority_court = court_counts.most_common(1)[0][0]
        group_records.append(
            {
                "source_id": cluster_id,
                "source_court": majority_court,
                "_members": members,
            }
        )

    n_groups = len(group_records)
    avg_group_size = max(1, len(pairs) // max(1, n_groups))
    group_val_size = (
        min(n_groups, max(1, val_size // avg_group_size)) if val_size else 0
    )
    group_test_size = (
        min(n_groups - group_val_size, max(1, test_size // avg_group_size))
        if test_size
        else 0
    )

    val_groups, test_groups = _stratified_split(
        group_records,
        val_size=group_val_size,
        test_size=group_test_size,
        seed=seed,
    )

    val_pairs: list[dict[str, Any]] = []
    for g in val_groups:
        val_pairs.extend(g["_members"])
        if len(val_pairs) >= val_size:
            break
    val_pairs = val_pairs[:val_size]

    test_pairs: list[dict[str, Any]] = []
    for g in test_groups:
        test_pairs.extend(g["_members"])
        if len(test_pairs) >= test_size:
            break
    test_pairs = test_pairs[:test_size]

    return val_pairs, test_pairs


# ---------- corpus enrichment with cluster_id ----------


def enrich_corpus_with_cluster_id(
    shard_dir: Path,
    corpus_in_path: Path,
    corpus_out_path: Path,
    max_unmatched_rate: float = 1.0,
) -> tuple[int, int, int]:
    """Enrich existing corpus_chunks.jsonl with cluster_id from CL shards.

    Reads each shard once to build opinion_id → cluster_id lookup, then
    streams corpus chunks adding cluster_id where matched. Existing
    cluster_id values are preserved (not overwritten).

    Unmatched rows are routed to ``unmatched_chunks.jsonl`` (dead-letter)
    alongside the output. If the unmatched rate exceeds
    ``max_unmatched_rate`` (fraction of total), a RuntimeError is raised
    after writing the dead-letter file.

    Writes ``<corpus_out>.summary.json`` with provenance (sha256, git_sha).

    Returns: (n_total_chunks, n_enriched, n_unmatched)
    """
    logger.info(f"building opinion_id → cluster_id from {shard_dir}")
    oid_to_cid: dict[int, int] = {}
    for shard in sorted(shard_dir.glob("shard_*.jsonl")):
        with shard.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "id" in r and "cluster_id" in r:
                    oid_to_cid[int(r["id"])] = int(r["cluster_id"])
    logger.info(f"  opinion_id → cluster_id: {len(oid_to_cid):,} entries")

    corpus_out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = corpus_out_path.with_suffix(corpus_out_path.suffix + ".tmp")
    dead_letter = corpus_out_path.parent / "unmatched_chunks.jsonl"
    dl_tmp = dead_letter.with_suffix(dead_letter.suffix + ".tmp")
    n_total = n_enriched = n_unmatched = 0
    with corpus_in_path.open(encoding="utf-8") as fin, tmp.open(
        "w", encoding="utf-8"
    ) as fout, dl_tmp.open("w", encoding="utf-8") as fdl:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            if "cluster_id" not in r:
                cid = oid_to_cid.get(int(r["opinion_id"]))
                if cid is not None:
                    r["cluster_id"] = cid
                    n_enriched += 1
                else:
                    n_unmatched += 1
                    fdl.write(json.dumps(r) + "\n")
            else:
                n_enriched += 1
            if "cluster_id" in r:
                fout.write(json.dumps(r) + "\n")
    tmp.rename(corpus_out_path)
    if n_unmatched > 0:
        dl_tmp.rename(dead_letter)
        logger.info(f"  unmatched routed to dead-letter: {dead_letter}")
    else:
        dl_tmp.unlink(missing_ok=True)
    logger.info(
        f"  total={n_total:,}  enriched={n_enriched:,}  unmatched={n_unmatched:,}"
    )

    # Sample first 20 unmatched opinion_ids from dead-letter for debugging
    sample_unmatched: list[int] = []
    if n_unmatched > 0 and dead_letter.exists():
        with dead_letter.open(encoding="utf-8") as f:
            for line in f:
                if len(sample_unmatched) >= 20:
                    break
                try:
                    sample_unmatched.append(int(json.loads(line)["opinion_id"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    from datetime import datetime, timezone
    # Provenance summary
    summary_path = corpus_out_path.with_suffix(".summary.json")
    summary = {
        "n_total": n_total,
        "n_enriched": n_enriched,
        "n_unmatched": n_unmatched,
        "unmatched_rate": round(n_unmatched / max(n_total, 1), 6),
        "sample_unmatched_opinion_ids": sample_unmatched,
        "shard_dir": str(shard_dir),
        "corpus_in_path": str(corpus_in_path),
        "corpus_out_path": str(corpus_out_path),
        "corpus_in_sha256": hashlib.sha256(corpus_in_path.read_bytes()).hexdigest(),
        "corpus_out_sha256": hashlib.sha256(corpus_out_path.read_bytes()).hexdigest()
        if corpus_out_path.stat().st_size > 0
        else "",
        "git_sha": _git_sha(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    rate = n_unmatched / max(n_total, 1)
    if rate > max_unmatched_rate:
        raise RuntimeError(
            f"unmatched rate {rate:.2%} exceeds threshold {max_unmatched_rate:.2%} "
            f"({n_unmatched:,}/{n_total:,}); see {dead_letter}"
        )
    return n_total, n_enriched, n_unmatched


# ---------- main (legacy cl_ids path) ----------


def main(
    shard_dir: Path = DEFAULT_SHARD_DIR,
    lepard_path: Path = DEFAULT_LEPARD,
    cl_ids_path: Path = DEFAULT_CL_IDS,
    court_map_path: Path = DEFAULT_COURT_MAP,
    out_dir: Path = DEFAULT_OUT_DIR,
    log_to_wandb: bool = False,
    resume: bool = True,
    seed: int = 0,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
) -> dict[str, Any]:
    from src.eda_schemas import BaselinePrepSummary

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, path in [
        ("shard_dir", shard_dir),
        ("lepard_path", lepard_path),
        ("cl_ids_path", cl_ids_path),
        ("court_map_path", court_map_path),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    logger.info("=" * 60)
    logger.info(f"MS3 baseline prep (seed={seed})")
    logger.info("=" * 60)

    cl_ids = _load_cl_ids(cl_ids_path)
    court_map = _load_court_map(court_map_path)

    raw_pairs = list(_iter_usable_gold(lepard_path, cl_ids))
    pairs = _annotate_source_court(raw_pairs, court_map)
    seen: set[tuple[int, int]] = set()
    deduped: list[dict[str, Any]] = []
    for p in pairs:
        k = (p["source_id"], p["dest_id"])
        if k not in seen:
            seen.add(k)
            deduped.append(p)
    logger.info(f"  unique pairs: {len(deduped):,}")

    val, test = _stratified_split(
        deduped, val_size=val_size, test_size=test_size, seed=seed
    )
    logger.info(f"  val: {len(val):,}  test: {len(test):,}")

    _write_jsonl(out_dir / "gold_pairs_val.jsonl", val)
    _write_jsonl(out_dir / "gold_pairs_test.jsonl", test)

    val_dist: dict[str, int] = defaultdict(int)
    test_dist: dict[str, int] = defaultdict(int)
    for p in val:
        val_dist[p["source_court"]] += 1
    for p in test:
        test_dist[p["source_court"]] += 1

    tok = _get_tokenizer()
    corpus_path = out_dir / "corpus_chunks.jsonl"
    ckpt_path = out_dir / "chunking_checkpoint.json"
    n_chunks, n_opinions = _chunk_corpus(
        shard_dir, corpus_path, ckpt_path, resume=resume, tok=tok
    )

    gold_pair_hashes = {
        fname: hashlib.sha256((out_dir / fname).read_bytes()).hexdigest()
        for fname in ("gold_pairs_val.jsonl", "gold_pairs_test.jsonl")
    }
    summary_data = {
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
    }
    validated = BaselinePrepSummary.model_validate(summary_data)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(validated.model_dump(), sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if log_to_wandb:
        _log_to_wandb(summary_data, out_dir)
    return summary_data


# ---------- main_verified ----------


def main_verified(
    verified_subset_path: Path,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 0,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
) -> dict[str, Any]:
    """Cluster-aware split using polars streaming load."""
    if not verified_subset_path.exists():
        raise FileNotFoundError(f"verified subset not found: {verified_subset_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"MS3 baseline prep VERIFIED (seed={seed})")
    logger.info("=" * 60)

    import polars as pl

    df = pl.read_ndjson(verified_subset_path)
    missing_cols = [c for c in REQUIRED_VERIFIED_FIELDS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"verified subset missing required columns: {missing_cols}"
        )

    pairs = df.select(list(REQUIRED_VERIFIED_FIELDS)).to_dicts()
    logger.info(f"  loaded pairs: {len(pairs):,}")

    needed = val_size + test_size
    if len(pairs) < needed:
        raise ValueError(
            f"insufficient verified pairs: have {len(pairs):,}, "
            f"need >= {needed:,} (val={val_size} + test={test_size})"
        )

    val, test = _stratified_split_verified(
        pairs, val_size=val_size, test_size=test_size, seed=seed
    )
    logger.info(f"  val: {len(val):,}  test: {len(test):,}")

    val_clusters = {p["source_cluster_id"] for p in val}
    test_clusters = {p["source_cluster_id"] for p in test}
    leaked = val_clusters & test_clusters
    if leaked:
        raise RuntimeError(
            f"FATAL: source_cluster_id leakage: {len(leaked)} clusters in both"
        )

    _write_jsonl(out_dir / "gold_pairs_val.jsonl", val)
    _write_jsonl(out_dir / "gold_pairs_test.jsonl", test)

    val_dist: dict[str, int] = defaultdict(int)
    test_dist: dict[str, int] = defaultdict(int)
    for p in val:
        val_dist[p["source_court"]] += 1
    for p in test:
        test_dist[p["source_court"]] += 1

    gold_pair_hashes = {
        fname: hashlib.sha256((out_dir / fname).read_bytes()).hexdigest()
        for fname in ("gold_pairs_val.jsonl", "gold_pairs_test.jsonl")
    }
    summary_data = {
        "schema_version": SCHEMA_VERSION,
        "verified_subset_path": str(verified_subset_path),
        "verified_subset_sha": hashlib.sha256(
            verified_subset_path.read_bytes()
        ).hexdigest(),
        "n_pairs_loaded": len(pairs),
        "gold_pairs_val": len(val),
        "gold_pairs_test": len(test),
        "val_court_distribution": dict(val_dist),
        "test_court_distribution": dict(test_dist),
        "n_unique_clusters_val": len(val_clusters),
        "n_unique_clusters_test": len(test_clusters),
        "n_clusters_leaked": 0,
        "gold_pair_hashes": gold_pair_hashes,
        "seed": seed,
        "git_sha": _git_sha(),
    }
    summary_path = out_dir / "verified_split_summary.json"
    summary_path.write_text(
        json.dumps(summary_data, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    logger.info(f"  Wrote {summary_path}")
    return summary_data


# ---------- CLI ----------


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MS3 baseline dataset prep.")
    ap.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    ap.add_argument("--lepard-path", type=Path, default=DEFAULT_LEPARD)
    ap.add_argument("--cl-ids-path", type=Path, default=DEFAULT_CL_IDS)
    ap.add_argument("--court-map-path", type=Path, default=DEFAULT_COURT_MAP)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--verified-subset",
        type=Path,
        default=None,
        help="If set, use cluster-aware verified path",
    )
    ap.add_argument(
        "--enrich-corpus",
        action="store_true",
        help="Run enrich_corpus_with_cluster_id and exit",
    )
    ap.add_argument("--corpus-in", type=Path, default=None)
    ap.add_argument("--corpus-out", type=Path, default=None)
    ap.add_argument("--log-to-wandb", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-size", type=int, default=VAL_SIZE)
    ap.add_argument("--test-size", type=int, default=TEST_SIZE)
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print(f"[baseline_prep] DRY RUN  args={vars(args)}")
        sys.exit(0)
    if args.enrich_corpus:
        if args.corpus_in is None or args.corpus_out is None:
            print("--enrich-corpus requires --corpus-in and --corpus-out")
            sys.exit(2)
        enrich_corpus_with_cluster_id(
            shard_dir=args.shard_dir,
            corpus_in_path=args.corpus_in,
            corpus_out_path=args.corpus_out,
        )
        sys.exit(0)
    if args.verified_subset is not None:
        main_verified(
            verified_subset_path=args.verified_subset,
            out_dir=args.out_dir,
            seed=args.seed,
            val_size=args.val_size,
            test_size=args.test_size,
        )
        sys.exit(0)
    main(
        shard_dir=args.shard_dir,
        lepard_path=args.lepard_path,
        cl_ids_path=args.cl_ids_path,
        court_map_path=args.court_map_path,
        out_dir=args.out_dir,
        log_to_wandb=args.log_to_wandb,
        resume=args.resume,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
    )
