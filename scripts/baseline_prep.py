"""MS3 Baseline dataset prep: chunk CL corpus + extract usable gold pairs.

Thin orchestration over:
    - src.lepard_cl_compat primitives (_load_cl_ids-equivalent)
    - Polars for corpus shard scanning
    - transformers.AutoTokenizer for 1024-subword, 128-overlap chunking

Outputs (data/processed/baseline/):
    corpus_chunks.jsonl        — one row per chunk (opinion_id, chunk_index, text)
    gold_pairs_val.jsonl       — 2K val pairs (stratified by source_court)
    gold_pairs_test.jsonl      — 45K test pairs (stratified by source_court)
    chunking_checkpoint.json   — per-shard resume state (atomic via os.replace)
    summary.json               — BaselinePrepSummary (Pydantic-validated)

Two split functions are provided:
  - _stratified_split: legacy cl_ids-based path (splits by source_id)
  - _stratified_split_verified: cluster-aware path for verified subset
    (splits by source_cluster_id to prevent retrieval-eval leakage)
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
    """Load CL opinion id universe from gzipped or plain text."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def _load_court_map(path: Path) -> dict[int, str]:
    """Load {opinion_id: court_id} from JSON (keys serialized as strings)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def _iter_usable_gold(
    lepard_path: Path,
    cl_ids: set[int],
) -> Iterator[dict[str, Any]]:
    """Stream LePaRD pairs, yielding only those where both endpoints are in CL."""
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
    """Add source_court from the matched-courts lookup."""
    return [
        {**p, "source_court": court_map.get(p["source_id"], "unknown")}
        for p in pairs
    ]


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
    """Chunk text into 1024-subword windows with 128-subword overlap.

    Matches README chunking policy (Revision 4). Returns list of
    {opinion_id, chunk_index, text} dicts.
    """
    tokens = tok.encode(text, add_special_tokens=False)
    if not tokens:
        return []
    stride = CHUNK_SIZE_SUBWORDS - CHUNK_OVERLAP_SUBWORDS  # 896
    chunks: list[dict[str, Any]] = []
    idx = 0
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE_SUBWORDS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tok.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(
            {
                "opinion_id": opinion_id,
                "chunk_index": idx,
                "text": chunk_text,
            }
        )
        idx += 1
        if end >= len(tokens):
            break
        start += stride
    return chunks


# ---------- checkpoint (atomic) ----------


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8"))["completed"])


def _save_checkpoint(path: Path, completed: set[str]) -> None:
    """Atomic write: tempfile.mkstemp + os.replace (repo convention)."""
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
    """Stream CL shards, chunk every opinion, write corpus_chunks.jsonl.

    Returns (n_chunks, n_opinions). Resumes from last checkpoint if
    resume=True; idempotent: counts pre-existing chunks from prior runs.
    """
    completed = _load_checkpoint(ckpt_path) if resume else set()
    shards = sorted(shard_dir.glob("shard_*.jsonl"))
    mode = "a" if completed else "w"
    n_chunks = 0
    n_opinions = 0

    # Idempotency: count pre-existing chunks from prior resumed runs
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
            _save_checkpoint(ckpt_path, completed)
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


# ---------- verified-subset consumer (lepard-eda branch port) -------------


def _iter_verified_subset(path: Path) -> Iterator[dict[str, Any]]:
    """Stream rows from data/processed/lepard_cl_verified_subset.jsonl.

    Output of scripts/build_lepard_cl_subset.py: each row has
    source_id, source_cluster_id, source_court, dest_id, quote,
    destination_context, quote_fuzzy_score.

    Raises json.JSONDecodeError on malformed lines (fail-fast — repo
    convention; matches _iter_usable_gold above).
    """
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _stratified_split_verified(
    pairs: list[dict[str, Any]],
    *,
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split verified pairs with cluster-aware leakage prevention.

    Critical retrieval-eval contract: same source_cluster_id (cited opinion)
    must NEVER appear in both val and test. Multiple LePaRD pairs cite the
    same opinion — without group-aware splitting, the same opinion text
    leaks across train/eval, inflating retrieval metrics.

    Algorithm (sklearn GroupShuffleSplit + stratification composition):
      1. Group pairs by source_cluster_id.
      2. Stratify groups (each cluster as one unit) by majority
         source_court within the group, using existing _stratified_split.
      3. Expand groups → flat pair lists, truncated to val_size / test_size.
    """
    # 1. Group pairs by source_cluster_id
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[int(pair["source_cluster_id"])].append(pair)

    # 2. Build group-level records: stratum = majority court in group
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

    # 3. Convert pair-budgets to group-budgets (avg group size)
    n_groups = len(group_records)
    avg_group_size = max(1, len(pairs) // max(1, n_groups))
    group_val_size = (
        min(n_groups, max(1, val_size // avg_group_size)) if val_size else 0
    )
    group_test_size = (
        min(
            n_groups - group_val_size,
            max(1, test_size // avg_group_size),
        )
        if test_size
        else 0
    )

    val_groups, test_groups = _stratified_split(
        group_records,
        val_size=group_val_size,
        test_size=group_test_size,
        seed=seed,
    )

    # 4. Expand groups → pairs, truncate to requested pair budgets
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


# ---------- main ----------


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
    """Run baseline prep pipeline. Returns summary dict."""
    from src.eda_schemas import BaselinePrepSummary

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info(f"MS3 baseline prep  (seed={seed})")
    logger.info("=" * 60)
    logger.info(f"  shard_dir      : {shard_dir}")
    logger.info(f"  lepard_path    : {lepard_path}")
    logger.info(f"  cl_ids_path    : {cl_ids_path}")
    logger.info(f"  court_map_path : {court_map_path}")
    logger.info(f"  out_dir        : {out_dir}")

    logger.info("\n[1/3] Loading CL id universe + court map")
    cl_ids = _load_cl_ids(cl_ids_path)
    court_map = _load_court_map(court_map_path)
    logger.info(f"  cl_ids     : {len(cl_ids):,}")
    logger.info(f"  court_map  : {len(court_map):,} matched ids")

    logger.info("\n[2/3] Extracting usable gold pairs")
    raw_pairs = list(_iter_usable_gold(lepard_path, cl_ids))
    pairs = _annotate_source_court(raw_pairs, court_map)
    seen: set[tuple[int, int]] = set()
    deduped: list[dict[str, Any]] = []
    for p in pairs:
        k = (p["source_id"], p["dest_id"])
        if k not in seen:
            seen.add(k)
            deduped.append(p)
    logger.info(f"  raw pairs       : {len(pairs):,}")
    logger.info(f"  unique pairs    : {len(deduped):,}")

    val, test = _stratified_split(
        deduped,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    logger.info(f"  val split       : {len(val):,}")
    logger.info(f"  test split      : {len(test):,}")

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

    logger.info("\n[3/3] Chunking CL corpus")
    tok = _get_tokenizer()
    corpus_path = out_dir / "corpus_chunks.jsonl"
    ckpt_path = out_dir / "chunking_checkpoint.json"
    n_chunks, n_opinions = _chunk_corpus(
        shard_dir,
        corpus_path,
        ckpt_path,
        resume=resume,
        tok=tok,
    )
    logger.info(f"  chunks written  : {n_chunks:,}")
    logger.info(f"  opinions        : {n_opinions:,}")

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
    logger.info(f"\nWrote summary.json -> {summary_path}")

    if log_to_wandb:
        _log_to_wandb(summary_data, out_dir)
    return summary_data


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MS3 baseline dataset prep.")
    ap.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    ap.add_argument("--lepard-path", type=Path, default=DEFAULT_LEPARD)
    ap.add_argument("--cl-ids-path", type=Path, default=DEFAULT_CL_IDS)
    ap.add_argument("--court-map-path", type=Path, default=DEFAULT_COURT_MAP)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--log-to-wandb", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-size", type=int, default=VAL_SIZE)
    ap.add_argument("--test-size", type=int, default=TEST_SIZE)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs + print fingerprint without running pipeline",
    )
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print("[baseline_prep] DRY RUN")
        print(f"  schema_version : {SCHEMA_VERSION}")
        print(f"  CHUNK_SIZE     : {CHUNK_SIZE_SUBWORDS}")
        print(f"  CHUNK_OVERLAP  : {CHUNK_OVERLAP_SUBWORDS}")
        print(f"  ENCODER_MODEL  : {ENCODER_MODEL}")
        print(f"  VAL_SIZE       : {VAL_SIZE}")
        print(f"  TEST_SIZE      : {TEST_SIZE}")
        print(f"  python         : {sys.version.split()[0]}")
        print(f"  git_sha        : {_git_sha()}")
        print(f"  args           : {vars(args)}")
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
