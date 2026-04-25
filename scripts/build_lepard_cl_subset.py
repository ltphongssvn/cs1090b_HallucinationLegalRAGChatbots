"""Build a verified LePaRD subset (adapted from lepard-eda branch).

Two-stage filter:
  Stage 1 — eyecite parses source_cite → (vol, reporter, page) → CL cluster_id
  Stage 2 — rapidfuzz partial_ratio(quote, opinion_text) ≥ threshold

Adapted for feature/data-acquisition branch:
  - LePaRD JSONL at repo root (lepard_train_4000000_rev0194f95.jsonl)
  - CL citations at data/raw/cl_bulk/citations-2025-12-31.csv.bz2
  - CL shards at data/raw/cl_federal_appellate_bulk/

Output:
  data/processed/lepard_cl_verified_subset.jsonl
  data/processed/lepard_cl_verified_subset.summary.json
"""
from __future__ import annotations

import argparse
import bz2
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEPARD_JSONL = REPO_ROOT / "lepard_train_4000000_rev0194f95.jsonl"
DEFAULT_SHARD_DIR = REPO_ROOT / "data" / "raw" / "cl_federal_appellate_bulk"
DEFAULT_CITATIONS_BZ2 = REPO_ROOT / "data" / "raw" / "cl_bulk" / "citations-2025-12-31.csv.bz2"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "lepard_cl_verified_subset.jsonl"
DEFAULT_SAMPLE = 10_000
DEFAULT_FUZZY_THRESHOLD = 80.0

FEDERAL_APPELLATE_KEYWORDS = [
    "first circuit", "second circuit", "third circuit", "fourth circuit",
    "fifth circuit", "sixth circuit", "seventh circuit", "eighth circuit",
    "ninth circuit", "tenth circuit", "eleventh circuit",
    "district of columbia circuit", "d.c. circuit", "federal circuit",
    "court of appeals",
]


def normalize_reporter(rep: str) -> str:
    return re.sub(r"\s+", " ", rep.strip()).lower()


def is_federal_appellate(court_name: str) -> bool:
    low = court_name.lower()
    return any(kw in low for kw in FEDERAL_APPELLATE_KEYWORDS)


def parse_source_cite(cite_str: str) -> tuple[str, str, str] | None:
    """Parse 'X v. Y, 100 F.2d 50 (1940)' → ('100', 'f.2d', '50')."""
    if not cite_str:
        return None
    try:
        from eyecite import get_citations
        from eyecite.models import FullCaseCitation
    except ImportError as e:
        raise RuntimeError("eyecite not installed — run: uv add eyecite") from e
    for c in get_citations(cite_str):
        if isinstance(c, FullCaseCitation):
            g = c.groups
            vol = g.get("volume", "").strip()
            rep = normalize_reporter(g.get("reporter", ""))
            page = g.get("page", "").strip()
            if vol and rep and page:
                return (vol, rep, page)
    return None


def load_cl_citations_index(path: Path) -> dict[tuple[str, str, str], int]:
    """Build (volume, reporter, page) → cluster_id from CL citations CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"CL citations file not found at {path}. "
            f"Download via: curl -L -o {path} "
            f"https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/"
            f"bulk-data/{path.name}"
        )
    print(f"\n[Stage 1] Building CL citations index from {path.name} ...")
    t0 = time.time()
    csv.field_size_limit(sys.maxsize)
    index: dict[tuple[str, str, str], int] = {}
    n = 0
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            vol = row.get("volume", "").strip()
            rep = normalize_reporter(row.get("reporter", ""))
            page = row.get("page", "").strip()
            cid = row.get("cluster_id", "").strip()
            if vol and rep and page and cid:
                key = (vol, rep, page)
                if key not in index:
                    try:
                        index[key] = int(cid)
                    except ValueError:
                        pass
            if n % 1_000_000 == 0:
                print(
                    f"  {n:,} rows, {len(index):,} keys ({time.time() - t0:.0f}s)",
                    file=sys.stderr,
                )
    print(
        f"  Done: {n:,} rows → {len(index):,} unique (vol, reporter, page) keys "
        f"in {time.time() - t0:.1f}s"
    )
    return index


def build_shard_text_index(
    shard_dir: Path, target_ids: set[int]
) -> dict[int, str]:
    """Scan shard JSONL files for {cluster_id: text} for target cluster_ids."""
    shards = sorted(shard_dir.glob("shard_*.jsonl"))
    if not shards:
        raise RuntimeError(f"No shard_*.jsonl files found in {shard_dir}")
    print(
        f"\n[Stage 2] Scanning {len(shards)} shards for "
        f"{len(target_ids):,} cluster IDs ..."
    )
    t0 = time.time()
    index: dict[int, str] = {}
    remaining = len(target_ids)
    for shard in shards:
        if not remaining:
            break
        with shard.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = rec.get("cluster_id")
                if cid in target_ids and cid not in index:
                    index[cid] = rec.get("text") or rec.get("raw_text") or ""
                    remaining -= 1
                    if not remaining:
                        break
    print(
        f"  Done: {len(index):,}/{len(target_ids):,} cluster IDs resolved "
        f"in {time.time() - t0:.1f}s"
    )
    if len(index) < len(target_ids):
        missing = len(target_ids) - len(index)
        print(
            f"  Note: {missing:,} cluster IDs not found in shards "
            f"(may be non-appellate or filtered out during extraction)"
        )
    return index


def fuzzy_match_quote(
    quote: str, opinion_text: str, threshold: float
) -> tuple[bool, float]:
    """Return (passed, score) using rapidfuzz partial_ratio."""
    try:
        from rapidfuzz import fuzz
    except ImportError as e:
        raise RuntimeError("rapidfuzz not installed — run: uv add rapidfuzz") from e
    if not opinion_text or not quote:
        return False, 0.0
    score = fuzz.partial_ratio(quote, opinion_text)
    return score >= threshold, score


def iter_lepard(n: int | None, seed: int = 42):
    """Iterate LePaRD rows from local JSONL, optionally sampled deterministically."""
    if not LEPARD_JSONL.exists():
        raise RuntimeError(
            f"LePaRD JSONL not found at {LEPARD_JSONL}. "
            "Run notebook Cell 7 to download via scripts/ingest_lepard.py."
        )
    print(f"  Loading LePaRD from local JSONL: {LEPARD_JSONL}")
    if n is None:
        with LEPARD_JSONL.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    with LEPARD_JSONL.open("rb") as fh:
        total = sum(1 for _ in fh)
    print(f"  Total LePaRD rows: {total:,}")
    rng = random.Random(seed)
    selected = set(rng.sample(range(total), min(n, total)))
    print(f"  Sampling {len(selected):,} rows (seed={seed})")
    with LEPARD_JSONL.open() as fh:
        for i, line in enumerate(fh):
            if i in selected:
                line = line.strip()
                if line:
                    yield json.loads(line)


def build_subset(
    sample_size: int = DEFAULT_SAMPLE,
    full_lepard: bool = False,
    appellate_only: bool = False,
    text_verify: bool = True,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    seed: int = 42,
    citations_path: Path = DEFAULT_CITATIONS_BZ2,
    shard_dir: Path = DEFAULT_SHARD_DIR,
    out_path: Path = DEFAULT_OUT,
) -> dict:
    t_start = time.time()
    cl_index = load_cl_citations_index(citations_path)

    n = None if full_lepard else sample_size
    print(
        f"\n[Stage 1] Iterating LePaRD "
        f"({'all rows' if n is None else f'{n:,} sampled'}) ..."
    )
    counters: Counter = Counter()
    parse_cache: dict[str, tuple[str, str, str] | None] = {}
    stage1_matched: list[dict] = []
    for row in iter_lepard(n, seed):
        counters["total"] += 1
        src_court = row.get("source_court") or ""
        if appellate_only and not is_federal_appellate(src_court):
            counters["skipped_non_appellate"] += 1
            continue
        src_cite = row.get("source_cite") or ""
        if src_cite not in parse_cache:
            parse_cache[src_cite] = parse_source_cite(src_cite)
        parsed = parse_cache[src_cite]
        if parsed is None:
            counters["parse_fail"] += 1
            continue
        cluster_id = cl_index.get(parsed)
        if cluster_id is None:
            counters["cite_no_match"] += 1
            continue
        counters["cite_matched"] += 1
        stage1_matched.append({
            "source_id": row.get("source_id"),
            "source_name": row.get("source_name") or "",
            "source_cite": src_cite,
            "source_court": src_court,
            "source_date": row.get("source_date") or "",
            "source_cluster_id": cluster_id,
            "dest_id": row.get("dest_id"),
            "dest_name": row.get("dest_name") or "",
            "dest_cite": row.get("dest_cite") or "",
            "dest_court": row.get("dest_court") or "",
            "dest_date": row.get("dest_date") or "",
            "passage_id": row.get("passage_id") or "",
            "quote": row.get("quote") or "",
            "destination_context": row.get("destination_context") or "",
        })
        if counters["total"] % 50_000 == 0:
            print(
                f"  [{counters['total']:,} rows] "
                f"cite_matched={counters['cite_matched']:,} "
                f"({time.time() - t_start:.0f}s)",
                file=sys.stderr,
            )
    print(
        f"\n  Stage 1 complete: {counters['cite_matched']:,} / "
        f"{counters['total'] - counters['skipped_non_appellate']:,} cite-matched "
        f"in {time.time() - t_start:.1f}s"
    )

    shards_available = shard_dir.exists() and any(shard_dir.glob("shard_*.jsonl"))
    if text_verify and not shards_available:
        print(
            f"\n  [warn] --text-verify requested but no shards found at {shard_dir}"
        )
        print("         Falling back to citation-match-only output.")
        text_verify = False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not text_verify:
        with out_path.open("w", encoding="utf-8") as fout:
            for rec in stage1_matched:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counters["text_verified"] = counters["cite_matched"]
        counters["text_failed"] = 0
    else:
        target_ids = {rec["source_cluster_id"] for rec in stage1_matched}
        text_index = build_shard_text_index(shard_dir, target_ids)

        print(
            f"\n[Stage 2] Fuzzy-matching quotes (threshold={fuzzy_threshold}) ..."
        )
        t2 = time.time()
        with out_path.open("w", encoding="utf-8") as fout:
            for rec in stage1_matched:
                cid = rec["source_cluster_id"]
                opinion_text = text_index.get(cid, "")
                if not opinion_text:
                    counters["text_no_shard"] += 1
                    continue
                passed, score = fuzzy_match_quote(
                    rec["quote"], opinion_text, fuzzy_threshold
                )
                if passed:
                    counters["text_verified"] += 1
                    out_row = {**rec, "quote_fuzzy_score": round(score, 1)}
                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                else:
                    counters["text_failed"] += 1
        print(
            f"  Stage 2 complete in {time.time() - t2:.1f}s: "
            f"{counters['text_verified']:,} verified, "
            f"{counters['text_failed']:,} failed, "
            f"{counters['text_no_shard']:,} cluster not in shards"
        )

    elapsed = time.time() - t_start
    eligible = counters["total"] - counters["skipped_non_appellate"]
    final = counters["text_verified"]
    summary = {
        "full_lepard_run": full_lepard,
        "appellate_only": appellate_only,
        "text_verify": text_verify,
        "fuzzy_threshold": fuzzy_threshold,
        "total_processed": counters["total"],
        "skipped_non_appellate": counters["skipped_non_appellate"],
        "parse_fail": counters["parse_fail"],
        "cite_no_match": counters["cite_no_match"],
        "cite_matched": counters["cite_matched"],
        "cite_match_rate_pct": round(
            100 * counters["cite_matched"] / max(1, eligible), 2
        ),
        "text_no_shard": counters.get("text_no_shard", 0),
        "text_failed": counters.get("text_failed", 0),
        "text_verified": final,
        "overall_yield_pct": round(100 * final / max(1, eligible), 2),
        "output_path": str(out_path),
        "output_size_mb": round(out_path.stat().st_size / 1e6, 1)
        if out_path.exists()
        else 0,
        "elapsed_sec": round(elapsed, 1),
    }
    print("\n" + "=" * 60)
    print("  LePaRD → CourtListener verified subset — final report")
    print("=" * 60)
    print(f"  Rows processed        {counters['total']:,}")
    if appellate_only:
        print(f"  Skipped (non-aplt)    {counters['skipped_non_appellate']:,}")
    print(f"  Parse failures        {counters['parse_fail']:,}")
    print(f"  Cite no match         {counters['cite_no_match']:,}")
    print(
        f"  Cite matched          {counters['cite_matched']:,}  "
        f"({summary['cite_match_rate_pct']}%)"
    )
    if text_verify:
        print(f"  Cluster not in shards {counters['text_no_shard']:,}")
        print(
            f"  Quote fuzzy failed    {counters['text_failed']:,}  "
            f"(threshold={fuzzy_threshold})"
        )
        print(f"  Text verified         {final:,}")
    print(
        f"  Final yield           {final:,}  "
        f"({summary['overall_yield_pct']}% of eligible)"
    )
    print(f"  Output                {out_path}  ({summary['output_size_mb']} MB)")
    print(f"  Elapsed               {elapsed:.1f}s")
    print("=" * 60)

    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary → {summary_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    ap.add_argument("--full-lepard", action="store_true")
    ap.add_argument("--appellate-only", action="store_true")
    ap.add_argument("--no-text-verify", dest="text_verify", action="store_false")
    ap.add_argument(
        "--fuzzy-threshold", type=float, default=DEFAULT_FUZZY_THRESHOLD
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--citations-file", type=Path, default=DEFAULT_CITATIONS_BZ2)
    ap.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    build_subset(
        sample_size=args.sample,
        full_lepard=args.full_lepard,
        appellate_only=args.appellate_only,
        text_verify=args.text_verify,
        fuzzy_threshold=args.fuzzy_threshold,
        seed=args.seed,
        citations_path=args.citations_file,
        shard_dir=args.shard_dir,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
