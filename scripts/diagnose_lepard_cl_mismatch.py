#!/usr/bin/env python3
"""
Resumable diagnostic + remediation pipeline for the LePaRD ↔ CourtListener
id-space mismatch discovered during Cell 15 all-zeros retrieval evaluation.

Root cause (confirmed)
----------------------
LePaRD `dest_id` / `source_id` are Caselaw Access Project (CAP) opinion ids,
NOT CourtListener `opinion.id`. The two id spaces are independent: a naive
numeric intersection produces spurious "matches" pointing to unrelated cases
(e.g. LePaRD dest_id=624356 = "Calhoon v. Sell" 1998 SD District; CL
opinion_id=624356 = "Singh v. US" 2012 9th Cir).

Remediation strategy
--------------------
Bridge CAP → CL via the citation reporter string (e.g. "71 F. Supp. 2d 990"):
    LePaRD.{dest,source}_cite
        → parse (volume, reporter, page)
        → CL citations table → cluster_id
        → CL opinions table → opinion.id
This yields a LePaRD_id → CL_opinion_id map that can rebuild gold pairs
correctly, unblocking Cells 12/13/14/15.

Pipeline stages
---------------
Each stage writes an output artifact AND a `.done` sentinel. On rerun, any
stage whose sentinel exists AND whose output is non-empty is skipped.
Delete the sentinel to force re-execution of a specific stage.

    Stage 1  parse LePaRD citations         → data/processed/diagnostic/lepard_id_to_citation.jsonl
    Stage 2  build CL citation index        → data/processed/diagnostic/cl_citation_to_cluster.jsonl
    Stage 3  join LePaRD → CL cluster_ids   → data/processed/diagnostic/lepard_id_to_cl_cluster.jsonl
    Stage 4  build cluster_id → opinion_id  → data/processed/diagnostic/cl_cluster_to_opinion.json
    Stage 5  final LePaRD id → CL opinion   → data/processed/diagnostic/lepard_id_to_cl_opinion.json
    Stage 6  validation sample              → data/processed/diagnostic/validation_report.json

Usage
-----
    # Full pipeline (resumes from last checkpoint automatically):
    .venv/bin/python scripts/diagnose_lepard_cl_mismatch.py

    # Force re-run of a single stage:
    .venv/bin/python scripts/diagnose_lepard_cl_mismatch.py --force-stage 4

    # Run only stages 1-3 (skip the expensive opinions-CSV scan):
    .venv/bin/python scripts/diagnose_lepard_cl_mismatch.py --max-stage 3

    # Background with logging (survives session drops):
    nohup .venv/bin/python scripts/diagnose_lepard_cl_mismatch.py \
        > logs/diagnose_$(date +%s).log 2>&1 &

Reproducibility
---------------
All outputs are deterministic given the same inputs. Stage 4 (opinions scan)
is the expensive one (~51GB bz2 → ~10-20 min sequential decompression) and
filters on-the-fly to only cluster_ids referenced by LePaRD, keeping memory
bounded. If interrupted mid-stage, rerun — completed stages are skipped via
sentinels.
"""
from __future__ import annotations

import argparse
import bz2
import csv
import json
import logging
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# ----- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LEPARD_JSONL = REPO_ROOT / "lepard_train_4000000_rev0194f95.jsonl"
CL_CITATIONS_CSV = REPO_ROOT / "data/raw/cl_bulk/citations-2025-12-31.csv.bz2"
CL_OPINIONS_CSV = REPO_ROOT / "data/raw/cl_bulk/opinions-2025-12-31.csv.bz2"
CORPUS_JSONL = REPO_ROOT / "data/processed/baseline/corpus_chunks.jsonl"

OUT_DIR = REPO_ROOT / "data/processed/diagnostic"
STAGE1_OUT = OUT_DIR / "lepard_id_to_citation.jsonl"
STAGE2_OUT = OUT_DIR / "cl_citation_to_cluster.jsonl"
STAGE3_OUT = OUT_DIR / "lepard_id_to_cl_cluster.jsonl"
STAGE4_OUT = OUT_DIR / "cl_cluster_to_opinion.json"
STAGE5_OUT = OUT_DIR / "lepard_id_to_cl_opinion.json"
STAGE6_OUT = OUT_DIR / "validation_report.json"

# ----- Logging ----------------------------------------------------------------


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("diagnose")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


log = _get_logger()

# ----- Checkpoint helpers -----------------------------------------------------


def _sentinel(path: Path) -> Path:
    """Sentinel file alongside an output artifact (`.done`)."""
    return path.with_suffix(path.suffix + ".done")


def _stage_complete(out: Path) -> bool:
    """A stage is complete iff sentinel exists AND output file is non-empty."""
    return _sentinel(out).exists() and out.exists() and out.stat().st_size > 0


def _mark_done(out: Path, meta: dict[str, Any]) -> None:
    _sentinel(out).write_text(
        json.dumps({**meta, "finished_at": time.time()}, indent=2, sort_keys=True),
    )


def _skip_or_run(stage_num: int, out: Path, force: set[int], max_stage: int) -> bool:
    """Return True iff this stage should run."""
    if stage_num > max_stage:
        log.info(f"stage {stage_num}: skipped (max_stage={max_stage})")
        return False
    if stage_num in force:
        log.info(f"stage {stage_num}: FORCED — deleting sentinel")
        _sentinel(out).unlink(missing_ok=True)
        return True
    if _stage_complete(out):
        log.info(
            f"stage {stage_num}: already complete ({out.name}, "
            f"{out.stat().st_size / 1e6:.1f} MB) — skipping"
        )
        return False
    return True


# ----- Stage 1: parse LePaRD citations ---------------------------------------

_CITE_RE = re.compile(r"(\d+)\s+([A-Z][A-Za-z\.\s\d]+?)\s+(\d+)\s*\(")


def _parse_citation(cite: str) -> tuple[str, str, str] | None:
    """Extract (volume, reporter, page) from a string like
    '71 F. Supp. 2d 990 (1998)'. Returns None if no match."""
    m = _CITE_RE.search(cite or "")
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


def stage1_parse_lepard(lepard_path: Path, out: Path) -> dict[str, int]:
    """Parse unique {source,dest}_id → (volume, reporter, page) from LePaRD."""
    log.info(f"stage 1: parsing LePaRD citations from {lepard_path.name}")
    seen: dict[int, tuple[str, str, str]] = {}
    n_rows = 0
    with lepard_path.open(encoding="utf-8") as f:
        for line in f:
            n_rows += 1
            r = json.loads(line)
            for side in ("dest", "source"):
                lid = int(r[f"{side}_id"])
                if lid in seen:
                    continue
                parsed = _parse_citation(r.get(f"{side}_cite", ""))
                if parsed is not None:
                    seen[lid] = parsed
            if n_rows % 500_000 == 0:
                log.info(
                    f"  [{n_rows:>10,} rows] unique parsed ids: {len(seen):>10,}"
                )

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for lid, (v, rep, pg) in seen.items():
            f.write(
                json.dumps(
                    {"lepard_id": lid, "volume": v, "reporter": rep, "page": pg}
                )
                + "\n"
            )
    tmp.rename(out)  # atomic

    meta = {"n_lepard_rows": n_rows, "n_ids_parsed": len(seen)}
    _mark_done(out, meta)
    log.info(f"stage 1: wrote {out.name}  ({out.stat().st_size / 1e6:.1f} MB)  {meta}")
    return meta


# ----- Stage 2: CL citation index --------------------------------------------


def stage2_index_cl_citations(cl_csv: Path, out: Path) -> dict[str, int]:
    """Build (volume, reporter, page) → cluster_id table from CL citations."""
    log.info(f"stage 2: indexing CL citations from {cl_csv.name}")
    csv.field_size_limit(sys.maxsize)

    n_rows = 0
    n_unique = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    seen_keys: set[tuple[str, str, str]] = set()
    with bz2.open(cl_csv, "rt", encoding="utf-8") as fin, tmp.open(
        "w", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            n_rows += 1
            try:
                key = (
                    row["volume"].strip(),
                    row["reporter"].strip(),
                    row["page"].strip(),
                )
                cid = int(row["cluster_id"])
            except (ValueError, KeyError):
                continue
            if key in seen_keys:
                continue  # keep first-seen cluster for collisions
            seen_keys.add(key)
            n_unique += 1
            fout.write(
                json.dumps(
                    {
                        "volume": key[0],
                        "reporter": key[1],
                        "page": key[2],
                        "cluster_id": cid,
                    }
                )
                + "\n"
            )
            if n_rows % 2_000_000 == 0:
                log.info(
                    f"  [{n_rows:>10,} rows] unique keys: {n_unique:>10,}"
                )
    tmp.rename(out)

    meta = {"n_citation_rows": n_rows, "n_unique_keys": n_unique}
    _mark_done(out, meta)
    log.info(f"stage 2: wrote {out.name}  ({out.stat().st_size / 1e6:.1f} MB)  {meta}")
    return meta


# ----- Stage 3: join LePaRD → CL cluster -------------------------------------


def _iter_jsonl(p: Path) -> Iterator[dict[str, Any]]:
    with p.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def stage3_join(lepard_cites: Path, cl_cites: Path, out: Path) -> dict[str, int]:
    """Join LePaRD ids to CL cluster_ids by (volume, reporter, page)."""
    log.info("stage 3: loading CL citation index into memory")
    cl_index: dict[tuple[str, str, str], int] = {}
    for row in _iter_jsonl(cl_cites):
        cl_index[(row["volume"], row["reporter"], row["page"])] = row["cluster_id"]
    log.info(f"  CL index: {len(cl_index):,} keys")

    log.info("stage 3: joining LePaRD citations → CL clusters")
    n_lepard = matched = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in _iter_jsonl(lepard_cites):
            n_lepard += 1
            key = (row["volume"], row["reporter"], row["page"])
            cid = cl_index.get(key)
            if cid is not None:
                matched += 1
                f.write(
                    json.dumps({"lepard_id": row["lepard_id"], "cl_cluster_id": cid})
                    + "\n"
                )
    tmp.rename(out)

    meta = {
        "n_lepard_ids": n_lepard,
        "n_matched": matched,
        "match_rate": round(matched / max(n_lepard, 1), 4),
    }
    _mark_done(out, meta)
    log.info(f"stage 3: wrote {out.name}  ({out.stat().st_size / 1e6:.1f} MB)  {meta}")
    return meta


# ----- Stage 4: cluster_id → opinion_id (filtered) ---------------------------


def stage4_cluster_to_opinion(
    lepard_cluster: Path, cl_opinions: Path, out: Path
) -> dict[str, int]:
    """Scan CL opinions CSV, keep only cluster_ids referenced by LePaRD."""
    log.info("stage 4: loading needed cluster_ids")
    needed: set[int] = set()
    for r in _iter_jsonl(lepard_cluster):
        needed.add(int(r["cl_cluster_id"]))
    log.info(f"  needed cluster_ids: {len(needed):,}")

    log.info(f"stage 4: scanning {cl_opinions.name} (single-threaded bz2, column-projected)")
    csv.field_size_limit(sys.maxsize)

    # The opinions CSV has 21 columns, several holding multi-MB text
    # (plain_text, html, xml_harvard, html_with_citations). csv.DictReader
    # parses every column for every row, which on a 51GB bz2 file stalls
    # on single rows with >100MB embedded text. We only need `id` (col 0)
    # and `cluster_id` (col 20), so we use csv.reader and index by position,
    # still respecting CSV quoting rules for embedded newlines/quotes.
    f = bz2.open(cl_opinions, "rt", encoding="utf-8")
    c2o: dict[int, list[int]] = defaultdict(list)
    n_rows = n_kept = 0
    with f:
        reader = csv.reader(f)
        header = next(reader)
        id_idx = header.index("id")
        cid_idx = header.index("cluster_id")
        log.info(f"  column layout: id=col{id_idx}, cluster_id=col{cid_idx}")
        for row in reader:
            n_rows += 1
            try:
                cid = int(row[cid_idx])
                if cid in needed:
                    c2o[cid].append(int(row[id_idx]))
                    n_kept += 1
            except (ValueError, IndexError, TypeError):
                continue
            if n_rows % 100_000 == 0:
                log.info(
                    f"  [{n_rows:>10,} rows] kept: {n_kept:>8,}  "
                    f"clusters matched: {len(c2o):>7,}/{len(needed):,}"
                )
    # Pick smallest opinion.id per cluster (primary/lead opinion)
    simple = {str(k): min(v) for k, v in c2o.items()}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(simple))
    tmp.rename(out)

    meta = {
        "n_opinions_rows": n_rows,
        "n_opinions_kept": n_kept,
        "n_clusters_matched": len(c2o),
        "n_clusters_needed": len(needed),
        "coverage": round(len(c2o) / max(len(needed), 1), 4),
    }
    _mark_done(out, meta)
    log.info(f"stage 4: wrote {out.name}  ({out.stat().st_size / 1e6:.1f} MB)  {meta}")
    return meta


# ----- Stage 5: final LePaRD_id → CL opinion_id ------------------------------


def stage5_final_map(
    lepard_cluster: Path, cluster_opinion: Path, out: Path
) -> dict[str, int]:
    """Compose LePaRD_id → cluster_id → opinion_id."""
    log.info("stage 5: composing final LePaRD → CL opinion map")
    c2o = json.loads(cluster_opinion.read_text())
    final: dict[str, int] = {}
    n_in = n_out = 0
    for r in _iter_jsonl(lepard_cluster):
        n_in += 1
        cid = str(r["cl_cluster_id"])
        oid = c2o.get(cid)
        if oid is not None:
            final[str(r["lepard_id"])] = int(oid)
            n_out += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(final))
    tmp.rename(out)

    meta = {
        "n_lepard_cluster_pairs": n_in,
        "n_final_mappings": n_out,
        "coverage": round(n_out / max(n_in, 1), 4),
    }
    _mark_done(out, meta)
    log.info(f"stage 5: wrote {out.name}  ({out.stat().st_size / 1e6:.1f} MB)  {meta}")
    return meta


# ----- Stage 6: validation ---------------------------------------------------


def stage6_validate(
    final_map: Path, corpus: Path, lepard: Path, out: Path, n_sample: int = 20
) -> dict[str, Any]:
    """Validate: for N sampled mappings, confirm the CL opinion's text
    contains the LePaRD `quote` snippet (evidence of true citation match)."""
    log.info(f"stage 6: validating on {n_sample}-sample")
    import random

    mapping: dict[str, int] = json.loads(final_map.read_text())
    log.info(f"  loaded {len(mapping):,} LePaRD → CL opinion mappings")

    rng = random.Random(0)
    sample_ids = rng.sample(list(mapping.keys()), min(n_sample, len(mapping)))
    sample_set = {int(x) for x in sample_ids}

    lepard_rows: dict[int, dict[str, Any]] = {}
    with lepard.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            did = int(r["dest_id"])
            if did in sample_set and did not in lepard_rows:
                lepard_rows[did] = {
                    "dest_name": r["dest_name"],
                    "dest_cite": r["dest_cite"],
                    "quote": r.get("quote", "") or "",
                }
            if len(lepard_rows) >= len(sample_set):
                break

    target_cl_oids = {mapping[str(lid)] for lid in sample_set if str(lid) in mapping}
    cl_texts: dict[int, str] = defaultdict(str)
    with corpus.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            oid = int(c["opinion_id"])
            if oid in target_cl_oids:
                cl_texts[oid] += " " + (c.get("text", "") or "")

    report = {"n_sample": 0, "n_quote_found_in_cl": 0, "details": []}
    for lid in sample_set:
        key = str(lid)
        if key not in mapping:
            continue
        cl_oid = mapping[key]
        lepard_meta = lepard_rows.get(lid)
        if lepard_meta is None:
            continue
        report["n_sample"] += 1
        quote = lepard_meta["quote"]
        snippet = quote[:60].strip()
        cl_text = cl_texts.get(cl_oid, "")
        found = snippet.lower() in cl_text.lower() if snippet else False
        if found:
            report["n_quote_found_in_cl"] += 1
        report["details"].append(
            {
                "lepard_id": lid,
                "cl_opinion_id": cl_oid,
                "dest_name": lepard_meta["dest_name"],
                "dest_cite": lepard_meta["dest_cite"],
                "quote_snippet": snippet[:80],
                "quote_found_in_cl_text": found,
                "cl_text_len": len(cl_text),
            }
        )

    report["hit_rate"] = round(
        report["n_quote_found_in_cl"] / max(report["n_sample"], 1), 3
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True))
    tmp.rename(out)

    meta = {
        "n_sample": report["n_sample"],
        "n_quote_found_in_cl": report["n_quote_found_in_cl"],
        "hit_rate": report["hit_rate"],
    }
    _mark_done(out, meta)
    log.info(
        f"stage 6: wrote {out.name}  {meta}  "
        f"(hit_rate > 0 confirms CAP→CL bridge works)"
    )
    return meta


# ----- Orchestration ----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--force-stage",
        type=int,
        action="append",
        default=[],
        help="force re-run of a specific stage (repeatable)",
    )
    ap.add_argument(
        "--max-stage", type=int, default=6, help="stop after this stage (1-6)"
    )
    ap.add_argument("--lepard", type=Path, default=LEPARD_JSONL)
    ap.add_argument("--cl-citations", type=Path, default=CL_CITATIONS_CSV)
    ap.add_argument("--cl-opinions", type=Path, default=CL_OPINIONS_CSV)
    ap.add_argument("--corpus", type=Path, default=CORPUS_JSONL)
    args = ap.parse_args(argv)
    force = set(args.force_stage)
    max_stage = args.max_stage

    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"output dir: {OUT_DIR}")
    log.info(f"force-stage: {sorted(force) if force else 'none'}  max-stage: {max_stage}")

    for label, p in [
        ("LePaRD", args.lepard),
        ("CL citations", args.cl_citations),
        ("CL opinions", args.cl_opinions),
    ]:
        if not p.exists():
            log.error(f"missing input {label}: {p}")
            return 2

    if _skip_or_run(1, STAGE1_OUT, force, max_stage):
        stage1_parse_lepard(args.lepard, STAGE1_OUT)
    if _skip_or_run(2, STAGE2_OUT, force, max_stage):
        stage2_index_cl_citations(args.cl_citations, STAGE2_OUT)
    if _skip_or_run(3, STAGE3_OUT, force, max_stage):
        stage3_join(STAGE1_OUT, STAGE2_OUT, STAGE3_OUT)
    if _skip_or_run(4, STAGE4_OUT, force, max_stage):
        stage4_cluster_to_opinion(STAGE3_OUT, args.cl_opinions, STAGE4_OUT)
    if _skip_or_run(5, STAGE5_OUT, force, max_stage):
        stage5_final_map(STAGE3_OUT, STAGE4_OUT, STAGE5_OUT)
    if _skip_or_run(6, STAGE6_OUT, force, max_stage):
        if args.corpus.exists():
            stage6_validate(STAGE5_OUT, args.corpus, args.lepard, STAGE6_OUT)
        else:
            log.warning(
                f"stage 6: corpus not found at {args.corpus} — skipping validation"
            )

    elapsed = time.perf_counter() - t0
    log.info(f"pipeline complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
