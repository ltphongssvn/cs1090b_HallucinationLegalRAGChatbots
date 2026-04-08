"""
LePaRD <-> CourtListener compatibility analysis.

Reproduces the cross-machine investigation (Colab A100 holds LePaRD,
ODD GPU Cluster L4 holds CourtListener) as a single-machine, file-based
analysis runnable from committed GitHub fixtures.

Inputs (all paths configurable; defaults point at tests/fixtures/):
  - LePaRD JSONL sample (source_id, dest_id, ...)
  - CourtListener opinion id set (gzipped newline-delimited ints)
  - Optional: matched-id -> court_id map (JSON) for court distribution

Outputs: a CompatReport dataclass with id overlap, pair overlap, and
court distribution -- identical numbers to the live cross-machine run.

CLI:
    uv run python -m src.lepard_cl_compat
    uv run python -m src.lepard_cl_compat --lepard path/to.jsonl --cl-ids path/to.txt.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
DEFAULT_LEPARD = DEFAULT_FIXTURES / "lepard_sample_1k.jsonl"
DEFAULT_CL_IDS = DEFAULT_FIXTURES / "cl_ids.txt.gz"
DEFAULT_COURT_MAP = DEFAULT_FIXTURES / "cl_matched_courts.json"


# ---------- data containers ----------


@dataclass(frozen=True)
class IdOverlap:
    lepard_unique_ids: int
    cl_total_ids: int
    overlap: int
    overlap_pct_of_lepard: float
    lepard_max: int
    cl_max: int
    lepard_ids_above_cl_max: int


@dataclass(frozen=True)
class PairOverlap:
    total_rows: int
    unique_pairs: int
    unique_sources: int
    unique_dests: int
    both_in_cl: int
    source_only_in_cl: int
    dest_only_in_cl: int
    neither_in_cl: int
    usable_pct: float


@dataclass(frozen=True)
class CompatReport:
    id_overlap: IdOverlap
    pair_overlap: PairOverlap
    court_distribution: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id_overlap": asdict(self.id_overlap),
            "pair_overlap": asdict(self.pair_overlap),
            "court_distribution": dict(self.court_distribution),
        }


# ---------- loaders ----------


def load_lepard_pairs(path):
    """Return (source_id, dest_id) tuples -- duplicates preserved."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LePaRD sample not found: {path}")
    pairs = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"malformed JSON at line {i} of {path}: {e}") from e
            for key in ("source_id", "dest_id"):
                if key not in r:
                    raise ValueError(f"missing required key {key!r} at line {i} of {path}")
            pairs.append((int(r["source_id"]), int(r["dest_id"])))
    return pairs


def load_cl_ids(path):
    """Load CourtListener opinion ids. Supports .gz and plain text."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CL id file not found: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return {int(line) for line in f if line.strip()}


def load_court_map(path):
    """Load matched-id -> court_id map. Returns empty dict if missing."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


# ---------- analysis ----------


def compute_id_overlap(pairs, cl_ids):
    lepard_ids = set()
    for s, d in pairs:
        lepard_ids.add(s)
        lepard_ids.add(d)
    overlap = lepard_ids & cl_ids
    cl_max = max(cl_ids) if cl_ids else 0
    return IdOverlap(
        lepard_unique_ids=len(lepard_ids),
        cl_total_ids=len(cl_ids),
        overlap=len(overlap),
        overlap_pct_of_lepard=100.0 * len(overlap) / len(lepard_ids) if lepard_ids else 0.0,
        lepard_max=max(lepard_ids) if lepard_ids else 0,
        cl_max=cl_max,
        lepard_ids_above_cl_max=sum(1 for x in lepard_ids if x > cl_max),
    )


def compute_pair_overlap(pairs, cl_ids):
    unique = set(pairs)
    both = src_only = dst_only = neither = 0
    for s, d in unique:
        s_in = s in cl_ids
        d_in = d in cl_ids
        if s_in and d_in:
            both += 1
        elif s_in:
            src_only += 1
        elif d_in:
            dst_only += 1
        else:
            neither += 1
    n = len(unique)
    return PairOverlap(
        total_rows=len(pairs),
        unique_pairs=n,
        unique_sources=len({s for s, _ in unique}),
        unique_dests=len({d for _, d in unique}),
        both_in_cl=both,
        source_only_in_cl=src_only,
        dest_only_in_cl=dst_only,
        neither_in_cl=neither,
        usable_pct=100.0 * both / n if n else 0.0,
    )


def analyze_court_distribution(pairs, cl_ids, court_map):
    """Return court_id -> count for LePaRD ids that are present in CL."""
    if not court_map:
        return {}
    matched = set()
    for s, d in pairs:
        if s in cl_ids:
            matched.add(s)
        if d in cl_ids:
            matched.add(d)
    dist = {}
    for mid in matched:
        court = court_map.get(mid)
        if court is not None:
            dist[court] = dist.get(court, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: -kv[1]))


def run_full_analysis(
    lepard_path=DEFAULT_LEPARD,
    cl_ids_path=DEFAULT_CL_IDS,
    court_map_path=DEFAULT_COURT_MAP,
):
    pairs = load_lepard_pairs(lepard_path)
    cl_ids = load_cl_ids(cl_ids_path)
    court_map = load_court_map(court_map_path)
    return CompatReport(
        id_overlap=compute_id_overlap(pairs, cl_ids),
        pair_overlap=compute_pair_overlap(pairs, cl_ids),
        court_distribution=analyze_court_distribution(pairs, cl_ids, court_map),
    )


# ---------- presentation ----------


def format_report(report):
    io_ = report.id_overlap
    po = report.pair_overlap
    lines = [
        "=" * 60,
        "LePaRD <-> CourtListener compatibility analysis",
        "=" * 60,
        "",
        "[1] ID-level overlap",
        f"  LePaRD unique ids:       {io_.lepard_unique_ids:,}",
        f"  CL total ids:            {io_.cl_total_ids:,}",
        f"  Overlap:                 {io_.overlap:,} ({io_.overlap_pct_of_lepard:.1f}% of LePaRD)",
        f"  LePaRD id range max:     {io_.lepard_max:,}",
        f"  CL id range max:         {io_.cl_max:,}",
        f"  LePaRD ids > CL max:     {io_.lepard_ids_above_cl_max:,} (newer CL snapshot)",
        "",
        "[2] Pair-level overlap (both endpoints required for gold label)",
        f"  Total rows:              {po.total_rows:,}",
        f"  Unique pairs:            {po.unique_pairs:,}",
        f"  Unique sources / dests:  {po.unique_sources:,} / {po.unique_dests:,}",
        f"  Both endpoints in CL:    {po.both_in_cl:,} ({po.usable_pct:.1f}%)  <- USABLE GOLD",
        f"  Source only in CL:       {po.source_only_in_cl:,}",
        f"  Dest only in CL:         {po.dest_only_in_cl:,}",
        f"  Neither in CL:           {po.neither_in_cl:,}",
    ]
    if report.court_distribution:
        lines += ["", "[3] Court distribution of matched CL ids"]
        total = sum(report.court_distribution.values())
        lines.append(f"  Total matched: {total}")
        for court, n in report.court_distribution.items():
            lines.append(f"    {court}: {n}")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------- CLI ----------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--lepard", type=Path, default=DEFAULT_LEPARD, help="LePaRD JSONL sample")
    ap.add_argument("--cl-ids", type=Path, default=DEFAULT_CL_IDS, help="CL id set (.txt or .txt.gz)")
    ap.add_argument("--court-map", type=Path, default=DEFAULT_COURT_MAP, help="Matched id -> court_id JSON")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable report")
    args = ap.parse_args()

    report = run_full_analysis(args.lepard, args.cl_ids, args.court_map)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()
