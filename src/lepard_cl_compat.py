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

Semantic notes:
  - usable_pct is a UNIQUE-PAIR usability metric, not row-weighted.
    Duplicates in the LePaRD JSONL affect total_rows but not usable_pct.
  - court_distribution counts UNIQUE matched CL ids (not per-row occurrences).
  - Designed for fixture-scale analysis (1K-100K pairs). For full 4M+ corpus
    runs, consider a streaming/DuckDB-backed variant.
  - CLI policy: --min-usable-pct gate is evaluated BEFORE --write-valid-pairs
    export. If the gate fails, no export file is written.
  - Strict int validation: source_id / dest_id must be native JSON integers.
    Floats (1.5), bools (true), and strings ("123") are rejected with line context.
  - Strict CL id validation: must be canonical positive decimal integers
    (no leading zeros, no sign characters, not zero).

CLI:
    uv run python -m src.lepard_cl_compat
    uv run python -m src.lepard_cl_compat --lepard path/to.jsonl --cl-ids path/to.txt.gz
    uv run python -m src.lepard_cl_compat --min-usable-pct 70.0            # CI gate
    uv run python -m src.lepard_cl_compat --write-valid-pairs gold.jsonl   # emit usable pairs
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "DEFAULT_FIXTURES",
    "DEFAULT_LEPARD",
    "DEFAULT_CL_IDS",
    "DEFAULT_COURT_MAP",
    "IdOverlap",
    "PairOverlap",
    "CompatReport",
    "load_lepard_pairs",
    "load_cl_ids",
    "load_court_map",
    "compute_id_overlap",
    "compute_pair_overlap",
    "analyze_court_distribution",
    "extract_valid_pairs",
    "write_valid_pairs_jsonl",
    "build_report",
    "run_full_analysis",
    "format_report",
    "main",
]

DEFAULT_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
DEFAULT_LEPARD = DEFAULT_FIXTURES / "lepard_sample_1k.jsonl"
DEFAULT_CL_IDS = DEFAULT_FIXTURES / "cl_ids.txt.gz"
DEFAULT_COURT_MAP = DEFAULT_FIXTURES / "cl_matched_courts.json"


# ---------- data containers ----------


@dataclass(frozen=True)
class IdOverlap:
    lepard_unique_ids: int
    cl_unique_ids: int
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
    court_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- loaders ----------


def load_lepard_pairs(path: Path | str) -> list[tuple[int, int]]:
    """Return (source_id, dest_id) tuples -- duplicates preserved.

    Raises ValueError with line context on malformed JSON, missing
    required keys, or values that are not native JSON integers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LePaRD sample not found: {path}")
    pairs: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON at line {line_number} of {path}: {exc}") from exc
            for required_key in ("source_id", "dest_id"):
                if required_key not in record:
                    raise ValueError(f"missing required key {required_key!r} at line {line_number} of {path}")
                value = record[required_key]
                if type(value) is not int:
                    raise ValueError(
                        f"{required_key!r} must be int, got {type(value).__name__} "
                        f"{value!r} at line {line_number} of {path}"
                    )
            pairs.append((record["source_id"], record["dest_id"]))
    return pairs


def load_cl_ids(path: Path | str) -> set[int]:
    """Load CourtListener opinion ids. Supports .gz and plain text.

    Strict canonical-positive-decimal validation: rejects sign characters
    (+1, -5), leading zeros (001), and zero itself.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CL id file not found: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    ids: set[int] = set()
    with opener(path, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.isdigit() or (len(stripped) > 1 and stripped[0] == "0"):
                raise ValueError(
                    f"non-canonical CL id {stripped!r} at line {line_number} of {path} "
                    f"(expected positive decimal, no sign, no leading zeros)"
                )
            value = int(stripped)
            if value == 0:
                raise ValueError(
                    f"non-canonical CL id {stripped!r} at line {line_number} of {path} "
                    f"(zero is not a valid CourtListener opinion id)"
                )
            ids.add(value)
    return ids


def load_court_map(path: Path | str) -> dict[int, str]:
    """Load matched-id -> court_id map. Returns empty dict if missing."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


# ---------- analysis ----------


def compute_id_overlap(pairs: list[tuple[int, int]], cl_ids: set[int]) -> IdOverlap:
    lepard_ids: set[int] = set()
    for source_id, dest_id in pairs:
        lepard_ids.add(source_id)
        lepard_ids.add(dest_id)
    overlap = lepard_ids & cl_ids
    cl_max = max(cl_ids) if cl_ids else 0
    return IdOverlap(
        lepard_unique_ids=len(lepard_ids),
        cl_unique_ids=len(cl_ids),
        overlap=len(overlap),
        overlap_pct_of_lepard=100.0 * len(overlap) / len(lepard_ids) if lepard_ids else 0.0,
        lepard_max=max(lepard_ids) if lepard_ids else 0,
        cl_max=cl_max,
        lepard_ids_above_cl_max=sum(1 for x in lepard_ids if x > cl_max),
    )


def compute_pair_overlap(pairs: list[tuple[int, int]], cl_ids: set[int]) -> PairOverlap:
    """Compute pair-level bucket counts.

    usable_pct = both_in_cl / unique_pairs (UNIQUE-PAIR usability,
    not row-weighted). Duplicates in `pairs` affect total_rows only.
    """
    unique_pairs = set(pairs)
    both_in_cl = source_only_in_cl = dest_only_in_cl = neither_in_cl = 0
    for source_id, dest_id in unique_pairs:
        is_source_present = source_id in cl_ids
        is_dest_present = dest_id in cl_ids
        if is_source_present and is_dest_present:
            both_in_cl += 1
        elif is_source_present:
            source_only_in_cl += 1
        elif is_dest_present:
            dest_only_in_cl += 1
        else:
            neither_in_cl += 1
    unique_count = len(unique_pairs)
    return PairOverlap(
        total_rows=len(pairs),
        unique_pairs=unique_count,
        unique_sources=len({src for src, _ in unique_pairs}),
        unique_dests=len({dst for _, dst in unique_pairs}),
        both_in_cl=both_in_cl,
        source_only_in_cl=source_only_in_cl,
        dest_only_in_cl=dest_only_in_cl,
        neither_in_cl=neither_in_cl,
        usable_pct=100.0 * both_in_cl / unique_count if unique_count else 0.0,
    )


def analyze_court_distribution(
    pairs: list[tuple[int, int]],
    cl_ids: set[int],
    court_map: dict[int, str],
) -> dict[str, int]:
    """Return court_id -> count of UNIQUE matched CL ids (not per-row occurrences).

    Sorted by count descending, then court_id ascending (deterministic tie-break).
    """
    if not court_map:
        return {}
    matched_ids: set[int] = set()
    for source_id, dest_id in pairs:
        if source_id in cl_ids:
            matched_ids.add(source_id)
        if dest_id in cl_ids:
            matched_ids.add(dest_id)
    counts: Counter[str] = Counter(court_map[matched_id] for matched_id in matched_ids if matched_id in court_map)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def extract_valid_pairs(pairs: list[tuple[int, int]], cl_ids: set[int]) -> list[tuple[int, int]]:
    """Return deduplicated pairs where BOTH endpoints exist in CL (usable gold).

    Deduplication uses ``dict.fromkeys`` to preserve first-occurrence order
    deterministically (``set(pairs)`` would break determinism across runs).
    """
    return [
        (source_id, dest_id) for source_id, dest_id in dict.fromkeys(pairs) if source_id in cl_ids and dest_id in cl_ids
    ]


def write_valid_pairs_jsonl(pairs: list[tuple[int, int]], cl_ids: set[int], out_path: Path | str) -> int:
    """Write usable gold pairs to JSONL. Returns count written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid = extract_valid_pairs(pairs, cl_ids)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for source_id, dest_id in valid:
            f.write(json.dumps({"source_id": source_id, "dest_id": dest_id}) + "\n")
    return len(valid)


def build_report(
    pairs: list[tuple[int, int]],
    cl_ids: set[int],
    court_map: dict[int, str],
) -> CompatReport:
    """Build CompatReport from already-loaded data. Pure, no I/O."""
    return CompatReport(
        id_overlap=compute_id_overlap(pairs, cl_ids),
        pair_overlap=compute_pair_overlap(pairs, cl_ids),
        court_distribution=analyze_court_distribution(pairs, cl_ids, court_map),
    )


def run_full_analysis(
    lepard_path: Path | str = DEFAULT_LEPARD,
    cl_ids_path: Path | str = DEFAULT_CL_IDS,
    court_map_path: Path | str = DEFAULT_COURT_MAP,
) -> CompatReport:
    """Load inputs and build report. Convenience wrapper around build_report."""
    pairs = load_lepard_pairs(lepard_path)
    cl_ids = load_cl_ids(cl_ids_path)
    court_map = load_court_map(court_map_path)
    return build_report(pairs, cl_ids, court_map)


# ---------- presentation ----------


def format_report(report: CompatReport) -> str:
    id_ov = report.id_overlap
    pair_ov = report.pair_overlap
    lines = [
        "=" * 60,
        "LePaRD <-> CourtListener compatibility analysis",
        "=" * 60,
        "",
        "[1] ID-level overlap",
        f"  LePaRD unique ids:       {id_ov.lepard_unique_ids:,}",
        f"  CL unique ids:           {id_ov.cl_unique_ids:,}",
        f"  Overlap:                 {id_ov.overlap:,} ({id_ov.overlap_pct_of_lepard:.1f}% of LePaRD)",
        f"  LePaRD id range max:     {id_ov.lepard_max:,}",
        f"  CL id range max:         {id_ov.cl_max:,}",
        f"  LePaRD ids > CL max:     {id_ov.lepard_ids_above_cl_max:,} "
        f"(heuristic: may indicate misaligned or differently-sourced id spaces)",
        "",
        "[2] Pair-level overlap (both endpoints required for gold label)",
        f"  Total rows:              {pair_ov.total_rows:,}",
        f"  Unique pairs:            {pair_ov.unique_pairs:,}",
        f"  Unique sources / dests:  {pair_ov.unique_sources:,} / {pair_ov.unique_dests:,}",
        f"  Both endpoints in CL:    {pair_ov.both_in_cl:,} ({pair_ov.usable_pct:.1f}%)  <- USABLE GOLD",
        f"  Source only in CL:       {pair_ov.source_only_in_cl:,}",
        f"  Dest only in CL:         {pair_ov.dest_only_in_cl:,}",
        f"  Neither in CL:           {pair_ov.neither_in_cl:,}",
    ]
    if report.court_distribution:
        lines += ["", "[3] Court distribution of matched CL ids"]
        total = sum(report.court_distribution.values())
        lines.append(f"  Total matched with known court: {total}")
        for court, count in report.court_distribution.items():
            lines.append(f"    {court}: {count}")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------- CLI ----------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--lepard", type=Path, default=DEFAULT_LEPARD, help="LePaRD JSONL sample")
    ap.add_argument("--cl-ids", type=Path, default=DEFAULT_CL_IDS, help="CL id set (.txt or .txt.gz)")
    ap.add_argument("--court-map", type=Path, default=DEFAULT_COURT_MAP, help="Matched id -> court_id JSON")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable report")
    ap.add_argument(
        "--min-usable-pct",
        type=float,
        default=None,
        help="CI gate: exit non-zero if usable_pct falls below this threshold (evaluated BEFORE export)",
    )
    ap.add_argument(
        "--write-valid-pairs",
        type=Path,
        default=None,
        help="Write deduplicated usable gold pairs (both endpoints in CL) to this JSONL path",
    )
    args = ap.parse_args()

    pairs = load_lepard_pairs(args.lepard)
    cl_ids = load_cl_ids(args.cl_ids)
    court_map = load_court_map(args.court_map)
    report = build_report(pairs, cl_ids, court_map)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(format_report(report))

    if args.min_usable_pct is not None:
        actual = report.pair_overlap.usable_pct
        if actual < args.min_usable_pct:
            print(
                f"ERROR: usable_pct {actual:.2f}% below threshold {args.min_usable_pct:.2f}%",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.write_valid_pairs is not None:
        n = write_valid_pairs_jsonl(pairs, cl_ids, args.write_valid_pairs)
        print(f"[write-valid-pairs] wrote {n} pairs to {args.write_valid_pairs}", file=sys.stderr)


if __name__ == "__main__":
    main()
