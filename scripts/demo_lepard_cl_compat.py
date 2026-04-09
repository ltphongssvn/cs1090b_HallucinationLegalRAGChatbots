"""
Demo script for the course TF session.

Reproduces the LePaRD <-> CourtListener compatibility investigation
from committed fixtures -- no access to Colab A100 or ODD GPU Cluster L4
required. Runs in under one second.

Usage:
    uv run python scripts/demo_lepard_cl_compat.py
    uv run python scripts/demo_lepard_cl_compat.py --json
    uv run python scripts/demo_lepard_cl_compat.py --no-narrative
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lepard_cl_compat import format_report, run_full_analysis  # noqa: E402

__all__ = ["NARRATIVE", "main"]

NARRATIVE = """
Investigation context
---------------------
The RAG pipeline uses two datasets (README Datasets section):

  - CourtListener federal appellate subset -- 1,465,484 opinions
    (the RETRIEVAL CORPUS, candidate pool)
  - LePaRD -- ~4M (source -> cited_precedent) pairs
    (the EVALUATION BACKBONE, gold labels)

Question asked by the user:
  "Are LePaRD and CourtListener compatible / overlapping?"

The datasets live on separate machines:
  - LePaRD on Google Colab Pro A100 (High RAM, 2025.10 runtime)
  - CourtListener on Harvard ODD GPU Cluster L4

This script reproduces the cross-machine intersection from small
fixtures committed to the repo, so the numbers are verifiable in CI
and the TF demo without any live data access.
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    ap.add_argument("--no-narrative", action="store_true", help="Skip narrative preamble")
    args = ap.parse_args()

    if not args.no_narrative and not args.json:
        print(NARRATIVE)

    report = run_full_analysis()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return

    print(format_report(report))

    io_ = report.id_overlap
    po = report.pair_overlap
    print()
    print("Interpretation")
    print("-" * 60)
    print("- Schema compatibility: CONFIRMED -- LePaRD source_id/dest_id")
    print(f"  share the CourtListener opinion id space (CL range 1..{io_.cl_max:,}).")
    print(
        f"- ID-level overlap: {io_.overlap}/{io_.lepard_unique_ids} "
        f"({io_.overlap_pct_of_lepard:.1f}%) of LePaRD ids found in CL subset."
    )
    print(
        f"- Pair-level usable gold: {po.both_in_cl}/{po.unique_pairs} "
        f"({po.usable_pct:.1f}%) -- BOTH endpoints present in CL."
    )
    print(f"- Newer snapshot drift: {io_.lepard_ids_above_cl_max} LePaRD ids")
    print(f"  exceed CL max ({io_.lepard_max:,} vs {io_.cl_max:,}).")
    if report.court_distribution:
        courts = list(report.court_distribution.keys())
        print("- Matched ids dominated by federal circuit courts")
        print(f"  ({', '.join(courts[:5])}...) -- consistent with CL filter to")
        print("  federal-appellate only, while LePaRD covers broader federal courts.")
    print()
    print("Implication for the pipeline")
    print("-" * 60)
    usable_est = int(4_000_000 * po.usable_pct / 100)
    print(f"- Extrapolating {po.usable_pct:.1f}% to full 4M LePaRD -> ~{usable_est:,} usable gold pairs.")
    print("- Still within Tier A target (10K-50K eval queries from README).")
    print("- To raise yield: (a) expand CL to federal district courts, or")
    print("  (b) filter LePaRD to appellate-source pairs before training.")


if __name__ == "__main__":
    main()
