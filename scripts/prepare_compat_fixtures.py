"""
Generate committable fixtures for src/lepard_cl_compat.py from live data.

One-time setup script run across two machines to capture the investigation
inputs into small files that can be committed to GitHub.

Workflow:
  1. On Google Colab Pro A100 (has LePaRD):
         uv run python scripts/prepare_compat_fixtures.py lepard \\
             --lepard data/raw/lepard/lepard_train_1000_rev0194f95.jsonl

     Produces: tests/fixtures/lepard_sample_1k.jsonl (~1 MB)

  2. Commit and pull on cluster.

  3. On Harvard OnDemand L4 Cluster (has CourtListener):
         uv run python scripts/prepare_compat_fixtures.py cl \\
             --cl-dir data/raw/cl_federal_appellate_bulk

     Produces:
         tests/fixtures/cl_ids.txt.gz          (~6-8 MB)
         tests/fixtures/cl_matched_courts.json (small)

  4. Commit all three files to GitHub.

After commit, anyone can run the analysis on any machine:
    uv run python -m src.lepard_cl_compat
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import shutil
from pathlib import Path

__all__ = ["prepare_lepard", "prepare_cl", "main"]


def prepare_lepard(lepard_path: Path, out_dir: Path) -> None:
    """Copy LePaRD JSONL sample to fixtures directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "lepard_sample_1k.jsonl"
    shutil.copy2(lepard_path, dest)
    size_mb = dest.stat().st_size / 1e6
    with dest.open(encoding="utf-8") as f:
        n = sum(1 for _ in f)
    print(f"[lepard] wrote {dest} ({size_mb:.2f} MB, {n} rows)")


def prepare_cl(cl_dir: Path, out_dir: Path, lepard_sample: Path | None) -> None:
    """Scan CL shards, write gzipped id set and matched-courts map."""
    out_dir.mkdir(parents=True, exist_ok=True)

    lepard_ids: set[int] = set()
    if lepard_sample and lepard_sample.exists():
        with lepard_sample.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                lepard_ids.add(int(r["source_id"]))
                lepard_ids.add(int(r["dest_id"]))
        print(f"[cl] loaded {len(lepard_ids)} LePaRD ids for court mapping")

    cl_ids: set[int] = set()
    matched_courts: dict[int, str] = {}

    shards = sorted(glob.glob(str(cl_dir / "shard_*.jsonl")))
    if not shards:
        raise FileNotFoundError(f"No shards found in {cl_dir}")

    for i, shard in enumerate(shards, 1):
        with open(shard, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                oid = int(r["id"])
                cl_ids.add(oid)
                if oid in lepard_ids:
                    matched_courts[oid] = str(r.get("court_id", "unknown"))
        if i % 20 == 0 or i == len(shards):
            print(f"[cl] scanned {i}/{len(shards)} shards, {len(cl_ids):,} ids so far")

    ids_path = out_dir / "cl_ids.txt.gz"
    with gzip.open(ids_path, "wt", encoding="utf-8") as f:
        for oid in sorted(cl_ids):
            f.write(f"{oid}\n")
    print(f"[cl] wrote {ids_path} ({ids_path.stat().st_size / 1e6:.2f} MB, {len(cl_ids):,} ids)")

    if matched_courts:
        courts_path = out_dir / "cl_matched_courts.json"
        with courts_path.open("w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in sorted(matched_courts.items())}, f, indent=2)
        print(f"[cl] wrote {courts_path} ({len(matched_courts)} matched ids)")
    else:
        print("[cl] no LePaRD sample provided; skipping court map")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    p_lepard = sub.add_parser("lepard", help="Copy LePaRD sample (run on Colab)")
    p_lepard.add_argument("--lepard", type=Path, required=True)
    p_lepard.add_argument("--out-dir", type=Path, default=Path("tests/fixtures"))

    p_cl = sub.add_parser("cl", help="Scan CL shards (run on cluster)")
    p_cl.add_argument("--cl-dir", type=Path, required=True)
    p_cl.add_argument("--out-dir", type=Path, default=Path("tests/fixtures"))
    p_cl.add_argument(
        "--lepard-sample",
        type=Path,
        default=Path("tests/fixtures/lepard_sample_1k.jsonl"),
    )

    args = ap.parse_args()
    if args.mode == "lepard":
        prepare_lepard(args.lepard, args.out_dir)
    else:
        prepare_cl(args.cl_dir, args.out_dir, args.lepard_sample)


if __name__ == "__main__":
    main()
