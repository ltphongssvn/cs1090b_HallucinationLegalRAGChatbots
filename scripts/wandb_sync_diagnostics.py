"""W&B offline-run sync diagnostics + sync + cleanup.

Reproducible end-to-end utility that consolidates the manual diagnostic
sequence used to resolve the 45-run offline sync queue:

  1. Inventory pending offline runs (count + total size + earliest/latest).
  2. Verify network reachability to api.wandb.ai:443.
  3. Verify W&B authentication (env var or ~/.netrc).
  4. Optionally batch-sync pending runs (`--sync`).
  5. Optionally clean already-synced run directories (`--clean`).

Usage:
    .venv/bin/python scripts/wandb_sync_diagnostics.py --check-only
    .venv/bin/python scripts/wandb_sync_diagnostics.py --sync
    .venv/bin/python scripts/wandb_sync_diagnostics.py --sync --clean

Designed for Harvard ODD GPU cluster context where compute nodes ran
W&B in offline mode; this script syncs from a node with internet.
"""
from __future__ import annotations
import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path


def inventory_offline_runs(wandb_dir: Path) -> dict:
    if not wandb_dir.is_dir():
        return {"count": 0, "total_bytes": 0, "earliest": None, "latest": None}
    runs = sorted(wandb_dir.glob("offline-run-*"))
    total = 0
    for r in runs:
        for f in r.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return {
        "count": len(runs),
        "total_bytes": total,
        "earliest": runs[0].name if runs else None,
        "latest": runs[-1].name if runs else None,
    }


def check_network(host: str = "api.wandb.ai", port: int = 443, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True, f"REACHABLE: {host}:{port}"
    except Exception as e:
        return False, f"BLOCKED: {host}:{port} ({e})"


def check_auth() -> dict:
    key_env = os.environ.get("WANDB_API_KEY", "")
    netrc = Path.home() / ".netrc"
    has_netrc_entry = False
    if netrc.exists():
        try:
            has_netrc_entry = "api.wandb.ai" in netrc.read_text()
        except Exception:
            has_netrc_entry = False
    return {
        "wandb_api_key_set": bool(key_env),
        "wandb_api_key_len": len(key_env),
        "netrc_exists": netrc.exists(),
        "netrc_has_wandb": has_netrc_entry,
        "wandb_mode": os.environ.get("WANDB_MODE", "(unset → online default)"),
    }


def run_sync_all(wandb_bin: str = ".venv/bin/wandb") -> int:
    print("\n>>> wandb sync --sync-all\n")
    proc = subprocess.run([wandb_bin, "sync", "--sync-all"], check=False)
    return proc.returncode


def run_clean(wandb_bin: str = ".venv/bin/wandb") -> int:
    print("\n>>> wandb sync --clean --clean-force\n")
    proc = subprocess.run(
        [wandb_bin, "sync", "--clean", "--clean-force"], check=False
    )
    return proc.returncode


def show_pending(wandb_bin: str = ".venv/bin/wandb") -> int:
    print("\n>>> wandb sync --show 50\n")
    proc = subprocess.run([wandb_bin, "sync", "--show", "50"], check=False)
    return proc.returncode



def verify_cloud(sync_log_ids: set[str] | None = None) -> int:
    """Query W&B API and verify all synced offline runs are visible in cloud.

    Lists every run across every project for the configured entity, classifies
    them into (a) runs from the current sync session and (b) prior-session
    legitimate runs, and prints a per-project breakdown plus reconciliation.
    """
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed", file=sys.stderr)
        return 1

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY", "phl690-harvard-extension-schol")

    print("\n" + "=" * 78)
    print(" CLOUD VERIFICATION (W&B API)")
    print("=" * 78)
    print(f"  Entity: {entity}\n")

    try:
        projects = list(api.projects(entity=entity))
    except Exception as e:
        print(f"  ERROR querying projects: {e}", file=sys.stderr)
        return 1

    print(f"  {'PROJECT':<48}  {'COUNT':>6}")
    print("  " + "-" * 56)

    all_runs: list[tuple[str, str, str, str, str]] = []
    total = 0
    for proj in projects:
        try:
            runs = list(api.runs(f"{entity}/{proj.name}"))
        except Exception:
            runs = []
        print(f"  {proj.name[:47]:<48}  {len(runs):>6}")
        total += len(runs)
        for r in runs:
            all_runs.append((proj.name, r.id, r.name or "-",
                             str(r.created_at), r.job_type or "-"))
    print("  " + "-" * 56)
    print(f"  {'TOTAL':<48}  {total:>6}")
    print(f"\n  Total cloud runs: {total}")

    # If a sync log was provided, reconcile
    if sync_log_ids is not None:
        cloud_ids = {rid for _, rid, *_ in all_runs}
        missing = sync_log_ids - cloud_ids
        extras = cloud_ids - sync_log_ids
        print("\n" + "-" * 78)
        print(f"  Sync-log runs:                                {len(sync_log_ids):>4}")
        print(f"  Verified in cloud:                            {len(sync_log_ids) - len(missing):>4}")
        print(f"  Missing in cloud (synced locally, not seen):  {len(missing):>4}")
        for m in missing:
            print(f"    - {m}")
        print(f"  Prior-session runs (legitimate, pre-sync):    {len(extras):>4}")
        if extras:
            print("\n  Prior-session runs (sample, max 20):")
            extra_runs = sorted(
                [r for r in all_runs if r[1] in extras],
                key=lambda x: x[3],
            )
            print(f"    {'PROJECT':<25} {'RUN_ID':<12} {'CREATED':<22} {'JOB_TYPE'}")
            for proj, rid, _, created, jt in extra_runs[:20]:
                print(f"    {proj[:24]:<25} {rid:<12} {created:<22} {jt}")
            if len(extras) > 20:
                print(f"    ... and {len(extras) - 20} more")
    return 0



def verify_cloud(sync_log_ids: set[str] | None = None) -> int:
    """Query W&B API and verify all synced offline runs are visible in cloud.

    Lists every run across every project for the configured entity, classifies
    them into (a) runs from the current sync session and (b) prior-session
    legitimate runs, and prints a per-project breakdown plus reconciliation.
    """
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed", file=sys.stderr)
        return 1

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY", "phl690-harvard-extension-schol")

    print("\n" + "=" * 78)
    print(" CLOUD VERIFICATION (W&B API)")
    print("=" * 78)
    print(f"  Entity: {entity}\n")

    try:
        projects = list(api.projects(entity=entity))
    except Exception as e:
        print(f"  ERROR querying projects: {e}", file=sys.stderr)
        return 1

    print(f"  {'PROJECT':<48}  {'COUNT':>6}")
    print("  " + "-" * 56)

    all_runs: list[tuple[str, str, str, str, str]] = []
    total = 0
    for proj in projects:
        try:
            runs = list(api.runs(f"{entity}/{proj.name}"))
        except Exception:
            runs = []
        print(f"  {proj.name[:47]:<48}  {len(runs):>6}")
        total += len(runs)
        for r in runs:
            all_runs.append((proj.name, r.id, r.name or "-",
                             str(r.created_at), r.job_type or "-"))
    print("  " + "-" * 56)
    print(f"  {'TOTAL':<48}  {total:>6}")
    print(f"\n  Total cloud runs: {total}")

    # If a sync log was provided, reconcile
    if sync_log_ids is not None:
        cloud_ids = {rid for _, rid, *_ in all_runs}
        missing = sync_log_ids - cloud_ids
        extras = cloud_ids - sync_log_ids
        print("\n" + "-" * 78)
        print(f"  Sync-log runs:                                {len(sync_log_ids):>4}")
        print(f"  Verified in cloud:                            {len(sync_log_ids) - len(missing):>4}")
        print(f"  Missing in cloud (synced locally, not seen):  {len(missing):>4}")
        for m in missing:
            print(f"    - {m}")
        print(f"  Prior-session runs (legitimate, pre-sync):    {len(extras):>4}")
        if extras:
            print("\n  Prior-session runs (sample, max 20):")
            extra_runs = sorted(
                [r for r in all_runs if r[1] in extras],
                key=lambda x: x[3],
            )
            print(f"    {'PROJECT':<25} {'RUN_ID':<12} {'CREATED':<22} {'JOB_TYPE'}")
            for proj, rid, _, created, jt in extra_runs[:20]:
                print(f"    {proj[:24]:<25} {rid:<12} {created:<22} {jt}")
            if len(extras) > 20:
                print(f"    ... and {len(extras) - 20} more")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Run diagnostics only; no sync, no clean.")
    parser.add_argument("--sync", action="store_true",
                        help="Sync all pending offline runs to W&B cloud.")
    parser.add_argument("--clean", action="store_true",
                        help="Delete local synced run directories after sync.")
    parser.add_argument("--verify-cloud", action="store_true",
                        help="Query W&B API and list all runs visible in cloud.")
    parser.add_argument("--sync-log", default=None,
                        help="Path to text file with one run-id per line; reconciles against cloud.")
    parser.add_argument("--wandb-dir", default="wandb",
                        help="Path to W&B working directory (default: wandb).")
    args = parser.parse_args()

    wandb_dir = Path(args.wandb_dir)

    print("=" * 78)
    print(" OFFLINE-RUN QUEUE INVENTORY")
    print("=" * 78)
    inv = inventory_offline_runs(wandb_dir)
    print(f"  Pending offline runs : {inv['count']}")
    print(f"  Total size           : {inv['total_bytes'] / 1024**2:.2f} MB "
          f"({inv['total_bytes']:,} bytes)")
    if inv["count"]:
        print(f"  Earliest             : {inv['earliest']}")
        print(f"  Latest               : {inv['latest']}")

    print("\n" + "=" * 78)
    print(" NETWORK REACHABILITY")
    print("=" * 78)
    ok, msg = check_network()
    print(f"  {msg}")
    if not ok:
        print("  --> Industry best practice: sync from a login/edge node with internet,")
        print("      or set up SSH tunnel/proxy from this compute node.")

    print("\n" + "=" * 78)
    print(" AUTHENTICATION")
    print("=" * 78)
    auth = check_auth()
    key_state = (f"set ({auth['wandb_api_key_len']} chars)"
                 if auth["wandb_api_key_set"] else "unset")
    netrc_state = ("exists, contains api.wandb.ai" if auth["netrc_has_wandb"]
                   else ("exists, no W&B entry" if auth["netrc_exists"]
                         else "missing"))
    print(f"  WANDB_API_KEY env : {key_state}")
    print(f"  ~/.netrc          : {netrc_state}")
    print(f"  WANDB_MODE        : {auth['wandb_mode']}")

    if args.verify_cloud:
        sync_log_ids: set[str] | None = None
        if args.sync_log:
            sync_log_path = Path(args.sync_log)
            if sync_log_path.is_file():
                sync_log_ids = {
                    line.strip() for line in sync_log_path.read_text().splitlines()
                    if line.strip() and not line.startswith("#")
                }
        rc = verify_cloud(sync_log_ids=sync_log_ids)
        if not args.sync and not args.clean:
            return rc

    if args.check_only or (not args.sync and not args.clean and not args.verify_cloud):
        print("\n[check-only] no mutations performed.")
        if not args.check_only:
            print("Re-run with --sync to upload pending runs, or --clean to delete synced dirs.")
        return 0

    if not ok:
        print("\nERROR: cannot proceed with sync — network unreachable.", file=sys.stderr)
        return 1
    if not (auth["wandb_api_key_set"] or auth["netrc_has_wandb"]):
        print("\nERROR: no W&B credentials found.", file=sys.stderr)
        print("  Set WANDB_API_KEY or run `wandb login` first.", file=sys.stderr)
        return 1

    rc = 0
    if args.sync:
        rc |= run_sync_all()
        show_pending()  # confirm queue empty
    if args.clean:
        rc |= run_clean()

    print("\n" + "=" * 78)
    print(" POST-RUN INVENTORY")
    print("=" * 78)
    inv2 = inventory_offline_runs(wandb_dir)
    print(f"  Remaining offline runs : {inv2['count']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
