"""Cell 25 logic: W&B reproducibility audit (config + lineage + queue + tests).

Extracted from the notebook so the cell itself contains only orchestration.
Entry point: :func:`run`. Returns a dict of structured diagnostics; the
notebook prints them in `cell_timer` context.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(".").resolve()
EXCLUDE_PARTS = {".venv", ".git", ".ipynb_checkpoints", "__pycache__"}

HELPER_PATTERNS = {
    "get_git_sha": r"\bget_git_sha\s*\(",
    "get_run_group": r"\bget_run_group\s*\(",
    "get_wandb_entity": r"\bget_wandb_entity\s*\(",
    "get_wandb_project": r"\bget_wandb_project\s*\(",
    "link_input_artifacts": r"\blink_input_artifacts\s*\(",
}

PIPELINE_STAGES = [
    ("scripts/baseline_prep.py", "baseline-prep", [], ["baseline-prep"]),
    ("scripts/baseline_bm25.py", "baseline-bm25", ["baseline-prep"], ["baseline-bm25"]),
    ("scripts/baseline_bge_m3.py", "baseline-bge-m3", ["baseline-prep"], ["baseline-bge-m3"]),
    ("scripts/baseline_rrf.py", "rrf", ["baseline-bm25", "baseline-bge-m3"], ["baseline-rrf"]),
    ("scripts/baseline_reranker.py", "reranker", ["baseline-rrf"], ["baseline-reranker"]),
    (
        "scripts/rag_generate.py",
        "rag",
        ["baseline-bm25", "baseline-bge-m3", "baseline-rrf", "baseline-reranker"],
        ["rag-generations"],
    ),
    ("scripts/hallucination_judge.py", "hallucination-judge", ["rag-generations"], ["hallucination-judgments"]),
    ("scripts/stratified_eval.py", "stratified-eval", ["baseline-<retriever>"], []),
]

TEST_FILES = [
    "tests/test_wandb_reproducibility.py",
    "tests/test_wandb_reproducibility_round2.py",
    "tests/test_wandb_lineage.py",
    "tests/test_wandb_sync_diagnostics.py",
]


def _py_files(root: str) -> list[Path]:
    out = []
    for p in (REPO / root).rglob("*.py"):
        if any(part in EXCLUDE_PARTS for part in p.parts):
            continue
        out.append(p)
    return out


def collect_wandb_config() -> dict[str, Any]:
    from src.repro import get_git_sha, get_run_group, get_wandb_entity, get_wandb_project

    return {
        "WANDB_ENTITY": (os.environ.get("WANDB_ENTITY", ""), get_wandb_entity("phl690-harvard-extension-schol")),
        "WANDB_PROJECT": (os.environ.get("WANDB_PROJECT", ""), get_wandb_project("cs1090b")),
        "WANDB_MODE": (os.environ.get("WANDB_MODE", ""), os.environ.get("WANDB_MODE", "online")),
        "WANDB_RUN_GROUP": (os.environ.get("WANDB_RUN_GROUP", ""), get_run_group("ms4-baselines")),
        "git_sha_short": get_git_sha(short=True),
        "git_sha_full": get_git_sha(short=False),
    }


def collect_helper_counts() -> dict[str, list[str]]:
    counts: dict[str, list[str]] = {k: [] for k in HELPER_PATTERNS}
    for f in _py_files("src") + _py_files("scripts"):
        text = f.read_text(errors="ignore")
        for name, pat in HELPER_PATTERNS.items():
            if re.search(pat, text):
                counts[name].append(f.relative_to(REPO).as_posix())
    return counts


def collect_lineage() -> list[dict[str, Any]]:
    rows = []
    for rel, job_type, inputs, outputs in PIPELINE_STAGES:
        p = REPO / rel
        if not p.is_file():
            rows.append({"script": rel, "job_type": job_type, "found": False, "inputs": inputs, "outputs": outputs})
            continue
        text = p.read_text(errors="ignore")
        rows.append(
            {
                "script": rel,
                "job_type": job_type,
                "found": True,
                "has_init": "wandb.init(" in text,
                "has_seed": "seed" in text,
                "has_group": "group=" in text or "WANDB_RUN_GROUP" in text,
                "has_use": "use_artifact" in text or "link_input_artifacts" in text,
                "has_log": "log_artifact" in text,
                "inputs": inputs,
                "outputs": outputs,
            }
        )
    return rows


def collect_queue(skip_network: bool = False) -> dict[str, Any]:
    wandb_dir = REPO / "wandb"
    runs: list[Path] = []
    total_bytes = 0
    if wandb_dir.is_dir():
        runs = sorted(wandb_dir.glob("offline-run-*"))
        for r in runs:
            for f in r.rglob("*"):
                if f.is_file():
                    total_bytes += f.stat().st_size
    net_ok, net_msg = False, "(skipped)"
    if not skip_network:
        try:
            socket.create_connection(("api.wandb.ai", 443), timeout=5)
            net_ok, net_msg = True, "✓ REACHABLE: api.wandb.ai:443"
        except Exception as e:
            net_ok, net_msg = False, f"✗ BLOCKED: {e}"
    key_env = os.environ.get("WANDB_API_KEY", "")
    netrc = Path.home() / ".netrc"
    netrc_has_wandb = netrc.exists() and "api.wandb.ai" in netrc.read_text(errors="ignore")
    return {
        "count": len(runs),
        "total_bytes": total_bytes,
        "earliest": runs[0].name if runs else None,
        "latest": runs[-1].name if runs else None,
        "net_ok": net_ok,
        "net_msg": net_msg,
        "auth_ok": bool(key_env) or netrc_has_wandb,
        "auth_src": "WANDB_API_KEY env" if key_env else ("~/.netrc" if netrc_has_wandb else "(none)"),
    }


def run_test_suite() -> tuple[int, list[str]]:
    proc = subprocess.run(
        [".venv/bin/pytest", *TEST_FILES, "--no-header", "-q", "--tb=no", "--color=no"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip().splitlines()


def render(cfg: dict, helpers: dict, lineage: list, queue: dict, rc: int, test_lines: list[str]) -> None:
    """Print all sections in canonical format."""
    print("=" * 100)
    print(" W&B CONFIG RESOLUTION (env var → src.repro default)")
    print("=" * 100)
    print(f"\n  {'KEY':<18}  {'ENV VALUE':<35}  RESOLVED")
    print("  " + "-" * 97)
    for key in ["WANDB_ENTITY", "WANDB_PROJECT", "WANDB_MODE", "WANDB_RUN_GROUP"]:
        env_val, resolved_val = cfg[key]
        print(f"  {key:<18}  {(env_val or '(unset)'):<35}  {resolved_val}")
    print(f"\n  git_sha (short-12)            : {cfg['git_sha_short']}")
    print(f"  git_sha (full-40)             : {cfg['git_sha_full']}")

    print("\n" + "=" * 100)
    print(" REPRODUCIBILITY HELPERS — call sites across src/ + scripts/")
    print("=" * 100)
    print(f"\n  {'HELPER':<22}  CALL SITES")
    print("  " + "-" * 97)
    for name, sites in helpers.items():
        print(f"  {name:<22}  {len(sites):>3}  {', '.join(sorted(sites)[:3])}" + (" ..." if len(sites) > 3 else ""))

    print("\n" + "=" * 100)
    print(" PIPELINE STAGE LINEAGE MAP")
    print("=" * 100)
    print(f"\n  {'SCRIPT':<36}  {'JOB_TYPE':<22}  WIRED")
    print("  " + "-" * 97)
    for r in lineage:
        if not r["found"]:
            print(f"  {r['script']:<36}  {r['job_type']:<22}  (file not found)")
            continue
        flags = (
            ("init" if r["has_init"] else "----")
            + " "
            + ("seed" if r["has_seed"] else "----")
            + " "
            + ("grp" if r["has_group"] else "---")
            + " "
            + ("use" if r["has_use"] else "---")
            + " "
            + ("log" if r["has_log"] else "---")
        )
        print(f"  {r['script']:<36}  {r['job_type']:<22}  {flags}")
        for inp in r["inputs"]:
            print(f"  {'':<36}  {'':<22}    ← use_artifact({inp}:latest)")
        for out in r["outputs"]:
            print(f"  {'':<36}  {'':<22}    → log_artifact({out})")

    print("\n" + "=" * 100)
    print(" OFFLINE RUN QUEUE + SYNC READINESS")
    print("=" * 100)
    print(f"\n  Pending offline runs : {queue['count']}")
    print(f"  Total size           : {queue['total_bytes'] / 1024**2:.2f} MB ({queue['total_bytes']:,} bytes)")
    if queue["count"]:
        print(f"  Earliest             : {queue['earliest']}")
        print(f"  Latest               : {queue['latest']}")
    else:
        print("  Status               : ✓ queue empty — all runs synced to W&B cloud")
    print(f"\n  Network              : {queue['net_msg']}")
    print(f"  Authentication       : {'✓' if queue['auth_ok'] else '✗'} {queue['auth_src']}")
    if queue["count"]:
        print("\n  To sync queued runs (from a node with internet):")
        print("    .venv/bin/python scripts/wandb_sync_diagnostics.py --sync --clean")

    print("\n" + "=" * 100)
    print(" REPRODUCIBILITY TEST SUITE")
    print("=" * 100)
    for line in test_lines[-12:]:
        print(f"  {line}")
    print(f"\n  rc = {rc}  ({'PASS' if rc == 0 else 'FAIL'})")
    print("\n  CLI equivalent:")
    print("    pytest tests/test_wandb_reproducibility.py \\")
    print("           tests/test_wandb_reproducibility_round2.py \\")
    print("           tests/test_wandb_lineage.py \\")
    print("           tests/test_wandb_sync_diagnostics.py -v")


def run(skip_pytest: bool = False, skip_network: bool = False) -> dict[str, Any]:
    cfg = collect_wandb_config()
    helpers = collect_helper_counts()
    lineage = collect_lineage()
    queue = collect_queue(skip_network=skip_network)
    if skip_pytest:
        rc, test_lines = 0, ["(pytest skipped)"]
    else:
        rc, test_lines = run_test_suite()
    return {
        "wandb_config": cfg,
        "helper_counts": helpers,
        "lineage": lineage,
        "queue": queue,
        "rc": rc,
        "test_lines": test_lines,
    }
