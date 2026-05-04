# scripts/aggregate_judgments.py
"""Aggregate judgments.jsonl into judgment_summary.json per ablation.

Used to recover summary files when judging completed but summary write was
interrupted. Idempotent: re-running overwrites with current state.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.hallucination_judge import (
    SCHEMA_VERSION,
    _aggregate_scores,
    _git_sha,
)

LABEL_TO_ABLATION = {
    "no_rag": "none",
    "bm25_rag": "bm25",
    "bge_m3_rag": "bge_m3",
    "rrf_rag": "rrf",
    "reranker_rag": "reranker",
}


def aggregate_one(
    *,
    judgments_path: Path,
    summary_path: Path,
    ablation: str,
    ablation_label: str,
    judge_model: str = "gpt-4o-mini",
    limit: int | None = None,
    top_k_context: int = 5,
    max_chunks_per_cluster: int = 2,
) -> dict[str, Any]:
    judgments_path = Path(judgments_path)
    scores: list[dict[str, Any]] = []
    with judgments_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scores.append({"label": json.loads(line)["label"]})
    agg = _aggregate_scores(scores)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "ablation": ablation,
        "ablation_label": ablation_label,
        "judge_model": judge_model,
        "n_total": agg["n_total"],
        "n_judged": agg["n_judged"],
        "n_unknown": agg["n_unknown"],
        "faithful_rate": agg["faithful_rate"],
        "partial_rate": agg["partial_rate"],
        "hallucinated_rate": agg["hallucinated_rate"],
        "judging_seconds": 0.0,
        "limit": limit,
        "top_k_context": top_k_context,
        "max_chunks_per_cluster": max_chunks_per_cluster,
        "git_sha": _git_sha(),
        "results_hash": hashlib.sha256(judgments_path.read_bytes()).hexdigest(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Aggregate judgments.jsonl into judgment_summary.json per ablation.",
    )
    ap.add_argument("--judge-root", type=Path,
                    default=Path("data/processed/hallucination"))
    ap.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    judge_root = args.judge_root
    if not judge_root.is_dir():
        print(f"FAIL: {judge_root} not a directory")
        return 2

    print(f"=== aggregating judgments in {judge_root} ===")
    print(
        f"  {'Ablation':<14} {'n_total':>8} {'Faithful':>10} {'Partial':>10} {'Halluc.':>10}"
    )
    for label, ablation in LABEL_TO_ABLATION.items():
        jp = judge_root / label / "judgments.jsonl"
        sp = judge_root / label / "judgment_summary.json"
        if not jp.is_file():
            continue
        s = aggregate_one(
            judgments_path=jp,
            summary_path=sp,
            ablation=ablation,
            ablation_label=label,
            judge_model=args.judge_model,
            limit=args.limit,
        )
        print(
            f"  {label:<14} {s['n_total']:>8,} {s['faithful_rate']:>10.4f} "
            f"{s['partial_rate']:>10.4f} {s['hallucinated_rate']:>10.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
