# scripts/merge_rag_generations.py
"""Merge per-rank RAG generation shards into a single results file + summary.

Called by scripts/rag_generate_multigpu.sbatch after all 4 query-shard
workers complete. Replaces inline `python -c` heredocs with a testable,
reusable module mirroring scripts/merge_reranker_shards.py.

Usage
-----
    .venv/bin/python scripts/merge_rag_generations.py \
        --ablation reranker \
        --out-root data/processed/rag \
        --world-size 4 \
        --seed 0

What it does
------------
1. Locate `<out_root>/<ablation_label>/generations.rank{NNN}.jsonl` shards
2. Verify shard count matches --world-size
3. Concatenate shards in rank order
4. Cross-check merged row count == sum of per-rank n_generated
5. Aggregate per-rank summaries: max-of-rank wall-times, sum tokens
6. Write merged `generations.jsonl` + `generation_summary.json`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

# Ensure repo root on sys.path so `scripts.rag_generate` resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.rag_generate import ABLATION_CONFIGS, GENERATOR_MODEL, SCHEMA_VERSION, _git_sha


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("merge_rag_generations")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[merge_rag_generations] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


def _concat_shards_in_order(shard_paths: list[Path], merged_path: Path) -> None:
    """Concatenate per-rank JSONL shards in rank order. Skips blank lines."""
    Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as fout:
        for shard in sorted(shard_paths):
            if not Path(shard).exists():
                continue
            with shard.open(encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)


def merge(
    *,
    ablation: str,
    out_root: Path,
    world_size: int,
    seed: int = 0,
) -> dict:
    if ablation not in ABLATION_CONFIGS:
        raise KeyError(
            f"unknown ablation {ablation!r}; valid: {sorted(ABLATION_CONFIGS.keys())}"
        )
    cfg = ABLATION_CONFIGS[ablation]
    label = cfg["label"]
    ablation_dir = Path(out_root) / label
    if not ablation_dir.is_dir():
        raise FileNotFoundError(f"ablation dir does not exist: {ablation_dir}")

    shard_paths = sorted(ablation_dir.glob("generations.rank*.jsonl"))
    if len(shard_paths) != world_size:
        raise RuntimeError(
            f"expected {world_size} generations.rank*.jsonl shards in "
            f"{ablation_dir}, got {len(shard_paths)}: {[p.name for p in shard_paths]}"
        )
    rank_summary_paths = sorted(ablation_dir.glob("generation_summary.rank*.json"))
    if len(rank_summary_paths) != world_size:
        raise RuntimeError(
            f"expected {world_size} generation_summary.rank*.json files in "
            f"{ablation_dir}, got {len(rank_summary_paths)}"
        )

    merged_results = ablation_dir / "generations.jsonl"
    merged_summary_path = ablation_dir / "generation_summary.json"

    logger.info(
        f"merging {len(shard_paths)} shards (ablation={ablation}) -> {merged_results}"
    )
    _concat_shards_in_order(shard_paths, merged_results)

    # Aggregate summaries
    rank_summaries = [json.loads(p.read_text()) for p in rank_summary_paths]
    rank0 = rank_summaries[0]
    n_queries_total = int(rank0["n_queries_total"])
    n_generated_total = sum(int(s["n_generated"]) for s in rank_summaries)
    n_total_tokens_out = sum(int(s["n_total_tokens_out"]) for s in rank_summaries)
    n_emitted = sum(1 for _ in merged_results.open(encoding="utf-8"))
    if n_emitted != n_generated_total:
        raise RuntimeError(
            f"merged row count {n_emitted:,} != sum n_generated "
            f"{n_generated_total:,}"
        )

    results_hash = hashlib.sha256(merged_results.read_bytes()).hexdigest()

    # Worst-case (max) per-rank wall-times
    encoder_load_seconds = max(float(s["encoder_load_seconds"]) for s in rank_summaries)
    text_index_seconds = max(float(s["text_index_seconds"]) for s in rank_summaries)
    generation_seconds = max(float(s["generation_seconds"]) for s in rank_summaries)

    merged_summary = {
        "schema_version": SCHEMA_VERSION,
        "ablation": ablation,
        "ablation_label": label,
        "n_queries_total": n_queries_total,
        "n_generated_total": n_generated_total,
        "n_total_tokens_out": n_total_tokens_out,
        "top_k_context": int(rank0["top_k_context"]),
        "max_new_tokens": int(rank0["max_new_tokens"]),
        "batch_size": int(rank0["batch_size"]),
        "max_length": int(rank0["max_length"]),
        "max_chunks_per_cluster": int(rank0["max_chunks_per_cluster"]),
        "generator_model": GENERATOR_MODEL,
        "device": rank0["device"],
        "device_name": rank0["device_name"],
        "world_size": world_size,
        "encoder_load_seconds": encoder_load_seconds,
        "text_index_seconds": text_index_seconds,
        "generation_seconds": generation_seconds,
        "seed": seed,
        "git_sha": _git_sha(),
        "results_hash": results_hash,
    }
    merged_summary_path.write_text(
        json.dumps(merged_summary, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    logger.info(
        f"merged: n_queries_total={n_queries_total:,}  "
        f"n_generated={n_generated_total:,}  tokens={n_total_tokens_out:,}  "
        f"hash={results_hash[:16]}"
    )
    logger.info(f"wrote {merged_results}")
    logger.info(f"wrote {merged_summary_path}")
    return merged_summary


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Merge per-rank RAG generation shards into unified results + summary.",
    )
    ap.add_argument("--ablation", type=str, required=True,
                    choices=sorted(ABLATION_CONFIGS.keys()))
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    merge(
        ablation=args.ablation,
        out_root=args.out_root,
        world_size=args.world_size,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
