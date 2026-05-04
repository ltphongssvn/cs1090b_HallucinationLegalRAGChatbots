# scripts/run_hallucination_judge_parallel.py
"""Parallel launcher for hallucination judging across all RAG ablations.

Runs N ablations concurrently (each its own subprocess), each writing to
its own judgments.jsonl. Resume-safe: re-running picks up where killed.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ABLATIONS = ["none", "bm25", "bge_m3", "rrf", "reranker"]
DEFAULT_MAX_PARALLEL = 4
RETRIEVAL_DIR_FOR_ABLATION = {
    "none":     "data/processed/baseline/cleaned",
    "bm25":     "data/processed/baseline/cleaned",
    "bge_m3":   "data/processed/baseline/cleaned",
    "rrf":      "data/processed/baseline/cleaned",
    "reranker": "data/processed/baseline/cleaned/finetuned",
}
LABEL_FOR_ABLATION = {
    "none": "no_rag", "bm25": "bm25_rag", "bge_m3": "bge_m3_rag",
    "rrf": "rrf_rag", "reranker": "reranker_rag",
}


def _build_arg_parser():
    ap = argparse.ArgumentParser(description="Parallel hallucination judge launcher.")
    ap.add_argument("--ablations", nargs="+", default=DEFAULT_ABLATIONS)
    ap.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    ap.add_argument("--rag-root", default="data/processed/rag")
    ap.add_argument("--gold-path", default="data/processed/baseline/cleaned/gold_pairs_test.jsonl")
    ap.add_argument("--corpus-path", default="data/processed/baseline/corpus_chunks_cleaned.jsonl")
    ap.add_argument("--out-root", default="data/processed/hallucination")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=None)
    return ap


def _source_env(env: dict) -> dict:
    if env.get("OPENAI_API_KEY"):
        return env
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return env
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("export OPENAI_API_KEY="):
            env["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip("\"\'")
    return env


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    env = _source_env(env)
    if not env.get("OPENAI_API_KEY"):
        print("FAIL: OPENAI_API_KEY not in env or .env")
        return 2

    # Build job specs
    jobs: list[tuple[str, list[str], Path]] = []
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    for ablation in args.ablations:
        label = LABEL_FOR_ABLATION[ablation]
        gen_path = Path(args.rag_root) / label / "generations.jsonl"
        if not gen_path.is_file():
            print(f"[parallel] SKIP {ablation}: missing {gen_path}")
            continue
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "hallucination_judge.py"),
            "--ablation", ablation,
            "--rag-root", args.rag_root,
            "--gold-path", args.gold_path,
            "--retrieval-dir", RETRIEVAL_DIR_FOR_ABLATION[ablation],
            "--corpus-path", args.corpus_path,
            "--out-root", args.out_root,
            "--judge-model", args.judge_model,
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        log_path = log_dir / f"hallucination_judge_{ablation}.log"
        jobs.append((ablation, cmd, log_path))

    # Run with bounded parallelism
    running: list[tuple[str, subprocess.Popen, Path]] = []
    pending = list(jobs)
    rc_total = 0

    # Fill initial batch up to max_parallel concurrently
    while pending and len(running) < args.max_parallel:
        ablation, cmd, log_path = pending.pop(0)
        log_fh = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        running.append((ablation, proc, log_path))
        print(f"[parallel] STARTED {ablation} (pid={proc.pid}) -> {log_path}", flush=True)

    # Wait for completion; start new ones as slots free up
    while running:
        completed_idx = None
        for i, (ablation, proc, log_path) in enumerate(running):
            rc = proc.poll()
            if rc is not None:
                print(f"[parallel] DONE {ablation} rc={rc}", flush=True)
                if rc != 0:
                    rc_total = 1
                completed_idx = i
                break
        if completed_idx is not None:
            running.pop(completed_idx)
            if pending:
                ablation, cmd, log_path = pending.pop(0)
                log_fh = log_path.open("w", encoding="utf-8")
                proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
                running.append((ablation, proc, log_path))
                print(f"[parallel] STARTED {ablation} (pid={proc.pid}) -> {log_path}", flush=True)
        else:
            time.sleep(10)

    return rc_total


if __name__ == "__main__":
    sys.exit(main())
