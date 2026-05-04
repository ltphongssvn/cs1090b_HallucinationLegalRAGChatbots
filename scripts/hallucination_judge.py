# scripts/hallucination_judge.py
"""MS5 LLM-as-judge for citation faithfulness in RAG generations.

For each (question, generated_answer, retrieved_contexts) triple produced by
scripts/rag_generate.py, asks an external judge model whether the answer is
FAITHFUL, PARTIAL, or HALLUCINATED with respect to the contexts. Aggregates
into a per-ablation faithfulness rate that's the central MS5 deliverable.

Why an API judge (not local)
----------------------------
FaithBench (arxiv.org/abs/2505.04847) shows o3-mini-high reaches 84% balanced
accuracy on faithfulness judgment, beating ensemble of local 70B models.
Judge calls are ~20K total per ablation × 4 ablations = 80K calls — at
gpt-4o-mini's pricing this is ~$5-15 total, vs hundreds of GPU-hours
for an equivalently-strong local judge.

Score labels (FaithBench-aligned)
---------------------------------
FAITHFUL     : every claim in the answer is supported by the contexts
PARTIAL      : some claims supported, others unsupported
HALLUCINATED : at least one claim contradicts or fabricates beyond contexts
UNKNOWN      : judge response could not be parsed (rare; recovered via retry)

Outputs (data/processed/hallucination/<ablation>/)
--------------------------------------------------
judgments.jsonl       — per-query judge label + raw response
judgment_summary.json — aggregate faithfulness/partial/hallucinated rates
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Ensure repo root on sys.path so `scripts.rag_generate` resolves regardless
# of how this script is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = "1.0.0"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_BATCH_SIZE = 8        # concurrent API requests
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SEC = 60
DEFAULT_TEMPERATURE = 0.0     # deterministic verdicts

SCORE_LABELS = ("FAITHFUL", "PARTIAL", "HALLUCINATED")
_VALID_LABELS = set(SCORE_LABELS) | {"UNKNOWN"}

DEFAULT_GOLD = Path("data/processed/baseline/cleaned/gold_pairs_test.jsonl")
DEFAULT_RAG_ROOT = Path("data/processed/rag")
DEFAULT_RETRIEVAL_DIR = Path("data/processed/baseline/cleaned")
DEFAULT_CORPUS = Path("data/processed/baseline/corpus_chunks_cleaned.jsonl")
DEFAULT_OUT_ROOT = Path("data/processed/hallucination")


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("hallucination_judge")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[hallucination_judge] %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg


logger = _get_logger()


# ---------- I/O ----------


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------- prompt construction ----------


JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of citation faithfulness in legal-RAG outputs. "
    "Given a legal question, a candidate answer, and the retrieved court-opinion "
    "contexts the answer was supposed to be based on, decide whether the answer "
    "is supported by the contexts. Respond with exactly one verdict word from "
    "{FAITHFUL, PARTIAL, HALLUCINATED} on the first line, followed by a brief "
    "one-sentence reason on the second line.\n\n"
    "  FAITHFUL     = every factual claim in the answer is supported by the contexts.\n"
    "  PARTIAL      = some claims are supported, others are not.\n"
    "  HALLUCINATED = at least one claim contradicts the contexts or is fabricated.\n"
)


def _build_judge_prompt(
    *,
    question: str,
    generation: str,
    contexts: list[str],
) -> str:
    """Build the judge user message. Contexts may be empty (no-RAG ablation)."""
    if contexts:
        ctx_block = "\n\n".join(
            f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )
    else:
        ctx_block = "[No contexts were provided to the answerer.]"
    return (
        f"QUESTION:\n{question}\n\n"
        f"CANDIDATE ANSWER:\n{generation}\n\n"
        f"RETRIEVED CONTEXTS:\n{ctx_block}\n\n"
        "Verdict (FAITHFUL / PARTIAL / HALLUCINATED) and reason:"
    )


# ---------- response parsing ----------


def _parse_judge_response(raw: str) -> dict[str, Any]:
    """Extract a SCORE_LABEL from the judge's free-form text.

    Strategy: look for the first SCORE_LABEL token. Order matters since
    HALLUCINATED contains 'PARTIAL' as substring? No — checking with
    word-level scan to avoid that. Returns {label, raw}.
    """
    if not raw or not raw.strip():
        return {"label": "UNKNOWN", "raw": raw}
    upper = raw.upper()
    # Check explicit labels in priority order: PARTIAL/FAITHFUL/HALLUCINATED
    # as standalone tokens (word-boundary). We accept the FIRST one that appears.
    import re
    matches: list[tuple[int, str]] = []
    for label in SCORE_LABELS:
        m = re.search(rf"\b{label}\b", upper)
        if m:
            matches.append((m.start(), label))
    if not matches:
        return {"label": "UNKNOWN", "raw": raw}
    matches.sort(key=lambda t: t[0])
    return {"label": matches[0][1], "raw": raw}


# ---------- aggregate scoring ----------


def _aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute faithfulness rates over a list of {label} dicts.

    Rates are computed over judged-only (excluding UNKNOWN).
    """
    n_total = len(scores)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_judged": 0,
            "n_unknown": 0,
            "faithful_rate": 0.0,
            "partial_rate": 0.0,
            "hallucinated_rate": 0.0,
        }
    counts = {label: 0 for label in SCORE_LABELS}
    n_unknown = 0
    for s in scores:
        label = s.get("label", "UNKNOWN")
        if label in counts:
            counts[label] += 1
        else:
            n_unknown += 1
    n_judged = n_total - n_unknown
    if n_judged == 0:
        return {
            "n_total": n_total,
            "n_judged": 0,
            "n_unknown": n_unknown,
            "faithful_rate": 0.0,
            "partial_rate": 0.0,
            "hallucinated_rate": 0.0,
        }
    return {
        "n_total": n_total,
        "n_judged": n_judged,
        "n_unknown": n_unknown,
        "faithful_rate": counts["FAITHFUL"] / n_judged,
        "partial_rate": counts["PARTIAL"] / n_judged,
        "hallucinated_rate": counts["HALLUCINATED"] / n_judged,
    }


# ---------- judge API call ----------


def _call_judge_with_retries(
    *,
    client: Any,
    model: str,
    user_msg: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> str:
    """Call the judge API with simple retry; return raw response text or empty."""
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=DEFAULT_TEMPERATURE,
                timeout=timeout_sec,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(min(2**attempt, 10))
    logger.info(f"  judge call failed after {max_retries} retries: {last_err}")
    return ""


# ---------- provenance ----------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


# ---------- main ----------


def main(
    *,
    ablation: str,
    rag_root: Path = DEFAULT_RAG_ROOT,
    gold_path: Path = DEFAULT_GOLD,
    retrieval_dir: Path = DEFAULT_RETRIEVAL_DIR,
    corpus_path: Path = DEFAULT_CORPUS,
    out_root: Path = DEFAULT_OUT_ROOT,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    top_k_context: int = 5,
    max_chunks_per_cluster: int = 2,
    limit: int | None = None,
) -> dict[str, Any]:
    """Judge faithfulness for one ablation's generations against retrieved contexts."""
    # Lazy import OpenAI client so unit tests don't require the package
    from openai import OpenAI  # type: ignore[import-not-found]

    # Reuse the rag_generate ablation registry + cluster text loader for
    # consistent context reconstruction (judge sees same contexts as generator)
    from scripts.rag_generate import (
        ABLATION_CONFIGS, _load_cluster_text_index, _load_queries,
    )
    if ablation not in ABLATION_CONFIGS:
        raise KeyError(
            f"unknown ablation {ablation!r}; valid: {sorted(ABLATION_CONFIGS.keys())}"
        )
    cfg = ABLATION_CONFIGS[ablation]
    rag_dir = Path(rag_root) / cfg["label"]
    gen_path = rag_dir / "generations.jsonl"
    if not gen_path.is_file():
        raise FileNotFoundError(f"generations missing: {gen_path}")

    out_dir = Path(out_root) / cfg["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = out_dir / "judgments.jsonl"
    summary_path = out_dir / "judgment_summary.json"

    logger.info("=" * 60)
    logger.info(f"MS5 hallucination judging  ablation={ablation}  judge={judge_model}")
    logger.info("=" * 60)

    # Align queries / generations / retrieval rows
    queries = _load_queries(Path(gold_path))
    gens = list(_iter_jsonl(gen_path))
    if len(queries) != len(gens):
        raise ValueError(
            f"queries ({len(queries):,}) and generations ({len(gens):,}) "
            f"have different row counts"
        )

    retrieval_rows: list[dict[str, Any]] = []
    cluster_text: dict[int, str] = {}
    if cfg["results_filename"] is not None:
        results_path = Path(retrieval_dir) / cfg["results_filename"]
        retrieval_rows = list(_iter_jsonl(results_path))
        if len(retrieval_rows) != len(queries):
            raise ValueError("retrieval-results row count mismatch")
        # Build cluster text index limited to candidates we'll show the judge
        needed: set[int] = set()
        for r in retrieval_rows:
            for hit in r["retrieved"][:top_k_context]:
                needed.add(int(hit["cluster_id"]))
        cluster_text = _load_cluster_text_index(
            Path(corpus_path),
            max_chunks_per_cluster=max_chunks_per_cluster,
            cluster_filter=needed,
        )

    n_to_judge = limit if limit is not None else len(queries)

    # Resume: load any existing judgments for this ablation
    already_judged: set[tuple[int, int]] = set()
    scores: list[dict[str, Any]] = []
    if judgments_path.is_file():
        with judgments_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                already_judged.add((int(row["source_id"]), int(row["dest_id"])))
                scores.append({"label": row["label"]})
        logger.info(
            f"  RESUMING: {len(already_judged):,} judgments already in "
            f"{judgments_path.name}, will skip those"
        )

    logger.info(f"  judging up to {n_to_judge:,} of {len(queries):,} (limit={limit})")

    client = OpenAI()
    log_interval = max(1, n_to_judge // 50)

    t0 = time.perf_counter()
    # Append mode (preserves prior judgments on resume)
    with judgments_path.open("a", encoding="utf-8", buffering=1) as fout:
        n_skipped_resume = 0
        for i in range(n_to_judge):
            q = queries[i]
            g = gens[i]
            if (q["source_id"], q["dest_id"]) != (g["source_id"], g["dest_id"]):
                raise ValueError(
                    f"alignment mismatch at row {i}: query={q['source_id'],q['dest_id']} "
                    f"vs gen={g['source_id'],g['dest_id']}"
                )
            if (int(q["source_id"]), int(q["dest_id"])) in already_judged:
                n_skipped_resume += 1
                continue
            contexts: list[str] = []
            if retrieval_rows:
                rr = retrieval_rows[i]
                for hit in rr["retrieved"][:top_k_context]:
                    cid = int(hit["cluster_id"])
                    txt = cluster_text.get(cid)
                    if txt:
                        contexts.append(txt)
            user_msg = _build_judge_prompt(
                question=q["query_text"],
                generation=g["generation"],
                contexts=contexts,
            )
            raw = _call_judge_with_retries(
                client=client,
                model=judge_model,
                user_msg=user_msg,
                max_retries=max_retries,
            )
            parsed = _parse_judge_response(raw)
            row = {
                "source_id": q["source_id"],
                "dest_id": q["dest_id"],
                "source_cluster_id": q["source_cluster_id"],
                "ablation": ablation,
                "label": parsed["label"],
                "raw": parsed["raw"],
            }
            fout.write(json.dumps(row, allow_nan=False) + "\n")
            scores.append({"label": parsed["label"]})
            if (i + 1) % log_interval == 0:
                elapsed = time.perf_counter() - t0
                logger.info(
                    f"    [{i+1:,}/{n_to_judge:,}] elapsed={elapsed:.0f}s  "
                    f"throughput={ (i+1)/max(elapsed,1):.2f} judgments/sec"
                )

    judging_seconds = time.perf_counter() - t0
    logger.info(f"  judging done in {judging_seconds:.1f}s")

    agg = _aggregate_scores(scores)
    results_hash = hashlib.sha256(judgments_path.read_bytes()).hexdigest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "ablation": ablation,
        "ablation_label": cfg["label"],
        "judge_model": judge_model,
        "n_total": agg["n_total"],
        "n_judged": agg["n_judged"],
        "n_unknown": agg["n_unknown"],
        "faithful_rate": agg["faithful_rate"],
        "partial_rate": agg["partial_rate"],
        "hallucinated_rate": agg["hallucinated_rate"],
        "judging_seconds": round(judging_seconds, 3),
        "limit": limit,
        "top_k_context": top_k_context,
        "max_chunks_per_cluster": max_chunks_per_cluster,
        "git_sha": _git_sha(),
        "results_hash": results_hash,
    }
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(f"  faithful_rate     : {agg['faithful_rate']:.4f}")
    logger.info(f"  partial_rate      : {agg['partial_rate']:.4f}")
    logger.info(f"  hallucinated_rate : {agg['hallucinated_rate']:.4f}")
    logger.info(f"  wrote -> {summary_path}")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="MS5 LLM-as-judge faithfulness scorer.",
    )
    ap.add_argument("--ablation", type=str, required=True)
    ap.add_argument("--rag-root", type=Path, default=DEFAULT_RAG_ROOT)
    ap.add_argument("--gold-path", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    ap.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    ap.add_argument("--top-k-context", type=int, default=5)
    ap.add_argument("--max-chunks-per-cluster", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of queries judged (for smoke runs)")
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        print(
            f"[hallucination_judge] DRY RUN  schema={SCHEMA_VERSION}  "
            f"judge={args.judge_model}  ablation={args.ablation}  "
            f"git_sha={_git_sha()}  args={vars(args)}"
        )
        sys.exit(0)
    main(
        ablation=args.ablation,
        rag_root=args.rag_root,
        gold_path=args.gold_path,
        retrieval_dir=args.retrieval_dir,
        corpus_path=args.corpus_path,
        out_root=args.out_root,
        judge_model=args.judge_model,
        max_retries=args.max_retries,
        top_k_context=args.top_k_context,
        max_chunks_per_cluster=args.max_chunks_per_cluster,
        limit=args.limit,
    )
