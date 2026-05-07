# Reproducibility Runbook

This document tells a TF reviewer or new collaborator how to reproduce every experiment result in this repository from a clean machine, end-to-end.

## 1. Clone and pin to the published commit

```bash
git clone https://github.com/ltphongssvn/cs1090b_HallucinationLegalRAGChatbots.git
cd cs1090b_HallucinationLegalRAGChatbots
git checkout <commit-sha-from-paper>     # pin to published artifact set
```

Every `*_summary.json` file under `data/processed/` records the `git_sha` of the commit that produced it. Use that SHA here.

## 2. Restore the Python environment

```bash
bash setup.sh              # creates .venv, runs `uv sync` against uv.lock
source .venv/bin/activate
```

`uv.lock` is the pinned dependency manifest; do not regenerate it during reproduction.

## 3. Restore data artifacts via DVC

Source data and intermediate retrieval/RAG outputs are tracked by DVC, not git. Pull them from the configured remote:

```bash
dvc pull                   # downloads everything referenced by *.dvc pointers
```

After this, `artifacts/`, `data/raw/`, and `data/processed/` will match the state at the pinned commit.

## 4. Configure Weights & Biases

The pipeline writes telemetry to W&B. Two paths are supported:

### 4a. Online — log to your own W&B workspace

```bash
export WANDB_ENTITY=your-team-or-username
export WANDB_PROJECT=hallucination-legal-rag
export WANDB_API_KEY=<your-key>
```

`WANDB_ENTITY` and `WANDB_PROJECT` override the script defaults so you do not need to edit any source. All scripts read them via `src.repro.get_wandb_entity` / `get_wandb_project`.

### 4b. Offline — capture runs locally, sync later

The Harvard ODD GPU cluster has no outbound internet from compute nodes. Use offline mode:

```bash
export WANDB_MODE=offline
```

Runs are written to `wandb/offline-run-*` directories. From any node with internet access, sync them to the cloud with:

```bash
wandb sync wandb/offline-run-*
```

Note: `run.use_artifact()` is unsupported in offline mode (W&B issue [#5309](https://github.com/wandb/wandb/issues/5309)). The lineage helpers in `src/wandb_lineage.py` detect this and skip silently — lineage edges only appear after `wandb sync` against an online project.

## 5. Run group taxonomy

Related W&B runs cluster under shared `group=` labels for at-a-glance comparison:

| Stage             | Group prefix         |
|-------------------|----------------------|
| EDA               | `ms3-eda-<sha12>`    |
| Dataset probe     | `dataset-probe-<sha12>` |
| Data quality gate | `data-quality-<sha12>` |
| Baselines / RRF / reranker / RAG / judge | `ms4-baselines-<sha12>` |

Set `WANDB_RUN_GROUP` to override the auto-generated prefix.

## 6. Reproduce the pipeline

Each stage is a self-contained script that reads from DVC-tracked inputs and writes to DVC-tracked outputs. Run in order:

```bash
# Stage 1: ingest + clean
uv run python scripts/ingest_lepard.py
uv run python scripts/clean_corpus.py
uv run python scripts/clean_gold_pairs.py
uv run python scripts/baseline_prep.py --verified-subset data/processed/lepard_cl_verified_subset.jsonl

# Stage 2: retrievers
uv run python scripts/baseline_bm25.py --verified --log-to-wandb
uv run python scripts/baseline_bge_m3.py --verified --log-to-wandb

# Stage 3: fusion + reranking
uv run python scripts/baseline_rrf.py \
    --bm25-path data/processed/baseline/cleaned/bm25_results.jsonl \
    --bge-m3-path data/processed/baseline/cleaned/bge_m3_results.jsonl \
    --out-dir data/processed/baseline/cleaned
uv run python scripts/baseline_reranker.py --score-mode maxp

# Stage 4: RAG + hallucination evaluation
uv run python scripts/rag_generate.py
uv run python scripts/hallucination_judge.py
uv run python scripts/aggregate_judgments.py

# Stage 5: stratified evaluation
uv run python scripts/stratified_eval.py \
    --gold-path data/processed/baseline/cleaned/gold_pairs_test.jsonl \
    --results-path data/processed/baseline/cleaned/reranker_results.jsonl
```

Every script captures `git_sha` and `seed` in its `wandb.init(config=...)`, so any run page on W&B uniquely identifies the code state and the deterministic seed used.

## 7. Verify a reproduction

After running stages 1–5, compare local SHA-256 hashes against the published `*_summary.json` files:

```bash
python -c "import hashlib, json; \
  s = json.load(open('data/processed/baseline/cleaned/bm25_summary.json')); \
  h = hashlib.sha256(open('data/processed/baseline/cleaned/bm25_results.jsonl','rb').read()).hexdigest(); \
  print('match' if h == s['results_hash'] else f'MISMATCH local={h} published={s[\"results_hash\"]}')"
```

A mismatch means either: (a) you are on a different `git_sha` than the published artifact, (b) `uv.lock` has drifted, or (c) hardware differs in a way that affects floating-point reductions (rare for these scripts; deterministic seeds + `torch.use_deterministic_algorithms(True)` mitigate).

## 8. Cluster-specific notes

On the Harvard ODD `gpu-dy-gpu-cr-*` nodes:

- Set `GIT_COMMIT_SHA` env var if running outside a git working tree (CI containers): the `_git_sha()` helper in `src/repro.py` honours it.
- 4× L4 GPUs are available per node. `baseline_bge_m3.py` shards corpus across `--world-size 4 --rank {0..3}`.
- Outbound network is firewalled. Use `WANDB_MODE=offline` (see §4b) and `dvc remote` configured for an internal S3 endpoint.
