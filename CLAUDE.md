# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A Harvard CS1090B research project studying hallucination in legal RAG chatbots. The pipeline compares BM25, BGE-M3 dense retrieval, and a hybrid reranker using CourtListener federal appellate opinions as the corpus and LePaRD citation pairs as gold-standard retrieval ground truth. The generator is an API-based LLM; hallucination is measured across three tiers (retrieval grounding, NLI contradiction, citation existence).

## Environment setup

```bash
bash setup.sh          # full bootstrap: uv venv, pinned deps, GPU checks, pre-push hook
SKIP_GPU=1 bash setup.sh   # CPU-only (no CUDA checks)
```

All commands that invoke Python or pytest must use `.venv/bin/` explicitly — never the system Python.

## Common commands

```bash
# Tests
make test-unit          # fast gate (pure function tests, no I/O)
make test-contract      # schema + provenance + quality threshold tests
make check              # lint + typecheck + unit + contract (pre-push gate)
make test-cov           # full suite with HTML/XML coverage (≥80% required)

# Run a single test file
.venv/bin/python -m pytest tests/test_s3_discovery.py -v

# Run a single test by name
.venv/bin/python -m pytest tests/test_filter_logic.py::test_federal_filter -v

# Quality
make lint               # ruff check src/ tests/
make format             # ruff --fix + ruff format
make typecheck          # mypy src/

# Data (DVC)
.venv/bin/dvc pull data/raw/cl_federal_appellate_bulk.dvc lepard_train_4000000_rev0194f95.jsonl.dvc
.venv/bin/dvc status    # show what's missing from local cache
```

## Test markers

| Marker | Meaning |
|--------|---------|
| `unit` | Pure functions, no I/O — runs everywhere |
| `contract` | Schema, provenance, quality thresholds |
| `integration` | Pipeline on frozen fixtures |
| `artifact` | Requires `setup.sh` artifacts on disk |
| `gpu` | Requires CUDA |

Default `make test` runs `not gpu`. The pre-push hook runs `make check` (unit + contract only).

## Data architecture

All large files are DVC-tracked in `s3://cs1090b-hallucinationlegalragchatbots/dvc` (us-east-2). The `.dvc` pointer files are committed to git; the actual data is not.

| Path | Contents | Size |
|------|----------|------|
| `data/raw/cl_federal_appellate_bulk/` | 162 JSONL shards, ~10k opinions each | 45 GB |
| `data/raw/cl_bulk/` | 4 raw CourtListener bulk CSVs | 61 GB |
| `lepard_train_4000000_rev0194f95.jsonl` | LePaRD 4M citation pairs | 5.8 GB |
| `data/processed/baseline/corpus_chunks.jsonl` | 5.79M BGE-M3 tokenized chunks | 26 GB |
| `data/processed/baseline/bm25_results.jsonl` | BM25 top-100 per query | 230 MB |
| `data/processed/baseline/bge_m3_results.jsonl` | BGE-M3 top-100 per query | 232 MB |

The raw CSVs in `cl_bulk/` are **not needed** for anything downstream of the shards. Only pull them if you need to rerun `src.pipeline.run_pipeline()` from scratch.

The citations CSV (`lepard_eda/cl_bulk/citations-2026-03-31.csv.bz2`) is auto-downloaded from the **public** CourtListener S3 bucket by `scripts/build_lepard_cl_subset.py` on first run — no DVC pull needed.

## Pipeline stages (notebook cell map)

The notebook `notebooks/Project_Group_#43_MS3_GPU_v01.ipynb` is thin orchestration only — no business logic lives in it. The internal "Cell N" labels in code comments are the authoritative cell identifiers (they do not match the visual cell position in Jupyter).

| Cell label | What it does | Key module/script |
|-----------|-------------|------------------|
| Cell 1–3 | Bootstrap, GPU preflight, reproducibility config | `src.repro`, `src.environment` |
| Cell 4 | Download CourtListener bulk CSVs from public S3 | `src.s3_discovery`, `src.bulk_download` |
| Cell 5 | Filter chain + extract 1.46M opinions to 162 JSONL shards | `src.pipeline.run_pipeline` |
| Cell 6 | 8-gate RAG-readiness probe (Polars full-corpus scan) | `src.dataset_probe` |
| Cell 7 | Ingest LePaRD 4M pairs from HF Hub at pinned revision | `scripts/ingest_lepard.py` |
| Cell 8 | LePaRD ↔ CourtListener compatibility audit | `src.lepard_cl_compat` |
| Cell 9 | NaN/encoding/parse quality gate over all shards | `scripts/audit_jsonl_nan.py` |
| Cell 10 | EDA: CourtListener corpus distributions | `scripts/eda_ms3_corpus.py` |
| Cell 11 | EDA: LePaRD × CourtListener compatibility | `scripts/eda_ms3_lepard.py` |
| Cell 11b | Build verified LePaRD-CL subset (citation + fuzzy text match) | `scripts/build_lepard_cl_subset.py` |
| Cell 12 | Chunk 1.46M opinions + extract 47K gold pairs | `scripts/baseline_prep.py` |
| Cell 13 | BM25 baseline retrieval (SLURM sbatch, ~70 min, ~90 GB RAM) | `scripts/baseline_bm25.py` |
| Cell 14 | BGE-M3 dense baseline (SLURM sbatch, 4× L4 GPU, ~4 hrs) | `scripts/baseline_bge_m3.py` |
| Cell 15 | Evaluate both baselines: Hit@k, MRR, NDCG@10 | `scripts/baseline_eval.py` |

## `src/` module responsibilities

- **`pipeline.py`** — top-level orchestrator: `run_pipeline()` + `validate_pipeline()`. Idempotent: reads manifest and skips completed shards.
- **`filter_chain.py`** — courts → dockets → clusters three-stage filter to isolate 13 federal appellate courts.
- **`extract.py`** — streams opinions through the filter and writes JSONL shards with SHA-256 checksums.
- **`s3_discovery.py`** / **`bulk_download.py`** — anonymous access to the public CourtListener S3 bucket (`com-courtlistener-storage`). No credentials needed.
- **`config.py`** (`PipelineConfig`) — single source of truth for all paths, S3 URLs, filter lists, and thresholds.
- **`repro.py`** — sets `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG`, seeds all RNGs. Must be called first in every notebook cell and CLI script.
- **`environment.py`** — asserts exact library versions and GPU capability at startup.
- **`model_loader.py`** — single place for all HF model/tokenizer loads. Enforces BGE-M3 CLS pooling via runtime assertion (not mean pooling).
- **`dataset_probe.py`** — Polars `scan_ndjson` 8-gate full-corpus readiness check.
- **`lepard_cl_compat.py`** — citation resolution + fuzzy text overlap to verify LePaRD pairs exist in the CL corpus.
- **`schemas.py`** / **`data_contracts.py`** — Pydantic models for all pipeline outputs; used for validation in tests and manifest verification.
- **`manifest.py`** — write/read/validate per-shard SHA-256 manifest. Pipeline is idempotent based on this file.
- **`wandb_logger.py`** — W&B integration; logs model pooling config, VRAM stats, and per-phase diagnostics.

## Critical constraints

**BGE-M3 pooling**: uses CLS pooling, not mean pooling. This is enforced by a runtime assertion in `src/model_loader.py`. Do not override.

**Sequential model loading**: only one model loaded into VRAM at a time. After each phase: delete DataLoader + iterator, then `gc.collect()` + `torch.cuda.empty_cache()`. Budget: BGE-M3 ~2.3 GB, reranker ~2 GB, generator ~14–15 GB, NLI ~3 GB — all within 23.7 GB.

**GPU addressing**: always `.to("cuda")` or `.to("cuda:0")`. Never hardcode a physical GPU ordinal — SLURM remaps the allocated GPU to index 0.

**Version pins**: `transformers==4.41.2`, `tokenizers==0.19.1` are hard-pinned. Do not upgrade without re-certifying all smoke tests.

**Reproducibility**: `src.repro.configure()` must be the first call in any script or notebook that touches models or data. `PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are set by the Makefile and `setup.sh`; the Makefile exports them for all `make` targets.

**Polars corpus scan**: `src/dataset_probe.py` always uses `scan_ndjson` (Polars, CPU-only) for the full 1.46M-opinion scan — never load the corpus into a GPU-resident structure.

## SLURM jobs

Cells 13 (BM25) and 14 (BGE-M3) submit SLURM `sbatch` jobs rather than running inline because both exceed SSH session lifetimes and RAM limits of the general partition. BM25 requires the `gpu` partition for its ~90 GB RSS peak. Both cells poll for completion and print the SLURM log path. Idempotency: each cell skips submission if its `*_summary.json` already validates successfully.

## DVC remote

```
remote: s3-dvc
url:    s3://cs1090b-hallucinationlegalragchatbots/dvc
region: us-east-2
```

Requires AWS credentials. Run `aws configure` or set `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in `.env`.
