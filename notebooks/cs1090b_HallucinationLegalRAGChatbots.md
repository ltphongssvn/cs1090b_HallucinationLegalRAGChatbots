---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: hallucination-legal-rag
  display_name: HallucinationLegalRAG (3.11.9)
  language: python
---

```{code-cell} ipython3
# Cell 1: Environment Setup & GPU Verification (TDD)
# Project: HallucinationLegalRAGChatbots
# Notebook: cs1090b_hw2.ipynb
#
# Clean Code: Thin orchestration. All logic in src/environment.py.
# No side effects on import — checks wrapped in run_environment_checks().
#
# Reproducibility: src/repro.configure() MUST be called first — before any
# import of torch, transformers, or other ML libraries. This guarantees
# notebook/CLI parity: identical PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG,
# TOKENIZERS_PARALLELISM, torch.use_deterministic_algorithms, and cuDNN flags
# regardless of whether the code runs in JupyterLab or from the CLI.
# See src/repro.py for full rationale.
#
# Failure isolation: run_preflight_checks() is a hard gate that validates ALL
# critical preconditions BEFORE any expensive GPU training begins. This prevents
# wasted GPU hours from misconfigured environments discovered mid-run.
# If preflight fails, the notebook raises immediately with an actionable message.

# --- Step 0: Reproducibility (must be FIRST — before torch import) ---
import sys, os
os.chdir(os.path.dirname(os.getcwd()))  # go up to project root
sys.path.insert(0, os.getcwd())

from src.repro import configure as _configure_repro
repro_cfg = _configure_repro(verbose=True)

# --- Step 1: Remaining imports (torch now imported safely after repro config) ---
import logging
from src.environment import (
    run_environment_checks,
    run_preflight_checks,
    get_environment_summary,
    REQUIRED_DEPS,
)
from src.timer import cell_timer

logger = logging.getLogger("cell1")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("  %(message)s"))
    logger.addHandler(handler)

with cell_timer("Cell 1 — Environment Setup & GPU Verification", logger=logger):
    # --- TDD Contract ---
    logger.info("=" * 60)
    logger.info("  TDD RED→GREEN: Environment Contract")
    logger.info("=" * 60)
    assert run_environment_checks(logger=logger), "Environment contract violated"

    # --- Preflight Gate (hard stop before any expensive training) ---
    # Validates: GPU count/name/VRAM, disk space, repro config integrity,
    # uv.lock hash match, and src/repro.py presence.
    # Raises PreflightError with actionable message on any failure.
    logger.info(f"\n{'=' * 60}")
    logger.info("  Preflight Checks — Failure Isolation Gate")
    logger.info("=" * 60)
    run_preflight_checks(logger=logger, repro_cfg=repro_cfg)

    # --- Repro Config Summary ---
    logger.info(f"\n{'=' * 60}")
    logger.info("  Reproducibility Config (src/repro.configure)")
    logger.info("=" * 60)
    for k, v in repro_cfg.items():
        logger.info(f"  {k:<36} {v}")

    # --- Environment Summary ---
    env = get_environment_summary()
    logger.info(f"\n{'=' * 60}")
    logger.info("  Verified Environment")
    logger.info("=" * 60)
    for pkg, constraint in REQUIRED_DEPS.items():
        logger.info(f"  {pkg:<16} {env[pkg]:<12} (requires {constraint or 'any'})")
    logger.info(f"  {'GPU':<16} {env['gpu']}")
    logger.info(f"  {'GPU Memory':<16} {env['gpu_memory_gb']} GB")
    logger.info(f"  {'CUDA':<16} {env['cuda']}")
    logger.info(f"\n✓ Environment ready — all preflight checks passed, safe to proceed")
```

```{code-cell} ipython3
# Cell 2: Data Acquisition — CourtListener Bulk Data (TDD)
# Project: HallucinationLegalRAGChatbots
# Notebook: cs1090b_hw2.ipynb
#
# PDF Spec Traceability:
#   [DATA]    "CourtListener — 10M+ U.S. court opinions (CC BY-ND 4.0)"
#   [SCOPE]   "federal appellate opinions, ~500K cases"
#   [DELIVER] "acquisition and preprocessing of data"

import os
import logging
from src.config import PipelineConfig
from src.pipeline import run_pipeline, validate_pipeline
from src.exceptions import PipelineError
from src.timer import cell_timer

logger = logging.getLogger("cell2")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("  %(message)s"))
    logger.addHandler(handler)

with cell_timer("Cell 2 — Data Acquisition Pipeline + TDD", logger=logger):
    # --- Config ---
    config = PipelineConfig(
        shard_size=int(os.environ.get("CL_SHARD_SIZE", 10000)),
    )

    # --- Run ---
    logger.info("=" * 60)
    logger.info("  Data Acquisition Pipeline")
    logger.info("=" * 60)

    manifest = run_pipeline(config=config, logger=logger)

    # --- TDD ---
    logger.info(f"\n{'=' * 60}")
    logger.info("  TDD Contract Tests")
    logger.info("=" * 60)

    try:
        validate_pipeline(config=config, manifest_data=manifest, logger=logger,
                          shard_strategy="sample")  # "all" for final experiments
        logger.info("\n✓ All contract tests passed")
    except PipelineError as error:
        logger.error(f"\n✗ {error}")
        raise

    # --- Summary (from manifest, zero rescans) ---
    logger.info(f"\n{'=' * 60}")
    logger.info("  Dataset Summary")
    logger.info("=" * 60)
    logger.info(f"Shards:     {manifest['num_shards']} × ~{config.shard_size:,}")
    logger.info(f"Total:      {manifest['num_cases']:,} cases")
    logger.info(f"Scanned:    {manifest.get('scanned', '?'):,}")
    logger.info(f"Skipped:    {manifest.get('skipped_empty', 0):,} empty, "
                f"{manifest.get('skipped_parse', 0):,} parse errors")
    logger.info(f"Schema:     {manifest.get('schema', [])}")

    text_length_stats = manifest.get("text_length_stats", {})
    if text_length_stats:
        logger.info(f"\nText: mean={text_length_stats['mean']:,} "
                    f"median={text_length_stats['median']:,} "
                    f"min={text_length_stats['min']:,} max={text_length_stats['max']:,}")

    court_distribution = manifest.get("court_distribution", {})
    if court_distribution:
        logger.info(f"\nCircuits ({len(court_distribution)}):")
        for court_id in sorted(court_distribution, key=court_distribution.get, reverse=True):
            logger.info(f"  {court_id:<10} {court_distribution[court_id]:>8,}")

    text_sources = manifest.get("text_source_counts", {})
    if text_sources:
        logger.info(f"\nText sources:")
        for source_name, count in text_sources.items():
            if count > 0:
                logger.info(f"  {source_name:<25} {count:>8,} "
                            f"({count/manifest['num_cases']*100:.1f}%)")
    logger.info(f"OCR-extracted: {manifest.get('ocr_extracted_count', 0):,}")

    filter_chain = manifest.get("filter_chain", {})
    if filter_chain:
        logger.info(f"\nChain: {filter_chain['courts']} courts → "
                    f"{filter_chain['dockets']:,} dockets → "
                    f"{filter_chain['clusters']:,} clusters")

    logger.info(f"\nPDF: '~500K federal appellate' → extracted {manifest['num_cases']:,}")
```
