"""Shared Pydantic artifact schemas for MS3 EDA pipelines.

Single source of truth for summary.json contracts emitted by:
    - scripts/eda_ms3_corpus.py   -> EdaCorpusSummary
    - scripts/eda_ms3_lepard.py   -> EdaLepardSummary

Tests and scripts import from here to prevent schema drift.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EdaCorpusSummary(BaseModel):
    """Contract for logs/eda_ms3/summary.json (CourtListener corpus EDA)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    n_total: int = Field(ge=0)
    n_after_filter: int = Field(ge=0)
    n_short_lt_100: int = Field(ge=0)
    text_length_mean: float = Field(ge=0, allow_inf_nan=False)
    text_length_median: float = Field(ge=0, allow_inf_nan=False)
    text_length_mean_filtered: float = Field(ge=0, allow_inf_nan=False)
    text_length_median_filtered: float = Field(ge=0, allow_inf_nan=False)
    filter_threshold: int
    circuit_counts: dict[str, int]
    circuit_order: list[str]
    chart_ranges: dict[str, list[int]]
    chart_overflow_counts: dict[str, int]
    corpus_manifest_sha: str = Field(min_length=64, max_length=64)
    figure_hashes: dict[str, str]
    git_sha: str


class EdaLepardSummary(BaseModel):
    """Contract for logs/eda_ms3_lepard/summary.json (LePaRD × CL compat)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    total_rows: int = Field(ge=0)
    unique_pairs: int = Field(ge=0)
    lepard_unique_ids: int = Field(ge=0)
    cl_unique_ids: int = Field(ge=0)
    overlap_ids: int = Field(ge=0)
    both_in_cl: int = Field(ge=0)
    source_only: int = Field(ge=0)
    dest_only: int = Field(ge=0)
    neither: int = Field(ge=0)
    usable_pct: float = Field(ge=0, le=100, allow_inf_nan=False)
    court_distribution: dict[str, int]
    figure_hashes: dict[str, str] = Field(
        description="SHA256 hex of each emitted PNG (64 lowercase hex chars)",
    )
    git_sha: str


class BaselinePrepSummary(BaseModel):
    """Contract for data/processed/baseline/summary.json (baseline dataset prep)."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str
    corpus_chunks: int = Field(ge=0)
    n_opinions_chunked: int = Field(ge=0)
    gold_pairs_total: int = Field(ge=0)
    gold_pairs_train: int = Field(ge=0)
    gold_pairs_val: int = Field(ge=0)
    gold_pairs_test: int = Field(ge=0)
    val_court_distribution: dict[str, int]
    test_court_distribution: dict[str, int]
    seed: int
    git_sha: str
    corpus_manifest_sha: str = Field(min_length=64, max_length=64)
