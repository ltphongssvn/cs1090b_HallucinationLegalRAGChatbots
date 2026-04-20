"""Shared Pydantic artifact schemas for MS3 EDA pipelines.

Single source of truth for summary.json contracts emitted by:
    - scripts/eda_ms3_corpus.py   -> EdaCorpusSummary
    - scripts/eda_ms3_lepard.py   -> EdaLepardSummary

Tests and scripts import from here to prevent schema drift.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    gold_pair_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="SHA256 hex of each gold_pairs_*.jsonl (64 lowercase hex chars)",
    )


class BaselineBM25Summary(BaseModel):
    """Contract for data/processed/baseline/bm25_summary.json."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str
    n_queries: int = Field(ge=0)
    n_corpus_chunks: int = Field(ge=0)
    n_unique_opinions: int = Field(ge=0)
    top_k: int = Field(ge=1)
    bm25_k1: float = Field(gt=0)
    bm25_b: float = Field(ge=0, le=1)
    index_build_seconds: float = Field(ge=0, allow_inf_nan=False)
    retrieval_seconds: float = Field(ge=0, allow_inf_nan=False)
    seed: int
    git_sha: str
    results_hash: str = Field(min_length=64, max_length=64)


class BaselineBM25ResultLine(BaseModel):
    """Contract for each line in data/processed/baseline/bm25_results.jsonl."""

    model_config = ConfigDict(extra="forbid")
    source_id: int
    dest_id: int
    retrieved: list[dict[str, float | int]]


class BaselineBgeM3Summary(BaseModel):
    """Contract for data/processed/baseline/bge_m3_summary.json."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    n_queries: int = Field(ge=0)
    n_corpus_chunks: int = Field(ge=0)
    n_unique_opinions: int = Field(ge=0)
    top_k: int = Field(ge=1)
    encoder_model: str
    embedding_dim: int = Field(ge=1)
    device: str
    device_name: str
    encode_batch_size: int = Field(ge=1)
    similarity_metric: str = Field(pattern=r"^(cosine|dot|euclidean)$")
    normalize_embeddings: bool
    max_length: int = Field(ge=1)
    dtype: str
    world_size: int = Field(ge=1)
    shard_rank: int = Field(ge=0)
    encoder_load_seconds: float = Field(ge=0, allow_inf_nan=False)
    index_build_seconds: float = Field(ge=0, allow_inf_nan=False)
    query_encode_seconds: float = Field(ge=0, allow_inf_nan=False)
    retrieval_seconds: float = Field(ge=0, allow_inf_nan=False)
    seed: int
    git_sha: str = Field(pattern=r"^[a-f0-9]{12}$")
    results_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _unique_opinions_le_chunks(self) -> BaselineBgeM3Summary:
        if self.n_unique_opinions > self.n_corpus_chunks:
            raise ValueError(f"n_unique_opinions ({self.n_unique_opinions}) > n_corpus_chunks ({self.n_corpus_chunks})")
        return self


class RetrievalHit(BaseModel):
    """Strict contract for a single opinion-level hit in retrieved list."""

    model_config = ConfigDict(extra="forbid")
    opinion_id: int
    score: float = Field(ge=-1.0, le=2.0)  # cosine similarity range (normalized)


class BaselineBgeM3ResultLine(BaseModel):
    """Contract for each line in data/processed/baseline/bge_m3_results.jsonl."""

    model_config = ConfigDict(extra="forbid")
    source_id: int
    dest_id: int
    retrieved: list[RetrievalHit]


class BaselineEvalSummary(BaseModel):
    """MS3 retrieval evaluation output — Hit@k, MRR, NDCG for BM25 + BGE-M3."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    n_queries: int = Field(ge=1)
    k_values: list[int] = Field(min_length=1)
    ndcg_k: int = Field(ge=1)

    # BM25 metrics
    bm25_hit_at_k: dict[str, float]
    bm25_mrr: float = Field(ge=0.0, le=1.0)
    bm25_ndcg_at_10: float = Field(ge=0.0, le=1.0)
    bm25_results_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    # BGE-M3 metrics
    bge_m3_hit_at_k: dict[str, float]
    bge_m3_mrr: float = Field(ge=0.0, le=1.0)
    bge_m3_ndcg_at_10: float = Field(ge=0.0, le=1.0)
    bge_m3_results_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    # Paired comparison
    bge_m3_wins: int = Field(ge=0)
    bm25_wins: int = Field(ge=0)
    ties: int = Field(ge=0)

    # Provenance
    git_sha: str = Field(pattern=r"^[a-f0-9]{12}$")
    seed: int = Field(ge=0)
