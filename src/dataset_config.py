# src/dataset_config.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/dataset_config.py
"""Typed configuration for pile-of-law subset ingestion and materialisation.

This module defines two dataclasses that jointly describe **what** data a
run consumes and **how** a previously-materialised artifact was produced:

* :class:`DatasetConfig` — the Hydra-injectable input config. Callers
  set it either directly in Python or via
  ``configs/data/legal_rag.yaml``.
* :class:`ArtifactManifest` — provenance metadata written next to every
  materialised artifact as ``artifact_manifest.json``, enabling strict
  compatibility checks before a later run reuses that artifact.

Design notes
------------
* **Two data sources**: ``"artifact"`` (the training default) loads from
  a preprocessed local directory, while ``"hf"`` streams directly from
  the Hugging Face Hub and requires ``trust_remote_code``. Production
  training runs should never touch the Hub; the ``"hf"`` path exists
  solely for the one-off ingestion step.
* **Pinned revision**: ``revision`` stores the exact Git SHA of the
  pile-of-law dataset repository. Unpinned loads would silently change
  corpus contents between runs.
* **Pure data**: no I/O, no validation against the filesystem. Those
  concerns live in :mod:`src.dataset_loader`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal


@dataclass
class DatasetConfig:
    """Hydra-injectable configuration for a pile-of-law subset.

    All fields are overridable via ``configs/data/legal_rag.yaml`` or by
    direct kwargs. Instances are typically constructed from a Hydra
    ``DictConfig`` with :meth:`from_hydra`.

    Attributes:
        dataset_id: Hugging Face Hub dataset identifier. Defaults to
            ``"pile-of-law/pile-of-law"``.
        subset: Named subset inside ``dataset_id``; defaults to the
            CourtListener opinions split used by the RAG corpus.
        split: HF split name (``"train"``, ``"validation"``, ``"test"``).
        revision: Pinned Git SHA of the dataset repository. Required
            for reproducible runs — unpinned loads can silently shift
            corpus contents.
        reproducible: When ``True``, all downstream loaders must enforce
            deterministic behaviour (fixed seeds, no streaming in
            artifact mode unless explicitly allowed, sorted file order).
        min_text_length: Rows whose concatenated text is shorter than
            this many characters are dropped. Mirrors the threshold in
            :class:`src.config.PipelineConfig`.
        streaming: Use HF datasets streaming mode. Required for the
            full pile-of-law (too large for local disk) but introduces
            non-determinism in artifact mode unless
            :attr:`allow_streaming_in_artifact_mode` is also ``True``.
        text_fields: Ordered tuple of column names to search for the
            canonical document text. First non-empty value wins.
        required_fields: Columns that must be present and non-null in
            every row; rows missing any are rejected.
        data_source: Either ``"artifact"`` (local preprocessed dir,
            training default) or ``"hf"`` (live HF Hub, ingestion only).
        artifact_path: Filesystem path to the preprocessed artifact when
            ``data_source == "artifact"``. Ignored otherwise.
        allow_streaming_in_artifact_mode: Escape hatch to permit
            streaming during artifact loads. Off by default because
            streaming breaks exact-rerun reproducibility.
    """

    dataset_id: str = "pile-of-law/pile-of-law"
    subset: str = "r_courtlistener_opinions"
    split: str = "train"
    revision: str = "0dc9f2c26b42af4cb6330f36d6146e82f9117a3b"  # pragma: allowlist secret
    reproducible: bool = True
    min_text_length: int = 50
    streaming: bool = True
    text_fields: tuple[str, ...] = field(default_factory=lambda: ("text", "contents"))
    required_fields: frozenset[str] = field(
        default_factory=lambda: frozenset({"created_timestamp", "downloaded_timestamp", "url"})
    )
    data_source: Literal["hf", "artifact"] = "artifact"
    artifact_path: str | None = None
    # Streaming adds non-determinism in artifact mode; disable for exact reruns.
    allow_streaming_in_artifact_mode: bool = False

    @classmethod
    def from_hydra(cls, cfg: object) -> "DatasetConfig":
        """Instantiate from a Hydra ``DictConfig`` or plain ``dict``.

        Accepts three input shapes:

        1. A plain Python ``dict`` — used directly.
        2. An OmegaConf ``DictConfig`` — resolved via
           ``OmegaConf.to_container(cfg, resolve=True)`` so
           interpolations are materialised before field assignment.
        3. Anything else dict-like — passed through ``dict(cfg)`` as a
           best-effort fallback.

        The method also normalises two collection fields that Hydra
        loads as plain lists: ``text_fields`` is coerced to ``tuple``
        and ``required_fields`` to ``frozenset``, preserving the
        immutability guarantees declared on the dataclass.

        Args:
            cfg: The configuration object. Typically a Hydra
                ``DictConfig`` produced by ``@hydra.main``.

        Returns:
            A fully-populated :class:`DatasetConfig`.

        Raises:
            TypeError: ``cfg`` contains keys not defined on the
                dataclass (surfaced by ``cls(**d)``).
        """
        d: Dict[str, Any]
        if isinstance(cfg, dict):
            d = dict(cfg)
        else:
            try:
                from omegaconf import OmegaConf  # type: ignore[import]

                d = dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
            except ImportError:
                d = dict(cfg)  # type: ignore[call-overload]
        if "text_fields" in d:
            d["text_fields"] = tuple(d["text_fields"])
        if "required_fields" in d:
            d["required_fields"] = frozenset(d["required_fields"])
        return cls(**d)


@dataclass
class ArtifactManifest:
    """Provenance metadata emitted beside every materialised artifact.

    Written as ``artifact_manifest.json`` next to the data payload. A
    downstream run loads the manifest first and calls
    :meth:`is_compatible_with` against its own :class:`DatasetConfig`
    to decide whether the artifact can be reused as-is or must be
    rebuilt.

    Attributes:
        source_dataset_id: Hugging Face dataset identifier used during
            ingestion.
        subset: Subset name used during ingestion.
        revision: Exact Git SHA of the dataset repo at ingestion time.
        ingestion_timestamp: ISO-8601 UTC timestamp of the build.
        loader_version: Semantic version of :mod:`src.dataset_loader`
            at build time. Bump on any behavioural change so stale
            artifacts are rejected.
        schema_version: Version of the row schema in :mod:`src.schemas`.
        row_count: Number of rows actually written to the artifact.
        artifact_checksum: Aggregate content hash of the artifact
            payload files (e.g. SHA-256 over sorted file contents).
        hf_datasets_version: Version of the ``datasets`` library that
            performed the load — surfaces silent upstream schema drift.
    """

    source_dataset_id: str
    subset: str
    revision: str  # pragma: allowlist secret
    ingestion_timestamp: str
    loader_version: str
    schema_version: str
    row_count: int
    artifact_checksum: str  # pragma: allowlist secret
    hf_datasets_version: str

    def is_compatible_with(self, config: "DatasetConfig") -> list[str]:
        """Check this artifact against a target :class:`DatasetConfig`.

        Compares the three identity fields that determine corpus
        contents: ``dataset_id``, ``subset``, and ``revision``. Other
        manifest fields (row count, checksum, library versions) are
        deliberately **not** checked here — they are reported by the
        manifest collector for auditing but must not gate reuse, since
        a benign :mod:`src.dataset_loader` bugfix would otherwise
        invalidate every existing artifact.

        Args:
            config: The target configuration a caller wants to load.

        Returns:
            A list of human-readable incompatibility descriptions. An
            empty list means the artifact is safe to reuse.
        """
        issues: list[str] = []
        if self.source_dataset_id != config.dataset_id:
            issues.append(f"dataset_id mismatch: {self.source_dataset_id!r} vs {config.dataset_id!r}")
        if self.subset != config.subset:
            issues.append(f"subset mismatch: {self.subset!r} vs {config.subset!r}")
        if self.revision != config.revision:
            issues.append(f"revision mismatch: {self.revision!r} vs {config.revision!r}")
        return issues
