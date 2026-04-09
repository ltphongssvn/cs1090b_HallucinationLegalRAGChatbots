# src/dataset_loader.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/dataset_loader.py
"""Reproducible loader for pile-of-law subsets with strict provenance guards.

This module is the single entry point the training and evaluation code
uses to obtain rows from a pile-of-law subset. It wraps the Hugging Face
``datasets`` library with three layers of safety that the raw library
does not provide:

1. **Artifact boundary**: in reproducible mode, live Hub loads are
   forbidden because the pile-of-law builder requires
   ``trust_remote_code=True``, which executes arbitrary maintainer code
   at load time. Reproducible runs must load from an immutable local
   artifact produced by a separate ingestion step.
2. **Revision pinning**: when reproducibility is on, the configured
   ``revision`` must be a 40-char hex commit SHA; mutable refs like
   ``main`` or ``HEAD`` are rejected because their resolved content can
   change between runs.
3. **Manifest compatibility**: when an artifact carries an
   ``artifact_manifest.json``, it is diffed against the current
   :class:`~src.dataset_config.DatasetConfig` before load, and any
   identity mismatch aborts the run.

All row-level logic (validation, normalization, filtering) is delegated
to :mod:`src.row_validator` and :mod:`src.row_normalizer`; this module
orchestrates only.
"""
from __future__ import annotations

import collections
import re
from typing import Any, Iterable, Iterator

from src.dataset_config import DatasetConfig
from src.row_normalizer import RowNormalizer
from src.row_validator import RowValidator

#: Matches a canonical 40-character lowercase Git commit SHA.
HEX_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

#: Refs that resolve to different commits over time and are therefore
#: banned under reproducible mode.
_MUTABLE_REFS = {"main", "master", "latest", "HEAD", ""}


def _histogram(values: list[int], bins: int = 10) -> list[dict[str, Any]]:
    """Compute an equal-width histogram over integer values.

    Used by :meth:`DatasetLoader.log_stats` to produce a compact
    token-length distribution suitable for W&B logging.

    Args:
        values: Raw integer samples. An empty list produces an empty
            histogram.
        bins: Number of equal-width buckets. Defaults to 10.

    Returns:
        A list of ``{"range": "lo-hi", "count": n}`` dicts, one per
        bucket. When all values are equal, a single bucket is returned
        to avoid a zero-width division.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"range": f"{lo}-{hi}", "count": len(values)}]
    width = (hi - lo) / bins
    buckets = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        buckets[idx] += 1
    return [{"range": f"{int(lo + i * width)}-{int(lo + (i + 1) * width)}", "count": c} for i, c in enumerate(buckets)]


class DatasetLoader:
    """Load, validate, and stream normalized rows from a pile-of-law subset.

    The loader owns a :class:`RowValidator` and :class:`RowNormalizer`
    derived from the injected :class:`DatasetConfig`, and exposes three
    classes of operations:

    * :meth:`load` — return a raw iterable from either a local artifact
      or the HF Hub, with all reproducibility guards applied.
    * :meth:`iter_valid_rows` — yield only rows that pass validation,
      after normalization.
    * Filtering helpers (:meth:`filter_by_date_range`,
      :meth:`filter_by_court`, :meth:`filter_min_text_tokens`) —
      generators that compose with :meth:`load` and each other.

    Plus :meth:`get_provenance` for run-metadata capture and
    :meth:`log_stats` for pre-training dataset telemetry.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Initialise with a fully-populated :class:`DatasetConfig`.

        Args:
            config: Immutable pipeline configuration. The loader keeps a
                reference; callers must not mutate it after construction.
        """
        self._config = config
        self._validator = RowValidator(config)
        self._normalizer = RowNormalizer(config, self._validator)

    def load(self) -> Iterable[dict[str, Any]]:
        """Return a raw row iterable, enforcing all reproducibility guards.

        Decision flow:

        1. In reproducible mode, reject ``data_source == "hf"`` outright
           because the pile-of-law builder requires ``trust_remote_code``.
        2. In reproducible mode, reject any ``revision`` that is not a
           40-char hex SHA (mutable refs resolve differently over time).
        3. For ``data_source == "artifact"``:
           * require ``artifact_path`` to be set;
           * reject ``streaming=True`` in reproducible mode unless
             :attr:`DatasetConfig.allow_streaming_in_artifact_mode`
             is explicitly ``True`` (streaming introduces iteration-order
             non-determinism);
           * validate the artifact manifest against the current config;
           * dispatch to ``datasets.load_from_disk``.
        4. For ``data_source == "hf"``: dispatch to ``datasets.load_dataset``
           with ``trust_remote_code=True`` (ingestion-only path).

        Returns:
            An iterable of raw dataset rows. The concrete type is either
            a ``datasets.Dataset`` (artifact path, non-streaming) or a
            ``datasets.IterableDataset`` (streaming), both of which
            yield ``dict[str, Any]`` rows.

        Raises:
            RuntimeError: Any reproducibility guard failed.
            ValueError: ``data_source == "artifact"`` but
                ``artifact_path`` is unset.
        """
        # Block HF Hub in reproducible mode — trust_remote_code is a security violation
        if self._config.reproducible and self._config.data_source == "hf":
            raise RuntimeError(
                "Reproducibility violation: HF Hub loading requires trust_remote_code=True "
                "which executes arbitrary remote code. Use a preprocessed immutable artifact "
                "for training. Set data_source='hf' with reproducible=False for ingestion only."
            )

        # Enforce immutable revision
        if self._config.reproducible and (
            self._config.revision in _MUTABLE_REFS or HEX_REVISION_RE.fullmatch(self._config.revision) is None
        ):
            raise RuntimeError(
                f"Reproducibility violation: revision={self._config.revision!r} is mutable. "
                "Set REVISION to a 40-char commit SHA, or set reproducible=False."
            )

        if self._config.data_source == "artifact":
            if not self._config.artifact_path:
                raise ValueError(
                    "data_source='artifact' requires artifact_path to be set. "
                    "Run the ingestion pipeline first: make ingest-dataset"
                )
            if (
                self._config.reproducible
                and self._config.streaming
                and not self._config.allow_streaming_in_artifact_mode
            ):
                raise RuntimeError(
                    "Reproducibility violation: streaming=True in artifact mode adds "
                    "non-determinism. Set allow_streaming_in_artifact_mode=True to "
                    "explicitly opt in, or set streaming=False for exact reruns."
                )
            self._validate_artifact_manifest()
            from datasets import load_from_disk  # type: ignore[import]

            return load_from_disk(self._config.artifact_path)  # type: ignore[return-value]

        # data_source == "hf" — ingestion mode only
        from datasets import load_dataset

        return load_dataset(  # type: ignore[return-value]
            self._config.dataset_id,
            self._config.subset,
            split=self._config.split,
            streaming=self._config.streaming,
            revision=self._config.revision,
            trust_remote_code=True,  # required by pile-of-law custom builder
        )

    def _validate_artifact_manifest(self) -> None:
        """Abort if a present manifest is incompatible with this config.

        The manifest file is optional — artifacts produced before the
        manifest format existed are allowed to load silently. When the
        file *is* present, identity fields (``dataset_id``, ``subset``,
        ``revision``) must match via
        :meth:`ArtifactManifest.is_compatible_with`.

        Raises:
            RuntimeError: The manifest exists and at least one identity
                field differs from the current config.
        """
        import json
        import pathlib

        assert self._config.artifact_path is not None
        manifest_path = pathlib.Path(self._config.artifact_path) / "artifact_manifest.json"
        if not manifest_path.exists():
            return  # manifest absent — warn not fail (may predate manifest)

        from src.dataset_config import ArtifactManifest

        data = json.loads(manifest_path.read_text())
        manifest = ArtifactManifest(**data)
        issues = manifest.is_compatible_with(self._config)
        if issues:
            raise RuntimeError(
                "Artifact manifest incompatible with current config:\n" + "\n".join(f"  - {i}" for i in issues)
            )

    def iter_valid_rows(self, source: Iterable[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
        """Yield only validated, normalized rows.

        Invalid rows are **silently skipped**. Callers that need reject
        counts or per-row failure reasons should use
        :class:`RowValidator` directly rather than this convenience
        method.

        Args:
            source: Optional pre-built row iterable. When ``None``, the
                method calls :meth:`load` internally. Passing an
                explicit source lets callers compose filters without
                re-running the full load path.

        Yields:
            Normalized row dicts with schema-conforming keys.
        """
        rows = source if source is not None else self.load()
        for row in rows:
            if self._validator.validate(row) == []:
                yield self._normalizer.normalize(row)

    def get_provenance(self) -> dict[str, Any]:
        """Return a run-provenance dict for inclusion in experiment metadata.

        The dict captures the exact dataset identity plus the installed
        ``datasets`` library version, so downstream consumers (W&B,
        manifest writer) can later detect silent upstream drift.

        Returns:
            Mapping with keys ``dataset``, ``subset``, ``split``,
            ``revision``, ``hf_datasets_version``, ``reproducible``.
        """
        import datasets

        return {
            "dataset": self._config.dataset_id,
            "subset": self._config.subset,
            "split": self._config.split,
            "revision": self._config.revision,
            "hf_datasets_version": datasets.__version__,
            "reproducible": self._config.reproducible,
        }

    def log_stats(
        self,
        source: Iterable[dict[str, Any]],
        tokenizer: Any,
        max_samples: int = 1000,
    ) -> dict[str, Any]:
        """Compute a pre-training telemetry snapshot for W&B logging.

        Iterates at most ``max_samples`` rows from ``source``, tokenises
        the normalised text of each valid row, and aggregates:

        * token-length summary statistics (min/avg/max + histogram);
        * raw character-length average;
        * court-ID distribution;
        * valid / skipped counts;
        * full provenance (via :meth:`get_provenance`).

        Args:
            source: Row iterable to sample. The caller controls whether
                this is the raw load or an already-filtered stream.
            tokenizer: Anything with an ``encode(str) -> list[int]``
                method. Typically a HF fast tokenizer.
            max_samples: Hard cap on rows consumed from ``source``.
                Defaults to 1000 — enough to characterise the
                distribution without stalling job startup.

        Returns:
            A flat dict combining the computed statistics with the
            provenance fields, safe to pass to ``wandb.log``.
        """
        token_lengths: list[int] = []
        text_lengths: list[int] = []
        court_counts: collections.Counter[str] = collections.Counter()
        skipped = 0

        for i, row in enumerate(source):
            if i >= max_samples:
                break
            if self._validator.validate(row) != []:
                skipped += 1
                continue
            normalized = self._normalizer.normalize(row)
            text = normalized["text"]
            token_lengths.append(len(tokenizer.encode(text)))
            text_lengths.append(len(text))
            court_id = str(row.get("court_id", "unknown"))
            court_counts[court_id] += 1

        n = len(token_lengths)
        stats: dict[str, Any] = {
            "n_valid": n,
            "n_skipped": skipped,
            "avg_token_length": sum(token_lengths) / n if n else 0,
            "min_token_length": min(token_lengths) if n else 0,
            "max_token_length": max(token_lengths) if n else 0,
            "avg_text_length_chars": sum(text_lengths) / n if n else 0,
            "court_distribution": dict(court_counts),
            "token_length_histogram": _histogram(token_lengths),
            "tokenizer_name": getattr(tokenizer, "name_or_path", str(tokenizer)),
            **self.get_provenance(),
        }
        return stats

    def filter_by_date_range(
        self,
        source: Iterable[dict[str, Any]],
        start: str,
        end: str,
        date_field: str = "created_timestamp",
    ) -> Iterator[dict[str, Any]]:
        """Yield rows whose date field falls in ``[start, end]`` inclusive.

        Uses the regex :data:`src.row_normalizer._TS_EXTRACT_RE` to
        extract a ``YYYY-MM-DD`` prefix from whatever timestamp format
        the row carries (the upstream pile-of-law subset uses several).

        Args:
            source: Row iterable to filter.
            start: Lower bound as ``YYYY-MM-DD`` (inclusive).
            end: Upper bound as ``YYYY-MM-DD`` (inclusive).
            date_field: Row key holding the timestamp. Defaults to
                ``"created_timestamp"``.

        Yields:
            Rows whose parsed date is within bounds. Rows with a missing
            or unparseable date field are dropped.
        """
        from src.row_normalizer import _TS_EXTRACT_RE

        for row in source:
            ts = str(row.get(date_field, ""))
            match = _TS_EXTRACT_RE.search(ts)
            if not match:
                continue
            date_part = match.group(0)[:10]
            if start <= date_part <= end:
                yield row

    def filter_by_court(
        self,
        source: Iterable[dict[str, Any]],
        court_ids: list[str],
        court_field: str = "court_id",
    ) -> Iterator[dict[str, Any]]:
        """Yield rows whose court identifier is in the allow-list.

        Args:
            source: Row iterable to filter.
            court_ids: Allow-list of court slugs (e.g. ``["ca9", "cadc"]``).
                Converted to a ``set`` internally for O(1) lookup.
            court_field: Row key holding the court identifier. Defaults
                to ``"court_id"``.

        Yields:
            Rows whose court field stringifies to one of ``court_ids``.
        """
        allowed = set(court_ids)
        for row in source:
            if str(row.get(court_field, "")) in allowed:
                yield row

    def filter_min_text_tokens(
        self,
        source: Iterable[dict[str, Any]],
        min_tokens: int,
        tokenizer: Any,
    ) -> Iterator[dict[str, Any]]:
        """Yield rows whose text has at least ``min_tokens`` tokens.

        Falls back to whitespace-split word count when ``tokenizer`` is
        ``None``, so the helper remains usable in lightweight tests
        that do not want to pay the cost of instantiating a real
        tokenizer.

        Args:
            source: Row iterable to filter.
            min_tokens: Inclusive lower bound on token count.
            tokenizer: Anything with an ``encode(str) -> list[int]``
                method, or ``None`` to use ``str.split``.

        Yields:
            Rows meeting the token-count threshold.
        """
        for row in source:
            text = str(row.get("text", row.get("contents", "")))
            count = len(tokenizer.encode(text)) if tokenizer is not None else len(text.split())
            if count >= min_tokens:
                yield row
