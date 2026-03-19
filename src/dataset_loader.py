# src/dataset_loader.py
import collections
import re
from typing import Any, Iterable, Iterator

from src.dataset_config import DatasetConfig
from src.row_normalizer import RowNormalizer
from src.row_validator import RowValidator

HEX_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_MUTABLE_REFS = {"main", "master", "latest", "HEAD", ""}


def _histogram(values: list[int], bins: int = 10) -> list[dict[str, Any]]:
    """Return a simple histogram as a list of {range, count} dicts."""
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
    """Loads, validates, and streams normalized rows from a pile-of-law subset."""

    def __init__(self, config: DatasetConfig) -> None:
        self._config = config
        self._validator = RowValidator(config)
        self._normalizer = RowNormalizer(config, self._validator)

    def load(self) -> Iterable[dict[str, Any]]:
        """Load dataset at pinned revision. Raises RuntimeError if revision is mutable."""
        if self._config.reproducible and (
            self._config.revision in _MUTABLE_REFS or HEX_REVISION_RE.fullmatch(self._config.revision) is None
        ):
            raise RuntimeError(
                f"Reproducibility violation: revision={self._config.revision!r} is mutable. "
                "Set reproducible=False to opt into exploration mode."
            )

        from datasets import load_dataset

        return load_dataset(  # type: ignore[return-value]
            self._config.dataset_id,
            self._config.subset,
            split=self._config.split,
            streaming=self._config.streaming,
            revision=self._config.revision,
        )

    def iter_valid_rows(self, source: Iterable[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
        """Yield only validated, normalized rows. Invalid rows silently skipped."""
        rows = source if source is not None else self.load()
        for row in rows:
            if self._validator.validate(row) == []:
                yield self._normalizer.normalize(row)

    def get_provenance(self) -> dict[str, Any]:
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
        """Compute dataset statistics for W&B logging before training.

        Computes: token length histogram, avg text length, court distribution.
        Call once before any training run — log result to W&B via wandb.log().

        Args:
            source: iterable of raw (pre-validation) rows
            tokenizer: tokenizer used for fine-tuning (must match training tokenizer)
            max_samples: cap samples to keep this fast on large streaming datasets
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
        """Yield rows whose date_field falls within [start, end] (YYYY-MM-DD inclusive)."""
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
        """Yield rows whose court_field is in court_ids."""
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
        """Yield rows whose text has at least min_tokens tokens (whitespace-split approximation
        used when tokenizer is None; otherwise uses tokenizer.encode)."""
        for row in source:
            text = str(row.get("text", row.get("contents", "")))
            count = len(tokenizer.encode(text)) if tokenizer is not None else len(text.split())
            if count >= min_tokens:
                yield row
