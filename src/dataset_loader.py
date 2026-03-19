# src/dataset_loader.py
# Single-responsibility: load and stream validated rows from HF Hub.
import re
from typing import Any, Iterable, Iterator

from src.dataset_config import DatasetConfig
from src.row_normalizer import RowNormalizer
from src.row_validator import RowValidator

HEX_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_MUTABLE_REFS = {"main", "master", "latest", "HEAD", ""}


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
