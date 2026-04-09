# src/lightning_datamodule.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/lightning_datamodule.py
"""PyTorch / Lightning adapters for the pile-of-law :class:`DatasetLoader`.

Two thin wrappers that turn the validator-filtered row stream from
:class:`src.dataset_loader.DatasetLoader` into the shapes the training
stack expects:

* :class:`CourtListenerIterableDataset` — a PyTorch ``IterableDataset``
  that tokenises each row on the fly and yields a dict of tensors plus
  provenance URLs.
* :class:`CourtListenerDataModule` — a :class:`lightning.LightningDataModule`
  that builds a :class:`torch.utils.data.DataLoader` over the iterable
  dataset so a :class:`lightning.Trainer` can consume it directly.

Both classes import their heavyweight dependencies (``torch``,
``lightning``) lazily via :mod:`importlib.util` so this module stays
importable on CI and documentation builds that do not ship a GPU stack.
``ImportError`` is raised only when a caller actually instantiates one
of the wrappers without the required library installed.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from src.dataset_config import DatasetConfig
from src.dataset_loader import DatasetLoader

#: ``True`` if :mod:`torch` is importable in the current environment.
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

#: ``True`` if either the new ``lightning`` package or the legacy
#: ``pytorch_lightning`` package is importable. Both are accepted.
LIGHTNING_AVAILABLE = (
    importlib.util.find_spec("lightning") is not None or importlib.util.find_spec("pytorch_lightning") is not None
)


class CourtListenerIterableDataset:
    """PyTorch-compatible iterable dataset over validated pile-of-law rows.

    Wraps a :class:`DatasetLoader` and a tokenizer. Each iteration step
    pulls the next validated, normalised row, tokenises its text field
    with fixed-length padding/truncation, and yields a dict containing
    the two model inputs (``input_ids``, ``attention_mask``) plus the
    row's ``url`` and ``source_url`` for provenance tracking through
    the training loop.

    Example:
        >>> config = DatasetConfig()
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> ds = CourtListenerIterableDataset(DatasetLoader(config), tokenizer)
        >>> loader = DataLoader(ds, batch_size=32)
    """

    def __init__(self, loader: DatasetLoader, tokenizer: Any, max_length: int = 512) -> None:
        """Initialise the iterable dataset.

        Args:
            loader: A configured :class:`DatasetLoader`. Its
                :meth:`~DatasetLoader.iter_valid_rows` drives iteration.
            tokenizer: Anything with an ``__call__`` signature
                compatible with HF ``PreTrainedTokenizer`` (kwargs
                ``max_length``, ``truncation``, ``padding``,
                ``return_tensors``).
            max_length: Sequence length passed to the tokenizer.
                Defaults to 512, matching BERT-family limits.

        Raises:
            ImportError: :mod:`torch` is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for CourtListenerIterableDataset")
        self._loader = loader
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __iter__(self):  # type: ignore[override]
        """Yield one tokenised example per validated row.

        Each yielded dict contains ``input_ids`` and ``attention_mask``
        as 1-D tensors (batch dim squeezed), plus the row's ``url`` and
        ``source_url`` strings so downstream consumers can cite the
        exact provenance of every sample in a training batch.
        """
        for row in self._loader.iter_valid_rows():
            encoding = self._tokenizer(
                row["text"],
                max_length=self._max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            yield {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "url": row["url"],
                "source_url": row["source_url"],
            }


class CourtListenerDataModule:
    """Lightning :class:`DataModule` wrapping :class:`CourtListenerIterableDataset`.

    Hydra-friendly construction: all knobs are plain kwargs so a
    ``configs/data/legal_rag.yaml`` can instantiate the module directly.
    The :class:`DatasetLoader` is built lazily inside :meth:`setup` so
    the constructor itself performs no I/O — a property Lightning
    requires for distributed training setup.

    Example:
        >>> config = DatasetConfig()
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> dm = CourtListenerDataModule(config, tokenizer, batch_size=32)
        >>> trainer = Trainer(...)
        >>> trainer.fit(model, datamodule=dm)
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Any,
        batch_size: int = 32,
        num_workers: int = 0,
        max_length: int = 512,
    ) -> None:
        """Initialise the data module.

        Args:
            config: Dataset configuration consumed by the inner
                :class:`DatasetLoader`.
            tokenizer: HF-compatible tokenizer.
            batch_size: Per-device batch size for the produced
                :class:`DataLoader`.
            num_workers: DataLoader worker count. Defaults to 0
                because the underlying stream is single-producer; set
                >0 only after switching to a shardable source.
            max_length: Sequence length forwarded to
                :class:`CourtListenerIterableDataset`.

        Raises:
            ImportError: Neither ``lightning`` nor ``pytorch_lightning``
                is installed.
        """
        if not LIGHTNING_AVAILABLE:
            raise ImportError("lightning or pytorch_lightning is required for CourtListenerDataModule")
        self._config = config
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._max_length = max_length

    def setup(self, _stage: str | None = None) -> None:
        """Build the :class:`DatasetLoader` and wrap it for iteration.

        Lightning calls this on every rank after process-group init,
        which is exactly when we want the loader to materialise — any
        reproducibility guard failures surface here rather than at
        module construction.
        """
        loader = DatasetLoader(self._config)
        self._dataset = CourtListenerIterableDataset(loader, self._tokenizer, self._max_length)

    def train_dataloader(self):  # type: ignore[override]
        """Return a :class:`torch.utils.data.DataLoader` over the iterable dataset."""
        from torch.utils.data import DataLoader  # type: ignore[import]

        return DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def get_provenance(self) -> dict[str, Any]:
        """Return the underlying loader's provenance dict for run-metadata logging."""
        return DatasetLoader(self._config).get_provenance()
