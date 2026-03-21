# src/lightning_datamodule.py
# PyTorch IterableDataset + LightningDataModule wrapping DatasetLoader.
import importlib.util
from typing import Any

from src.dataset_config import DatasetConfig
from src.dataset_loader import DatasetLoader

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
LIGHTNING_AVAILABLE = (
    importlib.util.find_spec("lightning") is not None or importlib.util.find_spec("pytorch_lightning") is not None
)


class CourtListenerIterableDataset:
    """
    PyTorch IterableDataset wrapping DatasetLoader.
    Yields tokenized batches ready for a training loop.

    Usage:
        config = DatasetConfig()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        ds = CourtListenerIterableDataset(DatasetLoader(config), tokenizer)
        loader = DataLoader(ds, batch_size=32)
    """

    def __init__(self, loader: DatasetLoader, tokenizer: Any, max_length: int = 512) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for CourtListenerIterableDataset")
        self._loader = loader
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __iter__(self):  # type: ignore[override]
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
    """
    LightningDataModule wrapping CourtListenerIterableDataset.

    Usage (Hydra):
        config = DatasetConfig()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dm = CourtListenerDataModule(config, tokenizer, batch_size=32)
        trainer = Trainer(...)
        trainer.fit(model, datamodule=dm)
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Any,
        batch_size: int = 32,
        num_workers: int = 0,
        max_length: int = 512,
    ) -> None:
        if not LIGHTNING_AVAILABLE:
            raise ImportError("lightning or pytorch_lightning is required for CourtListenerDataModule")
        self._config = config
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._max_length = max_length

    def setup(self, _stage: str | None = None) -> None:
        loader = DatasetLoader(self._config)
        self._dataset = CourtListenerIterableDataset(loader, self._tokenizer, self._max_length)

    def train_dataloader(self):  # type: ignore[override]
        from torch.utils.data import DataLoader  # type: ignore[import]

        return DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def get_provenance(self) -> dict[str, Any]:
        return DatasetLoader(self._config).get_provenance()
