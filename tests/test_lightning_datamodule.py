# tests/test_lightning_datamodule.py
from unittest.mock import MagicMock, patch

import pytest

import src.lightning_datamodule as ldm


@pytest.fixture
def mock_config():
    from src.dataset_config import DatasetConfig

    return DatasetConfig()


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    encoding = MagicMock()
    encoding.__getitem__ = lambda self, key: MagicMock()
    tok.return_value = encoding
    return tok


# --- CourtListenerIterableDataset ---


def test_dataset_init_raises_if_no_torch():
    with patch.object(ldm, "TORCH_AVAILABLE", False):
        with pytest.raises(ImportError, match="torch is required"):
            ldm.CourtListenerIterableDataset(MagicMock(), MagicMock())


def test_dataset_init_succeeds_with_torch(mock_tokenizer):
    with patch.object(ldm, "TORCH_AVAILABLE", True):
        loader = MagicMock()
        ds = ldm.CourtListenerIterableDataset(loader, mock_tokenizer, max_length=128)
        assert ds._max_length == 128
        assert ds._loader is loader
        assert ds._tokenizer is mock_tokenizer


def test_dataset_iter_yields_batches():
    with patch.object(ldm, "TORCH_AVAILABLE", True):
        import torch

        loader = MagicMock()
        loader.iter_valid_rows.return_value = [
            {"text": "some legal text", "url": "http://a.com", "source_url": "http://b.com"}
        ]
        tok = MagicMock()
        tensor = torch.zeros(512, dtype=torch.long)
        encoding = {"input_ids": tensor.unsqueeze(0), "attention_mask": tensor.unsqueeze(0)}
        tok.return_value = encoding
        ds = ldm.CourtListenerIterableDataset(loader, tok, max_length=512)
        batches = list(ds)
        assert len(batches) == 1
        assert "input_ids" in batches[0]
        assert "attention_mask" in batches[0]
        assert batches[0]["url"] == "http://a.com"


# --- CourtListenerDataModule ---


def test_datamodule_init_raises_if_no_lightning(mock_config, mock_tokenizer):
    with patch.object(ldm, "LIGHTNING_AVAILABLE", False):
        with pytest.raises(ImportError, match="lightning or pytorch_lightning is required"):
            ldm.CourtListenerDataModule(mock_config, mock_tokenizer)


def test_datamodule_init_succeeds(mock_config, mock_tokenizer):
    with patch.object(ldm, "LIGHTNING_AVAILABLE", True):
        with patch.object(ldm, "TORCH_AVAILABLE", True):
            dm = ldm.CourtListenerDataModule(mock_config, mock_tokenizer, batch_size=16)
            assert dm._batch_size == 16


def test_datamodule_setup_creates_dataset(mock_config, mock_tokenizer):
    with patch.object(ldm, "LIGHTNING_AVAILABLE", True):
        with patch.object(ldm, "TORCH_AVAILABLE", True):
            with patch("src.lightning_datamodule.DatasetLoader"):
                dm = ldm.CourtListenerDataModule(mock_config, mock_tokenizer)
                dm.setup()
                assert hasattr(dm, "_dataset")
                assert isinstance(dm._dataset, ldm.CourtListenerIterableDataset)


def test_datamodule_train_dataloader(mock_config, mock_tokenizer):
    with patch.object(ldm, "LIGHTNING_AVAILABLE", True):
        with patch.object(ldm, "TORCH_AVAILABLE", True):
            with patch("src.lightning_datamodule.DatasetLoader"):
                with patch("torch.utils.data.DataLoader") as MockDL:
                    dm = ldm.CourtListenerDataModule(mock_config, mock_tokenizer)
                    dm.setup()
                    dm.train_dataloader()
                    MockDL.assert_called_once()


def test_datamodule_get_provenance(mock_config, mock_tokenizer):
    with patch.object(ldm, "LIGHTNING_AVAILABLE", True):
        with patch.object(ldm, "TORCH_AVAILABLE", True):
            with patch("src.lightning_datamodule.DatasetLoader") as MockLoader:
                MockLoader.return_value.get_provenance.return_value = {"source": "test"}
                dm = ldm.CourtListenerDataModule(mock_config, mock_tokenizer)
                prov = dm.get_provenance()
                assert prov == {"source": "test"}
