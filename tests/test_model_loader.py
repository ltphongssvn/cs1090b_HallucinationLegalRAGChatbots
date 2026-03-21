# tests/test_model_loader.py
from unittest.mock import MagicMock, patch


def test_load_model_calls_auto_model():
    mock_model = MagicMock()
    with patch("transformers.AutoModel.from_pretrained", return_value=mock_model) as mock_fn:
        from src.model_loader import load_model

        result = load_model("bert-base-uncased")
        mock_fn.assert_called_once_with("bert-base-uncased", use_safetensors=True)
        assert result is mock_model


def test_load_model_passes_kwargs():
    with patch("transformers.AutoModel.from_pretrained") as mock_fn:
        from src.model_loader import load_model

        load_model("bert-base-uncased", trust_remote_code=False)
        mock_fn.assert_called_once_with("bert-base-uncased", use_safetensors=True, trust_remote_code=False)


def test_load_tokenizer_calls_auto_tokenizer():
    mock_tok = MagicMock()
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok) as mock_fn:
        from src.model_loader import load_tokenizer

        result = load_tokenizer("bert-base-uncased")
        mock_fn.assert_called_once_with("bert-base-uncased", use_fast=True)
        assert result is mock_tok


def test_load_tokenizer_passes_kwargs():
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_fn:
        from src.model_loader import load_tokenizer

        load_tokenizer("bert-base-uncased", do_lower_case=True)
        mock_fn.assert_called_once_with("bert-base-uncased", use_fast=True, do_lower_case=True)
