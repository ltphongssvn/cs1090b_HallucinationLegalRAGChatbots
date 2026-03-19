# src/model_loader.py
# Centralized model loading — always uses safetensors to avoid pickle RCE.
from typing import Any


def load_model(model_name: str, **kwargs: Any) -> Any:
    """Load a HuggingFace model enforcing safetensors format."""
    from transformers import AutoModel  # type: ignore[import-untyped]

    return AutoModel.from_pretrained(
        model_name,
        use_safetensors=True,  # refuse .bin/.pt pickle weights
        **kwargs,
    )


def load_tokenizer(model_name: str, **kwargs: Any) -> Any:
    """Load tokenizer — vocab files only, no weight pickle risk."""
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
