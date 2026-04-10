# src/model_loader.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/model_loader.py
"""Centralised Hugging Face model and tokenizer loaders with safety defaults.

This module is the single place where the project loads model weights
and tokenizers from the Hugging Face Hub or a local cache. Centralising
the load path lets us enforce two project-wide invariants that
``transformers`` itself does not enforce by default:

1. **Safetensors only.** ``use_safetensors=True`` is passed on every
   model load so a ``.bin`` / ``.pt`` pickle file cannot be executed as
   arbitrary code during deserialisation. The Hub serves both formats
   for most popular models; forcing safetensors eliminates the pickle
   RCE attack surface at zero cost to functionality.
2. **Fast tokenizers only.** ``use_fast=True`` on every tokenizer load
   guarantees the Rust-backed tokenizer is used, which is both safer
   (no Python pickle path) and materially faster on the sequence
   lengths this project handles.

Using these wrappers instead of calling ``AutoModel.from_pretrained``
directly means a future audit can grep a single module to verify both
invariants hold across the entire codebase.
"""

from __future__ import annotations

from typing import Any


def load_model(model_name: str, **kwargs: Any) -> Any:
    """Load a Hugging Face model, refusing pickle weight formats.

    Forwards every keyword argument to
    :meth:`transformers.AutoModel.from_pretrained` with
    ``use_safetensors=True`` pinned. A model that only publishes
    ``.bin`` weights on the Hub will raise at load time rather than
    silently deserialising a pickle — this is the intended behaviour.

    Args:
        model_name: Hub repo id (e.g. ``"bert-base-uncased"``) or a
            local directory containing a safetensors checkpoint.
        **kwargs: Any other kwarg accepted by
            :meth:`AutoModel.from_pretrained` (``torch_dtype``,
            ``device_map``, ``revision``, ``cache_dir``, …).

    Returns:
        The loaded model instance. Concrete type depends on the
        checkpoint's architecture.

    Raises:
        OSError: No safetensors weights are available for
            ``model_name``.
    """
    from transformers import AutoModel  # type: ignore[import-untyped]

    return AutoModel.from_pretrained(
        model_name,
        use_safetensors=True,  # refuse .bin/.pt pickle weights
        **kwargs,
    )


def load_tokenizer(model_name: str, **kwargs: Any) -> Any:
    """Load a Hugging Face fast tokenizer for ``model_name``.

    Tokenizer vocab files (``tokenizer.json``, ``vocab.txt``, etc.)
    carry no pickle risk, but ``use_fast=True`` is still forced to
    guarantee the Rust implementation is used throughout the pipeline
    for consistent encoding behaviour and performance.

    Args:
        model_name: Hub repo id or local directory containing the
            tokenizer files.
        **kwargs: Any other kwarg accepted by
            :meth:`AutoTokenizer.from_pretrained`.

    Returns:
        A fast tokenizer instance (subclass of
        :class:`transformers.PreTrainedTokenizerFast`).
    """
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
