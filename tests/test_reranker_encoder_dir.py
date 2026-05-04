# tests/test_reranker_encoder_dir.py
"""Tests that baseline_reranker accepts --encoder-dir to load fine-tuned weights."""

import pytest


@pytest.fixture
def reranker_module():
    from scripts import baseline_reranker

    return baseline_reranker


@pytest.mark.contract
class TestEncoderDirSupport:
    def test_default_encoder_dir_constant(self, reranker_module):
        assert hasattr(reranker_module, "DEFAULT_ENCODER_DIR")
        # Default = None means use RERANKER_MODEL from HF hub
        assert reranker_module.DEFAULT_ENCODER_DIR is None

    def test_main_accepts_encoder_dir_kwarg(self, reranker_module):
        import inspect

        sig = inspect.signature(reranker_module.main)
        assert "encoder_dir" in sig.parameters

    def test_arg_parser_has_encoder_dir(self, reranker_module):
        ap = reranker_module._build_arg_parser()
        # Try parsing with --encoder-dir flag
        args = ap.parse_args(["--encoder-dir", "/tmp/some/path"])
        assert str(args.encoder_dir) == "/tmp/some/path"
