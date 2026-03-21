import pytest

pytestmark = pytest.mark.unit

# tests/test_exceptions.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/tests/test_exceptions.py

import pytest

from src.exceptions import (
    DiscoveryError,
    DownloadError,
    ExtractionError,
    FilterChainError,
    ManifestError,
    PipelineError,
    ValidationError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_pipeline_error(self):
        for exc_class in [
            DiscoveryError,
            DownloadError,
            FilterChainError,
            ExtractionError,
            ValidationError,
            ManifestError,
        ]:
            assert issubclass(exc_class, PipelineError)

    def test_catchable_as_pipeline_error(self):
        with pytest.raises(PipelineError):
            raise ValidationError("test")

    def test_message_preserved(self):
        with pytest.raises(ValidationError, match="shard missing"):
            raise ValidationError("shard missing")
