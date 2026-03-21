# src/exceptions.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/src/exceptions.py
# SRP: Custom pipeline exceptions for explicit error handling.


class PipelineError(Exception):
    """Base exception for all pipeline errors."""


class DiscoveryError(PipelineError):
    """S3 bulk file discovery failed."""


class DownloadError(PipelineError):
    """Bulk file download failed."""


class FilterChainError(PipelineError):
    """Court/docket/cluster filter chain failed."""


class ExtractionError(PipelineError):
    """Opinion extraction failed."""


class ValidationError(PipelineError):
    """TDD contract test failed."""


class ManifestError(PipelineError):
    """Manifest read/write/validation failed."""
