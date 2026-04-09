# src/exceptions.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/exceptions.py
"""Custom exception hierarchy for the CourtListener ingest pipeline.

Every pipeline stage raises a subclass of :class:`PipelineError` rather
than a bare :class:`Exception`, so callers can catch pipeline failures
without accidentally swallowing unrelated errors (``KeyboardInterrupt``,
``ImportError``, test-framework assertions, etc.).

The hierarchy mirrors the pipeline stages in execution order:

    PipelineError
    ├── DiscoveryError       — src.s3_discovery
    ├── DownloadError        — src.bulk_download
    ├── FilterChainError     — src.filter_chain
    ├── ExtractionError      — src.extract
    ├── ValidationError      — src.validation / TDD contract tests
    └── ManifestError        — src.manifest / src.manifest_collector

Design note: these classes carry no extra state. Rich failure context
(row numbers, file paths, S3 keys) should be passed as the exception
message so it surfaces unchanged in logs and manifests.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every pipeline-stage failure.

    Catch this to handle any pipeline error generically (e.g. in a
    top-level CLI wrapper); catch a concrete subclass to handle a
    specific stage's failure differently.
    """


class DiscoveryError(PipelineError):
    """Raised when S3 bulk-file discovery fails.

    Typical causes: the bucket is unreachable, the expected filename
    prefix returns zero objects, or the listing is truncated past the
    configured page limit.
    """


class DownloadError(PipelineError):
    """Raised when a bulk-file download fails on both the CLI and HTTP paths.

    See :mod:`src.bulk_download` for the two-tier strategy. This
    exception only surfaces once *both* paths have been exhausted.
    """


class FilterChainError(PipelineError):
    """Raised when the court → docket → cluster → opinion join fails.

    Indicates an upstream schema mismatch or a missing join key, not a
    transient I/O problem.
    """


class ExtractionError(PipelineError):
    """Raised when per-row opinion-text extraction fails irrecoverably.

    Per-row failures normally route to the quarantine file; this
    exception is reserved for failures that invalidate the entire
    extraction (e.g. the opinions CSV is truncated mid-row).
    """


class ValidationError(PipelineError):
    """Raised when a TDD contract test or post-run invariant fails.

    Distinct from per-row validation rejection, which is a *data*
    condition and does not raise. This exception signals a *pipeline*
    condition (e.g. total row count below :attr:`PipelineConfig.min_expected_total`).
    """


class ManifestError(PipelineError):
    """Raised when the run manifest cannot be written, read, or parsed.

    A healthy run always produces a manifest; any failure to do so is
    treated as a pipeline failure because downstream consumers rely on
    the manifest for provenance.
    """
