# src/row_validator.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/row_validator.py
"""Schema-contract validator for pile-of-law rows.

Holds the row-level "is this usable?" decision for the pipeline. A row
is valid iff:

1. Every name in :attr:`DatasetConfig.required_fields` is present as a
   dict key.
2. At least one name in :attr:`DatasetConfig.text_fields` is present,
   its value is a :class:`str`, and that string is at least
   :attr:`DatasetConfig.min_text_length` characters long.

The validator is pure: it never mutates the row and never performs
I/O. All errors are returned as a list so callers can report every
problem at once instead of only the first.
"""

from __future__ import annotations

from typing import Any

from src.dataset_config import DatasetConfig


class RowValidator:
    """Validate rows against a :class:`DatasetConfig` schema contract.

    Instances are stateless with respect to rows and safe to reuse
    across any number of validate calls. Shared between
    :class:`DatasetLoader` and :class:`RowNormalizer` so the "valid"
    and "normalisable" predicates can never drift.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Initialise with the pipeline configuration.

        Args:
            config: Dataset configuration supplying
                ``required_fields``, ``text_fields``, and
                ``min_text_length``.
        """
        self._config = config

    def validate(self, row: dict[str, Any]) -> list[str]:
        """Return every validation error for ``row``; empty list means valid.

        Checks, in order: required-field presence, text-field
        resolution, text-field type, text-field length. Missing the
        text field short-circuits later checks (no point measuring
        something that does not exist), but missing required fields
        does not — callers benefit from seeing both problems in one
        report.

        Args:
            row: A raw pile-of-law row.

        Returns:
            A list of human-readable error strings. An empty list
            indicates the row is ready for :class:`RowNormalizer`.
        """
        errors: list[str] = []
        missing = self._config.required_fields - set(row.keys())
        if missing:
            errors.append(f"Missing required fields: {sorted(missing)}")
        text_field = self.resolve_text_field(row)
        if text_field is None:
            errors.append(f"No text field found in {sorted(row.keys())}")
            return errors
        value = row[text_field]
        if not isinstance(value, str):
            errors.append(f"{text_field} must be str, got {type(value).__name__!r}: {value!r}")
            return errors
        if len(value) < self._config.min_text_length:
            errors.append(f"{text_field} too short: {len(value)} < {self._config.min_text_length}")
        return errors

    def resolve_text_field(self, row: dict[str, Any]) -> str | None:
        """Return the first :attr:`DatasetConfig.text_fields` entry present in ``row``.

        Iteration order matches the config's declared priority, so
        e.g. ``"text"`` beats ``"contents"`` when both are present.

        Returns:
            The resolved field name, or ``None`` if no candidate key
            is present in the row.
        """
        return next((k for k in self._config.text_fields if k in row), None)
