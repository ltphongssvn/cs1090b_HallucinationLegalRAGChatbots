# src/row_normalizer.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/row_normalizer.py
"""Canonical-form normaliser for pre-validated pile-of-law rows.

Turns a row that has already passed :class:`RowValidator` into the
canonical shape the rest of the pipeline expects:

* A single ``text`` key holding the chosen content field, stripped.
* ISO-8601 ``created_timestamp`` and ``downloaded_timestamp`` with a
  trailing ``Z`` when the source indicated UTC.
* A ``source_url`` alias of the row's ``url`` for downstream citation.
* A ``_source_text_field`` debug key recording which upstream column
  the canonical ``text`` came from.

Design notes
------------
* **Validation boundary**: :meth:`RowNormalizer.normalize` re-checks
  validation and raises :class:`ValueError` on failure. This is a
  belt-and-braces guard — callers should already filter via
  :meth:`DatasetLoader.iter_valid_rows` — but makes misuse loud.
* **Timestamp parsing** is deliberately forgiving: a regex first
  extracts the date-or-datetime substring from the raw value, then a
  priority list of ``strptime`` formats decides the output shape.
  Unparseable timestamps become the empty string rather than raising.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from src.dataset_config import DatasetConfig
from src.row_validator import RowValidator

#: Timestamp parse attempts as ``(strptime_format, preserves_time, preserves_tz)``.
#: Tried in order; the first match wins.
_TS_FORMATS: list[tuple[str, bool, bool]] = [
    ("%Y-%m-%dT%H:%M:%S%z", True, True),
    ("%Y-%m-%dT%H:%M:%S", True, False),
    ("%Y-%m-%d", False, False),
]

#: Extracts a YYYY-MM-DD date or an optional ISO-8601 datetime with
#: trailing ``Z``, numeric offset, or fractional seconds.
_TS_EXTRACT_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2}|[+-]\d{4}|\.\d+Z?)?)?")


class RowNormalizer:
    """Project validated rows onto the canonical pipeline row shape.

    The normaliser is stateless with respect to the rows it processes —
    it only holds references to its :class:`DatasetConfig` and the
    validator it shares with :class:`DatasetLoader`. Instances are
    therefore safe to reuse across any number of rows.
    """

    def __init__(self, config: DatasetConfig, validator: RowValidator) -> None:
        """Initialise with the pipeline config and validator.

        Args:
            config: Dataset configuration, used indirectly for the
                validator's field lists.
            validator: The same :class:`RowValidator` instance the
                caller uses to filter rows; sharing ensures validate
                and normalize decisions stay in lock-step.
        """
        self._config = config
        self._validator = validator

    def normalize(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return a canonicalised copy of ``row``.

        Re-validates defensively; picks the highest-priority text
        field via :meth:`RowValidator.resolve_text_field`; copies it
        into ``text`` (dropping the original key if it differed);
        normalises both timestamps; and aliases ``url`` as
        ``source_url``.

        Args:
            row: A raw pile-of-law row that has already passed
                :meth:`RowValidator.validate`.

        Returns:
            A new dict ready for sharding or tokenisation.

        Raises:
            ValueError: ``row`` fails revalidation. The message lists
                every validation error and points to
                :meth:`DatasetLoader.iter_valid_rows` as the fix.
        """
        errors = self._validator.validate(row)
        if errors:
            raise ValueError(
                f"normalize() called on invalid row — validate first.\n"
                f"Errors: {errors}\n"
                f"Use DatasetLoader.iter_valid_rows() to enforce this automatically."
            )
        normalized = dict(row)
        text_field = self._validator.resolve_text_field(row)
        assert text_field is not None  # guaranteed by validate()
        text = str(row[text_field]).strip()
        normalized["_source_text_field"] = text_field
        if text_field != "text":
            normalized.pop(text_field, None)
        normalized["text"] = text
        normalized["created_timestamp"] = self._normalize_timestamp(str(row.get("created_timestamp", "")))
        normalized["downloaded_timestamp"] = self._normalize_timestamp(str(row.get("downloaded_timestamp", "")))
        normalized["source_url"] = str(row["url"])
        return normalized

    def _normalize_timestamp(self, ts: str) -> str:
        """Parse ``ts`` into a canonical ISO-8601 string, or ``""`` on failure.

        Extracts the first date-or-datetime substring via
        :data:`_TS_EXTRACT_RE`, normalises a trailing ``Z`` to
        ``+00:00`` so :func:`datetime.strptime` can handle it, then
        walks :data:`_TS_FORMATS` in priority order. UTC datetimes are
        emitted with a trailing ``Z``; other offsets round-trip via
        :meth:`datetime.isoformat`; dates without a time stay as
        ``YYYY-MM-DD``.

        Returns:
            The canonical string, or ``""`` if nothing matched.
        """
        candidate = _TS_EXTRACT_RE.search(ts)
        if not candidate:
            return ""
        raw = candidate.group(0).replace("Z", "+00:00")
        for fmt, preserves_time, preserves_tz in _TS_FORMATS:
            try:
                parsed = datetime.strptime(raw, fmt)
                if preserves_tz and parsed.tzinfo is not None:
                    if parsed.tzinfo == timezone.utc:
                        return parsed.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
                    return parsed.isoformat()
                if preserves_time:
                    return parsed.strftime("%Y-%m-%dT%H:%M:%S")
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return ""
