# src/row_normalizer.py
# Single-responsibility: normalize validated rows into canonical pipeline form.
import re
from datetime import datetime, timezone
from typing import Any

from src.dataset_config import DatasetConfig
from src.row_validator import RowValidator

_TS_FORMATS: list[tuple[str, bool, bool]] = [
    ("%Y-%m-%dT%H:%M:%S%z", True, True),
    ("%Y-%m-%dT%H:%M:%S", True, False),
    ("%Y-%m-%d", False, False),
]

_TS_EXTRACT_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2}|[+-]\d{4}|\.\d+Z?)?)?"
)


class RowNormalizer:
    """Normalizes pre-validated rows into canonical pipeline form."""

    def __init__(self, config: DatasetConfig, validator: RowValidator) -> None:
        self._config = config
        self._validator = validator

    def normalize(self, row: dict[str, Any]) -> dict[str, Any]:
        """Normalize a pre-validated row. Raises ValueError if validation fails."""
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
