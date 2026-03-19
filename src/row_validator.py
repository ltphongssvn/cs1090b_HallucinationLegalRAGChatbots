# src/row_validator.py
# Single-responsibility: validate dataset rows against schema contract.
from typing import Any

from src.dataset_config import DatasetConfig


class RowValidator:
    """Validates rows against a DatasetConfig schema contract."""

    def __init__(self, config: DatasetConfig) -> None:
        self._config = config

    def validate(self, row: dict[str, Any]) -> list[str]:
        """Return all validation errors. Empty list = valid."""
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
        return next((k for k in self._config.text_fields if k in row), None)
