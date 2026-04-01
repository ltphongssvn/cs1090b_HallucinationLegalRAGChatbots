"""Stub — implementation pending. Tests must fail RED before implementation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

_STRING_NAN_VALUES: frozenset[str] = frozenset()
_NAN_REPAIR_PATTERN = re.compile(r"(?!)")  # never matches — stub


@dataclass(frozen=True)
class ShardHealth:
    shard: str
    total_lines: int
    nan_lines: int
    nan_fields: dict[str, int]


@dataclass(frozen=True)
class DatasetHealth:
    total_lines: int
    nan_lines: int
    nan_shards: int
    total_shards: int
    nan_fields: dict[str, int]
    contaminated_shards: list[str]

    @property
    def clean_pct(self) -> float:
        raise NotImplementedError

    def gate_verdict(self) -> str:
        raise NotImplementedError


def _has_nan(value: Any) -> bool:
    raise NotImplementedError


def _nan_fields(obj: dict[str, Any]) -> list[str]:
    raise NotImplementedError


def audit_shard(shard_path: Path) -> ShardHealth:
    raise NotImplementedError


def audit_dataset(input_dir: Path) -> DatasetHealth:
    raise NotImplementedError


def repair_shard(shard_path: Path, dry_run: bool = False) -> tuple[int, int]:
    raise NotImplementedError
