# src/data_contracts.py
"""Statistical data contracts over pipeline manifest stats.

Three contracts with configurable thresholds: row count floor, court
balance (max share of any single court), and text length distribution
floor (mean and p5). Composed by :func:`run_all_contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import polars as pl

FILTER_MIN_CHARS = 100


def valid_record_expr() -> pl.Expr:
    """Canonical filter expression: record passes iff text_length >= FILTER_MIN_CHARS.

    Single-source-of-truth used by EDA, baseline indexers, and data gates.
    """
    return pl.col("text_length") >= FILTER_MIN_CHARS


class DataContractError(RuntimeError):
    """Raised when a data contract fails in strict mode."""


@dataclass(frozen=True)
class ContractResult:
    name: str
    passed: bool
    message: str


def check_row_count_floor(manifest: Dict[str, Any], min_rows: int = 10_000) -> ContractResult:
    """Row count must meet a minimum floor."""
    n = int(manifest.get("num_cases", 0))
    if n >= min_rows:
        return ContractResult("row_count_floor", True, f"{n:,} >= {min_rows:,}")
    return ContractResult("row_count_floor", False, f"{n:,} below floor {min_rows:,}")


def check_court_balance(manifest: Dict[str, Any], max_share: float = 0.5) -> ContractResult:
    """No single court may exceed ``max_share`` of the total."""
    cd = manifest.get("court_distribution", {})
    if not cd:
        return ContractResult("court_balance", False, "no court_distribution in manifest")
    total = sum(cd.values())
    if total == 0:
        return ContractResult("court_balance", False, "zero total rows")
    top_court = max(cd, key=cd.get)
    top_share = cd[top_court] / total
    if top_share <= max_share:
        return ContractResult("court_balance", True, f"top={top_court} share={top_share:.1%}")
    return ContractResult("court_balance", False, f"court {top_court} dominates with {top_share:.1%} > {max_share:.1%}")


def check_text_length_distribution(
    manifest: Dict[str, Any],
    min_mean: int = 1000,
    min_p5: int = 100,
) -> ContractResult:
    """Mean and p5 text length must meet minimum floors."""
    tls = manifest.get("text_length_stats", {})
    if not tls:
        return ContractResult("text_length_distribution", False, "no text_length_stats")
    mean = int(tls.get("mean", 0))
    p5 = int(tls.get("p5", 0))
    if mean >= min_mean and p5 >= min_p5:
        return ContractResult("text_length_distribution", True, f"mean={mean:,} p5={p5:,}")
    return ContractResult(
        "text_length_distribution", False, f"mean={mean} (need >={min_mean}) p5={p5} (need >={min_p5})"
    )


def run_all_contracts(manifest: Dict[str, Any], strict: bool = False) -> List[ContractResult]:
    """Run every contract. If ``strict``, raise on any failure."""
    results = [
        check_row_count_floor(manifest),
        check_court_balance(manifest),
        check_text_length_distribution(manifest),
    ]
    if strict:
        failed = [r for r in results if not r.passed]
        if failed:
            msg = "; ".join(f"{r.name}: {r.message}" for r in failed)
            raise DataContractError(f"data contracts failed: {msg}")
    return results
