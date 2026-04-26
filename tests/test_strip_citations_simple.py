"""TDD for RE2-based linear-time citation stripper.

For corpus-side cleaning where eyecite is too slow. RE2 guarantees
linear-time matching — no catastrophic backtracking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from strip_citations_simple import strip_citations  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "raw,leak",
    [
        ("Brown v. Board, 347 U.S. 483 (1954)", "347 U.S. 483"),
        ("See 71 F. Supp. 2d 990", "71 F. Supp. 2d 990"),
        ("123 F.3d 456 (9th Cir. 1999)", "123 F.3d 456"),
        ("130 S. Ct. 876", "130 S. Ct. 876"),
        ("1 L. Ed. 2d 100", "1 L. Ed. 2d 100"),
        ("As discussed, 347 U.S. at 489", "347 U.S. at 489"),
        ("Smith, 123 F.3d at 460", "123 F.3d at 460"),
        ("500 F.2d 1234", "500 F.2d 1234"),
    ],
)
def test_citation_string_removed(raw: str, leak: str) -> None:
    assert leak not in strip_citations(raw)


@pytest.mark.parametrize(
    "raw,must_keep",
    [
        ("The court must consider both equity and law.", "equity"),
        ("Stare decisis requires courts to follow precedent.", "stare decisis"),
        ("Plaintiff argues breach of contract.", "breach"),
        ("Summary judgment is appropriate.", "summary judgment"),
    ],
)
def test_prose_preserved(raw: str, must_keep: str) -> None:
    assert must_keep.lower() in strip_citations(raw).lower()


def test_empty_string() -> None:
    assert strip_citations("") == ""


def test_idempotent() -> None:
    text = "Brown v. Board, 347 U.S. 483 (1954) is settled law."
    once = strip_citations(text)
    assert strip_citations(once) == once


def test_returns_str() -> None:
    assert isinstance(strip_citations("anything"), str)


def test_linear_time_pathological_input() -> None:
    """720KB of repeating citations must complete in <2s (RE2 linear-time)."""
    import time

    text = "See 347 U.S. 483 (1954). Id. at 489. " * 20000
    t0 = time.perf_counter()
    result = strip_citations(text)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"too slow: {elapsed:.2f}s on {len(text):,} chars"
    assert "347 U.S. 483" not in result


def test_handles_underscore_artifact() -> None:
    """LePaRD ____ masking token is normalized."""
    cleaned = strip_citations("The court held that ____ applies here.")
    assert "____" not in cleaned
