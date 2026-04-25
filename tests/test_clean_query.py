"""TDD tests for query cleaning.

Cleaning contract:
  - Mask FullCaseCitation, ShortCaseCitation, SupraCitation, IdCitation,
    ReferenceCitation spans (eyecite-detected) → replace with single space
  - Strip case-name spans (plaintiff/defendant) when extracted by eyecite
  - Strip LePaRD masking artifacts (4+ underscores)
  - Strip orphan year parentheticals (1700-2099)
  - Idempotent: clean(clean(x)) == clean(x)
  - Deterministic: clean(x) is reproducible across runs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clean_query import clean_destination_context  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def cleaner():
    return clean_destination_context


# ----- parametrized: citation strings must NOT survive ----------------------


@pytest.mark.parametrize(
    "raw,leak_token",
    [
        # U.S. Reports
        ("Brown v. Board, 347 U.S. 483 (1954)", "347 U.S. 483"),
        ("Roe v. Wade, 410 U.S. 113", "410 U.S. 113"),
        ("Miranda v. Arizona, 384 U.S. 436 (1966)", "Miranda"),
        # F. Supp.
        ("See 71 F. Supp. 2d 990", "F. Supp. 2d"),
        # F.3d
        ("United States v. Smith, 123 F.3d 456 (9th Cir. 1999)", "123 F.3d 456"),
        ("Doe v. Roe, 999 F.3d 100 (1st Cir. 2020)", "999 F.3d 100"),
        # F.2d
        ("Adams Indus. v. Corp., 500 F.2d 1234 (2d Cir. 1974)", "500 F.2d 1234"),
        # S. Ct.
        ("Citizens United v. FEC, 558 U.S. 310, 130 S. Ct. 876 (2010)", "130 S. Ct. 876"),
        # L. Ed.
        ("Doe v. Roe, 1 L. Ed. 2d 100", "1 L. Ed. 2d 100"),
        # In re / Ex parte
        ("In re Gault, 387 U.S. 1 (1967)", "387 U.S. 1"),
        ("Ex parte Young, 209 U.S. 123 (1908)", "209 U.S. 123"),
        # Short forms
        ("As discussed, 347 U.S. at 489", "347 U.S."),
        ("Cf. 410 U.S. at 116", "410 U.S."),
        ("Smith, 123 F.3d at 460", "123 F.3d"),
        # Id. / supra
        ("The Court held that... Id. at 489.", "Id."),
        ("see id. at 12", "id."),
        ("Brown, supra, at 489", "supra"),
    ],
)
def test_citation_token_removed(cleaner, raw: str, leak_token: str) -> None:
    cleaned = cleaner(raw)
    assert leak_token not in cleaned, f"leakage: {leak_token!r} in {cleaned!r}"


# ----- parametrized: legal prose without citations preserved ---------------


@pytest.mark.parametrize(
    "raw,must_keep",
    [
        ("The court must consider both equity and law.", "equity"),
        ("Stare decisis requires courts to follow precedent.", "stare decisis"),
        ("Plaintiff argues breach of contract.", "breach"),
        ("Summary judgment is appropriate when no genuine issue exists.", "summary judgment"),
    ],
)
def test_non_citation_content_preserved(cleaner, raw: str, must_keep: str) -> None:
    cleaned = cleaner(raw)
    assert must_keep.lower() in cleaned.lower()


# ----- artifact normalization ----------------------------------------------


def test_strips_lepard_underscore_artifact(cleaner) -> None:
    assert "____" not in cleaner("The court held that ____ applies here.")


def test_strips_long_underscore_run(cleaner) -> None:
    assert "____" not in cleaner("ruling _________________________ binding")


# ----- exact-equality contract --------------------------------------------


def test_empty_string_returns_empty(cleaner) -> None:
    assert cleaner("") == ""


def test_only_underscores_returns_empty(cleaner) -> None:
    assert cleaner("_______").strip() == ""


def test_returns_str(cleaner) -> None:
    assert isinstance(cleaner("any input"), str)


# ----- determinism + idempotence ------------------------------------------


def test_idempotent(cleaner) -> None:
    text = "Plaintiff argues that Brown v. Board, 347 U.S. 483 (1954), controls."
    assert cleaner(cleaner(text)) == cleaner(text)


def test_deterministic_across_calls(cleaner) -> None:
    text = "See Brown v. Board, 347 U.S. 483 (1954)."
    assert cleaner(text) == cleaner(text)


# ----- canonical multi-leak end-to-end ------------------------------------


def test_canonical_example_strips_all_known_leak_vectors(cleaner) -> None:
    text = "See Brown v. Board, 347 U.S. 483 (1954) for further discussion."
    cleaned = cleaner(text)
    for leak in ("Brown", "Board", "347 U.S. 483", "1954"):
        assert leak not in cleaned, f"leak {leak!r} survived in {cleaned!r}"
    assert "discussion" in cleaned.lower()


# ----- property-based fuzz ------------------------------------------------


@pytest.mark.property
@given(
    text=st.text(
        alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
        min_size=0,
        max_size=500,
    )
)
@settings(max_examples=50, deadline=None)
def test_property_never_raises(cleaner, text: str) -> None:
    assert isinstance(cleaner(text), str)


@pytest.mark.property
@given(
    text=st.text(
        alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
        min_size=0,
        max_size=200,
    )
)
@settings(max_examples=30, deadline=None)
def test_property_idempotent_on_random(cleaner, text: str) -> None:
    assert cleaner(cleaner(text)) == cleaner(text)


# ----- batch API for corpus-level cleaning (Tier 2) -----------------------


def test_clean_destination_context_batch_exists() -> None:
    """For corpus-level cleaning we need a batch API on top of the unit fn."""
    from clean_query import clean_destination_context_batch

    inputs = [
        "Brown v. Board, 347 U.S. 483 (1954)",
        "Plain prose without citations.",
        "",
    ]
    out = clean_destination_context_batch(inputs)
    assert len(out) == 3
    assert "347 U.S." not in out[0]
    assert "Plain prose" in out[1]
    assert out[2] == ""


def test_clean_destination_context_batch_streams_jsonl(tmp_path: Path) -> None:
    """Batch should be able to stream JSONL → JSONL with field replacement."""
    from clean_query import clean_jsonl_field

    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(
        json.dumps({"source_id": 1, "destination_context": "Brown v. Board, 347 U.S. 483 (1954)"})
        + "\n"
        + json.dumps({"source_id": 2, "destination_context": "Plain text."})
        + "\n"
    )
    n = clean_jsonl_field(inp, out, field="destination_context")
    assert n == 2
    rows = [json.loads(line) for line in out.open()]
    assert "347 U.S." not in rows[0]["destination_context"]
    assert rows[0]["source_id"] == 1
    assert rows[1]["destination_context"].strip() != ""
