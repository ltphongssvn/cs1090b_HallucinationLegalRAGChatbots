"""RE2-based linear-time citation stripper for corpus cleaning.

Designed for the 27GB corpus where eyecite hangs on pathological inputs.
Trade-off vs eyecite:
  - eyecite: extracts case-name spans (plaintiff/defendant), handles
    supra/id chains. Slow, can hang.
  - this stripper: only removes citation strings (volume reporter page).
    Fast, mathematically guaranteed linear-time. No case-name extraction.

Acceptable trade-off for corpus side: we don't need to mask "Brown" the
word from corpus text — only the "347 U.S. 483" reporter strings that
would leak retrieval signal. Case-name leakage is handled query-side
(scripts/clean_query.py uses eyecite on 45K queries, not 7.8M chunks).
"""
from __future__ import annotations

import re2

# Reporter abbreviations covering U.S. federal + state caselaw.
# Bounded patterns (no nested quantifiers) → RE2 linear time.
_REPORTERS = (
    r"U\.\s?S\.|"           # U.S. / U. S.
    r"S\.\s?Ct\.|"          # S.Ct. / S. Ct.
    r"L\.\s?Ed\.\s?2d|"     # L.Ed.2d / L. Ed. 2d
    r"L\.\s?Ed\.|"          # L.Ed.
    r"F\.\s?Supp\.\s?2d|"   # F.Supp.2d / F. Supp. 2d
    r"F\.\s?Supp\.\s?3d|"
    r"F\.\s?Supp\.|"        # F.Supp.
    r"F\.\s?2d|F\.\s?3d|F\.\s?4th|"  # F.2d/3d/4th
    r"P\.\s?2d|P\.\s?3d|"   # P.2d/3d
    r"N\.\s?E\.\s?2d|N\.\s?E\.|"
    r"N\.\s?W\.\s?2d|N\.\s?W\.|"
    r"S\.\s?E\.\s?2d|S\.\s?E\.|"
    r"S\.\s?W\.\s?2d|S\.\s?W\.\s?3d|S\.\s?W\.|"
    r"A\.\s?2d|A\.\s?3d|"
    r"B\.R\.|"              # bankruptcy
    r"Fed\.\s?Cl\.|"
    r"WL|Stat\.|U\.S\.C\.|CFR"
)

# Full citation: 347 U.S. 483 — number, reporter, number
_FULL_CITE = re2.compile(rf"\b\d+\s+(?:{_REPORTERS})\s+\d+(?:[-,]\d+)?")
# Short cite: 347 U.S. at 489
_SHORT_CITE = re2.compile(rf"\b\d+\s+(?:{_REPORTERS})\s+at\s+\d+(?:[-,]\d+)?")
# Page-only short: at 489 (after a citation, contextual; we strip later patterns)
# Id./supra references
_ID_SUPRA = re2.compile(r"\b(?:Id\.|id\.|supra)(?:\s*,\s*(?:at\s+)?\d+(?:[-,]\d+)?)?", re2.IGNORECASE)
# Year parenthetical: (1954), (1954) — strip orphans after citation removal
_YEAR_PAREN = re2.compile(r"\(\s*(?:1[7-9]\d{2}|20\d{2})\s*\)")
# LePaRD masking token
_UNDERSCORE = re2.compile(r"_{4,}")
# Multi-whitespace collapse
_WHITESPACE = re2.compile(r"\s+")


def strip_citations(text: str) -> str:
    """Strip citation strings, supra/id refs, year parentheticals, underscore artifacts.

    Linear time guaranteed by RE2. Idempotent.
    """
    if not text:
        return ""
    # Order matters: short cites first (they contain "at"), then full, then refs
    text = _SHORT_CITE.sub(" ", text)
    text = _FULL_CITE.sub(" ", text)
    text = _ID_SUPRA.sub(" ", text)
    text = _YEAR_PAREN.sub(" ", text)
    text = _UNDERSCORE.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text
