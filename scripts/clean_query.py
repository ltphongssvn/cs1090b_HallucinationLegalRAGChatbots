"""Mask citations from LePaRD destination_context to prevent retrieval leakage.

Strategy
--------
Use eyecite (already in project deps, used by build_lepard_cl_subset.py) to
extract all citation spans, then replace each span with a single space.
This handles:
  - FullCaseCitation: "Brown v. Board, 347 U.S. 483 (1954)"
  - ShortCaseCitation: "347 U.S. at 489"
  - SupraCitation: "Brown, supra, at 489"
  - IdCitation: "Id. at 489"
  - ReferenceCitation: "Brown at 552"

For each citation, eyecite reports a span (start, end). We use those spans
to build the cleaned text. We also extract case-name spans (plaintiff/
defendant) from FullCaseCitation metadata and mask those separately.

Additional cleaning:
  - LePaRD uses 4+ underscores as a passage-mask token; remove these
  - Collapse multi-space runs to a single space
  - Strip leading/trailing whitespace

Why eyecite (not regex)
-----------------------
Eyecite is trained on 55M+ citations, handles parallel citations, supra
chains, and id resolution natively. It's already in the dependency tree
and the canonical CL/CAP citation parser. Regex would be brittle.

Idempotence
-----------
clean_destination_context(clean_destination_context(x)) == clean_destination_context(x).
After first cleaning, no citation tokens remain → second pass is a no-op
on citations; only whitespace normalization runs again with the same fixed
output.
"""
from __future__ import annotations

import re
from pathlib import Path

# 4+ underscores: LePaRD passage masking token
_UNDERSCORE_RUN_RE = re.compile(r"_{4,}")
# Orphaned year-only parenthetical, e.g. " (1954)" left after citation strip.
# Year range 1700-2099 covers all U.S. federal opinions.
_ORPHAN_YEAR_RE = re.compile(r"\(\s*(1[7-9]\d{2}|20\d{2})\s*\)")
# Multi-whitespace collapse
_WHITESPACE_RUN_RE = re.compile(r"\s+")


def clean_destination_context(text: str) -> str:
    """Mask citations + case names + LePaRD artifacts from query text.

    Args:
        text: raw destination_context from LePaRD.

    Returns:
        cleaned text with citation spans removed, case names masked, and
        whitespace normalized. Empty input returns empty string.
    """
    if not text:
        return ""

    # Strip LePaRD underscore artifact first (it can confuse eyecite)
    text = _UNDERSCORE_RUN_RE.sub(" ", text)

    try:
        from eyecite import get_citations
    except ImportError as e:
        raise RuntimeError("eyecite required — run: uv add eyecite") from e

    citations = get_citations(plain_text=text)

    # Collect spans to remove: citation spans + case-name spans
    spans: list[tuple[int, int]] = []
    for cit in citations:
        try:
            span = cit.span()
            if span is not None:
                spans.append(span)
        except Exception:
            continue
        # Also remove plaintiff / defendant name spans if eyecite extracted them.
        # These typically appear immediately before the volume/reporter in the
        # original text and are a primary leakage vector.
        meta = getattr(cit, "metadata", None)
        if meta is None:
            continue
        for attr in ("plaintiff", "defendant"):
            name = getattr(meta, attr, None) or ""
            name = name.strip()
            if not name or len(name) < 2:
                continue
            # Locate every occurrence of the case name in the text. We search
            # globally (not just before the span) because reference citations
            # like "Brown, supra" need the leading "Brown" stripped too.
            start = 0
            while True:
                idx = text.find(name, start)
                if idx < 0:
                    break
                spans.append((idx, idx + len(name)))
                start = idx + len(name)

    if not spans:
        text = _ORPHAN_YEAR_RE.sub(" ", text)
        return _WHITESPACE_RUN_RE.sub(" ", text).strip()

    # Merge overlapping/adjacent spans, then build cleaned text
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    parts: list[str] = []
    cursor = 0
    for s, e in merged:
        if s > cursor:
            parts.append(text[cursor:s])
        parts.append(" ")  # placeholder for the masked span
        cursor = e
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "".join(parts)
    cleaned = _ORPHAN_YEAR_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RUN_RE.sub(" ", cleaned).strip()
    return cleaned


# ---------- batch + JSONL streaming ----------


def clean_destination_context_batch(texts: list[str]) -> list[str]:
    """Clean a list of destination_context strings.

    Pure-Python loop — eyecite has no native batch API. For corpus-scale
    cleaning use clean_jsonl_field which streams I/O.
    """
    return [clean_destination_context(t or "") for t in texts]


def clean_jsonl_field(
    input_path: Path,
    output_path: Path,
    *,
    field: str = "destination_context",
    log_every: int = 100_000,
) -> int:
    """Stream JSONL, clean the named field on each row, write to output.

    Atomic write via .tmp + rename. Returns total rows written.
    """
    import json as _json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    n = 0
    with input_path.open(encoding="utf-8") as fin, tmp.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            row = _json.loads(line)
            if field in row:
                row[field] = clean_destination_context(row[field] or "")
            fout.write(_json.dumps(row) + "\n")
            n += 1
            if n % log_every == 0:
                print(f"[clean_query] {n:,} rows cleaned", flush=True)
    tmp.rename(output_path)
    return n
