# src/dataset_probe.py
# Path: cs1090b_HallucinationLegalRAGChatbots/src/dataset_probe.py
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

HEX_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_MUTABLE_REFS = {"main", "master", "latest", "HEAD", ""}

_TS_FORMATS: list[tuple[str, bool, bool]] = [
    ("%Y-%m-%dT%H:%M:%S%z", True, True),
    ("%Y-%m-%dT%H:%M:%S", True, False),
    ("%Y-%m-%d", False, False),
]

_TS_EXTRACT_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2}|[+-]\d{4}|\.\d+Z?)?)?"
)

# Pinned to HEAD commit of pile-of-law/pile-of-law as of 2026-03-18.
# Update with:
#   from huggingface_hub import list_repo_commits
#   list(list_repo_commits('pile-of-law/pile-of-law', repo_type='dataset'))[0].commit_id
PINNED_REVISION = "0dc9f2c26b42af4cb6330f36d6146e82f9117a3b"  # pragma: allowlist secret


class CourtListenerDatasetProbe:
    """
    Schema contract and access layer for pile-of-law/pile-of-law
    subset r_courtlistener_opinions.

    Reproducibility contract:
      - REPRODUCIBLE=True (default) enforces a pinned 40-char SHA at load() time.
        Set REPRODUCIBLE=False only for fast exploration — never for training runs.
      - trust_remote_code is never passed.
      - Provenance is probe-level — call get_provenance() once at training start.

    validate_row / normalize_row contract:
      - validate_row() returns all errors; empty list means valid.
      - normalize_row() requires a pre-validated row. Raises ValueError if
        validation fails — callers cannot accidentally normalize invalid rows.
      - iter_valid_rows() is the preferred pipeline entry point.

    REQUIRED_FIELDS intentionally excludes text-variant keys ('text', 'contents').
    Text field presence and type are enforced separately via resolve_text_field().
    """

    DATASET_ID = "pile-of-law/pile-of-law"
    SUBSET = "r_courtlistener_opinions"
    SPLIT = "train"
    REVISION = PINNED_REVISION
    PROBE_VERSION = "1.0"
    REPRODUCIBLE = True
    REQUIRED_FIELDS: frozenset[str] = frozenset({"created_timestamp", "downloaded_timestamp", "url"})
    TEXT_FIELDS: tuple[str, ...] = ("text", "contents")
    MIN_TEXT_LENGTH = 50

    def load(self, streaming: bool = True) -> Iterable[dict[str, Any]]:
        """Load dataset at pinned revision. trust_remote_code never passed.

        Raises RuntimeError if REPRODUCIBLE=True and REVISION is a mutable ref.
        Single-pass semantics: wrap in iter() to enforce single-pass explicitly.
        """
        if self.REPRODUCIBLE and (self.REVISION in _MUTABLE_REFS or HEX_REVISION_RE.fullmatch(self.REVISION) is None):
            raise RuntimeError(
                f"Reproducibility violation: REVISION={self.REVISION!r} is mutable. "
                "Set REVISION to a 40-char commit SHA, or set REPRODUCIBLE=False "
                "to explicitly opt into non-deterministic exploration mode."
            )

        from datasets import load_dataset

        return load_dataset(  # type: ignore[return-value]
            self.DATASET_ID,
            self.SUBSET,
            split=self.SPLIT,
            streaming=streaming,
            revision=self.REVISION,
        )

    def get_provenance(self) -> dict[str, Any]:
        """Return full provenance dict for W&B / experiment logging.
        Provenance is probe-level — log once at training start, not per-row.
        """
        import datasets

        return {
            "dataset": self.DATASET_ID,
            "subset": self.SUBSET,
            "split": self.SPLIT,
            "revision": self.REVISION,
            "hf_datasets_version": datasets.__version__,
            "probe_version": self.PROBE_VERSION,
            "reproducible": self.REPRODUCIBLE,
        }

    def iter_valid_rows(self, source: Iterable[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
        """Yield only validated, normalized rows. Invalid rows are skipped.
        Invariant: downstream code never sees invalid rows.
        Preferred pipeline entry point — enforces validate-then-normalize automatically.
        """
        rows = source if source is not None else self.load()
        for row in rows:
            if not self.validate_row(row):
                yield self.normalize_row(row)

    def validate_row(self, row: dict[str, Any]) -> list[str]:
        """Return all validation errors. Empty list = valid."""
        errors: list[str] = []

        missing = self.REQUIRED_FIELDS - set(row.keys())
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

        if len(value) < self.MIN_TEXT_LENGTH:
            errors.append(f"{text_field} too short: {len(value)} < {self.MIN_TEXT_LENGTH}")
        return errors

    def normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Normalize a pre-validated row into canonical form.

        Precondition: row must pass validate_row() with no errors.
        Raises ValueError if precondition is violated — programming error, not data error.
        Use iter_valid_rows() to enforce the contract automatically in pipelines.

        Shallow copy preserves all upstream fields (jurisdiction, court, judge).
        Old text field removed if renamed to avoid RAM duplication.
        Provenance NOT embedded — call get_provenance() at training start.

        Invariants after validate_row() passes:
          - resolve_text_field() always returns non-None (text_field guaranteed present)
          - 'url' always present (in REQUIRED_FIELDS) so source_url assigned unconditionally
        """
        errors = self.validate_row(row)
        if errors:
            raise ValueError(
                f"normalize_row() called on invalid row — validate first.\n"
                f"Errors: {errors}\n"
                f"Use iter_valid_rows() to enforce validate-then-normalize automatically."
            )

        normalized = dict(row)

        text_field = self.resolve_text_field(row)
        # Guaranteed non-None: validate_row() ensures a TEXT_FIELDS key is present.
        assert text_field is not None
        text = str(row[text_field]).strip()

        # Record which field the text came from — downstream models may need this
        # to distinguish scraping-variant schemas across pile-of-law subsets.
        normalized["_source_text_field"] = text_field
        if text_field != "text":
            normalized.pop(text_field, None)

        normalized["text"] = text
        normalized["created_timestamp"] = self._normalize_timestamp(str(row.get("created_timestamp", "")))
        normalized["downloaded_timestamp"] = self._normalize_timestamp(str(row.get("downloaded_timestamp", "")))
        # source_url is an intentional pipeline alias for url.
        # Downstream RAG components reference source_url consistently,
        # decoupling them from the raw field name which varies across
        # pile-of-law subsets (some use url, others use href or link).
        # 'url' is always present: it is in REQUIRED_FIELDS, enforced by validate_row().
        normalized["source_url"] = str(row["url"])

        return normalized

    def _normalize_timestamp(self, ts: str) -> str:
        """Parse and normalize timestamp using datetime — not just regex extraction.

        Validates actual date semantics (rejects '9999-99-99', '2022-13-45', etc.).
        Preserves the most precise valid format found:
          datetime+tz > datetime > date > '' (unparseable or invalid).

        Non-UTC offsets (e.g. +05:00) are preserved via isoformat() to retain
        the original timezone context — legal filing times are jurisdiction-local.
        """
        candidate = _TS_EXTRACT_RE.search(ts)
        if not candidate:
            return ""
        raw = candidate.group(0)
        normalized_raw = raw.replace("Z", "+00:00")

        for fmt, preserves_time, preserves_tz in _TS_FORMATS:
            try:
                parsed = datetime.strptime(normalized_raw, fmt)
                if preserves_tz and parsed.tzinfo is not None:
                    if parsed.tzinfo == timezone.utc:
                        return parsed.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
                    # Non-UTC: preserve original offset via isoformat()
                    return parsed.isoformat()
                if preserves_time:
                    return parsed.strftime("%Y-%m-%dT%H:%M:%S")
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return ""

    def resolve_text_field(self, row: dict[str, Any]) -> str | None:
        """Return the first available text field name, or None."""
        return next((k for k in self.TEXT_FIELDS if k in row), None)

    def get_text(self, row: dict[str, Any]) -> str:
        """Extract text content. Raises ValueError if no text field found.
        Strict accessor — callers never implement fallback logic themselves.
        """
        field = self.resolve_text_field(row)
        if field is None:
            raise ValueError(f"No text field in row keys: {sorted(row.keys())}")
        return str(row[field])


class ModelQualitySignals:
    """
    Model-relevant quality signals for RAG pipeline rows.
    These are soft warnings — not schema violations — returned as a list of
    (signal_name, detail) tuples. Empty list = no signals triggered.
    Callers decide whether to filter, flag, or pass rows downstream.
    """

    HTML_RE = re.compile(r"<[a-zA-Z][^>]{0,100}>")
    CITATION_RE = re.compile(r"\d+\s+[A-Z][a-z]*\.?\s*(?:\d+d?|App\.?|Supp\.?)")
    BOILERPLATE_PHRASES = (
        "all rights reserved",
        "this page intentionally left blank",
        "unpublished disposition",
        "not for publication",
        "do not cite",
    )

    @classmethod
    def check(cls, row: dict[str, Any], text_field: str = "text") -> list[tuple[str, str]]:
        """Return quality signals for a normalized row. Empty = clean."""
        signals: list[tuple[str, str]] = []
        text: str = row.get(text_field, "")

        # Approximate token length (whitespace-split — not tokenizer-specific)
        token_count = len(text.split())
        if token_count < 20:
            signals.append(("truncated_document", f"~{token_count} tokens — likely truncated"))
        if token_count > 100_000:
            signals.append(("gigantic_document", f"~{token_count} tokens — may exceed model context"))

        # HTML remnants
        if cls.HTML_RE.search(text):
            signals.append(("html_remnants", "HTML tags detected — scraping artifact"))

        # Unicode normalization check (NFC vs raw)
        import unicodedata

        if unicodedata.normalize("NFC", text) != text:
            signals.append(("unicode_not_nfc", "Text is not NFC-normalized"))

        # Boilerplate detection
        lower = text.lower()
        for phrase in cls.BOILERPLATE_PHRASES:
            if phrase in lower:
                signals.append(("boilerplate", f"Boilerplate phrase detected: {phrase!r}"))
                break

        # Citation density (very low = may not be a real opinion)
        citation_count = len(cls.CITATION_RE.findall(text))
        if token_count > 100 and citation_count == 0:
            signals.append(("no_citations", "No legal citations found — may be non-opinion text"))

        return signals
