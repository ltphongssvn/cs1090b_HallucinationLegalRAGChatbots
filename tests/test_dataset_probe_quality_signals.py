# tests/test_dataset_probe_quality_signals.py
# Model-relevant quality signal tests for RAG pipeline.
import pytest

from src.dataset_probe import ModelQualitySignals, validate_schema

pytestmark = pytest.mark.unit

CLEAN_ROW = {
    "text": (
        "The court held in Smith v. Jones, 42 F.3d 100 (9th Cir. 1994), "
        "that the defendant failed to establish a genuine issue of material fact. "
        "See also Brown v. Board, 55 F.3d 200. The motion for summary judgment "
        "is GRANTED pursuant to Federal Rule of Civil Procedure 56(c)."
    )
}

_VALID_ROW: dict = {
    "id": "1",
    "court_id": "ca9",
    "text": "placeholder",
    "text_length": 500,
    "text_source": "plain_text",
    "citation_count": 3,
    "citation_density": 0.05,
    "is_precedential": True,
    "text_entropy": 4.2,
    "token_count": 100,
    "paragraph_count": 5,
}


class TestModelQualitySignals:
    def test_clean_opinion_has_no_signals(self) -> None:
        assert ModelQualitySignals.check(CLEAN_ROW) == []

    def test_truncated_document_flagged(self) -> None:
        row = {"text": "Motion denied."}
        signals = ModelQualitySignals.check(row)
        assert any(s[0] == "truncated_document" for s in signals)

    def test_html_remnants_flagged(self) -> None:
        row = {"text": "<p>The court held that <b>defendant</b> failed. " * 5}
        signals = ModelQualitySignals.check(row)
        assert any(s[0] == "html_remnants" for s in signals)

    def test_boilerplate_not_for_publication_flagged(self) -> None:
        row = {"text": "NOT FOR PUBLICATION. " + "The court affirms the ruling below. " * 20}
        signals = ModelQualitySignals.check(row)
        assert any(s[0] == "boilerplate" for s in signals)

    def test_boilerplate_do_not_cite_flagged(self) -> None:
        row = {"text": "Do not cite this opinion. " + "The court affirms the ruling. " * 20}
        signals = ModelQualitySignals.check(row)
        assert any(s[0] == "boilerplate" for s in signals)

    def test_no_citations_flagged_for_long_noncitation_text(self) -> None:
        row = {"text": "The court considered the matter and issued its ruling. " * 12}
        signals = ModelQualitySignals.check(row)
        assert any(s[0] == "no_citations" for s in signals)

    def test_citation_dense_text_passes_citation_check(self) -> None:
        row = {
            "text": (
                "See Smith v. Jones, 42 F.3d 100; Brown v. Board, 55 F.3d 200; "
                "Davis v. City, 12 F.Supp 300. The court affirms. " * 3
            )
        }
        signals = ModelQualitySignals.check(row)
        assert not any(s[0] == "no_citations" for s in signals)

    def test_unicode_not_nfc_flagged(self) -> None:
        nfd_text = ("caf\u0065\u0301 " * 30) + "The court held. " * 5
        row = {"text": nfd_text}
        signals = ModelQualitySignals.check(row)
        assert any(s[0] == "unicode_not_nfc" for s in signals)

    def test_nfc_normalized_text_passes(self) -> None:
        import unicodedata

        nfc_text = unicodedata.normalize("NFC", "café " * 10 + "The court held. " * 5)
        row = {"text": nfc_text}
        signals = ModelQualitySignals.check(row)
        assert not any(s[0] == "unicode_not_nfc" for s in signals)

    def test_signals_are_soft_not_schema_violations(self) -> None:
        """Quality signals must not affect validate_schema() — they are advisory only."""
        row = dict(_VALID_ROW)
        row["text"] = "<p>Not for publication.</p> " * 10  # triggers HTML + boilerplate signals
        row["text_length"] = len(row["text"])
        result = validate_schema([row])
        assert result["pass"] is True  # schema passes despite quality issues
        signals = ModelQualitySignals.check(row)
        assert len(signals) > 0  # but signals fire

    def test_multiple_signals_can_fire_simultaneously(self) -> None:
        row = {"text": "<div>Not for publication. Motion denied.</div>"}
        signals = ModelQualitySignals.check(row)
        names = {s[0] for s in signals}
        assert len(names) >= 2
