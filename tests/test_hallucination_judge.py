# tests/test_hallucination_judge.py
"""Tests for scripts.hallucination_judge — LLM-as-judge for citation faithfulness.

The judge model itself is an external API call (mocked in tests). Pure-Python
helpers (prompt construction, response parsing, score aggregation) are
unit-tested. End-to-end integration is verified manually with a small batch
before committing to the full 20,877-query × 4-ablation judge run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def judge_module() -> Any:
    from scripts import hallucination_judge

    return hallucination_judge


# ---------- contract ----------


@pytest.mark.contract
class TestContract:
    def test_constants(self, judge_module: Any) -> None:
        assert judge_module.DEFAULT_JUDGE_MODEL == "gpt-4o-mini"
        assert judge_module.DEFAULT_BATCH_SIZE >= 1
        assert judge_module.DEFAULT_MAX_RETRIES >= 1

    def test_score_labels(self, judge_module: Any) -> None:
        # Three discrete faithfulness labels per FaithBench-style protocol
        labels = judge_module.SCORE_LABELS
        assert set(labels) == {"FAITHFUL", "PARTIAL", "HALLUCINATED"}

    def test_main_callable(self, judge_module: Any) -> None:
        assert callable(getattr(judge_module, "main", None))

    def test_build_judge_prompt_callable(self, judge_module: Any) -> None:
        assert callable(getattr(judge_module, "_build_judge_prompt", None))

    def test_parse_judge_response_callable(self, judge_module: Any) -> None:
        assert callable(getattr(judge_module, "_parse_judge_response", None))

    def test_aggregate_scores_callable(self, judge_module: Any) -> None:
        assert callable(getattr(judge_module, "_aggregate_scores", None))

    def test_schema_version(self, judge_module: Any) -> None:
        import re

        assert re.match(r"^\d+\.\d+\.\d+$", judge_module.SCHEMA_VERSION)


# ---------- prompt construction ----------


@pytest.mark.unit
class TestBuildJudgePrompt:
    def test_includes_question_and_answer(self, judge_module: Any) -> None:
        p = judge_module._build_judge_prompt(
            question="What is the rule?",
            generation="The rule is X per Smith v. Jones.",
            contexts=["Context A about X."],
        )
        assert "What is the rule?" in p
        assert "The rule is X per Smith v. Jones." in p

    def test_includes_contexts(self, judge_module: Any) -> None:
        p = judge_module._build_judge_prompt(
            question="q",
            generation="a",
            contexts=["alpha", "beta"],
        )
        assert "alpha" in p
        assert "beta" in p

    def test_no_context_no_rag_label(self, judge_module: Any) -> None:
        """For ablation=none, contexts is empty; prompt should still work."""
        p = judge_module._build_judge_prompt(
            question="q",
            generation="a",
            contexts=[],
        )
        assert "q" in p and "a" in p

    def test_asks_for_label_in_response(self, judge_module: Any) -> None:
        """Prompt must instruct the judge to emit one of the SCORE_LABELS."""
        p = judge_module._build_judge_prompt(
            question="q",
            generation="a",
            contexts=[],
        )
        assert any(label in p for label in ("FAITHFUL", "PARTIAL", "HALLUCINATED"))


# ---------- response parsing ----------


@pytest.mark.unit
class TestParseJudgeResponse:
    def test_extracts_faithful(self, judge_module: Any) -> None:
        result = judge_module._parse_judge_response("Verdict: FAITHFUL\nReason: matches context.")
        assert result["label"] == "FAITHFUL"

    def test_extracts_partial(self, judge_module: Any) -> None:
        result = judge_module._parse_judge_response(
            "After review, the answer is PARTIAL — only the first claim is supported."
        )
        assert result["label"] == "PARTIAL"

    def test_extracts_hallucinated(self, judge_module: Any) -> None:
        result = judge_module._parse_judge_response("HALLUCINATED. The cited case does not appear in the contexts.")
        assert result["label"] == "HALLUCINATED"

    def test_unknown_response_returns_unknown(self, judge_module: Any) -> None:
        result = judge_module._parse_judge_response("I don't know how to answer.")
        assert result["label"] == "UNKNOWN"

    def test_preserves_raw_response(self, judge_module: Any) -> None:
        raw = "Verdict: FAITHFUL\nReason: yes."
        result = judge_module._parse_judge_response(raw)
        assert result["raw"] == raw


# ---------- aggregate scoring ----------


@pytest.mark.unit
class TestAggregateScores:
    def test_pure_faithful(self, judge_module: Any) -> None:
        scores = [
            {"label": "FAITHFUL"},
            {"label": "FAITHFUL"},
            {"label": "FAITHFUL"},
        ]
        agg = judge_module._aggregate_scores(scores)
        assert agg["n_total"] == 3
        assert agg["faithful_rate"] == pytest.approx(1.0)
        assert agg["hallucinated_rate"] == pytest.approx(0.0)

    def test_mixed(self, judge_module: Any) -> None:
        scores = [
            {"label": "FAITHFUL"},
            {"label": "PARTIAL"},
            {"label": "HALLUCINATED"},
            {"label": "HALLUCINATED"},
        ]
        agg = judge_module._aggregate_scores(scores)
        assert agg["n_total"] == 4
        assert agg["faithful_rate"] == pytest.approx(0.25)
        assert agg["partial_rate"] == pytest.approx(0.25)
        assert agg["hallucinated_rate"] == pytest.approx(0.5)

    def test_unknown_excluded_from_rates(self, judge_module: Any) -> None:
        """UNKNOWN responses count in n_total but not in label rates."""
        scores = [
            {"label": "FAITHFUL"},
            {"label": "UNKNOWN"},
        ]
        agg = judge_module._aggregate_scores(scores)
        assert agg["n_total"] == 2
        assert agg["n_unknown"] == 1
        assert agg["n_judged"] == 1
        # Rates over judged-only
        assert agg["faithful_rate"] == pytest.approx(1.0)

    def test_empty_returns_zeros(self, judge_module: Any) -> None:
        agg = judge_module._aggregate_scores([])
        assert agg["n_total"] == 0
        assert agg["faithful_rate"] == 0.0
        assert agg["hallucinated_rate"] == 0.0


# ---------- malformed inputs ----------


@pytest.mark.unit
class TestMalformedInputs:
    def test_empty_response_returns_unknown(self, judge_module: Any) -> None:
        assert judge_module._parse_judge_response("")["label"] == "UNKNOWN"

    def test_whitespace_only_returns_unknown(self, judge_module: Any) -> None:
        assert judge_module._parse_judge_response("   \n  ")["label"] == "UNKNOWN"
