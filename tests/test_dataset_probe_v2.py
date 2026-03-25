# tests/test_dataset_probe_v2.py
"""
TDD contract tests for the 7 actionable improvements to src/dataset_probe.py:
  1. CourtListenerDatasetProbe as real orchestrator with W&B hook
  4. ProbeConfig fields for 50_000 cap and subsample sizes
  6. tests/fixtures/courtlistener_sample.jsonl with real-format records
  10. B6 entropy spot-check assertion
  13. --ci-mode flag -> sys.exit(1) if all_passed=False
  15. Optional W&B logging hook in run_probe()
  16. probe_version in provenance block
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dataset_probe import (
    CourtListenerDatasetProbe,
    ProbeConfig,
    gate_b6_text_entropy_distribution,
    run_probe,
)

pytestmark = pytest.mark.unit

FIXTURE_JSONL = Path("tests/fixtures/courtlistener_sample.jsonl")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_records(n: int, **overrides) -> list[dict]:
    base = {
        "id": "1",
        "cluster_id": "c1",
        "docket_id": "d1",
        "court_id": "ca9",
        "court_name": "Ninth Circuit",
        "case_name": "Smith v. Jones",
        "date_filed": "2020-01-01",
        "precedential_status": "Published",
        "opinion_type": "majority",
        "extracted_by_ocr": False,
        "raw_text": "Smith v. Jones, 123 F.3d 456.",
        "text": "Smith v. Jones, 123 F.3d 456. " + ("The court held. " * 60),
        "text_length": 2000,
        "text_source": "plain_text",
        "cleaning_flags": [],
        "source": "courtlistener",
        "token_count": 400,
        "paragraph_count": 5,
        "citation_count": 3,
        "text_hash": "abc123",
        "citation_density": 0.05,
        "is_precedential": True,
        "text_entropy": 4.2,
    }
    records = []
    for i in range(n):
        r = dict(base)
        r["id"] = str(i)
        r.update(overrides)
        records.append(r)
    return records


@pytest.fixture
def sample_shard_dir(tmp_path: Path) -> Path:
    shard = tmp_path / "shard_000.jsonl"
    with open(shard, "w") as fh:
        for r in _make_records(50):
            fh.write(json.dumps(r) + "\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Item 4 — ProbeConfig fields for magic numbers
# ---------------------------------------------------------------------------


class TestProbeConfigMagicNumbers:
    def test_has_a13_text_cap_chars(self):
        """50_000 char cap in A13 must be a ProbeConfig field."""
        assert hasattr(ProbeConfig(), "a13_text_cap_chars")

    def test_a13_text_cap_chars_default(self):
        assert ProbeConfig().a13_text_cap_chars == 50_000

    def test_has_a11_subsample_n(self):
        assert hasattr(ProbeConfig(), "a11_subsample_n")

    def test_a11_subsample_n_default(self):
        assert ProbeConfig().a11_subsample_n == 200

    def test_has_a12_subsample_n(self):
        assert hasattr(ProbeConfig(), "a12_subsample_n")

    def test_a12_subsample_n_default(self):
        assert ProbeConfig().a12_subsample_n == 500

    def test_has_a13_subsample_n(self):
        assert hasattr(ProbeConfig(), "a13_subsample_n")

    def test_a13_subsample_n_default(self):
        assert ProbeConfig().a13_subsample_n == 200

    def test_custom_subsample_n_accepted(self):
        cfg = ProbeConfig(a11_subsample_n=50)
        assert cfg.a11_subsample_n == 50

    def test_custom_text_cap_accepted(self):
        cfg = ProbeConfig(a13_text_cap_chars=10_000)
        assert cfg.a13_text_cap_chars == 10_000

    def test_all_new_fields_in_provenance(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        config = report["provenance"]["probe_config"]
        assert "a13_text_cap_chars" in config
        assert "a11_subsample_n" in config
        assert "a12_subsample_n" in config
        assert "a13_subsample_n" in config


# ---------------------------------------------------------------------------
# Item 6 — Committed fixture JSONL with real-format records
# ---------------------------------------------------------------------------


class TestFixtureJSONL:
    def test_fixture_file_exists(self):
        """tests/fixtures/courtlistener_sample.jsonl must exist in the repo."""
        assert FIXTURE_JSONL.exists(), (
            f"Fixture file not found: {FIXTURE_JSONL}. "
            "Create tests/fixtures/courtlistener_sample.jsonl with real-format records."
        )

    def test_fixture_is_valid_jsonl(self):
        records = [json.loads(line) for line in FIXTURE_JSONL.read_text().splitlines() if line.strip()]
        assert len(records) >= 5

    def test_fixture_has_all_23_schema_fields(self):
        EXPECTED_FIELDS = {
            "id", "cluster_id", "docket_id", "court_id", "court_name",
            "case_name", "date_filed", "precedential_status", "opinion_type",
            "extracted_by_ocr", "raw_text", "text", "text_length", "text_source",
            "cleaning_flags", "source", "token_count", "paragraph_count",
            "citation_count", "text_hash", "citation_density", "is_precedential",
            "text_entropy",
        }
        records = [json.loads(line) for line in FIXTURE_JSONL.read_text().splitlines() if line.strip()]
        for r in records:
            missing = EXPECTED_FIELDS - set(r.keys())
            assert not missing, f"Fixture record missing fields: {missing}"

    def test_fixture_is_deterministic(self):
        r1 = [json.loads(l) for l in FIXTURE_JSONL.read_text().splitlines() if l.strip()]
        r2 = [json.loads(l) for l in FIXTURE_JSONL.read_text().splitlines() if l.strip()]
        assert r1 == r2

    def test_fixture_text_length_matches_text(self):
        records = [json.loads(l) for l in FIXTURE_JSONL.read_text().splitlines() if l.strip()]
        for r in records:
            assert abs(r["text_length"] - len(r["text"])) < 10, (
                "text_length field must match actual text length"
            )

    def test_fixture_usable_as_shard(self, tmp_path):
        """Fixture JSONL must be loadable by iter_shards_with_audit."""
        from src.dataset_probe import iter_shards_with_audit
        import shutil
        shard_dir = tmp_path / "fixture_shards"
        shard_dir.mkdir()
        shutil.copy(FIXTURE_JSONL, shard_dir / "courtlistener_sample.jsonl")
        audit = iter_shards_with_audit(shard_dir)
        assert len(audit["records"]) >= 5
        assert audit["total_parse_errors"] == 0


# ---------------------------------------------------------------------------
# Item 10 — B6 entropy spot-check assertion
# ---------------------------------------------------------------------------


class TestB6EntropySpotCheck:
    def test_b6_includes_spot_check_result(self):
        """B6 gate must include a spot_check field verifying formula consistency."""
        records = _make_records(20)
        r = gate_b6_text_entropy_distribution(records)
        assert "spot_check" in r

    def test_b6_spot_check_passes_when_formula_consistent(self):
        import math
        text = "the court held that the defendant was liable"
        words = text.split()
        freq: dict[str, float] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        n = len(words)
        expected_entropy = -sum((c / n) * math.log2(c / n) for c in freq.values())
        records = _make_records(5, text=text, text_entropy=round(expected_entropy, 4))
        r = gate_b6_text_entropy_distribution(records)
        assert r["spot_check"]["consistent"] is True

    def test_b6_spot_check_flags_formula_drift(self):
        """B6 spot-check must flag when stored text_entropy deviates from computed."""
        records = _make_records(5, text_entropy=999.0)
        r = gate_b6_text_entropy_distribution(records)
        assert r["spot_check"]["consistent"] is False

    def test_b6_spot_check_reports_max_deviation(self):
        records = _make_records(10)
        r = gate_b6_text_entropy_distribution(records)
        assert "max_deviation" in r["spot_check"]


# ---------------------------------------------------------------------------
# Item 13 — --ci-mode flag
# ---------------------------------------------------------------------------


class TestCIMode:
    def test_ci_mode_exits_1_when_gates_fail(self, tmp_path):
        """--ci-mode must sys.exit(1) when all_passed=False."""
        shard = tmp_path / "shards" / "s.jsonl"
        shard.parent.mkdir()
        # Write records that will fail A9 (all zero citations)
        bad_records = _make_records(100, citation_count=0)
        with open(shard, "w") as fh:
            for r in bad_records:
                fh.write(json.dumps(r) + "\n")

        result = subprocess.run(
            [
                sys.executable, "-m", "src.dataset_probe",
                "--data-dir", str(shard.parent),
                "--subset", "100",
                "--output", str(tmp_path / "r.json"),
                "--skip-tokenizer",
                "--skip-spacy",
                "--ci-mode",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_ci_mode_exits_0_when_all_pass(self, sample_shard_dir, tmp_path):
        """--ci-mode must exit 0 when all gates pass."""
        result = subprocess.run(
            [
                sys.executable, "-m", "src.dataset_probe",
                "--data-dir", str(sample_shard_dir),
                "--subset", "50",
                "--output", str(tmp_path / "r.json"),
                "--skip-tokenizer",
                "--skip-spacy",
                "--ci-mode",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_ci_mode_without_flag_never_exits_1(self, tmp_path):
        """Without --ci-mode, probe must always exit 0 even on gate failures."""
        shard = tmp_path / "shards" / "s.jsonl"
        shard.parent.mkdir()
        bad_records = _make_records(100, citation_count=0)
        with open(shard, "w") as fh:
            for r in bad_records:
                fh.write(json.dumps(r) + "\n")

        result = subprocess.run(
            [
                sys.executable, "-m", "src.dataset_probe",
                "--data-dir", str(shard.parent),
                "--subset", "100",
                "--output", str(tmp_path / "r.json"),
                "--skip-tokenizer",
                "--skip-spacy",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Item 15 — Optional W&B logging hook in run_probe()
# ---------------------------------------------------------------------------


class TestWandbLoggingHook:
    def test_run_probe_accepts_log_to_wandb_param(self, sample_shard_dir, tmp_path):
        """run_probe() must accept log_to_wandb=False without error."""
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
            log_to_wandb=False,
        )
        assert "gates" in report

    @patch("src.dataset_probe.wandb")
    def test_run_probe_calls_wandb_when_enabled(self, mock_wandb, sample_shard_dir, tmp_path):
        """run_probe() must call wandb.log when log_to_wandb=True."""
        mock_wandb.run = MagicMock()
        run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
            log_to_wandb=True,
        )
        assert mock_wandb.log.called

    def test_run_probe_default_does_not_log_wandb(self, sample_shard_dir, tmp_path):
        """Default run_probe() must not require W&B to be configured."""
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert report is not None


# ---------------------------------------------------------------------------
# Item 16 — probe_version in provenance block
# ---------------------------------------------------------------------------


class TestProbeVersion:
    def test_provenance_has_probe_version(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "probe_version" in report["provenance"]

    def test_probe_version_is_semver_or_date_string(self, sample_shard_dir, tmp_path):
        report = run_probe(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        v = report["provenance"]["probe_version"]
        assert isinstance(v, str) and len(v) >= 3

    def test_probe_version_constant_exported(self):
        from src.dataset_probe import PROBE_VERSION
        assert isinstance(PROBE_VERSION, str)
        assert len(PROBE_VERSION) >= 3


# ---------------------------------------------------------------------------
# Item 1 — CourtListenerDatasetProbe as real orchestrator
# ---------------------------------------------------------------------------


class TestCourtListenerDatasetProbeOrchestrator:
    def test_probe_exposes_config(self):
        probe = CourtListenerDatasetProbe()
        assert isinstance(probe.config, ProbeConfig)

    def test_probe_accepts_custom_config(self):
        cfg = ProbeConfig(a11_subsample_n=10)
        probe = CourtListenerDatasetProbe(config=cfg)
        assert probe.config.a11_subsample_n == 10

    def test_probe_run_returns_report_with_shard_audit(self, sample_shard_dir, tmp_path):
        probe = CourtListenerDatasetProbe()
        report = probe.run(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "shard_audit" in report

    def test_probe_run_returns_report_with_provenance(self, sample_shard_dir, tmp_path):
        probe = CourtListenerDatasetProbe()
        report = probe.run(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
        )
        assert "probe_version" in report["provenance"]

    def test_probe_run_accepts_log_to_wandb(self, sample_shard_dir, tmp_path):
        probe = CourtListenerDatasetProbe()
        report = probe.run(
            data_dir=sample_shard_dir,
            subset=20,
            output=tmp_path / "r.json",
            skip_tokenizer=True,
            skip_spacy=True,
            log_to_wandb=False,
        )
        assert report is not None
