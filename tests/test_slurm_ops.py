"""Tests for src.ops.slurm_job — a reusable SLURM job monitoring utility."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def slurm_module():
    from src.ops import slurm_job

    return slurm_job


class TestParseDuration:
    def test_hms(self, slurm_module) -> None:
        assert slurm_module._parse_duration("01:30:45") == 5445

    def test_days_hms(self, slurm_module) -> None:
        # Format: "1-02:30:00" = 1 day + 2h 30m
        assert slurm_module._parse_duration("1-02:30:00") == 95400

    def test_zero(self, slurm_module) -> None:
        assert slurm_module._parse_duration("00:00:00") == 0

    def test_unlimited(self, slurm_module) -> None:
        assert slurm_module._parse_duration("UNLIMITED") is None

    def test_invalid_raises(self, slurm_module) -> None:
        with pytest.raises(ValueError):
            slurm_module._parse_duration("not a duration")


class TestParseSacctLine:
    def test_basic(self, slurm_module) -> None:
        line = "95397|00:30:15|08:00:00|RUNNING"
        info = slurm_module._parse_sacct_line(line)
        assert info.job_id == "95397"
        assert info.elapsed_seconds == 1815
        assert info.time_limit_seconds == 28800
        assert info.state == "RUNNING"

    def test_batch_step_skipped(self, slurm_module) -> None:
        # .batch / .extern steps should return None
        assert slurm_module._parse_sacct_line("95397.batch|00:30:15|08:00:00|RUNNING") is None

    def test_malformed_raises(self, slurm_module) -> None:
        with pytest.raises(ValueError):
            slurm_module._parse_sacct_line("garbage")


class TestGetJobStatus:
    def test_returns_jobstatus(self, slurm_module) -> None:
        fake_stdout = "95397|00:30:15|08:00:00|RUNNING\n95397.batch|00:30:14|08:00:00|RUNNING\n"
        with patch.object(
            slurm_module.subprocess,
            "check_output",
            return_value=fake_stdout,
        ):
            status = slurm_module.get_job_status(95397)
        assert status.job_id == "95397"
        assert status.elapsed_seconds == 1815
        assert status.time_limit_seconds == 28800
        assert status.state == "RUNNING"
        assert 0.0 < status.elapsed_fraction < 0.1

    def test_remaining_seconds(self, slurm_module) -> None:
        fake_stdout = "95397|00:30:00|08:00:00|RUNNING\n"
        with patch.object(slurm_module.subprocess, "check_output", return_value=fake_stdout):
            status = slurm_module.get_job_status(95397)
        assert status.remaining_seconds == 27000  # 8h - 30min

    def test_will_exceed_flag(self, slurm_module) -> None:
        # Elapsed 7h, projected completion 9h (from progress) vs 8h limit → exceed
        fake_stdout = "95397|07:00:00|08:00:00|RUNNING\n"
        with patch.object(slurm_module.subprocess, "check_output", return_value=fake_stdout):
            status = slurm_module.get_job_status(95397)
        # 7h elapsed, 1h remaining, 87.5% through
        assert status.elapsed_fraction > 0.85

    def test_job_not_found_raises(self, slurm_module) -> None:
        with patch.object(slurm_module.subprocess, "check_output", return_value=""):
            with pytest.raises(LookupError):
                slurm_module.get_job_status(99999999)


class TestCli:
    def test_cli_prints_summary(self, slurm_module, capsys) -> None:
        fake_stdout = "95397|00:30:15|08:00:00|RUNNING\n"
        with patch.object(slurm_module.subprocess, "check_output", return_value=fake_stdout):
            slurm_module.main(["95397"])
        captured = capsys.readouterr()
        assert "95397" in captured.out
        assert "RUNNING" in captured.out
        assert "00:30:15" in captured.out or "1815" in captured.out


class TestExtendedStatus:
    def test_extended_has_fields(self, slurm_module) -> None:
        # Extended format: JobID|Elapsed|TimeLimit|State|ExitCode|MaxRSS|AllocTRES|JobName
        fake = "95397|00:30:15|08:00:00|RUNNING|0:0|2048K|cpu=48,mem=160G|bge_m3_baseline\n"
        with patch.object(slurm_module.subprocess, "check_output", return_value=fake):
            ext = slurm_module.get_extended_status(95397)
        assert ext.job_id == "95397"
        assert ext.exit_code == "0:0"
        assert ext.max_rss == "2048K"
        assert ext.alloc_tres == "cpu=48,mem=160G"
        assert ext.job_name == "bge_m3_baseline"

    def test_extended_batch_step_skipped(self, slurm_module) -> None:
        fake = "95397.batch|00:30:15|08:00:00|RUNNING|0:0|1024K|cpu=48|batch\n95397|00:30:15|08:00:00|RUNNING|0:0|2048K|cpu=48,mem=160G|bge_m3\n"
        with patch.object(slurm_module.subprocess, "check_output", return_value=fake):
            ext = slurm_module.get_extended_status(95397)
        assert ext.job_id == "95397"


class TestJsonOutput:
    def test_cli_json_flag(self, slurm_module, capsys) -> None:
        import json as _json

        fake = "95397|00:30:15|08:00:00|RUNNING\n"
        with patch.object(slurm_module.subprocess, "check_output", return_value=fake):
            slurm_module.main(["95397", "--json"])
        captured = capsys.readouterr()
        payload = _json.loads(captured.out)
        assert payload["job_id"] == "95397"
        assert payload["state"] == "RUNNING"
        assert payload["elapsed_seconds"] == 1815
        assert payload["time_limit_seconds"] == 28800
        assert "elapsed_fraction" in payload
        assert "remaining_seconds" in payload
