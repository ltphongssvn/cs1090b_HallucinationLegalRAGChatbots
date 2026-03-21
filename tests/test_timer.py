# tests/test_timer.py
# TDD RED: Cell execution timer.
import logging
import time

import pytest

pytestmark = pytest.mark.unit

from src.timer import cell_timer


class TestCellTimer:
    def test_prints_elapsed(self, capsys):
        """Timer prints elapsed time to stdout."""
        with cell_timer("Test"):
            time.sleep(0.1)
        output = capsys.readouterr().out
        assert "Test completed in" in output

    def test_logs_elapsed(self):
        """Timer logs elapsed time when logger provided."""
        msgs: list = []
        logger = logging.getLogger("test_timer")
        h = logging.Handler()
        h.emit = lambda r: msgs.append(r.getMessage())  # type: ignore
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
        with cell_timer("Cell 2", logger=logger):
            time.sleep(0.1)
        assert any("Cell 2 completed in" in m for m in msgs)

    def test_shows_minutes(self, capsys):
        """Timer formats minutes correctly for longer runs."""
        with cell_timer("Long", _override_elapsed=125.5):
            pass
        output = capsys.readouterr().out
        assert "2m" in output

    def test_shows_hours(self, capsys):
        """Timer formats hours for very long runs."""
        with cell_timer("VeryLong", _override_elapsed=7384.2):
            pass
        output = capsys.readouterr().out
        assert "2h" in output

    def test_no_crash_on_exception(self, capsys):
        """Timer still prints even if cell code raises."""
        with pytest.raises(ValueError):
            with cell_timer("Error"):
                raise ValueError("boom")
        output = capsys.readouterr().out
        assert "Error completed in" in output
