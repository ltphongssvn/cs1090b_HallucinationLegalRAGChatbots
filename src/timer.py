# src/timer.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/src/timer.py
"""Wall-clock timer context manager for notebook cells and CLI blocks.

Provides a single zero-dependency :func:`cell_timer` helper that logs
or prints the elapsed time when its ``with`` block exits, with units
(s/m/h) scaled to the magnitude so short blocks don't report
``"0h 0m 1.2s"`` and long training loops don't report raw seconds.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator, Optional


@contextmanager
def cell_timer(
    label: str = "Cell",
    logger: Optional[Any] = None,
    _override_elapsed: Optional[float] = None,
) -> Generator[None, None, None]:
    """Time the enclosed block and emit a human-scaled duration on exit.

    The ``finally`` clause runs even on exception, so a cell that
    raises still reports how long it ran before failing. Output
    format adapts to magnitude:

    * ``< 1 minute``  → ``"X.Ys"``
    * ``< 1 hour``    → ``"Xm Y.Ys"``
    * ``>= 1 hour``   → ``"Xh Ym Z.Zs"``

    Example:
        >>> from src.timer import cell_timer
        >>> with cell_timer("Cell 1", logger=logger):
        ...     # ... cell code ...

    Args:
        label: Display name shown in the output line.
        logger: Optional logger. When ``None``, the message is printed
            to stdout; otherwise it is emitted at INFO level.
        _override_elapsed: Test-only hook to inject a fixed elapsed
            value so the format branches can be unit-tested without
            actually sleeping.
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = _override_elapsed if _override_elapsed is not None else (time.time() - start)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours >= 1:
            msg = f"⏱ {label} completed in {int(hours)}h {int(minutes)}m {seconds:.1f}s"
        elif minutes >= 1:
            msg = f"⏱ {label} completed in {int(minutes)}m {seconds:.1f}s"
        else:
            msg = f"⏱ {label} completed in {seconds:.1f}s"
        if logger:
            logger.info(msg)
        else:
            print(msg)
