# src/timer.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_hw2/src/timer.py
# SRP: Cell execution timer for notebook visibility.

import time
from contextlib import contextmanager
from typing import Any, Generator, Optional


@contextmanager
def cell_timer(
    label: str = "Cell",
    logger: Optional[Any] = None,
    _override_elapsed: Optional[float] = None,
) -> Generator[None, None, None]:
    """Context manager that prints elapsed time when a notebook cell completes.

    Usage in notebook:
        from src.timer import cell_timer
        with cell_timer("Cell 1", logger=logger):
            # ... cell code ...

    Args:
        label: Display name for the cell.
        logger: If provided, logs instead of printing.
        _override_elapsed: Test-only override for formatting tests.
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
