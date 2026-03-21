# conftest.py
# Path: cs1090b_HallucinationLegalRAGChatbots/conftest.py
#
# Root pytest configuration — loaded before any test module.
#
# Reproducibility: configure() is called here so that all pytest runs
# (unit, contract, integration, gpu) execute under identical reproducibility
# settings as notebook Cell 1 and CLI scripts. Without this, test results
# can differ from notebook/CLI results due to different RNG seeds, cuDNN
# algorithm selection, or missing CUBLAS_WORKSPACE_CONFIG — silently breaking
# the notebook/CLI/test parity that src/repro.py was built to enforce.
#
# Note: configure() loads .env with override=False, so any env vars already
# set in the shell (e.g. by setup.sh or CI) take precedence — this is safe.
import logging
import sys
from pathlib import Path

# Ensure src/ is importable regardless of how pytest is invoked
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("conftest")


def pytest_configure(config):
    """
    Called before test collection. Applies reproducibility settings globally
    so every test session runs under the same conditions as notebook/CLI.
    """
    try:
        from src.repro import configure
        repro_cfg = configure(verbose=False)
        logger.debug(
            "conftest: reproducibility configured — "
            "PYTHONHASHSEED=%s RANDOM_SEED=%s deterministic_algorithms=%s",
            repro_cfg.get("PYTHONHASHSEED"),
            repro_cfg.get("random_seed"),
            repro_cfg.get("deterministic_algorithms"),
        )
    except FileNotFoundError:
        # .env not present — likely a fresh clone before setup.sh has run.
        # Warn but do not block test collection so CI can still run checks.
        logger.warning(
            "conftest: .env not found — reproducibility settings NOT applied. "
            "Run bash setup.sh to generate .env and src/repro.py."
        )
    except ImportError:
        # src/repro.py not yet generated — same situation.
        logger.warning(
            "conftest: src.repro not importable — reproducibility settings NOT applied. "
            "Run bash setup.sh to generate src/repro.py."
        )
    except AssertionError as e:
        # configure() verification failed — surface as a clear error.
        raise RuntimeError(
            f"conftest: src.repro.configure() verification failed: {e}\n"
            "This means pytest is running with incorrect reproducibility settings.\n"
            "Fix: ensure CUBLAS_WORKSPACE_CONFIG and PYTHONHASHSEED are not "
            "overridden in ~/.bashrc, then re-run bash setup.sh."
        ) from e
