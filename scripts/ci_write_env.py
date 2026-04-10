# scripts/ci_write_env.py
# Project: HallucinationLegalRAGChatbots
# Path: cs1090b_HallucinationLegalRAGChatbots/scripts/ci_write_env.py
"""Minimal .env + directory bootstrap for CI runs.

CI containers do not run the full ``bash setup.sh`` — they do not have
GPUs, cluster schedulers, or the bandwidth budget for a fresh venv
build. But they still need the four reproducibility env vars that
:mod:`src.repro` expects in ``.env``, plus empty ``src/`` and ``logs/``
directories so imports and log writes succeed.

This script is the CI-only shortcut: it writes a minimal ``.env`` with
``PYTHONHASHSEED``, ``CUBLAS_WORKSPACE_CONFIG``,
``TOKENIZERS_PARALLELISM``, and ``RANDOM_SEED`` (all matching the
values the full setup.sh would produce), then ensures ``src/`` and
``logs/`` exist. Idempotent — safe to re-run.
"""

from __future__ import annotations

from pathlib import Path

root = Path(__file__).resolve().parent.parent
env_path = root / ".env"
env_path.write_text(
    "export PYTHONHASHSEED=0\n"
    "export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
    "export TOKENIZERS_PARALLELISM=false\n"
    "export RANDOM_SEED=0\n"
)
(root / "src").mkdir(exist_ok=True)
(root / "logs").mkdir(exist_ok=True)
print(f"Wrote {env_path}")
