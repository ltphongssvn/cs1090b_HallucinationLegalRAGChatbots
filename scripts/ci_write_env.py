# scripts/ci_write_env.py
# Write .env and ensure src/ exists for CI runs (no cluster setup.sh needed).
import os
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
