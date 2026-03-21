# HallucinationLegalRAGChatbots

[![CI](https://github.com/ltphongssvn/cs1090b_HallucinationLegalRAGChatbots/actions/workflows/ci.yml/badge.svg)](https://github.com/ltphongssvn/cs1090b_HallucinationLegalRAGChatbots/actions/workflows/ci.yml)

GPU pipeline comparing retrieval architectures (TF-IDF, CNN, LSTM, BERT bi-encoder, KG-augmented) to reduce hallucination in legal RAG chatbots.

**Hardware:** 4x NVIDIA L4 GPUs | Python 3.11.9 | torch 2.0.1+cu117 | CUDA 11.7 (driver 12.8)

## Quick Start
```bash
# Clone and install hooks (required once)
git clone https://github.com/ltphongssvn/cs1090b_HallucinationLegalRAGChatbots.git
cd cs1090b_HallucinationLegalRAGChatbots
uv run pre-commit install && uv run pre-commit install --hook-type pre-push

# Full GPU setup
bash setup.sh

# CPU-only (no GPU allocation needed)
SKIP_GPU=1 bash setup.sh

# Preview all side effects without executing
DRY_RUN=1 bash setup.sh
```

## Setup Modes

| Mode | Command |
|---|---|
| Full GPU setup | `bash setup.sh` |
| CPU-only | `SKIP_GPU=1 bash setup.sh` |
| Dry run | `DRY_RUN=1 bash setup.sh` |
| Quiet (CI) | `LOG_LEVEL=0 bash setup.sh` |
| Verbose | `LOG_LEVEL=2 bash setup.sh` |
| No download | `NO_DOWNLOAD=1 bash setup.sh` |
| No Jupyter | `NO_JUPYTER=1 bash setup.sh` |
| Single step | `STEP=<fn_name> bash setup.sh` |

## Notebook Cell 1 (required first line)
```python
from src.repro import configure
repro_cfg = configure()
```

## Testing
```bash
# Unit tests (fast, no GPU)
uv run pytest tests/ -m unit -v

# Contract tests
uv run pytest tests/ -m contract -v

# GPU tests (requires allocation)
uv run pytest tests/ -m gpu -v

# All with coverage
uv run pytest tests/ -m "unit or contract" --cov=src --cov-report=term-missing

# Shell script tests (requires bats-core)
bats tests/shell/

# All pre-commit hooks
uv run pre-commit run --all-files
```

## Coverage Enforcement

80% per-file minimum enforced at three levels:
- **pre-push hook** — blocks push if below threshold
- **CI pipeline** — `unit-tests` job with coverage report
- **pyproject.toml** — `fail_under = 80`

## CI/CD Pipeline

| Job | Trigger | Purpose |
|---|---|---|
| lint | push/PR | ruff + mypy |
| shell-tests | push/PR | bats-core hook tests |
| unit-tests | push/PR | pytest unit + coverage |
| cpu-smoke | push/PR | CPU forward pass + FAISS IVF |
| security | push/PR | pip-audit + CycloneDX SBOM |

## Module Map

| File | Responsibility |
|---|---|
| `scripts/lib.sh` | Constants, colors, step framework, messaging, guards |
| `scripts/bootstrap_env.sh` | uv, lockfile, venv, deps, drift |
| `scripts/validate_gpu.sh` | GPU detection, hardware policy, smoke tests |
| `scripts/setup_nlp.sh` | spaCy model download and verification |
| `scripts/setup_notebook.sh` | Repro env, repro module, stability, kernel |
| `scripts/validate_tests.sh` | Tiered test execution (5 tiers) |
| `scripts/manifest.sh` | Environment manifest + SBOM |
| `scripts/download_datasets.sh` | Legal HF corpus download |
| `src/repro.py` | Reproducibility config (notebook/CLI parity) |
| `src/environment.py` | Preflight checks and environment verification |
| `src/drift_check.py` | 5-tier dependency drift detection |
| `src/manifest_collector.py` | Environment provenance collection |

## Security

See [SECURITY.md](SECURITY.md) for full security practices.

Key protections:
- **detect-secrets** baseline scan on every commit
- **pre-push hooks** block .env, SSH keys, model binaries
- **pip-audit** CVE scan in CI and pre-push
- **CycloneDX SBOM** generated each CI run → `logs/sbom.json`

## Reproducibility

All experiments are reproducible via:
- `uv.lock` — pinned dependency snapshot
- `src/repro.py` — seeds + deterministic flags
- `logs/environment_manifest.json` — full provenance (git SHA, hardware, freeze snapshot, SLURM job)

## Git Workflow (GitFlow)
```
main (production) ← develop (integration) ← feature/* (work)
```

After cloning: `uv run pre-commit install && uv run pre-commit install --hook-type pre-push`
