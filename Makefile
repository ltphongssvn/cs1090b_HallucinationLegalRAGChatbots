# Makefile
# Path: cs1090b_HallucinationLegalRAGChatbots/Makefile
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
.DEFAULT_GOAL := help

SRC_DIR    ?= src
TESTS_DIR  ?= tests
MARKER     ?= not gpu
VENV_BIN   := .venv/bin
PYTEST     := $(VENV_BIN)/python -m pytest $(TESTS_DIR)/ --strict-markers -v
LOG_DIR    := logs
REPRO_LOG  := $(LOG_DIR)/repro_$(shell date +%Y%m%d_%H%M%S).log
BATS       := $(shell command -v bats 2>/dev/null || echo $(TESTS_DIR)/shell/bats-core/bin/bats)

.PHONY: setup cpu dry-run test test-all test-unit test-contract test-integration \
        test-regression test-gpu test-cov test-shell test-shell-artifacts \
        test-all-with-shell lint fmt format typecheck check ci repro \
        clean clean-all dvc-init install-hooks help

setup:
	bash setup.sh
	$(MAKE) install-hooks

cpu:
	SKIP_GPU=1 bash setup.sh
	$(MAKE) install-hooks

dry-run:
	DRY_RUN=1 bash setup.sh

install-hooks:
	@printf '#!/usr/bin/env bash\nset -euo pipefail\nmake check\n' > .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo " pre-push hook installed — make check enforced before every push"

# --- Shell tests (bats-core) ---
test-shell:
	@if [ ! -x "$(BATS)" ]; then \
		echo "ERROR: bats not found at '$(BATS)'"; \
		echo "       Install: npm install -g bats"; \
		echo "       or: git clone https://github.com/bats-core/bats-core $(TESTS_DIR)/shell/bats-core"; \
		exit 1; \
	fi
	@echo "Running shell tests via bats..."
	$(BATS) --tap \
		$(TESTS_DIR)/shell/test_lib.bats \
		$(TESTS_DIR)/shell/test_bootstrap_env.bats \
		$(TESTS_DIR)/shell/test_preflight.bats \
		$(TESTS_DIR)/shell/test_failure_paths.bats

# Artifact-verification tests (require setup.sh to have been run first)
test-shell-artifacts:
	@if [ ! -x "$(BATS)" ]; then \
		echo "ERROR: bats not found"; exit 1; \
	fi
	@echo "Running artifact verification tests..."
	$(BATS) --tap $(TESTS_DIR)/shell/test_artifact_verification.bats

# All shell tests including artifact verification
test-shell-all:
	@if [ ! -x "$(BATS)" ]; then \
		echo "ERROR: bats not found"; exit 1; \
	fi
	$(BATS) --tap \
		$(TESTS_DIR)/shell/test_lib.bats \
		$(TESTS_DIR)/shell/test_bootstrap_env.bats \
		$(TESTS_DIR)/shell/test_preflight.bats \
		$(TESTS_DIR)/shell/test_failure_paths.bats \
		$(TESTS_DIR)/shell/test_artifact_verification.bats

test: setup
	$(PYTEST) -m "$(MARKER)"

test-all: setup
	$(PYTEST) -n auto

test-all-with-shell: test-shell-all test-all

test-unit:
	$(MAKE) test MARKER=unit

test-contract:
	$(MAKE) test MARKER=contract

test-integration:
	$(MAKE) test MARKER=integration

test-regression:
	$(MAKE) test MARKER=regression

test-gpu:
	$(MAKE) test MARKER=gpu

test-cov:
	$(PYTEST) -n auto --cov=$(SRC_DIR) --cov-report=html --cov-report=xml

lint:
	$(VENV_BIN)/python -m ruff check $(SRC_DIR)/ $(TESTS_DIR)/

format:
	$(VENV_BIN)/python -m ruff check --fix $(SRC_DIR)/ $(TESTS_DIR)/
	$(VENV_BIN)/python -m ruff format $(SRC_DIR)/ $(TESTS_DIR)/

fmt: format

typecheck:
	$(VENV_BIN)/python -m mypy $(SRC_DIR)/

check: lint typecheck test-unit test-contract

ci: setup check

repro: clean-all setup
	@mkdir -p $(LOG_DIR)
	@echo "Repro run started: $$(date)" | tee $(REPRO_LOG)
	$(PYTEST) -n auto 2>&1 | tee -a $(REPRO_LOG); \
	EXIT=$$?; \
	if [ $$EXIT -ne 0 ]; then \
		echo "REPRO FAILED — see $(REPRO_LOG)"; \
		exit $$EXIT; \
	fi
	@echo "Repro run completed: $$(date)" | tee -a $(REPRO_LOG)
	@echo "Full reproducibility run complete — log: $(REPRO_LOG)"

clean:
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf .venv
	@echo "WARNING: .venv removed — run make setup to rebuild"

dvc-init:
	@if [ ! -d "data/raw/cl_federal_appellate_bulk" ]; then \
		echo "ERROR: DATA_DIR does not exist. Run Cell 2 pipeline first."; \
		exit 1; \
	fi
	$(VENV_BIN)/dvc status 2>/dev/null || $(VENV_BIN)/dvc init
	$(VENV_BIN)/dvc add data/raw/cl_federal_appellate_bulk

help:
	@echo "Usage: make <target> [MARKER=<pytest-marker>]"
	@echo ""
	@echo "  Setup:"
	@echo "    setup              — full bootstrap + install pre-push hook (GPU)"
	@echo "    cpu                — bootstrap without GPU checks (SKIP_GPU=1)"
	@echo "    dry-run            — preview all side effects (DRY_RUN=1 bash setup.sh)"
	@echo "    install-hooks      — install git pre-push hook"
	@echo ""
	@echo "  Tests:"
	@echo "    test               — run tests matching MARKER (default: 'not gpu')"
	@echo "    test-all           — full Python suite in parallel"
	@echo "    test-shell         — shell helper tests via bats-core (no artifacts)"
	@echo "    test-shell-artifacts — verify .venv/manifest/kernelspec contents"
	@echo "    test-shell-all     — all shell tests including artifact verification"
	@echo "    test-all-with-shell— shell-all + full Python suite"
	@echo "    test-unit          — MARKER=unit"
	@echo "    test-contract      — MARKER=contract"
	@echo "    test-integration   — MARKER=integration"
	@echo "    test-regression    — MARKER=regression"
	@echo "    test-gpu           — MARKER=gpu"
	@echo "    test-cov           — parallel tests with HTML+XML coverage"
	@echo ""
	@echo "  Quality:"
	@echo "    lint               — ruff check"
	@echo "    format             — ruff --fix + ruff format"
	@echo "    typecheck          — mypy check"
	@echo "    check              — lint + typecheck + unit + contract"
	@echo "    ci                 — setup + check (CI entry point)"
	@echo ""
	@echo "  Maintenance:"
	@echo "    repro              — clean-all + setup + test-all with timestamped log"
	@echo "    clean              — remove caches (preserves .venv)"
	@echo "    clean-all          — remove caches + .venv"
	@echo "    dvc-init           — initialize DVC for data dir"

# --- Security ---
audit:
	@echo "Running vulnerability audit (pip-audit against OSV)..."
	$(VENV_BIN)/python -m pip_audit --format=json --progress-spinner=off \
		--requirement <($(VENV_BIN)/pip list --format=freeze) 2>&1 | \
		$(VENV_BIN)/python -c "
import json,sys
data=json.load(sys.stdin)
vulns=[d for d in data.get('dependencies',[]) if d.get('vulns')]
if vulns:
    print(f'VULNERABILITIES FOUND: {len(vulns)} package(s)')
    for d in vulns:
        for v in d['vulns']:
            print(f'  {d[\"name\"]}=={d[\"version\"]}: {v[\"id\"]} — {v.get(\"description\",\"\")[:120]}')
    sys.exit(1)
else:
    print('No known vulnerabilities found.')
"

sbom:
	@echo "Generating CycloneDX SBOM..."
	mkdir -p logs
	$(VENV_BIN)/python -m cyclonedx_py environment --of JSON --output-file logs/sbom.json
	$(VENV_BIN)/python -m cyclonedx_py environment --of XML  --output-file logs/sbom.xml
	@echo "SBOM written: logs/sbom.json + logs/sbom.xml"

# Dataset probe targets
add-probe: ## Stage and commit all dataset probe files with standard message
	git add src/dataset_probe.py src/dataset_config.py src/dataset_loader.py \
		src/row_validator.py src/row_normalizer.py src/lightning_datamodule.py \
		tests/test_dataset_probe_contract.py tests/test_dataset_probe_normalization.py \
		tests/test_dataset_probe_reproducibility.py tests/test_dataset_probe_loader.py \
		tests/test_dataset_probe_edge_cases.py tests/test_dataset_probe_fixtures.py \
		tests/test_dataset_probe_quality_signals.py tests/test_dataset_loader.py \
		tests/fixtures/ configs/data/
	git commit -m "chore: update dataset probe and associated tests"

check-secrets: ## Run detect-secrets scan and update baseline
	.venv/bin/detect-secrets scan --exclude-files '.*\.lock$$|.*\.log$$' > .secrets.baseline
	git add .secrets.baseline
