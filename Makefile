.PHONY: help install install-dev install-notebook format lint type-check test test-fast security audit pipeline pipeline-fast notebook docs clean clean-outputs clean-cache pre-commit ci

PYTHON ?= python
SRC := src
TESTS := tests
NOTEBOOK := notebooks/demo.ipynb

.DEFAULT: help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.lock

install-dev: ## Install with development tools (linters, formatters, type-checker)
	$(PYTHON) -m pip install -e ".[dev]"

install-notebook: ## Install optional notebook dependencies
	$(PYTHON) -m pip install -e ".[notebook]"

format: ## Auto-format code with black + isort + ruff
	black $(SRC) $(TESTS) run_pipeline.py
	isort $(SRC) $(TESTS) run_pipeline.py
	ruff check --fix $(SRC) $(TESTS) run_pipeline.py

lint: ## Run linters (ruff)
	ruff check $(SRC) $(TESTS) run_pipeline.py
	black --check $(SRC) $(TESTS) run_pipeline.py
	isort --check-only $(SRC) $(TESTS) run_pipeline.py

type-check: ## Run mypy
	mypy $(SRC)

test: ## Run full test suite with coverage (>= 60%)
	pytest

test-fast: ## Run tests without coverage
	pytest -q --no-cov

security: ## Run bandit security audit
	bandit -r $(SRC) data run_pipeline.py

audit: lint type-check security test-fast ## Run full audit (lint + types + security + tests)

pipeline: ## Run the end-to-end pipeline (regenerates dataset + reports + figures)
	$(PYTHON) run_pipeline.py

pipeline-fast: ## Run pipeline skipping synthetic dataset regeneration
	$(PYTHON) -c "import run_pipeline; run_pipeline.main(regenerate_data=False)"

notebook: ## Execute the demo notebook in place (regenerates outputs)
	$(PYTHON) notebooks/_build.py

docs: ## Open project documentation
	@echo "Documentation lives in docs/ and README.md"

clean-outputs: ## Remove generated outputs (figures, reports)
	rm -rf outputs/figures/* outputs/reports/*

clean-cache: ## Remove Python cache directories
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

clean: clean-outputs clean-cache ## Remove all generated artifacts and caches

pre-commit: ## Install pre-commit hooks
	$(PYTHON) -m pip install pre-commit
	pre-commit install

ci: audit ## Run everything CI would run