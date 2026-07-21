# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-22

### Added
- Modern `pyproject.toml` packaging with tool configuration for black, ruff, isort, mypy, pytest
- `Makefile` with install, lint, test, format, security, pipeline, notebook, docs targets
- `.editorconfig` and `.pre-commit-config.yaml` for cross-IDE consistency
- `Dockerfile` and `docker-compose.yml` for fully reproducible runs
- `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`
- `CITATION.cff` for GitHub-native academic citation
- `src/__init__.py` exposing `__version__`, `__author__`, `__email__`
- GitHub Issue and Pull Request templates
- Architecture Decision Record `docs/ADR-0001-rdd-milp-choice.md`
- `docs/references.md` with academic citations for RDD methods
- Mermaid architecture diagram and CI/MIT/Python badges in `README.md`

### Changed
- `README.md` rewritten to industry-standard structure (Features, Quickstart, Architecture, Configuration, Roadmap, Citation)
- `.github/workflows/ci.yml` enhanced with lint job, Python 3.11/3.12 matrix, and coverage upload
- `requirements.lock` expanded with optional `[notebook]` and `[dev]` extras

## [0.1.0] - 2026-07-21

### Added
- Initial release: RDD + MILP collections allocation engine
- Regression Discontinuity Design (IK bandwidth, local linear regression, triangular kernel, placebo cutoff, density test, covariate balance)
- Mixed-Integer Linear Programming allocation (PuLP/CBC)
- Three-scenario KPI evaluation: no-contact / current / MILP-optimized
- Synthetic bank debt-recovery dataset (LendingClub-calibrated)
- 9 unit tests across data loader, MILP optimizer, and RDD modules
- Bandit security audit (clean across 1,295 LoC)
- `docs/resume_blueprint.md` and `docs/security_audit.md`

[Unreleased]: https://github.com/404Piyush/collections-causal-allocation/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/404Piyush/collections-causal-allocation/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/404Piyush/collections-causal-allocation/releases/tag/v0.1.0