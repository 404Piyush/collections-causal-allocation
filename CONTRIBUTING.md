# Contributing

Thank you for your interest in improving this project. Contributions of all
sizes are welcome — bug fixes, documentation improvements, new model variants,
test coverage, and performance work.

## Development setup

This project requires **Python 3.11+**.

```bash
# 1. Clone and enter the repo
git clone https://github.com/404Piyush/collections-causal-allocation.git
cd collections-causal-allocation

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# 3. Install runtime + dev dependencies
make install-dev

# 4. Install pre-commit hooks
make pre-commit
```

## Development workflow

1. **Create a branch** off `main`: `git checkout -b feat/your-feature`.
2. **Make focused commits** with clear messages (`feat:`, `fix:`, `docs:`, `refactor:`).
3. **Run the full audit locally** before pushing:
   ```bash
   make audit
   ```
   This runs `ruff` + `black --check` + `isort --check-only` + `mypy` + `bandit` + `pytest`.
4. **Push** and **open a Pull Request** against `main`.
5. **Wait for CI**: lint, type-check, security, tests on Python 3.11 and 3.12 must all pass.

## Code standards

- **Style**: PEP 8 with a 100-char line length (enforced by `black`).
- **Imports**: sorted by `isort` (`black`-compatible profile).
- **Linting**: `ruff` with `E, W, F, I, B, UP, N, C4, RET, SIM` rule sets.
- **Types**: gradual typing encouraged for public APIs (`mypy --ignore-missing-imports`).
- **Tests**: every public function should have at least one test in `tests/`.
- **Docstrings**: Google-style for modules and public functions.

## Commit message convention

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `perf`, `build`.

## Adding a new model or scenario

1. Place the implementation in the appropriate subpackage (`src/rdd/`, `src/milp/`, `src/evaluation/`).
2. Add an entry to `src/config.py` for any new constants.
3. Wire it into `run_pipeline.py` so it runs as part of the end-to-end pipeline.
4. Add unit tests in `tests/` mirroring the module name.
5. Update `README.md` and `docs/` as needed.

## Reporting security issues

Please **do not** open a public GitHub issue for security vulnerabilities. See
[`SECURITY.md`](./SECURITY.md) for the responsible-disclosure process.

## Code of conduct

By participating you agree to abide by the [Code of Conduct](./CODE_OF_CONDUCT.md).

## Questions?

Open a GitHub Discussion or Issue — no question is too small.