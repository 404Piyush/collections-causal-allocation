# Security policy

## Supported versions

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately by emailing **piyushutkar123@gmail.com** with:

- A clear description of the issue and impact
- Steps to reproduce, or a proof-of-concept
- The commit hash or release tag affected

You should receive an acknowledgement within **3 business days**, and a fix
or mitigation plan within **14 days** for confirmed issues.

## Scope

This is a portfolio / research codebase — there is no production deployment.
Typical issues of interest:

- Unsafe deserialization in `data/build_dataset.py`
- Unvalidated user inputs to the MILP optimizer (resource exhaustion)
- CSV injection in the output reports (`outputs/reports/*.csv`)
- Dependency vulnerabilities (see `requirements.lock`)

## Security tooling

- **bandit** runs in CI on every push and PR (`bandit -r src data run_pipeline.py`).
- The current audit (commit `cd69fd5`) reports **0 issues** across 1,295 lines of code.
- See `docs/security_audit.md` for the full report (8 informational findings, all non-blocking).
- Dependency updates are tracked via GitHub Dependabot (recommended enable).

## Out-of-scope

- Issues in third-party libraries (file upstream).
- Theoretical attacks without a concrete exploitation path.