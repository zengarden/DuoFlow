# Repository Guidelines

## Project Structure & Module Organization

- `tiny_meanflow/`: library source code (core modules live here).
- `tests/`: pytest-based tests (e.g., `tests/test_*.py`).
- `pyproject.toml`: package metadata + tool configuration (pytest/mypy/ruff).
- `Makefile`: common developer workflows (`make help` lists targets).
- `tox.ini`: CI-style test/typecheck matrix across Python versions.

## Build, Test, and Development Commands

This project uses `uv` for dependency management and running tools.

- `make install`: create/update the virtualenv via `uv sync`.
- `make install-dev`: install deps + pre-commit hooks (equivalent to `make install` + `make install-hooks`).
- `make check`: verify lockfile (`uv lock --locked`), run linters/formatters via `pre-commit`, then run `mypy`.
- `make test`: run `pytest` plus module doctests (`--doctest-modules`).
- `tox`: run tests/type-checking in isolated envs (requires the configured Python versions installed).
- `make build`: build a wheel into `dist/`.

## Coding Style & Naming Conventions

- Python style: 4-space indentation; `snake_case` for functions/variables, `PascalCase` for classes.
- Formatting/imports: enforced via pre-commit (Black + isort) with a 120 char line length (see `.pre-commit-config.yaml`).
- Typing: keep functions typed; `mypy` is configured with `disallow_untyped_defs = true` in `pyproject.toml`.

## Testing Guidelines

- Framework: `pytest` (see `[tool.pytest.ini_options]` in `pyproject.toml`).
- Naming: test files `tests/test_*.py`, test functions `test_*`.
- Prefer small, deterministic tests; add coverage for bug fixes and new behavior.

## Commit & Pull Request Guidelines

- Commits: follow a Conventional Commits-style prefix (`feat: ...`, `fix: ...`, `docs: ...`), as used in the current history.
- PRs: include a short description, rationale, and how to verify (paste commands like `make check` / `make test`). Link relevant issues and update docs/README when behavior changes.

## Agent-Specific Notes

- Prefer Make targets over ad-hoc commands to match CI behavior.
- Don’t commit generated artifacts or caches (e.g., `.venv/`, `.uv-cache/`, `.cache/`, `dist/`).
