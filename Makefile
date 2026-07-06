.PHONY: install install-dev install-hooks check test build clean-build help

# Keep uv's cache inside the repository by default. This avoids permission issues
# in constrained environments (e.g. CI/CD or sandboxed runners) where `$HOME`
# may not be writable.
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR

# Keep pre-commit's cache/logs inside the repository for the same reason.
PRE_COMMIT_HOME ?= $(CURDIR)/.cache/pre-commit
export PRE_COMMIT_HOME

.PHONY: install
install: ## Install the virtual environment
	@echo "🚀 Creating virtual environment using uv"
	@uv sync

.PHONY: install-hooks
install-hooks: ## Install the pre-commit hooks
	@uv run pre-commit install

.PHONY: install-dev
install-dev: install install-hooks ## Install the virtual environment + pre-commit hooks

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --doctest-modules

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
