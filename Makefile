.PHONY: install
install: ## Create the virtual environment and install dependencies
	@echo "Creating virtual environment using uv"
	@uv sync
	@uv run prek install

.PHONY: lock
lock: ## Update uv.lock
	@echo "Updating uv.lock"
	@uv lock

.PHONY: export
export: ## Export requirements.txt from uv.lock
	@echo "Exporting requirements.txt"
	@uv export -o requirements.txt --format requirements.txt

.PHONY: run
run: ## Run the Gradio app
	@.venv/bin/python app.py

.PHONY: test
test: ## Run tests
	@.venv/bin/pytest

.PHONY: check
check: ## Run code quality checks
	@echo "Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "Running pre-commit hooks via prek"
	@uv run prek run -a

.PHONY: lint
lint: ## Run ruff lint
	@uv run ruff check .

.PHONY: fmt
fmt: ## Run ruff formatter
	@uv run ruff format .

.PHONY: docs-test
docs-test: ## Build docs with mkdocs
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Serve docs locally
	@uv run mkdocs serve

.PHONY: help
help:
	@awk -F '## ' '/^[A-Za-z_-]+:.*##/ { target = $$1; sub(/:.*/, "", target); printf "\033[36m%-20s\033[0m %s\n", target, $$2 }' $(MAKEFILE_LIST)

.PHONY: compose-up
compose-up: ## Build and start app + seekdb with docker compose
	@docker compose up --build

.PHONY: compose-down
compose-down: ## Stop docker compose
	@docker compose down

.PHONY: compose-logs
compose-logs: ## Tail docker compose logs
	@docker compose logs -f

.PHONY: docker-build
docker-build: ## Build a single-container image for ModelScope Docker Studio
	@docker build -t endless-context:latest .

.DEFAULT_GOAL := help
