.PHONY: help install lint lint-fix format format-check type test test-cov check clean dev bun-install bun-dev bun-start

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with poetry
	poetry install

lint:  ## Run ruff linter (check only)
	poetry run ruff check server

lint-fix:  ## Run ruff linter and auto-fix issues
	poetry run ruff check --fix server

format:  ## Format code with ruff
	poetry run ruff format server

format-check:  ## Check code formatting without changing files
	poetry run ruff format --check server

type:  ## Run pyright type checker
	poetry run pyright server

test:  ## Run pytest tests
	poetry run pytest

test-cov:  ## Run pytest with coverage report
	poetry run pytest --cov=server --cov-report=html --cov-report=term-missing

check: lint type test  ## Run all checks (lint + type + test)

clean:  ## Clean cache and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage 2>/dev/null || true

dev:  ## Run development server (Python) - Use: make dev ARGS="--model base --backend cuda"
	poetry run python -m server.main --config server-config.example.yaml $(ARGS)

dev-gpu:  ## Run with GPU config (CUDA backend, large model)
	poetry run python -m server.main --config server-config.example.yaml --backend cuda --model large-v3-turbo

dev-mlx:  ## Run with MLX config (Apple Silicon)
	poetry run python -m server.main --config server-config.example.yaml --backend mlx --model mlx-community/distil-whisper-large-v3

bun-install:  ## Install Bun client dependencies
	cd sandbox && bun install

bun-dev:  ## Run Bun client in watch mode
	cd sandbox && bun run dev

bun-start:  ## Run Bun client (production)
	cd sandbox && bun run start
