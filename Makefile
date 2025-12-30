.PHONY: help install install-dev test test-cov lint format type-check clean run docker-build docker-run

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install
	playwright install

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term

test-property: ## Run property-based tests only
	pytest -m property

test-integration: ## Run integration tests only
	pytest -m integration

lint: ## Run linting
	flake8 src tests
	mypy src

format: ## Format code
	black src tests
	isort src tests

type-check: ## Run type checking
	mypy src

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

run: ## Run the application
	uvicorn sovereign_career_architect.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run in production mode
	uvicorn sovereign_career_architect.api.main:app --host 0.0.0.0 --port 8000

docker-build: ## Build Docker image
	docker build -t sovereign-career-architect .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env sovereign-career-architect

setup-env: ## Set up environment file
	cp .env.example .env
	@echo "Please edit .env with your API keys"

check: format lint type-check test ## Run all checks

ci: ## Run CI pipeline
	make format
	make lint
	make type-check
	make test-cov