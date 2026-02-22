.PHONY: test test-fast lint types check app docker docs

test:  ## Run full test suite with coverage
	pytest tests/ -v --cov --cov-report=term-missing

test-fast:  ## Run fast tests only (skip slow SA/zone tests)
	pytest tests/ -v -m "not slow" --cov --cov-report=term-missing

lint:  ## Run ruff linter
	ruff check src/ tests/

types:  ## Run mypy type checker
	mypy src/pymarxan/ --ignore-missing-imports

check:  ## Run all checks (lint + types + test)
	$(MAKE) lint types test

app:  ## Start the Shiny app
	shiny run src/pymarxan_app/app.py

docker:  ## Build and run Docker container
	docker compose up --build

docs:  ## Generate API documentation with pdoc
	pdoc src/pymarxan src/pymarxan_shiny -o docs/api
