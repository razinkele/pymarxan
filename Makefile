.PHONY: test test-fast lint types check bench app docker docs

test:  ## Run full test suite with coverage (skips perf-budget benches)
	pytest tests/ -v -m "not bench" --cov --cov-report=term-missing

test-fast:  ## Run fast tests only (skip slow SA/zone tests + benches)
	pytest tests/ -v -m "not slow and not bench" --cov --cov-report=term-missing

bench:  ## Run perf-budget benchmarks (Phase 18/19/20 SA inner-loop costs)
	pytest tests/ -v -m bench

lint:  ## Run ruff linter
	ruff check src/ tests/ examples/

types:  ## Run mypy type checker
	mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports

check:  ## Run all checks (lint + types + test)
	$(MAKE) lint types test

app:  ## Start the Shiny app
	shiny run src/pymarxan_app/app.py

docker:  ## Build and run Docker container
	docker compose up --build

docs:  ## Generate API documentation with pdoc
	pdoc src/pymarxan src/pymarxan_shiny -o docs/api
