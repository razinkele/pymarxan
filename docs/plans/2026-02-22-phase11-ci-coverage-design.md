# Phase 11: Testing & CI/CD — Coverage, Fixtures, Makefile, GitHub Actions

**Date:** 2026-02-22
**Status:** Design Document
**Goal:** Make pymarxan production-ready with CI pipeline, coverage enforcement, shared test fixtures, and dev command shortcuts.

---

## 1. Scope

| # | Item | Type | Purpose |
|---|------|------|---------|
| 1 | Root `conftest.py` with shared fixtures + markers | New | Deduplicate test setup, enable `@pytest.mark.slow` |
| 2 | Coverage config in `pyproject.toml` | Modify | Track coverage, enforce 80% minimum |
| 3 | `Makefile` | New | `make test`, `make lint`, `make types`, `make app` |
| 4 | GitHub Actions CI workflow | New | Run tests + lint + type-check on push/PR |

After this phase: CI runs on every push, coverage enforced at 80%, dev workflow standardized.

## 2. Root conftest.py

**File:** `tests/conftest.py`

**Shared fixtures:**
- `tiny_problem` — the 6 PU / 3 feature fixture from `tests/data/simple/` (used by ~20+ tests)
- `make_problem` — the synthetic generator currently only in `tests/benchmarks/conftest.py`, promoted to root so integration tests can use it too

**Pytest markers** (registered in `pyproject.toml`):
- `@pytest.mark.slow` — SA-heavy tests, run_mode pipelines, zone solver tests (>2s each)
- `@pytest.mark.integration` — phase integration tests

**CI benefit:** `pytest -m "not slow"` runs the fast suite (~15s) on every PR; the full suite runs on main branch pushes.

## 3. Coverage Config

**Changes to `pyproject.toml`:**

```toml
[tool.coverage.run]
source = ["src"]
omit = ["src/pymarxan_app/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = ["if __name__", "pragma: no cover", "raise NotImplementedError"]
```

Why omit `pymarxan_app`: It's pure Shiny wiring (imports + reactive plumbing). The modules it wires are all tested individually. Trying to hit coverage on reactive effects would require a full Shiny test harness for little value.

Pytest invocation: `pytest --cov --cov-report=term-missing` becomes the default in the Makefile. CI fails if coverage drops below 80%.

## 4. Makefile

```makefile
.PHONY: test test-fast lint types check app docker

test:           ## Run full test suite with coverage
	pytest tests/ -v --cov --cov-report=term-missing

test-fast:      ## Run fast tests only (skip slow SA/zone tests)
	pytest tests/ -v -m "not slow" --cov --cov-report=term-missing

lint:           ## Run ruff linter
	ruff check src/ tests/

types:          ## Run mypy type checker
	mypy src/pymarxan/ --ignore-missing-imports

check:          ## Run all checks (lint + types + test)
	$(MAKE) lint types test

app:            ## Start the Shiny app
	shiny run src/pymarxan_app/app.py

docker:         ## Build and run Docker container
	docker compose up --build
```

Assumes the venv is already activated.

## 5. GitHub Actions CI

**File:** `.github/workflows/ci.yml`

**Triggers:** push to `master`/`main`, pull requests.

**Matrix:** Python 3.11 and 3.12.

**Jobs:**
1. **lint** — `ruff check src/ tests/` (fast, ~5s)
2. **types** — `mypy src/pymarxan/ --ignore-missing-imports` (fast, ~10s)
3. **test** — `pytest tests/ -v --cov --cov-report=term-missing` (full suite, ~20s)

Lint and types run in parallel. Test runs after both pass.

**Dependencies:** `pip install -e ".[all]"` plus `networkx` (needed for connectivity metric tests).

**Coverage:** Reported in test output. No external service — can add Codecov later.

**No secrets needed.** No deployment steps. Pure CI.

## 6. File Changes Summary

### New Files (3)

| File | Purpose |
|------|---------|
| `tests/conftest.py` | Shared fixtures + marker registration |
| `Makefile` | Dev commands |
| `.github/workflows/ci.yml` | CI pipeline |

### Modified Files (1)

| File | Change |
|------|--------|
| `pyproject.toml` | Add `[tool.coverage.*]` sections, register pytest markers |

### Unchanged

- All 489 tests and their assertions
- All solver, model, I/O, calibration, and analysis code
- All 22 Shiny modules
- Dockerfile, docker-compose.yml
- `tests/benchmarks/conftest.py` (stays as-is, root conftest imports from it)

## 7. Testing Strategy

1. All 489 existing tests must continue to pass
2. `make test` runs full suite with coverage, exits 0
3. `make test-fast` skips slow-marked tests, exits 0
4. `make lint` exits 0
5. `make types` exits 0
6. `make check` runs all three in sequence
7. CI workflow syntax validates with `actionlint` or manual inspection
