# Phase 12: Documentation & Publishing — README, API Docs, PyPI Packaging

**Date:** 2026-02-22
**Status:** Design Document
**Goal:** Make pymarxan discoverable and usable by others with a README, auto-generated API docs, and PyPI-ready packaging.

---

## 1. Scope

| # | Item | Type | Purpose |
|---|------|------|---------|
| 1 | `README.md` | New | Project overview, quick start, architecture, dev commands |
| 2 | API docs with pdoc | New | Auto-generated HTML docs from docstrings |
| 3 | `make docs` target | Modify | One command to generate docs |
| 4 | PyPI-ready packaging polish | Modify | Classifiers, URLs, entry points in pyproject.toml |

## 2. README.md

Content structure:

1. **Title + one-line description** — "pymarxan: Modular Python toolkit for Marxan conservation planning"
2. **Badges** — CI status, Python version, license
3. **What is this** — 3-sentence overview: what Marxan does, what pymarxan adds, who it's for
4. **Quick Start** — `pip install pymarxan`, or clone + `pip install -e ".[all]"`, then `make app`
5. **Architecture** — Three-layer diagram (pymarxan-core / pymarxan-shiny / pymarxan-app) with one sentence each
6. **Solvers** — Table of 8 solvers with type (exact/heuristic/pipeline) and one-line description
7. **Development** — `make test`, `make lint`, `make check`, `make docs`
8. **Docker** — `make docker` or `docker compose up`
9. **License** — MIT

No tutorials or usage examples beyond quick start.

## 3. API Docs with pdoc

Setup:
- Add `pdoc` to `[project.optional-dependencies] dev` in pyproject.toml
- Generate: `pdoc src/pymarxan src/pymarxan_shiny -o docs/api --html`
- Output: `docs/api/` directory with static HTML

What gets documented:
- `pymarxan` package — all public modules (models, solvers, io, calibration, connectivity, analysis, zones)
- `pymarxan_shiny` package — all Shiny modules
- `pymarxan_app` excluded — just wiring, no public API

Makefile target:
```makefile
docs:  ## Generate API documentation
	pdoc src/pymarxan src/pymarxan_shiny -o docs/api --html
```

`.gitignore` addition: `docs/api/` — generated docs not committed.

## 4. PyPI-ready Packaging Polish

Changes to `pyproject.toml`:

```toml
authors = [{name = "pymarxan contributors"}]
readme = "README.md"
classifiers = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: GIS",
    "Intended Audience :: Science/Research",
]

[project.urls]
Repository = "https://github.com/pymarxan/pymarxan"
Documentation = "https://github.com/pymarxan/pymarxan"
```

No actual PyPI publish — just correct metadata for future `pip install pymarxan`.

## 5. File Changes Summary

### New Files (2)

| File | Purpose |
|------|---------|
| `README.md` | Project overview, quick start, dev commands |
| `.gitignore` | Exclude `docs/api/` generated output |

### Modified Files (2)

| File | Change |
|------|--------|
| `pyproject.toml` | Add pdoc dep, classifiers, authors, readme, URLs |
| `Makefile` | Add `docs` target |

### Unchanged

- All 489 tests
- All solver, model, I/O, calibration, and analysis code
- All 22 Shiny modules
- Dockerfile, docker-compose.yml, GitHub Actions CI

## 6. Testing Strategy

1. All 489 existing tests must continue to pass
2. `make docs` generates output in `docs/api/` without errors
3. `pip install -e ".[all]"` still works with updated pyproject.toml
4. README renders correctly (check with `python -m rich.markdown README.md` or visual inspection)
