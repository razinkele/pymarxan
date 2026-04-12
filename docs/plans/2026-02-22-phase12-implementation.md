# Phase 12: Documentation & Publishing — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make pymarxan discoverable and usable with a README, auto-generated API docs via pdoc, and PyPI-ready packaging metadata.

**Architecture:** Four additions: (1) README.md with project overview and quick start, (2) pdoc dependency + `make docs` target for API doc generation, (3) `.gitignore` update for generated docs, (4) pyproject.toml metadata polish (classifiers, authors, URLs). No code changes — purely documentation and packaging.

**Tech Stack:** pdoc, hatchling, Makefile, GitHub Actions (existing)

---

### Task 1: PyPI-ready Packaging Metadata

**Files:**
- Modify: `pyproject.toml:5-18`

**Step 1: Add project metadata**

In `pyproject.toml`, modify the `[project]` section. After line 10 (`license = "MIT"`), add:

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
```

Add a new section after `[tool.hatch.build.targets.wheel]`:

```toml
[project.urls]
Repository = "https://github.com/pymarxan/pymarxan"
Documentation = "https://github.com/pymarxan/pymarxan"
```

**Step 2: Add pdoc to dev dependencies**

In `[project.optional-dependencies] dev`, add `"pdoc>=14.0"` after `"pandas-stubs"`.

**Step 3: Verify packaging still works**

Run: `source .venv/bin/activate && pip install -e ".[all]" 2>&1 | tail -3`
Expected: Successfully installed (no errors).

**Step 4: Install pdoc**

Run: `source .venv/bin/activate && pip install pdoc>=14.0`
Expected: Successfully installed pdoc.

**Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add PyPI metadata, classifiers, pdoc dependency"
```

---

### Task 2: .gitignore Update

**Files:**
- Modify: `.gitignore`

**Step 1: Add docs/api/ and coverage artifacts to .gitignore**

Append to `.gitignore`:

```
docs/api/
htmlcov/
.coverage
```

**Step 2: Verify**

Run: `cat .gitignore`
Expected: Shows all existing entries plus new ones.

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add docs/api/ and coverage artifacts to .gitignore"
```

---

### Task 3: Makefile docs Target

**Files:**
- Modify: `Makefile`

**Step 1: Add docs target**

Read the current Makefile first. Add `docs` to the `.PHONY` line and add the target:

Update `.PHONY` line to:
```makefile
.PHONY: test test-fast lint types check app docker docs
```

Add after the `docker` target:

```makefile

docs:  ## Generate API documentation with pdoc
	pdoc src/pymarxan src/pymarxan_shiny -o docs/api --html
```

**IMPORTANT:** The recipe line MUST use a real tab character, not spaces.

**Step 2: Verify docs generation works**

Run: `source .venv/bin/activate && make docs`
Expected: Generates files in `docs/api/` without errors.

Run: `ls docs/api/`
Expected: `pymarxan/` and `pymarxan_shiny/` directories (or HTML files).

**Step 3: Clean up generated docs (they're gitignored)**

Run: `rm -rf docs/api/`

**Step 4: Commit**

```bash
git add Makefile
git commit -m "feat: add make docs target for pdoc API documentation"
```

---

### Task 4: README.md

**Files:**
- Create: `README.md`

**Step 1: Create README.md**

Create `README.md` at the project root:

```markdown
# pymarxan

Modular Python toolkit for Marxan conservation planning.

![CI](https://github.com/pymarxan/pymarxan/actions/workflows/ci.yml/badge.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## What is this?

[Marxan](https://marxansolutions.org/) is the world's most widely used conservation planning software, helping prioritize areas for biodiversity protection. **pymarxan** is a complete Python reimplementation that combines the classic Marxan algorithms with modern exact solvers, an interactive web UI, and a modular architecture for programmatic use.

It provides a pure Python core library for headless optimization, reusable Shiny UI components, and an assembled web application — all in one package.

## Quick Start

```bash
# Clone and install
git clone https://github.com/pymarxan/pymarxan.git
cd pymarxan
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Launch the web app
make app
```

Then open http://localhost:8000 in your browser.

## Architecture

```
pymarxan (three-layer monorepo)
├── src/pymarxan/          Core library: models, solvers, I/O, calibration, analysis
├── src/pymarxan_shiny/    Reusable Shiny UI modules (22 modules)
└── src/pymarxan_app/      Assembled 8-tab Shiny web application
```

- **pymarxan** — Pure Python, no UI dependencies. Use it in scripts, notebooks, or pipelines.
- **pymarxan_shiny** — Shiny for Python modules: maps, calibration, solver config, results.
- **pymarxan_app** — Wires all modules into a complete conservation planning application.

## Solvers

| Solver | Type | Description |
|--------|------|-------------|
| MIP (PuLP/CBC) | Exact | Mixed Integer Programming — guaranteed optimal solution |
| Simulated Annealing | Heuristic | Native Python SA with adaptive cooling |
| Marxan C++ Binary | Heuristic | Wraps the original Marxan executable |
| Zone SA | Heuristic | Multi-zone simulated annealing |
| Greedy Heuristic | Heuristic | 8 scoring modes (HEURTYPE 0-7) |
| Iterative Improvement | Heuristic | 4 refinement modes (ITIMPTYPE 0-3) |
| Run Mode Pipeline | Pipeline | Chains solvers per Marxan RUNMODE 0-6 |

## Development

```bash
make test        # Full test suite (489 tests) with coverage
make test-fast   # Skip slow SA tests (~479 tests, ~15s)
make lint        # Ruff linter
make types       # mypy type checker
make check       # All of the above
make docs        # Generate API docs with pdoc
```

## Docker

```bash
make docker
# or
docker compose up --build
```

The app will be available at http://localhost:8000.

## License

MIT
```

**Step 2: Verify README renders**

Run: `source .venv/bin/activate && python -c "
from pathlib import Path
readme = Path('README.md').read_text()
print(f'README.md: {len(readme)} chars, {len(readme.splitlines())} lines')
# Check key sections exist
for section in ['Quick Start', 'Architecture', 'Solvers', 'Development', 'Docker', 'License']:
    assert section in readme, f'Missing section: {section}'
print('All sections present.')
"`

Expected: Prints char/line counts and "All sections present."

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with project overview, quick start, and dev commands"
```

---

### Task 5: Full Regression

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `source .venv/bin/activate && make test 2>&1 | tail -5`
Expected: 489 passed, coverage at or above 75%.

**Step 2: Run lint**

Run: `source .venv/bin/activate && make lint`
Expected: All checks passed.

**Step 3: Verify docs generate**

Run: `source .venv/bin/activate && make docs && ls docs/api/ && rm -rf docs/api/`
Expected: docs/api/ populated, then cleaned up.

**Step 4: Verify app imports**

Run: `source .venv/bin/activate && python -c "from pymarxan_app.app import app; print('OK')"`
Expected: OK

**Step 5: Verify pip install works**

Run: `source .venv/bin/activate && pip install -e ".[all]" 2>&1 | tail -2`
Expected: Successfully installed.
