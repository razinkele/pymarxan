# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project-specific skills (use them)

This repo ships skills under `.claude/skills/` that encode hard-won workflow knowledge. Invoke them rather than re-deriving:

- **`marxan-testing`** — before running/debugging tests or claiming work passes. Covers the env footgun below.
- **`marxan-parity-check`** — when touching a solver, objective/penalty math, target resolution, or the Marxan I/O readers/writers, or before claiming "matches Marxan".
- **`release-pymarxan`** — when releasing, tagging, bumping the version, or publishing.
- **`multi-agent-design-review`** — before executing any non-trivial feature/phase/solver change.

## The test-environment footgun (read before running pytest)

Tests **must** run under the `shiny` micromamba env, **not** `.venv`. The `.venv` lacks `rasterio` and `ipyleaflet`, so most of `pymarxan.spatial` and the Shiny map modules fail there with a misleading `ModuleNotFoundError: No module named 'rasterio'` that looks like a code bug.

```bash
# Correct interpreter (has rasterio + ipyleaflet):
/opt/micromamba/envs/shiny/bin/pytest tests/ -x -q

# Or activate it first, then `make` targets work:
source /opt/micromamba/etc/profile.d/micromamba.sh && micromamba activate shiny
```

The `Makefile` and `.github/copilot-instructions.md` call bare `pytest`, which only works once that env is on `PATH`.

## Commands

```bash
make check        # lint + types + full test suite — the bar before any commit/PR/release
make test         # full suite with coverage (skips bench)
make test-fast    # skip slow SA/zone tests while iterating  (-m "not slow")
make lint         # ruff (E, F, I, UP; line length 99)
make types        # mypy (pymarxan + pymarxan_shiny)
make bench        # perf-budget benches ONLY — excluded from CI; flaky on slow machines; run deliberately
make app          # launch the Shiny web app on :8000
make docs         # pdoc API docs

# Single file / single test / by marker:
/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip.py -v
/opt/micromamba/envs/shiny/bin/pytest tests/ -k "test_blm_affects_solution" -v
/opt/micromamba/envs/shiny/bin/pytest tests/ -m "not slow and not spatial" -v
```

Markers: `slow` (SA/zone/pipeline), `integration`, `spatial` (geopandas), `bench`. Coverage threshold 75%; `src/pymarxan_app/` is excluded from coverage. Shared fixture `tiny_problem` (6 PUs, 3 features) comes from `tests/data/simple/` via `conftest.py`.

`test_solutions_are_different` is a known stochastic flake — it passes on rerun. Only treat it as real if it reproduces on rerun, or if any *other* test fails.

## Correctness anchor

`tests/data/simple` (native Marxan format) has a hand-verified exact optimum: reserve `{2, 4, 6}` at cost **35.0** (derivation in `docs/VALIDATION.md`). The MIP solver must hit 35.0; SA/greedy must land **at or above** it. A heuristic reporting a cost *below* 35.0 is a bug (a silently violated target), never a better solver. `examples/validate_marxan_parity.py` is the runnable harness, pinned by `tests/test_examples.py`.

## Architecture (big picture)

Three-layer monorepo under `src/`; the dependency direction is strict:

- **`pymarxan/`** — pure Python core, **no UI dependencies**. Usable in scripts/notebooks/pipelines.
- **`pymarxan_shiny/`** — reusable Shiny-for-Python UI modules (~26).
- **`pymarxan_app/`** — wires the modules into the full web app.

Central data flow:

```
File (*.dat) → load_project() → ConservationProblem → Solver.solve(problem, config) → list[Solution]
                                                                                          ↓
                                                                       analysis / save_project()
```

Two abstractions carry the design:

- **`ConservationProblem`** (`models/problem.py`) — a dataclass of pandas DataFrames (`planning_units`, `features`, `pu_vs_features`, optional `boundary`/`probability`/`connectivity`) plus a `parameters: dict` of Marxan settings. **New optional fields must be `kw_only=True` with defaults** to keep backward compatibility; use `copy_with(**overrides)` to derive modified copies (preserves subclass type). PU `status`: `0` normal, `1` initial-include, `2` locked-in, `3` locked-out. `validate()` returns a `list[str]` of errors (it does not raise).
- **`Solver`** ABC (`solvers/base.py`) — `solve(problem, config) -> list[Solution]`, plus `name()`, `supports_zones()`, `available()`. To add a solver, inherit and implement those three. `Solution` carries `selected: np.ndarray`, `cost`, `boundary`, `objective`, `targets_met`, optional `zone_assignment`.

Performance model: solvers use a `ProblemCache` with **incremental delta computation** (per-flip cost) rather than recomputing the full objective each iteration — preserve this when editing SA/zone inner loops (the `bench` marker guards per-flip cost). Pure NumPy vectorization, dense matrices; no Numba/Cython.

Marxan parity is the core value proposition: penalty curves, target rules (MISSLEVEL, SPF, prop), HEURTYPE/ITIMPTYPE/RUNMODE semantics, and PROBMODE chance constraints all intentionally mirror the Marxan C++ source. When a cost changes, suspect a subtle parity detail first (past bugs: hyperbolic-vs-linear penalty curve, PU-id vs amount-sorted greedy ordering, MISSLEVEL application).

The detailed subpackage table, solver matrix, I/O conventions, and Shiny module pattern live in **`.github/copilot-instructions.md`** — consult it rather than duplicating that detail here.

## Conventions

- Python 3.12+, `from __future__ import annotations` in every file, full type hints maintained.
- Domain models are dataclasses. Test layout mirrors `src/` (`tests/pymarxan/{subpackage}/`).
- Strongly TDD: write the failing test first (`superpowers:test-driven-development`).
- I/O comes in symmetric `read_*`/`write_*` pairs; `load_project`/`save_project` are the entry points (delimiter auto-detected).
- Any test that builds an `ipyleaflet.Map` outside a Shiny session needs the `_allow_widget_outside_session` fixture (`shinywidgets` hooks `ipywidgets` globally).

## Release & versioning

The version is read at runtime via `importlib.metadata.version("pymarxan")` and lives in **`pyproject.toml` only** — never hardcode it elsewhere. Releases go through `scripts/release.sh VERSION` (pass the bare version, e.g. `0.5.0`; `--dry-run` first; `--no-pypi` to skip PyPI). It enforces clean tree / on `main` / in sync with origin / non-empty CHANGELOG `[Unreleased]` / full release toolchain present / `make check` green before doing anything irreversible. See the `release-pymarxan` skill.

**Release environment:** the release needs `ruff`, `mypy`, `pytest` (with the spatial deps rasterio/ipyleaflet in pytest's own interpreter — see `marxan-testing`) on `PATH`, plus the `build` module importable by the build interpreter. On this machine these are split (`ruff` in `~/.local/bin`, `mypy`/`build` in `.venv`, `pytest`+rasterio in the `shiny` micromamba env), so a bare `scripts/release.sh` fails. Merge the tool dirs onto `PATH` (shiny first so its rasterio-capable pytest wins) and point the build interpreter at the one with `build` via `PYTHON=`:
```bash
PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" \
  PYTHON="$PWD/.venv/bin/python" scripts/release.sh 0.5.1 --no-pypi
```
The pre-flight verifies ruff/mypy/pytest on `PATH` and `$PYTHON -m build` up front, so a missing tool aborts cleanly before any commit/tag rather than mid-release (the v0.5.0 failure mode).

Design and multi-agent-review docs accumulate in `docs/plans/`.
