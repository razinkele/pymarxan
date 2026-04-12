# Codebase Review — Implementation Plan

**Date:** 2026-02-23
**Design:** `docs/plans/2026-02-23-codebase-review-design.md`
**Branch:** `codebase-review-fixes`

---

## Task 1: Fix `build_solution` missing penalty

**Files:** `src/pymarxan/solvers/utils.py`

1. In `build_solution()`, after computing `total_cost` and `total_boundary`:
   - Call `compute_penalty(problem, selected, pu_index)` to get penalty
   - Add cost threshold penalty when `COSTTHRESH > 0`
   - Set `objective = total_cost + blm * total_boundary + penalty + ct_penalty`
2. Write test: create a problem where penalty > 0, call `build_solution`,
   verify `solution.objective > solution.cost + blm * solution.boundary`
3. Verify: `pytest tests/pymarxan/solvers/ -x -q`

---

## Task 2: Apply `misslevel` in cache penalty computation

**Files:** `src/pymarxan/solvers/cache.py`

1. In `compute_full_objective()` line 220:
   - Change `self.feat_targets` to `self.feat_targets * self.misslevel`
2. In `compute_delta_objective()` lines 295-297:
   - Change both `self.feat_targets` references to
     `self.feat_targets * self.misslevel`
3. Write test: create a problem with MISSLEVEL=0.5, verify that the cache
   penalty is computed against 50% of targets
4. Verify: `pytest tests/pymarxan/solvers/ -x -q`

---

## Task 3: Guard `map_utils.py` ipyleaflet import

**Files:** `src/pymarxan_shiny/modules/mapping/map_utils.py`

1. Wrap `import ipyleaflet` in try/except:
   ```python
   try:
       import ipyleaflet
       _HAS_IPYLEAFLET = True
   except ImportError:
       _HAS_IPYLEAFLET = False
   ```
2. Add a guard at the top of `create_grid_map()`:
   ```python
   if not _HAS_IPYLEAFLET:
       raise ImportError("ipyleaflet is required for create_grid_map")
   ```
3. Verify: existing tests still pass (ipyleaflet IS installed in test env)
4. Verify: `pytest tests/pymarxan_shiny/test_map_utils.py -x -q`

---

## Task 4: Forward config in pipeline solver

**Files:** `src/pymarxan/solvers/run_mode.py`

1. In `_run_pipeline()`, pass `config` to all sub-solver `.solve()` calls:
   - `sa_cls().solve(problem, config)[0]` (already done in some places)
   - `heuristic_cls().solve(problem, config)[0]`
   - `ii_cls().solve(problem, config)[0]`
   - `ii_cls().improve(problem, sol)` — no config needed (improve takes a solution)
2. Write test: verify that pipeline mode 0 with seed produces deterministic
   results matching direct SA call with same seed
3. Verify: `pytest tests/pymarxan/solvers/test_run_mode.py -x -q`

---

## Task 5: High-priority fixes batch

**Files:** Multiple

1. **`or True` assertion** — `tests/pymarxan/calibration/test_blm.py:43`:
   Remove `or True`

2. **networkx dependency** — `pyproject.toml`:
   Add `"networkx>=3.0"` to `[project.optional-dependencies] shiny`

3. **pdoc version** — `pyproject.toml`:
   Change `"pdoc>=14.0"` to `"pdoc>=16.0"`

4. **Solver exports** — `src/pymarxan/solvers/__init__.py`:
   Add imports and exports for HeuristicSolver, IterativeImprovementSolver,
   RunModePipeline, ZoneSASolver

5. **Unused parameter** — `src/pymarxan_shiny/modules/mapping/network_view.py`:
   Remove `connectivity_pu_ids` from server signature.
   Update `src/pymarxan_app/app.py` call site (remove kwarg).

6. **MIPSolver.available()** — `src/pymarxan/solvers/mip_solver.py`:
   Simplify to `return True` (pulp is a core dependency).

7. Verify: `make test-fast`

---

## Task 6: Orphaned map_summary in fallback paths

**Files:** `comparison_map.py`, `frequency_map.py`, `network_view.py`

1. Move `map_summary` render.text inside `if _HAS_IPYLEAFLET:` block in each
   file (it's only used with the ipyleaflet UI path which has
   `output_text_verbatim("map_summary")`)
2. Verify: `pytest tests/pymarxan_shiny/ -x -q`

---

## Task 7: Code quality fixes

**Files:** Multiple

1. **Status constants** — `src/pymarxan/models/problem.py`:
   Add `STATUS_LOCKED_IN = 2`, `STATUS_LOCKED_OUT = 3` at module level.
   Update `simulated_annealing.py` and `mip_solver.py` to use them.

2. **Dead zone button** — `src/pymarxan_shiny/modules/zones/zone_config.py`:
   Remove the `input_action_button("load_zones", ...)` since there's no
   handler. Replace with informational text about loading zone projects.

3. Verify: `make test-fast`

---

## Task 8: Full regression

1. Run `make check` (lint + types + tests)
2. Verify coverage >= 75%
3. Fix any ruff/mypy issues introduced
