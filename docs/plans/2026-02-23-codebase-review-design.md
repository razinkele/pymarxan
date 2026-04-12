# Codebase Review — Inconsistencies & Optimizations

**Date:** 2026-02-23
**Scope:** Full codebase review of `pymarxan` / `pymarxan_shiny` / `pymarxan_app`

## Approach

Prioritize **correctness bugs first**, then dependency/packaging issues, then
code quality improvements. Group fixes into small, testable batches.

---

## Batch 1: Correctness Bugs (CRITICAL)

### 1.1 `build_solution` omits SPF penalty from objective

**File:** `src/pymarxan/solvers/utils.py:180`
**Bug:** `objective = total_cost + blm * total_boundary` — missing `+ penalty`.
Every solver that calls `build_solution` (SA, heuristic, iterative improvement,
pipeline) returns solutions with under-counted objective values.

**Fix:** Call `compute_penalty()` and add it to the objective. Also add
cost-threshold penalty when applicable.

### 1.2 `misslevel` stored but never applied in cache penalty

**File:** `src/pymarxan/solvers/cache.py` — `compute_full_objective` and
`compute_delta_objective` use `self.feat_targets` raw without applying
`misslevel`.

**Bug:** The Marxan spec says penalty = SPF × max(0, target×MISSLEVEL − held).
The cache applies penalty = SPF × max(0, target − held), ignoring MISSLEVEL.

**Fix:** In `compute_full_objective` and `compute_delta_objective`, use
`self.feat_targets * self.misslevel` instead of `self.feat_targets` in
shortfall computation.

### 1.3 `map_utils.py` unconditional ipyleaflet import

**File:** `src/pymarxan_shiny/modules/mapping/map_utils.py:4`
**Bug:** Crashes at import time if ipyleaflet isn't installed. The guard in
calling modules catches this, but direct imports of map_utils will fail.

**Fix:** Wrap in try/except with a module-level `_HAS_IPYLEAFLET` flag, or
move the import inside the function body. Simplest: add try/except with a
clear error message.

### 1.4 Pipeline solver ignores user config

**File:** `src/pymarxan/solvers/run_mode.py:108`
**Bug:** `sa_cls()` and `heuristic_cls()` created with default params. User
SA parameters (NUMITNS, NUMTEMP, PROP) and progress metadata are dropped.

**Fix:** Forward the `config` argument to sub-solver `.solve()` calls.
The sub-solvers already read NUMITNS/NUMTEMP from `problem.parameters`.

---

## Batch 2: High-Priority Fixes

### 2.1 `or True` no-op test assertion

**File:** `tests/pymarxan/calibration/test_blm.py:43`
**Fix:** Remove `or True`.

### 2.2 `networkx` undeclared in dependencies

**File:** `pyproject.toml`
**Fix:** Add `networkx>=3.0` to the `[project.optional-dependencies] shiny`
group (only connectivity metrics use it).

### 2.3 `pdoc>=14.0` → `>=16.0`

**File:** `pyproject.toml`
**Fix:** Bump lower bound.

### 2.4 Missing solver exports in `__init__.py`

**File:** `src/pymarxan/solvers/__init__.py`
**Fix:** Add HeuristicSolver, IterativeImprovementSolver, RunModePipeline,
ZoneSASolver to exports.

### 2.5 Unused `connectivity_pu_ids` parameter

**File:** `src/pymarxan_shiny/modules/mapping/network_view.py:88`
**Fix:** Remove the parameter from the server function signature. Update
`app.py` call site.

### 2.6 `MIPSolver.available()` unreachable guard

**File:** `src/pymarxan/solvers/mip_solver.py:33-36`
**Bug:** `import pulp` at module top means if pulp isn't installed, the entire
module fails to import before `available()` can ever run.

**Fix:** Move `import pulp` inside `solve()` and `available()`, or accept that
the module requires pulp (it's a core dependency). Since pulp is in the main
`dependencies` list, this is a cosmetic issue — just simplify `available()` to
`return True`.

### 2.7 Orphaned `map_summary` in fallback paths

**Files:** `comparison_map.py`, `frequency_map.py`, `network_view.py`
**Bug:** When `_HAS_IPYLEAFLET = False`, the `map_summary` render.text function
is still defined but the fallback UI doesn't include
`output_text_verbatim("map_summary")`.

**Fix:** Move `map_summary` inside the `if _HAS_IPYLEAFLET:` block, or add
the output element to the fallback UI.

---

## Batch 3: Code Quality

### 3.1 Status magic numbers → named constants

**Files:** `simulated_annealing.py:68-71`, `mip_solver.py:58-65`,
`iterative_improvement.py`

**Fix:** Define `STATUS_LOCKED_IN = 2`, `STATUS_LOCKED_OUT = 3` in
`models/problem.py` and use them.

### 3.2 Dead "Load Zone Project" button

**File:** `src/pymarxan_shiny/modules/zones/zone_config.py:14`
**Fix:** Either add a click handler or remove the button (currently misleading).

### 3.3 `_sync_solver_params` in-place mutation

**File:** `src/pymarxan_app/app.py:178-203`
**Fix:** Create a new dict and call `problem.set()` with a new problem, or
document that `_sync_solver_params` must run before solve is triggered (which
the reactive effect ordering already ensures).

---

## Out of Scope (deferred)

- Boundary logic deduplication (3 implementations) — risky refactor, no bug
- `ProblemCache.from_problem` iterrows optimization — one-time cost, not hot
- `SolverRegistry.available_solvers()` instantiation — rarely called
- Duplicate computation in map summary renderers — UX only, not a bug
- Empty `pymarxan_shiny/__init__.py` — no consumers currently
