# Phase 8: Solver UX — Run Panel, Progress, Convergence

**Date:** 2026-02-22
**Status:** Design Document
**Goal:** Professional solver execution experience with real-time progress monitoring and convergence visualization.

---

## 1. Problem Statement

The current app.py has solver execution logic inlined in the server function — a bare "Run Solver" button with no progress feedback, no iteration tracking, and duplicated solver instantiation code. Users click "Run" and see nothing until the solver finishes (or errors).

Phase 8 extracts solver execution into a proper run panel module with polling-based progress monitoring and adds a convergence plot for SA solvers.

## 2. Scope

### New Modules (3)

| Module | Purpose |
|---|---|
| `run_control/progress.py` | Thread-safe SolverProgress dataclass |
| `run_control/run_panel.py` | Start solver, progress bar, status log |
| `results/convergence.py` | SA objective-over-iterations plotly chart |

### Modified Files (3)

| File | Change |
|---|---|
| `solvers/simulated_annealing.py` | Add iteration history sampling + progress updates |
| `zones/solver.py` | Add run-level progress updates |
| `app.py` | Replace inline run logic with run_panel, add convergence tab |

### Unchanged

- Solver ABC (`base.py`) — no interface changes
- All other solvers (MIP, heuristic, iterative improvement, pipeline, binary)
- solver_picker module (already has SA/MIP params)
- All existing Shiny modules
- ProblemCache / ZoneProblemCache (Phase 7)
- All 386 existing tests

## 3. Progress Model

### 3.1 SolverProgress Dataclass

```python
@dataclass
class SolverProgress:
    status: str          # "idle" | "running" | "done" | "error"
    current_run: int     # Which solution run (1-based)
    total_runs: int      # num_solutions
    iteration: int       # Current iteration within run
    total_iterations: int
    best_objective: float
    message: str         # Human-readable status text
    error: str | None    # Error message if status == "error"
```

Thread safety: all fields are simple Python types with atomic reads/writes on CPython. No locks needed.

### 3.2 Integration Pattern

- `SolverProgress` reference passed via `SolverConfig.metadata["progress"]` (opt-in)
- SA and Zone SA solvers check for progress object and update it every ~1000 iterations
- MIP and other solvers: run panel wrapper updates run-level progress (run 1 of N, etc.)
- Solver ABC unchanged — progress is entirely opt-in

## 4. Run Panel Module

### 4.1 UI Components

- "Run Solver" primary action button
- Progress bar (Shiny `ui.progress`)
- Status text: "Run 3/10 — Iteration 450K/1M — Best: 1234.5"
- Scrollable log output (solver metadata as runs complete)

### 4.2 Server Logic

1. User clicks "Run" → spawn `threading.Thread` with solver
2. Thread updates shared `SolverProgress` as it runs
3. UI polls every 500ms via `reactive.invalidate_later(0.5)`
4. On completion: set `current_solution` and `all_solutions` reactive values
5. On error: display error in status area

### 4.3 What Moves Out of app.py

- `_run_solver()` reactive effect
- `run_status` and `run_log` render functions
- Solver instantiation logic (currently duplicated between `active_solver()` and `_run_solver()`)

## 5. Convergence Plot

### 5.1 Iteration History in SA Solver

Every ~1000 iterations, append to history dict:

```python
history = {
    "iteration": [],      # int
    "objective": [],       # float — current objective
    "best_objective": [],  # float — best seen so far
    "temperature": [],     # float — current temperature
}
```

For 1M iterations sampled every 1000: 1000 data points per run. Stored in `solution.metadata["history"]`.

### 5.2 Convergence Module

- Plotly line chart
- X-axis: iteration number
- Y-axis (primary): current objective + best objective (two lines)
- Y-axis (secondary): temperature (log scale, optional toggle)
- Dropdown to select which run to display
- Dependencies: plotly, shinywidgets

## 6. File Changes Summary

| Action | File |
|---|---|
| Create | `src/pymarxan_shiny/modules/run_control/__init__.py` |
| Create | `src/pymarxan_shiny/modules/run_control/progress.py` |
| Create | `src/pymarxan_shiny/modules/run_control/run_panel.py` |
| Create | `src/pymarxan_shiny/modules/results/convergence.py` |
| Modify | `src/pymarxan/solvers/simulated_annealing.py` |
| Modify | `src/pymarxan/zones/solver.py` |
| Modify | `src/pymarxan_app/app.py` |
| Create | `tests/pymarxan_shiny/test_run_panel.py` |
| Create | `tests/pymarxan_shiny/test_convergence.py` |
| Create | `tests/pymarxan/solvers/test_sa_history.py` |

## 7. Testing Strategy

1. All 386 existing tests must continue to pass
2. SA history tests: verify history is recorded, correct length, values decrease over time
3. SolverProgress tests: verify state transitions, thread-safe updates
4. Run panel tests: verify solver dispatch, progress polling, error handling
5. Convergence tests: verify chart data extraction from solution metadata
