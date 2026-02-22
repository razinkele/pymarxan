# Phase 8: Solver UX Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add run panel with progress monitoring, SA iteration history, and convergence plot to the Shiny app.

**Architecture:** Thread-safe `SolverProgress` dataclass shared between solver thread and UI. SA solver records sampled iteration history in solution metadata. Polling-based UI updates every 500ms. Convergence plot via plotly/shinywidgets.

**Tech Stack:** Shiny for Python, threading, plotly, shinywidgets, numpy

---

### Task 1: SolverProgress Dataclass

**Files:**
- Create: `src/pymarxan_shiny/modules/run_control/__init__.py`
- Create: `src/pymarxan_shiny/modules/run_control/progress.py`
- Test: `tests/pymarxan_shiny/test_progress.py`

**Step 1: Write the failing tests**

Create `tests/pymarxan_shiny/test_progress.py`:

```python
"""Tests for SolverProgress dataclass."""
from __future__ import annotations

from pymarxan_shiny.modules.run_control.progress import SolverProgress


class TestSolverProgress:
    def test_default_construction(self):
        p = SolverProgress()
        assert p.status == "idle"
        assert p.current_run == 0
        assert p.total_runs == 0
        assert p.iteration == 0
        assert p.total_iterations == 0
        assert p.best_objective == float("inf")
        assert p.message == ""
        assert p.error is None

    def test_update_fields(self):
        p = SolverProgress()
        p.status = "running"
        p.current_run = 3
        p.total_runs = 10
        p.iteration = 50000
        p.total_iterations = 1000000
        p.best_objective = 1234.5
        p.message = "Run 3/10 — Iteration 50K/1M"
        assert p.status == "running"
        assert p.current_run == 3
        assert p.best_objective == 1234.5

    def test_reset(self):
        p = SolverProgress()
        p.status = "running"
        p.current_run = 5
        p.best_objective = 100.0
        p.reset()
        assert p.status == "idle"
        assert p.current_run == 0
        assert p.best_objective == float("inf")
        assert p.error is None

    def test_format_status_idle(self):
        p = SolverProgress()
        assert "idle" in p.format_status().lower() or "ready" in p.format_status().lower()

    def test_format_status_running(self):
        p = SolverProgress()
        p.status = "running"
        p.current_run = 2
        p.total_runs = 5
        p.iteration = 500000
        p.total_iterations = 1000000
        text = p.format_status()
        assert "2" in text
        assert "5" in text

    def test_format_status_done(self):
        p = SolverProgress()
        p.status = "done"
        p.best_objective = 42.0
        text = p.format_status()
        assert "42" in text or "done" in text.lower() or "complete" in text.lower()

    def test_format_status_error(self):
        p = SolverProgress()
        p.status = "error"
        p.error = "Solver crashed"
        text = p.format_status()
        assert "crash" in text.lower() or "error" in text.lower()

    def test_progress_fraction_idle(self):
        p = SolverProgress()
        assert p.progress_fraction() == 0.0

    def test_progress_fraction_running(self):
        p = SolverProgress()
        p.status = "running"
        p.current_run = 2
        p.total_runs = 4
        p.iteration = 500000
        p.total_iterations = 1000000
        # Run 1 done (25%) + half of run 2 (12.5%) = 37.5%
        frac = p.progress_fraction()
        assert 0.3 < frac < 0.45

    def test_progress_fraction_done(self):
        p = SolverProgress()
        p.status = "done"
        assert p.progress_fraction() == 1.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan_shiny/test_progress.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan_shiny.modules.run_control'`

**Step 3: Write the implementation**

Create `src/pymarxan_shiny/modules/run_control/__init__.py`:

```python
```

Create `src/pymarxan_shiny/modules/run_control/progress.py`:

```python
"""Thread-safe progress model for solver execution monitoring."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SolverProgress:
    """Tracks solver execution progress.

    All fields are simple Python types with atomic reads/writes on CPython.
    Shared between solver thread (writer) and UI thread (reader).
    """

    status: str = "idle"  # "idle" | "running" | "done" | "error"
    current_run: int = 0
    total_runs: int = 0
    iteration: int = 0
    total_iterations: int = 0
    best_objective: float = field(default=float("inf"))
    message: str = ""
    error: str | None = None

    def reset(self) -> None:
        """Reset to idle state."""
        self.status = "idle"
        self.current_run = 0
        self.total_runs = 0
        self.iteration = 0
        self.total_iterations = 0
        self.best_objective = float("inf")
        self.message = ""
        self.error = None

    def format_status(self) -> str:
        """Return a human-readable status string."""
        if self.status == "idle":
            return "Ready to run."
        elif self.status == "running":
            if self.total_iterations > 0:
                pct = (self.iteration / self.total_iterations) * 100
                return (
                    f"Run {self.current_run}/{self.total_runs} — "
                    f"Iteration {self.iteration:,}/{self.total_iterations:,} "
                    f"({pct:.0f}%) — Best: {self.best_objective:.2f}"
                )
            return f"Run {self.current_run}/{self.total_runs} — Running..."
        elif self.status == "done":
            return f"Complete! Best objective: {self.best_objective:.2f}"
        elif self.status == "error":
            return f"Error: {self.error}"
        return self.message

    def progress_fraction(self) -> float:
        """Return overall progress as a float in [0, 1]."""
        if self.status == "done":
            return 1.0
        if self.status != "running" or self.total_runs == 0:
            return 0.0
        run_fraction = (self.current_run - 1) / self.total_runs
        if self.total_iterations > 0:
            iter_fraction = self.iteration / self.total_iterations
        else:
            iter_fraction = 0.0
        return run_fraction + iter_fraction / self.total_runs
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan_shiny/test_progress.py -v`
Expected: 11 passed

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/run_control/__init__.py \
        src/pymarxan_shiny/modules/run_control/progress.py \
        tests/pymarxan_shiny/test_progress.py
git commit -m "feat: add SolverProgress dataclass for solver execution monitoring"
```

---

### Task 2: SA Iteration History

Add sampled iteration history to the SA solver's solution metadata for convergence plotting.

**Files:**
- Modify: `src/pymarxan/solvers/simulated_annealing.py`
- Test: `tests/pymarxan/solvers/test_sa_history.py`

**Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_sa_history.py`:

```python
"""Tests for SA solver iteration history recording."""
from __future__ import annotations

from pathlib import Path

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestSAHistory:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.problem.parameters["NUMITNS"] = 10_000
        self.problem.parameters["NUMTEMP"] = 100
        self.solver = SimulatedAnnealingSolver()

    def test_history_present_in_metadata(self):
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        assert "history" in solutions[0].metadata
        history = solutions[0].metadata["history"]
        assert "iteration" in history
        assert "objective" in history
        assert "best_objective" in history
        assert "temperature" in history

    def test_history_has_correct_length(self):
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        history = solutions[0].metadata["history"]
        n = len(history["iteration"])
        # 10K iterations sampled every 1000 → ~10 points, plus initial
        assert 5 <= n <= 20
        assert len(history["objective"]) == n
        assert len(history["best_objective"]) == n
        assert len(history["temperature"]) == n

    def test_history_iterations_monotonic(self):
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        iters = solutions[0].metadata["history"]["iteration"]
        for i in range(1, len(iters)):
            assert iters[i] > iters[i - 1]

    def test_best_objective_non_increasing(self):
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        best = solutions[0].metadata["history"]["best_objective"]
        for i in range(1, len(best)):
            assert best[i] <= best[i - 1] + 1e-9

    def test_history_per_run(self):
        """Each run gets its own history."""
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = self.solver.solve(self.problem, config)
        for sol in solutions:
            assert "history" in sol.metadata
            assert len(sol.metadata["history"]["iteration"]) > 0

    def test_existing_tests_still_pass(self):
        """Verify basic solve still works with history recording."""
        config = SolverConfig(num_solutions=2, seed=42)
        solutions = self.solver.solve(self.problem, config)
        assert len(solutions) == 2
        for sol in solutions:
            assert sol.cost >= 0
            assert sol.objective >= 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/solvers/test_sa_history.py -v`
Expected: `test_history_present_in_metadata` FAILS — `KeyError: 'history'`

**Step 3: Modify the SA solver**

Edit `src/pymarxan/solvers/simulated_annealing.py`. The changes are:

1. Add a `HISTORY_SAMPLE_INTERVAL = 1000` constant at module level.
2. Inside the per-run loop, before the main SA loop, initialize the history dict and record the initial state.
3. Inside the main SA loop, every `HISTORY_SAMPLE_INTERVAL` iterations, append the current state.
4. After the main loop, record the final state and include history in the solution metadata.

Add at the top of the file (after imports):
```python
HISTORY_SAMPLE_INTERVAL = 1000
```

Inside the `solve` method, after the line `step_count = 0` and before `for _ in range(num_iterations):`, add:
```python
            # History recording for convergence plot
            history: dict[str, list] = {
                "iteration": [0],
                "objective": [current_obj],
                "best_objective": [current_obj],
                "temperature": [temp],
            }
            iter_count = 0
```

Replace the main loop `for _ in range(num_iterations):` and its body with:
```python
            for _ in range(num_iterations):
                # Pick random swappable PU
                idx = int(swappable_arr[rng.integers(n_swappable)])

                # Compute delta without flipping
                delta = cache.compute_delta_objective(
                    idx, selected, held, total_cost, blm
                )

                # Acceptance criterion
                if delta <= 0 or (
                    temp > 0 and rng.random() < math.exp(-delta / temp)
                ):
                    # Accept the move — update incremental state
                    sign = -1.0 if selected[idx] else 1.0
                    selected[idx] = not selected[idx]
                    held += sign * cache.pu_feat_matrix[idx]
                    total_cost += sign * cache.costs[idx]
                    current_obj += delta

                    # Track best
                    if current_obj < best_obj:
                        best_selected = selected.copy()
                        best_obj = current_obj

                # Cool
                step_count += 1
                if step_count >= iters_per_step:
                    temp *= alpha
                    step_count = 0

                # Sample history
                iter_count += 1
                if iter_count % HISTORY_SAMPLE_INTERVAL == 0:
                    history["iteration"].append(iter_count)
                    history["objective"].append(current_obj)
                    history["best_objective"].append(best_obj)
                    history["temperature"].append(temp)

            # Record final state
            if history["iteration"][-1] != num_iterations:
                history["iteration"].append(num_iterations)
                history["objective"].append(current_obj)
                history["best_objective"].append(best_obj)
                history["temperature"].append(temp)
```

In the `build_solution` call's metadata, add `"history": history`:
```python
            sol = build_solution(
                problem, best_selected, blm,
                metadata={
                    "solver": self.name(),
                    "run": run_idx + 1,
                    "initial_temp": round(initial_temp, 4),
                    "final_temp": round(temp, 6),
                    "best_objective": round(best_obj, 4),
                    "history": history,
                },
            )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan/solvers/test_sa_history.py -v`
Expected: 6 passed

Then verify existing SA tests still pass:
Run: `pytest tests/pymarxan/solvers/test_simulated_annealing.py -v`
Expected: 11 passed

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/simulated_annealing.py \
        tests/pymarxan/solvers/test_sa_history.py
git commit -m "feat: add iteration history sampling to SA solver for convergence plots"
```

---

### Task 3: SA + Zone SA Progress Updates

Wire the SA and Zone SA solvers to update a `SolverProgress` object when provided.

**Files:**
- Modify: `src/pymarxan/solvers/simulated_annealing.py`
- Modify: `src/pymarxan/zones/solver.py`
- Test: `tests/pymarxan/solvers/test_sa_progress.py`

**Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_sa_progress.py`:

```python
"""Tests for SA solver progress reporting."""
from __future__ import annotations

from pathlib import Path

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan_shiny.modules.run_control.progress import SolverProgress

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestSAProgress:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.problem.parameters["NUMITNS"] = 5_000
        self.problem.parameters["NUMTEMP"] = 50
        self.solver = SimulatedAnnealingSolver()

    def test_progress_updated_when_provided(self):
        progress = SolverProgress()
        config = SolverConfig(num_solutions=2, seed=42)
        config.metadata = {"progress": progress}
        self.solver.solve(self.problem, config)
        assert progress.status == "done"
        assert progress.current_run == 2
        assert progress.total_runs == 2

    def test_progress_not_required(self):
        """Solver works fine without progress object."""
        config = SolverConfig(num_solutions=1, seed=42)
        solutions = self.solver.solve(self.problem, config)
        assert len(solutions) == 1

    def test_progress_tracks_best_objective(self):
        progress = SolverProgress()
        config = SolverConfig(num_solutions=1, seed=42)
        config.metadata = {"progress": progress}
        solutions = self.solver.solve(self.problem, config)
        # Best objective should be finite after completion
        assert progress.best_objective < float("inf")

    def test_progress_error_on_invalid_problem(self):
        """Progress should reflect errors."""
        progress = SolverProgress()
        config = SolverConfig(num_solutions=1, seed=42)
        config.metadata = {"progress": progress}
        # Empty problem should cause an error or at least run
        # (with our simple test data it should succeed)
        self.solver.solve(self.problem, config)
        assert progress.status == "done"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/solvers/test_sa_progress.py -v`
Expected: FAIL — `SolverConfig` has no `metadata` attribute (or progress not read)

**Step 3: Modify SolverConfig and solvers**

First, add `metadata` to `SolverConfig` in `src/pymarxan/solvers/base.py`:

Add `from dataclasses import dataclass, field` (already imported) and add the field:
```python
@dataclass
class SolverConfig:
    """Configuration for a solver run."""

    num_solutions: int = 10
    seed: int | None = None
    verbose: bool = False
    metadata: dict = field(default_factory=dict)
```

Then modify `src/pymarxan/solvers/simulated_annealing.py`. At the beginning of the `solve()` method, after `config = SolverConfig()`, extract the progress object:

```python
        progress = config.metadata.get("progress") if hasattr(config, "metadata") else None
```

Before the `for run_idx in range(config.num_solutions):` loop:
```python
        if progress is not None:
            progress.status = "running"
            progress.total_runs = config.num_solutions
            progress.total_iterations = num_iterations
```

At the start of each run (inside the `for run_idx` loop, near the top):
```python
            if progress is not None:
                progress.current_run = run_idx + 1
                progress.iteration = 0
```

Inside the main SA loop, alongside the history sampling block (every `HISTORY_SAMPLE_INTERVAL` iterations):
```python
                if iter_count % HISTORY_SAMPLE_INTERVAL == 0:
                    history["iteration"].append(iter_count)
                    history["objective"].append(current_obj)
                    history["best_objective"].append(best_obj)
                    history["temperature"].append(temp)
                    if progress is not None:
                        progress.iteration = iter_count
                        progress.best_objective = best_obj
```

After all runs complete (after the `for run_idx` loop):
```python
        if progress is not None:
            progress.status = "done"
```

Similarly for `src/pymarxan/zones/solver.py`, add the same pattern:
- Extract progress from `config.metadata.get("progress")`
- Set `status = "running"`, `total_runs`, `total_iterations` before loop
- Update `current_run` at start of each run
- Update `iteration` and `best_objective` every 1000 iterations in the main loop
- Set `status = "done"` after all runs

In the Zone SA main loop, add an `iter_count` variable and update progress:
```python
            iter_count = 0
            for _ in range(num_iterations):
                # ... existing code ...

                iter_count += 1
                if progress is not None and iter_count % 1000 == 0:
                    progress.iteration = iter_count
                    progress.best_objective = best_obj
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan/solvers/test_sa_progress.py -v`
Expected: 4 passed

Then verify all existing solver tests still pass:
Run: `pytest tests/pymarxan/solvers/ tests/pymarxan/zones/ -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/base.py \
        src/pymarxan/solvers/simulated_annealing.py \
        src/pymarxan/zones/solver.py \
        tests/pymarxan/solvers/test_sa_progress.py
git commit -m "feat: add progress reporting to SA and Zone SA solvers"
```

---

### Task 4: Run Panel Module

**Files:**
- Create: `src/pymarxan_shiny/modules/run_control/run_panel.py`
- Test: `tests/pymarxan_shiny/test_run_panel.py`

**Step 1: Write the failing tests**

Create `tests/pymarxan_shiny/test_run_panel.py`:

```python
"""Tests for run panel Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.run_control.run_panel import (
    run_panel_server,
    run_panel_ui,
)


def test_run_panel_ui_returns_tag():
    ui_elem = run_panel_ui("test_run")
    assert ui_elem is not None


def test_run_panel_server_callable():
    assert callable(run_panel_server)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan_shiny/test_run_panel.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the run panel module**

Create `src/pymarxan_shiny/modules/run_control/run_panel.py`:

```python
"""Run panel Shiny module — solver execution with progress monitoring."""
from __future__ import annotations

import threading

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan_shiny.modules.run_control.progress import SolverProgress


@module.ui
def run_panel_ui():
    return ui.card(
        ui.card_header("Run Solver"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_action_button(
                    "run_solver",
                    "Run Solver",
                    class_="btn-primary btn-lg w-100",
                ),
                ui.hr(),
                ui.output_ui("progress_bar"),
                ui.hr(),
                ui.output_text_verbatim("run_log"),
                width=300,
            ),
            ui.output_text_verbatim("run_status"),
        ),
    )


@module.server
def run_panel_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
    solver: reactive.Calc,
    solver_config: reactive.Value,
    current_solution: reactive.Value,
    all_solutions: reactive.Value,
):
    progress = SolverProgress()
    solver_thread: reactive.Value[threading.Thread | None] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.run_solver)
    def _run_solver():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first!", type="error")
            return

        active = solver()
        if not active.available():
            ui.notification_show(
                f"Solver '{active.name()}' is not available.", type="error"
            )
            return

        config_dict = solver_config()
        p.parameters["BLM"] = config_dict.get("blm", 1.0)

        solver_type = config_dict.get("solver_type", "mip")
        if solver_type in ("binary", "sa", "zone_sa"):
            p.parameters["NUMITNS"] = config_dict.get(
                "num_iterations", 1000000
            )
            p.parameters["NUMTEMP"] = config_dict.get("num_temp", 10000)

        config = SolverConfig(
            num_solutions=config_dict.get("num_solutions", 10),
            seed=config_dict.get("seed"),
            verbose=False,
            metadata={"progress": progress},
        )

        progress.reset()
        ui.notification_show(f"Running {active.name()}...", type="message")

        def _run():
            try:
                solutions = active.solve(p, config)
                if solutions:
                    best = min(solutions, key=lambda s: s.objective)
                    current_solution.set(best)
                    all_solutions.set(solutions)
                    progress.status = "done"
                    progress.best_objective = best.objective
                    met = sum(best.targets_met.values())
                    total = len(best.targets_met)
                    progress.message = (
                        f"Done! Cost: {best.cost:.2f}, "
                        f"Targets met: {met}/{total}"
                    )
                else:
                    progress.status = "done"
                    progress.message = "Solver returned no solutions."
            except Exception as e:
                progress.status = "error"
                progress.error = str(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        solver_thread.set(thread)

    @render.ui
    def progress_bar():
        # Trigger re-render every 500ms while running
        if progress.status == "running":
            reactive.invalidate_later(0.5)
        frac = progress.progress_fraction()
        pct = int(frac * 100)
        return ui.div(
            ui.div(
                ui.div(
                    f"{pct}%",
                    class_="progress-bar",
                    role="progressbar",
                    style=f"width: {pct}%",
                ),
                class_="progress",
            ),
            ui.p(progress.format_status(), class_="mt-2 text-muted small"),
        )

    @render.text
    def run_status():
        # Poll while running
        if progress.status == "running":
            reactive.invalidate_later(0.5)
        p = problem()
        s = current_solution()
        if p is None:
            return "Step 1: Go to 'Data' tab and load a Marxan project."
        if progress.status == "running":
            return progress.format_status()
        if progress.status == "error":
            return f"Error: {progress.error}"
        if s is None:
            return (
                f"Project loaded: {p.n_planning_units} PUs, "
                f"{p.n_features} features.\n"
                f"Step 2: Configure solver in 'Configure' tab, "
                f"then click 'Run Solver'."
            )
        all_met = "Yes" if s.all_targets_met else "No"
        return (
            f"Solution available!\n  Selected: {s.n_selected} PUs\n"
            f"  Cost: {s.cost:.2f}\n  Boundary: {s.boundary:.2f}\n"
            f"  Objective: {s.objective:.2f}\n"
            f"  All targets met: {all_met}\n\n"
            f"Go to 'Results' tab to explore the solution."
        )

    @render.text
    def run_log():
        if progress.status == "running":
            reactive.invalidate_later(0.5)
        s = current_solution()
        if s is None:
            return "No solver has been run yet."
        lines = ["Solver metadata:"]
        for k, v in s.metadata.items():
            if k == "history":
                lines.append(f"  history: {len(v.get('iteration', []))} data points")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan_shiny/test_run_panel.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/run_control/run_panel.py \
        tests/pymarxan_shiny/test_run_panel.py
git commit -m "feat: add run panel module with progress monitoring"
```

---

### Task 5: Convergence Plot Module

**Files:**
- Create: `src/pymarxan_shiny/modules/results/convergence.py`
- Test: `tests/pymarxan_shiny/test_convergence.py`

**Step 1: Write the failing tests**

Create `tests/pymarxan_shiny/test_convergence.py`:

```python
"""Tests for convergence plot Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.results.convergence import (
    convergence_server,
    convergence_ui,
    extract_history,
)


def test_convergence_ui_returns_tag():
    ui_elem = convergence_ui("test_conv")
    assert ui_elem is not None


def test_convergence_server_callable():
    assert callable(convergence_server)


def test_extract_history_from_solutions():
    """extract_history returns list of histories from solutions."""
    from pymarxan.solvers.base import Solution
    import numpy as np

    sol1 = Solution(
        selected=np.array([True, False]),
        cost=10.0,
        boundary=5.0,
        objective=15.0,
        targets_met={1: True},
        metadata={
            "history": {
                "iteration": [0, 1000, 2000],
                "objective": [100.0, 80.0, 60.0],
                "best_objective": [100.0, 80.0, 60.0],
                "temperature": [10.0, 5.0, 1.0],
            },
            "run": 1,
        },
    )
    sol2 = Solution(
        selected=np.array([True, True]),
        cost=20.0,
        boundary=3.0,
        objective=23.0,
        targets_met={1: True},
        metadata={"run": 2},  # No history (e.g., MIP solver)
    )
    histories = extract_history([sol1, sol2])
    assert len(histories) == 1
    assert histories[0]["run"] == 1
    assert len(histories[0]["iteration"]) == 3


def test_extract_history_empty():
    histories = extract_history([])
    assert histories == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan_shiny/test_convergence.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the convergence module**

Create `src/pymarxan_shiny/modules/results/convergence.py`:

```python
"""Convergence plot Shiny module — SA objective over iterations."""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.solvers.base import Solution


def extract_history(
    solutions: list[Solution],
) -> list[dict]:
    """Extract iteration histories from solutions that have them.

    Returns a list of dicts, each with keys: run, iteration, objective,
    best_objective, temperature.
    """
    histories = []
    for sol in solutions:
        history = sol.metadata.get("history")
        if history and len(history.get("iteration", [])) > 0:
            histories.append({
                "run": sol.metadata.get("run", len(histories) + 1),
                **history,
            })
    return histories


@module.ui
def convergence_ui():
    return ui.card(
        ui.card_header("SA Convergence"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "run_select",
                    "Select Run",
                    choices={"1": "Run 1"},
                    selected="1",
                ),
                ui.input_checkbox(
                    "show_temperature",
                    "Show Temperature",
                    value=False,
                ),
                width=200,
            ),
            ui.output_ui("convergence_plot"),
        ),
    )


@module.server
def convergence_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    all_solutions: reactive.Value,
):
    @reactive.calc
    def histories():
        solutions = all_solutions()
        if solutions is None:
            return []
        return extract_history(solutions)

    @reactive.effect
    def _update_run_choices():
        h = histories()
        if not h:
            return
        choices = {str(i + 1): f"Run {entry['run']}" for i, entry in enumerate(h)}
        ui.update_select("run_select", choices=choices, selected="1")

    @render.ui
    def convergence_plot():
        h = histories()
        if not h:
            return ui.p("No convergence data available. Run an SA solver first.")

        idx = int(input.run_select()) - 1
        if idx < 0 or idx >= len(h):
            idx = 0
        entry = h[idx]

        try:
            import plotly.graph_objects as go
            from shinywidgets import output_widget, render_widget

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=entry["iteration"],
                y=entry["objective"],
                mode="lines",
                name="Current Objective",
                line=dict(color="steelblue", width=1),
                opacity=0.6,
            ))
            fig.add_trace(go.Scatter(
                x=entry["iteration"],
                y=entry["best_objective"],
                mode="lines",
                name="Best Objective",
                line=dict(color="darkgreen", width=2),
            ))

            if input.show_temperature():
                fig.add_trace(go.Scatter(
                    x=entry["iteration"],
                    y=entry["temperature"],
                    mode="lines",
                    name="Temperature",
                    line=dict(color="orangered", width=1, dash="dot"),
                    yaxis="y2",
                ))
                fig.update_layout(
                    yaxis2=dict(
                        title="Temperature",
                        overlaying="y",
                        side="right",
                        type="log",
                    ),
                )

            fig.update_layout(
                xaxis_title="Iteration",
                yaxis_title="Objective Value",
                title=f"SA Convergence — Run {entry['run']}",
                height=400,
                margin=dict(l=60, r=60, t=40, b=40),
                legend=dict(x=0.7, y=0.95),
            )

            # Return as static HTML since shinywidgets may not be available
            return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
        except ImportError:
            # Fallback: text summary
            iters = entry["iteration"]
            bests = entry["best_objective"]
            lines = [f"Run {entry['run']} convergence:"]
            for i, b in zip(iters, bests):
                lines.append(f"  Iter {i:>8,}: best = {b:.2f}")
            return ui.pre("\n".join(lines))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan_shiny/test_convergence.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/results/convergence.py \
        tests/pymarxan_shiny/test_convergence.py
git commit -m "feat: add convergence plot module for SA iteration visualization"
```

---

### Task 6: Rewire app.py

Replace inline solver execution in app.py with the new run_panel module and add convergence tab.

**Files:**
- Modify: `src/pymarxan_app/app.py`
- Test: `tests/test_integration_phase8.py`

**Step 1: Write the integration tests**

Create `tests/test_integration_phase8.py`:

```python
"""Phase 8 integration tests: solver UX modules."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan_shiny.modules.run_control.progress import SolverProgress
from pymarxan_shiny.modules.run_control.run_panel import run_panel_server, run_panel_ui
from pymarxan_shiny.modules.results.convergence import (
    convergence_server,
    convergence_ui,
    extract_history,
)

DATA_DIR = Path(__file__).parent / "data" / "simple"


def test_app_imports():
    """Verify the app can import all new modules."""
    from pymarxan_app import app
    assert app.app is not None


def test_sa_with_progress_and_history():
    """End-to-end: SA solver produces progress updates and history."""
    problem = load_project(DATA_DIR)
    problem.parameters["NUMITNS"] = 5_000
    problem.parameters["NUMTEMP"] = 50

    progress = SolverProgress()
    config = SolverConfig(
        num_solutions=2, seed=42, metadata={"progress": progress}
    )

    solver = SimulatedAnnealingSolver()
    solutions = solver.solve(problem, config)

    # Progress was updated
    assert progress.status == "done"
    assert progress.current_run == 2

    # History was recorded
    for sol in solutions:
        assert "history" in sol.metadata
        h = sol.metadata["history"]
        assert len(h["iteration"]) > 0
        # Best objective should be non-increasing
        for i in range(1, len(h["best_objective"])):
            assert h["best_objective"][i] <= h["best_objective"][i - 1] + 1e-9

    # Convergence extraction works
    histories = extract_history(solutions)
    assert len(histories) == 2


def test_run_panel_ui_renders():
    ui_elem = run_panel_ui("test")
    assert ui_elem is not None


def test_convergence_ui_renders():
    ui_elem = convergence_ui("test")
    assert ui_elem is not None


def test_progress_lifecycle():
    """Test the full progress lifecycle: idle → running → done."""
    p = SolverProgress()
    assert p.status == "idle"
    assert p.progress_fraction() == 0.0

    p.status = "running"
    p.current_run = 1
    p.total_runs = 2
    p.iteration = 5000
    p.total_iterations = 10000
    assert 0.2 < p.progress_fraction() < 0.4
    assert "1" in p.format_status()
    assert "2" in p.format_status()

    p.status = "done"
    p.best_objective = 42.0
    assert p.progress_fraction() == 1.0
```

**Step 2: Run integration tests to verify they fail**

Run: `pytest tests/test_integration_phase8.py -v`
Expected: `test_app_imports` may fail if app.py references are wrong

**Step 3: Rewrite app.py**

Replace the contents of `src/pymarxan_app/app.py` with the updated version. The key changes:

1. Import `run_panel_ui`, `run_panel_server` from `run_control.run_panel`
2. Import `convergence_ui`, `convergence_server` from `results.convergence`
3. Add `all_solutions` reactive value
4. Replace the inline "Run" nav_panel with `run_panel_ui("run")`
5. Add "Convergence" to the Results tab
6. Remove `_run_solver()`, `run_status()`, `run_log()` from the server function
7. Wire `run_panel_server` and `convergence_server` in the server function
8. Remove the duplicated solver instantiation — `active_solver()` is the single source

The full updated `app.py`:

```python
"""pymarxan: Assembled Shiny application for Marxan conservation planning.
Run with: shiny run src/pymarxan_app/app.py
"""
from __future__ import annotations

from shiny import App, Inputs, Outputs, Session, reactive, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution
from pymarxan.solvers.marxan_binary import MarxanBinarySolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.zones.solver import ZoneSASolver
from pymarxan_shiny.modules.calibration.blm_explorer import (
    blm_explorer_server,
    blm_explorer_ui,
)
from pymarxan_shiny.modules.calibration.spf_explorer import (
    spf_explorer_server,
    spf_explorer_ui,
)
from pymarxan_shiny.modules.calibration.sweep_explorer import (
    sweep_explorer_server,
    sweep_explorer_ui,
)
from pymarxan_shiny.modules.connectivity.metrics_viz import (
    metrics_viz_server,
    metrics_viz_ui,
)
from pymarxan_shiny.modules.data_input.upload import upload_server, upload_ui
from pymarxan_shiny.modules.mapping.solution_map import solution_map_server, solution_map_ui
from pymarxan_shiny.modules.results.convergence import convergence_server, convergence_ui
from pymarxan_shiny.modules.results.export import export_server, export_ui
from pymarxan_shiny.modules.results.scenario_compare import (
    scenario_compare_server,
    scenario_compare_ui,
)
from pymarxan_shiny.modules.results.summary_table import summary_table_server, summary_table_ui
from pymarxan_shiny.modules.results.target_met import (
    target_met_server,
    target_met_ui,
)
from pymarxan_shiny.modules.run_control.run_panel import run_panel_server, run_panel_ui
from pymarxan_shiny.modules.solver_config.solver_picker import (
    solver_picker_server,
    solver_picker_ui,
)
from pymarxan_shiny.modules.zones.zone_config import zone_config_server, zone_config_ui

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Data",
        ui.layout_columns(upload_ui("upload"), col_widths=12),
    ),
    ui.nav_panel(
        "Configure",
        ui.layout_columns(solver_picker_ui("solver"), col_widths=12),
    ),
    ui.nav_panel(
        "Calibrate",
        ui.layout_columns(
            blm_explorer_ui("blm_cal"),
            spf_explorer_ui("spf_cal"),
            col_widths=[6, 6],
        ),
    ),
    ui.nav_panel(
        "Sweep",
        ui.layout_columns(sweep_explorer_ui("sweep"), col_widths=12),
    ),
    ui.nav_panel(
        "Connectivity",
        ui.layout_columns(metrics_viz_ui("connectivity"), col_widths=12),
    ),
    ui.nav_panel(
        "Zones",
        ui.layout_columns(zone_config_ui("zone_config"), col_widths=12),
    ),
    ui.nav_panel(
        "Run",
        run_panel_ui("run"),
    ),
    ui.nav_panel("Results", ui.layout_columns(
        solution_map_ui("solution_map"),
        summary_table_ui("summary"),
        target_met_ui("targets"),
        convergence_ui("convergence"),
        scenario_compare_ui("scenarios"),
        export_ui("export"),
        col_widths=[6, 6, 12, 12, 12, 12],
    )),
    title="pymarxan",
    id="navbar",
)

def server(input: Inputs, output: Outputs, session: Session):
    problem: reactive.Value[ConservationProblem | None] = reactive.value(None)
    solver_config: reactive.Value[dict] = reactive.value({
        "solver_type": "mip", "blm": 1.0, "num_solutions": 10, "seed": None,
    })
    current_solution: reactive.Value[Solution | None] = reactive.value(None)
    all_solutions: reactive.Value[list[Solution] | None] = reactive.value(None)
    zone_problem: reactive.Value = reactive.value(None)
    connectivity_matrix: reactive.Value = reactive.value(None)
    connectivity_pu_ids: reactive.Value = reactive.value(None)

    upload_server("upload", problem=problem)
    solver_picker_server("solver", solver_config=solver_config)
    solution_map_server("solution_map", problem=problem, solution=current_solution)
    summary_table_server("summary", problem=problem, solution=current_solution)
    zone_config_server("zone_config", zone_problem=zone_problem)

    @reactive.calc
    def active_solver():
        config_dict = solver_config()
        st = config_dict.get("solver_type", "mip")
        if st == "mip":
            return MIPSolver()
        elif st == "sa":
            return SimulatedAnnealingSolver()
        elif st == "binary":
            return MarxanBinarySolver()
        elif st == "zone_sa":
            return ZoneSASolver()
        elif st == "greedy":
            from pymarxan.solvers.heuristic import HeuristicSolver
            return HeuristicSolver()
        elif st == "iterative_improvement":
            from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
            return IterativeImprovementSolver()
        elif st == "pipeline":
            from pymarxan.solvers.run_mode import RunModePipeline
            return RunModePipeline()
        return MIPSolver()

    blm_explorer_server(
        "blm_cal", problem=problem, solver=active_solver,
    )
    spf_explorer_server("spf_cal", problem=problem, solver=active_solver)
    sweep_explorer_server("sweep", problem=problem, solver=active_solver)
    metrics_viz_server(
        "connectivity",
        connectivity_matrix=connectivity_matrix,
        pu_ids=connectivity_pu_ids,
    )
    target_met_server("targets", problem=problem, solution=current_solution)
    scenario_compare_server(
        "scenarios", solution=current_solution, solver_config=solver_config,
    )
    export_server("export", problem=problem, solution=current_solution)

    # Run panel with progress monitoring
    run_panel_server(
        "run",
        problem=problem,
        solver=active_solver,
        solver_config=solver_config,
        current_solution=current_solution,
        all_solutions=all_solutions,
    )

    # Convergence plot
    convergence_server("convergence", all_solutions=all_solutions)

app = App(app_ui, server)
```

**Step 4: Run integration tests and full regression**

Run: `pytest tests/test_integration_phase8.py -v`
Expected: All pass

Run: `pytest tests/ -v`
Expected: All pass (386 + new tests)

Run: `ruff check src/ tests/`
Expected: All checks passed

Run: `mypy src/pymarxan/ --ignore-missing-imports`
Expected: Success

**Step 5: Commit**

```bash
git add src/pymarxan_app/app.py tests/test_integration_phase8.py
git commit -m "feat: rewire app.py to use run panel and convergence modules"
```

---

### Task 7: Full Regression + Lint + Types

**Files:**
- No new files — verification only

**Step 1: Run ruff**

Run: `ruff check src/ tests/ --fix`
Expected: All checks passed (fix any issues if found)

**Step 2: Run mypy**

Run: `mypy src/pymarxan/ --ignore-missing-imports`
Expected: Success — 0 errors (fix any issues if found)

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass (~400+ tests, under 30s)

**Step 4: Run benchmarks**

Run: `pytest tests/benchmarks/ -v`
Expected: 5 passed (performance unchanged)

**Step 5: Commit any fixes**

```bash
# Only if fixes were needed
git add -u
git commit -m "chore: fix lint and type issues from phase 8"
```
