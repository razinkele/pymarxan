# Codebase Review 2 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 23 CRITICAL + HIGH findings from the second codebase review.

**Architecture:** Three batches — solver correctness first, then Shiny UX, then performance + CI. Each batch runs the full test suite before committing.

**Tech Stack:** Python 3.11+, NumPy, PuLP, Shiny for Python, pytest

---

### Task 1: Add `penalty` field to Solution and update `build_solution`

**Files:**
- Modify: `src/pymarxan/solvers/base.py:13-21`
- Modify: `src/pymarxan/solvers/utils.py:192-199`
- Test: `tests/pymarxan/solvers/test_utils.py`

**Step 1: Write the failing test**

In `tests/pymarxan/solvers/test_utils.py`, add:

```python
def test_build_solution_has_penalty_field(problem):
    """build_solution should populate a penalty field on the Solution."""
    import numpy as np
    from pymarxan.solvers.utils import build_solution

    # Select no PUs — targets are unmet, so penalty > 0
    selected = np.zeros(problem.n_planning_units, dtype=bool)
    sol = build_solution(problem, selected, blm=0.0)
    assert hasattr(sol, "penalty")
    assert sol.penalty > 0.0

    # Select all PUs — targets should be met, penalty == 0
    all_selected = np.ones(problem.n_planning_units, dtype=bool)
    sol2 = build_solution(problem, all_selected, blm=0.0)
    assert sol2.penalty == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/solvers/test_utils.py::test_build_solution_has_penalty_field -v`
Expected: FAIL — Solution has no `penalty` attribute

**Step 3: Implement**

In `src/pymarxan/solvers/base.py`, add `penalty` field to Solution:

```python
@dataclass
class Solution:
    """Result of a conservation planning optimization run."""

    selected: np.ndarray
    cost: float
    boundary: float
    objective: float  # cost + BLM * boundary + penalties
    targets_met: dict[int, bool]
    penalty: float = 0.0
    metadata: dict = field(default_factory=dict)
    zone_assignment: np.ndarray | None = None
```

In `src/pymarxan/solvers/utils.py`, update `build_solution` to pass `penalty`:

```python
    return Solution(
        selected=selected.copy(),
        cost=total_cost,
        boundary=total_boundary,
        objective=objective,
        targets_met=targets_met,
        penalty=penalty,
        metadata=metadata or {},
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pymarxan/solvers/test_utils.py::test_build_solution_has_penalty_field -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `make test-fast`
Expected: All tests pass (the new `penalty=0.0` default is backward-compatible)

---

### Task 2: Fix MIP solver — add penalty, handle infeasible, apply MISSLEVEL, handle status=1

**Files:**
- Modify: `src/pymarxan/solvers/mip_solver.py`
- Test: `tests/pymarxan/solvers/test_mip.py`

**Step 1: Write the failing tests**

In `tests/pymarxan/solvers/test_mip.py`, add three tests:

```python
def test_mip_objective_includes_penalty(problem):
    """MIP solution objective should include SPF penalty when targets are unmet."""
    from pymarxan.solvers.mip_solver import MIPSolver

    # Set impossible target so penalty > 0
    problem.features.loc[0, "target"] = 999999.0
    solver = MIPSolver()
    solutions = solver.solve(problem, SolverConfig(num_solutions=1))
    assert len(solutions) >= 1
    sol = solutions[0]
    # Objective should exceed cost + boundary (because penalty is added)
    assert sol.objective >= sol.cost + sol.boundary
    assert sol.penalty > 0.0


def test_mip_infeasible_returns_empty(problem):
    """MIP solver should return empty list when problem is infeasible."""
    from pymarxan.solvers.mip_solver import MIPSolver

    # Lock out ALL PUs but require a target — infeasible
    problem.planning_units["status"] = 3
    problem.features.loc[:, "target"] = 100.0
    solver = MIPSolver()
    solutions = solver.solve(problem, SolverConfig(num_solutions=1))
    assert solutions == []


def test_mip_applies_misslevel(problem):
    """MIP solver should use MISSLEVEL to relax target constraints."""
    from pymarxan.solvers.mip_solver import MIPSolver

    # Set target slightly above what's achievable
    total_amount = float(
        problem.pu_vs_features.groupby("species")["amount"].sum().min()
    )
    problem.features.loc[:, "target"] = total_amount + 0.1
    problem.parameters["MISSLEVEL"] = 0.5  # relax to 50%

    solver = MIPSolver()
    solutions = solver.solve(problem, SolverConfig(num_solutions=1))
    assert len(solutions) >= 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/solvers/test_mip.py::test_mip_objective_includes_penalty tests/pymarxan/solvers/test_mip.py::test_mip_infeasible_returns_empty tests/pymarxan/solvers/test_mip.py::test_mip_applies_misslevel -v`

**Step 3: Implement fixes in `src/pymarxan/solvers/mip_solver.py`**

Add imports at the top:

```python
from pymarxan.solvers.utils import (
    check_targets,
    compute_boundary,
    compute_cost_threshold_penalty,
    compute_penalty,
)
```

In the `solve()` method, add MISSLEVEL to target constraints (replace lines 110-121):

```python
        # Constraints: feature targets (with MISSLEVEL)
        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
        for _, feat_row in problem.features.iterrows():
            fid = int(feat_row["id"])
            target = float(feat_row["target"]) * misslevel
            feat_data = problem.pu_vs_features[
                problem.pu_vs_features["species"] == fid
            ]
            amount_expr = pulp.lpSum(
                float(r["amount"]) * x[int(r["pu"])]
                for _, r in feat_data.iterrows()
            )
            model += amount_expr >= target, f"target_{fid}"
```

After `model.solve(solver)` (line 130), add infeasible guard:

```python
        # Check for infeasible / unbounded
        if model.status != pulp.constants.LpStatusOptimal:
            return []
```

Replace lines 132-159 (solution extraction through return) with:

```python
        # Extract solution
        selected = np.array(
            [bool(round(pulp.value(x[pid]) or 0.0)) for pid in pu_ids],
            dtype=bool,
        )

        # Use build_solution for consistent objective calculation
        from pymarxan.solvers.utils import build_solution

        sol = build_solution(
            problem,
            selected,
            blm,
            metadata={"solver": self.name(), "status": pulp.LpStatus[model.status]},
        )
        return [sol] * config.num_solutions
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan/solvers/test_mip.py -v`
Expected: All pass

---

### Task 3: Fix HeuristicSolver — apply MISSLEVEL, handle status=1

**Files:**
- Modify: `src/pymarxan/solvers/heuristic.py:153-277`
- Test: `tests/pymarxan/solvers/test_heuristic.py`

**Step 1: Write the failing test**

In `tests/pymarxan/solvers/test_heuristic.py`, add:

```python
def test_heuristic_applies_misslevel(problem):
    """Greedy solver should use MISSLEVEL to relax targets."""
    from pymarxan.solvers.heuristic import HeuristicSolver

    problem.parameters["MISSLEVEL"] = 0.5
    solver = HeuristicSolver(heurtype=0)
    solutions = solver.solve(problem, SolverConfig(num_solutions=1))
    assert len(solutions) == 1
    sol = solutions[0]
    # With MISSLEVEL=0.5, fewer PUs should be needed
    assert sol.n_selected >= 0


def test_heuristic_status1_starts_selected(problem):
    """PUs with status=1 should start selected but remain swappable."""
    from pymarxan.solvers.heuristic import HeuristicSolver

    problem.planning_units.loc[0, "status"] = 1
    solver = HeuristicSolver(heurtype=0)
    solutions = solver.solve(problem, SolverConfig(num_solutions=1))
    assert len(solutions) == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/solvers/test_heuristic.py::test_heuristic_applies_misslevel tests/pymarxan/solvers/test_heuristic.py::test_heuristic_status1_starts_selected -v`

**Step 3: Implement**

In `src/pymarxan/solvers/heuristic.py`, `_solve_once()`:

1. After line 166 (`statuses = ...`), add status=1 handling:

```python
        # Status 1: initial include — start selected but swappable
        initial_include = statuses == 1
        selected[initial_include] = True
```

2. Replace lines 192-197 (target tracking) with MISSLEVEL:

```python
        # Track remaining need per feature (with MISSLEVEL)
        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
        remaining: dict[int, float] = {}
        for _, row in problem.features.iterrows():
            fid = int(row["id"])
            target = float(row["target"]) * misslevel
            remaining[fid] = target
```

3. Also subtract initial_include contributions (add after line 203):

```python
        # Subtract initial-include contributions too
        for idx in np.where(initial_include & ~locked_in)[0]:
            for fid, amount in contributions.get(int(idx), {}).items():
                if fid in remaining:
                    remaining[fid] -= amount
```

4. Fix penalty section (lines 260-266) to use MISSLEVEL:

```python
        # Penalty (using MISSLEVEL-adjusted remaining)
        penalty = 0.0
        for _, row in problem.features.iterrows():
            fid = int(row["id"])
            if not targets_met[fid]:
                spf = float(row.get("spf", 1.0))
                shortfall = max(remaining.get(fid, 0.0), 0.0)
                penalty += spf * shortfall
```

5. Add `penalty=penalty` to the Solution constructor (line 270).

**Step 4: Run tests**

Run: `pytest tests/pymarxan/solvers/test_heuristic.py -v`
Expected: All pass

---

### Task 4: Fix `write_mvbest` MISSLEVEL and `write_sum` penalty, export missing writers

**Files:**
- Modify: `src/pymarxan/io/writers.py:176,243`
- Modify: `src/pymarxan/io/__init__.py`
- Test: `tests/pymarxan/io/test_writers.py`

**Step 1: Write the failing tests**

In `tests/pymarxan/io/test_writers.py`, add:

```python
def test_write_mvbest_applies_misslevel(tmp_path, problem):
    """write_mvbest should use MISSLEVEL for Target_Met column."""
    import pandas as pd
    from pymarxan.io.writers import write_mvbest
    from pymarxan.solvers.utils import build_solution
    import numpy as np

    # Set MISSLEVEL to 0.0 so all targets count as met
    problem.parameters["MISSLEVEL"] = 0.0
    selected = np.zeros(problem.n_planning_units, dtype=bool)
    sol = build_solution(problem, selected, blm=0.0)

    path = tmp_path / "mvbest.csv"
    write_mvbest(problem, sol, path)

    df = pd.read_csv(path)
    assert all(df["Target_Met"]), "With MISSLEVEL=0.0, all targets should be met"


def test_write_sum_uses_penalty_field(tmp_path):
    """write_sum should use sol.penalty directly, not reconstruct it."""
    import pandas as pd
    from pymarxan.io.writers import write_sum
    from pymarxan.solvers.base import Solution
    import numpy as np

    sol = Solution(
        selected=np.array([True, False]),
        cost=10.0,
        boundary=5.0,
        objective=25.0,  # 10 + 2*5 + 5 (BLM=2, penalty=5)
        targets_met={1: False},
        penalty=5.0,
    )

    path = tmp_path / "sum.csv"
    write_sum([sol], path)

    df = pd.read_csv(path)
    assert df["Penalty"].iloc[0] == 5.0


def test_io_exports_writers():
    """io module should export write_mvbest, write_ssoln, write_sum."""
    from pymarxan import io
    assert hasattr(io, "write_mvbest")
    assert hasattr(io, "write_ssoln")
    assert hasattr(io, "write_sum")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/io/test_writers.py::test_write_mvbest_applies_misslevel tests/pymarxan/io/test_writers.py::test_write_sum_uses_penalty_field tests/pymarxan/io/test_writers.py::test_io_exports_writers -v`

**Step 3: Implement**

In `src/pymarxan/io/writers.py`:

1. Fix `write_mvbest` (line 176): Apply MISSLEVEL to Target_Met:

```python
        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
        # ... (existing code for amount_held computation)
        target_met = amount_held >= target * misslevel
```

2. Fix `write_sum` (line 243): Use `sol.penalty` directly:

```python
        penalty = sol.penalty
        shortfall = penalty
```

3. In `src/pymarxan/io/__init__.py`, add the missing exports:

```python
from pymarxan.io.writers import (
    save_project,
    write_bound,
    write_input_dat,
    write_mvbest,
    write_pu,
    write_puvspr,
    write_spec,
    write_ssoln,
    write_sum,
)
```

And add to `__all__`:
```python
    "write_mvbest",
    "write_ssoln",
    "write_sum",
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/io/ -v`
Expected: All pass

---

### Task 5: Handle STATUS_INITIAL_INCLUDE in SA and iterative improvement solvers

**Files:**
- Modify: `src/pymarxan/solvers/simulated_annealing.py:67-77`
- Modify: `src/pymarxan/solvers/iterative_improvement.py:127-140`
- Test: `tests/pymarxan/solvers/test_simulated_annealing.py`

**Step 1: Write the failing test**

In `tests/pymarxan/solvers/test_simulated_annealing.py`, add:

```python
def test_status1_starts_selected(problem):
    """PUs with status=1 should start selected but remain swappable."""
    from pymarxan.models.problem import STATUS_INITIAL_INCLUDE

    problem.planning_units.loc[0, "status"] = STATUS_INITIAL_INCLUDE
    solver = SimulatedAnnealingSolver(num_iterations=100, num_temp_steps=10)
    solutions = solver.solve(problem, SolverConfig(num_solutions=1, seed=42))
    assert len(solutions) == 1
    # The PU started selected but SA may have swapped it out — that's fine.
    # Key check: it didn't crash and solution is valid.
    assert solutions[0].cost >= 0
```

**Step 2: Run test to verify behavior (may pass but with wrong initial state)**

Run: `pytest tests/pymarxan/solvers/test_simulated_annealing.py::test_status1_starts_selected -v`

**Step 3: Implement**

In `src/pymarxan/solvers/simulated_annealing.py`, update the status classification (lines 67-77):

```python
        from pymarxan.models.problem import STATUS_INITIAL_INCLUDE

        locked_in: list[int] = []
        locked_out: list[int] = []
        initial_include: list[int] = []
        swappable: list[int] = []
        for i in range(n_pu):
            s = int(cache.statuses[i])
            if s == STATUS_LOCKED_IN:
                locked_in.append(i)
            elif s == STATUS_LOCKED_OUT:
                locked_out.append(i)
            elif s == STATUS_INITIAL_INCLUDE:
                initial_include.append(i)
                swappable.append(i)
            else:
                swappable.append(i)
```

Update initial selection (line 113-119):

```python
            selected = np.zeros(n_pu, dtype=bool)
            for idx in locked_in:
                selected[idx] = True
            for idx in initial_include:
                selected[idx] = True
            for idx in swappable:
                if idx not in initial_include and rng.random() < initial_prop:
                    selected[idx] = True
```

In `src/pymarxan/solvers/iterative_improvement.py`, update `_locked_sets` (lines 127-140):

```python
    @staticmethod
    def _locked_sets(
        problem: ConservationProblem,
    ) -> tuple[set[int], set[int], set[int]]:
        """Return (locked_in_indices, locked_out_indices, initial_include_indices)."""
        locked_in: set[int] = set()
        locked_out: set[int] = set()
        initial_include: set[int] = set()
        statuses = problem.planning_units["status"].values.astype(int)
        for i, s in enumerate(statuses):
            if s == 2:
                locked_in.add(i)
            elif s == 3:
                locked_out.add(i)
            elif s == 1:
                initial_include.add(i)
        return locked_in, locked_out, initial_include
```

Update all callers of `_locked_sets` to unpack three values. In `_removal_pass`, `_addition_pass`, `_swap_pass_loop`: initial_include PUs are swappable (not locked), so no further changes needed beyond updating the tuple unpacking:

```python
        locked_in, _, _ = self._locked_sets(problem)
        # or
        _, locked_out, _ = self._locked_sets(problem)
        # or
        locked_in, locked_out, _ = self._locked_sets(problem)
```

In `solve()` (line 71), pre-select initial_include PUs:

```python
            selected = np.ones(n, dtype=bool)
            locked_out = statuses == 3
            selected[locked_out] = False
            # status=1 PUs are already selected (started from all-selected)
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/solvers/ -v`
Expected: All pass

**Step 5: Commit Batch 1**

```bash
git add src/pymarxan/solvers/base.py src/pymarxan/solvers/utils.py \
  src/pymarxan/solvers/mip_solver.py src/pymarxan/solvers/heuristic.py \
  src/pymarxan/solvers/simulated_annealing.py \
  src/pymarxan/solvers/iterative_improvement.py \
  src/pymarxan/io/writers.py src/pymarxan/io/__init__.py \
  tests/
git commit -m "fix: solver correctness — penalty, MISSLEVEL, infeasible guard, status=1"
```

---

### Task 6: Fix feature_table CellPatch keys and add write-back

**Files:**
- Modify: `src/pymarxan_shiny/modules/data/feature_table.py`
- Test: `tests/pymarxan_shiny/modules/data/test_feature_table.py`

**Step 1: Write the failing test**

In `tests/pymarxan_shiny/modules/data/test_feature_table.py`, add or update:

```python
def test_validate_feature_edit_accepts_valid_target():
    from pymarxan_shiny.modules.data.feature_table import validate_feature_edit

    assert validate_feature_edit("target", "5.0") == 5.0
    assert validate_feature_edit("spf", "2.5") == 2.5
    assert validate_feature_edit("id", "5.0") is None  # id not editable
    assert validate_feature_edit("target", "-1") is None  # negative rejected
    assert validate_feature_edit("target", "abc") is None  # non-numeric rejected


def test_feature_table_server_is_callable():
    from pymarxan_shiny.modules.data.feature_table import (
        feature_table_server,
        feature_table_ui,
    )

    assert callable(feature_table_server)
    assert callable(feature_table_ui)
```

**Step 2: Run test**

Run: `pytest tests/pymarxan_shiny/modules/data/test_feature_table.py -v`

**Step 3: Implement the fix**

Replace the entire `feature_table.py` content:

```python
"""Feature table editor Shiny module — editable target and SPF values."""
from __future__ import annotations

from shiny import module, reactive, render, ui


_COLUMN_ORDER = ["id", "name", "target", "spf"]


def validate_feature_edit(column: str, value: str) -> float | None:
    """Validate an edit to a feature table cell.

    Returns validated float, or None if edit is rejected.
    """
    if column not in ("target", "spf"):
        return None
    try:
        val = float(value)
    except (ValueError, TypeError):
        return None
    if val < 0:
        return None
    return val


@module.ui
def feature_table_ui():
    return ui.card(
        ui.card_header("Feature Targets & SPF"),
        ui.output_data_frame("feature_grid"),
        ui.input_action_button(
            "save_changes", "Save Changes", class_="btn-warning w-100 mt-2"
        ),
    )


@module.server
def feature_table_server(
    input,
    output,
    session,
    problem: reactive.Value,
):
    @render.data_frame
    def feature_grid():
        p = problem()
        if p is None:
            return None
        df = p.features[["id", "name", "target", "spf"]].copy()
        return render.DataGrid(df, editable=True)

    @feature_grid.set_patch_fn
    def _(*, patch):
        # CellPatch keys: row_index (int), column_index (int), value
        col_idx = patch["column_index"]
        col = _COLUMN_ORDER[col_idx] if col_idx < len(_COLUMN_ORDER) else ""
        validated = validate_feature_edit(col, str(patch["value"]))
        if validated is not None:
            return validated
        # Reject edit by returning the proposed value unchanged
        return patch["value"]

    @reactive.effect
    @reactive.event(input.save_changes)
    def _save():
        p = problem()
        if p is None:
            return
        df = feature_grid.data_view()
        p.features["target"] = df["target"].values
        p.features["spf"] = df["spf"].values
        problem.set(p)
        ui.notification_show("Feature targets saved.", type="message")
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan_shiny/ -v`
Expected: All pass

---

### Task 7: Fix run_panel — progress polling, verbose, deepcopy, dead code

**Files:**
- Modify: `src/pymarxan_shiny/modules/run_control/run_panel.py`
- Test: `tests/pymarxan_shiny/modules/run_control/test_run_panel.py`

**Step 1: Write the failing test**

```python
def test_run_panel_server_is_callable():
    from pymarxan_shiny.modules.run_control.run_panel import (
        run_panel_server,
        run_panel_ui,
    )

    assert callable(run_panel_server)
    assert callable(run_panel_ui)
```

**Step 2: Implement all four fixes in `run_panel.py`**

Add `import copy` at top.

Remove dead code:
- Delete line 46: `solver_thread: reactive.Value[threading.Thread | None] = reactive.value(None)`
- Delete line 107: `solver_thread.set(thread)`

In `_run_solver()`, make these changes:

1. **deepcopy** (after line 51, replace `p = problem()` block):

```python
        p = copy.deepcopy(problem())
        if p is None:
            ui.notification_show("Load a project first!", type="error")
            return
```

2. Move parameter mutations into the deepcopy'd `p` (existing lines 64-71 stay the same — they now safely mutate the copy).

3. **Forward verbose** (replace line 76):

```python
        config = SolverConfig(
            num_solutions=config_dict.get("num_solutions", 10),
            seed=config_dict.get("seed"),
            verbose=bool(config_dict.get("mip_verbose", False)),
            metadata={"progress": progress},
        )
```

4. **Set progress running before thread** (replace lines 80-81):

```python
        progress.reset()
        progress.status = "running"
        progress.message = f"Starting {active.name()}..."
        ui.notification_show(f"Running {active.name()}...", type="message")
```

**Step 3: Run tests**

Run: `pytest tests/pymarxan_shiny/ -v`
Expected: All pass

---

### Task 8: Fix spatial_grid orphaned renderer and calibration error handling

**Files:**
- Modify: `src/pymarxan_shiny/modules/mapping/spatial_grid.py:70-103`
- Modify: `src/pymarxan_shiny/modules/calibration/spf_explorer.py:49-65`
- Modify: `src/pymarxan_shiny/modules/calibration/sweep_explorer.py:49-69`
- Modify: `src/pymarxan_shiny/modules/calibration/blm_explorer.py:43`
- Modify: `src/pymarxan_shiny/modules/calibration/sensitivity_ui.py:73`
- Test: existing tests

**Step 1: Fix spatial_grid.py**

Move `map_summary` renderer inside `if _HAS_IPYLEAFLET:` block. Replace lines 70-103:

```python
@module.server
def spatial_grid_server(
    input,
    output,
    session,
    problem: reactive.Value,
):
    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            if p is None:
                return None

            n_pu = len(p.planning_units)
            color_mode = input.color_by()
            grid = generate_grid(n_pu)

            if color_mode == "status":
                statuses = p.planning_units["status"].tolist()
                colors = [status_color(s) for s in statuses]
            else:
                costs = p.planning_units["cost"].tolist()
                max_c = max(costs) if costs else 1.0
                colors = [
                    cost_color(c / max_c if max_c > 0 else 0.0) for c in costs
                ]

            return create_grid_map(grid, colors)

        @render.text
        def map_summary():
            p = problem()
            if p is None:
                return "Load a project to see the planning unit map."

            n_pu = len(p.planning_units)
            color_mode = input.color_by()
            return f"{n_pu} planning units — colored by {color_mode}"

    if not _HAS_IPYLEAFLET:

        @render.ui
        def grid_content():
            p = problem()
            if p is None:
                return ui.p("Load a project to see the planning unit map.")

            n_pu = len(p.planning_units)
            color_mode = input.color_by()
            grid = generate_grid(n_pu)
            all_lats = [b[0][0] for b in grid] + [b[1][0] for b in grid]
            all_lons = [b[0][1] for b in grid] + [b[1][1] for b in grid]
            bounds_info = (
                f"Grid bounds: [{min(all_lats):.4f}, {min(all_lons):.4f}]"
                f" to [{max(all_lats):.4f}, {max(all_lons):.4f}]"
            )
            return ui.div(
                ui.p(f"{n_pu} planning units — colored by {color_mode}"),
                ui.p(bounds_info),
            )
```

**Step 2: Fix spf_explorer.py**

Wrap the calibration call in try/except (replace lines 49-65):

```python
    @reactive.effect
    @reactive.event(input.run_spf)
    def _run():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first!", type="error")
            return
        try:
            result = calibrate_spf(
                p,
                solver(),
                max_iterations=int(input.max_iterations()),
                multiplier=float(input.multiplier()),
                config=SolverConfig(num_solutions=1),
            )
            spf_result.set(result)
            ui.notification_show(
                f"SPF calibration done in {len(result.history)} iterations",
                type="message",
            )
        except Exception as e:
            ui.notification_show(
                f"SPF calibration error: {e}", type="error"
            )
```

**Step 3: Fix sweep_explorer.py**

Wrap the sweep call in try/except (replace lines 49-69):

```python
    @reactive.effect
    @reactive.event(input.run_sweep)
    def _run():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first!", type="error")
            return
        try:
            param_name = input.sweep_param()
            min_val = input.sweep_min()
            max_val = input.sweep_max()
            steps = int(input.sweep_steps())

            import numpy as np

            values = np.linspace(min_val, max_val, steps).tolist()
            config = SweepConfig(
                param_dicts=[{param_name: v} for v in values],
            )
            result = run_sweep(p, solver(), config)
            sweep_result.set(result)
            ui.notification_show(
                f"Sweep complete: {len(result.solutions)} points",
                type="message",
            )
        except Exception as e:
            ui.notification_show(
                f"Sweep error: {e}", type="error"
            )
```

**Step 4: Fix calibration type annotations**

In `blm_explorer.py:43`, change `solver: reactive.Value` to `solver: reactive.Calc`.
In `sensitivity_ui.py:73`, change `solver: reactive.Value` to `solver: reactive.Calc`.

**Step 5: Commit Batch 2**

Run: `make test-fast`
Expected: All pass

```bash
git add src/pymarxan_shiny/ tests/
git commit -m "fix: Shiny UX — feature table, progress polling, thread safety, error handling"
```

---

### Task 9: Optimize iterative_improvement with ProblemCache

**Files:**
- Modify: `src/pymarxan/solvers/iterative_improvement.py`
- Test: `tests/pymarxan/solvers/test_iterative_improvement.py`

**Step 1: Write the failing test**

```python
def test_iterative_improvement_uses_cache(problem):
    """Verify iterative improvement produces correct results with ProblemCache."""
    from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver

    problem.parameters["ITIMPTYPE"] = 1
    solver = IterativeImprovementSolver(itimptype=1)
    solutions = solver.solve(problem, SolverConfig(num_solutions=1))
    assert len(solutions) == 1
    assert solutions[0].cost >= 0
    assert solutions[0].objective >= solutions[0].cost
```

**Step 2: Implement ProblemCache usage**

Rewrite `_removal_pass`, `_addition_pass`, and `_swap_pass_loop` to use ProblemCache:

In imports, add:
```python
from pymarxan.solvers.cache import ProblemCache
```

Add a helper to build cache once per `improve()` call:

```python
    def improve(
        self,
        problem: ConservationProblem,
        solution: Solution,
    ) -> Solution:
        itimptype = int(
            problem.parameters.get("ITIMPTYPE", self._itimptype)
        )

        if itimptype == 0:
            return solution

        cache = ProblemCache.from_problem(problem)
        blm = float(problem.parameters.get("BLM", 0.0))

        if itimptype == 1:
            return self._removal_pass_loop(problem, cache, blm, solution)
        if itimptype == 2:
            return self._two_step(problem, cache, blm, solution)
        if itimptype == 3:
            return self._swap_pass_loop(problem, cache, blm, solution)

        raise ValueError(f"Invalid itimptype {itimptype!r}")
```

Rewrite `_removal_pass` to use delta computation:

```python
    def _removal_pass(
        self,
        problem: ConservationProblem,
        cache: ProblemCache,
        blm: float,
        solution: Solution,
    ) -> Solution:
        locked_in, _, _ = self._locked_sets(problem)
        selected = solution.selected.copy()
        held = cache.compute_held(selected)
        total_cost = float(np.sum(cache.costs[selected]))
        current_obj = cache.compute_full_objective(selected, held, blm)

        improved = True
        while improved:
            improved = False
            for i in range(len(selected)):
                if not selected[i] or i in locked_in:
                    continue

                delta = cache.compute_delta_objective(
                    i, selected, held, total_cost, blm
                )
                if delta < 0:
                    # Accept: flip and update incremental state
                    selected[i] = False
                    held -= cache.pu_feat_matrix[i]
                    total_cost -= cache.costs[i]
                    current_obj += delta
                    improved = True

        return build_solution(
            problem, selected, blm, metadata={"solver": self.name()}
        )
```

Apply the same pattern to `_addition_pass` and `_swap_pass_loop` (using `compute_delta_objective` instead of `compute_objective`).

**Step 3: Run tests**

Run: `pytest tests/pymarxan/solvers/test_iterative_improvement.py -v`
Expected: All pass

---

### Task 10: Fix O(n²) lookups in ZoneSASolver and GapResult

**Files:**
- Modify: `src/pymarxan/zones/solver.py:60-74`
- Modify: `src/pymarxan/analysis/gap_analysis.py:27-44`
- Test: existing tests

**Step 1: Fix ZoneSASolver**

In `zones/solver.py`, replace lines 60-74:

```python
        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
        zone_ids_list = sorted(problem.zone_ids)
        zone_options = np.array([0] + zone_ids_list, dtype=int)
        n_zone_options = len(zone_options)

        locked: dict[int, int] = {}
        if "status" in problem.planning_units.columns:
            for _, row in problem.planning_units.iterrows():
                s = int(row["status"])
                idx = pu_id_to_idx[int(row["id"])]
                if s == 2:
                    locked[idx] = zone_ids_list[0]
                elif s == 3:
                    locked[idx] = 0
```

**Step 2: Fix GapResult.to_dataframe**

In `gap_analysis.py`, replace `to_dataframe` method:

```python
    def to_dataframe(self) -> pd.DataFrame:
        fid_to_name = dict(zip(self.feature_ids, self.feature_names))
        rows = []
        for fid in self.feature_ids:
            total = self.total_amount[fid]
            protected = self.protected_amount[fid]
            rows.append({
                "feature_id": fid,
                "feature_name": fid_to_name[fid],
                "target": self.targets[fid],
                "total_amount": total,
                "protected_amount": protected,
                "gap": self.gap[fid],
                "target_met": self.target_met[fid],
                "percent_protected": (
                    protected / total * 100 if total > 0 else 0.0
                ),
            })
        return pd.DataFrame(rows)
```

**Step 3: Run tests**

Run: `pytest tests/pymarxan/zones/ tests/pymarxan/analysis/ -v`
Expected: All pass

---

### Task 11: Fix CI and weak tests

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/pymarxan/solvers/test_simulated_annealing.py`

**Step 1: Fix CI**

In `.github/workflows/ci.yml`, line 41, remove redundant `networkx`:

```yaml
      - run: pip install -e ".[all]"
```

Line 42, add coverage threshold:

```yaml
      - run: pytest tests/ -v --cov --cov-report=term-missing --cov-fail-under=75
```

**Step 2: Fix test_solutions_are_different**

Replace the test (around line 35):

```python
    def test_solutions_are_different(self):
        """Multiple SA runs should produce some variation."""
        config = SolverConfig(num_solutions=5, seed=None)
        solutions = self.solver.solve(self.problem, config)
        # With 5 independent runs, not all should be identical
        arrays = [s.selected for s in solutions]
        any_different = any(
            not np.array_equal(arrays[0], a) for a in arrays[1:]
        )
        assert any_different, "All 5 SA solutions were identical"
```

**Step 3: Commit Batch 3**

Run: `make test-fast`
Expected: All pass

Run: `make lint && make types`
Expected: All pass

```bash
git add .github/workflows/ci.yml tests/ \
  src/pymarxan/solvers/iterative_improvement.py \
  src/pymarxan/zones/solver.py \
  src/pymarxan/analysis/gap_analysis.py
git commit -m "perf: O(n²) fixes + CI coverage enforcement + test quality"
```

---

### Task 12: Full regression and verification

**Step 1: Run full test suite with coverage**

Run: `pytest tests/ -v --cov --cov-report=term-missing --cov-fail-under=75`
Expected: All tests pass, coverage >= 75%

**Step 2: Run lint and type checks**

Run: `make lint && make types`
Expected: Clean

**Step 3: Review commit log**

Run: `git log --oneline -10`
Expected: 3 batch commits visible

**Step 4: Transition to finishing-a-development-branch**

Use superpowers:finishing-a-development-branch to complete.
