# Phase 5: Core Completeness & UI Polish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fill remaining design-doc gaps — add sensitivity analysis, greedy heuristic solver, gap analysis, and build Shiny modules for SPF calibration, connectivity visualization, and target achievement tracking.

**Architecture:** Follows the established three-layer pattern. New core modules in `pymarxan/` with corresponding Shiny modules in `pymarxan_shiny/`. Each core module is a pure function or dataclass with no UI dependencies. Each Shiny module wraps a core module with `@module.ui` / `@module.server`.

**Tech Stack:** Python 3.11+, NumPy, Pandas, Shiny for Python, networkx

---

### Task 1: Sensitivity Analysis Module

**Files:**
- Create: `src/pymarxan/calibration/sensitivity.py`
- Test: `tests/pymarxan/calibration/test_sensitivity.py`

**Context:** Sensitivity analysis varies feature targets (e.g., ±10%, ±20%) and measures how solutions change. This tells practitioners which targets drive the solution. The existing `calibrate_blm()` and `run_sweep()` patterns provide the template — iterate over parameter variations, solve, collect results.

**Step 1: Write the failing test**

```python
"""Tests for sensitivity analysis module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.calibration.sensitivity import (
    SensitivityConfig,
    SensitivityResult,
    run_sensitivity,
)
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


class _StubSolver(Solver):
    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        n = problem.n_planning_units
        total_target = sum(
            float(r["target"]) for _, r in problem.features.iterrows()
        )
        return [
            Solution(
                selected=np.ones(n, dtype=bool),
                cost=total_target * 2,
                boundary=5.0,
                objective=total_target * 2 + 5.0,
                targets_met={
                    int(r["id"]): True for _, r in problem.features.iterrows()
                },
            )
        ]

    def name(self) -> str:
        return "stub"

    def supports_zones(self) -> bool:
        return False


@pytest.fixture()
def problem() -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [10.0, 20.0, 15.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2],
        "name": ["bird", "mammal"],
        "target": [5.0, 8.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 2, 3],
        "amount": [3.0, 4.0, 5.0, 6.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
    )


def test_sensitivity_default_multipliers(problem: ConservationProblem):
    """Default config varies targets by ±10% and ±20%."""
    config = SensitivityConfig()
    result = run_sensitivity(problem, _StubSolver(), config)
    assert isinstance(result, SensitivityResult)
    # 2 features x 5 multipliers (0.8, 0.9, 1.0, 1.1, 1.2) = 10 runs
    assert len(result.runs) == 10


def test_sensitivity_custom_multipliers(problem: ConservationProblem):
    config = SensitivityConfig(multipliers=[0.5, 1.0, 1.5])
    result = run_sensitivity(problem, _StubSolver(), config)
    assert len(result.runs) == 6  # 2 features x 3 multipliers


def test_sensitivity_single_feature(problem: ConservationProblem):
    config = SensitivityConfig(
        feature_ids=[1], multipliers=[0.5, 1.0, 2.0],
    )
    result = run_sensitivity(problem, _StubSolver(), config)
    assert len(result.runs) == 3  # 1 feature x 3 multipliers


def test_sensitivity_to_dataframe(problem: ConservationProblem):
    config = SensitivityConfig(multipliers=[0.8, 1.0, 1.2])
    result = run_sensitivity(problem, _StubSolver(), config)
    df = result.to_dataframe()
    assert "feature_id" in df.columns
    assert "multiplier" in df.columns
    assert "cost" in df.columns
    assert "objective" in df.columns
    assert len(df) == 6


def test_sensitivity_baseline_at_1(problem: ConservationProblem):
    """Multiplier 1.0 should use original targets."""
    config = SensitivityConfig(multipliers=[1.0])
    result = run_sensitivity(problem, _StubSolver(), config)
    # Two runs (one per feature), both at multiplier 1.0
    for run in result.runs:
        assert run["multiplier"] == 1.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/calibration/test_sensitivity.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Target sensitivity analysis for conservation planning.

Varies feature targets (e.g., ±10%, ±20%) and measures how the
optimal solution changes. Helps practitioners understand which
targets drive the solution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


@dataclass
class SensitivityConfig:
    """Configuration for a target sensitivity analysis."""

    feature_ids: list[int] | None = None
    multipliers: list[float] = field(
        default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2]
    )
    solver_config: SolverConfig | None = None


@dataclass
class SensitivityResult:
    """Results of a target sensitivity analysis."""

    runs: list[dict]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.runs)


def run_sensitivity(
    problem: ConservationProblem,
    solver: Solver,
    config: SensitivityConfig,
) -> SensitivityResult:
    """Run target sensitivity analysis."""
    solver_config = config.solver_config or SolverConfig(num_solutions=1)

    if config.feature_ids is not None:
        feature_ids = config.feature_ids
    else:
        feature_ids = problem.features["id"].tolist()

    runs: list[dict] = []

    for fid in feature_ids:
        original_target = float(
            problem.features.loc[problem.features["id"] == fid, "target"].iloc[0]
        )
        for mult in config.multipliers:
            features_df = problem.features.copy()
            features_df.loc[features_df["id"] == fid, "target"] = (
                original_target * mult
            )
            modified = ConservationProblem(
                planning_units=problem.planning_units,
                features=features_df,
                pu_vs_features=problem.pu_vs_features,
                boundary=problem.boundary,
                parameters=problem.parameters,
            )
            sols = solver.solve(modified, solver_config)
            best = min(sols, key=lambda s: s.objective)
            runs.append({
                "feature_id": fid,
                "multiplier": mult,
                "target": original_target * mult,
                "cost": best.cost,
                "boundary": best.boundary,
                "objective": best.objective,
                "n_selected": best.n_selected,
                "all_targets_met": best.all_targets_met,
            })

    return SensitivityResult(runs=runs)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pymarxan/calibration/test_sensitivity.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan/calibration/sensitivity.py tests/pymarxan/calibration/test_sensitivity.py
git commit -m "feat: add target sensitivity analysis module"
```

---

### Task 2: Greedy Heuristic Solver

**Files:**
- Create: `src/pymarxan/solvers/heuristic.py`
- Test: `tests/pymarxan/solvers/test_heuristic.py`

**Context:** A fast greedy solver that selects planning units one-by-one based on cost-effectiveness. Useful as a quick baseline solution and for seeding SA. Follows the `Solver` ABC from `base.py`. Also register it in the default registry.

**Step 1: Write the failing test**

```python
"""Tests for greedy heuristic solver."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver


@pytest.fixture()
def simple_problem() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 20.0, 15.0, 25.0],
        "status": [0, 0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1, 2],
        "name": ["f1", "f2"],
        "target": [5.0, 4.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 3, 4],
        "amount": [3.0, 4.0, 5.0, 3.0],
    })
    bnd = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "boundary": [1.0, 1.0]})
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=bnd,
        parameters={"BLM": 0.0},
    )


def test_heuristic_solver_returns_solution(simple_problem: ConservationProblem):
    solver = HeuristicSolver()
    sols = solver.solve(simple_problem)
    assert len(sols) == 1
    assert sols[0].selected.dtype == bool


def test_heuristic_meets_targets(simple_problem: ConservationProblem):
    solver = HeuristicSolver()
    sols = solver.solve(simple_problem)
    assert sols[0].all_targets_met


def test_heuristic_prefers_cheap_units(simple_problem: ConservationProblem):
    """Greedy solver should prefer cheaper planning units."""
    solver = HeuristicSolver()
    sols = solver.solve(simple_problem)
    # PU 1 (cost=10, amount=3) and PU 3 (cost=15, amount=5) are most cost-effective
    selected_ids = set(
        simple_problem.planning_units.loc[sols[0].selected, "id"]
    )
    # The cheapest way to meet both targets should not include PU 4 (cost=25)
    assert 4 not in selected_ids


def test_heuristic_name():
    assert HeuristicSolver().name() == "greedy"


def test_heuristic_supports_zones():
    assert HeuristicSolver().supports_zones() is False


def test_heuristic_locked_in(simple_problem: ConservationProblem):
    """Status 2 PUs are always selected."""
    simple_problem.planning_units.loc[
        simple_problem.planning_units["id"] == 4, "status"
    ] = 2
    solver = HeuristicSolver()
    sols = solver.solve(simple_problem)
    idx = simple_problem.planning_units.index[
        simple_problem.planning_units["id"] == 4
    ][0]
    assert sols[0].selected[idx]


def test_heuristic_locked_out(simple_problem: ConservationProblem):
    """Status 3 PUs are never selected."""
    simple_problem.planning_units.loc[
        simple_problem.planning_units["id"] == 1, "status"
    ] = 3
    solver = HeuristicSolver()
    sols = solver.solve(simple_problem)
    idx = simple_problem.planning_units.index[
        simple_problem.planning_units["id"] == 1
    ][0]
    assert not sols[0].selected[idx]


def test_heuristic_multiple_solutions(simple_problem: ConservationProblem):
    """num_solutions > 1 returns multiple solutions with randomness."""
    solver = HeuristicSolver()
    config = SolverConfig(num_solutions=3, seed=42)
    sols = solver.solve(simple_problem, config)
    assert len(sols) == 3
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/solvers/test_heuristic.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Greedy heuristic solver for conservation planning.

Selects planning units one-by-one based on cost-effectiveness
(marginal contribution per unit cost). Fast baseline for comparison
with SA and MIP solvers.
"""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


class HeuristicSolver(Solver):
    """Greedy cost-effectiveness heuristic solver."""

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig(num_solutions=1)

        rng = np.random.default_rng(config.seed)
        solutions = []

        for _ in range(config.num_solutions):
            sol = self._solve_once(problem, rng)
            solutions.append(sol)

        return solutions

    def _solve_once(
        self,
        problem: ConservationProblem,
        rng: np.random.Generator,
    ) -> Solution:
        n = problem.n_planning_units
        pu_ids = problem.planning_units["id"].values
        costs = problem.planning_units["cost"].values.astype(float)
        statuses = problem.planning_units["status"].values.astype(int)

        selected = np.zeros(n, dtype=bool)

        # Lock-in (status 2) and lock-out (status 3)
        locked_in = statuses == 2
        locked_out = statuses == 3
        selected[locked_in] = True

        # Build feature contribution lookup: pu_index -> {fid: amount}
        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
        contributions: dict[int, dict[int, float]] = {}
        for _, row in problem.pu_vs_features.iterrows():
            pid = int(row["pu"])
            fid = int(row["species"])
            amount = float(row["amount"])
            idx = pu_id_to_idx.get(pid)
            if idx is not None:
                contributions.setdefault(idx, {})[fid] = amount

        # Track remaining need per feature
        remaining: dict[int, float] = {}
        for _, row in problem.features.iterrows():
            fid = int(row["id"])
            target = float(row["target"])
            remaining[fid] = target

        # Subtract locked-in contributions
        for idx in np.where(selected)[0]:
            for fid, amount in contributions.get(int(idx), {}).items():
                if fid in remaining:
                    remaining[fid] -= amount

        # Greedy loop: select most cost-effective PU until all targets met
        available = np.where(~selected & ~locked_out)[0]
        # Add small noise for diversity across runs
        noise = rng.uniform(0.0, 0.01, size=n)

        while any(r > 0 for r in remaining.values()) and len(available) > 0:
            best_idx = -1
            best_score = -1.0

            for idx in available:
                marginal = 0.0
                for fid, amount in contributions.get(int(idx), {}).items():
                    if remaining.get(fid, 0.0) > 0:
                        marginal += min(amount, remaining[fid])
                cost = max(costs[idx], 1e-10)
                score = marginal / cost + noise[idx]
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx < 0 or best_score <= 0:
                break

            selected[best_idx] = True
            for fid, amount in contributions.get(int(best_idx), {}).items():
                if fid in remaining:
                    remaining[fid] -= amount
            available = available[available != best_idx]

        # Compute objective
        blm = float(problem.parameters.get("BLM", 0.0))
        total_cost = float(costs[selected].sum())

        boundary_val = 0.0
        if problem.boundary is not None and blm > 0:
            for _, row in problem.boundary.iterrows():
                i = pu_id_to_idx.get(int(row["id1"]))
                j = pu_id_to_idx.get(int(row["id2"]))
                if i is not None and j is not None:
                    if selected[i] != selected[j]:
                        boundary_val += float(row["boundary"])

        # Check targets
        targets_met: dict[int, bool] = {}
        for _, row in problem.features.iterrows():
            fid = int(row["id"])
            targets_met[fid] = remaining.get(fid, 0.0) <= 0

        # Penalty
        penalty = 0.0
        for _, row in problem.features.iterrows():
            fid = int(row["id"])
            if not targets_met[fid]:
                spf = float(row.get("spf", 1.0))
                shortfall = max(remaining.get(fid, 0.0), 0.0)
                penalty += spf * shortfall

        objective = total_cost + blm * boundary_val + penalty

        return Solution(
            selected=selected,
            cost=total_cost,
            boundary=boundary_val,
            objective=objective,
            targets_met=targets_met,
            metadata={"solver": "greedy"},
        )

    def name(self) -> str:
        return "greedy"

    def supports_zones(self) -> bool:
        return False
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pymarxan/solvers/test_heuristic.py -v`
Expected: 8 PASSED

**Step 5: Register in default registry**

Edit `src/pymarxan/solvers/registry.py` — add `from pymarxan.solvers.heuristic import HeuristicSolver` inside `get_default_registry()` and register it as `"greedy"`:

```python
reg.register("greedy", HeuristicSolver)
```

**Step 6: Commit**

```bash
git add src/pymarxan/solvers/heuristic.py tests/pymarxan/solvers/test_heuristic.py src/pymarxan/solvers/registry.py
git commit -m "feat: add greedy heuristic solver and register in default registry"
```

---

### Task 3: Gap Analysis Module

**Files:**
- Create: `src/pymarxan/analysis/gap_analysis.py`
- Test: `tests/pymarxan/analysis/test_gap_analysis.py`

**Context:** Gap analysis compares current protection levels (PUs with status=2, i.e., already protected) against feature targets. It tells practitioners how much of each feature is already protected and how much additional protection is needed.

**Step 1: Write the failing test**

```python
"""Tests for gap analysis module."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.analysis.gap_analysis import GapResult, compute_gap_analysis
from pymarxan.models.problem import ConservationProblem


@pytest.fixture()
def problem_with_protection() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 20.0, 15.0, 25.0],
        "status": [2, 0, 2, 0],  # PUs 1 and 3 are protected (status=2)
    })
    feat = pd.DataFrame({
        "id": [1, 2],
        "name": ["bird", "mammal"],
        "target": [5.0, 8.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 2, 2, 2],
        "pu": [1, 2, 3, 2, 3, 4],
        "amount": [3.0, 4.0, 2.0, 5.0, 3.0, 4.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
    )


def test_gap_analysis_returns_result(problem_with_protection: ConservationProblem):
    result = compute_gap_analysis(problem_with_protection)
    assert isinstance(result, GapResult)


def test_gap_analysis_protected_amounts(problem_with_protection: ConservationProblem):
    result = compute_gap_analysis(problem_with_protection)
    # PUs 1 (amt=3) and 3 (amt=2) protected for feature 1 -> 5.0
    assert result.protected_amount[1] == pytest.approx(5.0)
    # PU 3 (amt=3) protected for feature 2 -> 3.0
    assert result.protected_amount[2] == pytest.approx(3.0)


def test_gap_analysis_gap_values(problem_with_protection: ConservationProblem):
    result = compute_gap_analysis(problem_with_protection)
    # Feature 1: target=5.0, protected=5.0 -> gap=0.0
    assert result.gap[1] == pytest.approx(0.0)
    # Feature 2: target=8.0, protected=3.0 -> gap=5.0
    assert result.gap[2] == pytest.approx(5.0)


def test_gap_analysis_target_met(problem_with_protection: ConservationProblem):
    result = compute_gap_analysis(problem_with_protection)
    assert result.target_met[1] is True
    assert result.target_met[2] is False


def test_gap_analysis_to_dataframe(problem_with_protection: ConservationProblem):
    result = compute_gap_analysis(problem_with_protection)
    df = result.to_dataframe()
    assert len(df) == 2
    assert "feature_id" in df.columns
    assert "target" in df.columns
    assert "protected_amount" in df.columns
    assert "gap" in df.columns
    assert "percent_protected" in df.columns


def test_gap_analysis_no_protection():
    """All status=0 means nothing is protected."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [10.0, 20.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["f1"], "target": [5.0], "spf": [1.0]})
    puvspr = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [3.0, 4.0]})
    p = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=puvspr)
    result = compute_gap_analysis(p)
    assert result.protected_amount[1] == pytest.approx(0.0)
    assert result.gap[1] == pytest.approx(5.0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/analysis/test_gap_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Gap analysis for conservation planning.

Compares current protection levels (PUs with status=2) against
feature targets to identify protection gaps.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pymarxan.models.problem import ConservationProblem


@dataclass
class GapResult:
    """Results of a gap analysis."""

    feature_ids: list[int]
    feature_names: list[str]
    targets: dict[int, float]
    total_amount: dict[int, float]
    protected_amount: dict[int, float]
    gap: dict[int, float]
    target_met: dict[int, bool]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for fid in self.feature_ids:
            total = self.total_amount[fid]
            protected = self.protected_amount[fid]
            rows.append({
                "feature_id": fid,
                "feature_name": self.feature_names[self.feature_ids.index(fid)],
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


def compute_gap_analysis(problem: ConservationProblem) -> GapResult:
    """Compute protection gap for each feature."""
    protected_pu_ids = set(
        problem.planning_units.loc[
            problem.planning_units["status"] == 2, "id"
        ]
    )

    feature_ids = problem.features["id"].tolist()
    feature_names = problem.features["name"].tolist()

    targets: dict[int, float] = {}
    for _, row in problem.features.iterrows():
        targets[int(row["id"])] = float(row["target"])

    total_amount = problem.feature_amounts()
    protected_amount: dict[int, float] = {fid: 0.0 for fid in feature_ids}

    for _, row in problem.pu_vs_features.iterrows():
        pid = int(row["pu"])
        fid = int(row["species"])
        if pid in protected_pu_ids and fid in protected_amount:
            protected_amount[fid] += float(row["amount"])

    gap: dict[int, float] = {}
    target_met: dict[int, bool] = {}
    for fid in feature_ids:
        shortfall = max(targets[fid] - protected_amount[fid], 0.0)
        gap[fid] = shortfall
        target_met[fid] = shortfall <= 0

    return GapResult(
        feature_ids=feature_ids,
        feature_names=feature_names,
        targets=targets,
        total_amount=total_amount,
        protected_amount=protected_amount,
        gap=gap,
        target_met=target_met,
    )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pymarxan/analysis/test_gap_analysis.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan/analysis/gap_analysis.py tests/pymarxan/analysis/test_gap_analysis.py
git commit -m "feat: add gap analysis module comparing current protection vs targets"
```

---

### Task 4: SPF Explorer Shiny Module

**Files:**
- Create: `src/pymarxan_shiny/modules/calibration/spf_explorer.py`
- Test: `tests/pymarxan_shiny/test_spf_module.py`

**Context:** A Shiny module that wraps the existing `calibrate_spf()` function. Users configure the max iterations and SPF multiplier, run calibration, and view the iteration history in a table.

**Step 1: Write the failing test**

```python
"""Tests for SPF explorer Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.calibration.spf_explorer import (
    spf_explorer_server,
    spf_explorer_ui,
)


def test_spf_explorer_ui_returns_tag():
    ui_elem = spf_explorer_ui("test_spf")
    assert ui_elem is not None


def test_spf_explorer_server_callable():
    assert callable(spf_explorer_server)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan_shiny/test_spf_module.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
"""SPF calibration explorer Shiny module.

Lets users run iterative SPF calibration and view the history.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.calibration.spf import SPFResult, calibrate_spf
from pymarxan.solvers.base import SolverConfig


@module.ui
def spf_explorer_ui():
    return ui.card(
        ui.card_header("SPF Calibration"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_numeric(
                    "max_iterations", "Max iterations", value=10, min=1, max=50,
                ),
                ui.input_numeric(
                    "multiplier", "SPF multiplier", value=2.0, min=1.1, max=10.0,
                ),
                ui.input_action_button(
                    "run_spf", "Run SPF Calibration", class_="btn-primary w-100",
                ),
                ui.hr(),
                ui.output_text("spf_status"),
                width=280,
            ),
            ui.output_data_frame("spf_history_table"),
        ),
    )


@module.server
def spf_explorer_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
    solver: reactive.Calc,
):
    spf_result: reactive.Value[SPFResult | None] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.run_spf)
    def _run():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first!", type="error")
            return
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

    @render.text
    def spf_status():
        r = spf_result()
        if r is None:
            return "Not yet run"
        met = r.solution.all_targets_met
        return f"All targets met: {'Yes' if met else 'No'} ({len(r.history)} iterations)"

    @render.data_frame
    def spf_history_table():
        import pandas as pd

        r = spf_result()
        if r is None:
            return None
        rows = []
        for h in r.history:
            rows.append({
                "iteration": h["iteration"],
                "unmet_count": h["unmet_count"],
            })
        return pd.DataFrame(rows)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pymarxan_shiny/test_spf_module.py -v`
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/calibration/spf_explorer.py tests/pymarxan_shiny/test_spf_module.py
git commit -m "feat: add SPF calibration explorer Shiny module"
```

---

### Task 5: Connectivity Visualization Shiny Module

**Files:**
- Create: `src/pymarxan_shiny/modules/connectivity/metrics_viz.py`
- Create: `src/pymarxan_shiny/modules/connectivity/__init__.py`
- Test: `tests/pymarxan_shiny/test_connectivity_module.py`

**Context:** A Shiny module that displays connectivity metrics (in-degree, out-degree, betweenness, eigenvector centrality) as a data table. Users provide a connectivity matrix via a reactive value.

**Step 1: Write the failing test**

```python
"""Tests for connectivity metrics visualization Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.connectivity.metrics_viz import (
    metrics_viz_server,
    metrics_viz_ui,
)


def test_metrics_viz_ui_returns_tag():
    ui_elem = metrics_viz_ui("test_conn")
    assert ui_elem is not None


def test_metrics_viz_server_callable():
    assert callable(metrics_viz_server)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan_shiny/test_connectivity_module.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `src/pymarxan_shiny/modules/connectivity/__init__.py` (empty).

Create `src/pymarxan_shiny/modules/connectivity/metrics_viz.py`:

```python
"""Connectivity metrics visualization Shiny module.

Displays connectivity metrics (in-degree, out-degree, betweenness,
eigenvector centrality) for the current connectivity matrix.
"""
from __future__ import annotations

import numpy as np

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.connectivity.metrics import (
    compute_betweenness_centrality,
    compute_eigenvector_centrality,
    compute_in_degree,
    compute_out_degree,
)


@module.ui
def metrics_viz_ui():
    return ui.card(
        ui.card_header("Connectivity Metrics"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_action_button(
                    "compute_metrics", "Compute Metrics",
                    class_="btn-primary w-100",
                ),
                ui.hr(),
                ui.output_text("metrics_status"),
                width=280,
            ),
            ui.output_data_frame("metrics_table"),
        ),
    )


@module.server
def metrics_viz_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    connectivity_matrix: reactive.Value,
    pu_ids: reactive.Value,
):
    metrics_df: reactive.Value = reactive.value(None)

    @reactive.effect
    @reactive.event(input.compute_metrics)
    def _compute():
        import pandas as pd

        matrix = connectivity_matrix()
        ids = pu_ids()
        if matrix is None:
            ui.notification_show("No connectivity matrix loaded!", type="error")
            return

        in_deg = compute_in_degree(matrix)
        out_deg = compute_out_degree(matrix)
        bc = compute_betweenness_centrality(matrix)
        ec = compute_eigenvector_centrality(matrix)

        df = pd.DataFrame({
            "pu_id": ids if ids is not None else list(range(matrix.shape[0])),
            "in_degree": in_deg,
            "out_degree": out_deg,
            "betweenness": bc,
            "eigenvector": ec,
        })
        metrics_df.set(df)
        ui.notification_show("Metrics computed", type="message")

    @render.text
    def metrics_status():
        m = connectivity_matrix()
        if m is None:
            return "No connectivity matrix loaded"
        return f"Matrix: {m.shape[0]} nodes"

    @render.data_frame
    def metrics_table():
        return metrics_df()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pymarxan_shiny/test_connectivity_module.py -v`
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/connectivity/__init__.py src/pymarxan_shiny/modules/connectivity/metrics_viz.py tests/pymarxan_shiny/test_connectivity_module.py
git commit -m "feat: add connectivity metrics visualization Shiny module"
```

---

### Task 6: Target Achievement Shiny Module

**Files:**
- Create: `src/pymarxan_shiny/modules/results/target_met.py`
- Test: `tests/pymarxan_shiny/test_target_met_module.py`

**Context:** A Shiny module that shows which conservation targets are met/unmet for the current solution, with a coloured status indicator per feature. Uses Solution.targets_met and the problem's feature names.

**Step 1: Write the failing test**

```python
"""Tests for target achievement Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.results.target_met import (
    target_met_server,
    target_met_ui,
)


def test_target_met_ui_returns_tag():
    ui_elem = target_met_ui("test_targets")
    assert ui_elem is not None


def test_target_met_server_callable():
    assert callable(target_met_server)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan_shiny/test_target_met_module.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
"""Target achievement Shiny module.

Shows which conservation targets are met/unmet for the current solution.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


@module.ui
def target_met_ui():
    return ui.card(
        ui.card_header("Target Achievement"),
        ui.output_data_frame("target_table"),
    )


@module.server
def target_met_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    @render.data_frame
    def target_table():
        import pandas as pd

        p = problem()
        s = solution()
        if p is None or s is None:
            return None

        rows = []
        for _, row in p.features.iterrows():
            fid = int(row["id"])
            met = s.targets_met.get(fid, False)
            rows.append({
                "feature_id": fid,
                "name": row["name"],
                "target": float(row["target"]),
                "met": "Yes" if met else "No",
            })
        return pd.DataFrame(rows)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pymarxan_shiny/test_target_met_module.py -v`
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/results/target_met.py tests/pymarxan_shiny/test_target_met_module.py
git commit -m "feat: add target achievement Shiny module"
```

---

### Task 7: Update App with Phase 5 Modules

**Files:**
- Modify: `src/pymarxan_app/app.py`
- Create: `tests/test_integration_phase5.py`

**Context:** Add the SPF explorer, target achievement, and connectivity modules to the app. Update the Calibrate tab to include SPF explorer alongside BLM. Add target achievement to Results. Add a Connectivity tab.

**Step 1: Write integration tests**

```python
"""Integration tests for Phase 5 features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.analysis.gap_analysis import compute_gap_analysis
from pymarxan.calibration.sensitivity import SensitivityConfig, run_sensitivity
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.registry import get_default_registry


def _problem() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 20.0, 15.0, 25.0],
        "status": [2, 0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1, 2],
        "name": ["bird", "mammal"],
        "target": [5.0, 4.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 2, 2],
        "pu": [1, 2, 3, 3, 4],
        "amount": [3.0, 4.0, 2.0, 5.0, 3.0],
    })
    bnd = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "boundary": [1.0, 1.0]})
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=bnd,
        parameters={"BLM": 1.0},
    )


def test_heuristic_end_to_end():
    """Greedy solver produces valid solution."""
    solver = HeuristicSolver()
    sols = solver.solve(_problem(), SolverConfig(num_solutions=1))
    assert len(sols) == 1
    assert sols[0].all_targets_met


def test_gap_analysis_end_to_end():
    """Gap analysis with partially protected problem."""
    result = compute_gap_analysis(_problem())
    df = result.to_dataframe()
    assert len(df) == 2
    assert "percent_protected" in df.columns


def test_sensitivity_end_to_end():
    """Sensitivity analysis with MIP solver."""
    config = SensitivityConfig(multipliers=[0.8, 1.0, 1.2])
    result = run_sensitivity(_problem(), MIPSolver(), config)
    assert len(result.runs) == 6  # 2 features x 3 multipliers


def test_registry_includes_greedy():
    """Default registry now includes greedy solver."""
    reg = get_default_registry()
    assert "greedy" in reg.list_solvers()
    solver = reg.create("greedy")
    sols = solver.solve(_problem(), SolverConfig(num_solutions=1))
    assert len(sols) == 1


def test_app_imports():
    """App module imports successfully with phase 5 modules."""
    import pymarxan_app.app  # noqa: F401
```

**Step 2: Update app.py**

Add imports for new modules. Update the Calibrate tab to include both BLM and SPF explorers. Add a Connectivity tab. Add target achievement to Results.

Key changes:
1. Import `spf_explorer_ui, spf_explorer_server`
2. Import `target_met_ui, target_met_server`
3. Import `metrics_viz_ui, metrics_viz_server`
4. Update Calibrate tab layout to include both BLM and SPF
5. Add Connectivity tab after Sweep
6. Add `target_met_ui("targets")` to Results tab
7. Wire server modules

**Step 3: Run tests**

Run: `python -m pytest tests/test_integration_phase5.py -v`
Expected: 5 PASSED

**Step 4: Commit**

```bash
git add src/pymarxan_app/app.py tests/test_integration_phase5.py
git commit -m "feat: update app with SPF, connectivity, target modules and phase 5 integration tests"
```

---

### Task 8: Lint and Cleanup

**Files:** All new files from Tasks 1-7

**Step 1: Run ruff**

Run: `ruff check src/ tests/ --fix`
Fix any remaining issues.

**Step 2: Run mypy**

Run: `mypy src/pymarxan/ --ignore-missing-imports`
Fix any type errors.

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass (~214 total)

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "chore: fix lint and type issues from phase 5"
```

---

## Summary

| Task | Module | Tests | Key concept |
|------|--------|-------|-------------|
| 1 | `calibration/sensitivity.py` | 5 | Vary targets, measure solution sensitivity |
| 2 | `solvers/heuristic.py` | 8 | Greedy cost-effectiveness solver |
| 3 | `analysis/gap_analysis.py` | 6 | Current protection vs targets |
| 4 | `calibration/spf_explorer.py` (Shiny) | 2 | SPF calibration UI |
| 5 | `connectivity/metrics_viz.py` (Shiny) | 2 | Connectivity metrics display |
| 6 | `results/target_met.py` (Shiny) | 2 | Target achievement dashboard |
| 7 | App update + integration tests | 5 | Wire everything together |
| 8 | Lint + cleanup | — | Code quality |

**Total new tests:** ~30
**Total estimated tests after Phase 5:** ~214+

**Parallelisable pairs:**
- Tasks 1 + 2 + 3 (all core modules — no dependencies between them)
- Tasks 4 + 5 + 6 (all Shiny modules — independent)
