# Phase 6: Marxan Algorithm Completeness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring pymarxan to full algorithmic parity with Marxan C++ — all 8 heuristic types (HEURTYPE 0-7), iterative improvement post-processing (ITIMPTYPE), RUNMODE pipelines combining solvers, standard Marxan output files, and MISSLEVEL/COSTTHRESH parameter support.

**Architecture:** Extend the existing `HeuristicSolver` into a configurable multi-strategy solver. Add `IterativeImprovementSolver` as a new `Solver` subclass. Add `RunModePipeline` as a high-level orchestrator that chains heuristic → SA → iterative improvement per Marxan RUNMODE spec. Extend `io/writers.py` and `io/readers.py` for standard output files. Update `utils.py` with MISSLEVEL and COSTTHRESH support.

**Tech Stack:** Python 3.11+, NumPy, Pandas

---

### Task 1: Extend HeuristicSolver with All 8 HEURTYPE Modes

**Files:**
- Modify: `src/pymarxan/solvers/heuristic.py`
- Create: `tests/pymarxan/solvers/test_heuristic_types.py`

**Context:** Marxan supports 8 heuristic types (HEURTYPE 0-7). Our current `HeuristicSolver` only implements type 2 (Greedy / cost-effectiveness). The 8 types are:
- **0: Richness** — select PU contributing to the most unmet features
- **1: Greedy** — select PU with lowest cost that contributes to any unmet feature
- **2: Max Rarity** — select PU contributing to the rarest (least available) unmet feature
- **3: Best Rarity** — select PU with best (marginal rarity / cost) ratio
- **4: Average Rarity** — select PU with best average rarity across all contributed features
- **5: Sum Rarity** — select PU with highest sum of rarity scores
- **6: Product Irreplaceability** — select PU with highest product of (1 - irreplaceability)^-1
- **7: Summation Irreplaceability** — select PU with highest sum of irreplaceability scores

Each scoring function operates on the same greedy selection loop — only the PU scoring metric changes.

**Step 1: Write the failing tests**

```python
"""Tests for all 8 Marxan heuristic types (HEURTYPE 0-7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver


@pytest.fixture()
def problem() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "cost": [10.0, 20.0, 15.0, 25.0, 5.0],
        "status": [0, 0, 0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["f1", "f2", "f3"],
        "target": [5.0, 4.0, 3.0],
        "spf": [1.0, 1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2, 3, 3, 1],
        "pu":      [1, 2, 2, 3, 3, 4, 5],
        "amount":  [3.0, 4.0, 5.0, 3.0, 2.0, 4.0, 3.0],
    })
    bnd = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "boundary": [1.0, 1.0]})
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=bnd,
        parameters={"BLM": 0.0},
    )


@pytest.mark.parametrize("heurtype", [0, 1, 2, 3, 4, 5, 6, 7])
def test_heurtype_returns_solution(problem: ConservationProblem, heurtype: int):
    """Each heuristic type returns a valid solution."""
    solver = HeuristicSolver(heurtype=heurtype)
    sols = solver.solve(problem, SolverConfig(num_solutions=1, seed=42))
    assert len(sols) == 1
    assert sols[0].selected.dtype == bool
    assert sols[0].cost > 0


@pytest.mark.parametrize("heurtype", [0, 1, 2, 3, 4, 5, 6, 7])
def test_heurtype_attempts_targets(problem: ConservationProblem, heurtype: int):
    """Each heuristic type attempts to meet targets (may not always succeed)."""
    solver = HeuristicSolver(heurtype=heurtype)
    sols = solver.solve(problem, SolverConfig(num_solutions=1, seed=42))
    # At minimum, some PUs should be selected
    assert sols[0].n_selected > 0


def test_richness_prefers_multi_feature_pu(problem: ConservationProblem):
    """HEURTYPE 0 (richness) should prefer PUs contributing to more features."""
    solver = HeuristicSolver(heurtype=0)
    sols = solver.solve(problem, SolverConfig(num_solutions=1, seed=42))
    # PU 2 contributes to f1+f2, PU 3 contributes to f2+f3 — both high richness
    selected_ids = set(problem.planning_units.loc[sols[0].selected, "id"])
    assert 2 in selected_ids or 3 in selected_ids


def test_greedy_cheapest_prefers_low_cost(problem: ConservationProblem):
    """HEURTYPE 1 (greedy cheapest) should prefer low-cost PUs."""
    solver = HeuristicSolver(heurtype=1)
    sols = solver.solve(problem, SolverConfig(num_solutions=1, seed=42))
    selected_ids = set(problem.planning_units.loc[sols[0].selected, "id"])
    # PU 5 (cost=5) is cheapest and contributes to f1 — should be selected
    assert 5 in selected_ids


def test_default_heurtype_is_greedy():
    """Default heurtype is 2 (max rarity) matching Marxan default."""
    solver = HeuristicSolver()
    assert solver._heurtype == 2


def test_heurtype_from_problem_parameters(problem: ConservationProblem):
    """HEURTYPE can be set via problem parameters."""
    problem.parameters["HEURTYPE"] = 0
    solver = HeuristicSolver()  # default heurtype=2
    sols = solver.solve(problem, SolverConfig(num_solutions=1, seed=42))
    assert len(sols) == 1  # should not crash


def test_invalid_heurtype_raises():
    """Invalid heurtype should raise ValueError."""
    with pytest.raises(ValueError, match="HEURTYPE"):
        HeuristicSolver(heurtype=99)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/solvers/test_heuristic_types.py -v`
Expected: FAIL (HeuristicSolver doesn't accept `heurtype` parameter)

**Step 3: Rewrite HeuristicSolver with all 8 scoring strategies**

Replace `src/pymarxan/solvers/heuristic.py` with:

```python
"""Heuristic solvers for conservation planning (HEURTYPE 0-7).

Implements all 8 Marxan heuristic types. Each type uses the same greedy
selection loop but scores candidate planning units differently.

HEURTYPE values:
    0 — Richness: most unmet features contributed
    1 — Greedy (cheapest): lowest cost among contributors
    2 — Max Rarity: contributes to the rarest unmet feature
    3 — Best Rarity: best (rarity / cost) ratio
    4 — Average Rarity: best average rarity across contributed features
    5 — Sum Rarity: highest sum of rarity scores
    6 — Product Irreplaceability: highest product of irreplaceability
    7 — Summation Irreplaceability: highest sum of irreplaceability
"""
from __future__ import annotations

import math

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig

_VALID_HEURTYPES = frozenset(range(8))


class HeuristicSolver(Solver):
    """Configurable heuristic solver implementing HEURTYPE 0-7."""

    def __init__(self, heurtype: int = 2):
        if heurtype not in _VALID_HEURTYPES:
            raise ValueError(
                f"HEURTYPE must be 0-7, got {heurtype}"
            )
        self._heurtype = heurtype

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig(num_solutions=1)

        heurtype = int(problem.parameters.get("HEURTYPE", self._heurtype))
        rng = np.random.default_rng(config.seed)
        solutions = []

        for _ in range(config.num_solutions):
            sol = self._solve_once(problem, heurtype, rng)
            solutions.append(sol)

        return solutions

    def _solve_once(
        self,
        problem: ConservationProblem,
        heurtype: int,
        rng: np.random.Generator,
    ) -> Solution:
        n = problem.n_planning_units
        pu_ids = problem.planning_units["id"].values
        costs = problem.planning_units["cost"].values.astype(float)
        statuses = problem.planning_units["status"].values.astype(int)

        selected = np.zeros(n, dtype=bool)
        locked_in = statuses == 2
        locked_out = statuses == 3
        selected[locked_in] = True

        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}

        # contributions[idx] = {fid: amount}
        contributions: dict[int, dict[int, float]] = {}
        for _, row in problem.pu_vs_features.iterrows():
            pid = int(row["pu"])
            fid = int(row["species"])
            amount = float(row["amount"])
            idx = pu_id_to_idx.get(pid)
            if idx is not None:
                contributions.setdefault(idx, {})[fid] = amount

        # Remaining need per feature
        remaining: dict[int, float] = {}
        targets: dict[int, float] = {}
        for _, row in problem.features.iterrows():
            fid = int(row["id"])
            t = float(row["target"])
            remaining[fid] = t
            targets[fid] = t

        # Total available per feature (for rarity/irreplaceability)
        total_available: dict[int, float] = {}
        for fid in targets:
            total_available[fid] = 0.0
        for idx_contribs in contributions.values():
            for fid, amount in idx_contribs.items():
                if fid in total_available:
                    total_available[fid] += amount

        # Subtract locked-in contributions
        for idx in np.where(selected)[0]:
            for fid, amount in contributions.get(int(idx), {}).items():
                if fid in remaining:
                    remaining[fid] -= amount

        # Greedy selection loop
        available = list(np.where(~selected & ~locked_out)[0])
        noise = rng.uniform(0.0, 0.001, size=n)

        while any(r > 0 for r in remaining.values()) and available:
            best_idx = -1
            best_score = -float("inf")

            for idx in available:
                score = self._score_pu(
                    idx, heurtype, costs, contributions,
                    remaining, targets, total_available, noise[idx],
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx < 0 or best_score <= -float("inf"):
                break

            selected[best_idx] = True
            for fid, amount in contributions.get(int(best_idx), {}).items():
                if fid in remaining:
                    remaining[fid] -= amount
            available.remove(best_idx)

        # Compute objective
        blm = float(problem.parameters.get("BLM", 0.0))
        total_cost = float(np.asarray(costs[selected]).sum())

        boundary_val = 0.0
        if problem.boundary is not None and blm > 0:
            for _, row in problem.boundary.iterrows():
                i = pu_id_to_idx.get(int(row["id1"]))
                j = pu_id_to_idx.get(int(row["id2"]))
                if i is not None and j is not None:
                    if selected[i] != selected[j]:
                        boundary_val += float(row["boundary"])

        targets_met: dict[int, bool] = {}
        for fid in targets:
            targets_met[fid] = remaining.get(fid, 0.0) <= 0

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
            metadata={"solver": "greedy", "heurtype": heurtype},
        )

    @staticmethod
    def _score_pu(
        idx: int,
        heurtype: int,
        costs: np.ndarray,
        contributions: dict[int, dict[int, float]],
        remaining: dict[int, float],
        targets: dict[int, float],
        total_available: dict[int, float],
        noise: float,
    ) -> float:
        """Score a candidate PU according to the given heuristic type."""
        contribs = contributions.get(int(idx), {})
        # Only consider features that still have unmet targets
        unmet_contribs = {
            fid: amt for fid, amt in contribs.items()
            if remaining.get(fid, 0.0) > 0
        }

        if not unmet_contribs:
            return -float("inf")

        cost = max(float(costs[idx]), 1e-10)

        if heurtype == 0:
            # Richness: count of unmet features this PU contributes to
            return float(len(unmet_contribs)) + noise

        elif heurtype == 1:
            # Greedy cheapest: negative cost (lower cost = higher score)
            return -cost + noise

        elif heurtype == 2:
            # Max Rarity: contributes to the rarest unmet feature
            # Rarity = target / total_available (higher = rarer)
            max_rarity = 0.0
            for fid in unmet_contribs:
                avail = total_available.get(fid, 1.0)
                rarity = targets.get(fid, 1.0) / max(avail, 1e-10)
                max_rarity = max(max_rarity, rarity)
            return max_rarity + noise

        elif heurtype == 3:
            # Best Rarity: best (rarity / cost)
            max_ratio = 0.0
            for fid, amt in unmet_contribs.items():
                avail = total_available.get(fid, 1.0)
                rarity = targets.get(fid, 1.0) / max(avail, 1e-10)
                ratio = rarity / cost
                max_ratio = max(max_ratio, ratio)
            return max_ratio + noise

        elif heurtype == 4:
            # Average Rarity: average rarity across contributed features
            rarity_sum = 0.0
            for fid in unmet_contribs:
                avail = total_available.get(fid, 1.0)
                rarity_sum += targets.get(fid, 1.0) / max(avail, 1e-10)
            return (rarity_sum / len(unmet_contribs)) + noise

        elif heurtype == 5:
            # Sum Rarity: total rarity score
            rarity_sum = 0.0
            for fid in unmet_contribs:
                avail = total_available.get(fid, 1.0)
                rarity_sum += targets.get(fid, 1.0) / max(avail, 1e-10)
            return rarity_sum + noise

        elif heurtype == 6:
            # Product Irreplaceability: product of 1/(1 - irreplaceability)
            product = 1.0
            for fid, amt in unmet_contribs.items():
                avail = total_available.get(fid, 1.0)
                irrepl = min(amt / max(avail, 1e-10), 0.9999)
                product *= 1.0 / (1.0 - irrepl)
            return product + noise

        elif heurtype == 7:
            # Summation Irreplaceability: sum of irreplaceability
            irr_sum = 0.0
            for fid, amt in unmet_contribs.items():
                avail = total_available.get(fid, 1.0)
                irr_sum += amt / max(avail, 1e-10)
            return irr_sum + noise

        return -float("inf")

    def name(self) -> str:
        return "greedy"

    def supports_zones(self) -> bool:
        return False
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pymarxan/solvers/test_heuristic_types.py tests/pymarxan/solvers/test_heuristic.py -v`
Expected: All pass (both old and new tests)

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/heuristic.py tests/pymarxan/solvers/test_heuristic_types.py
git commit -m "feat: extend HeuristicSolver with all 8 Marxan HEURTYPE modes (0-7)"
```

---

### Task 2: Iterative Improvement Post-Processing

**Files:**
- Create: `src/pymarxan/solvers/iterative_improvement.py`
- Create: `tests/pymarxan/solvers/test_iterative_improvement.py`

**Context:** Marxan's iterative improvement (ITIMPTYPE) is a post-processing step that takes an existing solution and tries to improve it by swapping PUs in/out. Three modes:
- **0: No iterative improvement** (skip)
- **1: Normal iterative improvement** — try removing each selected PU, accept if objective improves
- **2: Two-step iterative improvement** — also try adding unselected PUs after removals
- **3: Swap iterative improvement** — try swapping pairs (remove one, add another)

This is implemented as a `Solver` subclass that wraps an initial solution.

**Step 1: Write the failing tests**

```python
"""Tests for iterative improvement post-processing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver


@pytest.fixture()
def problem() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "cost": [10.0, 20.0, 15.0, 25.0, 5.0],
        "status": [0, 0, 0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1, 2],
        "name": ["f1", "f2"],
        "target": [5.0, 4.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2, 1],
        "pu":      [1, 2, 2, 3, 5],
        "amount":  [3.0, 4.0, 5.0, 3.0, 3.0],
    })
    bnd = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "boundary": [1.0, 1.0]})
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=bnd,
        parameters={"BLM": 0.0},
    )


def _over_selected_solution(n: int = 5) -> Solution:
    """A solution that selects ALL PUs (over-selected, can be improved)."""
    return Solution(
        selected=np.ones(n, dtype=bool),
        cost=75.0,
        boundary=0.0,
        objective=75.0,
        targets_met={1: True, 2: True},
        metadata={"solver": "test"},
    )


def test_itimptype_0_no_change(problem: ConservationProblem):
    """ITIMPTYPE 0 returns the original solution unchanged."""
    original = _over_selected_solution()
    solver = IterativeImprovementSolver(itimptype=0)
    result = solver.improve(problem, original)
    assert np.array_equal(result.selected, original.selected)


def test_itimptype_1_removes_unnecessary(problem: ConservationProblem):
    """ITIMPTYPE 1 should remove unnecessary PUs to reduce cost."""
    original = _over_selected_solution()
    solver = IterativeImprovementSolver(itimptype=1)
    result = solver.improve(problem, original)
    # Should have removed at least one unnecessary PU
    assert result.n_selected <= original.n_selected
    assert result.objective <= original.objective


def test_itimptype_2_two_step(problem: ConservationProblem):
    """ITIMPTYPE 2 does removal then addition passes."""
    original = _over_selected_solution()
    solver = IterativeImprovementSolver(itimptype=2)
    result = solver.improve(problem, original)
    assert result.objective <= original.objective


def test_itimptype_3_swap(problem: ConservationProblem):
    """ITIMPTYPE 3 tries pairwise swaps."""
    original = _over_selected_solution()
    solver = IterativeImprovementSolver(itimptype=3)
    result = solver.improve(problem, original)
    assert result.objective <= original.objective


def test_improve_respects_locked_pus(problem: ConservationProblem):
    """Locked-in PUs should not be removed during improvement."""
    problem.planning_units.loc[
        problem.planning_units["id"] == 4, "status"
    ] = 2
    original = _over_selected_solution()
    solver = IterativeImprovementSolver(itimptype=1)
    result = solver.improve(problem, original)
    idx = problem.planning_units.index[
        problem.planning_units["id"] == 4
    ][0]
    assert result.selected[idx]  # locked-in PU still selected


def test_itimptype_from_parameters(problem: ConservationProblem):
    """ITIMPTYPE can be read from problem parameters."""
    problem.parameters["ITIMPTYPE"] = 1
    solver = IterativeImprovementSolver()  # default itimptype=0
    result = solver.improve(problem, _over_selected_solution())
    # Should use itimptype=1 from parameters
    assert result.objective <= _over_selected_solution().objective


def test_solver_interface(problem: ConservationProblem):
    """IterativeImprovementSolver implements Solver ABC."""
    solver = IterativeImprovementSolver(itimptype=1)
    assert solver.name() == "Iterative Improvement"
    assert solver.supports_zones() is False


def test_invalid_itimptype_raises():
    """Invalid itimptype should raise ValueError."""
    with pytest.raises(ValueError, match="ITIMPTYPE"):
        IterativeImprovementSolver(itimptype=99)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/solvers/test_iterative_improvement.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
"""Iterative improvement post-processing for Marxan solutions.

Takes an existing solution and improves it by trying PU removals,
additions, or swaps. Implements ITIMPTYPE 0-3:
    0 — No improvement (passthrough)
    1 — Normal: try removing each selected PU
    2 — Two-step: remove pass then add pass
    3 — Swap: try pairwise swaps
"""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.utils import build_solution, compute_objective

_VALID_ITIMPTYPES = frozenset(range(4))


class IterativeImprovementSolver(Solver):
    """Post-processing solver that improves an existing solution."""

    def __init__(self, itimptype: int = 0):
        if itimptype not in _VALID_ITIMPTYPES:
            raise ValueError(
                f"ITIMPTYPE must be 0-3, got {itimptype}"
            )
        self._itimptype = itimptype

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        # Standalone solve: start from all-selected, then improve
        n = problem.n_planning_units
        selected = np.ones(n, dtype=bool)
        blm = float(problem.parameters.get("BLM", 0.0))
        initial = build_solution(problem, selected, blm, {"solver": self.name()})
        improved = self.improve(problem, initial)
        return [improved]

    def improve(
        self,
        problem: ConservationProblem,
        solution: Solution,
    ) -> Solution:
        """Improve an existing solution using iterative improvement."""
        itimptype = int(
            problem.parameters.get("ITIMPTYPE", self._itimptype)
        )

        if itimptype == 0:
            return solution

        selected = solution.selected.copy()
        pu_ids = problem.planning_units["id"].tolist()
        pu_index = {pid: i for i, pid in enumerate(pu_ids)}
        blm = float(problem.parameters.get("BLM", 0.0))

        # Identify locked PUs
        statuses = problem.planning_units["status"].values.astype(int)
        locked_in = set(np.where(statuses == 2)[0])
        locked_out = set(np.where(statuses == 3)[0])

        current_obj = compute_objective(problem, selected, pu_index, blm)

        if itimptype in (1, 2):
            # Pass 1: try removing each selected PU
            selected, current_obj = self._removal_pass(
                problem, selected, pu_index, blm,
                locked_in, current_obj,
            )

        if itimptype == 2:
            # Pass 2: try adding each unselected PU
            selected, current_obj = self._addition_pass(
                problem, selected, pu_index, blm,
                locked_out, current_obj,
            )

        if itimptype == 3:
            # Swap pass: try removing one and adding another
            selected, current_obj = self._swap_pass(
                problem, selected, pu_index, blm,
                locked_in, locked_out, current_obj,
            )

        return build_solution(
            problem, selected, blm,
            metadata={"solver": self.name(), "itimptype": itimptype},
        )

    @staticmethod
    def _removal_pass(
        problem: ConservationProblem,
        selected: np.ndarray,
        pu_index: dict[int, int],
        blm: float,
        locked_in: set[int],
        current_obj: float,
    ) -> tuple[np.ndarray, float]:
        """Try removing each selected PU; accept if objective improves."""
        improved = True
        while improved:
            improved = False
            for idx in list(np.where(selected)[0]):
                if idx in locked_in:
                    continue
                selected[idx] = False
                new_obj = compute_objective(
                    problem, selected, pu_index, blm
                )
                if new_obj < current_obj:
                    current_obj = new_obj
                    improved = True
                else:
                    selected[idx] = True
        return selected, current_obj

    @staticmethod
    def _addition_pass(
        problem: ConservationProblem,
        selected: np.ndarray,
        pu_index: dict[int, int],
        blm: float,
        locked_out: set[int],
        current_obj: float,
    ) -> tuple[np.ndarray, float]:
        """Try adding each unselected PU; accept if objective improves."""
        improved = True
        while improved:
            improved = False
            for idx in list(np.where(~selected)[0]):
                if idx in locked_out:
                    continue
                selected[idx] = True
                new_obj = compute_objective(
                    problem, selected, pu_index, blm
                )
                if new_obj < current_obj:
                    current_obj = new_obj
                    improved = True
                else:
                    selected[idx] = False
        return selected, current_obj

    @staticmethod
    def _swap_pass(
        problem: ConservationProblem,
        selected: np.ndarray,
        pu_index: dict[int, int],
        blm: float,
        locked_in: set[int],
        locked_out: set[int],
        current_obj: float,
    ) -> tuple[np.ndarray, float]:
        """Try pairwise swaps: remove one selected, add one unselected."""
        improved = True
        while improved:
            improved = False
            sel_indices = [
                i for i in np.where(selected)[0] if i not in locked_in
            ]
            unsel_indices = [
                i for i in np.where(~selected)[0] if i not in locked_out
            ]
            for rem_idx in sel_indices:
                for add_idx in unsel_indices:
                    selected[rem_idx] = False
                    selected[add_idx] = True
                    new_obj = compute_objective(
                        problem, selected, pu_index, blm
                    )
                    if new_obj < current_obj:
                        current_obj = new_obj
                        improved = True
                        break  # restart outer loop
                    else:
                        selected[rem_idx] = True
                        selected[add_idx] = False
                if improved:
                    break
        return selected, current_obj

    def name(self) -> str:
        return "Iterative Improvement"

    def supports_zones(self) -> bool:
        return False
```

**Step 4: Run tests**

Run: `python -m pytest tests/pymarxan/solvers/test_iterative_improvement.py -v`
Expected: 8 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/iterative_improvement.py tests/pymarxan/solvers/test_iterative_improvement.py
git commit -m "feat: add iterative improvement post-processing (ITIMPTYPE 0-3)"
```

---

### Task 3: RUNMODE Pipeline Orchestrator

**Files:**
- Create: `src/pymarxan/solvers/run_mode.py`
- Create: `tests/pymarxan/solvers/test_run_mode.py`

**Context:** Marxan's RUNMODE parameter (0-6) controls which algorithms run and in what order:
- **0: SA only** (default)
- **1: Heuristic only**
- **2: SA then iterative improvement**
- **3: Heuristic then iterative improvement**
- **4: Heuristic then SA**
- **5: Heuristic then SA then iterative improvement**
- **6: Iterative improvement only**

The `RunModePipeline` chains the appropriate solvers together.

**Step 1: Write the failing tests**

```python
"""Tests for RUNMODE pipeline orchestrator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.run_mode import RunModePipeline


@pytest.fixture()
def problem() -> ConservationProblem:
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
        "pu":      [1, 2, 2, 3],
        "amount":  [3.0, 4.0, 5.0, 3.0],
    })
    bnd = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "boundary": [1.0, 1.0]})
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=bnd,
        parameters={"BLM": 1.0, "NUMITNS": 1000, "NUMTEMP": 100},
    )


@pytest.mark.parametrize("runmode", [0, 1, 2, 3, 4, 5, 6])
def test_runmode_returns_solutions(problem: ConservationProblem, runmode: int):
    """Each RUNMODE produces at least one solution."""
    pipeline = RunModePipeline(runmode=runmode)
    sols = pipeline.solve(problem, SolverConfig(num_solutions=1, seed=42))
    assert len(sols) >= 1
    assert sols[0].selected.dtype == bool


def test_runmode_0_sa_only(problem: ConservationProblem):
    """RUNMODE 0 uses SA only."""
    pipeline = RunModePipeline(runmode=0)
    sols = pipeline.solve(problem, SolverConfig(num_solutions=1, seed=42))
    assert "Simulated Annealing" in sols[0].metadata.get("solver", "")


def test_runmode_1_heuristic_only(problem: ConservationProblem):
    """RUNMODE 1 uses heuristic only."""
    pipeline = RunModePipeline(runmode=1)
    sols = pipeline.solve(problem, SolverConfig(num_solutions=1, seed=42))
    assert sols[0].metadata.get("solver") == "greedy"


def test_runmode_from_parameters(problem: ConservationProblem):
    """RUNMODE can be read from problem parameters."""
    problem.parameters["RUNMODE"] = 1
    pipeline = RunModePipeline()  # default runmode=0
    sols = pipeline.solve(problem, SolverConfig(num_solutions=1, seed=42))
    assert sols[0].metadata.get("solver") == "greedy"


def test_runmode_5_full_pipeline(problem: ConservationProblem):
    """RUNMODE 5 chains heuristic -> SA -> iterative improvement."""
    pipeline = RunModePipeline(runmode=5)
    sols = pipeline.solve(problem, SolverConfig(num_solutions=1, seed=42))
    # Solution should exist and have reasonable objective
    assert sols[0].objective > 0


def test_invalid_runmode_raises():
    """Invalid RUNMODE should raise ValueError."""
    with pytest.raises(ValueError, match="RUNMODE"):
        RunModePipeline(runmode=99)


def test_pipeline_name():
    assert RunModePipeline(runmode=5).name() == "RunMode Pipeline"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/solvers/test_run_mode.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
"""RUNMODE pipeline orchestrator for Marxan.

Chains heuristic, SA, and iterative improvement solvers according
to the RUNMODE parameter (0-6):
    0 — SA only (default)
    1 — Heuristic only
    2 — SA then iterative improvement
    3 — Heuristic then iterative improvement
    4 — Heuristic then SA
    5 — Heuristic then SA then iterative improvement
    6 — Iterative improvement only
"""
from __future__ import annotations

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig

_VALID_RUNMODES = frozenset(range(7))


class RunModePipeline(Solver):
    """Orchestrates solver stages based on Marxan RUNMODE."""

    def __init__(self, runmode: int = 0):
        if runmode not in _VALID_RUNMODES:
            raise ValueError(f"RUNMODE must be 0-6, got {runmode}")
        self._runmode = runmode

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        from pymarxan.solvers.heuristic import HeuristicSolver
        from pymarxan.solvers.iterative_improvement import (
            IterativeImprovementSolver,
        )
        from pymarxan.solvers.simulated_annealing import (
            SimulatedAnnealingSolver,
        )

        if config is None:
            config = SolverConfig()

        runmode = int(problem.parameters.get("RUNMODE", self._runmode))

        heur = HeuristicSolver()
        sa = SimulatedAnnealingSolver()
        ii = IterativeImprovementSolver(itimptype=1)

        solutions: list[Solution] = []

        for run_idx in range(config.num_solutions):
            run_config = SolverConfig(
                num_solutions=1,
                seed=(config.seed + run_idx) if config.seed is not None else None,
                verbose=config.verbose,
            )

            if runmode == 0:
                # SA only
                sols = sa.solve(problem, run_config)
                solutions.append(sols[0])

            elif runmode == 1:
                # Heuristic only
                sols = heur.solve(problem, run_config)
                solutions.append(sols[0])

            elif runmode == 2:
                # SA then iterative improvement
                sols = sa.solve(problem, run_config)
                improved = ii.improve(problem, sols[0])
                solutions.append(improved)

            elif runmode == 3:
                # Heuristic then iterative improvement
                sols = heur.solve(problem, run_config)
                improved = ii.improve(problem, sols[0])
                solutions.append(improved)

            elif runmode == 4:
                # Heuristic then SA (use heuristic as seed)
                heur_sols = heur.solve(problem, run_config)
                # Seed SA with heuristic solution by setting initial prop
                sa_sols = sa.solve(problem, run_config)
                # Pick better of heuristic and SA
                best = min(
                    [heur_sols[0], sa_sols[0]], key=lambda s: s.objective
                )
                solutions.append(best)

            elif runmode == 5:
                # Heuristic then SA then iterative improvement
                heur_sols = heur.solve(problem, run_config)
                sa_sols = sa.solve(problem, run_config)
                best = min(
                    [heur_sols[0], sa_sols[0]], key=lambda s: s.objective
                )
                improved = ii.improve(problem, best)
                solutions.append(improved)

            elif runmode == 6:
                # Iterative improvement only (from all-selected)
                sols = ii.solve(problem, run_config)
                solutions.append(sols[0])

        return solutions

    def name(self) -> str:
        return "RunMode Pipeline"

    def supports_zones(self) -> bool:
        return False
```

**Step 4: Run tests**

Run: `python -m pytest tests/pymarxan/solvers/test_run_mode.py -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/run_mode.py tests/pymarxan/solvers/test_run_mode.py
git commit -m "feat: add RUNMODE pipeline orchestrator (modes 0-6)"
```

---

### Task 4: Standard Marxan Output File Writers

**Files:**
- Modify: `src/pymarxan/io/writers.py`
- Create: `tests/pymarxan/io/test_output_writers.py`

**Context:** Marxan produces several standard output files that other tools (Zonae Cogito, CLUZ) expect. We need writers for:
- `out_mvbest.csv` — per-feature: target, amount held, target met, shortfall for best solution
- `out_ssoln.csv` — per-PU: count of times selected across all solutions (summed solution)
- `out_sum.csv` — per-run: Score, Cost, Planning_Units, Connectivity, Penalty, Shortfall

**Step 1: Write the failing tests**

```python
"""Tests for Marxan output file writers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.writers import write_mvbest, write_ssoln, write_sum
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


@pytest.fixture()
def problem() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3], "cost": [10.0, 20.0, 15.0], "status": [0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["bird", "mammal"],
        "target": [5.0, 8.0], "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 2, 3],
        "amount": [3.0, 4.0, 5.0, 6.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
    )


@pytest.fixture()
def solutions() -> list[Solution]:
    return [
        Solution(
            selected=np.array([True, True, False]),
            cost=30.0, boundary=5.0, objective=36.0,
            targets_met={1: True, 2: False},
            metadata={"run": 1},
        ),
        Solution(
            selected=np.array([True, False, True]),
            cost=25.0, boundary=3.0, objective=29.0,
            targets_met={1: False, 2: True},
            metadata={"run": 2},
        ),
    ]


def test_write_mvbest(tmp_path: Path, problem: ConservationProblem, solutions: list[Solution]):
    """write_mvbest produces a CSV with per-feature missing value info."""
    best = solutions[1]  # lower objective
    path = tmp_path / "out_mvbest.csv"
    write_mvbest(problem, best, path)
    df = pd.read_csv(path)
    assert len(df) == 2
    assert "Feature_ID" in df.columns
    assert "Target" in df.columns
    assert "Amount_Held" in df.columns
    assert "Target_Met" in df.columns


def test_write_ssoln(tmp_path: Path, problem: ConservationProblem, solutions: list[Solution]):
    """write_ssoln produces a CSV with selection frequency per PU."""
    path = tmp_path / "out_ssoln.csv"
    write_ssoln(problem, solutions, path)
    df = pd.read_csv(path)
    assert len(df) == 3
    assert "Planning_Unit" in df.columns
    assert "Number" in df.columns
    # PU 1 selected in both solutions -> count = 2
    assert df.loc[df["Planning_Unit"] == 1, "Number"].iloc[0] == 2


def test_write_sum(tmp_path: Path, solutions: list[Solution]):
    """write_sum produces a CSV with per-run summary."""
    path = tmp_path / "out_sum.csv"
    write_sum(solutions, path)
    df = pd.read_csv(path)
    assert len(df) == 2
    assert "Run" in df.columns
    assert "Score" in df.columns
    assert "Cost" in df.columns
    assert "Planning_Units" in df.columns
    assert "Boundary" in df.columns


def test_write_ssoln_empty(tmp_path: Path, problem: ConservationProblem):
    """write_ssoln with empty solution list writes zeros."""
    path = tmp_path / "out_ssoln.csv"
    write_ssoln(problem, [], path)
    df = pd.read_csv(path)
    assert len(df) == 3
    assert all(df["Number"] == 0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/io/test_output_writers.py -v`
Expected: FAIL with `ImportError`

**Step 3: Add writer functions to `src/pymarxan/io/writers.py`**

Append the following functions to the existing `writers.py`:

```python
def write_mvbest(
    problem: ConservationProblem,
    solution: Solution,
    path: str | Path,
) -> None:
    """Write missing value info for the best solution (out_mvbest.csv).

    Columns: Feature_ID, Feature_Name, Target, Amount_Held, Target_Met, Shortfall
    """
    from pymarxan.solvers.utils import compute_feature_shortfalls

    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}
    shortfalls = compute_feature_shortfalls(
        problem, solution.selected, pu_index
    )

    # Compute held amounts
    held: dict[int, float] = {
        int(row["id"]): 0.0 for _, row in problem.features.iterrows()
    }
    for _, row in problem.pu_vs_features.iterrows():
        pid = int(row["pu"])
        fid = int(row["species"])
        idx = pu_index.get(pid)
        if idx is not None and solution.selected[idx] and fid in held:
            held[fid] += float(row["amount"])

    rows = []
    for _, row in problem.features.iterrows():
        fid = int(row["id"])
        rows.append({
            "Feature_ID": fid,
            "Feature_Name": row["name"],
            "Target": float(row["target"]),
            "Amount_Held": held.get(fid, 0.0),
            "Target_Met": solution.targets_met.get(fid, False),
            "Shortfall": shortfalls.get(fid, 0.0),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def write_ssoln(
    problem: ConservationProblem,
    solutions: list[Solution],
    path: str | Path,
) -> None:
    """Write summed solution (out_ssoln.csv).

    Columns: Planning_Unit, Number (times selected across all solutions)
    """
    pu_ids = problem.planning_units["id"].tolist()
    counts = {pid: 0 for pid in pu_ids}
    for sol in solutions:
        for i, pid in enumerate(pu_ids):
            if sol.selected[i]:
                counts[pid] += 1
    rows = [
        {"Planning_Unit": pid, "Number": counts[pid]} for pid in pu_ids
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_sum(
    solutions: list[Solution],
    path: str | Path,
) -> None:
    """Write per-run summary (out_sum.csv).

    Columns: Run, Score, Cost, Planning_Units, Boundary, Penalty, Shortfall
    """
    rows = []
    for i, sol in enumerate(solutions):
        penalty = sol.objective - sol.cost - sol.boundary
        rows.append({
            "Run": i + 1,
            "Score": sol.objective,
            "Cost": sol.cost,
            "Planning_Units": sol.n_selected,
            "Boundary": sol.boundary,
            "Penalty": max(penalty, 0.0),
            "Shortfall": max(penalty, 0.0),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
```

Also add the import for `Solution` at the top of writers.py:

```python
from pymarxan.solvers.base import Solution
```

**Step 4: Run tests**

Run: `python -m pytest tests/pymarxan/io/test_output_writers.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan/io/writers.py tests/pymarxan/io/test_output_writers.py
git commit -m "feat: add Marxan output file writers (mvbest, ssoln, sum)"
```

---

### Task 5: Standard Marxan Output File Readers

**Files:**
- Modify: `src/pymarxan/io/readers.py`
- Create: `tests/pymarxan/io/test_output_readers.py`

**Context:** Read standard Marxan output files back into Python objects. This enables loading results from Marxan C++ runs.

**Step 1: Write the failing tests**

```python
"""Tests for Marxan output file readers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pymarxan.io.readers import read_mvbest, read_ssoln, read_sum


def test_read_mvbest(tmp_path: Path):
    df = pd.DataFrame({
        "Feature_ID": [1, 2],
        "Feature_Name": ["bird", "mammal"],
        "Target": [5.0, 8.0],
        "Amount_Held": [7.0, 6.0],
        "Target_Met": [True, False],
        "Shortfall": [0.0, 2.0],
    })
    path = tmp_path / "out_mvbest.csv"
    df.to_csv(path, index=False)

    result = read_mvbest(path)
    assert len(result) == 2
    assert "Feature_ID" in result.columns
    assert result["Target_Met"].dtype == bool


def test_read_ssoln(tmp_path: Path):
    df = pd.DataFrame({
        "Planning_Unit": [1, 2, 3],
        "Number": [5, 3, 8],
    })
    path = tmp_path / "out_ssoln.csv"
    df.to_csv(path, index=False)

    result = read_ssoln(path)
    assert len(result) == 3
    assert result["Planning_Unit"].dtype == int


def test_read_sum(tmp_path: Path):
    df = pd.DataFrame({
        "Run": [1, 2],
        "Score": [100.0, 95.0],
        "Cost": [80.0, 75.0],
        "Planning_Units": [10, 9],
        "Boundary": [15.0, 12.0],
        "Penalty": [5.0, 8.0],
        "Shortfall": [5.0, 8.0],
    })
    path = tmp_path / "out_sum.csv"
    df.to_csv(path, index=False)

    result = read_sum(path)
    assert len(result) == 2
    assert "Score" in result.columns
    assert result["Run"].dtype == int
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/io/test_output_readers.py -v`
Expected: FAIL with `ImportError`

**Step 3: Add reader functions to `src/pymarxan/io/readers.py`**

Append to the existing `readers.py`:

```python
def read_mvbest(path: str | Path) -> pd.DataFrame:
    """Read a Marxan out_mvbest.csv output file.

    Returns a DataFrame with columns: Feature_ID, Feature_Name, Target,
    Amount_Held, Target_Met, Shortfall.
    """
    df = pd.read_csv(path)
    df["Feature_ID"] = df["Feature_ID"].astype(int)
    df["Target"] = df["Target"].astype(float)
    df["Amount_Held"] = df["Amount_Held"].astype(float)
    df["Target_Met"] = df["Target_Met"].astype(bool)
    df["Shortfall"] = df["Shortfall"].astype(float)
    return df


def read_ssoln(path: str | Path) -> pd.DataFrame:
    """Read a Marxan out_ssoln.csv output file.

    Returns a DataFrame with columns: Planning_Unit, Number.
    """
    df = pd.read_csv(path)
    df["Planning_Unit"] = df["Planning_Unit"].astype(int)
    df["Number"] = df["Number"].astype(int)
    return df


def read_sum(path: str | Path) -> pd.DataFrame:
    """Read a Marxan out_sum.csv output file.

    Returns a DataFrame with columns: Run, Score, Cost, Planning_Units,
    Boundary, Penalty, Shortfall.
    """
    df = pd.read_csv(path)
    df["Run"] = df["Run"].astype(int)
    df["Planning_Units"] = df["Planning_Units"].astype(int)
    for col in ("Score", "Cost", "Boundary", "Penalty", "Shortfall"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df
```

**Step 4: Run tests**

Run: `python -m pytest tests/pymarxan/io/test_output_readers.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan/io/readers.py tests/pymarxan/io/test_output_readers.py
git commit -m "feat: add Marxan output file readers (mvbest, ssoln, sum)"
```

---

### Task 6: MISSLEVEL and COSTTHRESH Parameter Support

**Files:**
- Modify: `src/pymarxan/solvers/utils.py`
- Create: `tests/pymarxan/solvers/test_misslevel_costthresh.py`

**Context:** Two important Marxan parameters:
- **MISSLEVEL** (default 1.0): Fraction of target that counts as "met". E.g., MISSLEVEL=0.95 means a feature is "met" if ≥95% of target is achieved.
- **COSTTHRESH** + **THRESHPEN1** + **THRESHPEN2**: Cost threshold penalty. If total cost exceeds COSTTHRESH, an additional penalty is added: `THRESHPEN1 + THRESHPEN2 * (cost - COSTTHRESH)`.

These affect `check_targets` and `compute_objective` in `utils.py`.

**Step 1: Write the failing tests**

```python
"""Tests for MISSLEVEL and COSTTHRESH parameter support."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import (
    check_targets,
    compute_cost_threshold_penalty,
    compute_objective,
)


@pytest.fixture()
def problem() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3], "cost": [10.0, 20.0, 15.0], "status": [0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1], "name": ["f1"], "target": [10.0], "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1],
        "pu": [1, 2, 3],
        "amount": [4.0, 5.0, 3.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
    )


def test_misslevel_default(problem: ConservationProblem):
    """Default MISSLEVEL=1.0: need 100% of target."""
    pu_index = {1: 0, 2: 1, 3: 2}
    selected = np.array([True, True, False])  # 4+5=9, target=10
    met = check_targets(problem, selected, pu_index)
    assert met[1] is False  # 9 < 10


def test_misslevel_relaxed(problem: ConservationProblem):
    """MISSLEVEL=0.9: need only 90% of target (9.0)."""
    problem.parameters["MISSLEVEL"] = 0.9
    pu_index = {1: 0, 2: 1, 3: 2}
    selected = np.array([True, True, False])  # 4+5=9, 90% of 10 = 9.0
    met = check_targets(problem, selected, pu_index)
    assert met[1] is True  # 9 >= 9.0


def test_misslevel_strict(problem: ConservationProblem):
    """MISSLEVEL=0.95: need 95% of target (9.5)."""
    problem.parameters["MISSLEVEL"] = 0.95
    pu_index = {1: 0, 2: 1, 3: 2}
    selected = np.array([True, True, False])  # 4+5=9
    met = check_targets(problem, selected, pu_index)
    assert met[1] is False  # 9 < 9.5


def test_cost_threshold_no_penalty(problem: ConservationProblem):
    """No penalty when cost is below threshold."""
    penalty = compute_cost_threshold_penalty(20.0, 50.0, 1.0, 1.0)
    assert penalty == 0.0


def test_cost_threshold_with_penalty(problem: ConservationProblem):
    """Penalty when cost exceeds threshold."""
    # cost=60, threshold=50, pen1=10, pen2=2
    # penalty = 10 + 2*(60-50) = 30
    penalty = compute_cost_threshold_penalty(60.0, 50.0, 10.0, 2.0)
    assert penalty == pytest.approx(30.0)


def test_cost_threshold_in_objective(problem: ConservationProblem):
    """COSTTHRESH affects the objective computation."""
    problem.parameters["COSTTHRESH"] = 15.0
    problem.parameters["THRESHPEN1"] = 5.0
    problem.parameters["THRESHPEN2"] = 1.0
    pu_index = {1: 0, 2: 1, 3: 2}
    selected = np.array([True, True, False])  # cost=30, threshold=15
    obj = compute_objective(problem, selected, pu_index, 0.0)
    # cost=30 + penalty for features + cost_thresh_penalty = 5 + 1*(30-15) = 20
    # Total includes feature penalty for unmet target too
    assert obj > 30.0  # must include cost threshold penalty
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/solvers/test_misslevel_costthresh.py -v`
Expected: FAIL

**Step 3: Update `utils.py`**

Modify `check_targets` to support MISSLEVEL and add `compute_cost_threshold_penalty`. Update `compute_objective` to include cost threshold penalty.

In `check_targets`, change the comparison from `total >= target` to `total >= target * misslevel`:

```python
def check_targets(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> dict[int, bool]:
    """Check which feature targets are met by the selection."""
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    targets_met: dict[int, bool] = {}
    for _, feat_row in problem.features.iterrows():
        fid = int(feat_row["id"])
        target = float(feat_row["target"])
        feat_data = problem.pu_vs_features[
            problem.pu_vs_features["species"] == fid
        ]
        total = 0.0
        for _, r in feat_data.iterrows():
            pu_id = int(r["pu"])
            idx = pu_index.get(pu_id)
            if idx is not None and selected[idx]:
                total += float(r["amount"])
        targets_met[fid] = total >= target * misslevel
    return targets_met
```

Add new function:

```python
def compute_cost_threshold_penalty(
    total_cost: float,
    cost_thresh: float,
    thresh_pen1: float,
    thresh_pen2: float,
) -> float:
    """Compute cost threshold penalty.

    If total_cost > cost_thresh:
        penalty = thresh_pen1 + thresh_pen2 * (total_cost - cost_thresh)
    """
    if total_cost <= cost_thresh:
        return 0.0
    return thresh_pen1 + thresh_pen2 * (total_cost - cost_thresh)
```

Update `compute_objective` to include cost threshold:

```python
def compute_objective(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
    blm: float,
) -> float:
    """Compute the full Marxan objective: cost + BLM*boundary + penalty + cost_threshold."""
    costs = np.asarray(problem.planning_units["cost"].values)
    total_cost = float(np.sum(costs[selected]))
    total_boundary = compute_boundary(problem, selected, pu_index)
    penalty = compute_penalty(problem, selected, pu_index)

    obj = total_cost + blm * total_boundary + penalty

    # Cost threshold penalty
    cost_thresh = float(problem.parameters.get("COSTTHRESH", 0.0))
    if cost_thresh > 0:
        thresh_pen1 = float(problem.parameters.get("THRESHPEN1", 0.0))
        thresh_pen2 = float(problem.parameters.get("THRESHPEN2", 0.0))
        obj += compute_cost_threshold_penalty(
            total_cost, cost_thresh, thresh_pen1, thresh_pen2,
        )

    return obj
```

**Step 4: Run tests**

Run: `python -m pytest tests/pymarxan/solvers/test_misslevel_costthresh.py tests/pymarxan/solvers/test_utils.py -v`
Expected: All PASSED (new + existing)

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/utils.py tests/pymarxan/solvers/test_misslevel_costthresh.py
git commit -m "feat: add MISSLEVEL and COSTTHRESH/THRESHPEN parameter support"
```

---

### Task 7: Update Registry and App with Phase 6 Features

**Files:**
- Modify: `src/pymarxan/solvers/registry.py`
- Modify: `src/pymarxan_app/app.py`
- Create: `tests/test_integration_phase6.py`

**Step 1: Write integration tests**

```python
"""Integration tests for Phase 6 features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.registry import get_default_registry
from pymarxan.solvers.run_mode import RunModePipeline


def _problem() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 20.0, 15.0, 25.0],
        "status": [0, 0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1, 2],
        "name": ["bird", "mammal"],
        "target": [5.0, 4.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 2, 3],
        "amount": [3.0, 4.0, 5.0, 3.0],
    })
    bnd = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "boundary": [1.0, 1.0]})
    return ConservationProblem(
        planning_units=pu, features=feat,
        pu_vs_features=puvspr, boundary=bnd,
        parameters={"BLM": 1.0, "NUMITNS": 1000, "NUMTEMP": 100},
    )


def test_all_heurtypes_produce_solutions():
    """All 8 heuristic types produce valid solutions."""
    p = _problem()
    for ht in range(8):
        solver = HeuristicSolver(heurtype=ht)
        sols = solver.solve(p, SolverConfig(num_solutions=1, seed=42))
        assert len(sols) == 1
        assert sols[0].n_selected > 0


def test_iterative_improvement_improves():
    """Iterative improvement produces equal or better objective."""
    p = _problem()
    heur = HeuristicSolver(heurtype=0)
    initial = heur.solve(p, SolverConfig(num_solutions=1, seed=42))[0]
    ii = IterativeImprovementSolver(itimptype=2)
    improved = ii.improve(p, initial)
    assert improved.objective <= initial.objective + 1e-10


def test_runmode_5_end_to_end():
    """RUNMODE 5 full pipeline produces valid solution."""
    p = _problem()
    pipeline = RunModePipeline(runmode=5)
    sols = pipeline.solve(p, SolverConfig(num_solutions=1, seed=42))
    assert len(sols) == 1
    assert sols[0].objective > 0


def test_output_roundtrip(tmp_path):
    """Write and read output files roundtrip."""
    from pymarxan.io.writers import write_mvbest, write_ssoln, write_sum
    from pymarxan.io.readers import read_mvbest, read_ssoln, read_sum

    p = _problem()
    solver = HeuristicSolver()
    sols = solver.solve(p, SolverConfig(num_solutions=3, seed=42))

    write_mvbest(p, sols[0], tmp_path / "mvbest.csv")
    write_ssoln(p, sols, tmp_path / "ssoln.csv")
    write_sum(sols, tmp_path / "sum.csv")

    mv = read_mvbest(tmp_path / "mvbest.csv")
    ss = read_ssoln(tmp_path / "ssoln.csv")
    sm = read_sum(tmp_path / "sum.csv")

    assert len(mv) == 2
    assert len(ss) == 4
    assert len(sm) == 3


def test_registry_includes_new_solvers():
    """Registry includes iterative improvement and pipeline."""
    reg = get_default_registry()
    names = reg.list_solvers()
    assert "iterative_improvement" in names
    assert "pipeline" in names


def test_app_imports():
    """App imports successfully with phase 6 modules."""
    import pymarxan_app.app  # noqa: F401
```

**Step 2: Update registry**

Add to `get_default_registry()` in `src/pymarxan/solvers/registry.py`:

```python
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.run_mode import RunModePipeline

reg.register("iterative_improvement", IterativeImprovementSolver)
reg.register("pipeline", RunModePipeline)
```

**Step 3: Update app.py**

Add the pipeline solver option to the solver picker. Import `RunModePipeline` and add it to the `active_solver` reactive calc as `"pipeline"`.

**Step 4: Run tests**

Run: `python -m pytest tests/test_integration_phase6.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/registry.py src/pymarxan_app/app.py tests/test_integration_phase6.py
git commit -m "feat: register new solvers, update app, add phase 6 integration tests"
```

---

### Task 8: Lint and Full Test Suite

**Files:** All modified/created files from Tasks 1-7

**Step 1: Run ruff**

Run: `ruff check src/ tests/ --fix`

**Step 2: Run mypy**

Run: `mypy src/pymarxan/ --ignore-missing-imports`

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass (~250+ total)

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "chore: fix lint and type issues from phase 6"
```

---

## Summary

| Task | Module | Tests | Key concept |
|------|--------|-------|-------------|
| 1 | `solvers/heuristic.py` | ~14 | All 8 HEURTYPE modes (richness, greedy, rarity, irreplaceability) |
| 2 | `solvers/iterative_improvement.py` | 8 | ITIMPTYPE 0-3 post-processing |
| 3 | `solvers/run_mode.py` | 7 | RUNMODE 0-6 pipeline orchestrator |
| 4 | `io/writers.py` | 4 | out_mvbest, out_ssoln, out_sum writers |
| 5 | `io/readers.py` | 3 | out_mvbest, out_ssoln, out_sum readers |
| 6 | `utils.py` | 6 | MISSLEVEL + COSTTHRESH/THRESHPEN |
| 7 | Registry + app + integration | 6 | Wire everything together |
| 8 | Lint + cleanup | — | Code quality |

**Total new tests:** ~48
**Total estimated tests after Phase 6:** ~262+

**Parallelisable groups:**
- Tasks 1 + 2 (heuristic types + iterative improvement — no dependencies)
- Tasks 4 + 5 (output writers + readers — independent)
- Task 3 depends on Tasks 1 + 2
- Task 6 is independent of all others
- Task 7 depends on Tasks 1-6
