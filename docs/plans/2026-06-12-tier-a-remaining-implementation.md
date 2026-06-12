# Remaining Tier A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the three remaining Tier-A features from the 2026-06-12 ecosystem survey — automatic/group target rules (A3), multi-scenario min-regret robustness (A4), and distribution smoothing (A6) — plus Shiny exposure of the already-built representation analysis.

**Architecture:** Each feature is an additive, self-contained module that follows the existing `pymarxan` patterns: a small pure-function/dataclass module under `src/pymarxan/`, a focused test file mirroring it, and (for the UI) one Shiny module mirroring `results/target_met.py`. No existing behaviour changes; everything is new surface. The four task groups are independent and commit separately.

**Tech Stack:** Python, numpy, pandas, scipy (already a core dependency), pulp (only indirectly, via `compute_objective`), pytest, Shiny for Python.

**Already-present — do NOT rebuild (verified in source):** `min_largest_shortfall` objective (`solvers/mip_solver.py:420`), `LinearConstraint`, `MinNeighborConstraint`, `ferrier_importance`, `compute_selection_frequency` (covers the "no-regrets overlap" half of A4), proportional `prop`-column targets (`io/readers.py:273`). The equity analysis (`analysis/equity.py`) and representation analysis (`analysis/representation.py`) are already implemented; only the **Shiny exposure** of representation remains (Task Group 4).

**Run tests with:** `/opt/micromamba/envs/shiny/bin/pytest`. Lint: `ruff check src/ tests/`. Types: `.venv/bin/mypy src/pymarxan/ --ignore-missing-imports`.

---

## Task Group 1 — A3: Automatic & group target rules

**Why:** Currently targets are set per-feature by hand, or proportionally via the `prop` column. prioritizr 8.1 ships a library of target-setting *rules* (relative, log-linear-by-range-size à la IUCN, grouped). This adds the three most-used as pure functions returning `{feature_id: target}`, plus an `apply_targets` helper.

**Files:**
- Create: `src/pymarxan/targets.py`
- Test: `tests/pymarxan/test_targets.py`

### Task 1.1: Relative targets

- [ ] **Step 1: Write the failing test**

Create `tests/pymarxan/test_targets.py`:

```python
"""Tests for automatic target-setting rules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.targets import (
    apply_targets,
    group_targets,
    loglinear_targets,
    relative_targets,
)


def _problem() -> ConservationProblem:
    planning_units = pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]}
    )
    features = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["common", "rare"],
            "target": [0.0, 0.0],
            "spf": [1.0, 1.0],
        }
    )
    # common total = 100 ; rare total = 10
    pu_vs_features = pd.DataFrame(
        {
            "species": [1, 1, 2],
            "pu": [1, 2, 3],
            "amount": [60.0, 40.0, 10.0],
        }
    )
    return ConservationProblem(planning_units, features, pu_vs_features)


def test_relative_targets_are_fraction_of_total():
    problem = _problem()
    targets = relative_targets(problem, 0.30)
    assert targets == {1: pytest.approx(30.0), 2: pytest.approx(3.0)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_targets.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan.targets'`

- [ ] **Step 3: Write minimal implementation**

Create `src/pymarxan/targets.py`:

```python
"""Automatic target-setting rules.

Set each feature's representation target by a rule rather than by hand.
Each rule returns a ``{feature_id: target_amount}`` mapping;
:func:`apply_targets` writes it onto a problem's features.

Mirrors prioritizr's ``add_relative_targets`` / ``add_auto_targets`` /
``add_group_targets`` (Hanson et al. 2024).
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from pymarxan.models.problem import ConservationProblem


def relative_targets(
    problem: ConservationProblem, fraction: float
) -> dict[int, float]:
    """Target = ``fraction`` of each feature's total amount."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    totals = problem.feature_amounts()
    return {
        int(f): fraction * float(totals.get(int(f), 0.0))
        for f in problem.features["id"]
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_targets.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/targets.py tests/pymarxan/test_targets.py
git commit -m "feat(targets): relative target rule (Tier A)"
```

### Task 1.2: Log-linear (range-size) targets

- [ ] **Step 1: Write the failing test**

Append to `tests/pymarxan/test_targets.py`:

```python
def test_loglinear_clamps_below_lower_and_above_upper():
    problem = _problem()
    # below lower_area (10 <= 10) -> upper_target fraction 1.0;
    # above upper_area (100 >= 100) -> lower_target fraction 0.1
    targets = loglinear_targets(
        problem,
        lower_area=10.0,
        lower_target=1.0,
        upper_area=100.0,
        upper_target=0.1,
    )
    # rare total 10 at/below lower_area -> 100% -> 10.0
    assert targets[2] == pytest.approx(10.0)
    # common total 100 at/above upper_area -> 10% -> 10.0
    assert targets[1] == pytest.approx(10.0)


def test_loglinear_interpolates_on_log_scale():
    # A feature whose total is the geometric midpoint of [10, 1000] is 100,
    # which sits halfway in log10 space, so its fraction is halfway too.
    planning_units = pd.DataFrame(
        {"id": [1], "cost": [1.0], "status": [0]}
    )
    features = pd.DataFrame(
        {"id": [1], "name": ["mid"], "target": [0.0], "spf": [1.0]}
    )
    pu_vs_features = pd.DataFrame(
        {"species": [1], "pu": [1], "amount": [100.0]}
    )
    problem = ConservationProblem(planning_units, features, pu_vs_features)
    targets = loglinear_targets(
        problem,
        lower_area=10.0,
        lower_target=1.0,
        upper_area=1000.0,
        upper_target=0.0,
    )
    # fraction = 1.0 + (0.0 - 1.0) * (log10(100)-log10(10))/(log10(1000)-log10(10))
    #          = 1.0 - 0.5 = 0.5 ; target = 0.5 * 100 = 50
    assert targets[1] == pytest.approx(50.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_targets.py::test_loglinear_interpolates_on_log_scale -q`
Expected: FAIL — `ImportError: cannot import name 'loglinear_targets'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/pymarxan/targets.py`:

```python
def loglinear_targets(
    problem: ConservationProblem,
    *,
    lower_area: float,
    lower_target: float,
    upper_area: float,
    upper_target: float,
) -> dict[int, float]:
    """IUCN-style range-size targets, interpolated log-linearly.

    Features whose total amount is at or below ``lower_area`` get the
    ``lower_target`` fraction; at or above ``upper_area`` they get
    ``upper_target``; in between, the fraction is interpolated linearly on
    ``log10`` of the total amount. The returned value is the fraction times
    the feature's total amount.
    """
    if lower_area <= 0 or upper_area <= 0:
        raise ValueError("lower_area and upper_area must be positive")
    if upper_area <= lower_area:
        raise ValueError("upper_area must exceed lower_area")

    totals = problem.feature_amounts()
    log_lo = math.log10(lower_area)
    log_hi = math.log10(upper_area)
    out: dict[int, float] = {}
    for f in problem.features["id"]:
        total = float(totals.get(int(f), 0.0))
        if total <= lower_area:
            frac = lower_target
        elif total >= upper_area:
            frac = upper_target
        else:
            t = (math.log10(total) - log_lo) / (log_hi - log_lo)
            frac = lower_target + (upper_target - lower_target) * t
        out[int(f)] = frac * total
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_targets.py -q`
Expected: PASS (all target tests)

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/targets.py tests/pymarxan/test_targets.py
git commit -m "feat(targets): log-linear range-size target rule (Tier A)"
```

### Task 1.3: Group targets + apply_targets

- [ ] **Step 1: Write the failing test**

Append to `tests/pymarxan/test_targets.py`:

```python
def test_group_targets_apply_group_fraction_to_members():
    problem = _problem()
    groups = {1: "abundant", 2: "scarce"}
    fractions = {"abundant": 0.10, "scarce": 0.50}
    targets = group_targets(problem, groups, fractions)
    # common total 100 * 0.10 = 10 ; rare total 10 * 0.50 = 5
    assert targets == {1: pytest.approx(10.0), 2: pytest.approx(5.0)}


def test_group_targets_unknown_group_raises():
    problem = _problem()
    with pytest.raises(ValueError, match="group"):
        group_targets(problem, {1: "x", 2: "y"}, {"x": 0.1})


def test_apply_targets_writes_targets_onto_features():
    problem = _problem()
    apply_targets(problem, {1: 30.0, 2: 3.0})
    by_id = dict(zip(problem.features["id"], problem.features["target"]))
    assert by_id == {1: pytest.approx(30.0), 2: pytest.approx(3.0)}


def test_apply_targets_leaves_unlisted_features_unchanged():
    problem = _problem()
    problem.features.loc[problem.features["id"] == 2, "target"] = 7.0
    apply_targets(problem, {1: 30.0})  # only feature 1 listed
    by_id = dict(zip(problem.features["id"], problem.features["target"]))
    assert by_id[1] == pytest.approx(30.0)
    assert by_id[2] == pytest.approx(7.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_targets.py -k "group or apply" -q`
Expected: FAIL — `ImportError: cannot import name 'group_targets'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/pymarxan/targets.py`:

```python
def group_targets(
    problem: ConservationProblem,
    groups: Mapping[int, str],
    fractions: Mapping[str, float],
) -> dict[int, float]:
    """Apply a per-group relative target to each member feature.

    ``groups`` maps feature id to a group label; ``fractions`` maps each
    group label to the fraction of total amount to target. Every group
    referenced in ``groups`` must have an entry in ``fractions``.
    """
    missing = {g for g in groups.values() if g not in fractions}
    if missing:
        raise ValueError(f"no fraction given for group(s): {sorted(missing)}")
    totals = problem.feature_amounts()
    out: dict[int, float] = {}
    for f in problem.features["id"]:
        fid = int(f)
        g = groups.get(fid)
        if g is not None:
            out[fid] = float(fractions[g]) * float(totals.get(fid, 0.0))
    return out


def apply_targets(
    problem: ConservationProblem, targets: Mapping[int, float]
) -> ConservationProblem:
    """Write ``{feature_id: target}`` onto the problem's features in place.

    Features not present in ``targets`` keep their existing target. Returns
    the same problem for chaining.
    """
    fmap = {int(k): float(v) for k, v in targets.items()}
    problem.features["target"] = [
        fmap.get(int(fid), float(t))
        for fid, t in zip(problem.features["id"], problem.features["target"])
    ]
    return problem
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_targets.py -q`
Expected: PASS (all 7 target tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff check src/pymarxan/targets.py tests/pymarxan/test_targets.py
.venv/bin/mypy src/pymarxan/targets.py --ignore-missing-imports
git add src/pymarxan/targets.py tests/pymarxan/test_targets.py
git commit -m "feat(targets): group targets + apply_targets helper (Tier A)"
```

---

## Task Group 2 — A6: Distribution smoothing

**Why:** Zonation's distribution smoothing spreads each unit's feature amount to nearby units via a dispersal kernel, so the optimiser values being near abundance, not just holding it. pymarxan has decay kernels (`connectivity/decay.py`) but no smoothing step.

**Files:**
- Create: `src/pymarxan/connectivity/smoothing.py`
- Test: `tests/pymarxan/connectivity/test_smoothing.py`

### Task 2.1: Pairwise distance matrix

- [ ] **Step 1: Write the failing test**

Create `tests/pymarxan/connectivity/test_smoothing.py`:

```python
"""Tests for dispersal-kernel distribution smoothing."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.connectivity.smoothing import (
    distance_matrix_from_points,
    smooth_distribution,
)


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    coords = np.array([[0.0, 0.0], [3.0, 4.0]])
    d = distance_matrix_from_points(coords)
    assert d.shape == (2, 2)
    assert d[0, 0] == pytest.approx(0.0)
    assert d[0, 1] == pytest.approx(5.0)  # 3-4-5 triangle
    assert d[1, 0] == pytest.approx(5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/connectivity/test_smoothing.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan.connectivity.smoothing'`

- [ ] **Step 3: Write minimal implementation**

Create `src/pymarxan/connectivity/smoothing.py`:

```python
"""Distribution smoothing via a dispersal kernel.

Spread each planning unit's feature amount to nearby units using a
negative-exponential dispersal kernel, so a solver values *being near*
abundance, not only holding it. This is the planning-unit (vector)
analogue of Zonation's distribution smoothing.
"""
from __future__ import annotations

import numpy as np

from pymarxan.connectivity.decay import negative_exponential


def distance_matrix_from_points(coords: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix from an ``(n, 2)`` coordinate array."""
    from scipy.spatial.distance import cdist

    coords = np.asarray(coords, dtype=float)
    return cdist(coords, coords)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/connectivity/test_smoothing.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/connectivity/smoothing.py tests/pymarxan/connectivity/test_smoothing.py
git commit -m "feat(connectivity): pairwise distance matrix for smoothing (Tier A)"
```

### Task 2.2: smooth_distribution (mass-conserving)

- [ ] **Step 1: Write the failing test**

Append to `tests/pymarxan/connectivity/test_smoothing.py`:

```python
def test_large_alpha_recovers_original_amounts():
    # With a very steep kernel, off-diagonal weights vanish and smoothing
    # is (near) the identity.
    amounts = np.array([5.0, 0.0, 0.0])
    distances = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    smoothed = smooth_distribution(amounts, distances, alpha=50.0)
    assert smoothed[0] == pytest.approx(5.0, abs=1e-6)
    assert smoothed[1] == pytest.approx(0.0, abs=1e-6)


def test_smoothing_is_mass_conserving_when_normalized():
    amounts = np.array([10.0, 0.0, 0.0])
    distances = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    smoothed = smooth_distribution(amounts, distances, alpha=0.5, normalize=True)
    assert smoothed.sum() == pytest.approx(10.0)
    # Mass spreads to neighbours, so the source loses some.
    assert smoothed[0] < 10.0
    assert smoothed[1] > 0.0


def test_unnormalized_accumulates_mass():
    amounts = np.array([1.0, 1.0])
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])
    smoothed = smooth_distribution(amounts, distances, alpha=0.5, normalize=False)
    # raw K @ amounts with K_ij = exp(-0.5*d): each entry 1 + exp(-0.5) > 1
    assert smoothed[0] == pytest.approx(1.0 + np.exp(-0.5))


def test_alpha_must_be_positive():
    amounts = np.array([1.0])
    distances = np.array([[0.0]])
    with pytest.raises(ValueError, match="alpha"):
        smooth_distribution(amounts, distances, alpha=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/connectivity/test_smoothing.py -k smooth -q`
Expected: FAIL — `ImportError: cannot import name 'smooth_distribution'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/pymarxan/connectivity/smoothing.py`:

```python
def smooth_distribution(
    amounts: np.ndarray,
    distances: np.ndarray,
    alpha: float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Smooth a per-unit feature distribution with a dispersal kernel.

    The kernel is ``K_ij = exp(-alpha * distance_ij)`` (so the diagonal is
    1). With ``normalize=True`` the kernel is column-normalised so total
    amount is conserved — each source unit's amount is redistributed across
    units in proportion to the kernel. With ``normalize=False`` the result
    is the raw ``K @ amounts`` accumulation (total grows).

    Args:
        amounts: Length-``n`` array of per-unit amounts for one feature.
        distances: ``(n, n)`` pairwise distance matrix.
        alpha: Decay rate (> 0); larger = more local.
        normalize: Conserve total amount (default True).

    Returns:
        Length-``n`` array of smoothed amounts.
    """
    # negative_exponential raises ValueError if alpha <= 0.
    kernel = negative_exponential(np.asarray(distances, dtype=float), alpha)
    if normalize:
        col_sums = kernel.sum(axis=0)
        kernel = kernel / col_sums
    result: np.ndarray = kernel @ np.asarray(amounts, dtype=float)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/connectivity/test_smoothing.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff check src/pymarxan/connectivity/smoothing.py tests/pymarxan/connectivity/test_smoothing.py
.venv/bin/mypy src/pymarxan/connectivity/smoothing.py --ignore-missing-imports
git add src/pymarxan/connectivity/smoothing.py tests/pymarxan/connectivity/test_smoothing.py
git commit -m "feat(connectivity): mass-conserving distribution smoothing (Tier A)"
```

---

## Task Group 3 — A4: Multi-scenario min-regret robustness

**Why:** Climate-robust planning chooses a plan that performs acceptably across scenarios, not just the one it was optimised for. `compute_selection_frequency` already gives the no-regrets overlap map; this adds the decision-theoretic **minimax-regret** choice and a builder that evaluates candidate plans across scenario problems via the existing `compute_objective`.

**Files:**
- Create: `src/pymarxan/analysis/robustness.py`
- Test: `tests/pymarxan/analysis/test_robustness.py`

### Task 3.1: minimax_regret decision function

- [ ] **Step 1: Write the failing test**

Create `tests/pymarxan/analysis/test_robustness.py`:

```python
"""Tests for multi-scenario robustness / minimax-regret selection."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.analysis.robustness import RegretResult, minimax_regret


def test_regret_matrix_subtracts_best_in_each_scenario():
    # rows = plans, cols = scenarios; entries are costs (lower better).
    # scenario 0 best = 10 (plan 0); scenario 1 best = 5 (plan 1).
    cost = np.array([[10.0, 20.0], [15.0, 5.0]])
    result = minimax_regret(cost)
    # regret = cost - column min
    assert result.regret_matrix.tolist() == [[0.0, 15.0], [5.0, 0.0]]


def test_minimax_regret_picks_plan_with_smallest_max_regret():
    cost = np.array([[10.0, 20.0], [15.0, 5.0]])
    result = minimax_regret(cost, plan_labels=["A", "B"])
    # plan A max regret = 15 ; plan B max regret = 5 -> choose B
    assert result.max_regret.tolist() == [15.0, 5.0]
    assert result.minimax_regret_plan == "B"


def test_minimax_cost_plan_picks_best_worst_case_cost():
    # plan A worst-case 20, plan B worst-case 15 -> robust choice B
    cost = np.array([[10.0, 20.0], [15.0, 15.0]])
    result = minimax_regret(cost, plan_labels=["A", "B"])
    assert result.minimax_cost_plan == "B"


def test_default_labels_are_indices():
    cost = np.array([[1.0, 2.0], [2.0, 1.0]])
    result = minimax_regret(cost)
    assert result.plan_labels == [0, 1]
    assert result.scenario_labels == [0, 1]


def test_result_is_dataclass():
    result = minimax_regret(np.array([[1.0]]))
    assert isinstance(result, RegretResult)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_robustness.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan.analysis.robustness'`

- [ ] **Step 3: Write minimal implementation**

Create `src/pymarxan/analysis/robustness.py`:

```python
"""Multi-scenario robustness: minimax-regret plan selection.

Given a cost matrix of candidate plans (rows) evaluated under scenarios
(columns), choose the plan that is most robust to which scenario turns out
to be true — either by minimax regret (smallest worst-case regret) or by
minimax cost (smallest worst-case cost).

The field's lighter-weight "no-regrets overlap" is already available via
:func:`pymarxan.analysis.selection_freq.compute_selection_frequency`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RegretResult:
    """Robustness analysis of plans across scenarios (costs, lower better)."""

    cost_matrix: np.ndarray
    regret_matrix: np.ndarray
    max_regret: np.ndarray
    plan_labels: list[Any]
    scenario_labels: list[Any]
    minimax_regret_plan: Any
    minimax_cost_plan: Any

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "plan": self.plan_labels,
                "max_regret": self.max_regret.tolist(),
                "worst_case_cost": self.cost_matrix.max(axis=1).tolist(),
            }
        )


def minimax_regret(
    cost_matrix: np.ndarray,
    plan_labels: list[Any] | None = None,
    scenario_labels: list[Any] | None = None,
) -> RegretResult:
    """Select the most robust plan from a plans-by-scenarios cost matrix.

    Args:
        cost_matrix: ``(n_plans, n_scenarios)`` array of costs (lower is
            better) — plan ``i`` evaluated under scenario ``j``.
        plan_labels: Optional labels for the rows (default: indices).
        scenario_labels: Optional labels for the columns (default: indices).

    Returns:
        A :class:`RegretResult` with the regret matrix, per-plan worst-case
        regret, the minimax-regret plan, and the minimax-cost plan.
    """
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be 2-D (plans x scenarios)")
    n_plans, n_scen = cost.shape
    plans = list(plan_labels) if plan_labels is not None else list(range(n_plans))
    scens = (
        list(scenario_labels)
        if scenario_labels is not None
        else list(range(n_scen))
    )

    best_per_scenario = cost.min(axis=0)  # best (lowest) cost in each column
    regret = cost - best_per_scenario  # broadcast over rows
    max_regret = regret.max(axis=1)
    worst_case_cost = cost.max(axis=1)

    return RegretResult(
        cost_matrix=cost,
        regret_matrix=regret,
        max_regret=max_regret,
        plan_labels=plans,
        scenario_labels=scens,
        minimax_regret_plan=plans[int(np.argmin(max_regret))],
        minimax_cost_plan=plans[int(np.argmin(worst_case_cost))],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_robustness.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/analysis/robustness.py tests/pymarxan/analysis/test_robustness.py
git commit -m "feat(analysis): minimax-regret robustness selection (Tier A)"
```

### Task 3.2: Build the cost matrix from scenario problems + plans

- [ ] **Step 1: Write the failing test**

Append to `tests/pymarxan/analysis/test_robustness.py`:

```python
import pandas as pd

from pymarxan.analysis.robustness import evaluate_plans_across_scenarios
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


def _problem(costs: list[float]) -> ConservationProblem:
    planning_units = pd.DataFrame(
        {"id": [1, 2], "cost": costs, "status": [0, 0]}
    )
    features = pd.DataFrame(
        {"id": [1], "name": ["sp"], "target": [1.0], "spf": [1.0]}
    )
    pu_vs_features = pd.DataFrame(
        {"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]}
    )
    return ConservationProblem(planning_units, features, pu_vs_features)


def _sol(selected: list[bool]) -> Solution:
    return Solution(
        selected=np.array(selected, dtype=bool),
        cost=0.0,
        boundary=0.0,
        objective=0.0,
        targets_met={},
    )


def test_evaluate_plans_builds_objective_matrix():
    # Two scenarios differ only in PU costs.
    scen_a = _problem([10.0, 20.0])  # PU1 cheap
    scen_b = _problem([20.0, 10.0])  # PU2 cheap
    plan_pick1 = _sol([True, False])  # selects PU1
    plan_pick2 = _sol([False, True])  # selects PU2

    matrix, plans, scens = evaluate_plans_across_scenarios(
        problems={"A": scen_a, "B": scen_b},
        solutions={"pick1": plan_pick1, "pick2": plan_pick2},
        blm=0.0,
    )
    # rows align to solutions order, cols to problems order.
    assert plans == ["pick1", "pick2"]
    assert scens == ["A", "B"]
    # pick1 (PU1) costs 10 under A, 20 under B ; pick2 (PU2) costs 20 / 10.
    assert matrix[plans.index("pick1"), scens.index("A")] == pytest.approx(10.0)
    assert matrix[plans.index("pick1"), scens.index("B")] == pytest.approx(20.0)
    assert matrix[plans.index("pick2"), scens.index("B")] == pytest.approx(10.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_robustness.py::test_evaluate_plans_builds_objective_matrix -q`
Expected: FAIL — `ImportError: cannot import name 'evaluate_plans_across_scenarios'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/pymarxan/analysis/robustness.py` (add imports at top: `from collections.abc import Mapping`, `from pymarxan.models.problem import ConservationProblem`, `from pymarxan.solvers.base import Solution`, `from pymarxan.solvers.utils import compute_objective`):

```python
def evaluate_plans_across_scenarios(
    problems: Mapping[str, ConservationProblem],
    solutions: Mapping[str, Solution],
    *,
    blm: float = 0.0,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Evaluate each plan's objective under every scenario problem.

    Args:
        problems: Mapping of scenario label to its :class:`ConservationProblem`.
        solutions: Mapping of plan label to a :class:`Solution` whose
            ``selected`` array is re-evaluated under each scenario.
        blm: Boundary-length modifier used for the objective.

    Returns:
        ``(cost_matrix, plan_labels, scenario_labels)`` where
        ``cost_matrix[i, j]`` is plan ``i`` evaluated under scenario ``j``.
        Feed it straight into :func:`minimax_regret`.
    """
    plan_labels = list(solutions.keys())
    scenario_labels = list(problems.keys())
    matrix = np.zeros((len(plan_labels), len(scenario_labels)))

    for j, slabel in enumerate(scenario_labels):
        problem = problems[slabel]
        pu_index = {int(pid): k for k, pid in enumerate(problem.planning_units["id"])}
        for i, plabel in enumerate(plan_labels):
            selected = np.asarray(solutions[plabel].selected, dtype=bool)
            matrix[i, j] = compute_objective(problem, selected, pu_index, blm)
    return matrix, plan_labels, scenario_labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_robustness.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff check src/pymarxan/analysis/robustness.py tests/pymarxan/analysis/test_robustness.py
.venv/bin/mypy src/pymarxan/analysis/robustness.py --ignore-missing-imports
git add src/pymarxan/analysis/robustness.py tests/pymarxan/analysis/test_robustness.py
git commit -m "feat(analysis): build plan-by-scenario cost matrix (Tier A)"
```

---

## Task Group 4 — Shiny exposure of the representation report

**Why:** `analysis/representation.py` is built and tested but has no UI. Add one Shiny results module mirroring `results/target_met.py`: a threshold slider plus a per-feature representation table. (Equity UI is deferred — it needs a per-PU group column the standard UI does not yet collect; tracked as a follow-up, not part of this plan.)

**Files:**
- Create: `src/pymarxan_shiny/modules/results/representation.py`
- Modify: `src/pymarxan_shiny/app.py` (imports near line 47-51; `ui.nav_panel` near line 148; server registration near line 317)
- Test: `tests/pymarxan_shiny/modules/results/test_representation_module.py`

### Task 4.1: Representation results module

- [ ] **Step 1: Write the failing test**

Create `tests/pymarxan_shiny/modules/results/test_representation_module.py`:

```python
"""Smoke test: the representation Shiny module imports and exposes ui/server."""
from __future__ import annotations


def test_module_exposes_ui_and_server():
    from pymarxan_shiny.modules.results.representation import (
        representation_server,
        representation_ui,
    )

    assert callable(representation_ui)
    assert callable(representation_server)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/modules/results/test_representation_module.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan_shiny.modules.results.representation'`

- [ ] **Step 3: Write minimal implementation**

Create `src/pymarxan_shiny/modules/results/representation.py`:

```python
"""Representation (30x30 / GBF Target 3) Shiny module.

Shows, for the current solution, what fraction of each feature is
represented and whether it clears a user-set policy threshold.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.analysis.representation import compute_representation
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup


@module.ui
def representation_ui():
    return ui.card(
        help_card_header("Representation (30x30)"),
        ui.p(
            "Per-feature share of each feature's total amount captured by "
            "the current solution, and whether it clears the policy "
            "threshold (default 30% for the Global Biodiversity Framework "
            "Target 3).",
            class_="text-muted small mb-3",
        ),
        ui.input_slider(
            "threshold", "Representation threshold", min=0, max=100, value=30, post="%"
        ),
        ui.output_text("summary"),
        ui.output_data_frame("rep_table"),
    )


@module.server
def representation_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    help_server_setup(input, "representation")

    @reactive.Calc
    def _result():  # type: ignore[valid-type]
        p = problem()
        s = solution()
        if p is None or s is None:
            return None
        return compute_representation(p, s, threshold=input.threshold() / 100.0)

    @render.text
    def summary():
        r = _result()
        if r is None:
            return "No solution yet."
        return (
            f"{r.n_features_meeting}/{len(r.feature_ids)} features meet the "
            f"{r.threshold:.0%} threshold."
        )

    @render.data_frame
    def rep_table():
        r = _result()
        if r is None:
            return None
        return r.to_dataframe()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/modules/results/test_representation_module.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/results/representation.py tests/pymarxan_shiny/modules/results/test_representation_module.py
git commit -m "feat(shiny): representation (30x30) results module (Tier A)"
```

### Task 4.2: Wire the module into the app

- [ ] **Step 1: Add the import**

In `src/pymarxan_shiny/app.py`, next to the other results imports (~line 51), add:

```python
from modules.results.representation import representation_server, representation_ui
```

- [ ] **Step 2: Add the nav panel**

In `app.py`, next to the Targets panel (~line 148), add a panel:

```python
            ui.nav_panel("Representation", representation_ui("representation")),
```

- [ ] **Step 3: Register the server**

In `app.py`, next to `target_met_server(...)` (~line 317), add:

```python
    representation_server("representation", problem=problem, solution=current_solution)
```

- [ ] **Step 4: Verify the app imports cleanly**

Run: `/opt/micromamba/envs/shiny/bin/python -c "import sys; sys.path.insert(0, 'src/pymarxan_shiny'); import app"`
Expected: no output, exit 0 (module imports without error)

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff check src/pymarxan_shiny/modules/results/representation.py
.venv/bin/mypy src/pymarxan_shiny/ --ignore-missing-imports
git add src/pymarxan_shiny/app.py
git commit -m "feat(shiny): wire representation panel into the app (Tier A)"
```

---

## Final verification

- [ ] **Run the full check suite**

```bash
ruff check src/ tests/ examples/
.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports
/opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" --deselect tests/test_release_script.py
```

Expected: 0 ruff errors, mypy "Success", all tests pass (≈1421 + ~23 new = ~1444). The known-flaky `test_solutions_are_different` may need a re-run.

- [ ] **Update CHANGELOG.md** — add `[Unreleased]` entries for targets, smoothing, robustness, and the representation UI under `### Added`.

- [ ] **Commit the changelog**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog for remaining Tier A features"
```

---

## Self-review notes

- **Spec coverage:** A3 (relative + log-linear + group targets) → Task Group 1; A6 (distribution smoothing) → Task Group 2; A4 (minimax-regret + cost-matrix builder) → Task Group 3; representation UI → Task Group 4. The equity *UI* is explicitly deferred (needs a per-PU group column the app does not collect) and is noted, not silently dropped.
- **Already-present items** (`min_largest_shortfall`, neighbour/linear constraints, Ferrier, selection-frequency overlap, `prop` targets) are called out at the top so no task rebuilds them.
- **Type consistency:** `relative_targets`/`loglinear_targets`/`group_targets` all return `dict[int, float]`; `apply_targets` consumes that. `evaluate_plans_across_scenarios` returns `(matrix, plan_labels, scenario_labels)` which is exactly `minimax_regret`'s input shape. `compute_objective(problem, selected, pu_index, blm)` matches `solvers/utils.py:449`.
- **Sequencing:** the four groups are independent; recommended order is 1 → 2 → 3 → 4 (targets and smoothing are the simplest). Each task commits on its own.
- **Bundle target:** ship all of Tier A (these four groups plus the already-merged equity + representation analyses) as a single **v0.5.0 "modern conservation planning"** minor, after the PyPI/JOSS adoption thread lands.
