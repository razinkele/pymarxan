# restoptr MESH greedy optimizer — implementation plan

> **For agentic workers:** TDD, one bite-sized step at a time. Steps use `- [ ]`. Tests under the
> `shiny` micromamba env.

**Goal:** Add `pymarxan.restoration.greedy_mesh_restore` — a benefit–cost greedy that chooses
restorable cells to maximize effective mesh size under a cost budget, returning a `RestorationResult`
(plan + budget–MESH frontier).

**Architecture:** New module `src/pymarxan/restoration/optimize.py`; a module function (like
`zonation.rank_removal`), NOT a Marxan `Solver`. Pure numpy + `compute_mesh`. No solver/objective
change.

**Tech stack:** Python 3.12+, numpy. `from __future__ import annotations`, full type hints.

## Global constraints

- `MESH = compute_mesh(grid, existing_habitat | restored)` — the optimizer never re-implements the
  measure; it calls `compute_mesh` directly (skipping `restore_mesh`'s per-call subset validation).
- Candidate gains are **re-evaluated every iteration** (adding a cell changes neighbours' gains —
  never cache). After picking `c*`, set `current` to that candidate's already-computed MESH.
- Pure new subpackage code — parity anchor (35.0) untouched.

---

### Task 1: `greedy_mesh_restore` + `RestorationResult`

**Files:**
- Create: `src/pymarxan/restoration/optimize.py`
- Modify: `src/pymarxan/restoration/__init__.py` (export both)
- Test: `tests/pymarxan/restoration/test_optimize.py`

**Interfaces produced:**
- `RestorationResult(restored, mesh, baseline_mesh, total_cost, n_restored, mesh_curve)` —
  `@dataclass(eq=False)`.
- `greedy_mesh_restore(problem, budget, *, criterion="gain_per_cost", connectivity="rook",
  cell_area=None) -> RestorationResult`.

- [ ] **Step 1: Write the failing tests.** Create `tests/pymarxan/restoration/test_optimize.py`:

```python
"""Tests for greedy_mesh_restore (MESH-maximizing restoration optimizer)."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.models.grid import GridGeometry
from pymarxan.restoration import (
    RestorationProblem,
    RestorationResult,
    compute_mesh,
    greedy_mesh_restore,
)


def _grid(nrow=1, ncol=5):
    return GridGeometry(x_min=0.0, y_max=float(nrow), cell_width=1.0, cell_height=1.0,
                        mask=np.ones((nrow, ncol), dtype=bool))


def _strip_problem(cost=None):
    # 1x5 strip; ends (0,4) already habitat (2 patches); middle 3 (1,2,3) restorable.
    g = _grid(1, 5)
    existing = np.array([True, False, False, False, True])
    restorable = np.array([False, True, True, True, False])
    return RestorationProblem(g, existing, restorable, cost)


def test_returns_restoration_result():
    rp = _strip_problem()
    res = greedy_mesh_restore(rp, budget=1.0)
    assert isinstance(res, RestorationResult)
    assert res.restored.dtype == bool
    assert res.n_restored == 1
    assert res.total_cost == pytest.approx(1.0)


def test_plan_is_subset_of_restorable_and_roundtrips():
    rp = _strip_problem()
    res = greedy_mesh_restore(rp, budget=2.0)
    assert not (res.restored & ~rp.restorable).any()  # ⊆ restorable
    assert res.mesh == pytest.approx(compute_mesh(rp.grid, rp.existing_habitat | res.restored).mesh)
    assert res.baseline_mesh == pytest.approx(rp.baseline_mesh().mesh)


def test_greedy_picks_bridge_cell_first():
    # ends habitat + one middle cell adjacent to an end gives the biggest merge. With cells (1,2,3)
    # restorable, restoring cell 1 extends the left end-patch {0} -> {0,1} (area 2), the largest
    # single-step gain. budget 1 -> exactly cell index 1 chosen.
    rp = _strip_problem()
    res = greedy_mesh_restore(rp, budget=1.0)
    assert list(np.flatnonzero(res.restored)) == [1]


def test_budget_fills_uniform_cost():
    rp = _strip_problem()
    # budget 3 (uniform cost 1) restores all 3 restorable middle cells -> whole strip habitat.
    res = greedy_mesh_restore(rp, budget=3.0)
    assert res.n_restored == 3
    assert res.mesh == pytest.approx(compute_mesh(rp.grid, np.ones(5, bool)).mesh)  # one patch, max


def test_mesh_curve_monotone_and_shaped():
    rp = _strip_problem()
    res = greedy_mesh_restore(rp, budget=2.0)
    curve = res.mesh_curve
    assert curve[0] == pytest.approx(res.baseline_mesh)
    assert curve[-1] == pytest.approx(res.mesh)
    assert len(curve) == res.n_restored + 1
    assert np.all(np.diff(curve) >= -1e-12)  # non-decreasing


def test_cost_budget_honored_nonuniform():
    # costs: middle cells expensive except cell 2 cheap; budget only affords the cheap one.
    cost = np.array([1.0, 5.0, 1.0, 5.0, 1.0])
    rp = _strip_problem(cost=cost)
    res = greedy_mesh_restore(rp, budget=1.0)
    assert res.total_cost <= 1.0
    assert list(np.flatnonzero(res.restored)) == [2]  # only cell 2 affordable


def test_gain_per_cost_vs_gain_differ():
    # cell 1 (cost 4) has the biggest raw gain (extends an end patch); cell 2 (cost 1) a smaller
    # gain but far better per-cost. budget 4: gain -> cell 1; gain_per_cost -> cheaper cells.
    cost = np.array([1.0, 4.0, 1.0, 1.0, 1.0])
    rp = _strip_problem(cost=cost)
    by_gain = greedy_mesh_restore(rp, budget=4.0, criterion="gain")
    by_ratio = greedy_mesh_restore(rp, budget=4.0, criterion="gain_per_cost")
    assert list(np.flatnonzero(by_gain.restored)) != list(np.flatnonzero(by_ratio.restored))


def test_zero_cost_cell_restored_first():
    cost = np.array([1.0, 1.0, 0.0, 1.0, 1.0])  # cell 2 free
    rp = _strip_problem(cost=cost)
    res = greedy_mesh_restore(rp, budget=0.0)  # only the free cell is affordable
    assert list(np.flatnonzero(res.restored)) == [2]
    assert res.total_cost == pytest.approx(0.0)


def test_budget_zero_no_restorable_empty_plan():
    rp = _strip_problem(cost=np.ones(5))
    res = greedy_mesh_restore(rp, budget=0.0)
    assert res.n_restored == 0
    assert res.mesh == pytest.approx(res.baseline_mesh)
    assert list(res.mesh_curve) == [pytest.approx(res.baseline_mesh)]


def test_no_restorable_cells():
    g = _grid(1, 3)
    rp = RestorationProblem(g, np.array([True, False, True]), np.zeros(3, bool))
    res = greedy_mesh_restore(rp, budget=10.0)
    assert res.n_restored == 0


def test_negative_budget_and_bad_criterion_raise():
    rp = _strip_problem()
    with pytest.raises(ValueError):
        greedy_mesh_restore(rp, budget=-1.0)
    with pytest.raises(ValueError):
        greedy_mesh_restore(rp, budget=1.0, criterion="nope")
```

- [ ] **Step 2: Run to verify they fail.**
Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/restoration/test_optimize.py -q`
Expected: FAIL — `greedy_mesh_restore` not importable.

- [ ] **Step 3: Implement `optimize.py`.**

```python
"""greedy_mesh_restore — benefit–cost greedy MESH maximization under a restoration budget.

A fast heuristic (MESH = Σarea²/A_total is supermodular, so no approximation guarantee; restoptr
uses an exact CP solver). Module function, not a Marxan Solver — mirrors zonation.rank_removal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymarxan.restoration.mesh import compute_mesh
from pymarxan.restoration.problem import RestorationProblem

_CRITERIA = ("gain_per_cost", "gain")


@dataclass(eq=False)  # numpy field breaks the auto __eq__ (repo convention)
class RestorationResult:
    restored: np.ndarray      # (n_pu,) bool — the chosen restoration plan
    mesh: float               # final post-restoration MESH
    baseline_mesh: float      # pre-restoration MESH
    total_cost: float
    n_restored: int
    mesh_curve: np.ndarray    # MESH after each step (index 0 = baseline)


def greedy_mesh_restore(
    problem: RestorationProblem,
    budget: float,
    *,
    criterion: str = "gain_per_cost",
    connectivity: str = "rook",
    cell_area: float | None = None,
) -> RestorationResult:
    """Greedily restore cells to maximize effective mesh size under a cost budget."""
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")

    grid = problem.grid
    existing = problem.existing_habitat
    cost = problem.cost
    assert cost is not None  # set in RestorationProblem.__post_init__

    restored = np.zeros(problem.n_pu, dtype=bool)
    spent = 0.0
    current = float(compute_mesh(grid, existing, connectivity=connectivity,
                                 cell_area=cell_area).mesh)
    baseline = current
    mesh_curve = [current]

    while True:
        best_idx = -1
        best_score = -np.inf
        best_mesh = current
        for c in problem.restorable_indices:
            c = int(c)
            if restored[c]:
                continue
            cc = float(cost[c])
            if spent + cc > budget:  # unaffordable
                continue
            trial = existing | restored
            trial[c] = True
            m = float(compute_mesh(grid, trial, connectivity=connectivity,
                                   cell_area=cell_area).mesh)
            gain = m - current
            if gain <= 0:
                continue  # no-op cell (e.g. already-habitat overlap)
            score = np.inf if cc == 0.0 else gain / cc
            if score > best_score:
                best_score = score
                best_idx = c
                best_mesh = m
        if best_idx < 0:
            break
        restored[best_idx] = True
        spent += float(cost[best_idx])
        current = best_mesh
        mesh_curve.append(current)

    return RestorationResult(
        restored=restored,
        mesh=current,
        baseline_mesh=baseline,
        total_cost=float(spent),
        n_restored=int(restored.sum()),
        mesh_curve=np.asarray(mesh_curve, dtype=float),
    )
```

Update `src/pymarxan/restoration/__init__.py`:

```python
from pymarxan.restoration.mesh import MeshResult, compute_mesh
from pymarxan.restoration.optimize import RestorationResult, greedy_mesh_restore
from pymarxan.restoration.problem import RestorationProblem

__all__ = [
    "MeshResult",
    "RestorationProblem",
    "RestorationResult",
    "compute_mesh",
    "greedy_mesh_restore",
]
```

- [ ] **Step 4: Run to verify pass.**
Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/restoration/test_optimize.py -q`
Expected: PASS. ruff + mypy clean
(`.venv/bin/mypy src/pymarxan/restoration/ --ignore-missing-imports`).

- [ ] **Step 5: Parity + CHANGELOG.**
Run: `/opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py` → 35.0.
Add to `CHANGELOG.md` `[Unreleased]`:

```markdown
- **MESH-maximizing restoration optimizer (restoptr-style, `pymarxan.restoration`).**
  `greedy_mesh_restore(problem, budget)` greedily restores cells (best marginal MESH gain per unit
  cost) under a cost budget, returning a `RestorationResult` (the restoration plan + a budget–MESH
  frontier `mesh_curve`). A fast heuristic — restoptr uses an exact constraint-programming solver.
  Completes the restoptr MESH arc: measure → data model → optimizer. SA refinement, `min_restore`,
  and connectivity indices (IIC/PC) are follow-ons.
```

- [ ] **Step 6: Full check + commit.**
Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Commit: `feat(restoration): greedy_mesh_restore — MESH-maximizing restoration optimizer`.

## Self-review

- **Spec coverage:** greedy benefit-cost + budget ✓; `gain_per_cost`/`gain` ✓; zero-cost → inf ✓;
  stop on `gain <= 0` ✓ (overlap no-op); monotone/shaped `mesh_curve` ✓; budget honored + subset +
  round-trip ✓; edge cases (budget 0 / no restorable / negative / bad criterion) ✓ — all have tests.
- **Placeholders:** none.
- **Type consistency:** `RestorationResult` fields match the tests; `compute_mesh(grid, mask, *,
  connectivity, cell_area)` call matches mesh.py; `restorable_indices` from `RestorationProblem`.
- **Design-review handoff:** architect / grounding / independent-redesign on the greedy correctness
  (recompute-all, `current = best_mesh` not `+=`, stop condition), the `gain_per_cost` zero-cost
  guard, and the round-trip `result.mesh == compute_mesh(existing|restored)`. Grounding should RUN
  the greedy on the strip problem and confirm the hand-reasoned picks. Science settled (MESH review).
```
