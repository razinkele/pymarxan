# Zonation Phase C — distribution smoothing implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional distribution-smoothing pre-step to the Zonation ranking — a `SmoothingSpec` config that smooths each feature column of the matrix before `rank_removal` runs, threaded through `rank_removal` and `ZonationSolver`.

**Architecture:** `SmoothingSpec.apply(q)` reuses `connectivity.smoothing.smooth_distribution` (the vector analogue of Zonation's distribution smoothing) per feature column. `rank_removal` calls it once, right after `build_pu_feature_matrix()`; the rest of the engine (ranking + curves) runs unchanged on the smoothed matrix. No new deps.

**Tech Stack:** Python 3.12+, NumPy, pandas, scipy (already a dep, via `distance_matrix_from_points`).

**Design spec:** `docs/plans/2026-07-14-zonation-phase-c-design.md` (read it first).

## Global Constraints

- Python 3.12+, `from __future__ import annotations` at the top of every new file, full type hints.
- Zero new third-party dependencies (reuses `pymarxan.connectivity.smoothing`).
- Tests **must** run under the `shiny` micromamba env: `/opt/micromamba/envs/shiny/bin/pytest`.
- Lint: ruff (E, F, I, UP; line length 99). Types: mypy clean. Coverage threshold 75%.
- The bar before done: `make check` green (lint + types + full suite).
- Commit after each task (TDD: failing test → implementation → passing test → commit).
- Reused API: `smooth_distribution(amounts, distances, alpha, *, normalize=True)` and `distance_matrix_from_points(coords)` from `pymarxan.connectivity.smoothing` (normalize=True conserves total).
- `SmoothingSpec` is `@dataclass(eq=False)` — numpy-array fields break the auto `__eq__` (and it becomes hashable).

## File Structure

- `src/pymarxan/zonation/smoothing.py` — `SmoothingSpec` dataclass.
- `src/pymarxan/zonation/rank_removal.py` — **modify**: add `smoothing` param + one-line pre-transform.
- `src/pymarxan/solvers/zonation_solver.py` — **modify**: add `smoothing` param + passthrough.
- `src/pymarxan/zonation/__init__.py` — **modify**: export `SmoothingSpec`.
- `tests/pymarxan/zonation/test_smoothing.py` — `SmoothingSpec` unit tests.
- `tests/pymarxan/zonation/test_rank_removal.py` — **append** the order-flip integration test.
- `tests/pymarxan/solvers/test_zonation_solver.py` — **append** the passthrough test.
- `CHANGELOG.md`.

**Reference oracle (order flip):** a single feature peaked on PU1
(`q=[[10],[0],[0]]`), PUs on a 1-D line `coords=[[0],[1],[2]]`. Without smoothing,
PU2/PU3 hold nothing → CAZ removal order `[2, 3, 1]` (by PU index among the ties).
With smoothing, PU2 (adjacent to the peak) inherits more value than PU3, so PU3 is
removed first → order `[3, 2, 1]`. The near-neighbor out-ranks the far cell.

---

### Task 1: `SmoothingSpec`

**Files:**
- Create: `src/pymarxan/zonation/smoothing.py`
- Test: `tests/pymarxan/zonation/test_smoothing.py`

**Interfaces:**
- Consumes: `smooth_distribution`, `distance_matrix_from_points` (`connectivity.smoothing`).
- Produces: `SmoothingSpec(alpha, coords=None, distances=None)` with `resolve_distances(n_pu)` and `apply(q) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/zonation/test_smoothing.py`:

```python
"""Tests for the Zonation SmoothingSpec (Phase C)."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.connectivity.smoothing import distance_matrix_from_points
from pymarxan.zonation.smoothing import SmoothingSpec


def test_point_mass_spreads_monotonically():
    spec = SmoothingSpec(alpha=1.0, coords=np.array([[0.0], [1.0], [2.0]]))
    q = np.array([[10.0], [0.0], [0.0]])  # feature peaked on PU1
    sm = spec.apply(q)[:, 0]
    assert sm[0] > sm[1] > sm[2] > 0  # decays with distance from the peak
    assert sm.sum() == pytest.approx(10.0)  # total conserved


def test_total_conserved_per_feature():
    spec = SmoothingSpec(alpha=0.5, coords=np.array([[0.0], [1.0], [2.0], [3.0]]))
    q = np.array([[3.0, 1.0], [0.0, 2.0], [5.0, 0.0], [0.0, 4.0]])
    sm = spec.apply(q)
    assert sm[:, 0].sum() == pytest.approx(q[:, 0].sum())
    assert sm[:, 1].sum() == pytest.approx(q[:, 1].sum())


def test_distances_matches_coords():
    coords = np.array([[0.0], [1.0], [2.0]])
    q = np.array([[10.0], [0.0], [0.0]])
    d = distance_matrix_from_points(coords)
    by_coords = SmoothingSpec(alpha=1.0, coords=coords).apply(q)
    by_dist = SmoothingSpec(alpha=1.0, distances=d).apply(q)
    assert np.allclose(by_coords, by_dist)


def test_alpha_must_be_positive():
    with pytest.raises(ValueError, match="alpha"):
        SmoothingSpec(alpha=0.0, coords=np.array([[0.0]]))


def test_requires_exactly_one_of_coords_or_distances():
    with pytest.raises(ValueError, match="exactly one"):
        SmoothingSpec(alpha=1.0)  # neither
    with pytest.raises(ValueError, match="exactly one"):
        SmoothingSpec(
            alpha=1.0, coords=np.array([[0.0]]), distances=np.array([[0.0]])
        )  # both


def test_resolve_distances_validates_shape():
    spec = SmoothingSpec(alpha=1.0, distances=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="distances must be"):
        spec.resolve_distances(3)  # wrong distances shape
    spec2 = SmoothingSpec(alpha=1.0, coords=np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="coords must be"):
        spec2.resolve_distances(3)  # 1-D coords
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_smoothing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymarxan.zonation.smoothing'`.

- [ ] **Step 3: Implement `SmoothingSpec`**

Create `src/pymarxan/zonation/smoothing.py`:

```python
"""Distribution-smoothing config for Zonation rank-removal (Phase C)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymarxan.connectivity.smoothing import (
    distance_matrix_from_points,
    smooth_distribution,
)


@dataclass(eq=False)
class SmoothingSpec:
    """Distribution-smoothing configuration for :func:`rank_removal`.

    Spreads each feature's amount to nearby planning units via a dispersal
    kernel before ranking (Zonation's distribution smoothing), reusing
    ``connectivity.smoothing``. Provide exactly one of ``coords`` (PU
    coordinates; distances computed via ``distance_matrix_from_points``) or a
    precomputed ``distances`` matrix. ``eq=False`` because numpy-array fields
    make the auto-generated ``__eq__`` raise on comparison.
    """

    alpha: float
    coords: np.ndarray | None = None
    distances: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if (self.coords is None) == (self.distances is None):
            raise ValueError("provide exactly one of 'coords' or 'distances'")

    def resolve_distances(self, n_pu: int) -> np.ndarray:
        if self.distances is not None:
            d = np.asarray(self.distances, dtype=float)
            if d.shape != (n_pu, n_pu):
                raise ValueError(f"distances must be ({n_pu}, {n_pu}), got {d.shape}")
            return d
        coords = np.asarray(self.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[0] != n_pu:
            raise ValueError(
                f"coords must be 2-D with {n_pu} rows, got shape {coords.shape}"
            )
        return distance_matrix_from_points(coords)

    def apply(self, q: np.ndarray) -> np.ndarray:
        """Return a copy of ``q`` with each feature column smoothed."""
        n_pu, n_feat = q.shape
        distances = self.resolve_distances(n_pu)
        smoothed = np.empty_like(q, dtype=float)
        for j in range(n_feat):
            smoothed[:, j] = smooth_distribution(q[:, j], distances, self.alpha)
        return smoothed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_smoothing.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/zonation/smoothing.py tests/pymarxan/zonation/test_smoothing.py
git commit -m "feat(zonation): SmoothingSpec — distribution-smoothing config reusing connectivity.smoothing"
```

---

### Task 2: Thread `smoothing` through the engine + solver + CHANGELOG

**Files:**
- Modify: `src/pymarxan/zonation/rank_removal.py`
- Modify: `src/pymarxan/solvers/zonation_solver.py`
- Modify: `src/pymarxan/zonation/__init__.py`
- Modify: `tests/pymarxan/zonation/test_rank_removal.py` (append)
- Modify: `tests/pymarxan/solvers/test_zonation_solver.py` (append)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `SmoothingSpec` (Task 1).
- Produces: `rank_removal(..., smoothing=None)` and `ZonationSolver(..., smoothing=None)`.

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/pymarxan/zonation/test_rank_removal.py` (it already defines `_problem`, imports `rank_removal`, `numpy as np`):

```python
def test_smoothing_changes_ranking():
    from pymarxan.zonation.smoothing import SmoothingSpec

    # feature peaked on PU1; without smoothing PU2/PU3 hold none (removed by
    # index -> [2,3,1]); with smoothing PU2 (near) inherits value, out-ranks PU3.
    problem = _problem([[10], [0], [0]], feat_ids=(1,))
    coords = np.array([[0.0], [1.0], [2.0]])
    plain = rank_removal(problem, rule="caz")
    smoothed = rank_removal(
        problem, rule="caz", smoothing=SmoothingSpec(alpha=1.0, coords=coords)
    )
    assert plain.removal_order == [2, 3, 1]
    assert smoothed.removal_order == [3, 2, 1]  # near-neighbor PU2 now out-ranks PU3
```

Append to `tests/pymarxan/solvers/test_zonation_solver.py` (it already defines `_problem`, imports `ZonationSolver`):

```python
def test_smoothing_passthrough_matches_engine():
    import numpy as np

    from pymarxan.zonation.rank_removal import rank_removal
    from pymarxan.zonation.smoothing import SmoothingSpec

    spec = SmoothingSpec(alpha=1.0, coords=np.array([[0.0], [1.0], [2.0]]))
    problem = _problem([[10], [0], [0]], feat_ids=(1,))
    sol = ZonationSolver(smoothing=spec, top_fraction=2 / 3).solve(problem)[0]
    engine_top = rank_removal(problem, smoothing=spec).top_fraction(2 / 3)
    selected_ids = {
        int(pid)
        for pid, s in zip(problem.planning_units["id"], sol.selected)
        if s
    }
    assert selected_ids == engine_top
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/test_rank_removal.py::test_smoothing_changes_ranking tests/pymarxan/solvers/test_zonation_solver.py::test_smoothing_passthrough_matches_engine -v`
Expected: FAIL — `rank_removal()`/`ZonationSolver()` got an unexpected keyword argument `smoothing`.

- [ ] **Step 3: Thread `smoothing` through `rank_removal`**

In `src/pymarxan/zonation/rank_removal.py`, add the import after the `result` import:

```python
from pymarxan.zonation.smoothing import SmoothingSpec
```

Add the parameter to the signature (after `use_cost: bool = True,`):

```python
    smoothing: SmoothingSpec | None = None,
```

And apply it immediately after the matrix is built — change:

```python
    q = problem.build_pu_feature_matrix()  # (n_pu, n_feat), rows = PU order
```

to:

```python
    q = problem.build_pu_feature_matrix()  # (n_pu, n_feat), rows = PU order
    if smoothing is not None:
        q = smoothing.apply(q)
```

- [ ] **Step 4: Thread `smoothing` through `ZonationSolver`**

In `src/pymarxan/solvers/zonation_solver.py`, add the import after the `rank_removal` import:

```python
from pymarxan.zonation.smoothing import SmoothingSpec
```

Add the parameter to `__init__` (after `use_cost: bool = True,`) and store it:

```python
        smoothing: SmoothingSpec | None = None,
```
```python
        self.smoothing = smoothing
```

And pass it in the `solve()` call to `rank_removal` (after `use_cost=self.use_cost,`):

```python
            smoothing=self.smoothing,
```

- [ ] **Step 5: Export `SmoothingSpec`**

In `src/pymarxan/zonation/__init__.py`, add the import and `__all__` entry:

```python
from pymarxan.zonation.rank_removal import rank_removal
from pymarxan.zonation.result import ZonationResult
from pymarxan.zonation.smoothing import SmoothingSpec

__all__ = ["SmoothingSpec", "ZonationResult", "rank_removal"]
```

- [ ] **Step 6: Run the integration tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zonation/ tests/pymarxan/solvers/test_zonation_solver.py -q`
Expected: PASS (all zonation + solver tests, including the 2 new integration tests).

- [ ] **Step 7: Add the CHANGELOG entry**

Under `## [Unreleased]` → `### Added` in `CHANGELOG.md` (add the headers if the section is empty):

```markdown
- **Zonation distribution smoothing (`SmoothingSpec`, Phase C).** An optional
  ``smoothing=SmoothingSpec(alpha, coords=...)`` on ``rank_removal`` /
  ``ZonationSolver`` spreads each feature's amount to nearby planning units via a
  mass-conserving dispersal kernel before ranking (Zonation's distribution
  smoothing), reusing ``connectivity.smoothing``. +8 tests.
```

- [ ] **Step 8: Run the full check**

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: `make check` green — 0 ruff, 0 mypy, full suite passes (previous count + 8). If `test_solutions_are_different` fails, rerun once (known SA flake).

Note: the CLAUDE.md `micromamba.sh` activation path may not exist on this machine; the `PATH=...` prefix above is the working invocation.

- [ ] **Step 9: Commit**

```bash
git add src/pymarxan/zonation/rank_removal.py src/pymarxan/solvers/zonation_solver.py src/pymarxan/zonation/__init__.py tests/pymarxan/zonation/test_rank_removal.py tests/pymarxan/solvers/test_zonation_solver.py CHANGELOG.md
git commit -m "feat(zonation): thread SmoothingSpec through rank_removal + ZonationSolver + CHANGELOG"
```

---

## Post-plan notes

- **Design review:** the user requested `multi-agent-design-review` before executing. Worth a light pass — the smoothing math (mass-conserving kernel is not doubly stochastic) and the reuse boundary with `connectivity.smoothing` are the things the scientific/grounding lenses should confirm.
- **Parity:** adds no Marxan-solver/objective math (smoothing is a pre-transform reusing the tested `connectivity.smoothing` kernel). The 35.0 anchor is untouched; a quick `marxan-parity-check` after `make check` confirms it.
- **Deferred (own spec):** Phase D Shiny panel + solver-picker wiring.
- **Scientific citations:** Moilanen 2005 / Lehtomäki & Moilanen 2013 (Zonation smoothing lineage; scite-verified in Phase A).
