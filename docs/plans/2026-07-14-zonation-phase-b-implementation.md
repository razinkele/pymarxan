# Zonation Phase B — `ZonationSolver` implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the Phase A `rank_removal` engine in a `Solver`-ABC adapter (`ZonationSolver`) that thresholds the priority ranking into one deterministic reserve, and register it as `"zonation"`.

**Architecture:** `ZonationSolver.solve()` runs `rank_removal`, takes the top `top_fraction` of the ranking via `ZonationResult.top_fraction`, builds a boolean selection, and returns a single `Solution` (rank map + curves in `Solution.metadata`) via the shared `build_solution`. Deterministic — one solution regardless of `num_solutions`. No new deps.

**Tech Stack:** Python 3.12+, NumPy, pandas.

**Design spec:** `docs/plans/2026-07-14-zonation-phase-b-design.md` (read it first).

## Global Constraints

- Python 3.12+, `from __future__ import annotations` at the top of every new file, full type hints.
- Zero new third-party dependencies.
- Tests **must** run under the `shiny` micromamba env: `/opt/micromamba/envs/shiny/bin/pytest`.
- Domain models are dataclasses; test layout mirrors `src/` under `tests/pymarxan/`.
- Lint: ruff (E, F, I, UP; line length 99). Types: mypy clean. Coverage threshold 75%.
- The bar before done: `make check` green (lint + types + full suite).
- Commit after each task (TDD: failing test → implementation → passing test → commit).
- `build_solution(problem, selected, blm, metadata)` (`solvers/utils.py:463`) fills cost/boundary/objective/targets_met from a boolean selection; BLM convention is `float(problem.parameters.get("BLM", 0.0))` (`heuristic.py:282`).
- `registry.get(name)` instantiates via `self._solvers[name]()` (no args), so `ZonationSolver()` must be valid with all defaults.

## File Structure

- `src/pymarxan/solvers/zonation_solver.py` — `ZonationSolver(Solver)`.
- `src/pymarxan/solvers/registry.py` — **modify** `get_default_registry()`: add the local import + `reg.register("zonation", ZonationSolver)`.
- `tests/pymarxan/solvers/test_zonation_solver.py`.
- `CHANGELOG.md` — `[Unreleased]` entry.

**Reference oracle:** **P1** `q=[[10,0],[0,10],[5,5]]`, uniform cost, CAZ ranking gives `priority_rank` PU3=1/3, PU1=2/3, PU2=1.0. So `top_fraction(2/3)` = `ceil(2)` = the two highest-ranked = `{PU1, PU2}` → `selected == [True, True, False]`, cost 2.0.

---

### Task 1: `ZonationSolver`

**Files:**
- Create: `src/pymarxan/solvers/zonation_solver.py`
- Test: `tests/pymarxan/solvers/test_zonation_solver.py`

**Interfaces:**
- Consumes: `rank_removal` + `ZonationResult.top_fraction` (Phase A); `Solver`/`Solution`/`SolverConfig` (`solvers/base.py`); `build_solution` (`solvers/utils.py`).
- Produces: `ZonationSolver(*, rule="caz", top_fraction=0.3, warp=1, weights=None, use_cost=True)` with `solve()`, `name()`, `supports_zones()`, `available()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_zonation_solver.py`:

```python
"""Tests for the ZonationSolver adapter (Phase B)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.zonation_solver import ZonationSolver


def _problem(q_rows, cost=None, status=None, feat_ids=(1, 2)) -> ConservationProblem:
    n_pu = len(q_rows)
    n_feat = len(q_rows[0])
    pu_ids = list(range(1, n_pu + 1))
    planning_units = pd.DataFrame(
        {
            "id": pu_ids,
            "cost": [1.0] * n_pu if cost is None else list(cost),
            "status": [0] * n_pu if status is None else list(status),
        }
    )
    features = pd.DataFrame(
        {
            "id": list(feat_ids),
            "name": [f"f{j}" for j in feat_ids],
            "target": [1.0] * n_feat,
            "spf": [1.0] * n_feat,
        }
    )
    rows = []
    for pu, row in zip(pu_ids, q_rows):
        for fid, amt in zip(feat_ids, row):
            if amt:
                rows.append({"species": fid, "pu": pu, "amount": float(amt)})
    pu_vs_features = pd.DataFrame(rows, columns=["species", "pu", "amount"])
    return ConservationProblem(planning_units, features, pu_vs_features)


P1 = [[10, 0], [0, 10], [5, 5]]  # CAZ ranks: PU3=1/3, PU1=2/3, PU2=1.0


def test_thresholded_reserve_hand_known():
    sol = ZonationSolver(rule="caz", top_fraction=2 / 3).solve(_problem(P1))[0]
    assert sol.selected.tolist() == [True, True, False]  # {PU1, PU2}
    assert sol.cost == pytest.approx(2.0)


def test_metadata_carries_ranking():
    sol = ZonationSolver(top_fraction=2 / 3).solve(_problem(P1))[0]
    assert sol.metadata["solver"] == "zonation"
    assert sol.metadata["rule"] == "caz"
    assert sol.metadata["priority_rank"][2] == pytest.approx(1.0)
    assert "prop_landscape_remaining" in sol.metadata["performance_curves"].columns


def test_build_solution_populated():
    sol = ZonationSolver(top_fraction=2 / 3).solve(_problem(P1))[0]
    assert len(sol.targets_met) == 2
    assert sol.all_targets_met  # {PU1, PU2} covers both features
    assert np.isfinite(sol.objective)


def test_top_fraction_controls_size_monotone():
    small = ZonationSolver(top_fraction=1 / 3).solve(_problem(P1))[0]
    big = ZonationSolver(top_fraction=1.0).solve(_problem(P1))[0]
    assert small.n_selected <= big.n_selected
    assert big.n_selected == 3  # top_fraction=1.0 selects all


def test_deterministic_single_solution():
    sols = ZonationSolver().solve(_problem(P1), SolverConfig(num_solutions=5))
    assert len(sols) == 1


def test_abc_surface():
    s = ZonationSolver()
    assert s.name() == "Zonation (rank-removal)"
    assert s.supports_zones() is False
    assert s.available() is True


def test_invalid_rule_raises():
    with pytest.raises(ValueError, match="rule"):
        ZonationSolver(rule="bogus")


def test_invalid_top_fraction_raises():
    with pytest.raises(ValueError, match="top_fraction"):
        ZonationSolver(top_fraction=0.0)
    with pytest.raises(ValueError, match="top_fraction"):
        ZonationSolver(top_fraction=1.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_zonation_solver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymarxan.solvers.zonation_solver'`.

- [ ] **Step 3: Implement the solver**

Create `src/pymarxan/solvers/zonation_solver.py`:

```python
"""Zonation rank-removal as a Solver-ABC adapter (Phase B)."""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.utils import build_solution
from pymarxan.zonation.rank_removal import rank_removal


class ZonationSolver(Solver):
    """Threshold a Zonation priority ranking into a single reserve.

    Runs :func:`pymarxan.zonation.rank_removal` and selects the top
    ``top_fraction`` of the ranking as the reserve. Deterministic: one ranking
    -> one reserve, so ``solve`` returns a length-1 list regardless of
    ``config.num_solutions``. The full ranking and performance curves ride in
    ``Solution.metadata``. Zonation ranks by biological loss and does not
    optimize to meet feature targets, so a low ``top_fraction`` may leave
    ``all_targets_met`` False by design.
    """

    def __init__(
        self,
        *,
        rule: str = "caz",
        top_fraction: float = 0.3,
        warp: int = 1,
        weights: dict[int, float] | None = None,
        use_cost: bool = True,
    ) -> None:
        if rule not in ("caz", "abf"):
            raise ValueError(f"rule must be 'caz' or 'abf', got {rule!r}")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
        self.rule = rule
        self.top_fraction = top_fraction
        self.warp = warp
        self.weights = weights
        self.use_cost = use_cost

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        result = rank_removal(
            problem,
            rule=self.rule,
            weights=self.weights,
            warp=self.warp,
            use_cost=self.use_cost,
        )
        selected_ids = result.top_fraction(self.top_fraction)
        selected = np.array(
            [int(pid) in selected_ids for pid in problem.planning_units["id"]],
            dtype=bool,
        )
        blm = float(problem.parameters.get("BLM", 0.0))
        meta = {
            "solver": "zonation",
            "rule": self.rule,
            "top_fraction": self.top_fraction,
            "priority_rank": result.priority_rank,
            "performance_curves": result.performance_curves,
        }
        return [build_solution(problem, selected, blm, metadata=meta)]

    def name(self) -> str:
        return "Zonation (rank-removal)"

    def supports_zones(self) -> bool:
        return False

    def available(self) -> bool:
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_zonation_solver.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/solvers/zonation_solver.py tests/pymarxan/solvers/test_zonation_solver.py
git commit -m "feat(zonation): ZonationSolver — Solver-ABC adapter thresholding the ranking"
```

---

### Task 2: Register + CHANGELOG + full-suite green

**Files:**
- Modify: `src/pymarxan/solvers/registry.py`
- Modify: `tests/pymarxan/solvers/test_zonation_solver.py` (append the registry test)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the failing registry test**

Append to `tests/pymarxan/solvers/test_zonation_solver.py`:

```python
def test_registered_in_default_registry():
    from pymarxan.solvers.registry import get_default_registry

    solver = get_default_registry().get("zonation")
    assert isinstance(solver, ZonationSolver)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_zonation_solver.py::test_registered_in_default_registry -v`
Expected: FAIL — `get("zonation")` raises `KeyError` (not registered yet).

- [ ] **Step 3: Register the solver**

In `src/pymarxan/solvers/registry.py`, inside `get_default_registry()`, add the local import (alphabetically, after the `simulated_annealing` import and before `zones.solver`):

```python
    from pymarxan.solvers.zonation_solver import ZonationSolver
```

and add the registration alongside the others (after the `zone_sa` line):

```python
    reg.register("zonation", ZonationSolver)
```

- [ ] **Step 4: Run the solver tests + a registry smoke check**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_zonation_solver.py -v`
Expected: PASS (9 tests, including the registry test).

- [ ] **Step 5: Add the CHANGELOG entry**

Under `## [Unreleased]` → `### Added` in `CHANGELOG.md` (add the headers if the section is empty):

```markdown
- **`ZonationSolver` (Zonation Phase B).** A `Solver`-ABC adapter over the Phase A
  rank-removal engine: ``ZonationSolver(rule=..., top_fraction=0.3).solve(problem)``
  thresholds the priority ranking into one deterministic reserve (rank map +
  performance curves in ``Solution.metadata``), registered as ``"zonation"`` in
  the default solver registry. +9 tests.
```

- [ ] **Step 6: Run the full check**

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: `make check` green — 0 ruff, 0 mypy, full suite passes (previous count + 9). If `test_solutions_are_different` fails, rerun once (known SA flake).

Note: the CLAUDE.md `micromamba.sh` activation path may not exist on this machine; the `PATH=...` prefix above is the working invocation (shiny env first for the rasterio-capable pytest).

- [ ] **Step 7: Commit**

```bash
git add src/pymarxan/solvers/registry.py tests/pymarxan/solvers/test_zonation_solver.py CHANGELOG.md
git commit -m "feat(zonation): register ZonationSolver as 'zonation' + CHANGELOG"
```

---

## Post-plan notes

- **Design review:** the user requested the four-perspective `multi-agent-design-review` before executing. Worth a light pass — the adapter is thin and additive, but the grounding lens should confirm the `build_solution`/registry/`top_fraction` wiring and that registering doesn't leak into the Shiny picker (verified in the spec: the picker uses a hardcoded choices list).
- **Parity:** adds no Marxan-solver/objective math (`build_solution` is the shared builder; the reserve is chosen by the Phase A ranking). The 35.0 anchor is untouched; a quick `marxan-parity-check` after `make check` confirms it.
- **Deferred (own specs):** Phase C distribution smoothing (reuse `connectivity.smoothing`), Phase D Shiny panel + solver-picker wiring.
- **Scientific citations:** Moilanen 2005/2007 + Lehtomäki & Moilanen 2013 (scite-verified in Phase A).
