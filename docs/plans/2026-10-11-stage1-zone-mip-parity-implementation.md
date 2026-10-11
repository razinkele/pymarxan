# Stage 1 (v0.39.0) Zone-MIP Parity and Solver Tooling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `ZoneMIPSolver` the budgeted objectives (`max_features`, `max_weighted_features`, `min_shortfall`, `min_largest_shortfall`) with prioritizr semantics. Add `min_shortfall`, relative shortfall and an exact two-pass `min_penalties` to `MIPSolver`. Add `MANDATORYALLOCATION`, `write_problem` and `presolve_check`, and deprecate the unused `pymarxan.objectives`.

**Architecture:**
- One solver-agnostic module, `src/pymarxan/solvers/_mip_objectives.py`. It holds `TargetTerm`, `ObjectiveContext`, `ObjectiveTerm`, the four builders and `solve_lexicographic`, and imports only pulp and the stdlib.
- Each MIP solver gets a private, warning-free `_build(problem, config)` and two public entry points, `build_model` and `solve`. Both run the warning preamble inline, so stacklevels do not change.
- Mandatory allocation lives only in `ZonalProblem.pu_zone_rules()`, so no solver code changes.
- `min_set` stays byte-identical to v0.38, pinned by canonical-model goldens recorded in Task 1 before any change.

**Tech Stack:** Python 3.12, numpy, pandas, PuLP 3.3 (CBC only in the `shiny` env; HiGHS is not installed), pytest.

**Spec:** `docs/plans/2026-10-11-stage1-zone-mip-parity-design.md` (rev 3). Review: `docs/plans/2026-10-11-stage1-zone-mip-parity-review.md`, which has the first pass and the second pass S1–S9.

## Global Constraints

- **File conventions.** Every file starts with `from __future__ import annotations` and has full type hints. ruff (E, F, I, UP; line length 99) and mypy must be clean. **Run ruff on the test files at every task handoff.**
- **Test interpreter.** Tests run ONLY with `/opt/micromamba/envs/shiny/bin/pytest` (never `.venv`).
  - In the worktree, run mypy as `/home/razinka/marxan/.venv/bin/mypy`.
  - Run non-pytest Python as `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python`.
- **Anchors must not move.**
  - `tests/data/simple` MIP = 35.0 **with BLM = 0**; the fixture's own BLM 1.0 gives objective 43.0.
  - MarZone anchor `(1, 2, 2)` at 7.0.
  - The zone `min_set` canonical model equals the v0.38 golden.
  - The core `min_set` / absolute `min_largest_shortfall` canonical models equal the v0.38 goldens.
- **Shortfall.**
  - Units are relative by default: shortfall / *R*, where *R* = target × MISSLEVEL.
  - Terms with *R* ≤ 0 are skipped by both shortfall objectives in both solvers.
  - `shortfall="absolute"` on `MIPSolver` reproduces v0.38 `min_largest_shortfall` exactly.
- **Ghost features.** A feature with target > 0 and no `pu_vs_features` rows is a term with `achieved = lpSum([])`. It is never met and has full shortfall.
- **Objective names.** Allowed names are `min_set`, `max_features`, `max_weighted_features`, `min_shortfall`, `min_largest_shortfall`, plus `min_penalties` on the core MIP only. The alias `max_n_targets_met` maps to `max_features`. Zone `min_penalties` is NOT in Stage 1.
- **Budget.** Budgeted objectives require `COSTBUDGET` and raise a `ValueError` naming the objective. Core `min_penalties` does not need one, as in v0.38.
- **Mandatory allocation.**
  - `MANDATORYALLOCATION` accepts 0, 1, 0.0 and 1.0. bool, str and any other value are a `validate()` error, and the accessor returns False for them.
  - Status 3 wins: those PUs stay unassigned.
  - It is a pymarxan-only key; MarZone does not read it.
- **Presolve messages.** `presolve_check(problem) -> list[str]` never raises. Every message starts with its code: `"P1: "` … `"P11: "`, and `"P0: "` for the `validate()` short-circuit. This prefix is a plan-level addition: it gives tests stable codes (first review, LOW).
- **Presolve flag.** `presolve_check` accepts only `True`, `False` or `"strict"`, checked by identity. `0` and `1` are rejected with `ValueError` at construction.
- **Metadata.** Both MIPs record `metadata["mip_objective_value"]` (natural sign: a count for `max_*`) and `metadata["shortfall"]`. The zone MIP also records `metadata["objective"]` (the canonical name).
- **Commits.** Messages reference `(stage1)` and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Benches.** Bench assertions are comparative, never absolute wall-clock.

## Review Focus

1. **`zone_costs` omits some (PU, zone) pairs** (common in hand-built MarZone projects). Budgeted objectives must treat a missing pair as cost 0 (the MarZone convention), not crash or treat it as forbidden. Test: Task 7 `test_budgeted_missing_zone_cost_is_free`.
2. **Every target is 0, so there are no terms.** `max_features` must return value 0 and `min_shortfall` value 0 with a feasible plan, not fail on an empty objective. Test: Task 7 `test_budgeted_no_terms`.
3. **Non-contiguous zone ids `{5, 9}` with `MANDATORYALLOCATION 1` and a budget.** Every eligible PU lands in 5 or 9, and the budget-floor message names the right cost. Test: Task 8 `test_mandatory_non_contiguous_zone_ids`.
4. **A NaN amount in `pu_vs_features`, or an `inf` cost.** `presolve_check` must report P1 (or P0 when `validate()` already rejects it) and must never raise. Test: Task 10 `test_presolve_nan_amount_never_raises`.
5. **One solver instance reused:** `build_model(p1)`, then `solve(p2)`, then `solve(p1)` with different objectives in the data. No state leaks; the results equal fresh-instance results. Test: Task 7 `test_reused_instance_no_state_leak`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/pymarxan/solvers/_mip_objectives.py` | `TargetTerm`, `ObjectiveTerm`, `ObjectiveContext`, `BUILDERS`, `resolve_objective`, `set_objective`, `solve_lexicographic`, `SHORTFALL_UNITS` | Create |
| `src/pymarxan/solvers/mip_solver.py` | `_build` / `build_model` / `solve` split; `shortfall=`, `presolve_check=`; `min_shortfall`; relative `min_largest_shortfall`; two-pass `min_penalties`; S9 and ghost fixes | Modify |
| `src/pymarxan/zones/mip_solver.py` | Same split; `objective=`, `shortfall=`, `presolve_check=`; `_zone_target_terms`; budgeted objectives; boundary-ignored warning; budget message | Modify |
| `src/pymarxan/zones/diagnostics.py` | `_zone_cost_lookup`, `min_forced_cost` | Modify |
| `src/pymarxan/zones/model.py` | `mandatory_allocation()`, the rule change, the `validate()` error, the status-3 issue | Modify |
| `src/pymarxan/solvers/presolve.py` | `presolve_check`, `PresolveWarning`, `scan_model`, `_validate_presolve_flag`, `_emit_presolve` | Create |
| `src/pymarxan/solvers/export.py` | `write_problem` | Create |
| `src/pymarxan/solvers/__init__.py`, `src/pymarxan/__init__.py` | Exports (lazy in `pymarxan`) | Modify |
| `src/pymarxan/objectives/__init__.py` | `DeprecationWarning` | Modify |
| `src/pymarxan_shiny/modules/solver_config/objective_selector.py` | Docstring: setting has no effect yet | Modify |
| `tests/pymarxan/lp_canon.py` | `canonical_model`, `captured_models` (test helper) | Create |
| `tests/data/golden/stage1_v038.json` | v0.38 goldens | Create |
| `tests/pymarxan/solvers/test_stage1_golden_v038.py` | Golden recorder and checker | Create |
| `tests/pymarxan/solvers/test_mip_objectives_module.py` | Builder unit tests | Create |
| `tests/pymarxan/solvers/test_mip_stage1.py` | Core MIP changes | Create |
| `tests/pymarxan/zones/test_zone_mip_objectives.py` | Zone objectives | Create |
| `tests/pymarxan/zones/test_mandatory_allocation.py` | Mandatory allocation | Create |
| `tests/pymarxan/solvers/test_write_problem.py` | Export | Create |
| `tests/pymarxan/solvers/test_presolve.py` | Presolve | Create |
| `tests/pymarxan/zones/test_zone_mip_random_oracle.py` | Random-instance enumeration oracle | Create |
| `tests/pymarxan/objectives/test_objectives.py` | Silence deprecation on import | Modify |
| `CHANGELOG.md`, `docs/TUTORIAL.md`, `docs/VALIDATION.md`, `docs/plans/2026-10-11-prioritizr-parity-roadmap.md` | Docs | Modify |

---

### Task 1: Worktree and v0.38 goldens (before any source change)

**Files:**
- Create: `tests/pymarxan/lp_canon.py`
- Create: `tests/pymarxan/solvers/test_stage1_golden_v038.py`
- Create: `tests/data/golden/stage1_v038.json` (generated)

**Interfaces:**
- Produces:
  - `canonical_model(model: pulp.LpProblem) -> dict`
  - `captured_models() -> ContextManager[list[pulp.LpProblem]]`
  - `make_relabs_problem() -> ConservationProblem`: the spec's Testing-5 fixture, reused by Task 5.
  - The golden JSON keys `zone_min_set_anchor`, `zone_min_set_zones`, `core_min_set_simple`, `core_mls_abs_relabs`, `core_mls_abs_infeasible`, `core_mls_abs_relabs_selected`, `core_min_penalties_simple_selected`.

- [ ] **Step 1: Create the worktree**

```bash
cd /home/razinka/marxan
git worktree add .worktrees/stage1-zone-mip -b feat/stage1-zone-mip-parity
cd .worktrees/stage1-zone-mip
```

All later paths are relative to this worktree.

- [ ] **Step 2: Write the canonical-model helper**

`tests/pymarxan/lp_canon.py`:

```python
"""Order-independent dump of a PuLP model, and a spy that captures models at solve()."""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pulp


def _r(v: float) -> float:
    return round(float(v), 12)


def canonical_model(model: pulp.LpProblem) -> dict[str, Any]:
    """Name -> (sense, rhs, sorted coefs) per constraint; sorted objective; variable bounds.

    JSON round-tripped so it compares equal to a dict loaded from a golden file.
    """
    obj = model.objective
    cons = {
        name: {
            "sense": int(c.sense),
            "rhs": _r(-c.constant),
            "coefs": sorted([v.name, _r(a)] for v, a in c.items()),
        }
        for name, c in model.constraints.items()
    }
    out = {
        "sense": int(model.sense),
        "objective": sorted([v.name, _r(a)] for v, a in obj.items()) if obj is not None else [],
        "objective_constant": _r(obj.constant) if obj is not None else 0.0,
        "constraints": cons,
        "variables": sorted([v.name, v.cat, v.lowBound, v.upBound] for v in model.variables()),
    }
    return json.loads(json.dumps(out))


@contextmanager
def captured_models() -> Iterator[list[pulp.LpProblem]]:
    """Record every LpProblem passed to LpProblem.solve (the solve still runs)."""
    models: list[pulp.LpProblem] = []
    orig = pulp.LpProblem.solve

    def spy(self: pulp.LpProblem, *args: Any, **kwargs: Any) -> Any:
        models.append(self)
        return orig(self, *args, **kwargs)

    pulp.LpProblem.solve = spy  # type: ignore[method-assign]
    try:
        yield models
    finally:
        pulp.LpProblem.solve = orig  # type: ignore[method-assign]
```

- [ ] **Step 3: Write the golden recorder and checker**

`tests/pymarxan/solvers/test_stage1_golden_v038.py`:

```python
"""v0.38 MIP models and selections that Stage 1 must not change (spec Testing 4, 5, 12).

GOLDEN was recorded on unmodified v0.38 (e5a066c). Regenerate only with a written reason:
    PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python \
        tests/pymarxan/solvers/test_stage1_golden_v038.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.zones import ZoneMIPSolver, load_zone_project
from tests.pymarxan.lp_canon import canonical_model, captured_models
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

DATA = Path(__file__).parent.parent.parent / "data"
GOLDEN_PATH = DATA / "golden" / "stage1_v038.json"
ONE = SolverConfig(num_solutions=1)


def make_relabs_problem(budget: float = 1.0) -> ConservationProblem:
    """Spec Testing 5: 3 PUs at cost 1. Feature a (target 100): PU1 60, PU3 30.
    Feature b (target 10): PU2 8, PU3 5. Absolute minimax -> PU1 (40), relative -> PU3 (0.7)."""
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "target": [100.0, 10.0], "spf": [1.0, 1.0],
    })
    pvf = pd.DataFrame({
        "species": [1, 1, 2, 2], "pu": [1, 3, 2, 3], "amount": [60.0, 30.0, 8.0, 5.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf,
        parameters={"COSTBUDGET": budget, "BLM": 0.0},
    )


def _infeasible_problem() -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "target": [10.0, 10.0], "spf": [1.0, 1.0],
    })
    pvf = pd.DataFrame({
        "species": [1, 1, 2, 2], "pu": [1, 2, 1, 2], "amount": [5.0, 5.0, 5.0, 5.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf,
        parameters={"COSTBUDGET": 1.0},
    )


def _zones_problem():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_zone_project(DATA / "zones")


def _model_and_selection(solver: Any, problem: Any) -> tuple[dict[str, Any], list[int]]:
    with captured_models() as models:
        sols = solver.solve(problem, ONE)
    sel = [int(v) for v in sols[0].selected] if sols else []
    return canonical_model(models[0]), sel


def record() -> dict[str, Any]:
    simple = load_project(DATA / "simple")
    out: dict[str, Any] = {}
    out["zone_min_set_anchor"], _ = _model_and_selection(ZoneMIPSolver(), make_anchor_problem())
    out["zone_min_set_zones"], _ = _model_and_selection(ZoneMIPSolver(), _zones_problem())
    out["core_min_set_simple"], _ = _model_and_selection(MIPSolver(), simple)
    mls = MIPSolver(objective="min_largest_shortfall")
    out["core_mls_abs_relabs"], out["core_mls_abs_relabs_selected"] = _model_and_selection(
        mls, make_relabs_problem(),
    )
    out["core_mls_abs_infeasible"], _ = _model_and_selection(mls, _infeasible_problem())
    _, out["core_min_penalties_simple_selected"] = _model_and_selection(
        MIPSolver(objective="min_penalties"), simple,
    )
    return out


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())


def test_zone_min_set_models_unchanged(golden: dict[str, Any]) -> None:
    got, _ = _model_and_selection(ZoneMIPSolver(), make_anchor_problem())
    assert got == golden["zone_min_set_anchor"]
    got, _ = _model_and_selection(ZoneMIPSolver(), _zones_problem())
    assert got == golden["zone_min_set_zones"]


def test_core_min_set_model_unchanged(golden: dict[str, Any]) -> None:
    got, _ = _model_and_selection(MIPSolver(), load_project(DATA / "simple"))
    assert got == golden["core_min_set_simple"]


def test_core_absolute_min_largest_shortfall_unchanged(golden: dict[str, Any]) -> None:
    solver = MIPSolver(objective="min_largest_shortfall", shortfall="absolute")
    got, sel = _model_and_selection(solver, make_relabs_problem())
    assert got == golden["core_mls_abs_relabs"]
    assert sel == golden["core_mls_abs_relabs_selected"] == [1, 0, 0]
    got, _ = _model_and_selection(solver, _infeasible_problem())
    assert got == golden["core_mls_abs_infeasible"]


def test_core_min_penalties_integral_selection_unchanged(golden: dict[str, Any]) -> None:
    _, sel = _model_and_selection(
        MIPSolver(objective="min_penalties"), load_project(DATA / "simple"),
    )
    assert sel == golden["core_min_penalties_simple_selected"]


if __name__ == "__main__":
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(record(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH}")
```

`test_core_absolute_min_largest_shortfall_unchanged` uses `shortfall="absolute"`, which v0.38 does not accept. **On v0.38 that test errors with `TypeError`; this is expected.** It turns green in Task 5. The other three pass now.

- [ ] **Step 4: Record the golden on unmodified v0.38**

```bash
PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python tests/pymarxan/solvers/test_stage1_golden_v038.py
python -c "import json;d=json.load(open('tests/data/golden/stage1_v038.json'));print(d['core_mls_abs_relabs_selected'], d['core_min_penalties_simple_selected'])"
```

Expected: `[1, 0, 0]`, followed by a 6-element 0/1 list. Check `git status`: `src/` must be clean, since this records v0.38.

- [ ] **Step 5: Run the checker**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_stage1_golden_v038.py -v`

Expected: 3 passed, 1 failed (`test_core_absolute_min_largest_shortfall_unchanged`: `TypeError ... shortfall`).

- [ ] **Step 6: Commit**

```bash
ruff check tests/pymarxan/lp_canon.py tests/pymarxan/solvers/test_stage1_golden_v038.py
git add tests/pymarxan/lp_canon.py tests/pymarxan/solvers/test_stage1_golden_v038.py tests/data/golden/stage1_v038.json
git commit -m "test(stage1): record v0.38 MIP model/selection goldens before any change

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Shared objective module `_mip_objectives.py`

**Files:**
- Create: `src/pymarxan/solvers/_mip_objectives.py`
- Test: `tests/pymarxan/solvers/test_mip_objectives_module.py`

**Interfaces:**
- Produces (exact):
  - `SHORTFALL_UNITS = ("relative", "absolute")`
  - `OBJECTIVE_ALIASES = {"max_n_targets_met": "max_features"}`
  - `resolve_objective(name: str, allowed: tuple[str, ...]) -> str`
  - `validate_shortfall(value: str) -> str`
  - `TargetTerm(key: str, kind: str, achieved: pulp.LpAffineExpression, required: float, weight: float)` (frozen)
  - `ObjectiveTerm(expr: pulp.LpAffineExpression, sense: int)` (frozen)
  - `ObjectiveContext(model, cost_expr, terms, prefix="")` with `.slack(term)` and `.met(term)`, both create-once
  - `BUILDERS: dict[str, Callable[[ObjectiveContext, str], ObjectiveTerm]]`, keys `max_features`, `max_weighted_features`, `min_shortfall`, `min_largest_shortfall`
  - `set_objective(model: pulp.LpProblem, term: ObjectiveTerm) -> None`
  - `solve_lexicographic(model, primary, secondary, solver, *, rel_tol=1e-9, abs_tol=1e-9) -> int` (returns a pulp status)
  - `NO_SOLUTION_STATUSES: frozenset[int]`

- [ ] **Step 1: Write the failing tests**

`tests/pymarxan/solvers/test_mip_objectives_module.py`:

```python
"""Unit tests for the solver-agnostic MIP objective builders (spec §1, D5)."""
from __future__ import annotations

import warnings

import pulp
import pytest

from pymarxan.solvers._mip_objectives import (
    BUILDERS,
    ObjectiveContext,
    ObjectiveTerm,
    TargetTerm,
    resolve_objective,
    set_objective,
    solve_lexicographic,
    validate_shortfall,
)

CBC = pulp.PULP_CBC_CMD(msg=False)
AMOUNTS = {"a": {1: 60.0, 3: 30.0}, "b": {2: 8.0, 3: 5.0}}
TARGETS = {"a": 100.0, "b": 10.0}


def _solve(objective: str, shortfall: str, *, ghost: bool = False) -> tuple[list[int], float]:
    m = pulp.LpProblem("t", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in (1, 2, 3)}
    terms = [
        TargetTerm(f"f{f}", "feature", pulp.lpSum(a * x[i] for i, a in AMOUNTS[f].items()),
                   TARGETS[f], 1.0)
        for f in AMOUNTS
    ]
    if ghost:
        terms.append(TargetTerm("fghost", "feature", pulp.lpSum([]), 5.0, 1.0))
    ctx = ObjectiveContext(m, pulp.lpSum(x.values()), terms)
    term = BUILDERS[objective](ctx, shortfall)
    set_objective(m, term)
    m += ctx.cost_expr <= 1, "cost_budget"
    m.solve(CBC)
    return [round(x[i].varValue or 0) for i in (1, 2, 3)], float(pulp.value(term.expr))


def test_min_largest_shortfall_relative_vs_absolute() -> None:
    assert _solve("min_largest_shortfall", "relative") == ([0, 0, 1], pytest.approx(0.7))
    assert _solve("min_largest_shortfall", "absolute") == ([1, 0, 0], pytest.approx(40.0))


def test_min_shortfall_values() -> None:
    # relative: PU2 and PU3 tie at 1.2; assert the value only.
    assert _solve("min_shortfall", "relative")[1] == pytest.approx(1.2)
    assert _solve("min_shortfall", "absolute") == ([1, 0, 0], pytest.approx(50.0))


def test_ghost_term_has_full_shortfall() -> None:
    # The ghost adds a constant 1.0 of relative shortfall and is never met.
    assert _solve("min_shortfall", "relative", ghost=True)[1] == pytest.approx(2.2)
    assert _solve("min_largest_shortfall", "relative", ghost=True)[1] == pytest.approx(1.0)
    assert _solve("max_features", "relative", ghost=True)[1] == pytest.approx(0.0)


def test_nonpositive_required_skipped_by_shortfall_builders() -> None:
    m = pulp.LpProblem("z", pulp.LpMinimize)
    x = pulp.LpVariable("x_1", cat="Binary")
    zero = TargetTerm("f0", "feature", pulp.lpSum([x]), 0.0, 1.0)  # MISSLEVEL 0 / target 0
    ctx = ObjectiveContext(m, pulp.lpSum([x]), [zero])
    term = BUILDERS["min_shortfall"](ctx, "relative")  # must not divide by zero
    set_objective(m, term)
    m.solve(CBC)
    assert pulp.LpStatus[m.status] == "Optimal"
    assert "short_f0" not in m.constraints


def test_slack_and_met_are_create_once_and_prefixed() -> None:
    m = pulp.LpProblem("c", pulp.LpMinimize)
    x = pulp.LpVariable("x_1", cat="Binary")
    t = TargetTerm("f1", "feature", pulp.lpSum([3 * x]), 3.0, 2.0)
    ctx = ObjectiveContext(m, pulp.lpSum([x]), [t], prefix="p_")
    assert ctx.slack(t) is ctx.slack(t)
    assert ctx.met(t) is ctx.met(t)
    assert {"p_short_f1", "p_met_f1"} <= set(m.constraints)


def test_compose_two_objectives_in_one_model_no_name_collision() -> None:
    m = pulp.LpProblem("both", pulp.LpMinimize)
    x = pulp.LpVariable("x_1", cat="Binary")
    t = TargetTerm("f1", "feature", pulp.lpSum([3 * x]), 3.0, 1.0)
    ctx = ObjectiveContext(m, pulp.lpSum([x]), [t])
    a = BUILDERS["min_shortfall"](ctx, "relative")
    b = BUILDERS["max_features"](ctx, "relative")
    set_objective(m, ObjectiveTerm(a.expr - b.expr, pulp.LpMinimize))
    m.solve(CBC)  # duplicate names would raise pulp.PulpError on solve
    assert pulp.LpStatus[m.status] == "Optimal"


def test_empty_objective_solves() -> None:
    m = pulp.LpProblem("e", pulp.LpMinimize)
    x = pulp.LpVariable("x", cat="Binary")
    m += x <= 1, "c"
    set_objective(m, ObjectiveTerm(pulp.lpSum([]), pulp.LpMinimize))
    m.solve(CBC)
    assert pulp.LpStatus[m.status] == "Optimal"


def test_resolve_objective_alias_and_errors() -> None:
    allowed = ("min_set", "max_features")
    assert resolve_objective("max_n_targets_met", allowed) == "max_features"
    assert resolve_objective("min_set", allowed) == "min_set"
    with pytest.raises(ValueError, match="max_n_targets_met"):
        resolve_objective("min_penalties", allowed)


def test_validate_shortfall() -> None:
    assert validate_shortfall("absolute") == "absolute"
    with pytest.raises(ValueError, match="relative"):
        validate_shortfall("percent")


def test_solve_lexicographic_beats_small_big_m() -> None:
    # PU1 costs 100 and holds 0.01 (target 0.01); a big-M of 102 would skip it.
    m = pulp.LpProblem("l", pulp.LpMinimize)
    x1 = pulp.LpVariable("x_1", cat="Binary")
    x2 = pulp.LpVariable("x_2", cat="Binary")
    s = pulp.LpVariable("s", lowBound=0)
    m += s >= 0.01 - 0.01 * x1, "short"
    status = solve_lexicographic(m, pulp.lpSum([s]), 100 * x1 + 1 * x2, CBC)
    assert pulp.LpStatus[status] == "Optimal"
    assert (x1.varValue, x2.varValue, s.varValue) == (1.0, 0.0, 0.0)
    assert "lex_primary_cap" in m.constraints


def test_solve_lexicographic_pass2_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    orig = pulp.LpProblem.solve
    calls = {"n": 0}

    def flaky(self: pulp.LpProblem, *a: object, **k: object) -> int:
        calls["n"] += 1
        result = orig(self, *a, **k)
        if calls["n"] == 2:
            self.status = pulp.LpStatusInfeasible
            for v in self.variables():
                v.varValue = None
        return result

    monkeypatch.setattr(pulp.LpProblem, "solve", flaky)
    m = pulp.LpProblem("l2", pulp.LpMinimize)
    x1 = pulp.LpVariable("x_1", cat="Binary")
    s = pulp.LpVariable("s", lowBound=0)
    m += s >= 1 - x1, "short"
    with pytest.warns(UserWarning, match="pass 2"):
        status = solve_lexicographic(m, pulp.lpSum([s]), 5 * x1, CBC)
    assert pulp.LpStatus[status] == "Optimal"
    assert x1.varValue == 1.0  # pass-1 values restored


def test_solve_lexicographic_infeasible_pass1_returns_status() -> None:
    m = pulp.LpProblem("inf", pulp.LpMinimize)
    x = pulp.LpVariable("x", cat="Binary")
    m += x >= 2, "impossible"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no fallback warning on pass-1 failure
        status = solve_lexicographic(m, pulp.lpSum([x]), pulp.lpSum([x]), CBC)
    assert status == pulp.LpStatusInfeasible
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_objectives_module.py -v`

Expected: collection error, `ModuleNotFoundError: No module named 'pymarxan.solvers._mip_objectives'`.

- [ ] **Step 3: Implement the module**

`src/pymarxan/solvers/_mip_objectives.py` (this was executed as a prototype during planning):

```python
"""Solver-agnostic MIP objective builders shared by MIPSolver and ZoneMIPSolver (Stage 1).

Each target is one ``TargetTerm`` (prioritizr: one row per target). Builders return an
``ObjectiveTerm`` in the objective's natural units and sense; the solver sets the model
objective (``set_objective``) and adds ``COSTBUDGET`` once. Imports only pulp and the stdlib.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import pulp

SHORTFALL_UNITS = ("relative", "absolute")
OBJECTIVE_ALIASES = {"max_n_targets_met": "max_features"}  # prioritizr >= 9 name
NO_SOLUTION_STATUSES = frozenset(
    {pulp.LpStatusInfeasible, pulp.LpStatusUnbounded, pulp.LpStatusUndefined}
)


def resolve_objective(name: str, allowed: tuple[str, ...]) -> str:
    """Canonical objective name; ``ValueError`` listing allowed names and aliases."""
    canonical = OBJECTIVE_ALIASES.get(name, name)
    if canonical not in allowed:
        names = allowed + tuple(a for a, c in OBJECTIVE_ALIASES.items() if c in allowed)
        raise ValueError(f"objective must be one of {names}, got {name!r}")
    return canonical


def validate_shortfall(value: str) -> str:
    if value not in SHORTFALL_UNITS:
        raise ValueError(f"shortfall must be one of {SHORTFALL_UNITS}, got {value!r}")
    return value


@dataclass(frozen=True)
class TargetTerm:
    """One target: ``achieved`` A(x) must reach ``required`` R (target × MISSLEVEL)."""

    key: str  # unique name stem: "z2_f5" (zone), "o_f5" (overall), "f5" (core)
    kind: str  # "zone" | "overall" | "feature"
    achieved: pulp.LpAffineExpression  # lpSum([]) for a feature with no rows
    required: float
    weight: float  # SPF (1.0 when the column is absent)


@dataclass(frozen=True)
class ObjectiveTerm:
    expr: pulp.LpAffineExpression
    sense: int  # pulp.LpMinimize | pulp.LpMaximize


@dataclass
class ObjectiveContext:
    model: pulp.LpProblem
    cost_expr: pulp.LpAffineExpression
    terms: list[TargetTerm]
    prefix: str = ""
    _slack: dict[str, pulp.LpVariable] = field(default_factory=dict, repr=False)
    _met: dict[str, pulp.LpVariable] = field(default_factory=dict, repr=False)

    def slack(self, term: TargetTerm) -> pulp.LpVariable:
        """s >= 0 with s >= R - A (constraint ``{prefix}short_{key}``), created once."""
        if term.key not in self._slack:
            s = pulp.LpVariable(f"{self.prefix}s_{term.key}", lowBound=0)
            self.model += s >= term.required - term.achieved, f"{self.prefix}short_{term.key}"
            self._slack[term.key] = s
        return self._slack[term.key]

    def met(self, term: TargetTerm) -> pulp.LpVariable:
        """Binary y with A >= R·y (constraint ``{prefix}met_{key}``), created once."""
        if term.key not in self._met:
            y = pulp.LpVariable(f"{self.prefix}y_{term.key}", cat="Binary")
            self.model += term.achieved >= term.required * y, f"{self.prefix}met_{term.key}"
            self._met[term.key] = y
        return self._met[term.key]


def _scale(term: TargetTerm, shortfall: str) -> float:
    return 1.0 / term.required if shortfall == "relative" else 1.0


def _shortfall_terms(ctx: ObjectiveContext) -> list[TargetTerm]:
    # R <= 0 (target 0 or MISSLEVEL 0): nothing to fall short of, and no division by zero.
    return [t for t in ctx.terms if t.required > 0]


def build_max_features(ctx: ObjectiveContext, shortfall: str) -> ObjectiveTerm:
    return ObjectiveTerm(pulp.lpSum(ctx.met(t) for t in ctx.terms), pulp.LpMaximize)


def build_max_weighted_features(ctx: ObjectiveContext, shortfall: str) -> ObjectiveTerm:
    return ObjectiveTerm(
        pulp.lpSum(t.weight * ctx.met(t) for t in ctx.terms), pulp.LpMaximize,
    )


def build_min_shortfall(ctx: ObjectiveContext, shortfall: str) -> ObjectiveTerm:
    """Σ w·s/R (relative, prioritizr) or Σ w·s (absolute)."""
    return ObjectiveTerm(
        pulp.lpSum(
            t.weight * _scale(t, shortfall) * ctx.slack(t) for t in _shortfall_terms(ctx)
        ),
        pulp.LpMinimize,
    )


def build_min_largest_shortfall(ctx: ObjectiveContext, shortfall: str) -> ObjectiveTerm:
    """m >= s/R (or s) for every term; minimise m. Unweighted, as in prioritizr."""
    m = pulp.LpVariable(f"{ctx.prefix}max_shortfall", lowBound=0)
    for t in _shortfall_terms(ctx):
        ctx.model += m >= _scale(t, shortfall) * ctx.slack(t), f"{ctx.prefix}maxs_{t.key}"
    return ObjectiveTerm(pulp.lpSum([m]), pulp.LpMinimize)


BUILDERS: dict[str, Callable[[ObjectiveContext, str], ObjectiveTerm]] = {
    "max_features": build_max_features,
    "max_weighted_features": build_max_weighted_features,
    "min_shortfall": build_min_shortfall,
    "min_largest_shortfall": build_min_largest_shortfall,
}


def set_objective(model: pulp.LpProblem, term: ObjectiveTerm) -> None:
    """Models stay LpMinimize; a maximised term is negated."""
    model.sense = pulp.LpMinimize
    model.setObjective(term.expr if term.sense == pulp.LpMinimize else -term.expr)


def solve_lexicographic(
    model: pulp.LpProblem,
    primary: pulp.LpAffineExpression,
    secondary: pulp.LpAffineExpression,
    solver: pulp.LpSolver,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> int:
    """Exact two-pass lexicographic minimisation (primary >= 0 assumed).

    Pass 1 minimises ``primary``. Pass 2 adds ``primary <= opt·(1+rel_tol)+abs_tol``
    (``"lex_primary_cap"``) and minimises ``secondary``. If pass 2 yields no solution the
    pass-1 values are restored, one ``UserWarning`` is emitted and pass 1's status returned.
    """
    model.sense = pulp.LpMinimize
    model.setObjective(primary)
    model.solve(solver)
    variables = model.variables()
    if model.status in NO_SOLUTION_STATUSES or all(v.varValue is None for v in variables):
        return int(model.status)
    status1 = int(model.status)
    snapshot = {v.name: v.varValue for v in variables}
    opt = float(pulp.value(primary) or 0.0)
    model += primary <= opt * (1.0 + rel_tol) + abs_tol, "lex_primary_cap"
    model.setObjective(secondary)
    model.solve(solver)
    if model.status in NO_SOLUTION_STATUSES or any(
        v.varValue is None for v in model.variables() if v.name in snapshot
    ):
        for v in model.variables():
            v.varValue = snapshot.get(v.name)
        warnings.warn(
            "lexicographic pass 2 (minimise cost at the optimal penalty) found no solution; "
            "returning the pass-1 solution",
            UserWarning,
            stacklevel=3,
        )
        return status1
    return int(model.status)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_objectives_module.py -v`

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
ruff check src/pymarxan/solvers/_mip_objectives.py tests/pymarxan/solvers/test_mip_objectives_module.py
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/solvers/_mip_objectives.py
git add src/pymarxan/solvers/_mip_objectives.py tests/pymarxan/solvers/test_mip_objectives_module.py
git commit -m "feat(stage1): solver-agnostic MIP objective builders and lexicographic solve

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Core `MIPSolver` bug fixes (S9 zero-cost PU, ghost `feat_met`, spf default)

**Files:**
- Modify: `src/pymarxan/solvers/mip_solver.py`, at `:395` (`features["spf"]`, two places), `:366-372` and `:400-406` (feat_met), and `:573` (has_values)
- Test: `tests/pymarxan/solvers/test_mip_stage1.py`
- Modify: `CHANGELOG.md` (`[Unreleased]` → `### Fixed`)

**Interfaces:**
- Produces: `_core_spf(problem: ConservationProblem) -> dict[int, float]`, a module-level helper in `mip_solver.py` that defaults to 1.0. It is reused in Task 5.

- [ ] **Step 1: Write the failing tests**

`tests/pymarxan/solvers/test_mip_stage1.py`:

```python
"""Stage 1 changes to the single-zone MIPSolver (spec §1 MIPSolver, S2, S9)."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

ONE = SolverConfig(num_solutions=1)


def _two_pu(cost2: float, *, spf: bool = True) -> ConservationProblem:
    """S2/S9 fixture: PU1 cost 100 holds 0.01 of feature 1 (target 0.01); PU2 holds nothing."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [100.0, cost2], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [0.01]})
    if spf:
        feat["spf"] = [1.0]
    pvf = pd.DataFrame({"species": [1], "pu": [1], "amount": [0.01]})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf, parameters={"BLM": 0.0},
    )


def _with_ghost(budget: float) -> ConservationProblem:
    """Feature 1 in PU1; feature 2 (target 5) has no pu_vs_features rows."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "ghost"], "target": [1.0, 5.0], "spf": [1.0, 7.0],
    })
    pvf = pd.DataFrame({"species": [1], "pu": [1], "amount": [1.0]})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf,
        parameters={"COSTBUDGET": budget, "BLM": 0.0},
    )


def test_zero_cost_featureless_pu_still_solves() -> None:
    # S9: v0.38 returned [] because x_2 never entered the model (varValue None).
    sols = MIPSolver().solve(_two_pu(0.0), ONE)
    assert len(sols) == 1
    assert sols[0].selected.tolist() == [True, False]


@pytest.mark.parametrize("objective", ["max_features", "max_weighted_features"])
def test_ghost_feature_not_counted_as_met(objective: str) -> None:
    sol = MIPSolver(objective=objective).solve(_with_ghost(1.0), ONE)[0]
    expected = 1.0  # only feature 1 (spf 1); v0.38 gave 2 / 8 (ghost met for free)
    assert sol.metadata["mip_objective_value"] == pytest.approx(expected)


def test_max_weighted_features_without_spf_column() -> None:
    p = _with_ghost(1.0)
    p = p.copy_with(features=p.features.drop(columns=["spf"]))
    sol = MIPSolver(objective="max_weighted_features").solve(p, ONE)[0]
    assert sol.metadata["mip_objective_value"] == pytest.approx(1.0)
```

These tests read `metadata["mip_objective_value"]`, which arrives in Task 5. **For this task, add it now as a minimal stub:** in the metadata dict at `mip_solver.py:587`, set `"mip_objective_value": -float(pulp.value(model.objective))` for `max_*` objectives and `float(pulp.value(model.objective))` otherwise. Task 5 replaces the stub with `built.objective_expr`.

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_stage1.py -v`

Expected: FAIL.
- `test_zero_cost_featureless_pu_still_solves`: `assert 0 == 1`.
- The ghost tests: `KeyError: 'mip_objective_value'`.
- No-spf: `KeyError: 'spf'`.

- [ ] **Step 3: Implement**

In `mip_solver.py`:

1. Add the module-level helper after `_warn_if_large_mip`:

```python
def _core_spf(problem: ConservationProblem) -> dict[int, float]:
    """feature id -> spf; 1.0 when the column is absent (as the zone MIP's _spf_lookup)."""
    ids = problem.features["id"].values
    if "spf" not in problem.features.columns:
        return {int(f): 1.0 for f in ids}
    return {
        int(f): float(s)
        for f, s in zip(ids, problem.features["spf"].values.astype(float), strict=True)
    }
```

2. In the `max_weighted_features` branch, replace the `feat_spf_arr` / `spf_by_id` lines with `spf_by_id = _core_spf(problem)`. In the `min_penalties` branch, do the same; Task 6 rewrites that branch.

3. After the target loop, just before the no-good cuts (line 523), fix ghost `feat_met`:

```python
        if feat_met:
            # Ghost features (target > 0, no pu_vs_features rows) can never be met; v0.38
            # left feat_met unconstrained so the maximiser counted them for free.
            for fi in range(len(feat_ids)):
                fid = int(feat_ids[fi])
                if fid not in feat_groups.groups and float(feat_targets[fi]) * misslevel > 0:
                    feat_met[fid].upBound = 0
```

4. Line 573: `has_values = any(pulp.value(x[pid]) is not None for pid in pu_ids)`. Extraction at :579 already reads `None` as 0. Comment: `# A PU in no constraint and with zero cost never enters the model (value None): read as 0 (S9).`

5. Add the metadata stub described in Step 1.

- [ ] **Step 4: Run the tests, then the MIP regression**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_stage1.py tests/pymarxan/solvers/test_mip_objectives.py tests/pymarxan/solvers/test_mip_solver.py tests/pymarxan/solvers/test_stage1_golden_v038.py -v`

Expected:
- the stage1 tests pass;
- `test_mip_objectives.py` and `test_mip_solver.py` pass;
- golden: 3 passed, 1 still failing (the `shortfall=` TypeError).

- [ ] **Step 5: CHANGELOG and commit**

Under `## [Unreleased]` → `### Fixed`:

```markdown
- `MIPSolver` returned no solution when a zero-cost planning unit held no features (its
  variable never entered the model) (stage1, S9).
- `MIPSolver` `max_features` / `max_weighted_features` counted a feature with a target but no
  `pu_vs_features` rows as met for free; it is now never met. `max_weighted_features` no
  longer needs an `spf` column (defaults to 1.0).
```

```bash
ruff check src/pymarxan/solvers/mip_solver.py tests/pymarxan/solvers/test_mip_stage1.py
git add -A src/pymarxan/solvers/mip_solver.py tests/pymarxan/solvers/test_mip_stage1.py CHANGELOG.md
git commit -m "fix(stage1): MIPSolver zero-cost featureless PU, ghost feat_met, spf default

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Core `MIPSolver` build/solve split (pure refactor)

**Files:**
- Modify: `src/pymarxan/solvers/mip_solver.py`
- Test: `tests/pymarxan/solvers/test_mip_stage1.py` (append)

**Interfaces:**
- Produces:
  - `_BuiltModel`, a dataclass in `mip_solver.py`:
    - `model: pulp.LpProblem`
    - `x: dict[int, pulp.LpVariable]`
    - `pu_ids: list[int]`
    - `blm: float`
    - `objective_expr: pulp.LpAffineExpression` (natural units)
    - `probmode: int`, `has_target2: bool`, `has_sep_active: bool`
    - `lex: tuple[pulp.LpAffineExpression, pulp.LpAffineExpression] | None = None`
  - `MIPSolver._build(problem, config) -> _BuiltModel`, which emits no warnings.
  - `MIPSolver.build_model(problem, config=None) -> pulp.LpProblem`.

  Deviation from spec §3: the probmode / TARGET2 / sep flags are carried on `_BuiltModel`, not recomputed in `solve()`. This gives one source and changes no behaviour.

- [ ] **Step 1: Write the failing tests** (append to `test_mip_stage1.py`)

```python
import warnings
from pathlib import Path

import pulp

from pymarxan.io.readers import load_project
from tests.pymarxan.lp_canon import canonical_model, captured_models

SIMPLE = Path(__file__).parent.parent.parent / "data" / "simple"


def test_build_model_equals_solved_model() -> None:
    p = load_project(SIMPLE)
    built = MIPSolver().build_model(p)
    with captured_models() as models:
        MIPSolver().solve(p, ONE)
    assert isinstance(built, pulp.LpProblem)
    assert canonical_model(built) == canonical_model(models[0])


def test_build_model_refuses_what_solve_refuses() -> None:
    p = load_project(SIMPLE)
    p = p.copy_with(parameters={**p.parameters, "PROBMODE": 3})
    with pytest.raises(NotImplementedError):
        MIPSolver(mip_chance_strategy="socp").build_model(p)


def test_large_mip_warning_once_with_caller_frame() -> None:
    p = load_project(SIMPLE)
    for call in (MIPSolver(warn_above_pu=1).solve, MIPSolver(warn_above_pu=1).build_model):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            call(p)
        large = [x for x in w if "warn_above_pu" in str(x.message)]
        assert len(large) == 1
        assert large[0].filename == __file__
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_stage1.py -k "build_model or large_mip" -v`

Expected: `AttributeError: 'MIPSolver' object has no attribute 'build_model'`.

- [ ] **Step 3: Implement the split**

1. Add `from dataclasses import dataclass` and define `_BuiltModel` as in Interfaces, with `lex` defaulting to `None`.

2. Rename the current body: `solve()` lines 165–542, from the `probmode` gating through the no-good cuts, become `def _build(self, problem: ConservationProblem, config: SolverConfig) -> _BuiltModel:`.
   - **Remove** the `_warn_if_large_mip(...)` call from the moved block.
   - Declare `objective_expr: pulp.LpAffineExpression | None = None` at the top of `_build`. The builder paths added in Tasks 5 and 6 set it.
   - End `_build` with:

```python
        if objective_expr is None:  # natural units: min_set / max_* / absolute minimax
            objective_expr = model.objective
            if self.objective in ("max_features", "max_weighted_features"):
                objective_expr = -model.objective
        return _BuiltModel(
            model=model, x=x, pu_ids=pu_ids, blm=blm, objective_expr=objective_expr,
            probmode=probmode, has_target2=has_target2, has_sep_active=has_sep_active,
        )
```

3. New public methods:

```python
    def build_model(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> pulp.LpProblem:
        """The PuLP model ``solve`` would hand to the backend (for ``write_problem``)."""
        if config is None:
            config = SolverConfig()
        _warn_if_large_mip(problem.n_planning_units, self.warn_above_pu)
        return self._build(problem, config).model

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig()
        # S3b: heads-up before building the model when the MILP is likely too large.
        _warn_if_large_mip(problem.n_planning_units, self.warn_above_pu)
        built = self._build(problem, config)
        model, x, pu_ids = built.model, built.x, built.pu_ids
        # ... the existing lines 544-600 (backend, model.solve, status check, extraction),
        # reading probmode / has_target2 / has_sep_active / blm from `built`, and
        # "mip_objective_value": float(pulp.value(built.objective_expr) or 0.0)
        # replacing the Task 3 stub.
```

`_warn_if_large_mip` keeps `stacklevel=3`: `warn` ← `_warn_if_large_mip` ← `solve`/`build_model` ← caller.

- [ ] **Step 4: Run the tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/ tests/test_examples.py -q -x`

Expected:
- everything passes except the one known golden failure (`shortfall=`);
- `test_core_min_set_model_unchanged` passes, so the refactor is byte-identical;
- `examples/validate_marxan_parity.py` gives 35.0.

- [ ] **Step 5: Commit**

```bash
ruff check src/pymarxan/solvers/mip_solver.py tests/pymarxan/solvers/test_mip_stage1.py
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/solvers/mip_solver.py
git add src/pymarxan/solvers/mip_solver.py tests/pymarxan/solvers/test_mip_stage1.py
git commit -m "refactor(stage1): MIPSolver _build/build_model/solve split (no behaviour change)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Core objectives: `min_shortfall`, `shortfall=`, relative `min_largest_shortfall`, alias

**Files:**
- Modify: `src/pymarxan/solvers/mip_solver.py`
- Test: `tests/pymarxan/solvers/test_mip_stage1.py` (append)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes from Task 2: `BUILDERS`, `ObjectiveContext`, `TargetTerm`, `resolve_objective`, `validate_shortfall`, `set_objective`.
- Consumes from Task 1: `make_relabs_problem` (import from `tests.pymarxan.solvers.test_stage1_golden_v038`).
- Produces:
  - `MIPSolver(objective=..., shortfall="relative", ...)`, which stores `self.objective` (canonical) and `self.shortfall`.
  - `_core_amount_exprs(problem, x, prob_map_m2) -> dict[int, pulp.LpAffineExpression]`: features with rows only, the same expressions as the target loop.
  - `_core_target_terms(problem, x, prob_map_m2) -> list[TargetTerm]`: one per feature with target > 0, with key `f"f{fid}"`, `required = target × MISSLEVEL` and weight from `_core_spf`.
  - `MIP_OBJECTIVES = ("min_set", "max_features", "max_weighted_features", "min_shortfall", "min_largest_shortfall", "min_penalties")`.

- [ ] **Step 1: Write the failing tests** (append)

```python
from tests.pymarxan.solvers.test_stage1_golden_v038 import make_relabs_problem


def test_relative_is_default_and_differs_from_absolute() -> None:
    rel = MIPSolver(objective="min_largest_shortfall").solve(make_relabs_problem(), ONE)[0]
    ab = MIPSolver(objective="min_largest_shortfall", shortfall="absolute").solve(
        make_relabs_problem(), ONE,
    )[0]
    assert rel.selected.tolist() == [False, False, True]
    assert rel.metadata["mip_objective_value"] == pytest.approx(0.7)
    assert rel.metadata["shortfall"] == "relative"
    assert ab.selected.tolist() == [True, False, False]
    assert ab.metadata["mip_objective_value"] == pytest.approx(40.0)


def test_min_shortfall_values() -> None:
    p = make_relabs_problem()
    rel = MIPSolver(objective="min_shortfall").solve(p, ONE)[0]
    ab = MIPSolver(objective="min_shortfall", shortfall="absolute").solve(p, ONE)[0]
    assert rel.metadata["mip_objective_value"] == pytest.approx(1.2)  # PU2/PU3 tie
    assert ab.selected.tolist() == [True, False, False]
    assert ab.metadata["mip_objective_value"] == pytest.approx(50.0)


def test_min_shortfall_requires_budget() -> None:
    p = make_relabs_problem()
    p = p.copy_with(parameters={"BLM": 0.0})
    with pytest.raises(ValueError, match="min_shortfall.*COSTBUDGET"):
        MIPSolver(objective="min_shortfall").solve(p, ONE)


def test_relative_counts_ghost_feature() -> None:
    sol = MIPSolver(objective="min_largest_shortfall").solve(_with_ghost(1.0), ONE)[0]
    assert sol.metadata["mip_objective_value"] == pytest.approx(1.0)


@pytest.mark.parametrize("misslevel", [0.0, 0.5])
def test_relative_misslevel_divisor(misslevel: float) -> None:
    p = make_relabs_problem()
    p = p.copy_with(parameters={**p.parameters, "MISSLEVEL": misslevel})
    sol = MIPSolver(objective="min_largest_shortfall").solve(p, ONE)[0]
    if misslevel == 0.0:
        assert sol.metadata["mip_objective_value"] == pytest.approx(0.0)  # every R <= 0 skipped
    else:
        # R_a = 50, R_b = 5: PU3 -> a 20/50 = 0.4, b 0 -> 0.4; PU1 -> b 5/5 = 1.0
        assert sol.selected.tolist() == [False, False, True]
        assert sol.metadata["mip_objective_value"] == pytest.approx(0.4)


def test_alias_and_bad_names() -> None:
    assert MIPSolver(objective="max_n_targets_met").objective == "max_features"
    with pytest.raises(ValueError, match="shortfall"):
        MIPSolver(shortfall="percent")
    with pytest.raises(ValueError, match="objective"):
        MIPSolver(objective="max_utility")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_stage1.py -v`

Expected: `TypeError: ... unexpected keyword argument 'shortfall'` and similar.

- [ ] **Step 3: Implement**

1. Constructor: add the `shortfall: str = "relative"` keyword. Replace the `objective` validation with:
   `self.objective = resolve_objective(objective, MIP_OBJECTIVES)` and `self.shortfall = validate_shortfall(shortfall)`.

2. Extract `_core_amount_exprs` from the target loop. The loop body then reads `amount_expr = amount_exprs[fid]`. Keep the `if fid in feat_groups.groups` structure, so the min_set / max / absolute expressions stay byte-identical.

```python
def _core_amount_exprs(
    problem: ConservationProblem,
    x: dict[int, pulp.LpVariable],
    prob_map_m2: dict[int, float],
) -> dict[int, pulp.LpAffineExpression]:
    """feature id -> Σ amount·x (PROBMODE 2: × (1 − p)) for features with rows."""
    out: dict[int, pulp.LpAffineExpression] = {}
    for fid, feat_data in problem.pu_vs_features.groupby("species"):
        pus = feat_data["pu"].values
        amounts = feat_data["amount"].values
        if prob_map_m2:
            out[int(fid)] = pulp.lpSum(
                float(amt) * (1.0 - prob_map_m2.get(int(pu), 0.0)) * x[int(pu)]
                for pu, amt in zip(pus, amounts)
            )
        else:
            out[int(fid)] = pulp.lpSum(float(amt) * x[int(pu)] for pu, amt in zip(pus, amounts))
    return out


def _core_target_terms(
    problem: ConservationProblem,
    x: dict[int, pulp.LpVariable],
    prob_map_m2: dict[int, float],
) -> list[TargetTerm]:
    """One term per feature with target > 0; ghosts get lpSum([]) (spec §1)."""
    exprs = _core_amount_exprs(problem, x, prob_map_m2)
    spf = _core_spf(problem)
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    terms: list[TargetTerm] = []
    for fid_raw, target in zip(
        problem.features["id"].values, problem.features["target"].values, strict=True,
    ):
        fid, t = int(fid_raw), float(target)
        if t <= 0:
            continue
        terms.append(TargetTerm(
            key=f"f{fid}", kind="feature", achieved=exprs.get(fid, pulp.lpSum([])),
            required=t * misslevel, weight=spf[fid],
        ))
    return terms
```

   The `groupby("species")` order is sorted by key. That does not matter for the expressions, whose term order is the row order within each group, as today.

3. In `_build`, move the `prob_map_m2` construction above the objective selection. Then route the new and relative objectives through the builders:

```python
        builder_path = self.objective == "min_shortfall" or (
            self.objective == "min_largest_shortfall" and self.shortfall == "relative"
        )
        if builder_path:
            cost_budget = problem.parameters.get("COSTBUDGET")
            if cost_budget is None:
                raise ValueError(
                    f"objective={self.objective!r} requires a COSTBUDGET parameter "
                    "(problem.parameters['COSTBUDGET'] = ...); without a budget every "
                    "shortfall is 0 by selecting all planning units."
                )
            ctx = ObjectiveContext(model, cost_expr, _core_target_terms(problem, x, prob_map_m2))
            term = BUILDERS[self.objective](ctx, self.shortfall)
            set_objective(model, term)
            model += cost_expr <= float(cost_budget), "cost_budget"
            objective_expr = term.expr
```

   The existing `min_largest_shortfall` branch becomes `elif self.objective == "min_largest_shortfall":` and is reached only for absolute. In the target loop, guard the per-feature constraints with `if not builder_path:`. The min_set / max / absolute / min_penalties arms are unchanged.

4. Metadata: add `"shortfall": self.shortfall`. `"mip_objective_value"` comes from `built.objective_expr`, and `_BuiltModel` gets `objective_expr = term.expr` on the builder path.

5. Docstrings: class docstring objective list; `min_largest_shortfall`: "relative (shortfall / (target × MISSLEVEL)) by default, as prioritizr; `shortfall='absolute'` is the v0.38 behaviour". Note that a ghost feature pins the relative minimax at 1.0 (`presolve_check` P5 flags it).

- [ ] **Step 4: Run the tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/ tests/test_tutorial_examples.py -q`

Expected: all pass, **including `test_core_absolute_min_largest_shortfall_unchanged`, which is now green**.

- [ ] **Step 5: CHANGELOG and commit**

```markdown
### Added
- `MIPSolver(objective="min_shortfall")` — minimise the SPF-weighted shortfall under
  `COSTBUDGET` (prioritizr `add_min_shortfall_objective`). `objective="max_n_targets_met"` is
  accepted as an alias of `max_features` (prioritizr ≥ 9 name). (stage1)
- `MIPSolver(shortfall="relative" | "absolute")`.
### Changed
- `MIPSolver(objective="min_largest_shortfall")` now minimises the largest **relative**
  shortfall (shortfall / (target × MISSLEVEL), as prioritizr) and counts features with no
  occurrences. Pass `shortfall="absolute"` for the v0.38 behaviour. (stage1)
```

```bash
ruff check src/pymarxan/solvers/mip_solver.py tests/pymarxan/solvers/test_mip_stage1.py
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/solvers/mip_solver.py
git add src/pymarxan/solvers/mip_solver.py tests/pymarxan/solvers/test_mip_stage1.py CHANGELOG.md
git commit -m "feat(stage1): MIPSolver min_shortfall, relative min_largest_shortfall, shortfall= kw

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Core `min_penalties` as an exact two-pass solve (S2)

**Files:**
- Modify: `src/pymarxan/solvers/mip_solver.py`
- Test: `tests/pymarxan/solvers/test_mip_stage1.py` (append)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `solve_lexicographic`, `ObjectiveContext`, `_core_target_terms` (Tasks 2 and 5).
- Produces: `_BuiltModel.lex = (primary, cost_expr)` for `min_penalties`. `build_model` exports the pass-1 model (objective = primary).

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_min_penalties_lexicographic_on_fractional_data() -> None:
    # v0.38: M = Σcost + 1 = 102; 0.01·102 < 100 so it left PU1 out (target unmet).
    sol = MIPSolver(objective="min_penalties").solve(_two_pu(1.0), ONE)[0]
    assert sol.selected.tolist() == [True, False]
    assert sol.cost == pytest.approx(100.0)
    assert sol.metadata["mip_objective_value"] == pytest.approx(0.0)  # SPF·shortfall


def test_min_penalties_cost_is_secondary() -> None:
    # Both PUs meet the target; pass 2 must take the cheaper one.
    pu = pd.DataFrame({"id": [1, 2], "cost": [5.0, 3.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [0.5]})
    pvf = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]})
    p = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pvf)
    sol = MIPSolver(objective="min_penalties").solve(p, ONE)[0]
    assert sol.selected.tolist() == [False, True]


def test_min_penalties_build_model_is_pass1() -> None:
    m = MIPSolver(objective="min_penalties").build_model(_two_pu(1.0))
    assert "lex_primary_cap" not in m.constraints
    assert {v.name for v, _ in m.objective.items()} == {"s_f1"}
```

The v0.38 integral-data guard is `test_core_min_penalties_integral_selection_unchanged` in the golden file.

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_stage1.py -k min_penalties -v`

Expected: FAIL. The first test gets `[False, False]` (the v0.38 big-M result); the build_model test fails on the variable names.

- [ ] **Step 3: Implement**

Replace the `min_penalties` branch in `_build`:

```python
        elif self.objective == "min_penalties":
            # Exact lexicographic (S2): pass 1 min Σ SPF·shortfall, pass 2 min cost at that
            # optimum (solve_lexicographic). v0.38's big-M (Σ cost + 1) only dominated when
            # SPF·Δshortfall >= 1, so fractional data could trade targets for cost. Not
            # prioritizr's add_min_penalties_objective (penalty-only, hard targets).
            ctx = ObjectiveContext(model, cost_expr, _core_target_terms(problem, x, prob_map_m2))
            primary = pulp.lpSum(t.weight * ctx.slack(t) for t in ctx.terms)
            model.setObjective(primary)  # build_model exports the pass-1 model
            lex = (primary, cost_expr)
            objective_expr = primary
```

Then:
- Remove the `elif self.objective == "min_penalties":` arm from the target loop; the builder's slack constraints replace it. The `builder_path` guard becomes `builder_path or self.objective == "min_penalties"`.
- Pass `lex=lex` to `_BuiltModel`, with `lex = None` initialised at the top.
- In `solve()`:

```python
        if built.lex is not None:
            status = solve_lexicographic(model, built.lex[0], built.lex[1], solver)
        else:
            model.solve(solver)
            status = model.status
```

and use `status` in place of `model.status` in the status check and the metadata.

- [ ] **Step 4: Run the tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/ tests/test_tutorial_examples.py -q`

Expected: all pass, including `test_core_min_penalties_integral_selection_unchanged` and the existing `test_min_penalties_*` in `test_mip_objectives.py`.

- [ ] **Step 5: CHANGELOG and commit**

Under `### Fixed`:

```markdown
- `MIPSolver(objective="min_penalties")` could trade target shortfall for cost on fractional
  data (its big-M only dominated when SPF·Δshortfall ≥ 1); it is now an exact two-pass
  lexicographic solve. `build_model` / `write_problem` export the pass-1 model. (stage1, S2)
```

```bash
ruff check src/pymarxan/solvers/mip_solver.py tests/pymarxan/solvers/test_mip_stage1.py
git add src/pymarxan/solvers/mip_solver.py tests/pymarxan/solvers/test_mip_stage1.py CHANGELOG.md
git commit -m "fix(stage1): exact two-pass lexicographic min_penalties

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `ZoneMIPSolver` split and budgeted objectives

**Files:**
- Modify: `src/pymarxan/zones/mip_solver.py`
- Modify: `src/pymarxan/zones/diagnostics.py`
- Test: `tests/pymarxan/zones/test_zone_mip_objectives.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: Task 2 (everything), and Task 5's core `MIPSolver` for the cross-solver oracle.
- Produces:
  - `ZoneMIPSolver(*, objective="min_set", shortfall="relative", mip_clump_strategy="drop", mip_sep_strategy="drop", mip_backend="auto")`. The `presolve_check` keyword comes in Task 10.
  - `ZONE_MIP_OBJECTIVES = ("min_set", "max_features", "max_weighted_features", "min_shortfall", "min_largest_shortfall")`
  - `ZoneMIPSolver.build_model(problem, config=None) -> pulp.LpProblem`
  - `_BuiltZoneModel(model, x, pu_ids, zone_ids, blm, objective_expr)`
  - `_zone_target_terms(problem, x, zone_ids) -> list[TargetTerm]`
  - `_overall_achieved_exprs(problem, x, zone_ids) -> dict[int, pulp.LpAffineExpression]`
  - In `zones/diagnostics.py`: `_zone_cost_lookup(problem) -> dict[tuple[int, int], float]` and `min_forced_cost(problem: ZonalProblem) -> float`.

- [ ] **Step 1: Write the failing tests**

`tests/pymarxan/zones/test_zone_mip_objectives.py`:

```python
"""ZoneMIPSolver budgeted objectives (spec §1, Testing 1-4, 6, 7; Review Focus 1, 2, 5)."""
from __future__ import annotations

import itertools
import warnings

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.zones import ZonalProblem, ZoneMIPSolver
from pymarxan.zones.diagnostics import ZoneMIPNoSolutionWarning, min_forced_cost
from tests.pymarxan.zones.marzone_anchor import (
    AMOUNT,
    CONTRIB,
    OVERALL_TARGET,
    ZONE_COSTS,
    ZONE_TARGET,
    make_anchor_problem,
)

ONE = SolverConfig(num_solutions=1)
BUDGETED = ("max_features", "max_weighted_features", "min_shortfall", "min_largest_shortfall")
BUDGETS = (0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 13)


def _with_budget(p: ZonalProblem, b: float) -> ZonalProblem:
    return p.copy_with(parameters={**p.parameters, "COSTBUDGET": b})


def _anchor_terms(a: tuple[int, ...]) -> list[tuple[float, float]]:
    """(achieved, required) for the anchor's two terms: overall, then zone (2, 1)."""
    overall = sum(AMOUNT * CONTRIB[z] for z in a if z)
    zone2 = sum(AMOUNT for z in a if z == 2)
    return [(overall, OVERALL_TARGET), (zone2, ZONE_TARGET)]


def _oracle(a: tuple[int, ...], objective: str, spf: float) -> float:
    terms = _anchor_terms(a)
    short = [max(0.0, r - got) / r for got, r in terms]
    met = [got >= r - 1e-9 for got, r in terms]
    if objective == "max_features":
        return float(sum(met))
    if objective == "max_weighted_features":
        return spf * sum(met)
    if objective == "min_shortfall":
        return spf * sum(short)
    return max(short)


def _cost(a: tuple[int, ...]) -> float:
    return sum(ZONE_COSTS[z][i] for i, z in enumerate(a) if z)


@pytest.mark.parametrize("objective", BUDGETED)
@pytest.mark.parametrize("spf", [1.0, 3.0])
@pytest.mark.parametrize("budget", BUDGETS)
def test_anchor_matches_enumeration(objective: str, spf: float, budget: float) -> None:
    feasible = [a for a in itertools.product([0, 1, 2], repeat=3) if _cost(a) <= budget + 1e-9]
    values = [_oracle(a, objective, spf) for a in feasible]
    best = max(values) if objective.startswith("max") else min(values)
    p = _with_budget(make_anchor_problem(spf=spf), budget)
    sol = ZoneMIPSolver(objective=objective).solve(p, ONE)[0]
    assert sol.metadata["mip_objective_value"] == pytest.approx(best, abs=1e-6)
    assert sol.metadata["objective"] == objective
    assert _cost(tuple(int(z) for z in sol.zone_assignment)) <= budget + 1e-9


def test_anchor_unique_argmins() -> None:
    def run(obj: str, b: float, spf: float = 1.0) -> tuple[tuple[int, ...], float]:
        s = ZoneMIPSolver(objective=obj).solve(
            _with_budget(make_anchor_problem(spf=spf), b), ONE,
        )[0]
        return tuple(int(z) for z in s.zone_assignment), s.metadata["mip_objective_value"]

    assert run("min_shortfall", 3) == ((2, 2, 2), pytest.approx(0.2))
    assert run("min_shortfall", 3, spf=3.0) == ((2, 2, 2), pytest.approx(0.6))
    assert run("min_largest_shortfall", 3) == ((2, 2, 2), pytest.approx(0.2))
    for obj in BUDGETED:
        assert run(obj, 7)[0] == (1, 2, 2)


def test_each_target_is_its_own_term() -> None:
    # D1: with B in [1, 6] the zone target (2,1) is met but the overall target is not.
    for b in range(1, 7):
        s = ZoneMIPSolver(objective="max_features").solve(
            _with_budget(make_anchor_problem(), b), ONE,
        )[0]
        assert s.metadata["mip_objective_value"] == pytest.approx(1.0)


def _two_feature_anchor() -> ZonalProblem:
    """Testing 3: add feature 2 (target 10, spf 5), amount 10 in PU3 only, contrib z1=1, z2=0."""
    p = make_anchor_problem()
    feats = pd.concat([
        p.features,
        pd.DataFrame({"id": [2], "name": ["f2"], "target": [10.0], "spf": [5.0]}),
    ], ignore_index=True)
    pvf = pd.concat([
        p.pu_vs_features, pd.DataFrame({"species": [2], "pu": [3], "amount": [10.0]}),
    ], ignore_index=True)
    contrib = pd.concat([
        p.zone_contributions,
        pd.DataFrame({"feature": [2, 2], "zone": [1, 2], "contribution": [1.0, 0.0]}),
    ], ignore_index=True)
    return _with_budget(
        p.copy_with(features=feats, pu_vs_features=pvf, zone_contributions=contrib), 7,
    )


def test_weighted_differs_from_count() -> None:
    p = _two_feature_anchor()
    cnt = ZoneMIPSolver(objective="max_features").solve(p, ONE)[0]
    wtd = ZoneMIPSolver(objective="max_weighted_features").solve(p, ONE)[0]
    assert tuple(cnt.zone_assignment) == (1, 2, 2)
    assert cnt.metadata["mip_objective_value"] == pytest.approx(2.0)
    assert tuple(wtd.zone_assignment) == (0, 0, 1)
    assert wtd.metadata["mip_objective_value"] == pytest.approx(5.0)


def test_budget_required() -> None:
    with pytest.raises(ValueError, match="min_shortfall.*COSTBUDGET"):
        ZoneMIPSolver(objective="min_shortfall").solve(make_anchor_problem(), ONE)


def test_boundary_ignored_warning_once() -> None:
    p = _with_budget(make_anchor_problem(blm=2.0), 7)
    p = p.copy_with(boundary=pd.DataFrame({"id1": [1], "id2": [2], "boundary": [1.0]}))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ZoneMIPSolver(objective="min_shortfall").solve(p, ONE)
    hits = [x for x in w if "ignores boundary penalties" in str(x.message)]
    assert len(hits) == 1
    assert hits[0].filename == __file__


def test_budget_floor_message() -> None:
    p = make_anchor_problem()
    p = p.copy_with(planning_units=p.planning_units.assign(status=[2, 0, 0]))  # PU1 -> zone 1
    p = _with_budget(p, 4.0)  # forced cost 5 > 4
    assert min_forced_cost(p) == pytest.approx(5.0)
    with pytest.warns(ZoneMIPNoSolutionWarning, match=r"at least 5 > COSTBUDGET 4"):
        assert ZoneMIPSolver(objective="max_features").solve(p, ONE) == []


def _core_problem() -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [2.0, 3.0, 4.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2, 3], "name": ["a", "b", "g"], "target": [10.0, 6.0, 4.0],
        "spf": [1.0, 2.0, 1.0],
    })
    pvf = pd.DataFrame({
        "species": [1, 1, 2, 2], "pu": [1, 2, 2, 3], "amount": [6.0, 5.0, 4.0, 3.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf,
        parameters={"COSTBUDGET": 5.0, "BLM": 0.0},
    )


def _as_one_zone(p: ConservationProblem) -> ZonalProblem:
    pu = p.planning_units
    return ZonalProblem(
        planning_units=pu, features=p.features, pu_vs_features=p.pu_vs_features,
        boundary=None, parameters=dict(p.parameters),
        zones=pd.DataFrame({"id": [1], "name": ["z"]}),
        zone_costs=pd.DataFrame({"pu": pu["id"], "zone": 1, "cost": pu["cost"]}),
        zone_contributions=pd.DataFrame({
            "feature": p.features["id"], "zone": 1, "contribution": 1.0,
        }),
        zone_targets=None, zone_boundary_costs=None,
    )


@pytest.mark.parametrize("objective", ["max_features", "min_shortfall", "min_largest_shortfall"])
@pytest.mark.parametrize("ghost", [False, True])
@pytest.mark.parametrize("misslevel", [1.0, 0.9])
def test_cross_solver_oracle(objective: str, ghost: bool, misslevel: float) -> None:
    p = _core_problem()
    if not ghost:
        p = p.copy_with(features=p.features[p.features["id"] != 3].reset_index(drop=True))
    p = p.copy_with(parameters={**p.parameters, "MISSLEVEL": misslevel})
    core = MIPSolver(objective=objective).solve(p, ONE)[0]
    zone = ZoneMIPSolver(objective=objective).solve(_as_one_zone(p), ONE)[0]
    assert zone.metadata["mip_objective_value"] == pytest.approx(
        core.metadata["mip_objective_value"], abs=1e-6,
    )


def test_budgeted_missing_zone_cost_is_free() -> None:
    p = make_anchor_problem()
    zc = p.zone_costs
    p = _with_budget(p.copy_with(zone_costs=zc[~((zc["pu"] == 1) & (zc["zone"] == 1))]), 2)
    s = ZoneMIPSolver(objective="max_features").solve(p, ONE)[0]
    # PU1 in zone 1 is free: (1, 2, 2) costs 2 and meets both targets.
    assert tuple(s.zone_assignment) == (1, 2, 2)
    assert s.metadata["mip_objective_value"] == pytest.approx(2.0)


def test_budgeted_no_terms() -> None:
    p = make_anchor_problem()
    p = _with_budget(
        p.copy_with(features=p.features.assign(target=0.0), zone_targets=None), 3,
    )
    for obj in ("max_features", "min_shortfall"):
        s = ZoneMIPSolver(objective=obj).solve(p, ONE)
        assert len(s) == 1
        assert s[0].metadata["mip_objective_value"] == pytest.approx(0.0)


def test_reused_instance_no_state_leak() -> None:
    solver = ZoneMIPSolver(objective="min_shortfall")
    p1 = _with_budget(make_anchor_problem(), 3)
    p2 = _with_budget(make_anchor_problem(spf=3.0), 7)
    solver.build_model(p1)
    a = solver.solve(p2, ONE)[0].metadata["mip_objective_value"]
    b = solver.solve(p1, ONE)[0].metadata["mip_objective_value"]
    assert a == pytest.approx(
        ZoneMIPSolver(objective="min_shortfall").solve(p2, ONE)[0].metadata["mip_objective_value"]
    )
    assert b == pytest.approx(0.2)


def test_min_set_metadata_and_alias() -> None:
    s = ZoneMIPSolver().solve(make_anchor_problem(), ONE)[0]
    assert s.metadata["objective"] == "min_set"
    assert s.metadata["mip_objective_value"] == pytest.approx(7.0)
    assert ZoneMIPSolver(objective="max_n_targets_met").objective == "max_features"
    with pytest.raises(ValueError, match="objective"):
        ZoneMIPSolver(objective="min_penalties")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_mip_objectives.py -q`

Expected: `TypeError: ... unexpected keyword argument 'objective'`, and an ImportError for `min_forced_cost`.

- [ ] **Step 3: Implement the diagnostics helpers** (`zones/diagnostics.py`)

```python
def _zone_cost_lookup(problem: ZonalProblem) -> dict[tuple[int, int], float]:
    """(pu, zone) -> cost; a missing pair costs 0 (MarZone; prioritizr NA means forbidden)."""
    zc = problem.zone_costs
    return {
        (int(p), int(z)): float(c)
        for p, z, c in zip(zc["pu"].values, zc["zone"].values, zc["cost"].values, strict=True)
    }


def min_forced_cost(problem: ZonalProblem) -> float:
    """Least total zone cost any rule-respecting assignment must pay.

    Σ over must-assign PUs of the cheapest allowed zone cost, plus min(0, cheapest) for PUs
    that may stay unassigned (a negative cost lowers the floor). COSTBUDGET below this is
    infeasible for every budgeted objective.
    """
    lookup = _zone_cost_lookup(problem)
    total = 0.0
    for pid, rule in zip(
        (int(p) for p in problem.planning_units["id"].values),
        problem.pu_zone_rules(),
        strict=True,
    ):
        costs = [lookup.get((pid, z), 0.0) for z in rule.allowed]
        if not costs:
            continue
        cheapest = min(costs)
        total += min(0.0, cheapest) if rule.may_be_unassigned else cheapest
    return total
```

- [ ] **Step 4: Implement the zone solver changes** (`zones/mip_solver.py`)

1. Imports:
   - `from dataclasses import dataclass`
   - from `pymarxan.solvers._mip_objectives`: `BUILDERS`, `NO_SOLUTION_STATUSES`, `ObjectiveContext`, `TargetTerm`, `resolve_objective`, `set_objective`, `validate_shortfall`
   - `_zone_cost_lookup` and `min_forced_cost` from diagnostics.

2. Module constant `ZONE_MIP_OBJECTIVES` (see Interfaces) and the dataclass:

```python
@dataclass
class _BuiltZoneModel:
    model: pulp.LpProblem
    x: dict[tuple[int, int], pulp.LpVariable]
    pu_ids: list[int]
    zone_ids: list[int]
    blm: float
    objective_expr: pulp.LpAffineExpression  # natural units (count for max_*)
```

3. Constructor: add `objective: str = "min_set"` and `shortfall: str = "relative"` before the strategy keywords. Store `self.objective = resolve_objective(objective, ZONE_MIP_OBJECTIVES)` and `self.shortfall = validate_shortfall(shortfall)`.

4. Overall expressions. Refactor `_add_overall_target_constraints` into:
   - `_overall_achieved_exprs(problem, x, zone_ids) -> dict[int, LpAffineExpression]`, which keeps the loop body and the `t <= 0: continue` skip, and returns `{fid: pulp.lpSum(terms)}`;
   - `_add_overall_target_constraints`, which iterates that dict in the same order, adding `exprs[fid] >= t * misslevel, f"overall_target_{fid}"`.

   Keep `t` from the features frame, so the constraint order and content are identical; the golden checks this.

5. Terms:

```python
def _zone_target_terms(
    problem: ZonalProblem,
    x: dict[tuple[int, int], pulp.LpVariable],
    zone_ids: list[int],
) -> list[TargetTerm]:
    """One term per (zone, feature) target and one per feature with overall target > 0 (D1)."""
    spf = _spf_lookup(problem)
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    zone_exprs = _zone_achieved_exprs(problem, x)
    terms = [
        TargetTerm(f"z{zid}_f{fid}", "zone", zone_exprs[(zid, fid)], target * misslevel,
                   spf.get(fid, 1.0))
        for (zid, fid), target in problem.zone_target_lookup().items()
    ]
    overall = _overall_achieved_exprs(problem, x, zone_ids)
    targets = dict(zip(
        (int(f) for f in problem.features["id"].values),
        problem.features["target"].values.astype(float),
        strict=True,
    ))
    terms += [
        TargetTerm(f"o_f{fid}", "overall", expr, targets[fid] * misslevel, spf.get(fid, 1.0))
        for fid, expr in overall.items()
    ]
    return terms
```

6. Split the work into three methods:

```python
    def _check(self, problem: ConservationProblem) -> ZonalProblem:
        if not isinstance(problem, ZonalProblem):
            raise TypeError(
                f"ZoneMIPSolver requires a ZonalProblem, got {type(problem).__name__}"
            )
        from pymarxan.solvers.separation import raise_if_separation_active
        raise_if_separation_active(problem, "ZoneMIPSolver")
        return problem

    def _warn_boundary_ignored(self, problem: ZonalProblem) -> None:
        if self.objective == "min_set":
            return
        blm = float(problem.parameters.get("BLM", 0.0))
        zbc = problem.zone_boundary_costs
        has_zbc = zbc is not None and bool((zbc["cost"].astype(float) != 0).any())
        if (blm > 0 and problem.boundary is not None) or has_zbc:
            warnings.warn(
                f"ZoneMIPSolver: objective {self.objective!r} ignores boundary penalties "
                "(BLM / zone boundary costs); prioritizr applies them — planned follow-up",
                UserWarning,
                stacklevel=3,
            )

    def build_model(
        self, problem: ZonalProblem, config: SolverConfig | None = None
    ) -> pulp.LpProblem:
        problem = self._check(problem)
        warn_zone_input_issues(problem, "ZoneMIPSolver")
        self._warn_boundary_ignored(problem)
        return self._build(problem, config or SolverConfig()).model
```

   `solve()` starts with the same three preamble lines (`_check`, `warn_zone_input_issues`, `_warn_boundary_ignored`), then `built = self._build(problem, config)`, then the existing solve, status and extraction code reading from `built`.
   - The status check uses `NO_SOLUTION_STATUSES`.
   - On no solution it calls `_warn_no_solution(problem, model.status, self.objective)`.
   - Metadata gains `"objective": self.objective`, `"shortfall": self.shortfall` and `"mip_objective_value": float(pulp.value(built.objective_expr) or 0.0)`.

7. `_build`: move lines 113–176 in.
   - **min_set** keeps them verbatim (ending with the two `_add_*_target_constraints`), with `objective_expr = model.objective`.
   - The **assignment block** (lines 120–134) is shared by both paths, unchanged.
   - **Budgeted** objectives use this instead of lines 136–176:

```python
        budget = problem.parameters.get("COSTBUDGET")
        if budget is None:
            raise ValueError(
                f"objective={self.objective!r} requires a COSTBUDGET parameter "
                "(problem.parameters['COSTBUDGET'] = ...); it caps the total zone cost."
            )
        zc_lookup = _zone_cost_lookup(problem)
        cost_expr = pulp.lpSum(
            zc_lookup.get((pid, zid), 0.0) * x[(pid, zid)]
            for pid in pu_ids for zid in zone_ids
        )
        ctx = ObjectiveContext(model, cost_expr, _zone_target_terms(problem, x, zone_ids))
        term = BUILDERS[self.objective](ctx, self.shortfall)
        set_objective(model, term)
        model += cost_expr <= float(budget), "cost_budget"
        objective_expr = term.expr
```

   The min_set path also switches its inline `zc_lookup` loop (lines 139–144) to `_zone_cost_lookup(problem)`. The dict is the same, so the canonical model is unchanged.

8. `_warn_no_solution(problem, status, objective)`: add the budget branch first.

```python
    if objective != "min_set" and status == pulp.constants.LpStatusInfeasible:
        floor = min_forced_cost(problem)
        budget = float(problem.parameters.get("COSTBUDGET", 0.0))
        if floor > budget:
            msg = (
                f"ZoneMIPSolver: infeasible. The planning units that must be assigned cost "
                f"at least {_fmt(floor)} > COSTBUDGET {_fmt(budget)}."
            )
        else:
            msg = f"ZoneMIPSolver: objective {objective!r} infeasible (status {name})."
        warnings.warn(msg, ZoneMIPNoSolutionWarning, stacklevel=3)
        return
```

   Import `_fmt` from diagnostics; it is already defined there. Otherwise the existing body is unchanged.

9. Class docstring: list the objectives, state the relative shortfall, say budgeted objectives drop hard targets and boundary terms, and give the prioritizr name mapping:
   - `max_features` = `max_n_targets_met`;
   - `max_weighted_features` = `max_n_targets_met` + `add_feature_weights`, which is **not** `max_wtd_sum`.

- [ ] **Step 5: Run the tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/ tests/pymarxan/solvers/test_stage1_golden_v038.py tests/pymarxan/analysis -q`

Expected: all pass, including `test_zone_min_set_models_unchanged`.

- [ ] **Step 6: CHANGELOG and commit**

```markdown
### Added
- `ZoneMIPSolver(objective=...)`: `max_features` (alias `max_n_targets_met`),
  `max_weighted_features`, `min_shortfall`, `min_largest_shortfall` under `COSTBUDGET` — each
  zone target and each overall target is its own term (prioritizr zones); relative shortfall
  by default (`shortfall="absolute"` available). Budgeted objectives drop hard targets and
  boundary penalties (one warning when BLM / zone boundary costs are set).
  `Solution.metadata["mip_objective_value"]` reports the objective. `ZoneMIPSolver.build_model`.
  (stage1)
```

```bash
ruff check src/pymarxan/zones/ tests/pymarxan/zones/test_zone_mip_objectives.py
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/zones/mip_solver.py src/pymarxan/zones/diagnostics.py
git add src/pymarxan/zones/mip_solver.py src/pymarxan/zones/diagnostics.py tests/pymarxan/zones/test_zone_mip_objectives.py CHANGELOG.md
git commit -m "feat(stage1): ZoneMIPSolver budgeted objectives, build_model, budget-floor message

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `MANDATORYALLOCATION`

**Files:**
- Modify: `src/pymarxan/zones/model.py` (`pu_zone_rules` final branch, `_pu_lock_issue_list` head, `validate()`, new accessor)
- Test: `tests/pymarxan/zones/test_mandatory_allocation.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces: `ZonalProblem.mandatory_allocation() -> bool` and the module helper `_mandatory_value(raw: object) -> int | None`.
- Consumes: `min_forced_cost` (Task 7).

- [ ] **Step 1: Write the failing tests**

`tests/pymarxan/zones/test_mandatory_allocation.py`:

```python
"""MANDATORYALLOCATION (spec §2, Testing 8; Review Focus 3)."""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pymarxan.io.readers import read_input_dat
from pymarxan.io.writers import write_input_dat
from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import (
    ZonalProblem,
    ZoneHeuristicSolver,
    ZoneIISolver,
    ZoneLockWarning,
    ZoneMIPSolver,
    ZoneSASolver,
    load_zone_project,
)
from pymarxan.zones.diagnostics import min_forced_cost
from pymarxan.zones.model import rule_violations
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

ONE = SolverConfig(num_solutions=1)
ZONES_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


def _on(p: ZonalProblem, value: object = 1) -> ZonalProblem:
    return p.copy_with(parameters={**p.parameters, "MANDATORYALLOCATION": value})


def _six_pu() -> ZonalProblem:
    """Anchor + PU4 (status 0), PU5 (status 3), PU6 (status 1); zone costs z1 = 8, z2 = 1.5."""
    p = make_anchor_problem()
    pu = pd.concat([
        p.planning_units,
        pd.DataFrame({"id": [4, 5, 6], "cost": [0.0] * 3, "status": [0, 3, 1]}),
    ], ignore_index=True)
    zc = pd.concat([
        p.zone_costs,
        pd.DataFrame({
            "pu": [4, 5, 6, 4, 5, 6], "zone": [1, 1, 1, 2, 2, 2],
            "cost": [8.0, 8.0, 8.0, 1.5, 1.5, 1.5],
        }),
    ], ignore_index=True)
    return p.copy_with(planning_units=pu, zone_costs=zc)


@pytest.mark.parametrize("value", [1, 1.0])
def test_accessor_accepts(value: object) -> None:
    p = _on(make_anchor_problem(), value)
    assert p.mandatory_allocation() is True
    assert p.validate() == []


@pytest.mark.parametrize("value", ["1", True, 2, None])
def test_accessor_rejects(value: object) -> None:
    p = _on(make_anchor_problem(), value)
    assert p.mandatory_allocation() is False
    assert any("MANDATORYALLOCATION" in e for e in p.validate())


def test_rules_free_pus_must_be_assigned() -> None:
    rules = _on(_six_pu()).pu_zone_rules()
    assert [r.may_be_unassigned for r in rules] == [False, False, False, False, True, False]
    assert rules[5].start == 1  # status 1 start unchanged


def test_mip_optimum_matches_enumeration() -> None:
    p = _on(_six_pu())
    sol = ZoneMIPSolver().solve(p, ONE)[0]
    assert tuple(int(z) for z in sol.zone_assignment) == (1, 2, 2, 2, 0, 2)
    assert sol.cost == pytest.approx(10.5)


def test_every_solver_respects_rules() -> None:
    p = _on(_six_pu())
    rules = p.pu_zone_rules()
    solvers = [
        ZoneMIPSolver(),
        ZoneSASolver(num_iterations=20_000, num_temp_steps=100),
        ZoneHeuristicSolver(),
        ZoneIISolver(),  # ITIMPTYPE is read from the parameters, set per case below
    ]
    for solver in solvers:
        cases = [2, 3] if isinstance(solver, ZoneIISolver) else [None]
        for itimp in cases:
            q = p if itimp is None else p.copy_with(parameters={**p.parameters, "ITIMPTYPE": itimp})
            for s in solver.solve(q, SolverConfig(num_solutions=1, seed=42)):
                assert rule_violations(rules, s.zone_assignment) == []
                zero = [k for k, z in enumerate(s.zone_assignment) if z == 0]
                assert zero == [4]  # only the status-3 PU


def test_heuristic_seed_changes_on_zones_data() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = load_zone_project(ZONES_DIR)
    p = p.copy_with(planning_units=p.planning_units.assign(status=0))
    off = ZoneHeuristicSolver().solve(p, SolverConfig(num_solutions=1, seed=1))[0]
    on = ZoneHeuristicSolver().solve(_on(p), SolverConfig(num_solutions=1, seed=1))[0]
    assert (off.zone_assignment == 0).any()  # v0.38 seed leaves free PUs unassigned
    assert (on.zone_assignment != 0).all()


def test_mip_zero_target_assigns_cheapest_zone() -> None:
    p = make_anchor_problem()
    p = p.copy_with(features=p.features.assign(target=0.0), zone_targets=None)
    off = ZoneMIPSolver().solve(p, ONE)[0]
    on = ZoneMIPSolver().solve(_on(p), ONE)[0]
    assert tuple(off.zone_assignment) == (0, 0, 0)
    assert tuple(on.zone_assignment) == (2, 2, 2)  # zone 2 costs 1 each


def test_status3_warning_without_lock_tables_once() -> None:
    p = _on(_six_pu())
    assert p.pu_locks is None and p.pu_zones is None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ZoneMIPSolver().solve(p, ONE)
    hits = [x for x in w if issubclass(x.category, ZoneLockWarning)]
    assert len(hits) == 1
    assert "locked out (status 3) while MANDATORYALLOCATION is 1" in str(hits[0].message)
    assert p.validate() == []  # a warning, not an error


def test_budget_floor_with_mandatory() -> None:
    p = _on(make_anchor_problem())
    assert min_forced_cost(p) == pytest.approx(3.0)  # three PUs at zone-2 cost 1
    p = p.copy_with(parameters={**p.parameters, "COSTBUDGET": 2.0})
    with pytest.warns(UserWarning, match=r"at least 3 > COSTBUDGET 2"):
        assert ZoneMIPSolver(objective="max_features").solve(p, ONE) == []


def test_single_zone_forces_every_eligible_pu() -> None:
    p = make_anchor_problem()
    p = p.copy_with(
        zones=p.zones.iloc[:1],
        zone_costs=p.zone_costs[p.zone_costs["zone"] == 1],
        zone_contributions=p.zone_contributions[p.zone_contributions["zone"] == 1],
        zone_targets=None,
    )
    sol = ZoneMIPSolver().solve(_on(p), ONE)[0]
    assert tuple(sol.zone_assignment) == (1, 1, 1)


def test_mandatory_non_contiguous_zone_ids() -> None:
    p = make_anchor_problem()
    remap = {1: 5, 2: 9}
    p = p.copy_with(
        zones=p.zones.assign(id=p.zones["id"].map(remap)),
        zone_costs=p.zone_costs.assign(zone=p.zone_costs["zone"].map(remap)),
        zone_contributions=p.zone_contributions.assign(
            zone=p.zone_contributions["zone"].map(remap)),
        zone_targets=p.zone_targets.assign(zone=p.zone_targets["zone"].map(remap)),
        parameters={**p.parameters, "MANDATORYALLOCATION": 1, "COSTBUDGET": 7.0},
    )
    sol = ZoneMIPSolver(objective="max_features").solve(p, ONE)[0]
    assert set(int(z) for z in sol.zone_assignment) <= {5, 9}
    assert 0 not in set(int(z) for z in sol.zone_assignment)
    tight = p.copy_with(parameters={**p.parameters, "COSTBUDGET": 2.0})
    with pytest.warns(UserWarning, match=r"at least 3 > COSTBUDGET 2"):
        ZoneMIPSolver(objective="max_features").solve(tight, ONE)


def test_input_dat_round_trip(tmp_path: Path) -> None:
    params = {"MANDATORYALLOCATION": 1, "BLM": 0.0}
    write_input_dat(params, tmp_path / "input.dat")
    assert read_input_dat(tmp_path / "input.dat")["MANDATORYALLOCATION"] == 1


def test_off_matches_enumerated_unassigned_optimum() -> None:
    # With the parameter off, the anchor still prefers (1, 2, 2) at 7; unassigned allowed.
    assert tuple(ZoneMIPSolver().solve(make_anchor_problem(), ONE)[0].zone_assignment) == (1, 2, 2)
    assert len(list(itertools.product([0, 1, 2], repeat=3))) == 27
```

`write_input_dat(params, path)` and the module-level `rule_violations(rules, assignment)` were verified while planning. The expected 10.5 at `(1, 2, 2, 2, 0, 2)` is quoted from spec Testing 8, which was enumerated in the first review. If the MIP disagrees, stop and report; do not edit the expected value.

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_mandatory_allocation.py -q`

Expected: `AttributeError: 'ZonalProblem' object has no attribute 'mandatory_allocation'`, among others.

- [ ] **Step 3: Implement** (`zones/model.py`)

Add a module helper next to `_known_id`:

```python
def _mandatory_value(raw: object) -> int | None:
    """0 or 1 for an accepted MANDATORYALLOCATION value (int/float 0 or 1), else None."""
    if isinstance(raw, bool | str) or not isinstance(raw, int | float):
        return None
    return int(raw) if raw in (0, 1) else None
```

Accessor, next to `zone_target_contrib`:

```python
    def mandatory_allocation(self) -> bool:
        """``MANDATORYALLOCATION`` = 1: every PU not locked out must take a zone.

        A pymarxan key (MarZone does not read it; prioritizr
        ``add_mandatory_allocation_constraints``). Use 1 only when the project's zones include
        MarZone's "available" zone; when pymarxan's zone 0 stands in for it, leave it 0.
        Invalid values are a ``validate()`` error and read as off.
        """
        return _mandatory_value(self.parameters.get("MANDATORYALLOCATION", 0)) == 1
```

In `pu_zone_rules`, read `mandatory = self.mandatory_allocation()` before the loop. The final branch becomes:

```python
                rule = PUZoneRule(
                    allowed=every, may_be_unassigned=not (mandatory and every), start=start,
                )
```

`every` is empty only with zero zones, and the guard keeps the assertion safe.

At the top of `_pu_lock_issue_list`, before the early return:

```python
        out: list[tuple[str, bool]] = []
        if self.mandatory_allocation() and "status" in self.planning_units.columns:
            locks = self.pu_lock_lookup()
            n_out = sum(
                1
                for p, s in zip(
                    self.planning_units["id"].values,
                    self.planning_units["status"].values,
                    strict=True,
                )
                if _known_id(s, {0, 1, 2, 3}) == STATUS_LOCKED_OUT and int(p) not in locks
            )
            if n_out:
                out.append((
                    f"{n_out} planning unit(s) are locked out (status 3) while "
                    "MANDATORYALLOCATION is 1; they stay unassigned",
                    False,
                ))
        if self.pu_locks is None and self.pu_zones is None:
            return out
```

Remove the old `out = []` line further down.

In `validate()`, after the `zone_target_contrib` check:

```python
        if _mandatory_value(self.parameters.get("MANDATORYALLOCATION", 0)) is None:
            errors.append(
                "MANDATORYALLOCATION must be 0 or 1 (int or float); got "
                f"{self.parameters.get('MANDATORYALLOCATION')!r}"
            )
```

- [ ] **Step 4: Run the tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/ tests/pymarxan/solvers/test_stage1_golden_v038.py -q`

Expected: all pass. `test_zone_golden_v037.py` must stay green, which confirms the parameter-off path is unchanged.

- [ ] **Step 5: CHANGELOG and commit**

```markdown
- `MANDATORYALLOCATION` (pymarxan key; prioritizr `add_mandatory_allocation_constraints`):
  1 forces every planning unit not locked out into a zone in all four zone solvers. Status-3
  units stay unassigned (one `ZoneLockWarning`). MarZone does not read this key. (stage1)
```

```bash
ruff check src/pymarxan/zones/model.py tests/pymarxan/zones/test_mandatory_allocation.py
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/zones/model.py
git add src/pymarxan/zones/model.py tests/pymarxan/zones/test_mandatory_allocation.py CHANGELOG.md
git commit -m "feat(stage1): MANDATORYALLOCATION via pu_zone_rules

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: `write_problem`

**Files:**
- Create: `src/pymarxan/solvers/export.py`
- Modify: `src/pymarxan/solvers/__init__.py` (export `write_problem`)
- Test: `tests/pymarxan/solvers/test_write_problem.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `MIPSolver.build_model` (Task 4) and `ZoneMIPSolver.build_model` (Task 7).
- Produces: `write_problem(problem, path, *, solver=None, config=None) -> pathlib.Path`.

- [ ] **Step 1: Write the failing tests**

```python
"""write_problem (spec §3, Testing 9)."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pulp
import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers import HeuristicSolver, MIPSolver, write_problem
from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZoneMIPSolver
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

SIMPLE = Path(__file__).parent.parent.parent / "data" / "simple"


def _simple(blm: float):
    p = load_project(SIMPLE)
    return p.copy_with(parameters={**p.parameters, "BLM": blm})


def _solve_mps(path: Path) -> float:
    _, model = pulp.LpProblem.fromMPS(str(path))
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return float(pulp.value(model.objective))


def _solve_lp_with_cbc(path: Path) -> float:
    out = subprocess.run(
        [pulp.PULP_CBC_CMD().path, str(path), "solve"], capture_output=True, text=True,
        check=True,
    ).stdout
    match = re.search(r"Objective value:\s+([-\d.eE+]+)", out)
    assert match, out
    return float(match.group(1))


@pytest.mark.parametrize("suffix", [".mps", ".lp"])
def test_simple_round_trip_blm0(tmp_path: Path, suffix: str) -> None:
    path = write_problem(_simple(0.0), tmp_path / f"m{suffix}")
    value = _solve_mps(path) if suffix == ".mps" else _solve_lp_with_cbc(path)
    assert value == pytest.approx(35.0)


def test_simple_round_trip_blm1_matches_in_process(tmp_path: Path) -> None:
    p = _simple(1.0)
    sol = MIPSolver().solve(p, SolverConfig(num_solutions=1))[0]
    path = write_problem(p, tmp_path / "m.mps")
    assert _solve_mps(path) == pytest.approx(sol.metadata["mip_objective_value"])
    assert _solve_mps(path) == pytest.approx(43.0)


def test_zone_round_trip_and_bound_survives(tmp_path: Path) -> None:
    import pandas as pd
    p = make_anchor_problem().copy_with(pu_zones=pd.DataFrame({"pu": [1], "zone": [1]}))
    path = write_problem(p, tmp_path / "z.mps")
    assert _solve_mps(path) == pytest.approx(7.0)
    text = path.read_text()
    assert "x_1_2" in text  # PU1 zone 2 forced to 0: an upper bound row in BOUNDS


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".lp"):
        write_problem(_simple(0.0), tmp_path / "m.txt")
    with pytest.raises(TypeError, match="build_model"):
        write_problem(_simple(0.0), tmp_path / "m.lp", solver=HeuristicSolver())
    with pytest.raises(TypeError, match="ZonalProblem"):
        write_problem(make_anchor_problem(), tmp_path / "m.lp", solver=MIPSolver())
    with pytest.raises(TypeError, match="ZonalProblem"):
        write_problem(_simple(0.0), tmp_path / "m.lp", solver=ZoneMIPSolver())


def test_default_solver_picks_by_problem_type(tmp_path: Path) -> None:
    assert write_problem(make_anchor_problem(), tmp_path / "z.lp").exists()
    assert write_problem(_simple(0.0), tmp_path / "c.lp").exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_write_problem.py -q`

Expected: `ImportError: cannot import name 'write_problem'`.

- [ ] **Step 3: Implement** `src/pymarxan/solvers/export.py`

```python
"""Export a MIP solver's model to LP / MPS (prioritizr ``write_problem``).

Deliberate exception to the ``read_*`` / ``write_*`` pairing convention: nothing reads a
model back into a problem. Time limit and gap are solver settings, not part of the model;
no-good cuts are included only when that ``config`` is passed. ``solver`` is a pymarxan
Solver instance (prioritizr's ``solver`` picks a backend).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.base import Solver, SolverConfig

_WRITERS = {".lp": "writeLP", ".mps": "writeMPS"}


def write_problem(
    problem: ConservationProblem,
    path: str | Path,
    *,
    solver: Solver | None = None,
    config: SolverConfig | None = None,
) -> Path:
    """Build ``solver``'s MIP for ``problem`` and write it to ``path`` (.lp or .mps)."""
    from pymarxan.zones.model import ZonalProblem

    out = Path(path)
    suffix = out.suffix.lower()
    if suffix not in _WRITERS:
        raise ValueError(f"write_problem: path must end in .lp or .mps, got {out.name!r}")
    zonal = isinstance(problem, ZonalProblem)
    if solver is None:
        if zonal:
            from pymarxan.zones.mip_solver import ZoneMIPSolver
            solver = ZoneMIPSolver()
        else:
            from pymarxan.solvers.mip_solver import MIPSolver
            solver = MIPSolver()
    build: Any = getattr(solver, "build_model", None)
    if build is None:
        raise TypeError(
            f"write_problem: {type(solver).__name__} has no build_model; use MIPSolver or "
            "ZoneMIPSolver"
        )
    if solver.supports_zones() != zonal:
        raise TypeError(
            f"write_problem: {type(solver).__name__} does not match the problem type "
            f"({'a ZonalProblem needs ZoneMIPSolver' if zonal else 'ZonalProblem solvers need a ZonalProblem'})"
        )
    model = build(problem, config)
    constant = float(model.objective.constant) if model.objective is not None else 0.0
    if constant != 0.0:
        warnings.warn(
            f"write_problem: the objective constant {constant} is dropped by LP/MPS writers",
            UserWarning,
            stacklevel=2,
        )
    getattr(model, _WRITERS[suffix])(str(out))
    return out
```

Wrap the long f-string to 99 columns. Export it from `pymarxan.solvers.__init__`.

- [ ] **Step 4: Run the tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_write_problem.py -v`

Expected: 7 passed. If `test_zone_round_trip_and_bound_survives` fails only on the `"x_1_2"` text check, the MPS writer may have dropped a variable with `upBound 0` because it appears in no constraint. Check with `grep x_1_2` on the file. That variable does appear in `assign_1`, so it should be written. Report the finding rather than weakening the assertion.

- [ ] **Step 5: CHANGELOG and commit**

```markdown
- `pymarxan.solvers.write_problem(problem, path, solver=None)` — write the MIP model to
  `.lp` / `.mps` (prioritizr `write_problem`); `MIPSolver.build_model` /
  `ZoneMIPSolver.build_model`. (stage1)
```

```bash
ruff check src/pymarxan/solvers/export.py tests/pymarxan/solvers/test_write_problem.py
git add src/pymarxan/solvers/export.py src/pymarxan/solvers/__init__.py tests/pymarxan/solvers/test_write_problem.py CHANGELOG.md
git commit -m "feat(stage1): write_problem (.lp/.mps) for both MIP solvers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: `presolve_check`, the solver flag, and the model scan

**Files:**
- Create: `src/pymarxan/solvers/presolve.py`
- Modify:
  - `src/pymarxan/solvers/__init__.py`
  - `src/pymarxan/__init__.py` (lazy `__getattr__`)
  - `src/pymarxan/solvers/mip_solver.py` and `src/pymarxan/zones/mip_solver.py` (`presolve_check=` keyword)
- Test: `tests/pymarxan/solvers/test_presolve.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces:
  - `presolve_check(problem) -> list[str]`
  - `class PresolveWarning(UserWarning)`
  - `scan_model(model: pulp.LpProblem) -> list[str]` (P10)
  - `_validate_presolve_flag(value: object) -> bool | str`
  - `_emit_presolve(flag, messages: list[str], solver_name: str) -> None` (stacklevel 3)
- Consumes: `min_forced_cost`, `_zone_cost_lookup`, `diagnose_zone_infeasibility` (all imported lazily inside functions).

- [ ] **Step 1: Write the failing tests**

`tests/pymarxan/solvers/test_presolve.py`:

```python
"""presolve_check (spec §4, Testing 10, 14; Review Focus 4)."""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers import MIPSolver, PresolveWarning, presolve_check
from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZoneMIPSolver, load_zone_project
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem

DATA = Path(__file__).parent.parent.parent / "data"
ONE = SolverConfig(num_solutions=1)


def _core(**over: object) -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [2.0], "spf": [1.0]})
    pvf = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [1.0, 2.0]})
    bnd = pd.DataFrame({"id1": [1], "id2": [2], "boundary": [1.0]})
    params = {"BLM": 0.0}
    kw = {"planning_units": pu, "features": feat, "pu_vs_features": pvf, "boundary": bnd,
          "parameters": params}
    kw.update(over)
    return ConservationProblem(**kw)  # type: ignore[arg-type]


def _codes(msgs: list[str]) -> set[str]:
    return {m.split(":", 1)[0] for m in msgs}


def test_fixtures_clean() -> None:
    assert presolve_check(load_project(DATA / "simple")) == []
    assert presolve_check(make_anchor_problem()) == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert presolve_check(load_zone_project(DATA / "zones")) == []


def test_validate_short_circuit() -> None:
    p = _core(features=pd.DataFrame({"id": [1], "name": ["a"]}))  # no target column
    msgs = presolve_check(p)
    assert len(msgs) == 1 and msgs[0].startswith("P0: problem fails validate()")


@pytest.mark.parametrize("code, problem", [
    ("P1", lambda: _core(planning_units=pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, -3.0, 3.0], "status": [0, 0, 0]}))),
    ("P2", lambda: _core(planning_units=pd.DataFrame(
        {"id": [1, 2, 3], "cost": [0.01, 2.0, 1.2e5], "status": [0, 0, 0]}))),
    ("P3", lambda: _core(pu_vs_features=pd.DataFrame(
        {"species": [1, 1], "pu": [1, 2], "amount": [1e-4, 1e4]}))),
    ("P4", lambda: _core(boundary=pd.DataFrame(
        {"id1": [1, 2], "id2": [2, 3], "boundary": [1e-4, 1e4]}))),
    ("P5", lambda: _core(features=pd.DataFrame(
        {"id": [1, 2], "name": ["a", "g"], "target": [2.0, 10.0], "spf": [1.0, 1.0]}))),
    ("P6", lambda: _core(features=pd.DataFrame(
        {"id": [1], "name": ["a"], "target": [50.0], "spf": [1.0]}))),
    ("P7a", lambda: _core(planning_units=pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [2, 2, 2]}))),
    ("P7b", lambda: _core(planning_units=pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [3, 3, 3]}))),
    ("P8", lambda: _core(parameters={"BLM": 1e8})),
    ("P9a", lambda: _core(
        planning_units=pd.DataFrame(
            {"id": [1, 2, 3], "cost": [1.0, 20.0, 3.0], "status": [0, 2, 0]}),
        parameters={"BLM": 0.0, "COSTBUDGET": 5.0})),
    ("P9b", lambda: _core(parameters={"BLM": 0.0, "COSTBUDGET": 0.5})),
    ("P11", lambda: _core(parameters={"BLM": 0.0, "COSTBUDGET": 100.0})),
])
def test_each_check_triggers(code: str, problem: object) -> None:
    msgs = presolve_check(problem())  # type: ignore[operator]
    assert code in _codes(msgs), msgs


def test_p6_ignores_locked_out_amounts() -> None:
    p = _core(
        planning_units=pd.DataFrame(
            {"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 3, 0]}),
    )  # target 2; PU2's amount 2 is locked out, only 1 is reachable
    assert "P6" in _codes(presolve_check(p))


def test_zone_checks_use_rules() -> None:
    p = make_anchor_problem()
    p = p.copy_with(parameters={**p.parameters, "MANDATORYALLOCATION": 1, "COSTBUDGET": 2.0})
    assert "P9a" in _codes(presolve_check(p))  # floor 3 > 2
    q = make_anchor_problem().copy_with(
        planning_units=make_anchor_problem().planning_units.assign(status=3))
    assert "P7b" in _codes(presolve_check(q))


def test_presolve_nan_amount_never_raises() -> None:
    p = _core(pu_vs_features=pd.DataFrame(
        {"species": [1, 1], "pu": [1, 2], "amount": [np.nan, 2.0]}))
    msgs = presolve_check(p)
    assert _codes(msgs) & {"P0", "P1"}
    q = _core(planning_units=pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, np.inf, 3.0], "status": [0, 0, 0]}))
    assert _codes(presolve_check(q)) & {"P0", "P1"}


def test_flag_values() -> None:
    for bad in (0, 1, "yes", None):
        with pytest.raises(ValueError, match="presolve_check"):
            MIPSolver(presolve_check=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="presolve_check"):
            ZoneMIPSolver(presolve_check=bad)  # type: ignore[arg-type]


def test_flag_true_warns_once_at_caller() -> None:
    p = _core(parameters={"BLM": 1e8})
    for call in (MIPSolver(presolve_check=True).solve,
                 MIPSolver(presolve_check=True).build_model):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            call(p)
        hits = [x for x in w if issubclass(x.category, PresolveWarning)]
        assert len(hits) == 1 and hits[0].filename == __file__
        assert "P8" in str(hits[0].message)


def test_flag_strict_raises_and_off_silent() -> None:
    p = _core(parameters={"BLM": 1e8})
    with pytest.raises(ValueError, match="P8"):
        MIPSolver(presolve_check="strict").solve(p, ONE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PresolveWarning)
        MIPSolver().solve(p, ONE)


def test_model_scan_p10_tiny_relative_target() -> None:
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [1e-8], "spf": [1.0]})
    p = _core(features=feat, parameters={"BLM": 0.0, "COSTBUDGET": 2.0})
    with pytest.warns(PresolveWarning, match="P10"):
        MIPSolver(objective="min_shortfall", presolve_check=True).solve(p, ONE)
    assert "P10" not in _codes(presolve_check(p))  # data-only function has no model


@pytest.mark.parametrize("module", [
    "pymarxan", "pymarxan.solvers", "pymarxan.zones", "pymarxan.solvers.presolve",
])
def test_import_order_no_cycle(module: str) -> None:
    code = f"import {module}; import pymarxan; pymarxan.presolve_check; pymarxan.PresolveWarning"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[3],
                   env={"PYTHONPATH": "src", "PATH": ""})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_presolve.py -q`

Expected: `ImportError: cannot import name 'PresolveWarning'`.

- [ ] **Step 3: Implement** `src/pymarxan/solvers/presolve.py`

```python
"""Pre-solve data checks (inspired by prioritizr's ``presolve_check``; the checks differ).

prioritizr runs its check by default on the compiled model with absolute thresholds
(1e6 / 1e-6); here the data checks are opt-in, add ratio checks (P2-P4, P8), and the
absolute scan of the built model (P10) runs only through a solver's ``presolve_check=``
flag. prioritizr's single-feature and >= 50 % empty-PU checks are not ported.
No zone imports at module level (``pymarxan.zones`` imports ``pymarxan.solvers``).
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pulp

if TYPE_CHECKING:
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.zones.model import ZonalProblem

RATIO_LIMIT = 1e6
ABS_HIGH = 1e6
ABS_LOW = 1e-6


class PresolveWarning(UserWarning):
    """``presolve_check=True`` found numerical or data problems before solving."""


def _validate_presolve_flag(value: object) -> bool | str:
    if value is True or value is False or (isinstance(value, str) and value == "strict"):
        return value  # type: ignore[return-value]
    raise ValueError(f"presolve_check must be True, False or 'strict', got {value!r}")


def _emit_presolve(flag: bool | str, messages: list[str], solver_name: str) -> None:
    # _emit_presolve <- solve()/build_model() <- caller: stacklevel 3.
    if not flag or not messages:
        return
    text = f"{solver_name}: presolve check: " + " | ".join(messages)
    if flag == "strict":
        raise ValueError(text)
    warnings.warn(text, PresolveWarning, stacklevel=3)


def _first(items: list[str], n: int = 5) -> str:
    more = f" (+{len(items) - n} more)" if len(items) > n else ""
    return "; ".join(items[:n]) + more


def _nonfinite(code: str, named: Iterable[tuple[str, Any]]) -> list[str]:
    out: list[str] = []
    for name, values in named:
        arr = np.asarray(values, dtype=float).ravel()
        bad = int((~np.isfinite(arr)).sum())
        if bad:
            out.append(f"{code}: {name} has {bad} non-finite value(s)")
    return out


def _ratio(code: str, label: str, values: Any) -> list[str]:
    arr = np.abs(np.asarray(values, dtype=float).ravel())
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 2:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if hi / lo > RATIO_LIMIT:
        return [f"{code}: {label} span {hi / lo:.1e}x (min {lo:.3g}, max {hi:.3g}): "
                "solvers may lose precision"]
    return []


def _blm_check(blm: float, boundary: Any, costs: Any) -> list[str]:
    bnd = np.abs(np.asarray(boundary, dtype=float))
    c = np.abs(np.asarray(costs, dtype=float))
    c = c[np.isfinite(c) & (c > 0)]
    if blm <= 0 or bnd.size == 0 or c.size == 0:
        return []
    ratio = blm * float(bnd.max()) / float(np.median(c))
    if ratio > RATIO_LIMIT or ratio < 1.0 / RATIO_LIMIT:
        return [f"P8: BLM x boundary is {ratio:.1e}x the median cost: "
                + ("boundary will dominate" if ratio > 1 else "boundary has no effect")]
    return []


def _budget_checks(budget: float | None, floor: float, cheapest: float | None,
                   ceiling: float) -> list[str]:
    if budget is None:
        return []
    out: list[str] = []
    if budget < floor - 1e-9:
        out.append(f"P9a: COSTBUDGET {budget:g} is below the cost that locked / must-assign "
                   f"units force ({floor:g}): infeasible")
    elif cheapest is not None and budget < cheapest - 1e-9:
        out.append(f"P9b: COSTBUDGET {budget:g} is below the cheapest selectable option "
                   f"({cheapest:g}): nothing can be selected")
    if budget >= ceiling and ceiling > 0:
        out.append(f"P11: COSTBUDGET {budget:g} reaches the total cost {ceiling:g}: "
                   "the budget does not constrain the solution")
    return out


def presolve_check(problem: ConservationProblem) -> list[str]:
    """Data problems likely to make a solve slow, imprecise or meaningless. Never raises.

    Returns ``[]`` when clean; every message starts with its code (``"P5: ..."``). When
    ``problem.validate()`` fails, returns one ``"P0: ..."`` message and skips the rest.
    """
    try:
        errors = problem.validate()
    except Exception as exc:  # validate() should not raise; presolve must not either
        return [f"P0: problem.validate() raised {exc!r}"]
    if errors:
        return [f"P0: problem fails validate(): {len(errors)} error(s); fix those first"]
    from pymarxan.zones.model import ZonalProblem

    try:
        if isinstance(problem, ZonalProblem):
            return _zone_checks(problem)
        return _core_checks(problem)
    except Exception as exc:  # a diagnostic must never crash the caller
        return [f"P0: presolve check failed: {exc!r}"]


def _common(problem: ConservationProblem, costs: Any, named_extra: list[tuple[str, Any]]
            ) -> list[str]:
    params = problem.parameters
    budget = params.get("COSTBUDGET")
    amounts = problem.pu_vs_features["amount"].to_numpy(dtype=float)
    targets = problem.features["target"].to_numpy(dtype=float)
    bnd = (problem.boundary["boundary"].to_numpy(dtype=float)
           if problem.boundary is not None else np.empty(0))
    named = [("cost", costs), ("amount", amounts), ("target", targets), ("boundary", bnd),
             ("BLM", [float(params.get("BLM", 0.0))])]
    if budget is not None:
        named.append(("COSTBUDGET", [float(budget)]))
    out = _nonfinite("P1", named + named_extra)
    neg = [f"{v:g}" for v in np.asarray(costs, dtype=float) if v < 0]
    if neg:
        out.append(f"P1: {len(neg)} negative cost(s): {_first(neg)}")
    out += _ratio("P2", "costs", costs)
    out += _ratio("P3", "amounts", amounts)
    out += _ratio("P4", "boundary lengths", bnd)
    out += _blm_check(float(params.get("BLM", 0.0)), bnd, costs)
    totals = problem.pu_vs_features.groupby("species")["amount"].sum()
    ghosts = [
        f"feature {int(f)} occurs in no planning unit but has target {t:g}"
        for f, t in zip(problem.features["id"].values, targets, strict=True)
        if t > 0 and not float(totals.get(int(f), 0.0)) > 0
    ]
    if ghosts:
        out.append("P5: " + _first(ghosts))
    return out


def _core_checks(problem: ConservationProblem) -> list[str]:
    pu = problem.planning_units
    costs = pu["cost"].to_numpy(dtype=float)
    status = (pu["status"].to_numpy(dtype=int) if "status" in pu.columns
              else np.zeros(len(pu), dtype=int))
    out = _common(problem, costs, [])
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    locked_out = {int(p) for p, s in zip(pu["id"].values, status, strict=True) if s == 3}
    pvf = problem.pu_vs_features
    avail = pvf[~pvf["pu"].isin(locked_out)].groupby("species")["amount"].sum()
    totals = pvf.groupby("species")["amount"].sum()
    short = [
        f"feature {int(f)} target {t * misslevel:g} exceeds its total amount "
        f"{float(avail.get(int(f), 0.0)):g} outside locked-out units"
        for f, t in zip(problem.features["id"].values,
                        problem.features["target"].to_numpy(dtype=float), strict=True)
        if t > 0 and float(totals.get(int(f), 0.0)) > 0
        and float(avail.get(int(f), 0.0)) < t * misslevel - 1e-9
    ]
    if short:
        out.append("P6: " + _first(short))
    if len(status) and (status == 2).all():
        out.append("P7a: every planning unit is locked in: nothing to optimise")
    if len(status) and (status == 3).all():
        out.append("P7b: every planning unit is locked out: nothing can be selected")
    free = costs[(status == 0) | (status == 1)]
    budget = problem.parameters.get("COSTBUDGET")
    out += _budget_checks(
        None if budget is None else float(budget),
        floor=float(costs[status == 2].sum()),
        cheapest=float(free.min()) if free.size else None,
        ceiling=float(np.clip(costs[status != 3], 0, None).sum()),
    )
    return out


def _zone_checks(problem: ZonalProblem) -> list[str]:
    from pymarxan.zones.diagnostics import (
        _zone_cost_lookup,
        diagnose_zone_infeasibility,
        min_forced_cost,
    )

    zc = problem.zone_costs["cost"].to_numpy(dtype=float)
    extra: list[tuple[str, Any]] = [
        ("zone contribution", problem.zone_contributions["contribution"].to_numpy(dtype=float)
         if problem.zone_contributions is not None else []),
    ]
    out = _common(problem, zc, extra)
    unreachable = diagnose_zone_infeasibility(problem)
    if unreachable:
        out.append("P6: " + _first(unreachable))
    rules = problem.pu_zone_rules()
    if rules and all(len(r.allowed) == 1 and not r.may_be_unassigned for r in rules):
        out.append("P7a: every planning unit is locked in: nothing to optimise")
    if rules and all(not r.allowed for r in rules):
        out.append("P7b: every planning unit is locked out: nothing can be selected")
    lookup = _zone_cost_lookup(problem)
    pids = [int(p) for p in problem.planning_units["id"].values]
    options = [[lookup.get((pid, z), 0.0) for z in r.allowed] for pid, r in zip(pids, rules)]
    free_costs = [c for opts, r in zip(options, rules) if r.may_be_unassigned for c in opts]
    budget = problem.parameters.get("COSTBUDGET")
    out += _budget_checks(
        None if budget is None else float(budget),
        floor=min_forced_cost(problem),
        cheapest=min(free_costs) if free_costs else None,
        ceiling=float(sum(max(0.0, max(opts)) for opts in options if opts)),
    )
    return out


def scan_model(model: pulp.LpProblem) -> list[str]:
    """P10: absolute coefficient / rhs magnitudes outside [1e-6, 1e6] (prioritizr thresholds)."""
    worst: tuple[float, str] | None = None
    count = 0

    def visit(value: float, where: str) -> None:
        nonlocal worst, count
        a = abs(value)
        if a > ABS_HIGH or 0 < a < ABS_LOW:
            count += 1
            score = a if a > 1 else 1.0 / a
            if worst is None or score > worst[0]:
                worst = (score, f"{where} = {value:.3g}")

    if model.objective is not None:
        for v, a in model.objective.items():
            visit(float(a), f"objective coefficient of {v.name}")
    for name, c in model.constraints.items():
        visit(float(-c.constant), f"rhs of {name}")
        for v, a in c.items():
            visit(float(a), f"coefficient of {v.name} in {name}")
    if worst is None or not math.isfinite(worst[0]):
        return [] if worst is None else [f"P10: {count} non-finite model value(s)"]
    return [f"P10: {count} model coefficient(s) outside [1e-6, 1e6]; worst: {worst[1]}"]
```

Then wire up the exports and the flag.

**Exports.** In `pymarxan/solvers/__init__.py`, export `PresolveWarning`, `presolve_check` and `write_problem`.

**`pymarxan/__init__.py`** — append:

```python
def __getattr__(name: str) -> object:
    # Lazy: ``import pymarxan`` must not import pandas / pulp / zones.
    if name in ("presolve_check", "PresolveWarning"):
        from pymarxan.solvers import presolve

        return getattr(presolve, name)
    raise AttributeError(f"module 'pymarxan' has no attribute {name!r}")
```

**Solver flag** (both solvers):
- Add the constructor keyword `presolve_check: bool | str = False` and store `self.presolve_check = _validate_presolve_flag(presolve_check)`.
- In both `solve` and `build_model`, directly after `_build`:

```python
        messages = presolve_check(problem) + scan_model(built.model) if self.presolve_check else []
        _emit_presolve(self.presolve_check, messages, self.name())
```

The name `presolve_check` collides with the attribute; import the function as `from pymarxan.solvers.presolve import presolve_check as _presolve_data`.

In `build_model`, `_build(...)` currently returns `.model` inline. Change it to `built = self._build(...)`, then the two lines above, then `return built.model`.

- [ ] **Step 4: Run the tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_presolve.py tests/pymarxan/solvers tests/pymarxan/zones -q`

Expected: all pass.

When a check fires on a fixture it was not meant for, read the message and fix the check's logic rather than the fixture. Example: P11 on `_core` with no budget cannot fire, because budget is None.

- [ ] **Step 5: CHANGELOG and commit**

```markdown
- `presolve_check(problem)` (also `pymarxan.presolve_check`) and `presolve_check=True |
  "strict"` on both MIP solvers — data and model-coefficient checks before solving, inspired by
  prioritizr's `presolve_check` (checks differ; opt-in). `PresolveWarning`. (stage1)
```

```bash
ruff check src/pymarxan/solvers/presolve.py src/pymarxan/__init__.py tests/pymarxan/solvers/test_presolve.py
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/solvers/presolve.py src/pymarxan/__init__.py
git add src/pymarxan/solvers/presolve.py src/pymarxan/solvers/__init__.py src/pymarxan/__init__.py src/pymarxan/solvers/mip_solver.py src/pymarxan/zones/mip_solver.py tests/pymarxan/solvers/test_presolve.py CHANGELOG.md
git commit -m "feat(stage1): presolve_check, solver presolve flag and model scan

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Deprecate `pymarxan.objectives` (S1)

**Files:**
- Modify: `src/pymarxan/objectives/__init__.py`
- Modify: `tests/pymarxan/objectives/test_objectives.py`
- Modify: `src/pymarxan_shiny/modules/solver_config/objective_selector.py` (docstring only)
- Test: `tests/pymarxan/objectives/test_objectives_deprecated.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the failing test**

```python
"""pymarxan.objectives is deprecated in v0.39 (spec §5, Testing 13)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
PROBE = (
    "import warnings\n"
    "with warnings.catch_warnings(record=True) as w:\n"
    "    warnings.simplefilter('always')\n"
    "    import {mod}\n"
    "print(sum(issubclass(x.category, DeprecationWarning) and 'objectives' in str(x.message)"
    " for x in w))\n"
)


def _count(mod: str) -> int:
    out = subprocess.run(
        [sys.executable, "-c", PROBE.format(mod=mod)], cwd=ROOT, check=True,
        capture_output=True, text=True, env={"PYTHONPATH": "src", "PATH": ""},
    ).stdout
    return int(out.strip())


def test_objectives_import_warns_once() -> None:
    assert _count("pymarxan.objectives") == 1


def test_plain_import_does_not_warn() -> None:
    assert _count("pymarxan") == 0
    assert _count("pymarxan.solvers") == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/objectives/test_objectives_deprecated.py -v`

Expected: `test_objectives_import_warns_once` fails, `assert 0 == 1`.

- [ ] **Step 3: Implement**

At the top of `src/pymarxan/objectives/__init__.py`, after the docstring and the `__future__` import:

```python
import os
import warnings

warnings.warn(
    "pymarxan.objectives is deprecated and will be removed in v0.42; use "
    "MIPSolver(objective=...) / ZoneMIPSolver(objective=...)",
    DeprecationWarning,
    # Attribute to the importer's line (stacklevel would land in importlib frames).
    skip_file_prefixes=(os.path.dirname(__file__), "<frozen importlib"),
)
```

This is a deviation from spec §5, which says `stacklevel=2`. The planning prototype showed that `skip_file_prefixes` attributes the warning to the importer's frame, so Python's default filter shows it. `stacklevel=2` points at importlib instead.

In `tests/pymarxan/objectives/test_objectives.py`, wrap the package import:

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from pymarxan.objectives import ...  # existing names, unchanged
```

In the `objective_selector.py` module docstring, add: "The selected objective is stored in `cfg['objective_type']` but no solver reads it yet (known gap; wiring planned in the Shiny follow-ups)."

- [ ] **Step 4: Run the tests**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/objectives tests/pymarxan_shiny -q -k "objective"`

Expected: pass.

- [ ] **Step 5: CHANGELOG and commit**

```markdown
### Deprecated
- `pymarxan.objectives` (unused `Objective` classes; their zone builders were incomplete and
  read `BUDGET`): use `MIPSolver(objective=...)` / `ZoneMIPSolver(objective=...)`. Removal in
  v0.42. (stage1, S1)
```

```bash
ruff check src/pymarxan/objectives/__init__.py tests/pymarxan/objectives/
git add src/pymarxan/objectives/__init__.py src/pymarxan_shiny/modules/solver_config/objective_selector.py tests/pymarxan/objectives/ CHANGELOG.md
git commit -m "chore(stage1): deprecate pymarxan.objectives

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Random-instance oracle, docs, and the gate

**Files:**
- Test: `tests/pymarxan/zones/test_zone_mip_random_oracle.py`
- Modify:
  - `docs/TUTORIAL.md` (§5 objective table, around line 159)
  - `docs/VALIDATION.md`
  - `docs/plans/2026-10-11-prioritizr-parity-roadmap.md`

- [ ] **Step 1: Write the oracle test**

```python
"""Seeded random-instance enumeration oracle for the zone budgeted objectives (S5)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZonalProblem, ZoneMIPSolver

N_PU, ZONES, N_F = 5, (1, 2, 3), 3
OBJECTIVES = ("max_features", "max_weighted_features", "min_shortfall", "min_largest_shortfall")


def _instance(seed: int, mandatory: int) -> tuple[ZonalProblem, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 10, size=(N_PU, len(ZONES))).astype(float)
    amount = rng.integers(0, 6, size=(N_PU, N_F)).astype(float)
    contrib = rng.choice([0.0, 0.5, 1.0], size=(len(ZONES), N_F))
    target = rng.integers(1, 12, size=N_F).astype(float)
    spf = rng.choice([1.0, 2.0, 5.0], size=N_F)
    ztarget = rng.integers(1, 8)  # one zone target: zone 2, feature 1
    budget = float(rng.uniform(cost.min(), cost.max(axis=1).sum()))
    pu = pd.DataFrame({"id": range(1, N_PU + 1), "cost": 0.0, "status": 0})
    pvf = pd.DataFrame([
        {"species": f + 1, "pu": i + 1, "amount": amount[i, f]}
        for i in range(N_PU) for f in range(N_F) if amount[i, f] > 0
    ])
    p = ZonalProblem(
        planning_units=pu,
        features=pd.DataFrame({"id": range(1, N_F + 1), "name": list("abc"),
                               "target": target, "spf": spf}),
        pu_vs_features=pvf, boundary=None,
        parameters={"BLM": 0.0, "COSTBUDGET": budget, "MANDATORYALLOCATION": mandatory},
        zones=pd.DataFrame({"id": list(ZONES), "name": ["z1", "z2", "z3"]}),
        zone_costs=pd.DataFrame([
            {"pu": i + 1, "zone": z, "cost": cost[i, k]}
            for i in range(N_PU) for k, z in enumerate(ZONES)
        ]),
        zone_contributions=pd.DataFrame([
            {"feature": f + 1, "zone": z, "contribution": contrib[k, f]}
            for k, z in enumerate(ZONES) for f in range(N_F)
        ]),
        zone_targets=pd.DataFrame({"zone": [2], "feature": [1], "target": [float(ztarget)]}),
        zone_boundary_costs=None,
    )
    data = {"cost": cost, "amount": amount, "contrib": contrib, "target": target,
            "spf": spf, "ztarget": np.array([ztarget], dtype=float), "budget": np.array([budget])}
    return p, data


def _best(objective: str, mandatory: int, d: dict[str, np.ndarray]) -> float:
    choices = ZONES if mandatory else (0, *ZONES)
    values: list[float] = []
    for a in itertools.product(choices, repeat=N_PU):
        spend = sum(d["cost"][i, z - 1] for i, z in enumerate(a) if z)
        if spend > d["budget"][0] + 1e-9:
            continue
        terms = []  # (achieved, required, weight)
        zone2 = sum(d["amount"][i, 0] for i, z in enumerate(a) if z == 2)
        terms.append((zone2, d["ztarget"][0], d["spf"][0]))
        for f in range(N_F):
            got = sum(d["amount"][i, f] * d["contrib"][z - 1, f] for i, z in enumerate(a) if z)
            terms.append((got, d["target"][f], d["spf"][f]))
        met = [got >= r - 1e-9 for got, r, _ in terms]
        rel = [max(0.0, r - got) / r for got, r, _ in terms]
        if objective == "max_features":
            values.append(float(sum(met)))
        elif objective == "max_weighted_features":
            values.append(sum(w for (_, _, w), m in zip(terms, met) if m))
        elif objective == "min_shortfall":
            values.append(sum(w * s for (_, _, w), s in zip(terms, rel)))
        else:
            values.append(max(rel))
    return max(values) if objective.startswith("max") else min(values)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("mandatory", [0, 1])
def test_zone_mip_matches_enumeration(seed: int, mandatory: int) -> None:
    p, d = _instance(seed, mandatory)
    for objective in OBJECTIVES:
        expected = _best(objective, mandatory, d)
        sols = ZoneMIPSolver(objective=objective).solve(p, SolverConfig(num_solutions=1))
        assert sols, (seed, mandatory, objective)
        assert sols[0].metadata["mip_objective_value"] == pytest.approx(expected, abs=1e-6), (
            seed, mandatory, objective,
        )
```

The budget is drawn from at least the minimum single-cell cost, but with mandatory allocation an infeasible draw is possible. If a `(seed, mandatory)` case returns `[]`, it is a fixture problem, not a solver bug. In that case, raise the lower bound to the mandatory floor (`cost.min(axis=1).sum()`) for `mandatory == 1`, and record that in the commit message.

- [ ] **Step 2: Run it**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_mip_random_oracle.py -v`

Expected: 20 passed. It is a new test over code already built, so it passes immediately. If any case fails, stop: that is a real defect. Debug it with `superpowers:systematic-debugging`; do not relax the oracle.

- [ ] **Step 3: Update the docs**

1. **`docs/TUTORIAL.md` §5 objective table.**
   - Add the `"min_shortfall"` row: "Minimise SPF-weighted **relative** shortfall | `COSTBUDGET`".
   - Change `"min_largest_shortfall"` to "Minimax over **relative** shortfall (`shortfall="absolute"` for v0.38 behaviour)".
   - Change `"min_penalties"` to "Exact lexicographic: SPF-weighted shortfall first, cost second (not prioritizr's `add_min_penalties_objective`)".
   - Add one paragraph on zone problems, `ZoneMIPSolver(objective=...)`, with the prioritizr name mapping table:

     | pymarxan | prioritizr ≥ 9 |
     |---|---|
     | `max_features` | `add_max_n_targets_met_objective` |
     | `max_weighted_features` | `add_max_n_targets_met_objective` + `add_feature_weights` (not `max_wtd_sum`) |
     | `min_shortfall` | `add_min_shortfall_objective` |
     | `min_largest_shortfall` | `add_min_largest_shortfall_objective` |

2. **`docs/VALIDATION.md`**, new subsection "Stage 1 zone-MIP objectives and mandatory allocation":
   - relative shortfall divisor R = target × MISSLEVEL (prioritizr has no MISSLEVEL);
   - ghost features count;
   - a missing zone cost is 0 (prioritizr: NA = forbidden);
   - MANDATORYALLOCATION is a pymarxan key; status 3 wins (prioritizr would be infeasible);
   - single-zone problems are allowed (prioritizr refuses);
   - SA never leaves a free PU unassigned even with the parameter off;
   - II at ITIMPTYPE 0/1 only reproduces its seed for must-assign PUs;
   - presolve differs from prioritizr (absolute vs ratio thresholds, opt-in);
   - core `min_penalties` is two-pass.

3. **The roadmap.** Mark Stage 1 shipped (v0.39). Record "remove `pymarxan.objectives` in v0.42" and "wire Shiny `objective_selector`" as follow-ups, and remove "min shortfall already at parity".

- [ ] **Step 4: Run the full gate**

```bash
PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:/home/razinka/marxan/.venv/bin:$PATH" make check
PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py
```

Expected:
- `make check` green: ruff, mypy and the full suite, coverage ≥ 75 %;
- the parity script reports MIP 35.0;
- `test_zone_golden_v037.py` and `test_stage1_golden_v038.py` are green.

Then run `make bench` once and compare against main. The only expected change is the zone-MIP model build, and `min_set` is unchanged.

- [ ] **Step 5: Commit**

```bash
ruff check tests/pymarxan/zones/test_zone_mip_random_oracle.py
git add tests/pymarxan/zones/test_zone_mip_random_oracle.py docs/TUTORIAL.md docs/VALIDATION.md docs/plans/2026-10-11-prioritizr-parity-roadmap.md
git commit -m "test(stage1): random-instance zone-MIP oracle; docs for v0.39 objectives

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage

| Spec item | Task |
|---|---|
| D1 terms / §1 TargetTerm | 2, 5 (core), 7 (zone) |
| D2 single budget | 5, 7 |
| D3 MANDATORYALLOCATION | 8 |
| D4 presolve function + flag | 10 |
| D5 builders module; supersedes objectives | 2, 11 |
| D6 objective set + alias; zone min_penalties excluded | 5, 7 |
| D7 relative units, `shortfall=` both solvers | 5, 7 |
| §1 core ghost fix, spf default | 3 |
| §1 S2 two-pass min_penalties | 2, 6 |
| §1 S9 zero-cost PU | 3 |
| §1 boundary-ignored warning; budget message; metadata | 7 |
| §2 accessor, rules, status-3 issue, `min_forced_cost`, single-zone | 7, 8 |
| §3 build/solve split; write_problem | 4, 7, 9 |
| §4 P0–P11, model scan P10, flag, lazy exports, import test | 10 |
| §5 deprecation | 11 |
| Testing 1–3, 6, 7 | 7 |
| Testing 4 (golden) | 1, then re-checked in 4, 7, 8 |
| Testing 5 | 1, 5 |
| Testing 8 | 8 |
| Testing 9 | 9 |
| Testing 10, 14 | 10 |
| Testing 11 (random oracle) | 12 |
| Testing 12 | 2, 6 |
| Testing 12b | 3 |
| Testing 13 | 11 |
| Testing 15 (gate) | 12 |

**Plan-level deviations from spec rev 3:**
- Presolve messages carry their code prefix (`P0`–`P11`).
- `_BuiltModel` carries the probmode / TARGET2 / sep flags rather than having `solve()` recompute them.
- The deprecation uses `skip_file_prefixes`, not `stacklevel=2` (executed during planning).
- P11's ceiling is Σ of the **most expensive** allowed option per PU, not the cheapest. Only that bound means the budget truly cannot bind.
