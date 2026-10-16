"""MultiObjectiveMIP bug-fix regressions and implied evaluation (Stage 3 spec §7.4, §7.5)."""
from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from typing import Any

import pandas as pd
import pulp
import pytest

import pymarxan.multiobjective as mo
import pymarxan.solvers.mip_solver as ms
from pymarxan.models.problem import ConservationProblem
from pymarxan.multiobjective import (
    Hierarchical,
    MaxFeatures,
    MaxWtdSum,
    MinBoundary,
    MinCost,
    MinLargestShortfall,
    MinShortfall,
    MultiObjectiveMIP,
    ObjectiveSpec,
    WeightedSum,
)
from pymarxan.solvers.base import Solution
from tests.pymarxan.scripted_solver import Scripted


def P(pu: dict[str, list[Any]], feat: dict[str, list[Any]], pvf: dict[str, list[Any]],
      **kw: Any) -> ConservationProblem:
    params = kw.pop("params", {})
    return ConservationProblem(
        planning_units=pd.DataFrame(pu), features=pd.DataFrame(feat),
        pu_vs_features=pd.DataFrame(pvf), parameters=params, **kw,
    )


def run(specs: Sequence[ObjectiveSpec], approach: Any, problem: ConservationProblem,
        **kw: Any) -> list[Solution]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return MultiObjectiveMIP(specs, approach, mip_backend="cbc", **kw).solve(problem)


def vals(sol: Solution) -> dict[str, float]:
    values: dict[str, float] = sol.metadata["objective_values"]
    return values


def sel(sol: Solution) -> list[int]:
    return [int(v) for v in sol.selected]


# --- caps (§7.4) -----------------------------------------------------------------


def test_negative_v_with_rel_tol_is_feasible() -> None:
    """Stage 1 maximises Σ −amount with PU1 locked in: v = −5. prioritizr's cap
    f >= v·(1 − r) = −2.5 is infeasible; ours, f >= v − (r·|v| + abs_tol), is not."""
    p = P({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [2, 0, 0]},
          {"id": [1], "name": ["a"], "target": [0.0]},
          {"species": [1, 1], "pu": [1, 2], "amount": [5.0, 1.0]})
    sol = run([MaxWtdSum(weights={1: -1.0}), MinCost()], Hierarchical(0.5), p)[0]
    assert sol.metadata["stage_reached"] == 2
    assert vals(sol) == {"max_wtd_sum": -5.0, "min_cost": 1.0}
    # No plan reaches prioritizr's cap.
    assert all(-(5.0 + 1.0 * x2) < -2.5 for x2 in (0, 1))


@pytest.mark.parametrize(("abs_tol", "expected"), [(1e-9, [1, 1, 1]), (0.5, [1, 1, 0])])
def test_v_zero_allows_exactly_abs_tol(abs_tol: float, expected: list[int]) -> None:
    """Best shortfall is 0 (no budget), so rel_tol = 10 gives no slack; abs_tol 0.5 lets
    stage 2 drop PU3 (relative shortfall 0.4) and save cost 5."""
    p = P({"id": [1, 2, 3], "cost": [1.0, 1.0, 5.0], "status": [0, 0, 0]},
          {"id": [1], "name": ["a"], "target": [5.0]},
          {"species": [1, 1, 1], "pu": [1, 2, 3], "amount": [2.0, 1.0, 2.0]})
    sol = run([MinShortfall(), MinCost()], Hierarchical(10.0, abs_tol=abs_tol), p)[0]
    assert sel(sol) == expected


def test_tolerance_boundary_cap_returns_cost_2() -> None:
    """Round 4: 0.7 + 0.1 = 0.7999999999999999 < 0.8, but CBC calls the target met.
    Recompute-only would cap at 0 and return the empty plan."""
    p = P({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [0.8]},
          {"species": [1, 1], "pu": [1, 2], "amount": [0.7, 0.1]})
    sol = run([MaxFeatures(), MinCost()], Hierarchical(rel_tol=0.0), p)[0]
    assert sol.cost == 2.0 and sel(sol) == [1, 1]
    # The reported count follows check_targets (the documented tolerance caveat).
    assert vals(sol)["max_features"] == 0.0


def test_incumbent_cap_keeps_both_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 5: a stage-1 incumbent meets both targets but leaves y_f2 = 0. Model-only
    would cap at 1 and let stage 2 drop a target."""
    p = P({"id": [1, 2, 3], "cost": [1.0, 1.0, 5.0], "status": [0, 0, 0]},
          {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0]},
          {"species": [1, 2, 1, 2], "pu": [1, 2, 3, 3], "amount": [1.0] * 4},
          params={"COSTBUDGET": 5.0})

    def loose(lp: pulp.LpProblem) -> None:
        lp.sol_status = pulp.LpSolutionIntegerFeasible
        values = {"x_1": 1.0, "x_2": 1.0, "x_3": 0.0, "y_f1": 1.0, "y_f2": 0.0}
        for v in lp.variables():
            if v.name in values:
                v.varValue = values[v.name]

    solver = Scripted({1: loose})
    monkeypatch.setattr(ms, "_make_pulp_solver", lambda *a, **k: solver)
    sol = run([MaxFeatures(), MinCost()], Hierarchical(0.0), p)[0]
    assert vals(sol) == {"max_features": 2.0, "min_cost": 2.0}
    assert sol.metadata["sol_status"] == 1  # stage 2's (proven optimal) status


def test_zero_cost_pu_without_amounts() -> None:
    p = P({"id": [1, 2], "cost": [1.0, 0.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [1.0]},
          {"species": [1], "pu": [1], "amount": [1.0]})
    sol = run([MinShortfall(), MinCost()], Hierarchical(0.0), p)[0]
    assert sel(sol) == [1, 0] and vals(sol) == {"min_shortfall": 0.0, "min_cost": 1.0}


def test_stage_reached_is_n_on_success() -> None:
    p = P({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [1.0]},
          {"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]})
    sol = run([MinShortfall(), MaxFeatures(), MinCost()], Hierarchical(0.0), p)[0]
    assert sol.metadata["stage_reached"] == 3


# --- reporting (§7.4) ------------------------------------------------------------


def test_objective_values_are_recomputed_not_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 0-weight MaxFeatures row leaves y free; a raw y_f1 = 0 is reported as met."""
    p = P({"id": [1, 2], "cost": [1.0, 1.0], "status": [2, 0]},
          {"id": [1], "name": ["a"], "target": [1.0]},
          {"species": [1], "pu": [1], "amount": [1.0]})
    raw: dict[str, float] = {}

    def zero_y(lp: pulp.LpProblem) -> None:
        for v in lp.variables():
            if v.name == "y_f1":
                v.varValue = 0.0
        raw["y_f1"] = 0.0

    monkeypatch.setattr(ms, "_make_pulp_solver", lambda *a, **k: Scripted({1: zero_y}))
    sol = run([MinCost(), MaxFeatures()], WeightedSum([1, 0]), p)[0]
    assert raw == {"y_f1": 0.0}
    assert vals(sol)["max_features"] == 1.0


def test_same_row_twice_gives_the_same_result() -> None:
    p = P({"id": [1, 2, 3, 4], "cost": [2.0, 3.0, 1.0, 4.0], "status": [0] * 4},
          {"id": [1, 2], "name": ["a", "b"], "target": [3.0, 2.0]},
          {"species": [1, 1, 2, 2], "pu": [1, 2, 3, 4], "amount": [2.0, 2.0, 1.0, 2.0]},
          params={"COSTBUDGET": 5.0})
    specs = [MinShortfall(), MinCost()]
    for approach in (WeightedSum([[1, 1], [1, 1]]), Hierarchical([[0.1], [0.1]])):
        a, b = run(specs, approach, p)
        assert sel(a) == sel(b) and vals(a) == vals(b)


# --- implied evaluation (§7.5) -----------------------------------------------------


def _fixture(kind: str) -> ConservationProblem:
    pu = {"id": [1, 2, 3, 4], "cost": [3.0, 3.0, 1.0, 2.0], "status": [0] * 4}
    feat = {"id": [1, 2], "name": ["a", "b"], "target": [2.0, 3.0]}
    pvf = {"species": [1, 1, 2, 2], "pu": [1, 3, 2, 4], "amount": [1.0, 1.5, 2.0, 2.0]}
    boundary = pd.DataFrame({
        "id1": [1, 1, 2, 3, 3, 4], "id2": [1, 2, 3, 3, 4, 4],
        "boundary": [1.0, 2.0, 1.0, 1.0, 3.0, 2.0],
    })
    params: dict[str, Any] = {"BLM": 0.0, "COSTBUDGET": 6.0}
    kw: dict[str, Any] = {"boundary": boundary}
    if kind == "blm":
        params["BLM"] = 0.5
    elif kind in ("conn_sym_dupes", "conn_asym"):
        kw["connectivity"] = pd.DataFrame({
            "id1": [1, 2, 3, 2], "id2": [2, 1, 4, 3], "value": [10.0, 4.0, 2.0, 1.5],
        })
        params["CONNECTIVITY_WEIGHT"] = 1.0
        params["ASYMMETRIC_CONNECTIVITY"] = int(kind == "conn_asym")
    else:  # probmode1_pos / probmode1_neg
        kw["probability"] = pd.DataFrame({"pu": [1, 2, 3, 4], "probability": [.5, .1, .3, 0]})
        params["PROBMODE"] = 1
        params["PROBABILITYWEIGHTING"] = 0.8 if kind == "probmode1_pos" else -1.0
    return P(pu, feat, pvf, params=params, **kw)


KINDS = ["blm", "conn_sym_dupes", "conn_asym", "probmode1_pos", "probmode1_neg"]
ALL = [MinCost(), MinShortfall(), MinLargestShortfall(), MaxFeatures(), MinBoundary()]


@pytest.mark.parametrize("kind", KINDS)
def test_objective_values_equal_the_model_at_an_optimal_row(
    kind: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw: list[list[float | None]] = []
    real = mo.evaluate_implied

    def spy(model: Any, x: Any, x_rounded: Any, implied: Any, terms: Any) -> Any:
        if len(terms) == len(ALL):
            raw.append([pulp.value(t.expr) for t in terms])
        return real(model, x, x_rounded, implied, terms)

    monkeypatch.setattr(mo, "evaluate_implied", spy)
    sol = run(ALL, WeightedSum([1, 2, 1, 1, 0.5]), _fixture(kind))[0]
    assert sol.metadata["sol_status"] == 1
    assert [vals(sol)[s.key] for s in ALL] == pytest.approx(raw[0], abs=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_min_cost_then_min_shortfall_reaches_stage_2(kind: str) -> None:
    """Round 6: v_rec from the Marxan-style compute_* helpers is unattainable here."""
    pu_status = [2, 2, 0, 0]
    p = _fixture(kind)
    p = p.copy_with(planning_units=p.planning_units.assign(status=pu_status))
    sol = run([MinCost(), MinShortfall()], Hierarchical(0.0), p)[0]
    assert sol.metadata["stage_reached"] == 2


def test_asymmetric_connectivity_locked_endpoints_reach_stage_2() -> None:
    """Round 6 (§7.5): Hierarchical([MinCost(), MinShortfall()]) with asymmetric
    connectivity and two locked-in endpoints."""
    p = P({"id": [1, 2, 3], "cost": [1.0] * 3, "status": [2, 2, 0]},
          {"id": [1], "name": ["a"], "target": [5.0]},
          {"species": [1], "pu": [3], "amount": [5.0]},
          connectivity=pd.DataFrame({"id1": [1], "id2": [2], "value": [10.0]}),
          params={"CONNECTIVITY_WEIGHT": 1.0, "ASYMMETRIC_CONNECTIVITY": 1})
    sol = run([MinCost(), MinShortfall()], Hierarchical(0.0), p)[0]
    assert sol.metadata["stage_reached"] == 2
    assert vals(sol) == {"min_cost": 2.0, "min_shortfall": 1.0}


def test_implied_point_is_feasible_and_attains_the_reported_values() -> None:
    """§5 invariant on oracle data with connectivity and probability: enumerate every plan,
    and the solver's reported weighted value is the enumerated optimum of the model's own
    expressions evaluated at implied points."""
    p = _fixture("conn_sym_dupes")
    p = p.copy_with(
        probability=pd.DataFrame({"pu": [1, 2, 3, 4], "probability": [.5, .1, .3, 0]}),
        parameters={**p.parameters, "PROBMODE": 1, "PROBABILITYWEIGHTING": 0.8},
    )
    specs = [MinCost(), MinShortfall()]
    sol = run(specs, WeightedSum([1, 3]), p)[0]
    # Enumerate with the same machinery: rebuild terms and evaluate each plan.
    from pymarxan.solvers._mip_objectives import evaluate_implied
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    base = MIPSolver()._build_base(p, SolverConfig())
    base.model += base.cost_expr <= 6.0, "cost_budget"
    ctx = base.context()
    terms = [s.build(ctx) for s in specs]
    best = None
    for plan in itertools.product((0.0, 1.0), repeat=4):
        values, feasible = evaluate_implied(
            base.model, base.x, dict(zip([1, 2, 3, 4], plan)), base.implied, terms,
        )
        if feasible:
            score = (values[0] + 3 * values[1]) / 4
            best = score if best is None else min(best, score)
    v = vals(sol)
    assert (v["min_cost"] + 3 * v["min_shortfall"]) / 4 == pytest.approx(best, abs=1e-9)
