"""evaluate_implied and solve_hierarchical (Stage 3 spec §4.1, §5, §7.4, §7.5).

Ports the spec-review cases: round 4 (tolerance boundary), round 5 (incumbent), round 6
(near-integral budget, asymmetric connectivity with locked endpoints), round 7 (stale slacks,
cap contradiction) and the status snapshot after a failed stage.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd
import pulp
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers._mip_objectives import (
    HierarchicalResult,
    ObjectiveContext,
    ObjectiveTerm,
    build_max_features,
    build_max_weighted_features,
    build_min_cost,
    build_min_largest_shortfall,
    build_min_shortfall,
    constraint_map,
    evaluate_implied,
    expr_value,
    solve_hierarchical,
)
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver, _BaseModel
from tests.pymarxan.scripted_solver import (
    Scripted,
    fail,
    not_solved_with_incumbent,
    optimal_without_solution,
)

Builder = Callable[..., ObjectiveTerm]


def P(pu: dict[str, list[Any]], feat: dict[str, list[Any]], pvf: dict[str, list[Any]],
      **kw: Any) -> ConservationProblem:
    params = kw.pop("params", {})
    return ConservationProblem(
        planning_units=pd.DataFrame(pu), features=pd.DataFrame(feat),
        pu_vs_features=pd.DataFrame(pvf), parameters=params, **kw,
    )


def build_case(
    problem: ConservationProblem, builders: list[Builder]
) -> tuple[_BaseModel, ObjectiveContext, list[ObjectiveTerm]]:
    base = MIPSolver()._build_base(problem, SolverConfig(), need_boundary=True)
    if "COSTBUDGET" in problem.parameters:
        base.model += base.cost_expr <= float(problem.parameters["COSTBUDGET"]), "cost_budget"
    ctx = base.context()
    return base, ctx, [b(ctx, "relative") for b in builders]


def hier(
    base: _BaseModel, terms: list[ObjectiveTerm], rel_tols: list[float], *,
    solver: pulp.LpSolver | None = None, abs_tol: float = 1e-9,
) -> HierarchicalResult:
    def evaluate(i: int) -> float | None:
        rounded = {pid: float(round(v.varValue or 0.0)) for pid, v in base.x.items()}
        vals, feasible = evaluate_implied(base.model, base.x, rounded, base.implied, [terms[i]])
        return vals[0] if feasible else None

    tracked = [*base.x.values(), *(v for v, _ in base.implied)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return solve_hierarchical(
            base.model, terms, rel_tols, abs_tol, solver or Scripted(), evaluate,
            tracked=tracked,
        )


def xs(base: _BaseModel) -> list[int]:
    return [int(round(v.varValue or 0.0)) for v in base.x.values()]


def caps(model: pulp.LpProblem) -> list[str]:
    return [n for n in constraint_map(model) if n.startswith("hier_cap_")]


TOL = P(  # round 4: 0.7 + 0.1 = 0.7999999999999999 < 0.8, but CBC calls the target met
    {"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
    {"id": [1], "name": ["a"], "target": [0.8]},
    {"species": [1, 1], "pu": [1, 2], "amount": [0.7, 0.1]},
)


# --- evaluate_implied -------------------------------------------------------------


def test_evaluate_implied_values_feasibility_and_restore() -> None:
    p = P({"id": [1, 2, 3], "cost": [1.0, 2.0, 0.0], "status": [0, 0, 0]},
          {"id": [1, 2], "name": ["a", "b"], "target": [2.0, 4.0]},
          {"species": [1, 2], "pu": [1, 2], "amount": [2.0, 4.0]})
    base, ctx, terms = build_case(p, [build_min_cost, build_min_largest_shortfall,
                                 build_max_features])
    assert "x_3" not in {v.name for v in base.model.variables()}  # zero cost, no rows
    for v, _ in base.implied:
        v.varValue = 9.0  # stale
    base.x[1].varValue, base.x[2].varValue, base.x[3].varValue = 0.9999999, 0.0, None
    before = {v.name: v.varValue for v in [*base.x.values(), *(v for v, _ in base.implied)]}
    vals, feasible = evaluate_implied(
        base.model, base.x, {1: 1.0, 2: 0.0, 3: 0.0}, base.implied, terms,
    )
    assert feasible
    assert vals == [1.0, 1.0, 1.0]  # cost 1; f2 fully short (relative 1.0); f1 met
    after = {v.name: v.varValue for v in [*base.x.values(), *(v for v, _ in base.implied)]}
    assert after == before  # every tracked value restored, x_3 still None


def test_evaluate_implied_infeasible_when_rounding_breaks_the_budget() -> None:
    # round 6: x1 = 1 − 5e-8 fits budget 1e7·(1 − 5e-8); x1 rounded to 1 does not.
    c1, eps = 1e7, 5e-8
    p = P({"id": [1, 2], "cost": [c1, 1.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [10.0]},
          {"species": [1], "pu": [1], "amount": [10.0]},
          params={"COSTBUDGET": c1 * (1 - eps)})
    base, _, terms = build_case(p, [build_max_features])
    vals, feasible = evaluate_implied(base.model, base.x, {1: 1.0, 2: 0.0}, base.implied, terms)
    assert vals == [1.0] and not feasible


def test_evaluate_implied_m_reads_fresh_slacks() -> None:
    # round 7 C5: stale slacks s = 3, m = 5; at x = (1, 1) the true largest shortfall is 0.
    p = P({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
          {"id": [1, 2], "name": ["a", "b"], "target": [2.0, 4.0]},
          {"species": [1, 2], "pu": [1, 2], "amount": [2.0, 4.0]})
    base, _, terms = build_case(p, [build_min_cost, build_min_largest_shortfall])
    for v, _ in base.implied:
        v.varValue = 3.0 if v.name.startswith("s_") else 5.0
    vals, _ = evaluate_implied(base.model, base.x, {1: 1.0, 2: 1.0}, base.implied, terms)
    assert vals == [2.0, 0.0]
    vals, _ = evaluate_implied(base.model, base.x, {1: 1.0, 2: 0.0}, base.implied, terms)
    assert vals == [1.0, 1.0]


# --- solve_hierarchical: the cap ---------------------------------------------------


@pytest.mark.parametrize("rel_tol", [0.0, 0.5])
def test_min_cap_is_v_plus_r_abs_v_plus_abs_tol(rel_tol: float) -> None:
    # Stage 1 min MinCost with a cost of -4 locked in: v = -4 < 0, so |v| matters.
    p = P({"id": [1, 2], "cost": [-4.0, 1.0], "status": [2, 0]},
          {"id": [1], "name": ["a"], "target": [1.0]},
          {"species": [1], "pu": [2], "amount": [1.0]})
    base, _, terms = build_case(p, [build_min_cost, build_min_shortfall])
    solver = Scripted()
    res = hier(base, terms, [rel_tol], solver=solver, abs_tol=0.25)
    assert res == HierarchicalResult(pulp.LpStatusOptimal, 1, 2)
    sense, rhs = solver.models[1]["hier_cap_1"]
    assert sense == pulp.LpConstraintLE
    assert rhs == pytest.approx(-4.0 + rel_tol * 4.0 + 0.25)
    assert caps(base.model) == []  # removed after the run


def test_max_cap_is_v_minus_r_abs_v_minus_abs_tol() -> None:
    p = P({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]},
          {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0]},
          {"species": [1, 2], "pu": [1, 2], "amount": [1.0, 1.0]})
    base, _, terms = build_case(p, [build_max_features, build_min_cost])
    solver = Scripted()
    hier(base, terms, [0.25], solver=solver, abs_tol=0.1)
    sense, rhs = solver.models[1]["hier_cap_1"]
    assert sense == pulp.LpConstraintGE
    assert rhs == pytest.approx(2.0 - (0.25 * 2.0 + 0.1))
    assert xs(base) == [1, 1, 0]  # Σ y >= 1.4 needs both targets: PU1 and PU2


def test_rel_tol_is_indexed_by_stage() -> None:
    p = P({"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0]},
          {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0]},
          {"species": [1, 2], "pu": [1, 2], "amount": [1.0, 1.0]})
    base, _, terms = build_case(p, [build_max_features, build_min_shortfall, build_min_cost])
    solver = Scripted()
    hier(base, terms, [0.5, 0.0], solver=solver, abs_tol=0.0)
    assert solver.models[2]["hier_cap_1"][1] == pytest.approx(2.0 - 0.5 * 2.0)
    assert solver.models[2]["hier_cap_2"][1] == pytest.approx(0.0)


# --- solve_hierarchical: v = better of v_model and v_rec ------------------------------


def test_tolerance_boundary_cap_keeps_the_model_target() -> None:
    # round 4 (§7.4): v_model = 1, v_rec = 0. Recompute-only caps at 0 and returns the
    # empty plan; best-of-both keeps the target and costs 2.
    base, _, terms = build_case(TOL, [build_max_features, build_min_cost])
    res = hier(base, terms, [0.0])
    assert res.stage_reached == 2
    assert xs(base) == [1, 1]


def test_incumbent_cap_uses_the_recomputed_value() -> None:
    # round 5 (§7.4): a stage-1 incumbent (sol_status 2) meets both targets but leaves
    # y_f2 = 0 (v_model = 1, v_rec = 2). The cap must use 2, so stage 2 keeps both targets.
    p = P({"id": [1, 2, 3], "cost": [1.0, 1.0, 5.0], "status": [0, 0, 0]},
          {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0]},
          {"species": [1, 2, 1, 2], "pu": [1, 2, 3, 3], "amount": [1.0, 1.0, 1.0, 1.0]},
          params={"COSTBUDGET": 5.0})
    base, _, terms = build_case(p, [build_max_features, build_min_cost])

    def loose(lp: pulp.LpProblem) -> None:
        lp.sol_status = pulp.LpSolutionIntegerFeasible
        values = {"x_1": 1.0, "x_2": 1.0, "x_3": 0.0, "y_f1": 1.0, "y_f2": 0.0}
        for v in lp.variables():
            if v.name in values:
                v.varValue = values[v.name]

    solver = Scripted({1: loose})
    res = hier(base, terms, [0.0], solver=solver)
    assert solver.models[1]["hier_cap_1"][1] == pytest.approx(2.0 - 1e-9)
    assert res.stage_reached == 2
    assert xs(base) == [1, 1, 0]  # both targets kept at cost 2


def test_near_integral_budget_falls_back_to_the_model_value() -> None:
    # round 6 (§7.5): rounding x1 = 1 − 5e-8 breaks COSTBUDGET, so v = v_model = 0.
    c1, eps = 1e7, 5e-8
    p = P({"id": [1, 2], "cost": [c1, 1.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [10.0]},
          {"species": [1], "pu": [1], "amount": [10.0]},
          params={"COSTBUDGET": c1 * (1 - eps)})
    base, _, terms = build_case(p, [build_max_features, build_min_cost])

    def near(lp: pulp.LpProblem) -> None:
        lp.sol_status = pulp.LpSolutionIntegerFeasible
        values = {"x_1": 1 - eps, "x_2": 0.0, "y_f1": 0.0}
        for v in lp.variables():
            if v.name in values:
                v.varValue = values[v.name]

    solver = Scripted({1: near})
    res = hier(base, terms, [0.0], solver=solver)
    assert solver.models[1]["hier_cap_1"][1] == pytest.approx(-1e-9)
    assert res.stage_reached == 2


def test_earlier_caps_are_checked_three_stages() -> None:
    # round 7 (§7.5): negative SPF on tolerance-boundary amounts. Skipping hier_cap_* rows
    # in the feasibility check lets cap 2 contradict cap 1 (stage_reached 2).
    p = P({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [0.8], "spf": [-1.0]},
          {"species": [1, 1], "pu": [1, 2], "amount": [0.7, 0.1]})
    base, _, terms = build_case(p, [build_max_features, build_max_weighted_features,
                               build_min_cost])
    res = hier(base, terms, [0.0, 0.0])
    assert res.stage_reached == 3


@pytest.mark.parametrize(("asym", "weighting"), [(1, None), (0, None), (0, -1.0), (0, 0.7)])
def test_connectivity_with_locked_endpoints_reaches_stage_2(
    asym: int, weighting: float | None,
) -> None:
    # round 6 (§7.5): v_rec is the model's own expression at the implied point.
    params: dict[str, Any] = {"CONNECTIVITY_WEIGHT": 1.0, "ASYMMETRIC_CONNECTIVITY": asym}
    kw: dict[str, Any] = {}
    if weighting is not None:
        params["PROBABILITYWEIGHTING"] = weighting
        kw["probability"] = pd.DataFrame({"pu": [1, 2, 3], "probability": [0.5] * 3})
    p = P({"id": [1, 2, 3], "cost": [1.0] * 3, "status": [2, 2, 0]},
          {"id": [1], "name": ["a"], "target": [5.0]},
          {"species": [1], "pu": [3], "amount": [5.0]},
          connectivity=pd.DataFrame({"id1": [1], "id2": [2], "value": [10.0]}),
          params=params, **kw)
    base, _, terms = build_case(p, [build_min_cost, build_min_shortfall])
    assert hier(base, terms, [0.0]).stage_reached == 2


def test_zero_cost_pu_in_no_row_does_not_break_v() -> None:
    p = P({"id": [1, 2], "cost": [1.0, 0.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [1.0]},
          {"species": [1], "pu": [1], "amount": [1.0]})
    base, _, terms = build_case(p, [build_min_cost, build_min_shortfall])
    res = hier(base, terms, [0.0])
    assert res.stage_reached == 2 and xs(base) == [0, 0]
    assert base.x[2].varValue is None


# --- solve_hierarchical: failures --------------------------------------------------


def test_stage_2_failure_restores_stage_1_and_its_status() -> None:
    # Budget 1: the best relative shortfall is 0.5, from PU1 only (PU2 costs 2). The failed
    # stage 2 leaves every x at 1 (shortfall 0): only a restore gives back 0.5 and [1, 0].
    p = P({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [2.0]},
          {"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]},
          params={"COSTBUDGET": 1.0})
    base, _, terms = build_case(p, [build_min_shortfall, build_min_cost])
    res = hier(base, terms, [0.0], solver=Scripted({2: fail}))
    assert res == HierarchicalResult(pulp.LpStatusOptimal, pulp.LpSolutionOptimal, 1)
    assert xs(base) == [1, 0]
    assert expr_value(terms[0].expr) == pytest.approx(0.5)
    assert caps(base.model) == []


def test_acceptance_follows_sol_status_not_status() -> None:
    p = P({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [2.0]},
          {"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]},
          params={"COSTBUDGET": 1.0})
    base, _, terms = build_case(p, [build_min_shortfall, build_min_cost])
    # status Optimal, sol_status NoSolutionFound: no solution.
    res = hier(base, terms, [0.0], solver=Scripted({2: optimal_without_solution}))
    assert res.stage_reached == 1
    # status NotSolved, sol_status IntegerFeasible (a time-limit incumbent): accepted.
    res = hier(base, terms, [0.0], solver=Scripted({2: not_solved_with_incumbent}))
    assert res == HierarchicalResult(pulp.LpStatusNotSolved, pulp.LpSolutionIntegerFeasible, 2)


def test_stage_1_failure_reports_stage_0() -> None:
    base, _, terms = build_case(TOL, [build_min_cost, build_min_shortfall])
    res = hier(base, terms, [0.0], solver=Scripted({1: fail}))
    assert res.stage_reached == 0 and caps(base.model) == []


def test_values_are_reset_before_each_solve() -> None:
    # A failed stage 2 that leaves no values must not see stage-1 values as a solution.
    base, _, terms = build_case(TOL, [build_min_cost, build_min_shortfall])
    seen: list[set[Any]] = []

    def record(lp: pulp.LpProblem) -> None:
        seen.append({v.varValue for v in lp.variables()})
        fail(lp)

    class Blank(Scripted):
        def actualSolve(self, lp: pulp.LpProblem, **kwargs: Any) -> int:  # noqa: N802
            if self.calls == 0:
                return super().actualSolve(lp)
            self.calls += 1
            record(lp)  # no inner solve: values are whatever the helper left
            return int(lp.status)

    res = hier(base, terms, [0.0], solver=Blank())
    assert seen == [{None}]
    assert res.stage_reached == 1
