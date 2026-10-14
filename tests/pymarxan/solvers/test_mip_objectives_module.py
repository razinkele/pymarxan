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
    assert (x1.varValue, x2.varValue) == (1.0, 0.0)
    assert s.varValue == pytest.approx(0.0, abs=1e-9)  # free in [0, 1e-9] under the cap
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
