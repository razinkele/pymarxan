"""Stage 3 builder protocol (spec §2): context fields, owner names, implied registry, builders."""
from __future__ import annotations

import pulp
import pytest

from pymarxan.solvers._mip_objectives import (
    BUILDERS,
    ObjectiveContext,
    TargetTerm,
    build_max_wtd_sum,
    build_min_boundary,
    build_min_cost,
    build_min_largest_shortfall,
)


def _coefs(expr: pulp.LpAffineExpression) -> dict[str, float]:
    return {v.name: float(a) for v, a in expr.items()}


def _ctx(**kw: object) -> tuple[ObjectiveContext, dict[int, pulp.LpVariable]]:
    m = pulp.LpProblem("p", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in (1, 2)}
    terms = [
        TargetTerm("f1", "feature", pulp.lpSum([2.0 * x[1]]), 2.0, 1.0),
        TargetTerm("f2", "feature", pulp.lpSum([4.0 * x[2]]), 4.0, 1.0),
    ]
    return ObjectiveContext(m, pulp.lpSum([x[1], 3 * x[2]]), terms, **kw), x  # type: ignore[arg-type]


def _apply(ctx: ObjectiveContext) -> None:
    for var, rule in ctx.implied:
        var.varValue = rule()


def test_context_defaults_keep_the_zone_call() -> None:
    a, _ = _ctx()
    b, _ = _ctx()
    assert a.blm == 0.0
    for expr in (a.boundary_expr, a.conn_expr, a.prob_expr):
        assert isinstance(expr, pulp.LpAffineExpression) and not expr.items()
    assert a.amount_exprs == {} and a.implied == []
    assert a.implied is not b.implied and a.boundary_expr is not b.boundary_expr


def test_slack_and_met_register_implied_rules_on_creation() -> None:
    ctx, x = _ctx()
    t1, t2 = ctx.terms
    s1, y2 = ctx.slack(t1), ctx.met(t2)
    ctx.slack(t1)  # created once: registered once
    assert [v.name for v, _ in ctx.implied] == ["s_f1", "y_f2"]
    x[1].varValue, x[2].varValue = 0.0, 1.0
    s1.varValue, y2.varValue = 7.0, 0.0  # stale solver values
    _apply(ctx)
    assert (s1.varValue, y2.varValue) == (2.0, 1.0)  # max(0, R - A); A >= R


def test_met_rule_is_exact_at_the_target() -> None:
    m = pulp.LpProblem("e", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in (1, 2)}
    t = TargetTerm("f1", "feature", pulp.lpSum([0.7 * x[1], 0.1 * x[2]]), 0.8, 1.0)
    ctx = ObjectiveContext(m, pulp.lpSum(x.values()), [t])
    y = ctx.met(t)
    x[1].varValue = x[2].varValue = 1.0  # 0.7 + 0.1 = 0.7999999999999999 < 0.8
    _apply(ctx)
    assert y.varValue == 0.0


def test_min_largest_shortfall_registers_m_after_its_slacks() -> None:
    ctx, x = _ctx()
    term = build_min_largest_shortfall(ctx, "relative")
    assert [v.name for v, _ in ctx.implied] == ["s_f1", "s_f2", "max_shortfall"]
    # Stale solver values: s = 3, m = 5. True values at x = (1, 0): s = (0, 4), m = 4/4 = 1.
    x[1].varValue, x[2].varValue = 1.0, 0.0
    for v, _ in ctx.implied:
        v.varValue = 3.0 if v.name.startswith("s_") else 5.0
    _apply(ctx)
    assert pulp.value(term.expr) == 1.0


def test_min_largest_shortfall_without_positive_targets_is_zero() -> None:
    m = pulp.LpProblem("z", pulp.LpMinimize)
    x = pulp.LpVariable("x_1", cat="Binary")
    ctx = ObjectiveContext(m, pulp.lpSum([x]), [TargetTerm("f0", "feature", 1 * x, 0.0, 1.0)])
    term = build_min_largest_shortfall(ctx, "absolute")
    _apply(ctx)
    assert pulp.value(term.expr) == 0.0


def test_owner_prefixes_owned_names_and_shares_slacks() -> None:
    ctx, _ = _ctx(prefix="p_")
    build_min_largest_shortfall(ctx, "relative", owner="a__")
    build_min_largest_shortfall(ctx, "absolute", owner="b__")
    names = set(ctx.model.constraints)
    assert {"p_a__maxs_f1", "p_b__maxs_f1", "p_short_f1", "p_short_f2"} <= names
    var_names = [v.name for v, _ in ctx.implied]
    assert var_names == ["p_s_f1", "p_s_f2", "p_a__max_shortfall", "p_b__max_shortfall"]


def test_default_owner_keeps_todays_names() -> None:
    ctx, _ = _ctx()
    BUILDERS["min_largest_shortfall"](ctx, "absolute")
    assert {"maxs_f1", "maxs_f2"} <= set(ctx.model.constraints)
    assert ctx.implied[-1][0].name == "max_shortfall"


def test_build_min_cost_is_min_set_objective() -> None:
    ctx, x = _ctx(blm=2.0)
    ctx.boundary_expr += 5.0 * x[1]
    ctx.conn_expr += -1.5 * x[2]
    ctx.prob_expr += 0.25 * x[1]
    term = build_min_cost(ctx, "relative")
    assert term.sense == pulp.LpMinimize
    assert _coefs(term.expr) == {"x_1": 1.0 + 10.0 + 0.25, "x_2": 3.0 - 1.5}


def test_build_min_boundary_ignores_blm() -> None:
    ctx, x = _ctx(blm=7.0)
    ctx.boundary_expr += 5.0 * x[1]
    term = build_min_boundary(ctx, "relative", owner="mb__")
    assert term.sense == pulp.LpMinimize
    assert _coefs(term.expr) == {"x_1": 5.0}
    ctx.boundary_expr += 1.0 * x[2]  # the term is a copy, not an alias
    assert _coefs(term.expr) == {"x_1": 5.0}


def test_build_max_wtd_sum_weights_and_ghosts() -> None:
    ctx, x = _ctx()
    ctx.amount_exprs = {1: pulp.lpSum([2.0 * x[1]]), 2: pulp.lpSum([4.0 * x[2]])}
    plain = build_max_wtd_sum(ctx, "relative")
    assert plain.sense == pulp.LpMaximize
    assert _coefs(plain.expr) == {"x_1": 2.0, "x_2": 4.0}
    # Feature 9 has no rows (ghost): it contributes 0, not a KeyError.
    weighted = build_max_wtd_sum(ctx, "relative", owner="w__", weights={1: -1.0, 2: 0.5, 9: 3.0})
    assert _coefs(weighted.expr) == {"x_1": -2.0, "x_2": 2.0}
    assert len(ctx.model.constraints) == 0  # owns no rows


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_builder_takes_owner(name: str) -> None:
    ctx, _ = _ctx()
    term = BUILDERS[name](ctx, "relative", owner=f"{name}__")
    assert isinstance(term.expr, pulp.LpAffineExpression)
    assert all(
        n.startswith(f"{name}__") or n.startswith(("short_", "met_"))
        for n in ctx.model.constraints
    )


def test_registry_has_the_stage3_builders() -> None:
    assert {"min_cost", "min_boundary", "max_wtd_sum"} <= set(BUILDERS)
