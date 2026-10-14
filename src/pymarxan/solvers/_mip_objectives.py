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
    A variable-free (constant) ``primary`` — no target terms — is a single solve of
    ``secondary``.
    """
    model.sense = pulp.LpMinimize
    if not primary.keys():
        model.setObjective(secondary)
        model.solve(solver)
        return int(model.status)
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
        model.status = status1
        warnings.warn(
            "lexicographic pass 2 (minimise cost at the optimal penalty) found no solution; "
            "returning the pass-1 solution",
            UserWarning,
            stacklevel=3,
        )
        return status1
    return int(model.status)
