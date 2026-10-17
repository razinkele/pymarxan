"""MIP objective builders shared by MIPSolver and ZoneMIPSolver (Stage 1), and the
multi-objective machinery used by ``pymarxan.multiobjective`` (Stage 3).

Each target is one ``TargetTerm`` (prioritizr: one row per target). Builders return an
``ObjectiveTerm`` in the objective's natural units and sense; the solver sets the model
objective (``set_objective``) and adds ``COSTBUDGET`` once. The four target builders work
on any context. ``build_min_cost``, ``build_min_boundary`` and ``build_max_wtd_sum`` are
core-only: they read ``ObjectiveContext`` fields that ``MIPSolver`` fills from
``_build_base`` and that zone contexts leave empty. Imports only pulp, the stdlib and the
stdlib-only ``_tolerance``.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import pulp

from pymarxan.solvers._tolerance import target_tol

SHORTFALL_UNITS = ("relative", "absolute")
OBJECTIVE_ALIASES = {"max_n_targets_met": "max_features"}  # prioritizr >= 9 name
NO_SOLUTION_STATUSES = frozenset(
    {pulp.LpStatusInfeasible, pulp.LpStatusUnbounded, pulp.LpStatusUndefined}
)
# A multi-objective solve is accepted only on these solution statuses (spec §4.1).
ACCEPTED_SOL_STATUSES = frozenset({pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible})


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


@dataclass(frozen=True, eq=False)  # holds an LpAffineExpression (a dict): no __eq__/hash
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


ImpliedRule = tuple[pulp.LpVariable, Callable[[], float]]


def expr_value(expr: pulp.LpAffineExpression) -> float:
    """Value of ``expr`` at the current ``varValue``s, reading ``None`` as 0.

    ``pulp.value`` returns ``None`` when any variable has no value (a zero-cost PU in no
    row is never written to the LP).
    """
    return float(expr.constant or 0.0) + sum(
        float(a) * float(v.varValue or 0.0) for v, a in expr.items()
    )


def _slack_rule(term: TargetTerm, fractional: bool = False) -> Callable[[], float]:
    """s = max(0, R − A); fractional: zeroed when ``<= target_tol(R)`` (Stage 4 §3.6)."""
    if not fractional:
        return lambda: max(0.0, term.required - expr_value(term.achieved))

    def rule() -> float:
        short = max(0.0, term.required - expr_value(term.achieved))
        return 0.0 if short <= target_tol(term.required) else short

    return rule


def _met_rule(term: TargetTerm, fractional: bool = False) -> Callable[[], float]:
    """y = 1 iff A >= R (fractional: A >= R − target_tol(R)): the ``check_targets`` rule."""
    if not fractional:
        return lambda: 1.0 if expr_value(term.achieved) >= term.required else 0.0
    required = term.required - target_tol(term.required)
    return lambda: 1.0 if expr_value(term.achieved) >= required else 0.0


@dataclass
class ObjectiveContext:
    """Shared state for building objectives on one model (spec §2).

    ``implied`` is the implied-value registry (spec §5): each auxiliary variable a builder
    creates is registered with a rule giving its value implied by x, after every variable
    the rule reads. ``MIPSolver`` fills the expression fields from ``_build_base``; zone
    contexts keep the defaults. ``fractional`` (non-binary decisions) selects the tolerant
    met / slack rules (Stage 4 §3.6).
    """

    model: pulp.LpProblem
    cost_expr: pulp.LpAffineExpression
    terms: list[TargetTerm]
    prefix: str = ""
    blm: float = 0.0
    boundary_expr: pulp.LpAffineExpression = field(default_factory=pulp.LpAffineExpression)
    conn_expr: pulp.LpAffineExpression = field(default_factory=pulp.LpAffineExpression)
    prob_expr: pulp.LpAffineExpression = field(default_factory=pulp.LpAffineExpression)
    amount_exprs: dict[int, pulp.LpAffineExpression] = field(default_factory=dict)
    implied: list[ImpliedRule] = field(default_factory=list)
    fractional: bool = False
    _slack: dict[str, pulp.LpVariable] = field(default_factory=dict, repr=False)
    _met: dict[str, pulp.LpVariable] = field(default_factory=dict, repr=False)

    def slack(self, term: TargetTerm) -> pulp.LpVariable:
        """s >= 0 with s >= R - A (constraint ``{prefix}short_{key}``), created once."""
        if term.key not in self._slack:
            s = pulp.LpVariable(f"{self.prefix}s_{term.key}", lowBound=0)
            self.model += s >= term.required - term.achieved, f"{self.prefix}short_{term.key}"
            self._slack[term.key] = s
            self.implied.append((s, _slack_rule(term, self.fractional)))
        return self._slack[term.key]

    def met(self, term: TargetTerm) -> pulp.LpVariable:
        """Binary y with A >= R·y (constraint ``{prefix}met_{key}``), created once."""
        if term.key not in self._met:
            y = pulp.LpVariable(f"{self.prefix}y_{term.key}", cat="Binary")
            self.model += term.achieved >= term.required * y, f"{self.prefix}met_{term.key}"
            self._met[term.key] = y
            self.implied.append((y, _met_rule(term, self.fractional)))
        return self._met[term.key]


def _scale(term: TargetTerm, shortfall: str) -> float:
    return 1.0 / term.required if shortfall == "relative" else 1.0


def _shortfall_terms(ctx: ObjectiveContext) -> list[TargetTerm]:
    # R <= 0 (target 0 or MISSLEVEL 0): nothing to fall short of, and no division by zero.
    return [t for t in ctx.terms if t.required > 0]


def build_max_features(
    ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
) -> ObjectiveTerm:
    return ObjectiveTerm(pulp.lpSum(ctx.met(t) for t in ctx.terms), pulp.LpMaximize)


def build_max_weighted_features(
    ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
) -> ObjectiveTerm:
    return ObjectiveTerm(
        pulp.lpSum(t.weight * ctx.met(t) for t in ctx.terms), pulp.LpMaximize,
    )


def build_min_shortfall(
    ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
) -> ObjectiveTerm:
    """Σ w·s/R (relative, prioritizr) or Σ w·s (absolute)."""
    return ObjectiveTerm(
        pulp.lpSum(
            t.weight * _scale(t, shortfall) * ctx.slack(t) for t in _shortfall_terms(ctx)
        ),
        pulp.LpMinimize,
    )


def build_min_largest_shortfall(
    ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
) -> ObjectiveTerm:
    """m >= s/R (or s) for every term with R > 0; minimise m. Unweighted, as in prioritizr.

    ``m`` is registered in ``ctx.implied`` after the loop, so its rule reads slacks whose
    rules have already run (spec §5).
    """
    m = pulp.LpVariable(f"{ctx.prefix}{owner}max_shortfall", lowBound=0)
    short = _shortfall_terms(ctx)
    for t in short:
        ctx.model += m >= _scale(t, shortfall) * ctx.slack(t), f"{ctx.prefix}{owner}maxs_{t.key}"
    slacks = [(_scale(t, shortfall), ctx.slack(t)) for t in short]
    ctx.implied.append(
        (m, lambda: max((k * float(s.varValue or 0.0) for k, s in slacks), default=0.0))
    )
    return ObjectiveTerm(pulp.lpSum([m]), pulp.LpMinimize)


def build_min_cost(
    ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
) -> ObjectiveTerm:
    """cost + BLM·boundary + connectivity + probability: what ``min_set`` minimises."""
    return ObjectiveTerm(
        ctx.cost_expr + ctx.blm * ctx.boundary_expr + ctx.conn_expr + ctx.prob_expr,
        pulp.LpMinimize,
    )


def build_min_boundary(
    ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
) -> ObjectiveTerm:
    """The boundary term alone, unweighted by BLM (compactness as its own objective)."""
    return ObjectiveTerm(ctx.boundary_expr.copy(), pulp.LpMinimize)


def build_max_wtd_sum(
    ctx: ObjectiveContext,
    shortfall: str,
    *,
    owner: str = "",
    weights: Mapping[int, float] | None = None,
    **params: Any,
) -> ObjectiveTerm:
    """Σ_j w_j·A_j, the weighted amount held (prioritizr ``max_wtd_sum``); w = 1 by default.

    A feature without ``pu_vs_features`` rows contributes 0.
    """
    if weights is None:
        weights = {fid: 1.0 for fid in ctx.amount_exprs}
    expr = pulp.LpAffineExpression()
    for fid, w in weights.items():
        expr += float(w) * ctx.amount_exprs.get(fid, pulp.LpAffineExpression())
    return ObjectiveTerm(expr, pulp.LpMaximize)


class BuilderFn(Protocol):
    def __call__(
        self, ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
    ) -> ObjectiveTerm: ...


BUILDERS: dict[str, BuilderFn] = {
    "max_features": build_max_features,
    "max_weighted_features": build_max_weighted_features,
    "min_shortfall": build_min_shortfall,
    "min_largest_shortfall": build_min_largest_shortfall,
    "min_cost": build_min_cost,
    "min_boundary": build_min_boundary,
    "max_wtd_sum": build_max_wtd_sum,
}


def set_objective(model: pulp.LpProblem, term: ObjectiveTerm) -> None:
    """Models stay LpMinimize; a maximised term is negated."""
    model.sense = pulp.LpMinimize
    model.setObjective(term.expr if term.sense == pulp.LpMinimize else -term.expr)


# MPS writes the RHS with 13 significant figures, so a cap at exactly v can exclude the stage
# optimum itself once |v| >~ 1e5; the floor keeps that rounding inside the cap.
_CAP_REL_FLOOR = 1e-9


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

    Pass 1 minimises ``primary``. Pass 2 adds
    ``primary <= opt + rel_tol·|opt| + max(abs_tol, 1e-9·|opt|)``
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
    slack = rel_tol * abs(opt) + max(abs_tol, _CAP_REL_FLOOR * abs(opt))
    model += primary <= opt + slack, "lex_primary_cap"
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


def constraint_map(model: pulp.LpProblem) -> dict[str, pulp.LpConstraint]:
    """Name -> row of ``model``, without PuLP 3.3's deprecated mapping access.

    PuLP 3.3 warns on every dict-style use of ``LpProblem.constraints`` (PuLP 4.0 makes
    it a method returning a list) and has no public row deletion; PuLP 2.x stores the
    dict on ``constraints`` itself. ``pyproject`` allows ``PuLP>=2.7,<4``.
    """
    rows = getattr(model, "_constraints", None)
    if isinstance(rows, dict):
        return rows
    legacy: dict[str, pulp.LpConstraint] = model.constraints
    return legacy


def _tracked(
    model: pulp.LpProblem, extra: Iterable[pulp.LpVariable]
) -> list[pulp.LpVariable]:
    """``model.variables()`` ∪ ``extra``, deduplicated (spec §5 tracked variable set).

    A PU with zero cost and no rows, and an auxiliary in an empty row, are not in
    ``model.variables()``; callers pass x and the registry as ``extra``.
    """
    seen: dict[int, pulp.LpVariable] = {}
    for v in [*model.variables(), *extra]:
        seen.setdefault(id(v), v)
    return list(seen.values())


def evaluate_implied(
    model: pulp.LpProblem,
    x: Mapping[int, pulp.LpVariable],
    x_rounded: Mapping[int, float],
    implied: Sequence[ImpliedRule],
    terms: Sequence[ObjectiveTerm],
) -> tuple[list[float], bool]:
    """Each term's value at the implied point of ``x_rounded``, and whether that point is
    feasible (spec §5).

    Sets x to ``x_rounded``, applies every implied rule in registry order, reads each
    ``term.expr``, checks **every** model row (``hier_cap_*`` included) with
    ``valid(1e-6)``, and restores every tracked ``varValue``.
    """
    tracked = _tracked(model, [*x.values(), *(v for v, _ in implied)])
    snapshot = [(v, v.varValue) for v in tracked]
    try:
        for pid, var in x.items():
            var.varValue = float(x_rounded[pid])
        for var, rule in implied:
            var.varValue = rule()
        values = [expr_value(t.expr) for t in terms]
        feasible = all(c.valid(1e-6) for c in constraint_map(model).values())
    finally:
        for var, value in snapshot:
            var.varValue = value
    return values, feasible


@dataclass(frozen=True)
class HierarchicalResult:
    """``status`` / ``sol_status`` of the last accepted stage; ``stage_reached`` = accepted
    stages (n on full success, 0 when stage 1 has no solution)."""

    status: int
    sol_status: int
    stage_reached: int


def solve_hierarchical(
    model: pulp.LpProblem,
    terms: Sequence[ObjectiveTerm],
    rel_tols: Sequence[float],
    abs_tol: float,
    solver: pulp.LpSolver,
    evaluate: Callable[[int], float | None],
    *,
    tracked: Iterable[pulp.LpVariable],
) -> HierarchicalResult:
    """Lexicographic optimisation of ``terms`` in list order (spec §4.1).

    Stage k < n is capped (row ``hier_cap_{k}``) at ``f ≤ v + r_k·|v| + s``
    (minimise) or ``f ≥ v − (r_k·|v| + s)`` (maximise), ``r_k = rel_tols[k-1]`` and
    ``s = max(abs_tol, 1e-9·|v|)``.
    ``v`` is the better, in the stage's sense, of the model value of ``term.expr`` and
    ``evaluate(k-1)`` (the implied value, ``None`` when the implied point is infeasible).
    Values are reset to ``None`` before each solve, and a solve is accepted only when
    ``model.sol_status`` is 1 or 2. On a failed stage the previous stage's values are
    restored and the loop stops. Cap rows are removed before returning, also on failure.
    ``tracked`` adds variables missing from ``model.variables()`` (x, the registry) to the
    reset / snapshot / restore. Raises nothing itself.
    """
    extra = list(tracked)
    caps: list[str] = []
    snapshot: list[tuple[pulp.LpVariable, float | None]] | None = None
    accepted = HierarchicalResult(0, 0, 0)
    try:
        for k, term in enumerate(terms, start=1):
            variables = _tracked(model, extra)
            for var in variables:
                var.varValue = None
            set_objective(model, term)
            model.solve(solver)
            if model.sol_status not in ACCEPTED_SOL_STATUSES:
                if snapshot is None:
                    return HierarchicalResult(int(model.status), int(model.sol_status), 0)
                for var in variables:
                    var.varValue = None
                for var, value in snapshot:
                    var.varValue = value
                return accepted
            snapshot = [(var, var.varValue) for var in _tracked(model, extra)]
            accepted = HierarchicalResult(int(model.status), int(model.sol_status), k)
            if k == len(terms):
                break
            v_model = expr_value(term.expr)
            v_rec = evaluate(k - 1)
            if v_rec is None:
                v = v_model
            elif term.sense == pulp.LpMinimize:
                v = min(v_model, v_rec)
            else:
                v = max(v_model, v_rec)
            slack = float(rel_tols[k - 1]) * abs(v) + max(abs_tol, _CAP_REL_FLOOR * abs(v))
            name = f"hier_cap_{k}"
            if term.sense == pulp.LpMinimize:
                model += term.expr <= v + slack, name
            else:
                model += term.expr >= v - slack, name
            caps.append(name)
        return accepted
    finally:
        for name in caps:
            del constraint_map(model)[name]
