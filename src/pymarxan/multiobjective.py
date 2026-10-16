"""Exact multi-objective planning on one ``ConservationProblem`` (Stage 3, prioritizr 9 parity).

Objectives are specs (``MinCost``, ``MinBoundary``, ``MinShortfall``, ``MinLargestShortfall``,
``MaxFeatures``, ``MaxWeightedFeatures``, ``MaxWtdSum``) combined by an approach:

- ``Hierarchical(rel_tol, abs_tol=1e-9)``: lexicographic, in list order. Each stage caps the
  objective just solved at ``v + r·|v| + s`` (minimise) or ``v − (r·|v| + s)``
  (maximise) and solves the next, with ``s = max(abs_tol, 1e-9·|v|)`` (the relative floor
  absorbs MPS rounding at large |v|). At ``v = 0`` only ``abs_tol`` is allowed.
- ``WeightedSum(weights)``: one solve of ``Σ (w/Σw)·s·f`` per weight row (s = −1 for a
  maximised objective). Objectives are **not** scaled: objectives in different units need
  weights scaled to their magnitudes (Das & Dennis 1997,
  https://doi.org/10.1007/BF01197559).

Targets are soft (only representation objectives use them) and ``COSTBUDGET`` is optional.
``weights_matrix`` / ``rel_tol_matrix`` build prioritizr's parameter sweeps and
``nondominated`` filters the resulting solutions. Deviations from prioritizr 9.0.1 are listed
in the CHANGELOG for v0.41.0 (Hanson et al. 2024, https://doi.org/10.1111/cobi.14376;
Williams & Kendall 2017, https://doi.org/10.1016/j.ecolmodel.2016.10.010).
"""
from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any, ClassVar

import numpy as np
import pulp

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers._mip_objectives import (
    ACCEPTED_SOL_STATUSES,
    BUILDERS,
    ObjectiveContext,
    ObjectiveTerm,
    evaluate_implied,
    set_objective,
    solve_hierarchical,
    validate_shortfall,
)
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver, _mip_solver_for, _warn_if_large_mip
from pymarxan.solvers.presolve import _emit_presolve, _presolve_data
from pymarxan.solvers.utils import build_solution

__all__ = [
    "Hierarchical",
    "MaxFeatures",
    "MaxWeightedFeatures",
    "MaxWtdSum",
    "MinBoundary",
    "MinCost",
    "MinLargestShortfall",
    "MinShortfall",
    "MultiObjectiveMIP",
    "ObjectiveSpec",
    "WeightedSum",
    "nondominated",
    "rel_tol_matrix",
    "weights_matrix",
]


# --- objective specs -------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveSpec:
    """One objective. ``key`` (``label or name``) names its reported value and owns its rows.

    ``label`` must be a Python identifier: PuLP rewrites other characters in names, so two
    distinct labels could collide.
    """

    name: ClassVar[str] = ""
    sense: ClassVar[str] = "min"
    label: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.label is not None and not (
            isinstance(self.label, str) and self.label.isidentifier()
        ):
            raise ValueError(f"label must be a Python identifier, got {self.label!r}")

    @property
    def key(self) -> str:
        return self.label or self.name

    def _shortfall(self) -> str:
        return "relative"

    def _params(self) -> dict[str, Any]:
        return {}

    def build(self, ctx: ObjectiveContext) -> ObjectiveTerm:
        """The builder's term; owned names are prefixed ``f"{key}__"``. Call once per ctx."""
        return BUILDERS[self.name](
            ctx, self._shortfall(), owner=f"{self.key}__", **self._params(),
        )


@dataclass(frozen=True)
class MinCost(ObjectiveSpec):
    """cost + BLM·boundary + connectivity + probability penalty (what ``min_set`` minimises).

    With BLM > 0 it already counts boundary, so a weighted sum with ``MinBoundary`` counts
    boundary twice.
    """

    name: ClassVar[str] = "min_cost"
    sense: ClassVar[str] = "min"


@dataclass(frozen=True)
class MinBoundary(ObjectiveSpec):
    """The boundary length alone, unweighted by BLM. Needs a boundary table."""

    name: ClassVar[str] = "min_boundary"
    sense: ClassVar[str] = "min"


@dataclass(frozen=True)
class MinShortfall(ObjectiveSpec):
    """Σ SPF·shortfall, relative (÷ target × MISSLEVEL, prioritizr) or absolute."""

    name: ClassVar[str] = "min_shortfall"
    sense: ClassVar[str] = "min"
    shortfall: str = "relative"

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_shortfall(self.shortfall)

    def _shortfall(self) -> str:
        return self.shortfall


@dataclass(frozen=True)
class MinLargestShortfall(ObjectiveSpec):
    """The largest (relative or absolute) shortfall over targets, unweighted."""

    name: ClassVar[str] = "min_largest_shortfall"
    sense: ClassVar[str] = "min"
    shortfall: str = "relative"

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_shortfall(self.shortfall)

    def _shortfall(self) -> str:
        return self.shortfall


@dataclass(frozen=True)
class MaxFeatures(ObjectiveSpec):
    """Count of targets met (features with target > 0)."""

    name: ClassVar[str] = "max_features"
    sense: ClassVar[str] = "max"


@dataclass(frozen=True)
class MaxWeightedFeatures(ObjectiveSpec):
    """SPF-weighted count of targets met."""

    name: ClassVar[str] = "max_weighted_features"
    sense: ClassVar[str] = "max"


def _feature_id(k: object) -> int:
    """An int, a numpy integer or an integral float; never a bool or str."""
    if isinstance(k, numbers.Integral) and not isinstance(k, bool):
        return int(k)
    if isinstance(k, numbers.Real) and not isinstance(k, bool) and float(k).is_integer():
        return int(float(k))
    raise ValueError(f"weights keys must be integral feature ids, got {k!r}")


@dataclass(frozen=True)
class MaxWtdSum(ObjectiveSpec):
    """Σ w_j · amount held_j (prioritizr ``max_wtd_sum``).

    ``weights``: ``None`` (every feature 1) or ``{feature_id: w}`` covering every feature in
    the problem (finite, any sign); coverage is checked at ``solve``.

    Caution (as prioritizr notes for its weighted-sum objectives): a weighted sum of amounts
    ignores complementarity. More of an already well-held feature scores as much as the
    same amount of a missing one, so the plan can leave features unrepresented; pair it
    with a shortfall objective when representation matters.
    """

    name: ClassVar[str] = "max_wtd_sum"
    sense: ClassVar[str] = "max"
    weights: Mapping[int, float] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.weights is not None:
            bad = sorted(k for k, w in self.weights.items() if not _finite(w))
            if bad:
                raise ValueError(f"weights must be finite numbers; bad feature ids {bad}")
            keys = [_feature_id(k) for k in self.weights]
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            if dupes:
                raise ValueError(f"weights has duplicate feature ids after coercion: {dupes}")
            object.__setattr__(
                self, "weights",
                {k: float(w) for k, w in zip(keys, self.weights.values(), strict=True)},
            )

    def _params(self) -> dict[str, Any]:
        return {"weights": self.weights}


# --- approaches -----------------------------------------------------------------


def _finite(v: object) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def _nonneg(name: str, v: object) -> float:
    if not _finite(v) or float(v) < 0:  # type: ignore[arg-type]
        raise ValueError(f"{name} values must be finite and >= 0, got {v!r}")
    return float(v)  # type: ignore[arg-type]


def _is_scalar(x: object) -> bool:
    return (isinstance(x, numbers.Real) and not isinstance(x, bool)) or (
        getattr(x, "ndim", None) == 0
    )


def _as_rows(name: str, value: object) -> list[list[float]]:
    """A float -> one row of one value; a 1-D sequence -> one row; a 2-D -> its rows."""
    if hasattr(value, "tolist"):  # numpy array or scalar
        value = value.tolist()
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return [[_nonneg(name, value)]]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{name} must be a number, a sequence or a 2-D sequence")
    items = list(value)
    if all(isinstance(v, Sequence) and not isinstance(v, str) for v in items) and items:
        return [[_nonneg(name, v) for v in row] for row in items]
    return [[_nonneg(name, v) for v in items]]


@dataclass(frozen=True)
class Hierarchical:
    """Lexicographic optimisation in list order (spec §4.1).

    ``rel_tol``: a float (every stage), n−1 floats, or a 2-D sequence (one row of n−1 per
    run); finite, >= 0, values > 1 allowed. The k-th tolerance caps the k-th stage.
    """

    rel_tol: float | Sequence[float] | Sequence[Sequence[float]]
    abs_tol: float = 1e-9

    def __post_init__(self) -> None:
        _as_rows("rel_tol", self.rel_tol)
        _nonneg("abs_tol", self.abs_tol)

    def rows(self, n: int) -> list[list[float]]:
        """One rel_tol row of n−1 values per run."""
        if _is_scalar(self.rel_tol):
            return [[float(self.rel_tol)] * (n - 1)]  # type: ignore[arg-type]
        rows = _as_rows("rel_tol", self.rel_tol)
        for i, row in enumerate(rows):
            if len(row) != n - 1:
                raise ValueError(
                    f"rel_tol row {i} has {len(row)} values; {n} objectives need {n - 1}"
                )
        return rows


@dataclass(frozen=True)
class WeightedSum:
    """One solve of ``Σ (w/Σw)·s·f`` per row (spec §4.2); no objective scaling.

    ``weights``: n floats, or a 2-D sequence (one row of n per run); finite, >= 0, and no
    row may sum to 0.
    """

    weights: Sequence[float] | Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        for i, row in enumerate(_as_rows("weights", self.weights)):
            if sum(row) == 0:
                raise ValueError(f"weights row {i} sums to 0")

    def rows(self, n: int) -> list[list[float]]:
        rows = _as_rows("weights", self.weights)
        for i, row in enumerate(rows):
            if len(row) != n:
                raise ValueError(f"weights row {i} has {len(row)} values; expected {n}")
        return rows


# --- prioritizr parameter matrices ------------------------------------------------------


def _check_matrix_args(n_objectives: int, n_values: int) -> tuple[int, int]:
    """Both sizes as ``int`` (numpy integers accepted, ``bool`` rejected)."""
    for name, v in (("n_objectives", n_objectives), ("n_values", n_values)):
        if not isinstance(v, numbers.Integral) or isinstance(v, bool) or v < 2:
            raise ValueError(f"{name} must be an integer >= 2, got {v!r}")
    return int(n_objectives), int(n_values)


def _grid(values: list[float], n_cols: int) -> list[list[float]]:
    """R ``expand.grid`` row order: the first column varies fastest."""
    return [list(reversed(combo)) for combo in product(values, repeat=n_cols)]


def _points(top: float, n_values: int, include_zeros: bool) -> list[float]:
    if include_zeros:
        return [top * i / (n_values - 1) for i in range(n_values)]
    return [top * i / n_values for i in range(1, n_values + 1)]


def weights_matrix(
    n_objectives: int,
    n_values: int,
    *,
    include_zeros: bool = True,
    include_extremes: bool = True,
) -> list[list[float]]:
    """prioritizr ``approach_weights_matrix`` (9.0.1), row for row.

    A grid of ``n_values`` points in [0, 1] (with ``include_zeros=False``, the
    ``n_values`` non-zero points of an ``n_values + 1`` grid); rows whose values are all
    equal, or with fewer than two entries > 1e-6, are dropped; the identity rows are
    prepended if ``include_extremes`` and a row of ones always comes first. Despite
    prioritizr's documentation, proportional duplicates such as (0.5, 0.25) and (1, 0.5)
    are kept.
    """
    n_objectives, n_values = _check_matrix_args(n_objectives, n_values)
    rows = [
        row for row in _grid(_points(1.0, n_values, include_zeros), n_objectives)
        if len(set(row)) > 1 and sum(v > 1e-6 for v in row) >= 2
    ]
    if include_extremes:
        identity = [
            [1.0 if i == j else 0.0 for j in range(n_objectives)] for i in range(n_objectives)
        ]
        rows = identity + rows
    return [[1.0] * n_objectives] + rows


def rel_tol_matrix(
    n_objectives: int,
    n_values: int,
    max: float,  # prioritizr's argument name
    *,
    include_zeros: bool = True,
    order: bool = True,
) -> list[list[float]]:
    """prioritizr ``approach_rel_tol_matrix`` (9.0.1), row for row.

    A grid over n−1 columns of ``n_values`` points in [0, max] (with
    ``include_zeros=False``, the ``n_values`` non-zero points of an ``n_values + 1`` grid).
    With ``order``, only rows that are non-increasing across columns are kept, so an
    earlier stage gets at least the tolerance of a later one.
    """
    n_objectives, n_values = _check_matrix_args(n_objectives, n_values)
    if not _finite(max) or max < 0:
        raise ValueError(f"max must be finite and >= 0, got {max!r}")
    rows = _grid(_points(float(max), n_values, include_zeros), n_objectives - 1)
    if order:
        rows = [r for r in rows if all(r[i - 1] >= r[i] for i in range(1, len(r)))]
    return rows


# --- nondominated -------------------------------------------------------------------


def nondominated(solutions: Sequence[Solution]) -> list[Solution]:
    """The solutions no other solution dominates, in input order.

    Uses ``metadata["objective_values"]`` and ``["objective_senses"]``. A solution is
    dominated if another is at least as good on every key and strictly better on one.
    Values are compared exactly. Of identical value vectors only the first is kept.
    """
    if not solutions:
        return []
    keys: list[str] | None = None
    senses: dict[str, str] = {}
    vectors: list[list[float]] = []
    for i, sol in enumerate(solutions):
        values = sol.metadata.get("objective_values")
        sol_senses = sol.metadata.get("objective_senses")
        if not isinstance(values, Mapping) or not isinstance(sol_senses, Mapping):
            raise ValueError(
                f"solution {i} lacks metadata 'objective_values' / 'objective_senses'"
            )
        if keys is None:
            keys, senses = sorted(values), dict(sol_senses)
        if sorted(values) != keys or dict(sol_senses) != senses or set(senses) != set(keys):
            raise ValueError(f"solution {i} has different objective keys or senses")
        # Orient every key as a minimisation.
        vectors.append([
            float(values[k]) if senses[k] == "min" else -float(values[k]) for k in keys
        ])

    def dominates(a: list[float], b: list[float]) -> bool:
        return all(p <= q for p, q in zip(a, b, strict=True)) and a != b

    kept: list[Solution] = []
    for i, vec in enumerate(vectors):
        if any(dominates(other, vec) for other in vectors):
            continue
        if any(vectors[j] == vec for j in range(i)):
            continue
        kept.append(solutions[i])
    return kept


# --- the solver -------------------------------------------------------------------


def _no_solution(status: int) -> str:
    """Why a solve has no accepted solution, for the RuntimeError text."""
    if status == pulp.LpStatusUnbounded:
        return (
            "is unbounded: an objective can improve without limit (e.g. a negative SPF "
            "on a shortfall objective, or negative boundary / connectivity values)"
        )
    return (
        f"found no solution (status {pulp.LpStatus.get(status, status)}); run "
        "presolve_check(problem) to look for infeasible data"
    )


class MultiObjectiveMIP:
    """Exact multi-objective MIP over one single-zone ``ConservationProblem`` (spec §5).

    ``objectives`` are specs; their effective keys (``label or name``) must be unique.
    ``approach`` is ``Hierarchical`` or ``WeightedSum``. The base model is
    ``MIPSolver``'s (locks, boundary, connectivity, probability, no-good cuts); targets are
    soft and ``COSTBUDGET``, when set, binds every stage and run. Each objective's term is
    built once per ``solve``.

    ``solve`` returns one ``Solution`` per approach row. ``cost`` / ``boundary`` /
    ``objective`` / ``targets_met`` are filled as ``MIPSolver`` fills them (Marxan-style);
    ``metadata["sol_status"]`` is PuLP's solution status (1 optimal, 2 feasible); under the
    Gurobi backend PuLP's ``GUROBI_CMD`` reports no solution status, so a time-limited
    incumbent may show 1 rather than 2.
    ``metadata["objective_values"]`` holds each objective's value at the plan, from the
    model's own expressions (so ``"min_cost"`` can differ from ``Solution.objective``, and
    under PROBMODE 2 met counts use the scaled amounts). ``config.num_solutions`` is
    ignored. ``MIP_TIME_LIMIT`` applies per stage and row, so a hierarchical sweep takes up
    to rows × n × T; with ``MIP_GAP`` > 0 or a time limit a stage's v is an incumbent's
    value. Stage 1 (or a weighted-sum row) with no solution raises ``RuntimeError``; a later
    stage with no solution returns the previous stage's plan with a ``UserWarning``.

    Met counts in ``objective_values`` follow the exact ``check_targets`` rule (A >= R), so
    on data at a target's tolerance boundary (e.g. amounts 0.7 + 0.1 against 0.8) the
    reported count can be below the value a hierarchical cap kept, which the solver reached
    within its tolerance. When the implied point of a stage's plan violates a model row
    (including an earlier stage's cap), that stage's cap falls back to the solver's own
    value, and later stages can as well.
    """

    def __init__(
        self,
        objectives: Sequence[ObjectiveSpec],
        approach: Hierarchical | WeightedSum,
        *,
        mip_backend: str = "auto",
        mip_chance_strategy: str = "drop",
        mip_clump_strategy: str = "drop",
        mip_sep_strategy: str = "drop",
        warn_above_pu: int | None = 200_000,
        presolve_check: bool | str = False,
    ) -> None:
        specs = list(objectives)
        if not specs:
            raise ValueError("objectives must hold at least one ObjectiveSpec")
        for spec in specs:
            if not isinstance(spec, ObjectiveSpec) or spec.name not in BUILDERS:
                raise TypeError(
                    f"objectives must be ObjectiveSpec subclasses with a builder "
                    f"(MinCost, MinShortfall, ...), got {spec!r}"
                )
        keys = [s.key for s in specs]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise ValueError(f"objectives have duplicate keys (label or name): {dupes}")
        if not isinstance(approach, Hierarchical | WeightedSum):
            raise TypeError(
                f"approach must be Hierarchical or WeightedSum, got {type(approach).__name__}"
            )
        self.objectives = specs
        self.approach = approach
        self.rows = approach.rows(len(specs))
        # Reuses MIPSolver's strategy, backend and presolve-flag validation.
        self._mip = MIPSolver(
            mip_backend=mip_backend, mip_chance_strategy=mip_chance_strategy,
            mip_clump_strategy=mip_clump_strategy, mip_sep_strategy=mip_sep_strategy,
            warn_above_pu=warn_above_pu, presolve_check=presolve_check,
        )

    def name(self) -> str:
        return "MultiObjectiveMIP"

    def _check(self, problem: ConservationProblem) -> None:
        from pymarxan.zones.model import ZonalProblem

        if isinstance(problem, ZonalProblem):
            raise TypeError(
                "MultiObjectiveMIP takes a single-zone ConservationProblem; ZonalProblem "
                "support is a follow-on (use ZoneMIPSolver)"
            )
        if problem.boundary is None and any(isinstance(s, MinBoundary) for s in self.objectives):
            raise ValueError(
                "MinBoundary needs a boundary table (problem.boundary is None): its "
                "objective would be the constant 0"
            )
        ids = {int(f) for f in problem.features["id"].values}
        for spec in self.objectives:
            if isinstance(spec, MaxWtdSum) and spec.weights is not None:
                missing = sorted(ids - set(spec.weights))
                unknown = sorted(set(spec.weights) - ids)
                if missing or unknown:
                    raise ValueError(
                        f"MaxWtdSum {spec.key!r} weights must cover every problem feature: "
                        f"missing {missing}, unknown {unknown}"
                    )

    def _weighted(self, terms: list[ObjectiveTerm], row: list[float]) -> ObjectiveTerm:
        total = sum(row)
        expr = pulp.LpAffineExpression()
        for w, term in zip(row, terms, strict=True):
            sign = 1.0 if term.sense == pulp.LpMinimize else -1.0
            expr += (w / total) * sign * term.expr
        return ObjectiveTerm(expr, pulp.LpMinimize)

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig()
        self._check(problem)
        mip = self._mip
        _warn_if_large_mip(problem.n_planning_units, mip.warn_above_pu)
        data = _presolve_data(mip.presolve_check, problem, self.name())
        need_boundary = any(isinstance(s, MinBoundary) for s in self.objectives)
        base = mip._build_base(problem, config, need_boundary=need_boundary)
        model, x = base.model, base.x
        budget = problem.parameters.get("COSTBUDGET")
        if budget is not None:
            model += base.cost_expr <= float(budget), "cost_budget"
        ctx = base.context()
        terms = [spec.build(ctx) for spec in self.objectives]  # once per solve (spec §3)
        hierarchical = isinstance(self.approach, Hierarchical)
        # P10 scans the first row's objective only (spec §9).
        set_objective(model, terms[0] if hierarchical else self._weighted(terms, self.rows[0]))
        _emit_presolve(mip.presolve_check, data, model, self.name(), stacklevel=3)

        solver, backend = _mip_solver_for(problem, mip.mip_backend, config.verbose)
        tracked = [*x.values(), *(v for v, _ in base.implied)]

        def rounded() -> dict[int, float]:
            return {pid: float(round(v.varValue or 0.0)) for pid, v in x.items()}

        def evaluate(i: int) -> float | None:
            values, feasible = evaluate_implied(
                model, x, rounded(), base.implied, [terms[i]],
            )
            return values[0] if feasible else None

        keys = [s.key for s in self.objectives]
        senses = {s.key: s.sense for s in self.objectives}
        solutions: list[Solution] = []
        for r, row in enumerate(self.rows):
            meta: dict[str, Any] = {"solver": self.name()}
            if isinstance(self.approach, Hierarchical):
                res = solve_hierarchical(
                    model, terms, row, self.approach.abs_tol, solver, evaluate,
                    tracked=tracked,
                )
                if res.stage_reached == 0:
                    raise RuntimeError(
                        f"hierarchical row {r}: stage 1 ({keys[0]}) {_no_solution(res.status)}"
                    )
                if res.stage_reached < len(terms):
                    k = res.stage_reached + 1
                    warnings.warn(
                        f"hierarchical row {r}: stage {k} ({keys[k - 1]}) found no "
                        f"solution; returning the stage {k - 1} plan (a small positive "
                        f"rel_tol, e.g. 1e-9, or a larger abs_tol usually helps)",
                        UserWarning,
                        stacklevel=2,
                    )
                status, sol_status = res.status, res.sol_status
                meta.update(approach="hierarchical", row=r, rel_tol=list(row),
                            stage_reached=res.stage_reached)
            else:
                for var in [*model.variables(), *tracked]:
                    var.varValue = None
                set_objective(model, self._weighted(terms, row))
                model.solve(solver)
                if model.sol_status not in ACCEPTED_SOL_STATUSES:
                    raise RuntimeError(
                        f"weighted-sum row {r} (weights {list(row)}) "
                        f"{_no_solution(int(model.status))}; the sweep is aborted"
                    )
                status, sol_status = int(model.status), int(model.sol_status)
                meta.update(approach="weighted_sum", row=r, weights=list(row))
            x_rounded = rounded()
            values, _ = evaluate_implied(model, x, x_rounded, base.implied, terms)
            selected = np.array([bool(x_rounded[pid]) for pid in base.pu_ids], dtype=bool)
            meta.update(
                status=pulp.LpStatus[status], sol_status=int(sol_status),
                objective_values=dict(zip(keys, values, strict=True)),
                objective_senses=dict(senses), mip_backend=backend,
            )
            meta.update(
                mip._strategy_metadata(base.probmode, base.has_target2, base.has_sep_active)
            )
            solutions.append(build_solution(problem, selected, base.blm, metadata=meta))
        return solutions
