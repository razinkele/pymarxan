"""MIP solver using PuLP for exact conservation planning optimization."""
from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pulp

from pymarxan.models.problem import (
    STATUS_LOCKED_IN,
    STATUS_LOCKED_OUT,
    ConservationProblem,
)
from pymarxan.solvers._backends import (
    _MIP_BACKEND_NAMES as _MIP_BACKEND_NAMES,
)
from pymarxan.solvers._backends import (
    _available_backends as _available_backends,
)
from pymarxan.solvers._backends import (
    _make_pulp_solver as _make_pulp_solver,
)
from pymarxan.solvers._decisions import (
    DECISIONS as DECISIONS,
)
from pymarxan.solvers._decisions import (
    SNAP_TOL as SNAP_TOL,
)
from pymarxan.solvers._decisions import (
    ZERO_TOL as ZERO_TOL,
)
from pymarxan.solvers._decisions import (
    _extract_selected as _extract_selected,
)
from pymarxan.solvers._decisions import (
    _validate_decisions as _validate_decisions,
)
from pymarxan.solvers._mip_objectives import (
    BUILDERS,
    ImpliedRule,
    ObjectiveContext,
    TargetTerm,
    build_min_cost,
    build_min_shortfall,
    expr_value,
    resolve_objective,
    set_objective,
    solve_lexicographic,
    validate_shortfall,
)
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.presolve import (
    _emit_presolve,
    _presolve_data,
    _validate_presolve_flag,
)
from pymarxan.solvers.utils import (
    _nonbinary_conflicts,
    _probability_map,
    _probmode,
    _probmode2_active,
    build_solution,
)


def _reject_nonbinary(
    problem: ConservationProblem, config: SolverConfig, decisions: str,
) -> None:
    """``ValueError`` for settings that only make sense with 0/1 decisions (spec §3.5)."""
    if decisions == "binary":
        return
    causes: list[str] = []
    if int(problem.parameters.get("PROBMODE", 0)) == 3:
        causes.append("PROBMODE 3 (Z-score chance constraints)")
    causes += _nonbinary_conflicts(problem)
    if config.metadata and config.metadata.get("forbidden_selections"):
        causes.append(
            "forbidden_selections (no-good cuts are only valid for 0/1 variables)"
        )
    conn = problem.connectivity
    if (
        float(problem.parameters.get("CONNECTIVITY_WEIGHT", 0.0)) > 0
        and conn is not None
        and (conn["value"].to_numpy(dtype=float) < 0).any()
    ):
        causes.append(
            "negative connectivity values (the linearised term would no longer equal "
            "min(x_i, x_j))"
        )
    if causes:
        raise ValueError(
            f"decisions={decisions!r} cannot be combined with " + "; ".join(causes)
            + ". Use decisions='binary'."
        )


def _validate_mip_strategy(
    name: str,
    value: str,
    allowed: tuple[str, ...],
    rejected_with_reason: dict[str, str] | None = None,
) -> None:
    """Validate a MIP strategy kwarg at __init__ time (round-3 M7).

    ``rejected_with_reason`` is ``{strategy_name: explanation}`` for values
    that ARE recognised but explicitly rejected with a different reason
    than 'not in allowed set' (e.g. ``"socp"`` for separation, which is
    combinatorial rather than conic).
    """
    if rejected_with_reason and value in rejected_with_reason:
        raise ValueError(
            f"{name}={value!r} is not valid — {rejected_with_reason[value]}"
        )
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


# Phase 21 MIP backend factory was extracted to solvers/_backends.py in Phase C0
# so the core, zone, and river MIP solvers share one importable home. The imports
# above re-export it for callers that still do
# `from pymarxan.solvers.mip_solver import _available_backends, _make_pulp_solver`.


def _warn_if_large_mip(n_pu: int, threshold: int | None) -> None:
    """Warn (once) when a MIP is likely too large to build/solve in practice.

    The MILP has one binary variable per planning unit (plus boundary variables when
    BLM > 0); CBC/HiGHS do not scale to ~1e6 binary variables and pulp is slow to even
    build the model at that size. Shared with the zone/river MIP solvers if they adopt it.
    """
    if threshold is not None and n_pu > threshold:
        warnings.warn(
            f"MIP on {n_pu} planning units (> warn_above_pu={threshold}): building and "
            "solving the MILP may be very slow or exhaust memory — CBC/HiGHS do not scale "
            "to ~1e6 binary variables (BLM adds boundary variables on top). Consider the "
            "'sa', 'greedy', or 'zonation' solvers, or pass warn_above_pu=None to silence.",
            stacklevel=3,
        )


def _mip_solver_for(
    problem: ConservationProblem, backend: str, verbose: bool,
) -> tuple[pulp.LpSolver, str]:
    """The PuLP solver for ``backend`` with the problem's ``MIP_TIME_LIMIT`` (default 300 s)
    and ``MIP_GAP`` (default 0), and the resolved backend name for metadata.

    Phase 21: CBC is the default under ``"auto"`` but HiGHS wins if available, so the
    concrete name is read from the solver, not from ``backend``. Shared by ``MIPSolver``
    and ``MultiObjectiveMIP``.
    """
    solver = _make_pulp_solver(
        backend,
        time_limit=int(problem.parameters.get("MIP_TIME_LIMIT", 300)),
        gap=float(problem.parameters.get("MIP_GAP", 0.0)),
        verbose=verbose,
    )
    name = (
        "highs" if isinstance(solver, pulp.HiGHS_CMD)
        else "gurobi" if isinstance(solver, pulp.GUROBI_CMD)
        else "cbc"
    )
    return solver, name


def _core_spf(problem: ConservationProblem) -> dict[int, float]:
    """feature id -> spf; 1.0 when the column is absent (as the zone MIP's _spf_lookup)."""
    ids = problem.features["id"].values
    if "spf" not in problem.features.columns:
        return {int(f): 1.0 for f in ids}
    return {
        int(f): float(s)
        for f, s in zip(ids, problem.features["spf"].values.astype(float), strict=True)
    }


MIP_OBJECTIVES = (
    "min_set",
    "max_features",
    "max_weighted_features",
    "min_shortfall",
    "min_largest_shortfall",
    "min_penalties",
)


def _core_amount_exprs(
    problem: ConservationProblem,
    x: dict[int, pulp.LpVariable],
    prob_map_m2: dict[int, float],
) -> dict[int, pulp.LpAffineExpression]:
    """feature id -> Σ amount·x (PROBMODE 2: × (1 − p)) for features with rows."""
    out: dict[int, pulp.LpAffineExpression] = {}
    for fid_raw, feat_data in problem.pu_vs_features.groupby("species"):
        fid = int(fid_raw)  # type: ignore[arg-type]  # pandas-stubs types the key as a union
        pus = feat_data["pu"].values
        amounts = feat_data["amount"].values
        if prob_map_m2:
            out[fid] = pulp.lpSum(
                float(amt) * (1.0 - prob_map_m2.get(int(pu), 0.0)) * x[int(pu)]
                for pu, amt in zip(pus, amounts)
            )
        else:
            out[fid] = pulp.lpSum(float(amt) * x[int(pu)] for pu, amt in zip(pus, amounts))
    return out


def _core_target_terms(
    problem: ConservationProblem,
    x: dict[int, pulp.LpVariable],
    prob_map_m2: dict[int, float],
    exprs: dict[int, pulp.LpAffineExpression] | None = None,
) -> list[TargetTerm]:
    """One term per feature with target > 0; ghosts get lpSum([]) (spec §1).

    ``exprs`` is the ``_core_amount_exprs`` map when the caller has already built it.
    """
    exprs = _core_amount_exprs(problem, x, prob_map_m2) if exprs is None else exprs
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


def _extract(
    x: dict[int, pulp.LpVariable],
    s: dict[int, pulp.LpVariable],
    pu_ids: list[int],
    decisions: str,
    lower: float,
) -> np.ndarray:
    """``_extract_selected`` on the model's variables: each PU is clipped to its own x
    upper bound (1.0 for locked-in), and PUs without an on/off variable pass ``None``."""
    return _extract_selected(
        [x[pid].varValue for pid in pu_ids],
        [s[pid].varValue if pid in s else None for pid in pu_ids]
        if decisions == "semicontinuous" else [],
        decisions,
        [float(x[pid].upBound if x[pid].upBound is not None else 1.0) for pid in pu_ids],
        lower,
    )


def _value(var: pulp.LpVariable) -> float:
    return float(var.varValue or 0.0)


def _abs_rule(a: pulp.LpVariable, b: pulp.LpVariable) -> Callable[[], float]:
    """Boundary auxiliary: |x_i − x_j|, as the abs rows bound it."""
    return lambda: abs(_value(a) - _value(b))


def _and_rule(a: pulp.LpVariable, b: pulp.LpVariable) -> Callable[[], float]:
    """Symmetric connectivity auxiliary: min(x_i, x_j) (x_i·x_j on 0/1)."""
    return lambda: min(_value(a), _value(b))


def _src_without_sink_rule(i: pulp.LpVariable, j: pulp.LpVariable) -> Callable[[], float]:
    """Asymmetric connectivity auxiliary: max(0, x_i − x_j) (x_i·(1 − x_j) on 0/1)."""
    return lambda: max(0.0, _value(i) - _value(j))


def _on_rule(x: pulp.LpVariable) -> Callable[[], float]:
    """Semicontinuous indicator: s_i = 1 iff x_i > 0."""
    return lambda: 1.0 if _value(x) > 0 else 0.0


@dataclass
class _BaseModel:
    """Result of ``MIPSolver._build_base``: the model every objective shares (spec §2)."""

    model: pulp.LpProblem
    x: dict[int, pulp.LpVariable]
    pu_ids: list[int]
    blm: float
    cost_expr: pulp.LpAffineExpression
    boundary_expr: pulp.LpAffineExpression  # unweighted
    conn_expr: pulp.LpAffineExpression
    prob_expr: pulp.LpAffineExpression
    amount_exprs: dict[int, pulp.LpAffineExpression]
    terms: list[TargetTerm]
    implied: list[ImpliedRule]
    probmode: int
    has_target2: bool
    has_sep_active: bool
    s: dict[int, pulp.LpVariable]  # semicontinuous on/off binaries; empty otherwise
    decisions: str
    upper: float
    lower: float

    def context(self) -> ObjectiveContext:
        """One ``ObjectiveContext`` sharing this base's expressions and registry."""
        return ObjectiveContext(
            self.model, self.cost_expr, self.terms, blm=self.blm,
            boundary_expr=self.boundary_expr, conn_expr=self.conn_expr,
            prob_expr=self.prob_expr, amount_exprs=self.amount_exprs, implied=self.implied,
            fractional=self.decisions != "binary",
        )

    def extract(self) -> np.ndarray:
        """``Solution.selected`` from the current ``varValue``s (spec §4)."""
        return _extract(self.x, self.s, self.pu_ids, self.decisions, self.lower)


@dataclass
class _BuiltModel:
    """Result of ``MIPSolver._build``: the PuLP model plus what ``solve`` needs."""

    model: pulp.LpProblem
    x: dict[int, pulp.LpVariable]
    pu_ids: list[int]
    blm: float
    objective_expr: pulp.LpAffineExpression  # natural units
    probmode: int
    has_target2: bool
    has_sep_active: bool
    s: dict[int, pulp.LpVariable]
    decisions: str
    upper: float
    lower: float
    lex: tuple[pulp.LpAffineExpression, pulp.LpAffineExpression] | None = None


class MIPSolver(Solver):
    """Solver that formulates the Marxan minimum-set problem as a MILP.

    Minimizes:
        sum(cost_i * x_i) + BLM * sum(boundary_ij * y_ij)

    Subject to:
        sum(amount_ij * x_i) >= target_j   for all features j
        y_ij >= x_i - x_j                  for boundary pairs (i != j)
        y_ij >= x_j - x_i                  for boundary pairs (i != j)
        x_i = 1                            for locked-in PUs (status=2)
        x_i = 0                            for locked-out PUs (status=3)
        x_i in {0, 1}

    Under PROBMODE 3 (Z-score chance constraints, Phase 18) the Z-score
    constraint involves ``√Σ σ²·x`` which is SOCP/QCP territory and not
    solvable by the default CBC backend. The ``mip_chance_strategy``
    knob controls behaviour:

    - ``"drop"`` (default): solve the deterministic relaxation; populate
      ``Solution.prob_shortfalls`` and ``Solution.prob_penalty`` from a
      post-hoc Z-score evaluation. User sees the gap.
    - ``"piecewise"``: tangent-line approximation of the chance constraint
      in CBC. Deferred to Phase 18.5.
    - ``"socp"``: exact SOCP formulation via Gurobi/CPLEX. Deferred to
      Phase 21.

    ``objective`` selects the formulation (Stage 1): ``"min_set"`` (default, above),
    ``"max_features"`` / ``"max_weighted_features"`` (count or SPF-weighted count of targets
    met under ``COSTBUDGET``; ``"max_n_targets_met"`` is an accepted alias of
    ``"max_features"``), ``"min_shortfall"`` (Σ SPF·shortfall under ``COSTBUDGET``),
    ``"min_largest_shortfall"`` (minimax shortfall under ``COSTBUDGET``) and
    ``"min_penalties"`` (Σ SPF·shortfall first, cost second). ``min_penalties`` is an exact
    two-pass lexicographic solve (pass 1 minimises Σ SPF·shortfall; pass 2 minimises cost
    with that optimum capped); it differs from prioritizr's ``add_min_penalties_objective``,
    which minimises penalties subject to hard targets (and an optional budget); here targets
    are soft and their shortfall is penalised by SPF, Marxan-style. ``MIP_TIME_LIMIT``
    applies to each pass. Like v0.38, ``min_penalties`` ignores the BLM / connectivity /
    probability terms (as do the budgeted objectives).

    ``shortfall`` sets the units of the two shortfall objectives: ``"relative"`` (default,
    shortfall / (target × MISSLEVEL), as prioritizr) or ``"absolute"`` (the v0.38 behaviour
    of ``min_largest_shortfall``). In relative units a feature with a target but no
    ``pu_vs_features`` rows has full shortfall and pins the minimax at 1.0
    (``presolve_check`` P5 flags it).

    ``Solution.metadata`` records ``objective``, ``mip_objective_value`` (natural sign: a
    count for ``max_*``) and ``shortfall`` — the configured units, recorded for every
    objective but meaningful only for the two shortfall objectives. ``max_features`` /
    ``max_weighted_features`` count only features with target > 0.

    ``decisions`` (Stage 4): ``"binary"`` (default), ``"proportion"`` (continuous
    ``x ∈ [0, upper]``; prioritizr's ``add_semicontinuous_decisions(u)`` is
    ``decisions="proportion", upper=u``) or ``"semicontinuous"`` (``x = 0`` or
    ``lower ≤ x ≤ upper``, a MILP with one on/off binary per PU). Non-binary solutions hold
    float64 proportions; locked-in PUs are fixed whole (x = 1.0) in every mode, ``upper`` and
    ``lower`` bound the free PUs. PROBMODE 3, clumping,
    separation, space targets, ``forbidden_selections`` and negative connectivity values
    are rejected with ``ValueError`` for non-binary decisions.

    ``presolve_check`` (default ``False``): ``True`` runs
    :func:`~pymarxan.solvers.presolve.presolve_check` plus a scan of the built model (P10)
    in ``solve`` / ``build_model`` and emits one ``PresolveWarning``; ``"strict"`` raises
    ``ValueError`` instead. The data checks run before the model is built: a non-finite
    value (P1) raises ``ValueError`` under both, since PuLP cannot build the model.
    """

    def __init__(
        self,
        *,
        mip_chance_strategy: str = "drop",
        mip_clump_strategy: str = "drop",
        mip_sep_strategy: str = "drop",
        mip_backend: str = "auto",
        objective: str = "min_set",
        shortfall: str = "relative",
        warn_above_pu: int | None = 200_000,
        presolve_check: bool | str = False,
        decisions: str = "binary",
        upper: float = 1.0,
        lower: float = 0.0,
    ) -> None:
        self.decisions, self.upper, self.lower = _validate_decisions(decisions, upper, lower)
        _validate_mip_strategy(
            "mip_chance_strategy", mip_chance_strategy,
            ("drop", "piecewise", "socp"),
        )
        _validate_mip_strategy(
            "mip_clump_strategy", mip_clump_strategy, ("drop", "big_m"),
        )
        _validate_mip_strategy(
            "mip_sep_strategy", mip_sep_strategy, ("drop", "big_m"),
            rejected_with_reason={
                "socp": (
                    "separation is a combinatorial constraint (greedy "
                    "maximum independent set), not a conic/probabilistic one. "
                    "Use 'drop' (default; gap reported on "
                    "Solution.sep_shortfalls) or 'big_m' (deferred)."
                ),
            },
        )
        if mip_backend not in _MIP_BACKEND_NAMES:
            raise ValueError(
                f"mip_backend must be one of {_MIP_BACKEND_NAMES}, got "
                f"{mip_backend!r}."
            )
        self.mip_chance_strategy = mip_chance_strategy
        self.mip_clump_strategy = mip_clump_strategy
        self.mip_sep_strategy = mip_sep_strategy
        self.mip_backend = mip_backend
        self.objective = resolve_objective(objective, MIP_OBJECTIVES)
        self.shortfall = validate_shortfall(shortfall)
        self.warn_above_pu = warn_above_pu
        self.presolve_check = _validate_presolve_flag(presolve_check)

    def name(self) -> str:
        return "MIP (PuLP)"

    def supports_zones(self) -> bool:
        return False

    def available(self) -> bool:
        return True  # pulp is a core dependency

    def _build_base(
        self,
        problem: ConservationProblem,
        config: SolverConfig,
        *,
        need_boundary: bool = False,
    ) -> _BaseModel:
        """Everything every objective shares (spec §2); no targets, budget or objective.

        The boundary rows are built when a boundary table exists and BLM > 0 or
        ``need_boundary``. Every auxiliary variable is registered in ``implied`` with the
        value its rows imply from x (spec §5).
        """
        # Stage 4: settings that need 0/1 decisions fail first, before any strategy check.
        _reject_nonbinary(problem, config, self.decisions)
        binary = self.decisions == "binary"

        # PROBMODE 3 gating — see class docstring for rationale.
        probmode = _probmode(problem)
        if probmode == 3:
            if self.mip_chance_strategy == "piecewise":
                raise NotImplementedError(
                    "mip_chance_strategy='piecewise' (piecewise-linear "
                    "approximation of the chance constraint in CBC) is "
                    "planned for Phase 18.5. Use 'drop' (default) or one "
                    "of the SA / heuristic / iterative-improvement solvers."
                )
            if self.mip_chance_strategy == "socp":
                raise NotImplementedError(
                    "mip_chance_strategy='socp' (exact SOCP formulation via "
                    "Gurobi/CPLEX) lands in Phase 21 when those backends "
                    "are wired in. Use 'drop' (default) or SA for now."
                )

        # TARGET2 / clumping gating (Phase 19). Like PROBMODE 3, the chance-
        # constrained-style CLUMPTYPE math is non-trivial in MILP; the
        # default "drop" strategy solves the deterministic relaxation and
        # build_solution reports the clump-shortfall gap post-hoc.
        has_target2 = (
            "target2" in problem.features.columns
            and (problem.features["target2"] > 0).any()
        )
        if has_target2 and self.mip_clump_strategy == "big_m":
            raise NotImplementedError(
                "mip_clump_strategy='big_m' (network-flow + big-M formulation "
                "of TARGET2 in CBC) is deferred to a later phase. Use 'drop' "
                "(default; post-hoc gap reported on Solution.clump_shortfalls) "
                "or the SA / iterative-improvement solvers for chance-"
                "constrained-style clumping optimality."
            )

        # SEPDISTANCE / SEPNUM gating (Phase 20). MIP "drop" runs the
        # deterministic relaxation and build_solution reports the
        # separation gap on Solution.sep_shortfalls post-hoc.
        has_sep_active = (
            "sepnum" in problem.features.columns
            and "sepdistance" in problem.features.columns
            and (
                (problem.features["sepnum"] > 1)
                & (problem.features["sepdistance"] > 0)
            ).any()
        )
        if has_sep_active and self.mip_sep_strategy == "big_m":
            raise NotImplementedError(
                "mip_sep_strategy='big_m' (pairwise-distance + big-M "
                "formulation of separation in CBC) is deferred to v0.3. "
                "Use 'drop' (default; post-hoc gap reported on "
                "Solution.sep_shortfalls) or SA / iterative-improvement."
            )

        blm = float(problem.parameters.get("BLM", 0.0))
        pu_ids = problem.planning_units["id"].tolist()
        implied: list[ImpliedRule] = []

        # Build the LP model
        model = pulp.LpProblem("MarxanMIP", pulp.LpMinimize)

        # Decision variables: x_i = 1 if PU i is selected (Stage 4: the selected share)
        x = {}
        s: dict[int, pulp.LpVariable] = {}
        pu_id_arr = problem.planning_units["id"].values
        pu_st_arr = problem.planning_units["status"].values.astype(int)
        for k in range(len(pu_id_arr)):
            pid = int(pu_id_arr[k])
            status = int(pu_st_arr[k])
            locked = status in (STATUS_LOCKED_IN, STATUS_LOCKED_OUT)
            if binary:
                x[pid] = pulp.LpVariable(f"x_{pid}", cat="Binary")
            else:
                # Locked-in units are whole (x = 1.0) in every mode; `upper` and the
                # semicontinuous band apply to free units only (review R3).
                x[pid] = pulp.LpVariable(
                    f"x_{pid}", lowBound=0,
                    upBound=1.0 if status == STATUS_LOCKED_IN else self.upper,
                    cat="Continuous",
                )
            if self.decisions == "semicontinuous" and not locked:
                s[pid] = pulp.LpVariable(f"s_{pid}", cat="Binary")
                model += x[pid] - self.upper * s[pid] <= 0, f"sc_ub_{pid}"
                model += x[pid] - self.lower * s[pid] >= 0, f"sc_lb_{pid}"
                implied.append((s[pid], _on_rule(x[pid])))
            if status == STATUS_LOCKED_IN:
                # prioritizr's lb = 1 with ub = upper < 1 is infeasible; here a locked-in
                # unit is fixed whole whatever `upper` is (documented deviation, R3).
                model += x[pid] == 1, f"locked_in_{pid}"
            elif status == STATUS_LOCKED_OUT:
                model += x[pid] == 0, f"locked_out_{pid}"

        # Cost component
        pu_cost_arr = problem.planning_units["cost"].values.astype(float)
        cost_expr = pulp.lpSum(
            float(pu_cost_arr[k]) * x[int(pu_id_arr[k])]
            for k in range(len(pu_id_arr))
        )

        # Unweighted boundary component with auxiliary variables
        boundary_expr = pulp.LpAffineExpression()
        y = {}
        if (blm > 0 or need_boundary) and problem.boundary is not None:
            b_id1 = problem.boundary["id1"].values
            b_id2 = problem.boundary["id2"].values
            b_val = problem.boundary["boundary"].values.astype(float)
            for bk in range(len(b_id1)):
                id1 = int(b_id1[bk])
                id2 = int(b_id2[bk])
                bval = float(b_val[bk])

                if id1 == id2:
                    # Self-boundary: external boundary cost when PU is selected
                    boundary_expr += bval * x[id1]
                else:
                    # Off-diagonal: linearize |x_i - x_j|
                    key = (min(id1, id2), max(id1, id2))
                    if key not in y:
                        y_var = pulp.LpVariable(
                            f"y_{key[0]}_{key[1]}", lowBound=0, upBound=1
                        )
                        y[key] = y_var
                        model += (
                            y_var >= x[key[0]] - x[key[1]],
                            f"abs1_{key[0]}_{key[1]}",
                        )
                        model += (
                            y_var >= x[key[1]] - x[key[0]],
                            f"abs2_{key[0]}_{key[1]}",
                        )
                        implied.append((y_var, _abs_rule(x[key[0]], x[key[1]])))
                    boundary_expr += bval * y[key]

        # Connectivity component with auxiliary variables
        conn_expr = pulp.LpAffineExpression()
        conn_weight = float(problem.parameters.get("CONNECTIVITY_WEIGHT", 0.0))
        asymmetric = bool(
            int(problem.parameters.get("ASYMMETRIC_CONNECTIVITY", 0))
        )
        if conn_weight > 0 and problem.connectivity is not None:
            conn = problem.connectivity
            c_id1 = conn["id1"].values
            c_id2 = conn["id2"].values
            c_val = conn["value"].values.astype(float)

            if asymmetric:
                # Penalize selecting source i without sink j:
                #   z_ij >= x_i - x_j, z_ij >= 0
                #   objective: +conn_weight * c_ij * z_ij
                # Duplicate (i, j) rows share one z, so their values add up.
                asym_z: dict[tuple[int, int], pulp.LpVariable] = {}
                for ck in range(len(c_id1)):
                    i = int(c_id1[ck])
                    j = int(c_id2[ck])
                    cval = float(c_val[ck])
                    z = asym_z.get((i, j))
                    if z is None:
                        z = pulp.LpVariable(
                            f"zc_{i}_{j}", lowBound=0, cat="Continuous",
                        )
                        asym_z[(i, j)] = z
                        model += z >= x[i] - x[j], f"conn_asym_{i}_{j}"
                        implied.append((z, _src_without_sink_rule(x[i], x[j])))
                    conn_expr += conn_weight * cval * z
            else:
                # Reward both selected via binary-AND linearization:
                #   z_ij <= x_i, z_ij <= x_j, z_ij >= x_i + x_j - 1
                #   objective: -conn_weight * c_ij * z_ij
                # The first row of a duplicated (i, j) / (j, i) pair wins. Non-binary
                # decisions: z is continuous in [0, 1] (a binary z would be forced to 0 by
                # fractional x); the reward drives it to min(x_i, x_j).
                seen: set[tuple[int, int]] = set()
                for ck in range(len(c_id1)):
                    i = int(c_id1[ck])
                    j = int(c_id2[ck])
                    key = (min(i, j), max(i, j))
                    if key in seen:
                        continue
                    seen.add(key)
                    cval = float(c_val[ck])
                    if binary:
                        z = pulp.LpVariable(f"zc_{key[0]}_{key[1]}", cat="Binary")
                    else:
                        z = pulp.LpVariable(
                            f"zc_{key[0]}_{key[1]}", lowBound=0, upBound=1,
                            cat="Continuous",
                        )
                    model += z <= x[key[0]], f"conn_and1_{key[0]}_{key[1]}"
                    model += z <= x[key[1]], f"conn_and2_{key[0]}_{key[1]}"
                    model += (
                        z >= x[key[0]] + x[key[1]] - 1,
                        f"conn_and3_{key[0]}_{key[1]}",
                    )
                    implied.append((z, _and_rule(x[key[0]], x[key[1]])))
                    conn_expr += -conn_weight * cval * z

        # Probability risk premium (Mode 1; the default with a probability table). The
        # table is read by the shared helpers in solvers/utils.py (v0.45 §3).
        prob_expr = pulp.LpAffineExpression()
        if problem.probability is not None and probmode == 1:
            prob_weight = float(
                problem.parameters.get("PROBABILITYWEIGHTING", 1.0)
            )
            if prob_weight > 0:
                prob_map = _probability_map(problem)
                prob_expr = pulp.lpSum(
                    prob_weight
                    * prob_map.get(int(pu_id_arr[k]), 0.0)
                    * float(pu_cost_arr[k])
                    * x[int(pu_id_arr[k])]
                    for k in range(len(pu_id_arr))
                )

        # Probability map for Mode 2 (expected amounts a·(1 − p) in the target rows)
        prob_map_m2: dict[int, float] = (
            _probability_map(problem) if _probmode2_active(problem) else {}
        )

        amount_exprs = _core_amount_exprs(problem, x, prob_map_m2)
        terms = _core_target_terms(problem, x, prob_map_m2, amount_exprs)

        # Phase 25: no-good cuts for portfolio generation. Each forbidden
        # selection vector contributes one constraint forcing at least
        # one binary variable to differ from it. Passed through
        # config.metadata["forbidden_selections"] as a list of bool arrays
        # indexed by pu_id_arr order.
        forbidden_selections = (
            config.metadata.get("forbidden_selections", [])
            if config.metadata else []
        )
        for forbid_idx, forbidden in enumerate(forbidden_selections):
            # diff_expr = Σ_{i: s_i=1} (1 - x_i) + Σ_{i: s_i=0} x_i
            # Must be ≥ 1 to ensure at least one variable flipped.
            diff_terms = []
            for k in range(len(pu_id_arr)):
                pid = int(pu_id_arr[k])
                if bool(forbidden[k]):
                    diff_terms.append(1 - x[pid])
                else:
                    diff_terms.append(x[pid])
            model += pulp.lpSum(diff_terms) >= 1, f"nogood_cut_{forbid_idx}"

        return _BaseModel(
            model=model, x=x, pu_ids=pu_ids, blm=blm, cost_expr=cost_expr,
            boundary_expr=boundary_expr, conn_expr=conn_expr, prob_expr=prob_expr,
            amount_exprs=amount_exprs, terms=terms, implied=implied, probmode=probmode,
            has_target2=bool(has_target2), has_sep_active=bool(has_sep_active),
            s=s, decisions=self.decisions, upper=self.upper, lower=self.lower,
        )

    def _build(self, problem: ConservationProblem, config: SolverConfig) -> _BuiltModel:
        """Build the PuLP model (no warnings, no solve): the base plus this objective."""
        base = self._build_base(problem, config)
        model = base.model
        ctx = base.context()
        lex: tuple[pulp.LpAffineExpression, pulp.LpAffineExpression] | None = None
        if self.objective == "min_set":
            term = build_min_cost(ctx, self.shortfall)
            set_objective(model, term)
            # Hard targets for every feature with pu_vs_features rows (a ghost adds none).
            misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
            for fid_raw, target in zip(
                problem.features["id"].values,
                problem.features["target"].values.astype(float),
                strict=True,
            ):
                fid = int(fid_raw)
                if fid in base.amount_exprs:
                    model += (
                        base.amount_exprs[fid] >= float(target) * misslevel,
                        f"target_{fid}",
                    )
        elif self.objective == "min_penalties":
            # Exact lexicographic (S2): pass 1 min Σ SPF·shortfall, pass 2 min cost at that
            # optimum (solve_lexicographic). v0.38's big-M (Σ cost + 1) only dominated when
            # SPF·Δshortfall >= 1, so fractional data could trade targets for cost. Not
            # prioritizr's add_min_penalties_objective (penalty-only, hard targets).
            term = build_min_shortfall(ctx, "absolute")
            model.setObjective(term.expr)  # build_model exports the pass-1 model
            lex = (term.expr, base.cost_expr)
        else:
            cost_budget = problem.parameters.get("COSTBUDGET")
            if cost_budget is None:
                raise ValueError(
                    f"objective={self.objective!r} requires a COSTBUDGET parameter "
                    "(problem.parameters['COSTBUDGET'] = ...); without a budget the "
                    "objective is degenerate (selecting every planning unit meets "
                    "every target)."
                )
            term = BUILDERS[self.objective](ctx, self.shortfall)
            set_objective(model, term)
            model += base.cost_expr <= float(cost_budget), "cost_budget"
        return _BuiltModel(
            model=model, x=base.x, pu_ids=base.pu_ids, blm=base.blm, objective_expr=term.expr,
            probmode=base.probmode, has_target2=base.has_target2,
            has_sep_active=base.has_sep_active, s=base.s, decisions=base.decisions,
            upper=base.upper, lower=base.lower, lex=lex,
        )

    def _strategy_metadata(
        self, probmode: int, has_target2: bool, has_sep_active: bool,
    ) -> dict[str, str]:
        """The ``mip_*_strategy`` keys for the gaps this problem has (shared with
        ``MultiObjectiveMIP``)."""
        meta: dict[str, str] = {}
        if probmode == 3:
            meta["mip_chance_strategy"] = self.mip_chance_strategy
        if has_target2:
            meta["mip_clump_strategy"] = self.mip_clump_strategy
        if has_sep_active:
            meta["mip_sep_strategy"] = self.mip_sep_strategy
        return meta

    def build_model(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> pulp.LpProblem:
        """The PuLP model ``solve`` would hand to the backend (for ``write_problem``).

        For ``min_penalties`` this is the pass-1 model (objective Σ SPF·shortfall).
        """
        if config is None:
            config = SolverConfig()
        _warn_if_large_mip(problem.n_planning_units, self.warn_above_pu)
        data = _presolve_data(
            self.presolve_check, problem, self.name(),
            decisions=self.decisions, upper=self.upper, lower=self.lower,
        )
        built = self._build(problem, config)
        _emit_presolve(self.presolve_check, data, built.model, self.name(), stacklevel=3)
        return built.model

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig()
        # S3b: heads-up before building the model when the MILP is likely too large.
        _warn_if_large_mip(problem.n_planning_units, self.warn_above_pu)
        data = _presolve_data(
            self.presolve_check, problem, self.name(),
            decisions=self.decisions, upper=self.upper, lower=self.lower,
        )
        built = self._build(problem, config)
        _emit_presolve(self.presolve_check, data, built.model, self.name(), stacklevel=3)
        model, x, pu_ids = built.model, built.x, built.pu_ids
        probmode, has_target2 = built.probmode, built.has_target2
        has_sep_active, blm = built.has_sep_active, built.blm

        solver, resolved_backend = _mip_solver_for(problem, self.mip_backend, config.verbose)
        if built.lex is not None:
            status = solve_lexicographic(model, built.lex[0], built.lex[1], solver)
        else:
            model.solve(solver)
            status = int(model.status)

        # Accept any solver outcome where decision variables have integer
        # incumbent values. CBC sets status=LpStatusNotSolved (not Optimal)
        # when MIP_TIME_LIMIT fires but a feasible solution was already found;
        # returning [] in that case silently loses a usable solution.
        infeasible_statuses = {
            pulp.constants.LpStatusInfeasible,
            pulp.constants.LpStatusUnbounded,
            pulp.constants.LpStatusUndefined,
        }
        # A PU in no constraint and with zero cost never enters the model (value None):
        # read as 0 (S9).
        has_values = any(pulp.value(x[pid]) is not None for pid in pu_ids)
        if status in infeasible_statuses or not has_values:
            return []

        # Extract solution (binary: rounded bools; otherwise snapped float64, spec §4)
        selected = _extract(x, built.s, pu_ids, built.decisions, built.lower)

        # build_solution populates Solution.prob_shortfalls + prob_penalty
        # automatically when PROBMODE==3. Under the 'drop' strategy, the
        # MIP solved the deterministic relaxation; the chance-constraint
        # gap is what build_solution reports post-hoc.
        # A zero-cost x never written to the LP has no value: expr_value reads it as 0.
        obj_value = expr_value(built.objective_expr)
        meta = {
            "solver": self.name(),
            "status": pulp.LpStatus[status],
            "mip_backend": resolved_backend,
            "objective": self.objective,
            "mip_objective_value": obj_value,
            "shortfall": self.shortfall,
        }
        meta.update(self._strategy_metadata(probmode, has_target2, has_sep_active))
        sol = build_solution(problem, selected, blm, metadata=meta)
        return [copy.deepcopy(sol) for _ in range(config.num_solutions)]
