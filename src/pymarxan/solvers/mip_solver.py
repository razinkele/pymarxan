"""MIP solver using PuLP for exact conservation planning optimization."""
from __future__ import annotations

import copy
import warnings
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
from pymarxan.solvers._mip_objectives import (
    BUILDERS,
    ObjectiveContext,
    TargetTerm,
    resolve_objective,
    set_objective,
    solve_lexicographic,
    validate_shortfall,
)
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.utils import build_solution


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
    are soft and their shortfall is penalised by SPF, Marxan-style.

    ``shortfall`` sets the units of the two shortfall objectives: ``"relative"`` (default,
    shortfall / (target × MISSLEVEL), as prioritizr) or ``"absolute"`` (the v0.38 behaviour
    of ``min_largest_shortfall``). In relative units a feature with a target but no
    ``pu_vs_features`` rows has full shortfall and pins the minimax at 1.0
    (``presolve_check`` P5 flags it).
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
    ) -> None:
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

    def name(self) -> str:
        return "MIP (PuLP)"

    def supports_zones(self) -> bool:
        return False

    def available(self) -> bool:
        return True  # pulp is a core dependency

    def _build(self, problem: ConservationProblem, config: SolverConfig) -> _BuiltModel:
        """Build the PuLP model (no warnings, no solve)."""
        objective_expr: pulp.LpAffineExpression | None = None
        lex: tuple[pulp.LpAffineExpression, pulp.LpAffineExpression] | None = None

        # PROBMODE 3 gating — see class docstring for rationale.
        probmode = int(problem.parameters.get("PROBMODE", 0))
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

        # Build the LP model
        model = pulp.LpProblem("MarxanMIP", pulp.LpMinimize)

        # Decision variables: x_i = 1 if PU i is selected
        x = {}
        pu_id_arr = problem.planning_units["id"].values
        pu_st_arr = problem.planning_units["status"].values.astype(int)
        for k in range(len(pu_id_arr)):
            pid = int(pu_id_arr[k])
            status = int(pu_st_arr[k])
            x[pid] = pulp.LpVariable(f"x_{pid}", cat="Binary")
            if status == STATUS_LOCKED_IN:
                model += x[pid] == 1, f"locked_in_{pid}"
            elif status == STATUS_LOCKED_OUT:
                model += x[pid] == 0, f"locked_out_{pid}"

        # Objective: cost component
        pu_cost_arr = problem.planning_units["cost"].values.astype(float)
        cost_expr = pulp.lpSum(
            float(pu_cost_arr[k]) * x[int(pu_id_arr[k])]
            for k in range(len(pu_id_arr))
        )

        # Boundary component with auxiliary variables
        boundary_expr = 0
        y = {}
        if blm > 0 and problem.boundary is not None:
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
                    boundary_expr += bval * y[key]

        # Connectivity component with auxiliary variables
        conn_expr = 0
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
                for ck in range(len(c_id1)):
                    i = int(c_id1[ck])
                    j = int(c_id2[ck])
                    cval = float(c_val[ck])
                    z = pulp.LpVariable(
                        f"zc_{i}_{j}", lowBound=0, cat="Continuous",
                    )
                    model += z >= x[i] - x[j], f"conn_asym_{i}_{j}"
                    conn_expr += conn_weight * cval * z
            else:
                # Reward both selected via binary-AND linearization:
                #   z_ij <= x_i, z_ij <= x_j, z_ij >= x_i + x_j - 1
                #   objective: -conn_weight * c_ij * z_ij
                seen: set[tuple[int, int]] = set()
                for ck in range(len(c_id1)):
                    i = int(c_id1[ck])
                    j = int(c_id2[ck])
                    key = (min(i, j), max(i, j))
                    if key in seen:
                        continue
                    seen.add(key)
                    cval = float(c_val[ck])
                    z = pulp.LpVariable(f"zc_{key[0]}_{key[1]}", cat="Binary")
                    model += z <= x[key[0]], f"conn_and1_{key[0]}_{key[1]}"
                    model += z <= x[key[1]], f"conn_and2_{key[0]}_{key[1]}"
                    model += (
                        z >= x[key[0]] + x[key[1]] - 1,
                        f"conn_and3_{key[0]}_{key[1]}",
                    )
                    conn_expr += -conn_weight * cval * z

        # Probability risk premium (Mode 1)
        prob_expr = 0
        prob_mode = int(problem.parameters.get("PROBMODE", 1))
        if problem.probability is not None and prob_mode == 1:
            prob_weight = float(
                problem.parameters.get("PROBABILITYWEIGHTING", 1.0)
            )
            if prob_weight > 0:
                prob_pu = problem.probability["pu"].values
                prob_val = problem.probability["probability"].values
                prob_map: dict[int, float] = {}
                for pk in range(len(prob_pu)):
                    prob_map[int(prob_pu[pk])] = float(prob_val[pk])
                prob_expr = pulp.lpSum(
                    prob_weight
                    * prob_map.get(int(pu_id_arr[k]), 0.0)
                    * float(pu_cost_arr[k])
                    * x[int(pu_id_arr[k])]
                    for k in range(len(pu_id_arr))
                )

        # Build probability map for Mode 2 (persistence-adjusted amounts)
        prob_map_m2: dict[int, float] = {}
        if (
            problem.probability is not None
            and prob_mode == 2
        ):
            p_pu = problem.probability["pu"].values
            p_val = problem.probability["probability"].values
            for pk in range(len(p_pu)):
                prob_map_m2[int(p_pu[pk])] = float(p_val[pk])

        amount_exprs = _core_amount_exprs(problem, x, prob_map_m2)

        # Phase 23: select objective formulation. Default "min_set" preserves
        # pre-Phase-23 behaviour.
        feat_met: dict[int, pulp.LpVariable] = {}
        slack: dict[int, pulp.LpVariable] = {}
        t_var: pulp.LpVariable | None = None
        feat_ids_init = problem.features["id"].values
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
            terms = _core_target_terms(problem, x, prob_map_m2, amount_exprs)
            ctx = ObjectiveContext(model, cost_expr, terms)
            term = BUILDERS[self.objective](ctx, self.shortfall)
            set_objective(model, term)
            model += cost_expr <= float(cost_budget), "cost_budget"
            objective_expr = term.expr
        elif self.objective == "max_features":
            # Binary z_j per feature; relaxed target ≥ target · z_j; cost
            # cap from COSTBUDGET. Maximise Σ z_j (LpMinimize → negate).
            cost_budget = problem.parameters.get("COSTBUDGET")
            if cost_budget is None:
                raise ValueError(
                    "objective='max_features' requires a COSTBUDGET parameter "
                    "on the problem (problem.parameters['COSTBUDGET'] = ...). "
                    "Without a budget the formulation is degenerate — every "
                    "feature target is trivially met by selecting all PUs."
                )
            cost_budget = float(cost_budget)
            for fi in range(len(feat_ids_init)):
                fid = int(feat_ids_init[fi])
                feat_met[fid] = pulp.LpVariable(f"feat_met_{fid}", cat="Binary")
            model += (
                -pulp.lpSum(feat_met[fid] for fid in feat_met),
                "objective",
            )
            model += cost_expr <= cost_budget, "cost_budget"
        elif self.objective == "max_weighted_features":
            # Binary z_j per feature; relaxed target ≥ target · z_j; cost cap
            # from COSTBUDGET. Maximise Σ spf_j·z_j (LpMinimize → negate) — the
            # spf-weighted count of targets met. With spf = branch length this
            # maximises phylogenetic diversity. This weights the *target-met*
            # indicator; it is NOT the objectives/ MaxUtility class (which sums
            # held amounts). Like max_features, a feature with no pu_vs_features
            # rows can never be met (feat_met is forced to 0 below). Solution.objective
            # is recomputed min_set-style and is NOT the weighted objective value.
            cost_budget = problem.parameters.get("COSTBUDGET")
            if cost_budget is None:
                raise ValueError(
                    "objective='max_weighted_features' requires a COSTBUDGET "
                    "parameter on the problem "
                    "(problem.parameters['COSTBUDGET'] = ...). Without a budget "
                    "the formulation is degenerate — every target is trivially "
                    "met by selecting all PUs."
                )
            cost_budget = float(cost_budget)
            spf_by_id = _core_spf(problem)
            for fi in range(len(feat_ids_init)):
                fid = int(feat_ids_init[fi])
                feat_met[fid] = pulp.LpVariable(f"feat_met_{fid}", cat="Binary")
            model += (
                -pulp.lpSum(spf_by_id[fid] * feat_met[fid] for fid in feat_met),
                "objective",
            )
            model += cost_expr <= cost_budget, "cost_budget"
        elif self.objective == "min_largest_shortfall":
            # Auxiliary slack_j ≥ 0, slack_j ≥ target_j - Σ amount·x;
            # t ≥ slack_j for every feature. Minimise t.
            cost_budget = problem.parameters.get("COSTBUDGET")
            if cost_budget is None:
                raise ValueError(
                    "objective='min_largest_shortfall' requires a COSTBUDGET "
                    "parameter (problem.parameters['COSTBUDGET'] = ...). "
                    "Without a budget the objective is degenerate (buying "
                    "every PU drives every shortfall to 0)."
                )
            cost_budget = float(cost_budget)
            t_var = pulp.LpVariable("max_shortfall", lowBound=0)
            for fi in range(len(feat_ids_init)):
                fid = int(feat_ids_init[fi])
                slack[fid] = pulp.LpVariable(f"slack_{fid}", lowBound=0)
                # Coupling slack ≤ t handled below via t ≥ slack.
            model += (t_var, "objective")
            model += cost_expr <= cost_budget, "cost_budget"
        elif self.objective == "min_penalties":
            # Exact lexicographic (S2): pass 1 min Σ SPF·shortfall, pass 2 min cost at that
            # optimum (solve_lexicographic). v0.38's big-M (Σ cost + 1) only dominated when
            # SPF·Δshortfall >= 1, so fractional data could trade targets for cost. Not
            # prioritizr's add_min_penalties_objective (penalty-only, hard targets).
            terms = _core_target_terms(problem, x, prob_map_m2, amount_exprs)
            ctx = ObjectiveContext(model, cost_expr, terms)
            primary = pulp.lpSum(t.weight * ctx.slack(t) for t in ctx.terms)
            model.setObjective(primary)  # build_model exports the pass-1 model
            lex = (primary, cost_expr)
            objective_expr = primary
        else:
            model += (
                cost_expr + blm * boundary_expr + conn_expr + prob_expr,
                "objective",
            )

        # Constraints: feature targets
        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))

        # Pre-group feature data for O(1) lookup instead of O(R) filtering
        feat_groups = problem.pu_vs_features.groupby("species")

        feat_ids = problem.features["id"].values
        feat_targets = problem.features["target"].values.astype(float)
        for fi in range(len(feat_ids)):
            fid = int(feat_ids[fi])
            target = float(feat_targets[fi]) * misslevel

            if fid in feat_groups.groups and not (
                builder_path or self.objective == "min_penalties"
            ):
                amount_expr = amount_exprs[fid]
                if self.objective in ("max_features", "max_weighted_features"):
                    # Relaxed target: only enforced when feat_met_j = 1.
                    # Maximising Σ (weight·)feat_met_j therefore picks as many
                    # (weighted) features as the cost budget will pay for.
                    model += (
                        amount_expr >= target * feat_met[fid],
                        f"target_{fid}",
                    )
                elif self.objective == "min_largest_shortfall":
                    # slack_j ≥ target_j − amount; t ≥ slack_j.
                    model += (
                        slack[fid] >= target - amount_expr,
                        f"slack_{fid}",
                    )
                    assert t_var is not None  # for mypy: set in init block above
                    model += t_var >= slack[fid], f"max_shortfall_{fid}"
                else:
                    model += amount_expr >= target, f"target_{fid}"

        if feat_met:
            # Ghost features (target > 0, no pu_vs_features rows) can never be met; v0.38
            # left feat_met unconstrained so the maximiser counted them for free.
            for fi in range(len(feat_ids)):
                fid = int(feat_ids[fi])
                if fid not in feat_groups.groups and float(feat_targets[fi]) * misslevel > 0:
                    feat_met[fid].upBound = 0

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

        if objective_expr is None:  # natural units: min_set / max_* / absolute minimax
            objective_expr = model.objective
            if self.objective in ("max_features", "max_weighted_features"):
                objective_expr = -model.objective
        return _BuiltModel(
            model=model, x=x, pu_ids=pu_ids, blm=blm, objective_expr=objective_expr,
            probmode=probmode, has_target2=bool(has_target2),
            has_sep_active=bool(has_sep_active), lex=lex,
        )

    def build_model(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> pulp.LpProblem:
        """The PuLP model ``solve`` would hand to the backend (for ``write_problem``).

        For ``min_penalties`` this is the pass-1 model (objective Σ SPF·shortfall).
        """
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
        probmode, has_target2 = built.probmode, built.has_target2
        has_sep_active, blm = built.has_sep_active, built.blm

        # Solve
        time_limit = int(problem.parameters.get("MIP_TIME_LIMIT", 300))
        gap = float(problem.parameters.get("MIP_GAP", 0.0))
        verbose = config.verbose
        # Phase 21: dispatch through the backend factory. CBC remains default
        # when mip_backend="auto" but HiGHS wins if available.
        solver = _make_pulp_solver(
            self.mip_backend,
            time_limit=time_limit, gap=gap, verbose=verbose,
        )
        # Record the resolved backend in metadata so users can confirm
        # which solver actually ran. "auto" gets resolved here, so capture
        # the concrete name post-factory rather than self.mip_backend.
        resolved_backend = (
            "highs" if isinstance(solver, pulp.HiGHS_CMD)
            else "gurobi" if isinstance(solver, pulp.GUROBI_CMD)
            else "cbc"
        )
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

        # Extract solution
        selected = np.array(
            [bool(round(pulp.value(x[pid]) or 0.0)) for pid in pu_ids],
            dtype=bool,
        )

        # build_solution populates Solution.prob_shortfalls + prob_penalty
        # automatically when PROBMODE==3. Under the 'drop' strategy, the
        # MIP solved the deterministic relaxation; the chance-constraint
        # gap is what build_solution reports post-hoc.
        # pulp.value() is None when any variable in the expression has no value (zero-cost
        # x never written to the LP); evaluate the objective reading those as 0.
        obj_expr = built.objective_expr
        obj_value = float(obj_expr.constant or 0.0) + sum(
            float(coef) * float(v.varValue or 0.0) for v, coef in obj_expr.items()
        )
        meta = {
            "solver": self.name(),
            "status": pulp.LpStatus[status],
            "mip_backend": resolved_backend,
            "objective": self.objective,
            "mip_objective_value": obj_value,
            "shortfall": self.shortfall,
        }
        if probmode == 3:
            meta["mip_chance_strategy"] = self.mip_chance_strategy
        if has_target2:
            meta["mip_clump_strategy"] = self.mip_clump_strategy
        if has_sep_active:
            meta["mip_sep_strategy"] = self.mip_sep_strategy
        sol = build_solution(problem, selected, blm, metadata=meta)
        return [copy.deepcopy(sol) for _ in range(config.num_solutions)]
