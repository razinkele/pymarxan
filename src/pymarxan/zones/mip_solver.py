"""Zone MIP solver using PuLP for exact multi-zone conservation planning."""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass

import numpy as np
import pulp

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers._mip_objectives import (
    BUILDERS,
    NO_SOLUTION_STATUSES,
    ObjectiveContext,
    TargetTerm,
    resolve_objective,
    set_objective,
    validate_shortfall,
)
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.presolve import (
    _emit_presolve,
    _presolve_data,
    _validate_presolve_flag,
)
from pymarxan.zones.diagnostics import (
    ZoneMIPNoSolutionWarning,
    _feature_groups,
    _fmt,
    _pu_zone_rules,
    _zone_cost_lookup,
    diagnose_zone_infeasibility,
    min_forced_cost,
)
from pymarxan.zones.model import ZonalProblem, warn_zone_input_issues
from pymarxan.zones.objective import _feature_arrays, build_zone_solution

ZONE_MIP_OBJECTIVES = (
    "min_set",
    "max_features",
    "max_weighted_features",
    "min_shortfall",
    "min_largest_shortfall",
)


@dataclass
class _BuiltZoneModel:
    model: pulp.LpProblem
    x: dict[tuple[int, int], pulp.LpVariable]
    pu_ids: list[int]
    zone_ids: list[int]
    blm: float
    objective_expr: pulp.LpAffineExpression  # natural units (count for max_*)


class ZoneMIPSolver(Solver):
    """Solver that formulates the MarZone multi-zone problem as a MILP.

    Decision variables:
        x[i,z] ∈ {0,1}  — PU i assigned to zone z

    Constraints:
        Σ_z x[i,z] == 1           PUs that must be assigned (``pu_zone_rules``)
        Σ_z x[i,z] <= 1           every other PU
        x[i,z] = 0                zones outside a PU's rule (status, pu_locks, pu_zones)
        zone-specific feature targets on raw amounts (hard + slack; weighted under
            ZONETARGETCONTRIB=1)
        overall feature targets on contribution-weighted amounts (hard; Watts eq. 6)
        Infeasible / no incumbent → ``[]`` plus one ``ZoneMIPNoSolutionWarning``
            (``diagnose_zone_infeasibility``)

    Objective ``min_set`` (default; minimize):
        zone costs + BLM * standard boundary + zone boundary costs + penalty

    Budgeted objectives (require ``COSTBUDGET``; Σ zone cost <= COSTBUDGET):
        ``max_features``           count of met targets (prioritizr ``max_n_targets_met``;
                                   the alias ``max_n_targets_met`` is accepted)
        ``max_weighted_features``  Σ SPF × met (prioritizr ``max_n_targets_met`` +
                                   ``add_feature_weights`` — **not** ``max_wtd_sum``)
        ``min_shortfall``          Σ SPF × shortfall
        ``min_largest_shortfall``  max shortfall (unweighted)

    Each zone target and each overall target (> 0) is its own term (prioritizr zones).
    Shortfall is relative by default (shortfall / (target × MISSLEVEL));
    ``shortfall="absolute"`` uses raw units. Budgeted objectives drop the hard targets and
    the boundary terms (one ``UserWarning`` when BLM / zone boundary costs would apply).
    ``Solution.metadata`` records ``objective``, ``shortfall`` and ``mip_objective_value``
    (natural sign: a count for ``max_*``). ``shortfall`` is the configured units, recorded for
    every objective but meaningful only for the two shortfall objectives.

    ``presolve_check`` (default ``False``): ``True`` runs
    :func:`~pymarxan.solvers.presolve.presolve_check` plus a scan of the built model (P10)
    in ``solve`` / ``build_model`` and emits one ``PresolveWarning``; ``"strict"`` raises
    ``ValueError`` instead. The data checks run before the model is built: a non-finite
    value (P1) raises ``ValueError`` under both, since PuLP cannot build the model.
    """

    def __init__(
        self,
        *,
        objective: str = "min_set",
        shortfall: str = "relative",
        mip_clump_strategy: str = "drop",
        mip_sep_strategy: str = "drop",
        mip_backend: str = "auto",
        presolve_check: bool | str = False,
    ) -> None:
        """Per-zone TARGET2 / SEPNUM are out of scope for v0.2 (see Phase
        19/20 design §"What's NOT in scope"). The kwargs are accepted for
        API symmetry with :class:`MIPSolver` and to forward-compatibly
        reject ``"big_m"`` / ``"socp"``; the deterministic ``"drop"`` path
        is the only one wired today.

        Phase 21: ``mip_backend`` lets users select between CBC (default),
        HiGHS, and Gurobi (when installed). Same factory as ``MIPSolver``.
        """
        self.objective = resolve_objective(objective, ZONE_MIP_OBJECTIVES)
        self.shortfall = validate_shortfall(shortfall)
        # Use the shared MIPSolver validator for consistency.
        from pymarxan.solvers.mip_solver import (
            _MIP_BACKEND_NAMES,
            _validate_mip_strategy,
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
                    "Use 'drop' (default) or 'big_m' (deferred)."
                ),
            },
        )
        if mip_backend not in _MIP_BACKEND_NAMES:
            raise ValueError(
                f"mip_backend must be one of {_MIP_BACKEND_NAMES}, got "
                f"{mip_backend!r}."
            )
        self.mip_clump_strategy = mip_clump_strategy
        self.mip_sep_strategy = mip_sep_strategy
        self.mip_backend = mip_backend
        self.presolve_check = _validate_presolve_flag(presolve_check)

    def name(self) -> str:
        return "Zone MIP (PuLP)"

    def supports_zones(self) -> bool:
        return True

    def supports_separation(self) -> bool:
        # Per-zone SEPDISTANCE / SEPNUM deferred to v0.3 (round-3 H1).
        return False

    def available(self) -> bool:
        return True

    def _check(self, problem: ConservationProblem) -> ZonalProblem:
        # Liskov: the Solver base takes ConservationProblem; zone solvers
        # specialise to ZonalProblem and verify at runtime. supports_zones()
        # advertises this so dispatchers route correctly.
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
        if problem.boundary is not None and (blm > 0 or has_zbc):
            warnings.warn(
                f"ZoneMIPSolver: objective {self.objective!r} ignores boundary penalties "
                "(BLM / zone boundary costs); prioritizr applies them — planned follow-up",
                UserWarning,
                stacklevel=3,
            )

    def _build(self, problem: ZonalProblem, config: SolverConfig) -> _BuiltZoneModel:
        """A fresh model and objective context on every call (no state across calls)."""
        blm = float(problem.parameters.get("BLM", 0.0))
        pu_ids = problem.planning_units["id"].tolist()
        rules = _pu_zone_rules(problem)
        zone_ids = sorted(problem.zone_ids)

        model = pulp.LpProblem("ZoneMIP", pulp.LpMinimize)

        # Decision variables: x[pu_id, zone_id] ∈ {0, 1}
        x: dict[tuple[int, int], pulp.LpVariable] = {}
        for k, pid in enumerate(pu_ids):
            pid = int(pid)
            rule = rules[k]
            for zid in zone_ids:
                var = pulp.LpVariable(f"x_{pid}_{zid}", cat="Binary")
                if zid not in rule.allowed:
                    var.upBound = 0
                x[(pid, zid)] = var
            assigned = pulp.lpSum(x[(pid, zid)] for zid in zone_ids)
            if rule.may_be_unassigned:
                model += assigned <= 1, f"assign_{pid}"
            else:
                model += assigned == 1, f"assign_{pid}"

        # Zone costs (a (pu, zone) pair missing from zone_costs costs 0, MarZone)
        zc_lookup = _zone_cost_lookup(problem)
        cost_expr = pulp.lpSum(
            zc_lookup.get((pid, zid), 0.0) * x[(pid, zid)]
            for pid in pu_ids
            for zid in zone_ids
        )

        if self.objective == "min_set":
            # Standard boundary (BLM): penalize perimeter of selected set
            boundary_expr = _build_standard_boundary_expr(
                problem, x, pu_ids, zone_ids, model
            )
            # Zone boundary costs between adjacent PUs in different zones
            zone_boundary_expr = _build_zone_boundary_expr(
                problem, x, pu_ids, zone_ids, model
            )
            # Zone-target penalty (slack) and hard constraints share one achieved expression
            zone_exprs = _zone_achieved_exprs(problem, x)
            penalty_expr, _ = _build_penalty_expr(problem, zone_exprs, model)
            model += (
                cost_expr
                + blm * boundary_expr
                + zone_boundary_expr
                + penalty_expr,
                "objective",
            )
            # Feature target constraints
            _add_zone_target_constraints(problem, zone_exprs, model)
            _add_overall_target_constraints(problem, x, zone_ids, model)
            objective_expr = model.objective
        else:
            budget = problem.parameters.get("COSTBUDGET")
            if budget is None:
                raise ValueError(
                    f"objective={self.objective!r} requires a COSTBUDGET parameter "
                    "(problem.parameters['COSTBUDGET'] = ...); it caps the total zone cost."
                )
            ctx = ObjectiveContext(model, cost_expr, _zone_target_terms(problem, x, zone_ids))
            term = BUILDERS[self.objective](ctx, self.shortfall)
            set_objective(model, term)
            model += cost_expr <= float(budget), "cost_budget"
            objective_expr = term.expr
        return _BuiltZoneModel(
            model=model, x=x, pu_ids=pu_ids, zone_ids=zone_ids, blm=blm,
            objective_expr=objective_expr,
        )

    def build_model(
        self, problem: ZonalProblem, config: SolverConfig | None = None
    ) -> pulp.LpProblem:
        """The PuLP model ``solve`` would hand to the backend (for ``write_problem``)."""
        problem = self._check(problem)
        warn_zone_input_issues(problem, "ZoneMIPSolver")
        self._warn_boundary_ignored(problem)
        data = _presolve_data(self.presolve_check, problem, self.name())
        built = self._build(problem, config or SolverConfig())
        _emit_presolve(self.presolve_check, data, built.model, self.name(), stacklevel=3)
        return built.model

    def solve(  # type: ignore[override]
        self, problem: ZonalProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        problem = self._check(problem)
        warn_zone_input_issues(problem, "ZoneMIPSolver")
        self._warn_boundary_ignored(problem)
        if config is None:
            config = SolverConfig()
        data = _presolve_data(self.presolve_check, problem, self.name())
        built = self._build(problem, config)
        _emit_presolve(self.presolve_check, data, built.model, self.name(), stacklevel=3)
        model, x, pu_ids, zone_ids = built.model, built.x, built.pu_ids, built.zone_ids

        # Solve via the Phase 21 backend factory.
        time_limit = int(problem.parameters.get("MIP_TIME_LIMIT", 300))
        gap = float(problem.parameters.get("MIP_GAP", 0.0))
        from pymarxan.solvers.mip_solver import _make_pulp_solver
        solver = _make_pulp_solver(
            self.mip_backend,
            time_limit=time_limit, gap=gap, verbose=config.verbose,
        )
        resolved_backend = (
            "highs" if isinstance(solver, pulp.HiGHS_CMD)
            else "gurobi" if isinstance(solver, pulp.GUROBI_CMD)
            else "cbc"
        )
        model.solve(solver)

        # Accept feasible-on-timeout solutions: CBC sets status=NotSolved when
        # the time limit fires before optimality is proved, even though a
        # usable integer incumbent exists. See mip_solver.py for rationale.
        has_values = any(var.varValue is not None for var in x.values())
        if model.status in NO_SOLUTION_STATUSES or not has_values:
            _warn_no_solution(problem, model.status, self.objective)
            return []

        # Extract zone assignment
        zone_assignment = np.zeros(len(pu_ids), dtype=int)
        for k, pid in enumerate(pu_ids):
            pid = int(pid)
            for zid in zone_ids:
                val = pulp.value(x[(pid, zid)]) or 0.0
                if round(val) == 1:
                    zone_assignment[k] = zid
                    break

        # pulp.value() is None when any variable has no value (a zero-cost x never written
        # to the LP); evaluate the objective reading those as 0.
        obj_expr = built.objective_expr
        obj_value = float(obj_expr.constant or 0.0) + sum(
            float(coef) * float(v.varValue or 0.0) for v, coef in obj_expr.items()
        )
        sol = build_zone_solution(problem, zone_assignment, built.blm, solver_name=self.name())
        # Merge, don't replace: build_zone_solution already recorded zone_targets_met
        # (issue #1: the old assignment silently discarded it).
        sol.metadata.update(
            {
                "status": pulp.LpStatus[model.status],
                "mip_backend": resolved_backend,
                "objective": self.objective,
                "shortfall": self.shortfall,
                "mip_objective_value": obj_value,
            }
        )
        return [copy.deepcopy(sol) for _ in range(config.num_solutions)]


def _build_standard_boundary_expr(
    problem: ZonalProblem,
    x: dict[tuple[int, int], pulp.LpVariable],
    pu_ids: list[int],
    zone_ids: list[int],
    model: pulp.LpProblem,
) -> pulp.LpAffineExpression:
    """Build standard boundary expression with auxiliary variables."""
    if problem.boundary is None:
        return pulp.lpSum([])

    # s[i] = Σ_z x[i,z]  (1 if PU selected in any zone, 0 otherwise)
    s: dict[int, pulp.LpAffineExpression] = {}
    for pid in pu_ids:
        pid = int(pid)
        s[pid] = pulp.lpSum(x[(pid, zid)] for zid in zone_ids)

    bnd = problem.boundary
    id1_col = bnd["id1"].values
    id2_col = bnd["id2"].values
    bval_col = bnd["boundary"].values.astype(float)

    expr = pulp.lpSum([])
    y: dict[tuple[int, int], pulp.LpVariable] = {}

    for bk in range(len(id1_col)):
        id1 = int(id1_col[bk])
        id2 = int(id2_col[bk])
        bval = float(bval_col[bk])

        if id1 == id2:
            # Self-boundary: perimeter when PU is selected
            if id1 in s:
                expr += bval * s[id1]
        else:
            # Pairwise: linearize |s[i] - s[j]|
            if id1 not in s or id2 not in s:
                continue
            key = (min(id1, id2), max(id1, id2))
            if key not in y:
                y_var = pulp.LpVariable(
                    f"y_std_{key[0]}_{key[1]}", lowBound=0, upBound=1
                )
                y[key] = y_var
                model += (
                    y_var >= s[key[0]] - s[key[1]],
                    f"bnd_abs1_{key[0]}_{key[1]}",
                )
                model += (
                    y_var >= s[key[1]] - s[key[0]],
                    f"bnd_abs2_{key[0]}_{key[1]}",
                )
            expr += bval * y[key]

    return expr


def _build_zone_boundary_expr(
    problem: ZonalProblem,
    x: dict[tuple[int, int], pulp.LpVariable],
    pu_ids: list[int],
    zone_ids: list[int],
    model: pulp.LpProblem,
) -> pulp.LpAffineExpression:
    """Build zone-specific boundary cost expression."""
    if problem.boundary is None or problem.zone_boundary_costs is None:
        return pulp.lpSum([])

    # Build zone boundary cost lookup
    zbc = problem.zone_boundary_costs
    zbc_lookup: dict[tuple[int, int], float] = {}
    for k in range(len(zbc)):
        z1 = int(zbc["zone1"].values[k])
        z2 = int(zbc["zone2"].values[k])
        zbc_lookup[(z1, z2)] = float(zbc["cost"].values[k])

    pu_set = set(int(p) for p in pu_ids)
    bnd = problem.boundary
    id1_col = bnd["id1"].values
    id2_col = bnd["id2"].values
    bval_col = bnd["boundary"].values.astype(float)

    expr = pulp.lpSum([])

    for bk in range(len(id1_col)):
        id1 = int(id1_col[bk])
        id2 = int(id2_col[bk])
        if id1 == id2:
            continue
        if id1 not in pu_set or id2 not in pu_set:
            continue
        bval = float(bval_col[bk])

        for z1 in zone_ids:
            for z2 in zone_ids:
                if z1 == z2:
                    continue
                zbc_cost = zbc_lookup.get((z1, z2), 0.0)
                if zbc_cost == 0.0:
                    continue
                # w[i,j,z1,z2] linearizes x[i,z1] * x[j,z2]
                w = pulp.LpVariable(
                    f"w_{id1}_{id2}_{z1}_{z2}", cat="Binary"
                )
                model += w <= x[(id1, z1)], f"w_le1_{id1}_{id2}_{z1}_{z2}"
                model += w <= x[(id2, z2)], f"w_le2_{id1}_{id2}_{z1}_{z2}"
                model += (
                    w >= x[(id1, z1)] + x[(id2, z2)] - 1,
                    f"w_ge_{id1}_{id2}_{z1}_{z2}",
                )
                expr += bval * zbc_cost * w

    return expr


def _spf_lookup(problem: ZonalProblem) -> dict[int, float]:
    """feature id -> spf (1.0 when the column is absent; one rule, ``_feature_arrays``)."""
    ids, _, spf = _feature_arrays(problem)
    return dict(zip(ids.tolist(), spf.tolist(), strict=True))


def _zone_achieved_exprs(
    problem: ZonalProblem,
    x: dict[tuple[int, int], pulp.LpVariable],
) -> dict[tuple[int, int], pulp.LpAffineExpression]:
    """Σ_i amount[i, f] × w[z, f] × x[i, z] for every effective (zone, feature) target.

    Targets come from ``zone_target_lookup``. Built once and shared by the penalty
    (slack) and the hard constraint. ``w`` is ``problem.zone_target_weight_matrix()``
    (raw by default, MarZone ``reserve.hpp:164``).
    """
    lookup = problem.zone_target_lookup()
    if not lookup:
        return {}
    weight = problem.zone_target_weight_matrix()
    zidx = problem.zone_index()
    fidx = problem.feature_index()
    groups = _feature_groups(problem)
    exprs: dict[tuple[int, int], pulp.LpAffineExpression] = {}
    for zid, fid in lookup:
        w = float(weight[zidx[zid], fidx[fid]])
        exprs[(zid, fid)] = pulp.lpSum(
            amt * w * x[(pid, zid)]
            for pid, amt in groups.get(fid, [])
            if (pid, zid) in x
        )
    return exprs


def _build_penalty_expr(
    problem: ZonalProblem,
    exprs: dict[tuple[int, int], pulp.LpAffineExpression],
    model: pulp.LpProblem,
) -> tuple[pulp.LpAffineExpression, dict]:
    """Build penalty expression for unmet zone targets (SPF * slack)."""
    lookup = problem.zone_target_lookup()
    if not lookup:
        return pulp.lpSum([]), {}
    spf_lookup = _spf_lookup(problem)
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    expr = pulp.lpSum([])
    slack_vars = {}
    for (zid, fid), target in lookup.items():
        slack = pulp.LpVariable(f"slack_{zid}_{fid}", lowBound=0, cat="Continuous")
        model += (
            slack >= target * misslevel - exprs[(zid, fid)],
            f"shortfall_{zid}_{fid}",
        )
        slack_vars[(zid, fid)] = slack
        expr += spf_lookup.get(fid, 1.0) * slack
    return expr, slack_vars


def _add_zone_target_constraints(
    problem: ZonalProblem,
    exprs: dict[tuple[int, int], pulp.LpAffineExpression],
    model: pulp.LpProblem,
) -> None:
    """Add hard zone-specific feature target constraints."""
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    for (zid, fid), target in problem.zone_target_lookup().items():
        model += (exprs[(zid, fid)] >= target * misslevel, f"zone_target_{zid}_{fid}")


def _overall_targets(problem: ZonalProblem) -> dict[int, float]:
    """feature id -> overall target (features order)."""
    return dict(zip(
        (int(f) for f in problem.features["id"].values),
        problem.features["target"].values.astype(float),
        strict=True,
    ))


def _overall_achieved_exprs(
    problem: ZonalProblem,
    x: dict[tuple[int, int], pulp.LpVariable],
    zone_ids: list[int],
) -> dict[int, pulp.LpAffineExpression]:
    """Σ_i Σ_z amount[i, f] × contribution[f, z] × x[i, z] per feature with target > 0.

    Watts et al. 2009 eq. 6; MarZone ``reserve.hpp:158-171``. Features order. A feature with
    no amounts anywhere gets ``lpSum([])``: as a hard constraint the model is infeasible (the
    solver then returns ``[]``), the honest answer rather than a silently met target.
    Contributions come from ``contribution_matrix()`` / ``zone_index()`` / ``feature_index()``
    exactly as ``_zone_achieved_exprs`` does (one convention per module).
    """
    contrib = problem.contribution_matrix()
    zidx = problem.zone_index()
    fidx = problem.feature_index()
    groups = _feature_groups(problem)
    exprs: dict[int, pulp.LpAffineExpression] = {}
    for fid, t in _overall_targets(problem).items():
        if t <= 0:
            continue
        col = fidx[fid]
        terms = []
        for pid, amt in groups.get(fid, []):
            for zid in zone_ids:
                c = float(contrib[zidx[zid], col])
                if c != 0.0 and (pid, zid) in x:
                    terms.append(amt * c * x[(pid, zid)])
        exprs[fid] = pulp.lpSum(terms)
    return exprs


def _add_overall_target_constraints(
    problem: ZonalProblem,
    x: dict[tuple[int, int], pulp.LpVariable],
    zone_ids: list[int],
    model: pulp.LpProblem,
) -> None:
    """Hard constraints ``_overall_achieved_exprs[f] >= target × MISSLEVEL`` (features order)."""
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    targets = _overall_targets(problem)
    for fid, expr in _overall_achieved_exprs(problem, x, zone_ids).items():
        model += (expr >= targets[fid] * misslevel, f"overall_target_{fid}")


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
    targets = _overall_targets(problem)
    terms += [
        TargetTerm(f"o_f{fid}", "overall", expr, targets[fid] * misslevel, spf.get(fid, 1.0))
        for fid, expr in _overall_achieved_exprs(problem, x, zone_ids).items()
    ]
    return terms


def _warn_no_solution(problem: ZonalProblem, status: int, objective: str = "min_set") -> None:
    """One ZoneMIPNoSolutionWarning explaining an empty result; the return stays []."""
    name = pulp.LpStatus.get(status, str(status))
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
    if status == pulp.constants.LpStatusInfeasible:
        try:
            unreachable = diagnose_zone_infeasibility(problem)
        except Exception as exc:  # a diagnostic must never turn [] into a crash
            msg = f"ZoneMIPSolver: infeasible (diagnostic failed: {exc!r})."
        else:
            note = (
                " (given the PU zone restrictions)"
                if problem.pu_lock_lookup() or problem.pu_zone_lookup()
                else ""
            )
            if unreachable:
                more = f" (+{len(unreachable) - 5} more)" if len(unreachable) > 5 else ""
                msg = (
                    "ZoneMIPSolver: infeasible. Unreachable targets: "
                    + "; ".join(unreachable[:5]) + more + note
                )
            else:
                msg = (
                    f"ZoneMIPSolver: infeasible, but every target is reachable on its own{note} "
                    "— targets compete for the same planning units. ZoneSASolver treats "
                    "targets as penalties and will return a best-effort plan."
                )
    elif status == pulp.constants.LpStatusNotSolved:
        msg = (
            "ZoneMIPSolver: no feasible solution found within the time limit (status "
            f"{name}); raise MIP_TIME_LIMIT or use ZoneSASolver."
        )
    else:
        msg = f"ZoneMIPSolver: no solution (status {name})."
    warnings.warn(msg, ZoneMIPNoSolutionWarning, stacklevel=3)
