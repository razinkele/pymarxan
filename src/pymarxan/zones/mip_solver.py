"""Zone MIP solver using PuLP for exact multi-zone conservation planning."""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pulp

from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.zones.diagnostics import (
    ZoneMIPNoSolutionWarning,
    _feature_groups,
    _pu_zone_rules,
    diagnose_zone_infeasibility,
)
from pymarxan.zones.model import ZonalProblem, warn_zone_target_issues
from pymarxan.zones.objective import _feature_arrays, build_zone_solution


class ZoneMIPSolver(Solver):
    """Solver that formulates the MarZone multi-zone problem as a MILP.

    Decision variables:
        x[i,z] ∈ {0,1}  — PU i assigned to zone z

    Constraints:
        Σ_z x[i,z] <= 1           each PU in at most one zone
        x[i,z] = 1                locked-in PUs (status=2, first zone)
        x[i,z] = 0 ∀z             locked-out PUs (status=3)
        zone-specific feature targets on raw amounts (hard + slack; weighted under
            ZONETARGETCONTRIB=1)
        overall feature targets on contribution-weighted amounts (hard; Watts eq. 6)

    Objective (minimize):
        zone costs + BLM * standard boundary + zone boundary costs + penalty
    """

    def __init__(
        self,
        *,
        mip_clump_strategy: str = "drop",
        mip_sep_strategy: str = "drop",
        mip_backend: str = "auto",
    ) -> None:
        """Per-zone TARGET2 / SEPNUM are out of scope for v0.2 (see Phase
        19/20 design §"What's NOT in scope"). The kwargs are accepted for
        API symmetry with :class:`MIPSolver` and to forward-compatibly
        reject ``"big_m"`` / ``"socp"``; the deterministic ``"drop"`` path
        is the only one wired today.

        Phase 21: ``mip_backend`` lets users select between CBC (default),
        HiGHS, and Gurobi (when installed). Same factory as ``MIPSolver``.
        """
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

    def name(self) -> str:
        return "Zone MIP (PuLP)"

    def supports_zones(self) -> bool:
        return True

    def supports_separation(self) -> bool:
        # Per-zone SEPDISTANCE / SEPNUM deferred to v0.3 (round-3 H1).
        return False

    def available(self) -> bool:
        return True

    def solve(  # type: ignore[override]
        self, problem: ZonalProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        # Liskov: the Solver base takes ConservationProblem; zone solvers
        # specialise to ZonalProblem and verify at runtime. supports_zones()
        # advertises this so dispatchers route correctly.
        if not isinstance(problem, ZonalProblem):
            raise TypeError(
                f"ZoneMIPSolver requires a ZonalProblem, got {type(problem).__name__}"
            )
        from pymarxan.solvers.separation import raise_if_separation_active
        raise_if_separation_active(problem, "ZoneMIPSolver")
        warn_zone_target_issues(problem, "ZoneMIPSolver")
        if config is None:
            config = SolverConfig()

        blm = float(problem.parameters.get("BLM", 0.0))
        pu_ids = problem.planning_units["id"].tolist()
        rules = _pu_zone_rules(problem)
        zone_ids = sorted(problem.zone_ids)

        model = pulp.LpProblem("ZoneMIP", pulp.LpMinimize)

        # Decision variables: x[pu_id, zone_id] ∈ {0, 1}
        x: dict[tuple[int, int], pulp.LpVariable] = {}
        for k, pid in enumerate(pu_ids):
            pid = int(pid)
            for zid in zone_ids:
                x[(pid, zid)] = pulp.LpVariable(f"x_{pid}_{zid}", cat="Binary")

            # Each PU in at most one zone
            model += (
                pulp.lpSum(x[(pid, zid)] for zid in zone_ids) <= 1,
                f"assign_{pid}",
            )

            rule = rules[k]
            if rule.forced:
                (forced_zone,) = rule.allowed
                model += x[(pid, forced_zone)] == 1, f"locked_in_{pid}"
            else:
                for zid in zone_ids:
                    if zid not in rule.allowed:
                        model += x[(pid, zid)] == 0, f"locked_out_{pid}_{zid}"

        # --- Objective components ---

        # 1. Zone costs
        zc = problem.zone_costs
        zc_lookup: dict[tuple[int, int], float] = {}
        for row_k in range(len(zc)):
            zc_lookup[
                (int(zc["pu"].values[row_k]), int(zc["zone"].values[row_k]))
            ] = float(zc["cost"].values[row_k])

        cost_expr = pulp.lpSum(
            zc_lookup.get((pid, zid), 0.0) * x[(pid, zid)]
            for pid in pu_ids
            for zid in zone_ids
        )

        # 2. Standard boundary (BLM): penalize perimeter of selected set
        boundary_expr = _build_standard_boundary_expr(
            problem, x, pu_ids, zone_ids, model
        )

        # 3. Zone boundary costs between adjacent PUs in different zones
        zone_boundary_expr = _build_zone_boundary_expr(
            problem, x, pu_ids, zone_ids, model
        )

        # 4. Zone-target penalty (slack) and hard constraints share one achieved expression
        zone_exprs = _zone_achieved_exprs(problem, x)
        penalty_expr, penalty_vars = _build_penalty_expr(problem, zone_exprs, model)

        model += (
            cost_expr
            + blm * boundary_expr
            + zone_boundary_expr
            + penalty_expr,
            "objective",
        )

        # --- Feature target constraints ---
        _add_zone_target_constraints(problem, zone_exprs, model)
        _add_overall_target_constraints(problem, x, zone_ids, model)

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
        infeasible_statuses = {
            pulp.constants.LpStatusInfeasible,
            pulp.constants.LpStatusUnbounded,
            pulp.constants.LpStatusUndefined,
        }
        first_pid = int(pu_ids[0])
        has_values = pulp.value(x[(first_pid, zone_ids[0])]) is not None
        if model.status in infeasible_statuses or not has_values:
            _warn_no_solution(problem, model.status)
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

        sol = build_zone_solution(problem, zone_assignment, blm, solver_name=self.name())
        # Merge, don't replace: build_zone_solution already recorded zone_targets_met
        # (issue #1: the old assignment silently discarded it).
        sol.metadata.update(
            {
                "status": pulp.LpStatus[model.status],
                "mip_backend": resolved_backend,
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


def _add_overall_target_constraints(
    problem: ZonalProblem,
    x: dict[tuple[int, int], pulp.LpVariable],
    zone_ids: list[int],
    model: pulp.LpProblem,
) -> None:
    """Hard constraints Σ_i Σ_z amount[i, f] × contribution[f, z] × x[i, z] >= target × MISSLEVEL.

    Watts et al. 2009 eq. 6; MarZone ``reserve.hpp:158-171``. One constraint per feature with
    ``target > 0``; a feature with no amounts anywhere makes the model infeasible (the solver
    then returns ``[]``), which is the honest answer rather than a silently met target.
    Contributions come from ``contribution_matrix()`` / ``zone_index()`` / ``feature_index()``
    exactly as ``_zone_achieved_exprs`` does (one convention per module).
    """
    contrib = problem.contribution_matrix()
    zidx = problem.zone_index()
    fidx = problem.feature_index()
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    groups = _feature_groups(problem)
    for fid, target in zip(
        problem.features["id"].values, problem.features["target"].values, strict=True,
    ):
        fid = int(fid)
        t = float(target)
        if t <= 0:
            continue
        col = fidx[fid]
        terms = []
        for pid, amt in groups.get(fid, []):
            for zid in zone_ids:
                c = float(contrib[zidx[zid], col])
                if c != 0.0 and (pid, zid) in x:
                    terms.append(amt * c * x[(pid, zid)])
        model += (pulp.lpSum(terms) >= t * misslevel, f"overall_target_{fid}")


def _warn_no_solution(problem: ZonalProblem, status: int) -> None:
    """One ZoneMIPNoSolutionWarning explaining an empty result; the return stays []."""
    name = pulp.LpStatus.get(status, str(status))
    if status == pulp.constants.LpStatusInfeasible:
        try:
            unreachable = diagnose_zone_infeasibility(problem)
            failed = ""
        except Exception as exc:  # a diagnostic must never turn [] into a crash
            unreachable, failed = [], f" (diagnostic failed: {exc!r})"
        if unreachable:
            more = f" (+{len(unreachable) - 5} more)" if len(unreachable) > 5 else ""
            msg = (
                "ZoneMIPSolver: infeasible. Unreachable targets: "
                + "; ".join(unreachable[:5]) + more
            )
        else:
            msg = (
                "ZoneMIPSolver: infeasible, but every target is reachable on its own — "
                "targets compete for the same planning units. ZoneSASolver treats "
                f"targets as penalties and will return a best-effort plan.{failed}"
            )
    elif status == pulp.constants.LpStatusNotSolved:
        msg = (
            "ZoneMIPSolver: no feasible solution found within the time limit (status "
            f"{name}); raise MIP_TIME_LIMIT or use ZoneSASolver."
        )
    else:
        msg = f"ZoneMIPSolver: no solution (status {name})."
    warnings.warn(msg, ZoneMIPNoSolutionWarning, stacklevel=3)
