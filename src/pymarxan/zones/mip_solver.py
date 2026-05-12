"""Zone MIP solver using PuLP for exact multi-zone conservation planning."""
from __future__ import annotations

import copy

import numpy as np
import pulp

from pymarxan.models.problem import (
    STATUS_LOCKED_IN,
    STATUS_LOCKED_OUT,
)
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import (
    check_zone_targets,
    compute_standard_boundary,
    compute_zone_cost,
    compute_zone_objective,
    compute_zone_penalty,
    compute_zone_shortfall,
)


class ZoneMIPSolver(Solver):
    """Solver that formulates the MarZone multi-zone problem as a MILP.

    Decision variables:
        x[i,z] ∈ {0,1}  — PU i assigned to zone z

    Constraints:
        Σ_z x[i,z] <= 1           each PU in at most one zone
        x[i,z] = 1                locked-in PUs (status=2, first zone)
        x[i,z] = 0 ∀z             locked-out PUs (status=3)
        zone-specific feature targets with contributions

    Objective (minimize):
        zone costs + BLM * standard boundary + zone boundary costs + penalty
    """

    def name(self) -> str:
        return "Zone MIP (PuLP)"

    def supports_zones(self) -> bool:
        return True

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
        if config is None:
            config = SolverConfig()

        blm = float(problem.parameters.get("BLM", 0.0))
        pu_ids = problem.planning_units["id"].tolist()
        pu_status = problem.planning_units["status"].values.astype(int)
        zone_ids = sorted(problem.zone_ids)

        model = pulp.LpProblem("ZoneMIP", pulp.LpMinimize)

        # Decision variables: x[pu_id, zone_id] ∈ {0, 1}
        x: dict[tuple[int, int], pulp.LpVariable] = {}
        for k, pid in enumerate(pu_ids):
            pid = int(pid)
            status = int(pu_status[k])
            for zid in zone_ids:
                x[(pid, zid)] = pulp.LpVariable(
                    f"x_{pid}_{zid}", cat="Binary"
                )

            # Each PU in at most one zone
            model += (
                pulp.lpSum(x[(pid, zid)] for zid in zone_ids) <= 1,
                f"assign_{pid}",
            )

            # Locked-in: assign to first zone
            if status == STATUS_LOCKED_IN:
                model += x[(pid, zone_ids[0])] == 1, f"locked_in_{pid}"
            # Locked-out: not in any zone
            elif status == STATUS_LOCKED_OUT:
                for zid in zone_ids:
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

        # 4. Zone-target penalty (big-M for unmet targets)
        penalty_expr, penalty_vars = _build_penalty_expr(
            problem, x, pu_ids, zone_ids, model
        )

        model += (
            cost_expr
            + blm * boundary_expr
            + zone_boundary_expr
            + penalty_expr,
            "objective",
        )

        # --- Feature target constraints ---
        _add_zone_target_constraints(problem, x, pu_ids, zone_ids, model)

        # Solve
        time_limit = int(problem.parameters.get("MIP_TIME_LIMIT", 300))
        gap = float(problem.parameters.get("MIP_GAP", 0.0))
        solver = pulp.PULP_CBC_CMD(
            msg=int(config.verbose), timeLimit=time_limit, gapRel=gap
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

        sol = _build_zone_solution(problem, zone_assignment, blm)
        sol.metadata = {
            "solver": self.name(),
            "status": pulp.LpStatus[model.status],
        }
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


def _build_penalty_expr(
    problem: ZonalProblem,
    x: dict[tuple[int, int], pulp.LpVariable],
    pu_ids: list[int],
    zone_ids: list[int],
    model: pulp.LpProblem,
) -> tuple[pulp.LpAffineExpression, dict]:
    """Build penalty expression for unmet zone targets (SPF * shortfall)."""
    if problem.zone_targets is None:
        return pulp.lpSum([]), {}

    # Build contribution lookup
    contrib_lookup: dict[tuple[int, int], float] = {}
    if problem.zone_contributions is not None:
        zc = problem.zone_contributions
        for k in range(len(zc)):
            fid = int(zc["feature"].values[k])
            zid = int(zc["zone"].values[k])
            contrib_lookup[(fid, zid)] = float(zc["contribution"].values[k])

    # Build SPF lookup
    spf_lookup: dict[int, float] = {}
    feat_ids = problem.features["id"].values
    feat_spf = (
        problem.features["spf"].values
        if "spf" in problem.features.columns
        else np.ones(len(feat_ids))
    )
    for i in range(len(feat_ids)):
        spf_lookup[int(feat_ids[i])] = float(feat_spf[i])

    # Pre-group pu_vs_features
    puvspr = problem.pu_vs_features
    feat_groups: dict[int, list[tuple[int, float]]] = {}
    pu_col = puvspr["pu"].values
    sp_col = puvspr["species"].values
    amt_col = puvspr["amount"].values
    for k in range(len(pu_col)):
        fid = int(sp_col[k])
        feat_groups.setdefault(fid, []).append((int(pu_col[k]), float(amt_col[k])))

    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    zt = problem.zone_targets
    zone_col = zt["zone"].values
    feat_col = zt["feature"].values
    target_col = zt["target"].values

    expr = pulp.lpSum([])
    slack_vars = {}

    for k in range(len(zone_col)):
        zid = int(zone_col[k])
        fid = int(feat_col[k])
        target = float(target_col[k]) * misslevel
        spf = spf_lookup.get(fid, 1.0)

        # Achieved for this (zone, feature): Σ_i amount[i,f] * contrib[f,z] * x[i,z]
        achieved_expr = pulp.lpSum([])
        for pid, amt in feat_groups.get(fid, []):
            contribution = contrib_lookup.get((fid, zid), 1.0)
            if (pid, zid) in x:
                achieved_expr += amt * contribution * x[(pid, zid)]

        # slack >= target - achieved (shortfall), slack >= 0
        slack = pulp.LpVariable(
            f"slack_{zid}_{fid}", lowBound=0, cat="Continuous"
        )
        model += (
            slack >= target - achieved_expr,
            f"shortfall_{zid}_{fid}",
        )
        slack_vars[(zid, fid)] = slack
        expr += spf * slack

    return expr, slack_vars


def _add_zone_target_constraints(
    problem: ZonalProblem,
    x: dict[tuple[int, int], pulp.LpVariable],
    pu_ids: list[int],
    zone_ids: list[int],
    model: pulp.LpProblem,
) -> None:
    """Add hard zone-specific feature target constraints."""
    if problem.zone_targets is None:
        return

    # Build contribution lookup
    contrib_lookup: dict[tuple[int, int], float] = {}
    if problem.zone_contributions is not None:
        zc = problem.zone_contributions
        for k in range(len(zc)):
            fid = int(zc["feature"].values[k])
            zid = int(zc["zone"].values[k])
            contrib_lookup[(fid, zid)] = float(zc["contribution"].values[k])

    # Pre-group pu_vs_features
    puvspr = problem.pu_vs_features
    feat_groups: dict[int, list[tuple[int, float]]] = {}
    pu_col = puvspr["pu"].values
    sp_col = puvspr["species"].values
    amt_col = puvspr["amount"].values
    for k in range(len(pu_col)):
        fid = int(sp_col[k])
        feat_groups.setdefault(fid, []).append((int(pu_col[k]), float(amt_col[k])))

    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    zt = problem.zone_targets
    zone_col = zt["zone"].values
    feat_col = zt["feature"].values
    target_col = zt["target"].values

    for k in range(len(zone_col)):
        zid = int(zone_col[k])
        fid = int(feat_col[k])
        target = float(target_col[k]) * misslevel

        achieved_expr = pulp.lpSum([])
        for pid, amt in feat_groups.get(fid, []):
            contribution = contrib_lookup.get((fid, zid), 1.0)
            if (pid, zid) in x:
                achieved_expr += amt * contribution * x[(pid, zid)]

        model += (
            achieved_expr >= target,
            f"zone_target_{zid}_{fid}",
        )


def _build_zone_solution(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    blm: float,
) -> Solution:
    """Build a Solution from a zone assignment array."""

    selected = zone_assignment > 0
    cost = compute_zone_cost(problem, zone_assignment)
    boundary = compute_standard_boundary(problem, zone_assignment)
    objective = compute_zone_objective(problem, zone_assignment, blm)
    penalty = compute_zone_penalty(problem, zone_assignment)
    shortfall = compute_zone_shortfall(problem, zone_assignment)

    # Feature-level targets (aggregate across zones)
    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}
    from pymarxan.solvers.utils import check_targets
    targets_met = check_targets(problem, selected, pu_index)

    zone_targets_met = check_zone_targets(problem, zone_assignment)

    return Solution(
        selected=selected,
        cost=cost,
        boundary=boundary,
        objective=objective,
        targets_met=targets_met,
        penalty=penalty,
        shortfall=shortfall,
        zone_assignment=zone_assignment,
        metadata={"zone_targets_met": zone_targets_met},
    )
