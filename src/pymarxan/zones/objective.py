"""Objective function components for multi-zone conservation planning."""
from __future__ import annotations

import numpy as np

from pymarxan.zones.model import ZonalProblem


def compute_zone_cost(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute total cost given zone assignments. Zone 0 = unassigned (no cost)."""
    pu_ids = problem.planning_units["id"].values
    zc = problem.zone_costs

    # Pre-build lookup: (pu, zone) -> cost
    zc_pu = zc["pu"].values
    zc_zone = zc["zone"].values
    zc_cost = zc["cost"].values.astype(np.float64)
    cost_lookup: dict[tuple[int, int], float] = {}
    for k in range(len(zc_pu)):
        cost_lookup[(int(zc_pu[k]), int(zc_zone[k]))] = float(zc_cost[k])

    total = 0.0
    for i in range(len(pu_ids)):
        zid = int(zone_assignment[i])
        if zid == 0:
            continue
        total += cost_lookup.get((int(pu_ids[i]), zid), 0.0)
    return total


def compute_zone_boundary(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute zone boundary cost between adjacent PUs in different zones."""
    if problem.boundary is None or problem.zone_boundary_costs is None:
        return 0.0

    pu_ids = problem.planning_units["id"].values
    pu_index = {int(pid): i for i, pid in enumerate(pu_ids)}

    # Pre-build zone boundary cost lookup
    zbc = problem.zone_boundary_costs
    zbc_lookup: dict[tuple[int, int], float] = {}
    z1_col = zbc["zone1"].values
    z2_col = zbc["zone2"].values
    cost_col = zbc["cost"].values
    for k in range(len(z1_col)):
        zbc_lookup[(int(z1_col[k]), int(z2_col[k]))] = float(cost_col[k])

    bnd = problem.boundary
    id1_col = bnd["id1"].values
    id2_col = bnd["id2"].values

    total = 0.0
    for k in range(len(id1_col)):
        id1 = int(id1_col[k])
        id2 = int(id2_col[k])
        if id1 == id2:
            continue

        idx1 = pu_index.get(id1)
        idx2 = pu_index.get(id2)
        if idx1 is None or idx2 is None:
            continue

        z1 = int(zone_assignment[idx1])
        z2 = int(zone_assignment[idx2])
        if z1 == 0 or z2 == 0 or z1 == z2:
            continue

        total += zbc_lookup.get((z1, z2), 0.0)

    return total


def compute_standard_boundary(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute standard (PU-level) boundary for selected PUs (zone > 0)."""
    if problem.boundary is None:
        return 0.0

    pu_ids = problem.planning_units["id"].values
    pu_index = {int(pid): i for i, pid in enumerate(pu_ids)}
    selected = zone_assignment > 0

    bnd = problem.boundary
    id1_col = bnd["id1"].values
    id2_col = bnd["id2"].values
    bval_col = bnd["boundary"].values.astype(np.float64)

    total = 0.0
    for k in range(len(id1_col)):
        id1 = int(id1_col[k])
        id2 = int(id2_col[k])
        bval = float(bval_col[k])

        if id1 == id2:
            idx = pu_index.get(id1)
            if idx is not None and selected[idx]:
                total += bval
        else:
            idx1 = pu_index.get(id1)
            idx2 = pu_index.get(id2)
            if idx1 is not None and idx2 is not None:
                if selected[idx1] != selected[idx2]:
                    total += bval
    return total


# ----------------------------------------------------------------------
# Overall (contribution-weighted) feature targets — MarZone reserve.hpp:158-171,
# Watts et al. 2009 eq. 6.
# ----------------------------------------------------------------------


def _feature_arrays(problem: ZonalProblem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(feature ids, targets × MISSLEVEL, spf) aligned with ``features`` order."""
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    ids = np.asarray(problem.features["id"].values, dtype=np.int64)
    targets = np.asarray(problem.features["target"].values, dtype=np.float64) * misslevel
    spf = (
        np.asarray(problem.features["spf"].values, dtype=np.float64)
        if "spf" in problem.features.columns
        else np.ones(len(ids), dtype=np.float64)
    )
    return ids, targets, spf


def compute_overall_achieved(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    *,
    amounts: np.ndarray | None = None,
) -> dict[int, float]:
    """A_f = Σ_i amount[i, f] × contribution[z_i, f] over PUs with z_i > 0.

    ``amounts`` (n_pu, n_feat) defaults to ``problem.build_pu_feature_matrix()``; the
    ``objectives`` package passes its (possibly probability-adjusted) effective amounts.
    """
    if amounts is None:
        amounts = problem.build_pu_feature_matrix()
    contrib = problem.contribution_matrix()
    zidx = problem.zone_index()
    rows = np.fromiter(
        (zidx.get(int(z), 0) for z in zone_assignment), dtype=np.int64, count=len(zone_assignment)
    )
    totals = (amounts * contrib[rows]).sum(axis=0)
    return {int(fid): float(totals[j]) for j, fid in enumerate(problem.features["id"].values)}


def compute_overall_shortfalls(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    *,
    achieved: dict[int, float] | None = None,
) -> dict[int, float]:
    """feature id -> max(0, target × MISSLEVEL − A_f) for features with target > 0.

    ``achieved`` is the dict ``compute_overall_achieved`` returns; ``None`` recomputes it.
    Callers that already hold it (``build_zone_solution``) pass it through so the PU ×
    feature matrix is built once per solution.
    """
    ids, targets, _ = _feature_arrays(problem)
    if achieved is None:
        achieved = compute_overall_achieved(problem, zone_assignment)
    return {
        int(fid): max(0.0, float(t) - achieved.get(int(fid), 0.0))
        for fid, t in zip(ids, targets, strict=True)
        if t > 0
    }


def check_overall_targets(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    *,
    achieved: dict[int, float] | None = None,
) -> dict[int, bool]:
    """feature id -> A_f >= target × MISSLEVEL (True for inert targets), every feature."""
    ids, targets, _ = _feature_arrays(problem)
    if achieved is None:
        achieved = compute_overall_achieved(problem, zone_assignment)
    return {
        int(fid): bool(t <= 0 or achieved.get(int(fid), 0.0) >= float(t))
        for fid, t in zip(ids, targets, strict=True)
    }


def compute_overall_penalty(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    *,
    achieved: dict[int, float] | None = None,
) -> float:
    """Σ_f spf_f × overall shortfall_f (approximation of MarZone's spf × penalty × proportion)."""
    ids, _, spf = _feature_arrays(problem)
    spf_of = {int(fid): float(s) for fid, s in zip(ids, spf, strict=True)}
    shortfalls = compute_overall_shortfalls(problem, zone_assignment, achieved=achieved)
    return float(sum(spf_of[fid] * sf for fid, sf in shortfalls.items()))


def compute_overall_shortfall(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    *,
    achieved: dict[int, float] | None = None,
) -> float:
    """Unweighted total overall shortfall."""
    shortfalls = compute_overall_shortfalls(problem, zone_assignment, achieved=achieved)
    return float(sum(shortfalls.values()))


def _compute_zone_achieved(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Achieved amount per (zone id, feature id): Σ_{i: z_i = k} amount[i, f] × w[k, f].

    ``w`` is ``problem.zone_target_weight_matrix()`` — all ones by default (MarZone
    ``reserve.hpp:164`` accumulates raw amounts for zone targets) or the contribution
    matrix under ``ZONETARGETCONTRIB = 1``.
    """
    weight = problem.zone_target_weight_matrix()
    zidx = problem.zone_index()
    fidx = problem.feature_index()
    amounts = problem.build_pu_feature_matrix()
    achieved: dict[tuple[int, int], float] = {}
    for zid, row in zidx.items():
        in_zone = zone_assignment == zid
        if not in_zone.any():
            continue
        totals = amounts[in_zone].sum(axis=0) * weight[row]
        for fid, col in fidx.items():
            if totals[col] != 0.0:
                achieved[(zid, fid)] = float(totals[col])
    return achieved


def check_zone_targets(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    *,
    zone_achieved: dict[tuple[int, int], float] | None = None,
) -> dict[tuple[int, int], bool]:
    """(zone id, feature id) -> Z_kf >= zone target × MISSLEVEL for listed zone targets.

    ``zone_achieved`` is the dict ``_compute_zone_achieved`` returns; ``None`` recomputes it.
    """
    if problem.zone_targets is None:
        return {}
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    if zone_achieved is None:
        zone_achieved = _compute_zone_achieved(problem, zone_assignment)
    zt = problem.zone_targets
    targets_met: dict[tuple[int, int], bool] = {}
    for zid, fid, target in zip(
        zt["zone"].values, zt["feature"].values, zt["target"].values, strict=True,
    ):
        key = (int(zid), int(fid))
        targets_met[key] = zone_achieved.get(key, 0.0) >= float(target) * misslevel
    return targets_met


def compute_zone_shortfalls(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    *,
    zone_achieved: dict[tuple[int, int], float] | None = None,
) -> dict[tuple[int, int], float]:
    """(zone id, feature id) -> max(0, zone target × MISSLEVEL − Z_kf) for listed zone targets."""
    if problem.zone_targets is None:
        return {}
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    if zone_achieved is None:
        zone_achieved = _compute_zone_achieved(problem, zone_assignment)
    zt = problem.zone_targets
    out: dict[tuple[int, int], float] = {}
    for zid, fid, target in zip(
        zt["zone"].values, zt["feature"].values, zt["target"].values, strict=True,
    ):
        key = (int(zid), int(fid))
        out[key] = max(0.0, float(target) * misslevel - zone_achieved.get(key, 0.0))
    return out


def compute_zone_penalty(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    *,
    zone_achieved: dict[tuple[int, int], float] | None = None,
) -> float:
    """Penalty for unmet zone targets: Σ spf_f × shortfall_kf."""
    shortfalls = compute_zone_shortfalls(problem, zone_assignment, zone_achieved=zone_achieved)
    if not shortfalls:
        return 0.0
    ids, _, spf = _feature_arrays(problem)
    spf_of = {int(fid): float(s) for fid, s in zip(ids, spf, strict=True)}
    return float(sum(spf_of.get(fid, 1.0) * sf for (_, fid), sf in shortfalls.items()))


def compute_zone_shortfall(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    *,
    zone_achieved: dict[tuple[int, int], float] | None = None,
) -> float:
    """Unweighted total shortfall across all zone targets."""
    shortfalls = compute_zone_shortfalls(problem, zone_assignment, zone_achieved=zone_achieved)
    return float(sum(shortfalls.values()))


def compute_zone_connectivity(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute connectivity penalty for zone assignments.

    For each edge (i, j) with connectivity value v, if both PUs are
    assigned to the same non-zero zone the value is a bonus (negative).
    Formula: -CONNECTIVITY_WEIGHT * Σ c_ij * [assignment[i] == assignment[j] && > 0]

    Returns 0.0 if no connectivity data or weight is zero.
    """
    if problem.connectivity is None:
        return 0.0

    conn_weight = float(problem.parameters.get("CONNECTIVITY_WEIGHT", 0.0))
    if conn_weight == 0.0:
        return 0.0

    pu_ids = problem.planning_units["id"].values
    pu_index = {int(pid): i for i, pid in enumerate(pu_ids)}

    conn = problem.connectivity
    id1_col = conn["id1"].values
    id2_col = conn["id2"].values
    val_col = conn["value"].values.astype(np.float64)

    total = 0.0
    for k in range(len(id1_col)):
        idx1 = pu_index.get(int(id1_col[k]))
        idx2 = pu_index.get(int(id2_col[k]))
        if idx1 is None or idx2 is None:
            continue
        z1 = int(zone_assignment[idx1])
        z2 = int(zone_assignment[idx2])
        if z1 > 0 and z1 == z2:
            total -= float(val_col[k])  # bonus for same-zone connection
    return conn_weight * total


def compute_zone_objective(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    blm: float,
) -> float:
    """Compute the full MarZone objective.
    Objective = zone_cost + BLM * standard_boundary + zone_boundary + penalty + connectivity
    """
    cost = compute_zone_cost(problem, zone_assignment)
    std_boundary = compute_standard_boundary(problem, zone_assignment)
    zone_boundary = compute_zone_boundary(problem, zone_assignment)
    penalty = compute_zone_penalty(problem, zone_assignment)
    connectivity = compute_zone_connectivity(problem, zone_assignment)
    return cost + blm * std_boundary + zone_boundary + penalty + connectivity
