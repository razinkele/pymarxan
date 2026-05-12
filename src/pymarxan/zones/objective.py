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


def _compute_zone_achieved(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Precompute achieved amounts per (zone, feature) pair.

    Returns dict mapping (zone_id, feature_id) -> achieved amount.
    """
    pu_ids = problem.planning_units["id"].values
    pu_index = {int(pid): i for i, pid in enumerate(pu_ids)}

    # Pre-build contribution lookup
    contrib_lookup: dict[tuple[int, int], float] = {}
    if problem.zone_contributions is not None:
        zc = problem.zone_contributions
        for k in range(len(zc)):
            fid = int(zc["feature"].values[k])
            zid = int(zc["zone"].values[k])
            contrib_lookup[(fid, zid)] = float(zc["contribution"].values[k])

    puvspr = problem.pu_vs_features
    pu_col = puvspr["pu"].values
    sp_col = puvspr["species"].values
    amt_col = puvspr["amount"].values

    achieved: dict[tuple[int, int], float] = {}
    for k in range(len(pu_col)):
        pid = int(pu_col[k])
        idx = pu_index.get(pid)
        if idx is None:
            continue
        zid = int(zone_assignment[idx])
        if zid == 0:
            continue
        fid = int(sp_col[k])
        contribution = contrib_lookup.get((fid, zid), 1.0)
        key = (zid, fid)
        achieved[key] = achieved.get(key, 0.0) + float(amt_col[k]) * contribution

    return achieved


def check_zone_targets(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> dict[tuple[int, int], bool]:
    """Check which zone-specific targets are met. Returns dict of (zone_id, feature_id) -> bool."""
    if problem.zone_targets is None:
        return {}

    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    achieved = _compute_zone_achieved(problem, zone_assignment)

    targets_met: dict[tuple[int, int], bool] = {}
    zt = problem.zone_targets
    zone_col = zt["zone"].values
    feat_col = zt["feature"].values
    target_col = zt["target"].values

    for k in range(len(zone_col)):
        zid = int(zone_col[k])
        fid = int(feat_col[k])
        target = float(target_col[k])
        targets_met[(zid, fid)] = achieved.get((zid, fid), 0.0) >= target * misslevel

    return targets_met


def compute_zone_penalty(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute penalty for unmet zone targets (SPF * shortfall)."""
    if problem.zone_targets is None:
        return 0.0

    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    achieved = _compute_zone_achieved(problem, zone_assignment)

    # Pre-build SPF lookup
    spf_lookup: dict[int, float] = {}
    feat_ids = problem.features["id"].values
    feat_spf = (
        problem.features["spf"].values
        if "spf" in problem.features.columns
        else np.ones(len(feat_ids))
    )
    for i in range(len(feat_ids)):
        spf_lookup[int(feat_ids[i])] = float(feat_spf[i])

    zt = problem.zone_targets
    zone_col = zt["zone"].values
    feat_col = zt["feature"].values
    target_col = zt["target"].values

    total = 0.0
    for k in range(len(zone_col)):
        zid = int(zone_col[k])
        fid = int(feat_col[k])
        target = float(target_col[k])
        shortfall = max(0.0, target * misslevel - achieved.get((zid, fid), 0.0))
        total += spf_lookup.get(fid, 1.0) * shortfall

    return total


def compute_zone_shortfall(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute raw (unweighted) total shortfall across all zone targets."""
    if problem.zone_targets is None:
        return 0.0

    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    achieved = _compute_zone_achieved(problem, zone_assignment)

    zt = problem.zone_targets
    zone_col = zt["zone"].values
    feat_col = zt["feature"].values
    target_col = zt["target"].values

    total = 0.0
    for k in range(len(zone_col)):
        zid = int(zone_col[k])
        fid = int(feat_col[k])
        target = float(target_col[k])
        total += max(0.0, target * misslevel - achieved.get((zid, fid), 0.0))

    return total


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
