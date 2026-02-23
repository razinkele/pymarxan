"""Objective function components for multi-zone conservation planning."""
from __future__ import annotations

import numpy as np

from pymarxan.zones.model import ZonalProblem


def compute_zone_cost(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute total cost given zone assignments. Zone 0 = unassigned (no cost)."""
    pu_ids = problem.planning_units["id"].tolist()
    total = 0.0
    for i, pid in enumerate(pu_ids):
        zid = int(zone_assignment[i])
        if zid == 0:
            continue
        total += problem.get_zone_cost(pid, zid)
    return total


def compute_zone_boundary(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute zone boundary cost between adjacent PUs in different zones."""
    if problem.boundary is None or problem.zone_boundary_costs is None:
        return 0.0

    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}

    zbc_lookup: dict[tuple[int, int], float] = {}
    for _, row in problem.zone_boundary_costs.iterrows():
        z1 = int(row["zone1"])
        z2 = int(row["zone2"])
        cost = float(row["cost"])
        zbc_lookup[(z1, z2)] = cost

    total = 0.0
    for _, row in problem.boundary.iterrows():
        id1 = int(row["id1"])
        id2 = int(row["id2"])
        if id1 == id2:
            continue

        idx1 = pu_index.get(id1)
        idx2 = pu_index.get(id2)
        if idx1 is None or idx2 is None:
            continue

        z1 = int(zone_assignment[idx1])
        z2 = int(zone_assignment[idx2])
        if z1 == 0 or z2 == 0:
            continue
        if z1 == z2:
            continue

        cost = zbc_lookup.get((z1, z2), 0.0)
        total += cost

    return total


def compute_standard_boundary(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute standard (PU-level) boundary for selected PUs (zone > 0)."""
    if problem.boundary is None:
        return 0.0

    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}
    selected = zone_assignment > 0

    total = 0.0
    for _, row in problem.boundary.iterrows():
        id1 = int(row["id1"])
        id2 = int(row["id2"])
        bval = float(row["boundary"])

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


def check_zone_targets(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> dict[tuple[int, int], bool]:
    """Check which zone-specific targets are met. Returns dict of (zone_id, feature_id) -> bool."""
    if problem.zone_targets is None:
        return {}

    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}

    targets_met: dict[tuple[int, int], bool] = {}
    for _, trow in problem.zone_targets.iterrows():
        zid = int(trow["zone"])
        fid = int(trow["feature"])
        target = float(trow["target"])

        contribution = problem.get_contribution(fid, zid)
        feat_data = problem.pu_vs_features[
            problem.pu_vs_features["species"] == fid
        ]

        achieved = 0.0
        for _, r in feat_data.iterrows():
            pid = int(r["pu"])
            idx = pu_index.get(pid)
            if idx is not None and int(zone_assignment[idx]) == zid:
                achieved += float(r["amount"]) * contribution

        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
        targets_met[(zid, fid)] = achieved >= target * misslevel

    return targets_met


def compute_zone_penalty(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute penalty for unmet zone targets (SPF * shortfall)."""
    if problem.zone_targets is None:
        return 0.0

    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}

    spf_lookup: dict[int, float] = {}
    for _, frow in problem.features.iterrows():
        spf_lookup[int(frow["id"])] = float(frow.get("spf", 1.0))

    total = 0.0
    for _, trow in problem.zone_targets.iterrows():
        zid = int(trow["zone"])
        fid = int(trow["feature"])
        target = float(trow["target"])

        contribution = problem.get_contribution(fid, zid)
        feat_data = problem.pu_vs_features[
            problem.pu_vs_features["species"] == fid
        ]

        achieved = 0.0
        for _, r in feat_data.iterrows():
            pid = int(r["pu"])
            idx = pu_index.get(pid)
            if idx is not None and int(zone_assignment[idx]) == zid:
                achieved += float(r["amount"]) * contribution

        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
        shortfall = max(0.0, target * misslevel - achieved)
        total += spf_lookup.get(fid, 1.0) * shortfall

    return total


def compute_zone_objective(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    blm: float,
) -> float:
    """Compute the full MarZone objective.
    Objective = zone_cost + BLM * standard_boundary + zone_boundary + penalty
    """
    cost = compute_zone_cost(problem, zone_assignment)
    std_boundary = compute_standard_boundary(problem, zone_assignment)
    zone_boundary = compute_zone_boundary(problem, zone_assignment)
    penalty = compute_zone_penalty(problem, zone_assignment)
    return cost + blm * std_boundary + zone_boundary + penalty
