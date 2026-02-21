"""Shared utility functions for conservation planning solvers."""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


def compute_boundary(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> float:
    """Compute total boundary length for a given selection.

    Diagonal entries (id1==id2) represent external boundary, added when PU is selected.
    Off-diagonal entries represent shared boundary, added when exactly one PU is selected.
    """
    if problem.boundary is None:
        return 0.0

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


def check_targets(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> dict[int, bool]:
    """Check which feature targets are met by the selection."""
    targets_met: dict[int, bool] = {}
    for _, feat_row in problem.features.iterrows():
        fid = int(feat_row["id"])
        target = float(feat_row["target"])
        feat_data = problem.pu_vs_features[
            problem.pu_vs_features["species"] == fid
        ]
        total = 0.0
        for _, r in feat_data.iterrows():
            pu_id = int(r["pu"])
            idx = pu_index.get(pu_id)
            if idx is not None and selected[idx]:
                total += float(r["amount"])
        targets_met[fid] = total >= target
    return targets_met


def compute_feature_shortfalls(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> dict[int, float]:
    """Compute the shortfall for each feature (target - achieved, min 0)."""
    shortfalls: dict[int, float] = {}
    for _, feat_row in problem.features.iterrows():
        fid = int(feat_row["id"])
        target = float(feat_row["target"])
        feat_data = problem.pu_vs_features[
            problem.pu_vs_features["species"] == fid
        ]
        achieved = 0.0
        for _, r in feat_data.iterrows():
            pu_id = int(r["pu"])
            idx = pu_index.get(pu_id)
            if idx is not None and selected[idx]:
                achieved += float(r["amount"])
        shortfalls[fid] = max(0.0, target - achieved)
    return shortfalls


def compute_penalty(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
) -> float:
    """Compute the total feature penalty (SPF * shortfall for each feature)."""
    shortfalls = compute_feature_shortfalls(problem, selected, pu_index)
    total = 0.0
    for _, feat_row in problem.features.iterrows():
        fid = int(feat_row["id"])
        spf = float(feat_row.get("spf", 1.0))
        total += spf * shortfalls.get(fid, 0.0)
    return total


def compute_objective(
    problem: ConservationProblem,
    selected: np.ndarray,
    pu_index: dict[int, int],
    blm: float,
) -> float:
    """Compute the full Marxan objective: cost + BLM*boundary + penalty."""
    costs = np.asarray(problem.planning_units["cost"].values)
    total_cost = float(np.sum(costs[selected]))
    total_boundary = compute_boundary(problem, selected, pu_index)
    penalty = compute_penalty(problem, selected, pu_index)
    return total_cost + blm * total_boundary + penalty


def build_solution(
    problem: ConservationProblem,
    selected: np.ndarray,
    blm: float,
    metadata: dict | None = None,
) -> Solution:
    """Build a complete Solution from a selection array."""
    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}

    costs = np.asarray(problem.planning_units["cost"].values)
    total_cost = float(np.sum(costs[selected]))
    total_boundary = compute_boundary(problem, selected, pu_index)
    targets_met = check_targets(problem, selected, pu_index)
    objective = total_cost + blm * total_boundary

    return Solution(
        selected=selected.copy(),
        cost=total_cost,
        boundary=total_boundary,
        objective=objective,
        targets_met=targets_met,
        metadata=metadata or {},
    )
