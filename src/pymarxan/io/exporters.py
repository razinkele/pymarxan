"""Export solver results to CSV and other formats."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pymarxan.analysis.selection_freq import SelectionFrequency
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


def export_solution_csv(
    problem: ConservationProblem,
    solution: Solution,
    path: Path | str,
) -> None:
    """Export a solution to CSV with planning unit details."""
    pu_ids = problem.planning_units["id"].tolist()
    costs = problem.planning_units["cost"].tolist()
    rows = []
    for i, (pid, cost) in enumerate(zip(pu_ids, costs)):
        rows.append({
            "planning_unit": pid,
            "cost": cost,
            "selected": int(solution.selected[i]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def export_summary_csv(
    problem: ConservationProblem,
    solution: Solution,
    path: Path | str,
) -> None:
    """Export target achievement summary to CSV."""
    pu_ids = problem.planning_units["id"].tolist()
    id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}

    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))

    # Precompute achieved per feature in one pass
    sel_set = {pid for pid, i in id_to_idx.items() if solution.selected[i]}
    sel_mask = problem.pu_vs_features["pu"].isin(sel_set)
    achieved_map = (
        problem.pu_vs_features.loc[sel_mask]
        .groupby("species")["amount"]
        .sum()
        .to_dict()
    )

    feat_ids = problem.features["id"].values
    feat_names = (
        problem.features["name"].values
        if "name" in problem.features.columns
        else [f"Feature {fid}" for fid in feat_ids]
    )
    feat_targets = (
        problem.features["target"].values.astype(float)
        if "target" in problem.features.columns
        else np.zeros(len(feat_ids))
    )

    rows = []
    for k in range(len(feat_ids)):
        fid = int(feat_ids[k])
        achieved = achieved_map.get(fid, 0.0)
        target = float(feat_targets[k])
        rows.append({
            "feature_id": fid,
            "feature_name": feat_names[k],
            "target": target,
            "achieved": achieved,
            "met": achieved >= target * misslevel,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def export_selection_frequency_csv(
    problem: ConservationProblem,
    freq: SelectionFrequency,
    path: Path | str,
) -> None:
    """Export selection frequency results to CSV."""
    pu_ids = problem.planning_units["id"].tolist()
    rows = []
    for i, pid in enumerate(pu_ids):
        rows.append({
            "planning_unit": pid,
            "frequency": float(freq.frequencies[i]),
            "count": int(freq.counts[i]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
