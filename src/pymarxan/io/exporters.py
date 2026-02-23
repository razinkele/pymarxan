"""Export solver results to CSV and other formats."""
from __future__ import annotations

from pathlib import Path

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

    rows = []
    for _, frow in problem.features.iterrows():
        fid = int(frow["id"])
        fname = frow.get("name", f"Feature {fid}")
        target = float(frow.get("target", 0.0))
        mask = problem.pu_vs_features["species"] == fid
        achieved = 0.0
        for _, arow in problem.pu_vs_features[mask].iterrows():
            pid = int(arow["pu"])
            if pid in id_to_idx and solution.selected[id_to_idx[pid]]:
                achieved += float(arow["amount"])
        rows.append({
            "feature_id": fid,
            "feature_name": fname,
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
