"""File writers for MarZone multi-zone projects."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pymarxan.solvers.base import Solution
    from pymarxan.zones.model import ZonalProblem


def write_zones(df: pd.DataFrame, path: str | Path) -> None:
    """Write a zones DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Zones data with columns ``id``, ``name``.
    path : str | Path
        Output file path.
    """
    df.to_csv(path, index=False)


def write_zone_costs(df: pd.DataFrame, path: str | Path) -> None:
    """Write a zone costs DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Zone costs data with columns ``pu``, ``zone``, ``cost``.
    path : str | Path
        Output file path.
    """
    df.to_csv(path, index=False)


def write_zone_contributions(
    df: pd.DataFrame, path: str | Path
) -> None:
    """Write a zone contributions DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Zone contributions with columns ``feature``, ``zone``,
        ``contribution``.
    path : str | Path
        Output file path.
    """
    df.to_csv(path, index=False)


def write_zone_targets(df: pd.DataFrame, path: str | Path) -> None:
    """Write a zone targets DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Zone targets with columns ``zone``, ``feature``, ``target``.
    path : str | Path
        Output file path.
    """
    df.to_csv(path, index=False)


def write_zone_boundary_costs(
    df: pd.DataFrame, path: str | Path
) -> None:
    """Write a zone boundary costs DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Zone boundary costs with columns ``zone1``, ``zone2``, ``cost``.
    path : str | Path
        Output file path.
    """
    df.to_csv(path, index=False)


def write_zone_solution(
    solution: Solution, path: str | Path
) -> None:
    """Write zone solution file with columns: planning_unit, zone.

    Zone 0 means the planning unit is not selected / unassigned.

    Parameters
    ----------
    solution : Solution
        A solution with ``zone_assignment`` populated.
    path : str | Path
        Output file path.
    """
    if solution.zone_assignment is None:
        zones = np.where(solution.selected, 1, 0)
    else:
        zones = solution.zone_assignment

    rows = [
        {"planning_unit": i + 1, "zone": int(z)}
        for i, z in enumerate(zones)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_zone_summary(
    problem: ZonalProblem,
    solutions: list[Solution],
    path: str | Path,
) -> None:
    """Write per-zone-feature target achievement summary across runs.

    Produces a CSV with one row per (zone, feature) combination showing
    the number of times the target was met across all solution runs.

    Columns: zone, feature, target, times_met, total_runs.

    Parameters
    ----------
    problem : ZonalProblem
        The zonal conservation planning problem.
    solutions : list[Solution]
        All solutions to summarize.
    path : str | Path
        Output file path.
    """
    zone_ids = sorted(problem.zones["id"].tolist())
    feat_ids = problem.features["id"].tolist()
    feat_targets = dict(
        zip(
            problem.features["id"].tolist(),
            problem.features["target"].astype(float).tolist(),
        )
    )

    pu_ids = problem.planning_units["id"].tolist()
    total_runs = len(solutions)

    # Build per-feature amounts keyed by (pu, species)
    puvspr = problem.pu_vs_features
    amount_lookup: dict[tuple[int, int], float] = {}
    for _, row in puvspr.iterrows():
        amount_lookup[(int(row["pu"]), int(row["species"]))] = float(
            row["amount"]
        )

    rows: list[dict] = []
    for zid in zone_ids:
        contribution_for_zone = {
            fid: problem.get_contribution(fid, zid) for fid in feat_ids
        }
        for fid in feat_ids:
            target = feat_targets.get(fid, 0.0)
            contrib = contribution_for_zone[fid]
            times_met = 0

            for sol in solutions:
                achieved = 0.0
                assignment = (
                    sol.zone_assignment
                    if sol.zone_assignment is not None
                    else np.where(sol.selected, 1, 0)
                )
                for idx, pid in enumerate(pu_ids):
                    if int(assignment[idx]) == zid:
                        amt = amount_lookup.get((pid, fid), 0.0)
                        achieved += amt * contrib
                if target <= 0 or achieved >= target:
                    times_met += 1

            rows.append({
                "zone": zid,
                "feature": fid,
                "target": target,
                "times_met": times_met,
                "total_runs": total_runs,
            })

    pd.DataFrame(rows).to_csv(path, index=False)
