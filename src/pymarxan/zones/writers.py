"""File writers for MarZone multi-zone projects."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pymarxan.zones.objective import (
    _compute_zone_achieved,
    check_overall_targets,
    check_zone_targets,
    compute_overall_achieved,
)

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


_SUMMARY_COLUMNS = [
    "tier", "zone", "feature", "target", "mean_achieved", "times_met", "total_runs",
]


def write_zone_summary(
    problem: ZonalProblem,
    solutions: list[Solution],
    path: str | Path,
) -> None:
    """Target-achievement summary across runs, computed by the objective module.

    Two row groups share one CSV (columns ``tier, zone, feature, target, mean_achieved,
    times_met, total_runs``):

    - ``tier == "overall"`` — one row per feature, ``zone`` 0: ``target`` is
      ``features.target``; ``mean_achieved`` the contribution-weighted amount
      (``compute_overall_achieved``); ``times_met`` counts runs where
      ``check_overall_targets`` is True.
    - ``tier == "zone"`` — one row per listed zone target: raw amounts by default
      (contribution-weighted under ``ZONETARGETCONTRIB = 1``), met per ``check_zone_targets``.

    Solutions without ``zone_assignment`` are read as "selected PUs in the first zone".
    """
    zone_ids = sorted(int(z) for z in problem.zones["id"].tolist())
    first_zone = zone_ids[0] if zone_ids else 1
    assignments = [
        np.asarray(sol.zone_assignment, dtype=int)
        if sol.zone_assignment is not None
        else np.where(np.asarray(sol.selected, dtype=bool), first_zone, 0)
        for sol in solutions
    ]
    total_runs = len(assignments)

    feat_ids = [int(f) for f in problem.features["id"].tolist()]
    feat_targets = dict(
        zip(feat_ids, problem.features["target"].astype(float).tolist(), strict=True)
    )
    overall_achieved = [compute_overall_achieved(problem, a) for a in assignments]
    overall_met = [check_overall_targets(problem, a) for a in assignments]

    rows: list[dict[str, object]] = []
    for fid in feat_ids:
        rows.append({
            "tier": "overall",
            "zone": 0,
            "feature": fid,
            "target": feat_targets[fid],
            "mean_achieved": (
                sum(a.get(fid, 0.0) for a in overall_achieved) / total_runs
                if total_runs else 0.0
            ),
            "times_met": sum(int(m.get(fid, False)) for m in overall_met),
            "total_runs": total_runs,
        })

    if problem.zone_targets is not None:
        zone_achieved = [_compute_zone_achieved(problem, a) for a in assignments]
        zone_met = [check_zone_targets(problem, a) for a in assignments]
        zt = problem.zone_targets
        for zid, fid, target in zip(
            zt["zone"].values, zt["feature"].values, zt["target"].values, strict=True,
        ):
            key = (int(zid), int(fid))
            rows.append({
                "tier": "zone",
                "zone": key[0],
                "feature": key[1],
                "target": float(target),
                "mean_achieved": (
                    sum(a.get(key, 0.0) for a in zone_achieved) / total_runs
                    if total_runs else 0.0
                ),
                "times_met": sum(int(m.get(key, False)) for m in zone_met),
                "total_runs": total_runs,
            })

    pd.DataFrame(rows, columns=_SUMMARY_COLUMNS).to_csv(path, index=False)
