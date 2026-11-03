"""File writers for MarZone multi-zone projects."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pymarxan.solvers.base import require_binary
from pymarxan.zones.objective import (
    check_overall_targets,
    check_zone_targets,
    compute_overall_achieved,
    compute_zone_achieved,
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
        require_binary(solution, "write_zone_solution")
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
    - ``tier == "zone"`` — one row per known zone target
      (``ZonalProblem.zone_target_lookup(include_inert=True)``: unknown ids omitted,
      duplicates last-wins, inert targets <= 0 reported as met): raw amounts by default
      (contribution-weighted under ``ZONETARGETCONTRIB = 1``), met per ``check_zone_targets``.

    Solutions without ``zone_assignment`` are read as "selected PUs in the first zone".
    """
    zone_ids = sorted(int(z) for z in problem.zones["id"].tolist())
    first_zone = zone_ids[0] if zone_ids else 1
    for sol in solutions:
        if sol.zone_assignment is None:
            require_binary(sol, "write_zone_summary")
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

    lookup = problem.zone_target_lookup(include_inert=True)
    if lookup:
        zone_achieved = [compute_zone_achieved(problem, a) for a in assignments]
        zone_met = [check_zone_targets(problem, a) for a in assignments]
        for key, target in lookup.items():
            rows.append({
                "tier": "zone",
                "zone": key[0],
                "feature": key[1],
                "target": target,
                "mean_achieved": (
                    sum(a.get(key, 0.0) for a in zone_achieved) / total_runs
                    if total_runs else 0.0
                ),
                "times_met": sum(int(m.get(key, False)) for m in zone_met),
                "total_runs": total_runs,
            })

    pd.DataFrame(rows, columns=_SUMMARY_COLUMNS).to_csv(path, index=False)


def _write_pu_zone_pairs(df: pd.DataFrame, path: str | Path) -> None:
    df[["pu", "zone"]].rename(columns={"pu": "puid", "zone": "zoneid"}).to_csv(
        path, index=False,
    )


def write_pu_lock(df: pd.DataFrame, path: str | Path) -> None:
    """Write ``pu_locks`` as MarZone ``pulock.dat`` (header ``puid,zoneid``)."""
    _write_pu_zone_pairs(df, path)


def write_pu_zone(df: pd.DataFrame, path: str | Path) -> None:
    """Write ``pu_zones`` as MarZone ``puzone.dat`` (header ``puid,zoneid``)."""
    _write_pu_zone_pairs(df, path)


# (parameter key, default file name, ZonalProblem attribute, writer)
_ZONE_FILES = (
    ("ZONESNAME", "zones.dat", "zones", write_zones),
    ("ZONECOSTNAME", "zonecost.dat", "zone_costs", write_zone_costs),
    ("ZONECONTRIBNAME", "zonecontrib.dat", "zone_contributions", write_zone_contributions),
    ("ZONETARGETNAME", "zonetarget.dat", "zone_targets", write_zone_targets),
    ("ZONEBOUNDCOSTNAME", "zoneboundcost.dat", "zone_boundary_costs",
     write_zone_boundary_costs),
    ("PULOCKNAME", "pulock.dat", "pu_locks", write_pu_lock),
    ("PUZONENAME", "puzone.dat", "pu_zones", write_pu_zone),
)


def save_zone_project(problem: ZonalProblem, project_dir: str | Path) -> None:
    """Save a ``ZonalProblem`` as a MarZone project directory ``load_zone_project`` reads
    back (the inverse of the loader).

    Writes the Marxan files via :func:`pymarxan.io.writers.save_project` plus
    ``zones.dat``, ``zonecost.dat`` and, when present, ``zonecontrib.dat``,
    ``zonetarget.dat``, ``zoneboundcost.dat``, ``pulock.dat`` and ``puzone.dat``, naming
    each in ``input.dat`` (the loader reads pulock/puzone only when named). An absent
    optional table removes a stale file of the same name from the input directory, and its
    ``*NAME`` key from the written parameters, so the reload cannot pick up old data.
    pymarxan's "unassigned" (no action) state has no MarZone C++ equivalent, so a saved
    project is guaranteed to reproduce in pymarxan, not in MarZone.
    """
    from pymarxan.io.writers import save_project

    project_dir = Path(project_dir)
    params = dict(problem.parameters)
    params.setdefault("INPUTDIR", "input")
    for key, default, attr, _ in _ZONE_FILES:
        if getattr(problem, attr) is not None:
            params.setdefault(key, default)
        elif key in params:
            del params[key]
    save_project(problem.copy_with(parameters=params), project_dir)
    input_dir = project_dir / str(params["INPUTDIR"])
    for key, default, attr, writer in _ZONE_FILES:
        table = getattr(problem, attr)
        if table is not None:
            writer(table, input_dir / str(params[key]))
        elif (input_dir / default).exists():
            (input_dir / default).unlink()
