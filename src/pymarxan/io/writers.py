"""Marxan file writers and project saver."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from pymarxan.models.problem import ConservationProblem

if TYPE_CHECKING:
    from pymarxan.solvers.base import Solution


def write_pu(df: pd.DataFrame, path: str | Path) -> None:
    """Write a planning units DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Planning units data with columns like ``id``, ``cost``, ``status``.
    path : str | Path
        Output file path.
    """
    df.to_csv(path, index=False)


def write_spec(df: pd.DataFrame, path: str | Path) -> None:
    """Write a species/features DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Features data with columns like ``id``, ``name``, ``target``, ``spf``.
    path : str | Path
        Output file path.
    """
    df.to_csv(path, index=False)


def write_puvspr(df: pd.DataFrame, path: str | Path) -> None:
    """Write a planning-unit-vs-species DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        PU vs species data with columns ``species``, ``pu``, ``amount``.
    path : str | Path
        Output file path.
    """
    df.to_csv(path, index=False)


def write_bound(df: pd.DataFrame, path: str | Path) -> None:
    """Write a boundary DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Boundary data with columns ``id1``, ``id2``, ``boundary``.
    path : str | Path
        Output file path.
    """
    df.to_csv(path, index=False)


def write_input_dat(params: dict[str, Any], path: str | Path) -> None:
    """Write Marxan parameters as a KEY VALUE file.

    Each parameter is written as a single line with key and value separated
    by a space.  Numeric values are formatted so that floats include a
    decimal point and ints do not.

    Parameters
    ----------
    params : dict[str, Any]
        Mapping of parameter names to their values.
    path : str | Path
        Output file path.
    """
    path = Path(path)
    with open(path, "w") as f:
        for key, value in params.items():
            if isinstance(value, float):
                # Ensure float values always have a decimal point
                formatted = f"{value:g}"
                if "." not in formatted and "e" not in formatted.lower():
                    formatted += ".0"
                f.write(f"{key} {formatted}\n")
            else:
                f.write(f"{key} {value}\n")


def save_project(problem: ConservationProblem, project_dir: str | Path) -> None:
    """Save a ConservationProblem to a Marxan project directory.

    Creates ``input.dat`` and an ``input/`` subdirectory containing
    ``pu.dat``, ``spec.dat``, ``puvspr.dat``, and optionally ``bound.dat``.

    Parameters
    ----------
    problem : ConservationProblem
        The problem to save.
    project_dir : str | Path
        Directory to save into (created if it doesn't exist).
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Default file names
    params = dict(problem.parameters)
    params.setdefault("INPUTDIR", "input")
    params.setdefault("PUNAME", "pu.dat")
    params.setdefault("SPECNAME", "spec.dat")
    params.setdefault("PUVSPRNAME", "puvspr.dat")
    params.setdefault("BOUNDNAME", "bound.dat")

    input_dir = project_dir / params["INPUTDIR"]
    input_dir.mkdir(parents=True, exist_ok=True)

    write_pu(problem.planning_units, input_dir / params["PUNAME"])
    write_spec(problem.features, input_dir / params["SPECNAME"])
    write_puvspr(problem.pu_vs_features, input_dir / params["PUVSPRNAME"])

    if problem.boundary is not None:
        write_bound(problem.boundary, input_dir / params["BOUNDNAME"])

    write_input_dat(params, project_dir / "input.dat")


def write_mvbest(
    problem: ConservationProblem, solution: Solution, path: str | Path
) -> None:
    """Write missing value info for the best solution.

    Produces a CSV with one row per feature showing how much of each
    feature's target is met by the selected planning units.

    Columns: Feature_ID, Feature_Name, Target, Amount_Held, Target_Met,
    Shortfall.

    Parameters
    ----------
    problem : ConservationProblem
        The conservation planning problem.
    solution : Solution
        The best solution to report on.
    path : str | Path
        Output file path.
    """
    from pymarxan.solvers.utils import compute_feature_shortfalls

    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    shortfalls = compute_feature_shortfalls(problem, solution.selected, pu_index)

    rows = []
    for _, feat_row in problem.features.iterrows():
        fid = int(feat_row["id"])
        target = float(feat_row["target"])
        name = feat_row["name"]

        # Compute amount held by summing amounts of selected PUs
        feat_data = problem.pu_vs_features[
            problem.pu_vs_features["species"] == fid
        ]
        amount_held = 0.0
        for _, r in feat_data.iterrows():
            pu_id = int(r["pu"])
            idx = pu_index.get(pu_id)
            if idx is not None and solution.selected[idx]:
                amount_held += float(r["amount"])

        shortfall = shortfalls.get(fid, 0.0)
        target_met = amount_held >= target * misslevel

        rows.append({
            "Feature_ID": fid,
            "Feature_Name": name,
            "Target": target,
            "Amount_Held": amount_held,
            "Target_Met": target_met,
            "Shortfall": shortfall,
        })

    pd.DataFrame(rows).to_csv(path, index=False)


def write_ssoln(
    problem: ConservationProblem,
    solutions: list[Solution],
    path: str | Path,
) -> None:
    """Write summed solution showing how many times each PU was selected.

    Produces a CSV with one row per planning unit, counting how many
    solutions included it.

    Columns: Planning_Unit, Number.

    Parameters
    ----------
    problem : ConservationProblem
        The conservation planning problem.
    solutions : list[Solution]
        All solutions to summarize.
    path : str | Path
        Output file path.
    """
    pu_ids = problem.planning_units["id"].tolist()
    counts = [0] * len(pu_ids)

    for sol in solutions:
        for i, selected in enumerate(sol.selected):
            if selected:
                counts[i] += 1

    rows = [
        {"Planning_Unit": pu_id, "Number": count}
        for pu_id, count in zip(pu_ids, counts)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_sum(solutions: list[Solution], path: str | Path) -> None:
    """Write per-run summary.

    Produces a CSV with one row per solution run containing cost,
    boundary, penalty, and other summary statistics.

    Columns: Run, Score, Cost, Planning_Units, Boundary, Penalty, Shortfall.

    Parameters
    ----------
    solutions : list[Solution]
        All solutions to summarize.
    path : str | Path
        Output file path.
    """
    rows = []
    for i, sol in enumerate(solutions, start=1):
        penalty = sol.penalty
        shortfall = penalty  # total shortfall approximated from penalty
        rows.append({
            "Run": i,
            "Score": sol.objective,
            "Cost": sol.cost,
            "Planning_Units": sol.n_selected,
            "Boundary": sol.boundary,
            "Penalty": penalty,
            "Shortfall": shortfall,
        })

    pd.DataFrame(rows).to_csv(path, index=False)
