"""Marxan file writers and project saver."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pymarxan.models.problem import ConservationProblem


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
