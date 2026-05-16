"""Marxan file readers and project loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pymarxan.models.problem import ConservationProblem


def _read_dat(path: str | Path) -> pd.DataFrame:
    """Read a .dat file, auto-detecting the delimiter (tab or comma).

    Parameters
    ----------
    path : str | Path
        Path to the .dat file.

    Returns
    -------
    pd.DataFrame
        Parsed data.
    """
    path = Path(path)
    with open(path) as f:
        first_line = f.readline()

    if "\t" in first_line:
        sep = "\t"
    else:
        sep = ","

    return pd.read_csv(path, sep=sep)


def read_pu(path: str | Path) -> pd.DataFrame:
    """Read a planning units (pu.dat) file.

    Casts ``id`` to int, ``cost`` to float, and ``status`` to int if present.

    Parameters
    ----------
    path : str | Path
        Path to pu.dat.

    Returns
    -------
    pd.DataFrame
        Planning units data.
    """
    df = _read_dat(path)
    df["id"] = df["id"].astype(int)
    df["cost"] = df["cost"].astype(float)
    if "status" in df.columns:
        df["status"] = df["status"].astype(int)
    else:
        df["status"] = 0
    return df


def read_spec(path: str | Path) -> pd.DataFrame:
    """Read a species/features (spec.dat) file.

    Casts ``id`` to int, ``target``/``prop``/``spf`` to float where present.

    Parameters
    ----------
    path : str | Path
        Path to spec.dat.

    Returns
    -------
    pd.DataFrame
        Features data.
    """
    df = _read_dat(path)
    df["id"] = df["id"].astype(int)
    for col in ("target", "prop", "spf", "ptarget"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "target" not in df.columns:
        df["target"] = 0.0
    if "spf" not in df.columns:
        df["spf"] = 1.0
    if "name" not in df.columns:
        df["name"] = [f"Feature_{fid}" for fid in df["id"]]
    if "ptarget" not in df.columns:
        # -1 is Marxan's "no probability target" sentinel.
        df["ptarget"] = -1.0
    return df


def read_puvspr(path: str | Path) -> pd.DataFrame:
    """Read a planning-unit-vs-species (puvspr.dat) file.

    Casts ``species`` and ``pu`` to int, ``amount`` to float.

    Parameters
    ----------
    path : str | Path
        Path to puvspr.dat.

    Returns
    -------
    pd.DataFrame
        Planning unit vs species data.
    """
    df = _read_dat(path)
    df["species"] = df["species"].astype(int)
    df["pu"] = df["pu"].astype(int)
    df["amount"] = df["amount"].astype(float)
    return df


def read_bound(path: str | Path) -> pd.DataFrame:
    """Read a boundary (bound.dat) file.

    Casts ``id1`` and ``id2`` to int, ``boundary`` to float.

    Parameters
    ----------
    path : str | Path
        Path to bound.dat.

    Returns
    -------
    pd.DataFrame
        Boundary data.
    """
    df = _read_dat(path)
    df["id1"] = df["id1"].astype(int)
    df["id2"] = df["id2"].astype(int)
    df["boundary"] = df["boundary"].astype(float)
    return df


def read_input_dat(path: str | Path) -> dict[str, Any]:
    """Read a Marxan input.dat parameter file.

    The file format is whitespace-separated KEY VALUE pairs, one per line.
    Lines beginning with ``#`` are treated as comments and ignored.
    Numeric values are auto-converted: int if no decimal point, float if
    decimal point is present.

    Parameters
    ----------
    path : str | Path
        Path to input.dat.

    Returns
    -------
    dict[str, Any]
        Mapping of parameter names to their values.
    """
    path = Path(path)
    params: dict[str, Any] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
            # Auto-convert numeric values
            try:
                if "." in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value
    return params


def read_probability(path: str | Path) -> pd.DataFrame:
    """Read a probability (prob.dat) file.

    Casts ``pu`` to int, ``probability`` to float.

    Parameters
    ----------
    path : str | Path
        Path to prob.dat.

    Returns
    -------
    pd.DataFrame
        Probability data with columns ``pu``, ``probability``.
    """
    df = _read_dat(path)
    df["pu"] = df["pu"].astype(int)
    df["probability"] = df["probability"].astype(float)
    return df


def _resolve_prop_targets(
    features: pd.DataFrame, pu_vs_features: pd.DataFrame
) -> pd.DataFrame:
    """Resolve proportional targets: effective_target = max(target, prop * total_amount)."""
    if "prop" not in features.columns:
        return features

    features = features.copy()
    if "target" not in features.columns:
        features["target"] = 0.0

    # 1. Compute total amount available for each species
    totals = pu_vs_features.groupby("species")["amount"].sum()

    # 2. Map totals to the features DataFrame based on 'id'
    #    (fillna(0.0) handles species present in features but absent in pu_vs_features)
    feature_totals = features["id"].map(totals).fillna(0.0)

    # 3. Vectorized update: target = max(target, prop * total)
    prop_targets = features["prop"].fillna(0.0) * feature_totals
    features["target"] = features["target"].clip(lower=prop_targets)

    return features


def load_project(project_dir: str | Path) -> ConservationProblem:
    """Load a full Marxan project from a directory.

    Reads ``input.dat`` from the directory, then loads all referenced data
    files (pu.dat, spec.dat, puvspr.dat, bound.dat) and returns a
    ``ConservationProblem``.

    Parameters
    ----------
    project_dir : str | Path
        Path to the project directory containing ``input.dat``.

    Returns
    -------
    ConservationProblem
        The loaded conservation problem.
    """
    project_dir = Path(project_dir)
    params = read_input_dat(project_dir / "input.dat")

    input_dir = project_dir / params.get("INPUTDIR", "input")

    pu_name = params.get("PUNAME", "pu.dat")
    spec_name = params.get("SPECNAME", "spec.dat")
    puvspr_name = params.get("PUVSPRNAME", "puvspr.dat")
    bound_name = params.get("BOUNDNAME", "bound.dat")

    planning_units = read_pu(input_dir / pu_name)
    features = read_spec(input_dir / spec_name)
    pu_vs_features = read_puvspr(input_dir / puvspr_name)

    boundary = None
    bound_path = input_dir / bound_name
    if bound_path.exists():
        boundary = read_bound(bound_path)

    # Probability data (optional)
    prob_name = params.get("PROBNAME", "prob.dat")
    probability = None
    prob_path = input_dir / prob_name
    if prob_path.exists():
        probability = read_probability(prob_path)

    features = _resolve_prop_targets(features, pu_vs_features)

    return ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
        boundary=boundary,
        parameters=params,
        probability=probability,
    )


def read_mvbest(path: str | Path) -> pd.DataFrame:
    """Read a Marxan output missing values best (out_mvbest.csv) file.

    Casts ``Feature_ID`` to int, ``Target``/``Amount_Held``/``Shortfall``
    to float, and ``Target_Met`` to bool.

    Parameters
    ----------
    path : str | Path
        Path to out_mvbest.csv.

    Returns
    -------
    pd.DataFrame
        Missing values best data.
    """
    df = pd.read_csv(path)
    df["Feature_ID"] = df["Feature_ID"].astype(int)
    for col in ("Target", "Amount_Held", "Shortfall"):
        df[col] = df[col].astype(float)
    df["Target_Met"] = df["Target_Met"].astype(bool)
    return df


def read_ssoln(path: str | Path) -> pd.DataFrame:
    """Read a Marxan output summed solution (out_ssoln.csv) file.

    Casts ``Planning_Unit`` and ``Number`` to int.

    Parameters
    ----------
    path : str | Path
        Path to out_ssoln.csv.

    Returns
    -------
    pd.DataFrame
        Summed solution data.
    """
    df = pd.read_csv(path)
    df["Planning_Unit"] = df["Planning_Unit"].astype(int)
    df["Number"] = df["Number"].astype(int)
    return df


def read_sum(path: str | Path) -> pd.DataFrame:
    """Read a Marxan output summary (out_sum.csv) file.

    Casts ``Run``/``Planning_Units`` to int and
    ``Score``/``Cost``/``Boundary``/``Penalty``/``Shortfall`` to float.

    Parameters
    ----------
    path : str | Path
        Path to out_sum.csv.

    Returns
    -------
    pd.DataFrame
        Summary data.
    """
    df = pd.read_csv(path)
    df["Run"] = df["Run"].astype(int)
    df["Planning_Units"] = df["Planning_Units"].astype(int)
    for col in ("Score", "Cost", "Boundary", "Penalty", "Shortfall"):
        df[col] = df[col].astype(float)
    return df
