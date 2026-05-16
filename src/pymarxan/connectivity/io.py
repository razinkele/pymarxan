"""I/O for connectivity matrices (edge lists and full matrices)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_connectivity_edgelist(
    path: str | Path,
    pu_ids: list[int],
    *,
    symmetric: bool = True,
) -> np.ndarray:
    """Read an edge list CSV and convert to NxN matrix. Expected columns: id1, id2, value."""
    df = pd.read_csv(path)
    return connectivity_to_matrix(df, pu_ids, symmetric=symmetric)


def read_connectivity_matrix(path: str | Path) -> np.ndarray:
    """Read a full connectivity matrix from CSV (first column/row = IDs)."""
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(float)


def connectivity_to_matrix(
    edgelist: pd.DataFrame,
    pu_ids: list[int],
    *,
    symmetric: bool = True,
) -> np.ndarray:
    """Convert an edge list DataFrame to NxN matrix.

    Duplicate ``(id1, id2)`` rows are summed (Marxan-style accumulation).
    Vectorized assignment would silently drop all but the last duplicate.
    """
    n = len(pu_ids)
    id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}
    matrix = np.zeros((n, n))

    # Sum duplicates first so users who provide both (A,B,v1) and (A,B,v2)
    # or repeated rows for the same pair don't lose data.
    deduped = (
        edgelist[["id1", "id2", "value"]]
        .groupby(["id1", "id2"], as_index=False, sort=False)["value"]
        .sum()
    )

    id1_col = deduped["id1"].values
    id2_col = deduped["id2"].values
    val_col = deduped["value"].values.astype(np.float64)

    idx1 = np.array([id_to_idx.get(int(v), -1) for v in id1_col], dtype=np.intp)
    idx2 = np.array([id_to_idx.get(int(v), -1) for v in id2_col], dtype=np.intp)
    valid = (idx1 >= 0) & (idx2 >= 0)

    matrix[idx1[valid], idx2[valid]] = val_col[valid]
    if symmetric:
        matrix[idx2[valid], idx1[valid]] = val_col[valid]

    return matrix


def connectivity_to_boundary(
    edgelist: pd.DataFrame,
    *,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Convert a connectivity edge list to a Marxan ``bound.dat``-compatible
    DataFrame so users can feed connectivity into the BLM penalty.

    Marxan's ``bound.dat`` columns are ``id1`` / ``id2`` / ``boundary``.
    This is a one-to-one mapping from a connectivity edge list, with an
    optional ``scale`` multiplier.

    Parameters
    ----------
    edgelist
        DataFrame with ``id1``, ``id2``, ``value`` columns.
    scale
        Multiplier applied to ``value`` before writing as ``boundary``.
        Negative ``scale`` is a common idiom: high connectivity → low
        boundary penalty (so the solver prefers keeping connected PUs
        together).

    Returns
    -------
    pd.DataFrame
        Columns ``id1``, ``id2``, ``boundary``. Zero-value rows dropped
        (bound.dat conventions don't include null boundaries).
    """
    df = edgelist[["id1", "id2", "value"]].copy()
    df["boundary"] = df["value"].astype(float) * float(scale)
    df = df.drop(columns=["value"])
    return df[df["boundary"] != 0.0].reset_index(drop=True)
