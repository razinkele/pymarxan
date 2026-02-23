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
    n = len(pu_ids)
    id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}
    matrix = np.zeros((n, n))
    for _, row in df.iterrows():
        i = id_to_idx.get(int(row["id1"]))
        j = id_to_idx.get(int(row["id2"]))
        if i is not None and j is not None:
            matrix[i, j] = float(row["value"])
            if symmetric:
                matrix[j, i] = float(row["value"])
    return matrix


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
    """Convert an edge list DataFrame to NxN matrix."""
    n = len(pu_ids)
    id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}
    matrix = np.zeros((n, n))
    for _, row in edgelist.iterrows():
        i = id_to_idx.get(int(row["id1"]))
        j = id_to_idx.get(int(row["id2"]))
        if i is not None and j is not None:
            matrix[i, j] = float(row["value"])
            if symmetric:
                matrix[j, i] = float(row["value"])
    return matrix
