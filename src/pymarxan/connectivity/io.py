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
    """Convert an edge list DataFrame to NxN matrix."""
    n = len(pu_ids)
    id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}
    matrix = np.zeros((n, n))

    id1_col = edgelist["id1"].values
    id2_col = edgelist["id2"].values
    val_col = edgelist["value"].values.astype(np.float64)

    # Map IDs to indices using vectorized lookup
    idx1 = np.array([id_to_idx.get(int(v), -1) for v in id1_col], dtype=np.intp)
    idx2 = np.array([id_to_idx.get(int(v), -1) for v in id2_col], dtype=np.intp)
    valid = (idx1 >= 0) & (idx2 >= 0)

    matrix[idx1[valid], idx2[valid]] = val_col[valid]
    if symmetric:
        matrix[idx2[valid], idx1[valid]] = val_col[valid]

    return matrix
