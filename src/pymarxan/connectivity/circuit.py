"""Circuit-theory (current-flow) connectivity (Tier B).

The least-cost-path measure in :mod:`pymarxan.connectivity.resistance`
keeps only the single cheapest route between two cells. Circuit theory
instead models the landscape as a resistor network and integrates *all*
routes: the connectivity between two cells is their **effective
resistance** (a.k.a. resistance distance), the standard Circuitscape
measure. Lower effective resistance = better connected.

The graph Laplacian linear system is solved with ``scipy.sparse``: one
sparse LU factorisation of the grounded Laplacian is reused across every
planning-unit pair.

See McRae et al. (2008), *Ecology*, https://doi.org/10.1890/07-1861.1.
"""
from __future__ import annotations

import numpy as np


def current_flow_to_matrix(
    raster: np.ndarray,
    coords: list[tuple[int, int]],
) -> np.ndarray:
    """Pairwise effective-resistance connectivity from a resistance raster.

    Models the raster as a resistor network (4-neighbour grid; each edge's
    resistance is the mean of the two cells it joins) and returns the
    effective resistance between every pair of planning units.

    Args:
        raster: Shape ``(rows, cols)``; ``raster[r, c]`` is the resistance
            of cell ``(r, c)`` (higher = poorer habitat). Values must be
            positive and finite.
        coords: ``(row, col)`` index per planning unit, within the raster.

    Returns:
        ``(n_pus, n_pus)`` symmetric matrix of effective resistances
        (resistance distance); zero diagonal. Lower = better connected.

    Raises:
        ValueError: If the raster is not 2-D, has a non-positive or
            non-finite value, or a coordinate lies outside the raster.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    raster = np.asarray(raster, dtype=float)
    if raster.ndim != 2:
        raise ValueError(f"raster must be 2-D; got shape {raster.shape}.")
    if not np.all(np.isfinite(raster)) or not np.all(raster > 0):
        raise ValueError("raster resistance values must be positive and finite.")

    n_rows, n_cols = raster.shape
    n_pus = len(coords)
    for i, (r, c) in enumerate(coords):
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            raise ValueError(
                f"PU {i} coord ({r}, {c}) lies outside raster shape "
                f"({n_rows}, {n_cols})."
            )

    matrix = np.zeros((n_pus, n_pus))
    n_cells = n_rows * n_cols
    if n_pus <= 1 or n_cells <= 1:
        return matrix

    def node(r: int, c: int) -> int:
        return int(r * n_cols + c)

    # Assemble the sparse Laplacian L = D - G. Each 4-neighbour edge gets a
    # conductance g = 1 / mean(resistance of its two cells); it contributes
    # -g to both off-diagonal entries and +g to each endpoint's diagonal.
    rows_idx: list[int] = []
    cols_idx: list[int] = []
    data: list[float] = []
    diag = np.zeros(n_cells)
    for r in range(n_rows):
        for c in range(n_cols):
            a = node(r, c)
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < n_rows and cc < n_cols:
                    b = node(rr, cc)
                    g = 1.0 / (0.5 * (raster[r, c] + raster[rr, cc]))
                    rows_idx += [a, b]
                    cols_idx += [b, a]
                    data += [-g, -g]
                    diag[a] += g
                    diag[b] += g
    rows_idx += list(range(n_cells))
    cols_idx += list(range(n_cells))
    data += diag.tolist()
    laplacian = csc_matrix((data, (rows_idx, cols_idx)), shape=(n_cells, n_cells))

    # Ground node 0 (drop its row/column) so the reduced Laplacian is
    # symmetric positive-definite and invertible. Factorise once.
    solver = splu(laplacian[1:, 1:].tocsc())

    cell_of = [node(r, c) for (r, c) in coords]
    for i in range(n_pus):
        ci = cell_of[i]
        for j in range(i + 1, n_pus):
            cj = cell_of[j]
            if ci == cj:
                continue
            # Inject +1 A at ci and -1 A at cj; with node 0 grounded
            # (v[0] = 0), the effective resistance is the voltage drop.
            injection = np.zeros(n_cells)
            injection[ci] += 1.0
            injection[cj] -= 1.0
            voltage = np.zeros(n_cells)
            voltage[1:] = solver.solve(injection[1:])
            r_eff = float(voltage[ci] - voltage[cj])
            matrix[i, j] = r_eff
            matrix[j, i] = r_eff

    return matrix
