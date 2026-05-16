"""Habitat-resistance least-cost-path connectivity (Phase 24).

Given a 2D habitat-resistance raster and a list of PU (x, y) coordinates,
compute the pairwise least-cost-path connectivity matrix. High raster
values are "expensive" cells (poor habitat); low values are cheap
(good corridor).

Uses networkx Dijkstra on a 4-neighbour grid graph. We deliberately
avoid the scikit-image dependency that ``route_through_array`` would
bring; networkx is already a pymarxan optional dep.

Cost: ``O(n_pus · n_cells · log(n_cells))`` for the all-pairs LCP
computation. For very large rasters consider sampling coarser PU
positions.
"""
from __future__ import annotations

import numpy as np


def habitat_resistance_to_matrix(
    raster: np.ndarray,
    coords: list[tuple[int, int]],
) -> np.ndarray:
    """Compute pairwise least-cost-path connectivity from a habitat
    resistance raster.

    Parameters
    ----------
    raster
        Shape ``(rows, cols)``. ``raster[r, c]`` is the cost of crossing
        cell ``(r, c)``. Higher = poorer habitat (more resistance).
    coords
        List of ``(row, col)`` integer indices into the raster — one per
        planning unit. Must lie within raster bounds.

    Returns
    -------
    np.ndarray
        ``(n_pus, n_pus)`` symmetric matrix of least-cost-path costs.
        Diagonal is zero.
    """
    import networkx as nx

    if raster.ndim != 2:
        raise ValueError(f"raster must be 2-D; got shape {raster.shape}.")

    n_rows, n_cols = raster.shape
    n_pus = len(coords)

    # Build the 4-neighbour grid graph. Edge weight between adjacent cells
    # is the mean of the two cell values (standard least-cost-path
    # convention — the cost of traversing the edge approximates the cost
    # of being in either endpoint).
    G = nx.Graph()
    for r in range(n_rows):
        for c in range(n_cols):
            G.add_node((r, c))
            if c + 1 < n_cols:
                w = 0.5 * (float(raster[r, c]) + float(raster[r, c + 1]))
                G.add_edge((r, c), (r, c + 1), weight=w)
            if r + 1 < n_rows:
                w = 0.5 * (float(raster[r, c]) + float(raster[r + 1, c]))
                G.add_edge((r, c), (r + 1, c), weight=w)

    # Validate PU coordinates fall inside the raster.
    for i, (r, c) in enumerate(coords):
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            raise ValueError(
                f"PU {i} coord ({r}, {c}) lies outside raster shape "
                f"({n_rows}, {n_cols})."
            )

    matrix = np.zeros((n_pus, n_pus))
    for i in range(n_pus):
        src = coords[i]
        # Single-source Dijkstra to every other node.
        lengths = nx.single_source_dijkstra_path_length(G, src, weight="weight")
        for j in range(i + 1, n_pus):
            dst = coords[j]
            cost = float(lengths.get(dst, np.inf))
            matrix[i, j] = cost
            matrix[j, i] = cost

    return matrix
