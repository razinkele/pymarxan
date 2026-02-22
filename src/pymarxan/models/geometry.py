"""Synthetic geometry generator for planning unit grids.

Generates rectangular bounding boxes arranged in a grid layout,
suitable for ipyleaflet.Rectangle visualization when spatial
data is not available in the Marxan dataset.
"""
from __future__ import annotations

import math


def generate_grid(
    n_pu: int,
    origin: tuple[float, float] = (0.0, 0.0),
    cell_size: float = 0.01,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Generate grid bounding boxes for n planning units.

    Parameters
    ----------
    n_pu : int
        Number of planning units.
    origin : tuple[float, float]
        (latitude, longitude) of the south-west corner of the grid.
    cell_size : float
        Size of each cell in degrees (~1km at equator for 0.01).

    Returns
    -------
    list[tuple[tuple[float, float], tuple[float, float]]]
        List of ((south, west), (north, east)) bounding boxes.
    """
    if n_pu <= 0:
        return []

    cols = math.ceil(math.sqrt(n_pu))
    boxes: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for i in range(n_pu):
        col = i % cols
        row = i // cols
        south = origin[0] + row * cell_size
        west = origin[1] + col * cell_size
        north = south + cell_size
        east = west + cell_size
        boxes.append(((south, west), (north, east)))

    return boxes
