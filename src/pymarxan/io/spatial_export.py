"""Spatial export of solutions to GeoPackage and Shapefile."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np

from pymarxan.solvers.base import Solution


def export_solution_spatial(
    planning_units: gpd.GeoDataFrame,
    solution: Solution,
    path: Path | str,
    driver: str = "GPKG",
) -> None:
    """Join solution selected status to PU geometries and export.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id`` and ``geometry`` columns.
    solution : Solution
        Solver result with ``selected`` boolean array (or float proportions, written as a
        float column rounded to 10 significant digits).
    path : Path or str
        Output file path.
    driver : str
        OGR driver name: ``"GPKG"`` (default) or ``"ESRI Shapefile"``.
    """
    gdf = planning_units.copy()
    if solution.is_fractional:
        gdf["selected"] = [float(format(float(v), ".10g")) for v in solution.selected]
    else:
        gdf["selected"] = solution.selected.astype(int)
    gdf.to_file(str(path), driver=driver)


def export_frequency_spatial(
    planning_units: gpd.GeoDataFrame,
    solutions: list[Solution],
    path: Path | str,
    driver: str = "GPKG",
) -> None:
    """Export selection frequency across multiple solution runs.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id`` and ``geometry`` columns.
    solutions : list[Solution]
        List of solver results.
    path : Path or str
        Output file path.
    driver : str
        OGR driver name: ``"GPKG"`` (default) or ``"ESRI Shapefile"``.
    """
    n_solutions = len(solutions)
    if n_solutions == 0:
        gdf = planning_units.copy()
        gdf["frequency"] = 0.0
        gdf["count"] = 0
        gdf.to_file(str(path), driver=driver)
        return

    # Proportions: count = Σ x (float), frequency = mean of x.
    fractional = any(sol.is_fractional for sol in solutions)
    counts = np.zeros(len(planning_units), dtype=np.float64 if fractional else int)
    for sol in solutions:
        counts += sol.selected.astype(np.float64 if fractional else int)

    gdf = planning_units.copy()
    gdf["count"] = counts
    gdf["frequency"] = counts / n_solutions
    gdf.to_file(str(path), driver=driver)
