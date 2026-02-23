"""Spatial data processing for conservation planning."""

from pymarxan.spatial.gadm import fetch_gadm, list_countries
from pymarxan.spatial.grid import compute_adjacency, generate_planning_grid
from pymarxan.spatial.wdpa import apply_wdpa_status, fetch_wdpa

__all__ = [
    "apply_wdpa_status",
    "compute_adjacency",
    "fetch_gadm",
    "fetch_wdpa",
    "generate_planning_grid",
    "list_countries",
]
