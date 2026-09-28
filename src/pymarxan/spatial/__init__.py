"""Spatial data processing for conservation planning.

The grid, boundary and vector-import helpers need only geopandas/shapely and are imported
eagerly. The raster and web-fetch helpers need ``rasterio`` / ``requests`` from the
``spatial`` extra and are resolved lazily on first attribute access (PEP 562), so a base
install can still ``import pymarxan.spatial.grid``.
"""
from __future__ import annotations

import importlib
from typing import Any

from pymarxan.spatial.boundary import compute_boundary
from pymarxan.spatial.grid import compute_adjacency, generate_planning_grid
from pymarxan.spatial.importers import import_features_from_vector, import_planning_units

# name -> submodule; everything here needs the ``spatial`` extra at import time
_LAZY: dict[str, str] = {
    "apply_cost_from_raster": "cost_surface",
    "apply_cost_from_vector": "cost_surface",
    "combine_cost_layers": "cost_surface",
    "intersect_raster_features": "feature_intersection",
    "intersect_vector_features": "feature_intersection",
    "fetch_gadm": "gadm",
    "list_countries": "gadm",
    "from_arrays": "raster",
    "from_rasters": "raster",
    "apply_wdpa_status": "wdpa",
    "fetch_wdpa": "wdpa",
}

__all__ = [
    "apply_cost_from_raster",
    "apply_cost_from_vector",
    "apply_wdpa_status",
    "combine_cost_layers",
    "compute_adjacency",
    "compute_boundary",
    "fetch_gadm",
    "fetch_wdpa",
    "from_arrays",
    "from_rasters",
    "generate_planning_grid",
    "import_features_from_vector",
    "import_planning_units",
    "intersect_raster_features",
    "intersect_vector_features",
    "list_countries",
]


def __getattr__(name: str) -> Any:
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f"{__name__}.{submodule}")
    except ImportError as exc:
        raise ImportError(
            f"pymarxan.spatial.{name} needs the 'spatial' extra "
            f"(pip install 'pymarxan[spatial]'): {exc}"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value  # cache so the next access is a plain lookup
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
