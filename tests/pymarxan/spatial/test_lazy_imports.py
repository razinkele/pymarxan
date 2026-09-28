"""Issue #4: pymarxan.spatial must import without rasterio/requests (the `spatial` extra)."""
from __future__ import annotations

import importlib
import sys

import pytest

LAZY_MODULES = [
    "pymarxan.spatial",
    "pymarxan.spatial.raster",
    "pymarxan.spatial.cost_surface",
    "pymarxan.spatial.feature_intersection",
    "pymarxan.spatial.gadm",
    "pymarxan.spatial.wdpa",
    "pymarxan.spatial.grid",
    "pymarxan.spatial.boundary",
    "pymarxan.spatial.importers",
]

ALL_NAMES = {
    "apply_cost_from_raster", "apply_cost_from_vector", "apply_wdpa_status",
    "combine_cost_layers", "compute_adjacency", "compute_boundary", "fetch_gadm",
    "fetch_wdpa", "from_arrays", "from_rasters", "generate_planning_grid",
    "import_features_from_vector", "import_planning_units", "intersect_raster_features",
    "intersect_vector_features", "list_countries",
}


@pytest.fixture
def without_extras(monkeypatch):
    """Simulate a base install: rasterio and requests unimportable, spatial not yet imported."""
    for mod in LAZY_MODULES:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setitem(sys.modules, "rasterio", None)
    monkeypatch.setitem(sys.modules, "requests", None)
    yield
    for mod in LAZY_MODULES:
        monkeypatch.delitem(sys.modules, mod, raising=False)


def test_grid_and_boundary_import_without_rasterio(without_extras):
    grid = importlib.import_module("pymarxan.spatial.grid")
    spatial = importlib.import_module("pymarxan.spatial")
    assert spatial.generate_planning_grid is grid.generate_planning_grid
    assert callable(spatial.compute_boundary)
    assert callable(spatial.import_planning_units)


def test_raster_helpers_raise_import_error_naming_the_extra(without_extras):
    spatial = importlib.import_module("pymarxan.spatial")
    with pytest.raises(ImportError, match="spatial"):
        _ = spatial.from_rasters
    with pytest.raises(ImportError, match="spatial"):
        _ = spatial.fetch_gadm


def test_public_surface_is_unchanged():
    import pymarxan.spatial as spatial

    assert set(spatial.__all__) == ALL_NAMES
    assert ALL_NAMES <= set(dir(spatial))
    # with the extra installed every name resolves to its source object
    from pymarxan.spatial.raster import from_rasters

    assert spatial.from_rasters is from_rasters


def test_unknown_attribute_still_raises_attribute_error():
    import pymarxan.spatial as spatial

    with pytest.raises(AttributeError):
        _ = spatial.no_such_helper
