# Phase 14: Planning Unit Grid Generation + GeoDataFrame Upgrade

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the `pymarxan.spatial` subpackage with square and hexagonal grid generation, compute adjacency from geometry, add a `has_geometry()` helper, and upgrade map modules to render real polygons when geometry is available.

**Architecture:** New `pymarxan.spatial.grid` module generates GeoDataFrames with Shapely polygons. A `has_geometry()` helper in `pymarxan.models.problem` lets map modules detect real geometry. Map modules fall back to synthetic `generate_grid()` when geometry is absent. A new Shiny module `grid_builder` provides the UI.

**Tech Stack:** geopandas, shapely (both already in core deps), ipyleaflet (shiny dep)

---

### Task 1: Create `pymarxan.spatial` package with `generate_planning_grid` (square)

**Files:**
- Create: `src/pymarxan/spatial/__init__.py`
- Create: `src/pymarxan/spatial/grid.py`
- Create: `tests/pymarxan/spatial/__init__.py`
- Create: `tests/pymarxan/spatial/test_grid.py`

**Step 1: Write the failing tests**

```python
# tests/pymarxan/spatial/__init__.py
# (empty)

# tests/pymarxan/spatial/test_grid.py
"""Tests for planning unit grid generation."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from pymarxan.spatial.grid import generate_planning_grid


class TestSquareGrid:
    def test_basic_square_grid(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
            grid_type="square",
        )
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 4  # 2x2 grid
        assert set(gdf.columns) >= {"id", "cost", "status", "geometry"}
        assert gdf["cost"].tolist() == [1.0] * 4
        assert gdf["status"].tolist() == [0] * 4

    def test_square_grid_ids_are_sequential(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.5, 1.0),
            cell_size=0.5,
        )
        assert gdf["id"].tolist() == list(range(1, len(gdf) + 1))

    def test_square_grid_crs(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
            crs="EPSG:4326",
        )
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_square_grid_geometries_are_polygons(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
        )
        for geom in gdf.geometry:
            assert isinstance(geom, Polygon)
            assert geom.is_valid

    def test_square_grid_no_overlaps(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
        )
        # Pairwise intersection area should be ~0
        for i in range(len(gdf)):
            for j in range(i + 1, len(gdf)):
                overlap = gdf.geometry.iloc[i].intersection(gdf.geometry.iloc[j]).area
                assert overlap < 1e-10

    def test_empty_bounds_returns_empty(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 0.0, 0.0),
            cell_size=0.5,
        )
        assert len(gdf) == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/spatial/test_grid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan.spatial'`

**Step 3: Implement square grid generation**

```python
# src/pymarxan/spatial/__init__.py
"""Spatial data processing for conservation planning."""

from pymarxan.spatial.grid import compute_adjacency, generate_planning_grid

__all__ = ["generate_planning_grid", "compute_adjacency"]
```

```python
# src/pymarxan/spatial/grid.py
"""Planning unit grid generation."""
from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import BaseGeometry, Polygon, box


def generate_planning_grid(
    bounds: tuple[float, float, float, float],
    cell_size: float,
    grid_type: str = "square",
    crs: str = "EPSG:4326",
    clip_to: BaseGeometry | None = None,
) -> gpd.GeoDataFrame:
    """Generate a planning unit grid as a GeoDataFrame.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) bounding box.
    cell_size : float
        Width/height of each cell in CRS units.
    grid_type : str
        ``"square"`` or ``"hexagonal"``.
    crs : str
        Coordinate reference system (default EPSG:4326).
    clip_to : BaseGeometry or None
        Optional polygon to clip the grid to.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: id (int), cost (float), status (int), geometry (Polygon).
    """
    if grid_type == "square":
        cells = _generate_square_cells(bounds, cell_size)
    elif grid_type == "hexagonal":
        cells = _generate_hex_cells(bounds, cell_size)
    else:
        raise ValueError(f"Unknown grid_type: {grid_type!r}. Use 'square' or 'hexagonal'.")

    if not cells:
        return gpd.GeoDataFrame(
            {"id": [], "cost": [], "status": [], "geometry": []},
            crs=crs,
        )

    if clip_to is not None:
        cells = [c for c in cells if c.centroid.within(clip_to)]

    n = len(cells)
    return gpd.GeoDataFrame(
        {
            "id": list(range(1, n + 1)),
            "cost": [1.0] * n,
            "status": [0] * n,
        },
        geometry=cells,
        crs=crs,
    )


def _generate_square_cells(
    bounds: tuple[float, float, float, float],
    cell_size: float,
) -> list[Polygon]:
    minx, miny, maxx, maxy = bounds
    cells: list[Polygon] = []
    y = miny
    while y < maxy - 1e-10:
        x = minx
        while x < maxx - 1e-10:
            cells.append(box(x, y, x + cell_size, y + cell_size))
            x += cell_size
        y += cell_size
    return cells


def _generate_hex_cells(
    bounds: tuple[float, float, float, float],
    cell_size: float,
) -> list[Polygon]:
    minx, miny, maxx, maxy = bounds
    # Flat-top hexagon: width = cell_size, height = cell_size * sqrt(3)/2
    w = cell_size
    h = cell_size * math.sqrt(3) / 2
    cells: list[Polygon] = []
    row = 0
    y = miny
    while y < maxy - 1e-10:
        x_offset = (w / 2) if row % 2 == 1 else 0.0
        x = minx + x_offset
        while x < maxx - 1e-10:
            cells.append(_flat_top_hex(x, y, cell_size))
            x += w
        y += h
        row += 1
    return cells


def _flat_top_hex(cx: float, cy: float, size: float) -> Polygon:
    """Create a flat-top hexagon centered at (cx, cy)."""
    half = size / 2
    h = size * math.sqrt(3) / 4
    return Polygon([
        (cx - half, cy),
        (cx - half / 2, cy + h),
        (cx + half / 2, cy + h),
        (cx + half, cy),
        (cx + half / 2, cy - h),
        (cx - half / 2, cy - h),
    ])


def compute_adjacency(planning_units: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute boundary DataFrame from shared edges between adjacent PUs.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id`` and ``geometry`` columns.

    Returns
    -------
    pd.DataFrame
        Columns: id1, id2, boundary (shared edge length).
    """
    rows: list[dict] = []
    geoms = planning_units.geometry.values
    ids = planning_units["id"].values

    for i in range(len(planning_units)):
        for j in range(i + 1, len(planning_units)):
            if geoms[i].touches(geoms[j]) or (
                geoms[i].intersection(geoms[j]).length > 1e-10
            ):
                shared = geoms[i].intersection(geoms[j]).length
                if shared > 1e-10:
                    rows.append({
                        "id1": int(ids[i]),
                        "id2": int(ids[j]),
                        "boundary": shared,
                    })

    return pd.DataFrame(rows, columns=["id1", "id2", "boundary"])
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan/spatial/test_grid.py -v`
Expected: 6 PASS

**Step 5: Commit**

```bash
git add src/pymarxan/spatial/ tests/pymarxan/spatial/
git commit -m "feat(spatial): add planning unit grid generation — square grids"
```

---

### Task 2: Add hexagonal grid generation and clipping

**Files:**
- Modify: `tests/pymarxan/spatial/test_grid.py`
- Modify: `src/pymarxan/spatial/grid.py` (already implemented above, tests verify)

**Step 1: Write the failing tests**

Add to `tests/pymarxan/spatial/test_grid.py`:

```python
from shapely.geometry import Polygon, box


class TestHexGrid:
    def test_basic_hex_grid(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 2.0, 2.0),
            cell_size=0.5,
            grid_type="hexagonal",
        )
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0
        for geom in gdf.geometry:
            assert isinstance(geom, Polygon)
            # Hexagons have 6 vertices (+ closing vertex = 7 coords)
            assert len(geom.exterior.coords) == 7

    def test_hex_grid_no_large_overlaps(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.3,
            grid_type="hexagonal",
        )
        # Hex tiles may have tiny floating-point overlaps, but not large ones
        for i in range(len(gdf)):
            for j in range(i + 1, len(gdf)):
                overlap = gdf.geometry.iloc[i].intersection(gdf.geometry.iloc[j]).area
                assert overlap < 0.01 * gdf.geometry.iloc[i].area


class TestClipping:
    def test_clip_to_polygon(self):
        clip_poly = box(0.0, 0.0, 0.7, 0.7)
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
            clip_to=clip_poly,
        )
        # Only cells whose centroid is within clip_poly are kept
        for geom in gdf.geometry:
            assert clip_poly.contains(geom.centroid)

    def test_clip_removes_cells_outside(self):
        clip_poly = box(0.0, 0.0, 0.3, 0.3)
        gdf_full = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
        )
        gdf_clipped = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
            clip_to=clip_poly,
        )
        assert len(gdf_clipped) < len(gdf_full)


class TestInvalidInput:
    def test_invalid_grid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown grid_type"):
            generate_planning_grid(
                bounds=(0.0, 0.0, 1.0, 1.0),
                cell_size=0.5,
                grid_type="triangle",
            )
```

**Step 2: Run tests to verify they pass** (implementation was done in Task 1)

Run: `pytest tests/pymarxan/spatial/test_grid.py -v`
Expected: 10 PASS

**Step 3: Commit**

```bash
git add tests/pymarxan/spatial/test_grid.py
git commit -m "test(spatial): add hexagonal grid and clipping tests"
```

---

### Task 3: Add `compute_adjacency` tests

**Files:**
- Modify: `tests/pymarxan/spatial/test_grid.py`

**Step 1: Write the failing tests**

Add to `tests/pymarxan/spatial/test_grid.py`:

```python
from pymarxan.spatial.grid import compute_adjacency


class TestAdjacency:
    def test_square_grid_adjacency(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
        )
        adj = compute_adjacency(gdf)
        assert set(adj.columns) == {"id1", "id2", "boundary"}
        # 2x2 grid: 4 shared edges (right, down for each applicable cell)
        assert len(adj) == 4
        assert all(adj["boundary"] > 0)

    def test_adjacency_ids_match_grid(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
        )
        adj = compute_adjacency(gdf)
        all_ids = set(gdf["id"])
        adj_ids = set(adj["id1"]) | set(adj["id2"])
        assert adj_ids <= all_ids

    def test_single_cell_no_adjacency(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 0.5, 0.5),
            cell_size=0.5,
        )
        assert len(gdf) == 1
        adj = compute_adjacency(gdf)
        assert len(adj) == 0
```

**Step 2: Run tests**

Run: `pytest tests/pymarxan/spatial/test_grid.py::TestAdjacency -v`
Expected: 3 PASS

**Step 3: Commit**

```bash
git add tests/pymarxan/spatial/test_grid.py
git commit -m "test(spatial): add adjacency computation tests"
```

---

### Task 4: Add `has_geometry()` helper to problem model

**Files:**
- Modify: `src/pymarxan/models/problem.py`
- Create: `tests/pymarxan/spatial/test_geometry_detection.py`

**Step 1: Write the failing tests**

```python
# tests/pymarxan/spatial/test_geometry_detection.py
"""Tests for has_geometry detection."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from pymarxan.models.problem import ConservationProblem, has_geometry


def _make_features():
    return pd.DataFrame({
        "id": [1], "name": ["f1"], "target": [10.0], "spf": [1.0],
    })


def _make_puvspr():
    return pd.DataFrame({"species": [1], "pu": [1], "amount": [20.0]})


class TestHasGeometry:
    def test_plain_dataframe_no_geometry(self):
        p = ConservationProblem(
            planning_units=pd.DataFrame({
                "id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0],
            }),
            features=_make_features(),
            pu_vs_features=_make_puvspr(),
        )
        assert has_geometry(p) is False

    def test_geodataframe_with_geometry(self):
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        p = ConservationProblem(
            planning_units=gdf,
            features=_make_features(),
            pu_vs_features=_make_puvspr(),
        )
        assert has_geometry(p) is True

    def test_geodataframe_empty_geometry(self):
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(
            {"id": [1], "cost": [1.0], "status": [0]},
            geometry=[Point()],  # empty geometry
            crs="EPSG:4326",
        )
        p = ConservationProblem(
            planning_units=gdf,
            features=_make_features(),
            pu_vs_features=_make_puvspr(),
        )
        assert has_geometry(p) is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/spatial/test_geometry_detection.py -v`
Expected: FAIL — `ImportError: cannot import name 'has_geometry'`

**Step 3: Implement `has_geometry`**

Add to `src/pymarxan/models/problem.py` at the top (after imports):

```python
import geopandas as gpd
```

Add after the `ConservationProblem` class:

```python
def has_geometry(problem: ConservationProblem) -> bool:
    """Check if planning_units has real spatial geometry."""
    return (
        isinstance(problem.planning_units, gpd.GeoDataFrame)
        and "geometry" in problem.planning_units.columns
        and not problem.planning_units.geometry.is_empty.all()
    )
```

Also update `src/pymarxan/models/__init__.py`:

```python
from pymarxan.models.problem import ConservationProblem, has_geometry

__all__ = ["ConservationProblem", "has_geometry"]
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/spatial/test_geometry_detection.py -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add src/pymarxan/models/problem.py src/pymarxan/models/__init__.py tests/pymarxan/spatial/test_geometry_detection.py
git commit -m "feat(models): add has_geometry() helper for GeoDataFrame detection"
```

---

### Task 5: Upgrade `map_utils.py` to render real GeoDataFrame polygons

**Files:**
- Modify: `src/pymarxan_shiny/modules/mapping/map_utils.py`
- Create: `tests/pymarxan_shiny/test_map_utils.py`

**Step 1: Write the failing tests**

```python
# tests/pymarxan_shiny/test_map_utils.py
"""Tests for map_utils geometry rendering."""
from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

try:
    from pymarxan_shiny.modules.mapping.map_utils import create_geo_map
    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


@pytest.mark.skipif(not _HAS_IPYLEAFLET, reason="ipyleaflet not installed")
class TestCreateGeoMap:
    def test_creates_map_from_geodataframe(self):
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2)],
            crs="EPSG:4326",
        )
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        m = create_geo_map(gdf, colors)
        import ipyleaflet
        assert isinstance(m, ipyleaflet.Map)

    def test_map_has_correct_number_of_layers(self):
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        colors = ["#ff0000", "#00ff00"]
        m = create_geo_map(gdf, colors)
        # Default layers (tile) + our geo layers
        geo_layers = [l for l in m.layers if hasattr(l, "data")]
        assert len(geo_layers) == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan_shiny/test_map_utils.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_geo_map'`

**Step 3: Implement `create_geo_map`**

Add to `src/pymarxan_shiny/modules/mapping/map_utils.py`:

```python
import json


def create_geo_map(
    gdf: "gpd.GeoDataFrame",
    colors: list[str],
    center: tuple[float, float] | None = None,
    zoom: int = 10,
) -> "ipyleaflet.Map":
    """Create ipyleaflet Map with GeoJSON layers from a GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Must have a geometry column with Polygon geometries.
    colors : list[str]
        One hex color string per row in gdf.
    center : optional (lat, lon)
        Map center. Auto-computed from gdf bounds if not provided.
    zoom : int
        Initial zoom level.
    """
    if not _HAS_IPYLEAFLET:
        raise ImportError("ipyleaflet is required for create_geo_map")

    if center is None:
        b = gdf.total_bounds  # [minx, miny, maxx, maxy]
        center = ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2)

    m = ipyleaflet.Map(center=center, zoom=zoom)

    for idx, (_, row) in enumerate(gdf.iterrows()):
        geojson_data = json.loads(row.geometry.__geo_interface__.__class__.__module__
                                  and json.dumps(row.geometry.__geo_interface__))
        geo_json = ipyleaflet.GeoJSON(
            data={
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {},
            },
            style={
                "color": colors[idx] if idx < len(colors) else "#999999",
                "fillColor": colors[idx] if idx < len(colors) else "#999999",
                "fillOpacity": 0.7,
                "weight": 1,
            },
        )
        m.add(geo_json)

    return m
```

**Note:** Simplify the GeoJSON creation — remove the awkward intermediate step:

```python
def create_geo_map(
    gdf: "gpd.GeoDataFrame",
    colors: list[str],
    center: tuple[float, float] | None = None,
    zoom: int = 10,
) -> "ipyleaflet.Map":
    """Create ipyleaflet Map with GeoJSON layers from a GeoDataFrame."""
    if not _HAS_IPYLEAFLET:
        raise ImportError("ipyleaflet is required for create_geo_map")

    if center is None:
        b = gdf.total_bounds  # [minx, miny, maxx, maxy]
        center = ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2)

    m = ipyleaflet.Map(center=center, zoom=zoom)

    for idx, (_, row) in enumerate(gdf.iterrows()):
        color = colors[idx] if idx < len(colors) else "#999999"
        geo_json = ipyleaflet.GeoJSON(
            data={
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {},
            },
            style={
                "color": color,
                "fillColor": color,
                "fillOpacity": 0.7,
                "weight": 1,
            },
        )
        m.add(geo_json)

    return m
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan_shiny/test_map_utils.py -v`
Expected: 2 PASS (or SKIP if no ipyleaflet)

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/map_utils.py tests/pymarxan_shiny/test_map_utils.py
git commit -m "feat(mapping): add create_geo_map for real GeoDataFrame rendering"
```

---

### Task 6: Upgrade `solution_map.py` to use real geometry when available

**Files:**
- Modify: `src/pymarxan_shiny/modules/mapping/solution_map.py`

**Step 1: Understand the change**

The current `solution_map_server` always calls `generate_grid(n_pu)` for synthetic rectangles. After this change:
1. Import `has_geometry` from `pymarxan.models.problem`
2. If `has_geometry(p)` is True, call `create_geo_map(p.planning_units, colors)`
3. If False, fall back to current `generate_grid()` + `create_grid_map()` path

**Step 2: Implement the change**

Modify `src/pymarxan_shiny/modules/mapping/solution_map.py`:

Add `has_geometry` import at top:
```python
from pymarxan.models.problem import has_geometry
```

Replace the `map()` render function inside `if _HAS_IPYLEAFLET:` block:

```python
        @render_widget
        def map():
            p = problem()
            s = solution()
            if p is None or s is None:
                return None

            n_pu = len(p.planning_units)
            colors = [
                "#2ecc71" if s.selected[i] else "#95a5a6"
                for i in range(n_pu)
            ]

            if has_geometry(p):
                return create_geo_map(p.planning_units, colors)

            grid = generate_grid(n_pu)
            return create_grid_map(grid, colors)
```

Also add `create_geo_map` to the try-import block:

```python
    from pymarxan_shiny.modules.mapping.map_utils import create_geo_map, create_grid_map
```

**Step 3: Run full test suite to verify no regression**

Run: `pytest tests/ -x -q`
Expected: All tests pass (521+)

**Step 4: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/solution_map.py
git commit -m "feat(mapping): solution_map renders real geometry when available"
```

---

### Task 7: Upgrade remaining 4 map modules for geometry support

**Files:**
- Modify: `src/pymarxan_shiny/modules/mapping/spatial_grid.py`
- Modify: `src/pymarxan_shiny/modules/mapping/frequency_map.py`
- Modify: `src/pymarxan_shiny/modules/mapping/comparison_map.py`
- Modify: `src/pymarxan_shiny/modules/mapping/network_view.py`

**Step 1: Apply same pattern to all 4 modules**

Each module needs:
1. Add `from pymarxan.models.problem import has_geometry` import
2. Add `create_geo_map` to the try-import block alongside `create_grid_map`
3. In the `map()` render function, check `has_geometry(p)` — if True, use `create_geo_map`, otherwise fall back to `generate_grid()` + `create_grid_map()`

The pattern is identical across all modules — only the color function differs per module.

**Step 2: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/
git commit -m "feat(mapping): upgrade all map modules to render real geometry"
```

---

### Task 8: Create `grid_builder` Shiny module

**Files:**
- Create: `src/pymarxan_shiny/modules/spatial/__init__.py`
- Create: `src/pymarxan_shiny/modules/spatial/grid_builder.py`
- Modify: `src/pymarxan_app/app.py` (add to Data tab)
- Create: `tests/pymarxan_shiny/test_grid_builder.py`

**Step 1: Write the failing test**

```python
# tests/pymarxan_shiny/test_grid_builder.py
"""Tests for grid builder Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.spatial.grid_builder import grid_builder_server, grid_builder_ui


def test_grid_builder_ui_callable():
    assert callable(grid_builder_ui)


def test_grid_builder_server_callable():
    assert callable(grid_builder_server)
```

**Step 2: Run to verify failure**

Run: `pytest tests/pymarxan_shiny/test_grid_builder.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement**

```python
# src/pymarxan_shiny/modules/spatial/__init__.py
# (empty)

# src/pymarxan_shiny/modules/spatial/grid_builder.py
"""Grid builder Shiny module — generate planning unit grids."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.spatial.grid import compute_adjacency, generate_planning_grid

try:
    from shinywidgets import output_widget, render_widget
    from pymarxan_shiny.modules.mapping.map_utils import create_geo_map
    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


@module.ui
def grid_builder_ui():
    map_output = output_widget("grid_map") if _HAS_IPYLEAFLET else ui.output_ui("grid_map_text")
    return ui.card(
        ui.card_header("Generate Planning Grid"),
        ui.layout_columns(
            ui.input_numeric("minx", "Min X (lon)", value=0.0),
            ui.input_numeric("miny", "Min Y (lat)", value=0.0),
            ui.input_numeric("maxx", "Max X (lon)", value=1.0),
            ui.input_numeric("maxy", "Max Y (lat)", value=1.0),
            col_widths=[3, 3, 3, 3],
        ),
        ui.layout_columns(
            ui.input_numeric("cell_size", "Cell Size", value=0.1, min=0.001),
            ui.input_select("grid_type", "Grid Type", {"square": "Square", "hexagonal": "Hexagonal"}),
            col_widths=[6, 6],
        ),
        ui.input_action_button("generate", "Generate Grid", class_="btn-primary"),
        map_output,
        ui.output_text_verbatim("grid_info"),
    )


@module.server
def grid_builder_server(input, output, session, problem: reactive.Value):

    @reactive.effect
    @reactive.event(input.generate)
    def _generate():
        bounds = (input.minx(), input.miny(), input.maxx(), input.maxy())
        gdf = generate_planning_grid(
            bounds=bounds,
            cell_size=input.cell_size(),
            grid_type=input.grid_type(),
        )
        if len(gdf) == 0:
            ui.notification_show("No cells generated. Check bounds and cell size.", type="warning")
            return

        boundary = compute_adjacency(gdf)
        import pandas as pd
        p = ConservationProblem(
            planning_units=gdf,
            features=pd.DataFrame({"id": [], "name": [], "target": [], "spf": []}),
            pu_vs_features=pd.DataFrame({"species": [], "pu": [], "amount": []}),
            boundary=boundary if len(boundary) > 0 else None,
        )
        problem.set(p)
        ui.notification_show(f"Generated {len(gdf)} planning units.", type="message")

    @render.text
    def grid_info():
        p = problem()
        if p is None:
            return "No grid generated yet."
        n = len(p.planning_units)
        has_bnd = p.boundary is not None and len(p.boundary) > 0
        return f"{n} planning units | Boundary edges: {len(p.boundary) if has_bnd else 0}"

    if _HAS_IPYLEAFLET:
        @render_widget
        def grid_map():
            p = problem()
            if p is None:
                return None
            colors = ["#3498db"] * len(p.planning_units)
            return create_geo_map(p.planning_units, colors)

    if not _HAS_IPYLEAFLET:
        @render.ui
        def grid_map_text():
            p = problem()
            if p is None:
                return ui.p("Generate a grid to see the preview.")
            return ui.p(f"Grid preview: {len(p.planning_units)} planning units")
```

**Step 4: Add to app.py**

Add import at top of `app.py`:
```python
from pymarxan_shiny.modules.spatial.grid_builder import grid_builder_server, grid_builder_ui
```

Add `grid_builder_ui("grid_gen")` to the Data tab layout, and `grid_builder_server("grid_gen", problem=problem)` to the server function.

**Step 5: Run tests**

Run: `pytest tests/pymarxan_shiny/test_grid_builder.py -v`
Expected: 2 PASS

**Step 6: Commit**

```bash
git add src/pymarxan_shiny/modules/spatial/ src/pymarxan_app/app.py tests/pymarxan_shiny/test_grid_builder.py
git commit -m "feat(shiny): add grid builder module — generate PU grids from UI"
```

---

### Task 9: Add `[spatial]` optional deps + `@pytest.mark.spatial` marker + regression

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py` (if marker registration needed)

**Step 1: Add `spatial` dependency group and marker**

In `pyproject.toml`, add to `[project.optional-dependencies]`:
```toml
spatial = ["rasterio>=1.3", "requests>=2.28"]
all = ["pymarxan[shiny,dev,spatial]"]
```

Add to `[tool.pytest.ini_options]` markers:
```toml
markers = [
    "slow: marks tests as slow (SA-heavy, zone solver, run_mode pipelines)",
    "integration: marks phase integration tests",
    "spatial: marks tests requiring geopandas/spatial features",
]
```

**Step 2: Run full regression**

Run: `pytest tests/ -v --cov --cov-report=term-missing --cov-fail-under=75`
Expected: All tests pass, coverage >= 75%

Run: `ruff check src/ tests/`
Expected: Clean

Run: `mypy src/pymarxan/ --ignore-missing-imports`
Expected: Clean

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add [spatial] optional deps, pytest marker, update [all]"
```

---

## Summary

| Task | Description | New Tests |
|------|-------------|-----------|
| 1 | Square grid generation + spatial package | 6 |
| 2 | Hex grid + clipping tests | 4 |
| 3 | Adjacency computation tests | 3 |
| 4 | `has_geometry()` helper | 3 |
| 5 | `create_geo_map` in map_utils | 2 |
| 6 | Upgrade solution_map for real geometry | 0 (regression) |
| 7 | Upgrade 4 remaining map modules | 0 (regression) |
| 8 | Grid builder Shiny module | 2 |
| 9 | Deps, marker, regression | 0 |
| **Total** | | **~20** |
