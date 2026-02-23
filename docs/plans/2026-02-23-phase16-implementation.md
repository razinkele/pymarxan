# Phase 16: Shapefile/GeoJSON Import + Custom Cost Surfaces

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Import planning units from GIS formats (shapefile, GeoJSON, GeoPackage), import features from vector overlays, and apply custom cost surfaces from vector layers. Raster support is deferred to a follow-up if `rasterio` proves problematic.

**Architecture:** New `pymarxan.spatial.importers` handles PU and feature import from vector files via geopandas. New `pymarxan.spatial.cost_surface` handles vector-based cost overlay. Both produce standard DataFrames/GeoDataFrames compatible with `ConservationProblem`. Shiny modules provide file upload + column mapping UI.

**Tech Stack:** geopandas (core dep), shapely (core dep). Raster support (rasterio) is out of scope for this phase.

---

### Task 1: Implement `import_planning_units` from vector files

**Files:**
- Create: `src/pymarxan/spatial/importers.py`
- Create: `tests/pymarxan/spatial/test_importers.py`
- Create: `tests/data/spatial/` directory with test fixtures

**Step 1: Create test fixture**

Create a small GeoJSON file for testing:

```json
// tests/data/spatial/test_pus.geojson
{
  "type": "FeatureCollection",
  "features": [
    {"type": "Feature", "properties": {"pu_id": 1, "pu_cost": 5.0, "lock": 0}, "geometry": {"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]}},
    {"type": "Feature", "properties": {"pu_id": 2, "pu_cost": 3.0, "lock": 2}, "geometry": {"type": "Polygon", "coordinates": [[[1,0],[2,0],[2,1],[1,1],[1,0]]]}},
    {"type": "Feature", "properties": {"pu_id": 3, "pu_cost": 7.0, "lock": 0}, "geometry": {"type": "Polygon", "coordinates": [[[0,1],[1,1],[1,2],[0,2],[0,1]]]}}
  ]
}
```

**Step 2: Write the failing tests**

```python
# tests/pymarxan/spatial/test_importers.py
"""Tests for GIS file importers."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from pymarxan.spatial.importers import import_features_from_vector, import_planning_units

SPATIAL_DATA = Path(__file__).parent.parent.parent / "data" / "spatial"


class TestImportPlanningUnits:
    def test_import_geojson_with_column_mapping(self):
        gdf = import_planning_units(
            SPATIAL_DATA / "test_pus.geojson",
            id_column="pu_id",
            cost_column="pu_cost",
            status_column="lock",
        )
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 3
        assert set(gdf.columns) >= {"id", "cost", "status", "geometry"}
        assert gdf["id"].tolist() == [1, 2, 3]
        assert gdf["cost"].tolist() == [5.0, 3.0, 7.0]
        assert gdf["status"].tolist() == [0, 2, 0]

    def test_import_defaults_missing_cost(self):
        # Create in-memory GeoDataFrame without cost column
        gdf_src = gpd.GeoDataFrame(
            {"myid": [1, 2]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        path = SPATIAL_DATA / "_temp_no_cost.geojson"
        gdf_src.to_file(path, driver="GeoJSON")
        try:
            gdf = import_planning_units(path, id_column="myid", cost_column="cost")
            assert all(gdf["cost"] == 1.0)
        finally:
            path.unlink(missing_ok=True)

    def test_import_defaults_missing_status(self):
        gdf_src = gpd.GeoDataFrame(
            {"id": [1, 2], "cost": [1.0, 2.0]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        path = SPATIAL_DATA / "_temp_no_status.geojson"
        gdf_src.to_file(path, driver="GeoJSON")
        try:
            gdf = import_planning_units(path, status_column=None)
            assert all(gdf["status"] == 0)
        finally:
            path.unlink(missing_ok=True)

    def test_import_invalid_id_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            import_planning_units(
                SPATIAL_DATA / "test_pus.geojson",
                id_column="nonexistent",
            )

    def test_import_preserves_crs(self):
        gdf = import_planning_units(
            SPATIAL_DATA / "test_pus.geojson",
            id_column="pu_id",
            cost_column="pu_cost",
        )
        assert gdf.crs is not None
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/pymarxan/spatial/test_importers.py::TestImportPlanningUnits -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 4: Implement**

```python
# src/pymarxan/spatial/importers.py
"""Import planning units and features from GIS files."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def import_planning_units(
    path: str | Path,
    id_column: str = "id",
    cost_column: str = "cost",
    status_column: str | None = "status",
) -> gpd.GeoDataFrame:
    """Import planning units from shapefile, GeoJSON, or GeoPackage.

    Parameters
    ----------
    path : str or Path
        Path to the GIS file.
    id_column : str
        Column name to use as planning unit ID.
    cost_column : str
        Column name to use as cost. If missing, defaults to 1.0.
    status_column : str or None
        Column name for status. If None or missing, defaults to 0.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: id, cost, status, geometry. CRS preserved from input.
    """
    gdf = gpd.read_file(path)

    # Validate ID column
    if id_column not in gdf.columns:
        raise ValueError(
            f"ID column '{id_column}' not found in file. "
            f"Available columns: {list(gdf.columns)}"
        )

    # Build output
    result = gpd.GeoDataFrame(crs=gdf.crs)
    result["id"] = gdf[id_column].astype(int)

    if cost_column in gdf.columns:
        result["cost"] = gdf[cost_column].astype(float)
    else:
        result["cost"] = 1.0

    if status_column is not None and status_column in gdf.columns:
        result["status"] = gdf[status_column].astype(int)
    else:
        result["status"] = 0

    result = result.set_geometry(gdf.geometry)
    return result


def import_features_from_vector(
    path: str | Path,
    planning_units: gpd.GeoDataFrame,
    feature_name: str,
    feature_id: int,
    amount_column: str | None = None,
) -> pd.DataFrame:
    """Compute feature amounts per PU via spatial overlay.

    Parameters
    ----------
    path : str or Path
        Path to feature vector file.
    planning_units : gpd.GeoDataFrame
        Must have ``id`` and ``geometry`` columns.
    feature_name : str
        Name for the feature (for display).
    feature_id : int
        Numeric ID for the feature.
    amount_column : str or None
        Column in the feature file to sum per PU.
        If None, uses area of intersection as amount.

    Returns
    -------
    pd.DataFrame
        Columns: species, pu, amount.
    """
    features_gdf = gpd.read_file(path)

    # Reproject if CRS differs
    if features_gdf.crs != planning_units.crs and features_gdf.crs is not None:
        features_gdf = features_gdf.to_crs(planning_units.crs)

    overlay = gpd.overlay(planning_units[["id", "geometry"]], features_gdf, how="intersection")

    rows = []
    if amount_column and amount_column in overlay.columns:
        grouped = overlay.groupby("id")[amount_column].sum()
        for pu_id, amount in grouped.items():
            if amount > 0:
                rows.append({"species": feature_id, "pu": int(pu_id), "amount": float(amount)})
    else:
        # Use intersection area
        overlay["_area"] = overlay.geometry.area
        grouped = overlay.groupby("id")["_area"].sum()
        for pu_id, area in grouped.items():
            if area > 0:
                rows.append({"species": feature_id, "pu": int(pu_id), "amount": float(area)})

    return pd.DataFrame(rows, columns=["species", "pu", "amount"])
```

Update `src/pymarxan/spatial/__init__.py` to add:
```python
from pymarxan.spatial.importers import import_features_from_vector, import_planning_units
```

And update `__all__`.

**Step 5: Run tests**

Run: `pytest tests/pymarxan/spatial/test_importers.py::TestImportPlanningUnits -v`
Expected: 5 PASS

**Step 6: Commit**

```bash
git add src/pymarxan/spatial/importers.py src/pymarxan/spatial/__init__.py tests/pymarxan/spatial/test_importers.py tests/data/spatial/
git commit -m "feat(spatial): add planning unit import from shapefile/GeoJSON/GeoPackage"
```

---

### Task 2: Implement `import_features_from_vector`

**Files:**
- Modify: `tests/pymarxan/spatial/test_importers.py`
- Create: `tests/data/spatial/test_features.geojson`

**Step 1: Create test fixture**

```json
// tests/data/spatial/test_features.geojson
{
  "type": "FeatureCollection",
  "features": [
    {"type": "Feature", "properties": {"habitat": "forest", "area_ha": 50.0}, "geometry": {"type": "Polygon", "coordinates": [[[0,0],[0.8,0],[0.8,0.8],[0,0.8],[0,0]]]}},
    {"type": "Feature", "properties": {"habitat": "wetland", "area_ha": 30.0}, "geometry": {"type": "Polygon", "coordinates": [[[1.2,0],[2,0],[2,0.8],[1.2,0.8],[1.2,0]]]}}
  ]
}
```

**Step 2: Write the failing tests**

Add to `tests/pymarxan/spatial/test_importers.py`:

```python
class TestImportFeaturesFromVector:
    def _make_pus(self):
        return gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2)],
            crs="EPSG:4326",
        )

    def test_import_with_area_overlap(self):
        pus = self._make_pus()
        df = import_features_from_vector(
            SPATIAL_DATA / "test_features.geojson",
            pus,
            feature_name="forest",
            feature_id=1,
        )
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"species", "pu", "amount"}
        assert all(df["species"] == 1)
        assert all(df["amount"] > 0)

    def test_import_with_amount_column(self):
        pus = self._make_pus()
        df = import_features_from_vector(
            SPATIAL_DATA / "test_features.geojson",
            pus,
            feature_name="forest",
            feature_id=1,
            amount_column="area_ha",
        )
        assert all(df["amount"] > 0)

    def test_no_overlap_returns_empty(self):
        # PUs far from features
        pus = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(100, 100, 101, 101)],
            crs="EPSG:4326",
        )
        df = import_features_from_vector(
            SPATIAL_DATA / "test_features.geojson",
            pus,
            feature_name="forest",
            feature_id=1,
        )
        assert len(df) == 0
```

**Step 3: Run tests**

Run: `pytest tests/pymarxan/spatial/test_importers.py::TestImportFeaturesFromVector -v`
Expected: 3 PASS (implementation done in Task 1)

**Step 4: Commit**

```bash
git add tests/pymarxan/spatial/test_importers.py tests/data/spatial/test_features.geojson
git commit -m "test(spatial): add vector feature import tests"
```

---

### Task 3: Implement `cost_surface.py` — vector cost overlay

**Files:**
- Create: `src/pymarxan/spatial/cost_surface.py`
- Create: `tests/pymarxan/spatial/test_cost_surface.py`
- Modify: `src/pymarxan/spatial/__init__.py`

**Step 1: Write the failing tests**

```python
# tests/pymarxan/spatial/test_cost_surface.py
"""Tests for cost surface processing."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pymarxan.spatial.cost_surface import apply_cost_from_vector, combine_cost_layers


def _make_pus():
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4], "cost": [1.0, 1.0, 1.0, 1.0], "status": [0, 0, 0, 0]},
        geometry=[
            box(0, 0, 1, 1), box(1, 0, 2, 1),
            box(0, 1, 1, 2), box(1, 1, 2, 2),
        ],
        crs="EPSG:4326",
    )


class TestApplyCostFromVector:
    def test_full_coverage_assigns_cost(self):
        pus = _make_pus()
        cost_layer = gpd.GeoDataFrame(
            {"cost_val": [10.0]},
            geometry=[box(0, 0, 2, 2)],  # Covers all PUs
            crs="EPSG:4326",
        )
        result = apply_cost_from_vector(pus, cost_layer, cost_column="cost_val")
        assert all(result["cost"] == 10.0)

    def test_partial_coverage_area_weighted(self):
        pus = _make_pus()
        # Cost layer covers left half only
        cost_layer = gpd.GeoDataFrame(
            {"cost_val": [20.0]},
            geometry=[box(0, 0, 1, 2)],
            crs="EPSG:4326",
        )
        result = apply_cost_from_vector(
            pus, cost_layer, cost_column="cost_val",
            aggregation="area_weighted_mean",
        )
        # PU 1 and 3 (fully covered) should have cost 20
        assert result.loc[result["id"] == 1, "cost"].iloc[0] == pytest.approx(20.0, abs=0.1)
        # PU 2 and 4 (not covered) should keep original cost
        assert result.loc[result["id"] == 2, "cost"].iloc[0] == pytest.approx(1.0, abs=0.1)

    def test_does_not_mutate_input(self):
        pus = _make_pus()
        cost_layer = gpd.GeoDataFrame(
            {"cost_val": [10.0]},
            geometry=[box(0, 0, 2, 2)],
            crs="EPSG:4326",
        )
        apply_cost_from_vector(pus, cost_layer, cost_column="cost_val")
        assert all(pus["cost"] == 1.0)


class TestCombineCostLayers:
    def test_equal_weight_combination(self):
        pus = _make_pus()
        layer1 = np.array([10.0, 20.0, 30.0, 40.0])
        layer2 = np.array([40.0, 30.0, 20.0, 10.0])
        result = combine_cost_layers(
            pus,
            layers=[("layer1", layer1), ("layer2", layer2)],
        )
        # Normalized: both should be [0, 0.33, 0.67, 1.0] and [1.0, 0.67, 0.33, 0]
        # Equal weight sum: all approximately 1.0
        costs = result["cost"].values
        assert np.allclose(costs, costs[0], atol=0.01)

    def test_weighted_combination(self):
        pus = _make_pus()
        layer1 = np.array([0.0, 0.0, 0.0, 0.0])
        layer2 = np.array([10.0, 10.0, 10.0, 10.0])
        result = combine_cost_layers(
            pus,
            layers=[("zero", layer1), ("ten", layer2)],
            weights=[0.0, 1.0],
        )
        # Only layer2 matters, all equal → all same cost
        assert len(set(result["cost"].round(4).tolist())) == 1

    def test_single_layer(self):
        pus = _make_pus()
        layer = np.array([5.0, 10.0, 15.0, 20.0])
        result = combine_cost_layers(pus, layers=[("costs", layer)])
        # Single layer normalized to [0, 0.33, 0.67, 1.0]
        assert result["cost"].iloc[0] < result["cost"].iloc[-1]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/spatial/test_cost_surface.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement**

```python
# src/pymarxan/spatial/cost_surface.py
"""Cost surface processing for conservation planning."""
from __future__ import annotations

import geopandas as gpd
import numpy as np


def apply_cost_from_vector(
    planning_units: gpd.GeoDataFrame,
    cost_layer: gpd.GeoDataFrame,
    cost_column: str,
    aggregation: str = "area_weighted_mean",
) -> gpd.GeoDataFrame:
    """Compute cost from vector overlay.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id``, ``cost``, and ``geometry`` columns.
    cost_layer : gpd.GeoDataFrame
        Vector layer with cost values.
    cost_column : str
        Column in cost_layer containing cost values.
    aggregation : str
        ``"area_weighted_mean"`` | ``"sum"`` | ``"max"``.

    Returns
    -------
    gpd.GeoDataFrame
        Copy with updated cost column. Original PU cost preserved where
        no overlay exists.
    """
    result = planning_units.copy()

    # Reproject if needed
    if cost_layer.crs != planning_units.crs and cost_layer.crs is not None:
        cost_layer = cost_layer.to_crs(planning_units.crs)

    overlay = gpd.overlay(
        planning_units[["id", "geometry"]],
        cost_layer[[cost_column, "geometry"]],
        how="intersection",
    )

    if len(overlay) == 0:
        return result

    overlay["_intersection_area"] = overlay.geometry.area

    new_costs = {}
    for pu_id, group in overlay.groupby("id"):
        pu_area = planning_units.loc[planning_units["id"] == pu_id, "geometry"].iloc[0].area
        if aggregation == "area_weighted_mean":
            weighted = (group[cost_column] * group["_intersection_area"]).sum()
            total_area = group["_intersection_area"].sum()
            if total_area > 0 and total_area >= pu_area * 0.01:
                new_costs[pu_id] = weighted / total_area
        elif aggregation == "sum":
            new_costs[pu_id] = group[cost_column].sum()
        elif aggregation == "max":
            new_costs[pu_id] = group[cost_column].max()

    for pu_id, cost in new_costs.items():
        result.loc[result["id"] == pu_id, "cost"] = cost

    return result


def combine_cost_layers(
    planning_units: gpd.GeoDataFrame,
    layers: list[tuple[str, np.ndarray]],
    weights: list[float] | None = None,
) -> gpd.GeoDataFrame:
    """Combine multiple cost arrays with optional weighting.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id`` and ``cost`` columns.
    layers : list of (name, array) tuples
        Each array has one value per PU.
    weights : list of float or None
        Per-layer weights. If None, equal weighting.

    Returns
    -------
    gpd.GeoDataFrame
        Copy with updated cost column (weighted sum of normalized layers).
    """
    result = planning_units.copy()
    n_layers = len(layers)
    if n_layers == 0:
        return result

    if weights is None:
        weights = [1.0 / n_layers] * n_layers

    combined = np.zeros(len(planning_units), dtype=float)
    for (name, values), w in zip(layers, weights):
        arr = np.asarray(values, dtype=float)
        # Min-max normalize
        vmin, vmax = arr.min(), arr.max()
        if vmax > vmin:
            normalized = (arr - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(arr)
        combined += w * normalized

    result["cost"] = combined
    return result
```

Update `src/pymarxan/spatial/__init__.py` to add:
```python
from pymarxan.spatial.cost_surface import apply_cost_from_vector, combine_cost_layers
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/spatial/test_cost_surface.py -v`
Expected: 6 PASS

**Step 5: Commit**

```bash
git add src/pymarxan/spatial/cost_surface.py src/pymarxan/spatial/__init__.py tests/pymarxan/spatial/test_cost_surface.py
git commit -m "feat(spatial): add vector cost surface overlay and multi-layer combination"
```

---

### Task 4: Create `import_wizard` Shiny module

**Files:**
- Create: `src/pymarxan_shiny/modules/spatial/import_wizard.py`
- Create: `tests/pymarxan_shiny/test_import_wizard.py`

**Step 1: Write the failing test**

```python
# tests/pymarxan_shiny/test_import_wizard.py
"""Tests for import wizard Shiny module."""
from pymarxan_shiny.modules.spatial.import_wizard import import_wizard_server, import_wizard_ui


def test_import_wizard_ui_callable():
    assert callable(import_wizard_ui)


def test_import_wizard_server_callable():
    assert callable(import_wizard_server)
```

**Step 2: Implement**

```python
# src/pymarxan_shiny/modules/spatial/import_wizard.py
"""GIS file import wizard Shiny module."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.spatial.grid import compute_adjacency
from pymarxan.spatial.importers import import_planning_units


@module.ui
def import_wizard_ui():
    return ui.card(
        ui.card_header("Import Planning Units from GIS File"),
        ui.input_file("pu_file", "Upload File (.shp, .geojson, .gpkg)", accept=[".shp", ".geojson", ".gpkg", ".json", ".zip"]),
        ui.layout_columns(
            ui.input_text("id_col", "ID Column", value="id"),
            ui.input_text("cost_col", "Cost Column", value="cost"),
            ui.input_text("status_col", "Status Column (optional)", value="status"),
            col_widths=[4, 4, 4],
        ),
        ui.input_action_button("import_btn", "Import", class_="btn-primary"),
        ui.output_text_verbatim("import_info"),
    )


@module.server
def import_wizard_server(input, output, session, problem: reactive.Value):

    @reactive.effect
    @reactive.event(input.import_btn)
    def _import():
        file_info = input.pu_file()
        if not file_info:
            ui.notification_show("Please upload a file first.", type="warning")
            return

        path = file_info[0]["datapath"]
        status_col = input.status_col() or None

        try:
            gdf = import_planning_units(
                path,
                id_column=input.id_col(),
                cost_column=input.cost_col(),
                status_column=status_col,
            )
            boundary = compute_adjacency(gdf)
            import pandas as pd
            p = ConservationProblem(
                planning_units=gdf,
                features=pd.DataFrame({"id": [], "name": [], "target": [], "spf": []}),
                pu_vs_features=pd.DataFrame({"species": [], "pu": [], "amount": []}),
                boundary=boundary if len(boundary) > 0 else None,
            )
            problem.set(p)
            ui.notification_show(f"Imported {len(gdf)} planning units.", type="message")
        except Exception as e:
            ui.notification_show(f"Import error: {e}", type="error")

    @render.text
    def import_info():
        p = problem()
        if p is None:
            return "Upload a GIS file and click Import."
        return f"{len(p.planning_units)} planning units loaded"
```

**Step 3: Run tests**

Run: `pytest tests/pymarxan_shiny/test_import_wizard.py -v`
Expected: 2 PASS

**Step 4: Commit**

```bash
git add src/pymarxan_shiny/modules/spatial/import_wizard.py tests/pymarxan_shiny/test_import_wizard.py
git commit -m "feat(shiny): add GIS file import wizard module"
```

---

### Task 5: Create `cost_upload` Shiny module

**Files:**
- Create: `src/pymarxan_shiny/modules/spatial/cost_upload.py`
- Create: `tests/pymarxan_shiny/test_cost_upload.py`

**Step 1: Write the failing test**

```python
# tests/pymarxan_shiny/test_cost_upload.py
"""Tests for cost upload Shiny module."""
from pymarxan_shiny.modules.spatial.cost_upload import cost_upload_server, cost_upload_ui


def test_cost_upload_ui_callable():
    assert callable(cost_upload_ui)


def test_cost_upload_server_callable():
    assert callable(cost_upload_server)
```

**Step 2: Implement**

```python
# src/pymarxan_shiny/modules/spatial/cost_upload.py
"""Cost surface upload Shiny module."""
from __future__ import annotations

import geopandas as gpd
from shiny import module, reactive, render, ui

from pymarxan.models.problem import has_geometry
from pymarxan.spatial.cost_surface import apply_cost_from_vector


@module.ui
def cost_upload_ui():
    return ui.card(
        ui.card_header("Custom Cost Surface"),
        ui.input_file("cost_file", "Upload Cost Layer (.shp, .geojson, .gpkg)", accept=[".shp", ".geojson", ".gpkg", ".json", ".zip"]),
        ui.layout_columns(
            ui.input_text("cost_col", "Cost Column", value="cost"),
            ui.input_select("aggregation", "Aggregation", {
                "area_weighted_mean": "Area-Weighted Mean",
                "sum": "Sum",
                "max": "Maximum",
            }),
            col_widths=[6, 6],
        ),
        ui.input_action_button("apply_cost", "Apply Cost Surface", class_="btn-primary"),
        ui.output_text_verbatim("cost_info"),
    )


@module.server
def cost_upload_server(input, output, session, problem: reactive.Value):

    @reactive.effect
    @reactive.event(input.apply_cost)
    def _apply():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first.", type="warning")
            return
        if not has_geometry(p):
            ui.notification_show("Planning units need geometry.", type="warning")
            return

        file_info = input.cost_file()
        if not file_info:
            ui.notification_show("Upload a cost layer file.", type="warning")
            return

        try:
            cost_layer = gpd.read_file(file_info[0]["datapath"])
            updated = apply_cost_from_vector(
                p.planning_units,
                cost_layer,
                cost_column=input.cost_col(),
                aggregation=input.aggregation(),
            )
            import copy
            new_problem = copy.deepcopy(p)
            new_problem.planning_units = updated
            problem.set(new_problem)
            ui.notification_show("Cost surface applied.", type="message")
        except Exception as e:
            ui.notification_show(f"Cost error: {e}", type="error")

    @render.text
    def cost_info():
        p = problem()
        if p is None:
            return "Load a project to apply cost surfaces."
        costs = p.planning_units["cost"]
        return f"Cost range: {costs.min():.2f} – {costs.max():.2f} (mean: {costs.mean():.2f})"
```

**Step 3: Run tests**

Run: `pytest tests/pymarxan_shiny/test_cost_upload.py -v`
Expected: 2 PASS

**Step 4: Commit**

```bash
git add src/pymarxan_shiny/modules/spatial/cost_upload.py tests/pymarxan_shiny/test_cost_upload.py
git commit -m "feat(shiny): add cost surface upload module"
```

---

### Task 6: Integrate into app.py + regression

**Files:**
- Modify: `src/pymarxan_app/app.py`

**Step 1: Add imports and wire up**

Add imports:
```python
from pymarxan_shiny.modules.spatial.import_wizard import import_wizard_server, import_wizard_ui
from pymarxan_shiny.modules.spatial.cost_upload import cost_upload_server, cost_upload_ui
```

Add to Data tab layout (alongside existing upload and grid_builder):
```python
import_wizard_ui("import_wiz"),
cost_upload_ui("cost"),
```

Add server calls:
```python
import_wizard_server("import_wiz", problem=problem)
cost_upload_server("cost", problem=problem)
```

**Step 2: Run full regression**

Run: `pytest tests/ -v --cov --cov-report=term-missing --cov-fail-under=75`
Expected: All tests pass, coverage >= 75%

Run: `ruff check src/ tests/`
Expected: Clean

**Step 3: Commit**

```bash
git add src/pymarxan_app/app.py
git commit -m "feat(app): integrate import wizard and cost upload into Data tab"
```

---

## Summary

| Task | Description | New Tests |
|------|-------------|-----------|
| 1 | `import_planning_units` from vector files | 5 |
| 2 | `import_features_from_vector` | 3 |
| 3 | `cost_surface.py` — vector cost overlay + combine | 6 |
| 4 | Import wizard Shiny module | 2 |
| 5 | Cost upload Shiny module | 2 |
| 6 | App integration + regression | 0 |
| **Total** | | **~18** |
