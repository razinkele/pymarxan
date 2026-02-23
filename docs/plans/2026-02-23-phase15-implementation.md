# Phase 15: GADM Admin Boundaries + WDPA Protected Areas

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add web API integration to fetch GADM administrative boundaries (geoBoundaries API) and WDPA protected areas (Protected Planet API), with Shiny modules for both.

**Architecture:** Two new modules in `pymarxan.spatial`: `gadm.py` fetches admin boundaries as GeoDataFrames, `wdpa.py` fetches protected areas and applies status overrides to planning units. Both use `requests` for HTTP. All API calls are mocked in tests.

**Tech Stack:** geopandas, shapely, requests (new `[spatial]` dep from Phase 14)

---

### Task 1: Implement `gadm.py` — fetch admin boundaries

**Files:**
- Create: `src/pymarxan/spatial/gadm.py`
- Create: `tests/pymarxan/spatial/test_gadm.py`
- Modify: `src/pymarxan/spatial/__init__.py`

**Step 1: Write the failing tests**

```python
# tests/pymarxan/spatial/test_gadm.py
"""Tests for GADM admin boundary fetching."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from pymarxan.spatial.gadm import fetch_gadm, list_countries

# Sample GeoJSON response from geoBoundaries API
_SAMPLE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"shapeName": "TestCountry"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        }
    ],
}

_SAMPLE_API_RESPONSE = {
    "boundarySource-1": "geoBoundaries",
    "gjDownloadURL": "https://example.com/geojson.geojson",
}


class TestFetchGadm:
    @patch("pymarxan.spatial.gadm.requests.get")
    def test_fetch_returns_geodataframe(self, mock_get):
        # First call: API metadata → returns gjDownloadURL
        # Second call: GeoJSON download
        meta_resp = MagicMock()
        meta_resp.status_code = 200
        meta_resp.json.return_value = _SAMPLE_API_RESPONSE

        geojson_resp = MagicMock()
        geojson_resp.status_code = 200
        geojson_resp.json.return_value = _SAMPLE_GEOJSON

        mock_get.side_effect = [meta_resp, geojson_resp]

        gdf = fetch_gadm("TST", admin_level=0)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1
        assert gdf.geometry.iloc[0].is_valid

    @patch("pymarxan.spatial.gadm.requests.get")
    def test_fetch_with_admin_name_filters(self, mock_get):
        multi_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"shapeName": "RegionA"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"shapeName": "RegionB"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]],
                    },
                },
            ],
        }

        meta_resp = MagicMock()
        meta_resp.status_code = 200
        meta_resp.json.return_value = _SAMPLE_API_RESPONSE

        geojson_resp = MagicMock()
        geojson_resp.status_code = 200
        geojson_resp.json.return_value = multi_geojson

        mock_get.side_effect = [meta_resp, geojson_resp]

        gdf = fetch_gadm("TST", admin_level=1, admin_name="RegionA")
        assert len(gdf) == 1

    @patch("pymarxan.spatial.gadm.requests.get")
    def test_fetch_invalid_country_raises(self, mock_get):
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = resp

        with pytest.raises(Exception):
            fetch_gadm("ZZZ", admin_level=0)


class TestListCountries:
    def test_returns_list_of_dicts(self):
        countries = list_countries()
        assert isinstance(countries, list)
        assert len(countries) > 0
        assert "iso3" in countries[0]
        assert "name" in countries[0]

    def test_contains_common_countries(self):
        countries = list_countries()
        iso3s = {c["iso3"] for c in countries}
        assert "USA" in iso3s
        assert "GBR" in iso3s
        assert "AUS" in iso3s
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/spatial/test_gadm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan.spatial.gadm'`

**Step 3: Implement**

```python
# src/pymarxan/spatial/gadm.py
"""Fetch administrative boundaries from geoBoundaries API."""
from __future__ import annotations

import geopandas as gpd
import requests

_GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen"

# Common countries (ISO 3166-1 alpha-3). Full list at geoboundaries.org.
_COUNTRIES = [
    {"iso3": "AFG", "name": "Afghanistan"},
    {"iso3": "ALB", "name": "Albania"},
    {"iso3": "DZA", "name": "Algeria"},
    {"iso3": "AGO", "name": "Angola"},
    {"iso3": "ARG", "name": "Argentina"},
    {"iso3": "AUS", "name": "Australia"},
    {"iso3": "AUT", "name": "Austria"},
    {"iso3": "BGD", "name": "Bangladesh"},
    {"iso3": "BEL", "name": "Belgium"},
    {"iso3": "BOL", "name": "Bolivia"},
    {"iso3": "BRA", "name": "Brazil"},
    {"iso3": "KHM", "name": "Cambodia"},
    {"iso3": "CMR", "name": "Cameroon"},
    {"iso3": "CAN", "name": "Canada"},
    {"iso3": "CHL", "name": "Chile"},
    {"iso3": "CHN", "name": "China"},
    {"iso3": "COL", "name": "Colombia"},
    {"iso3": "COD", "name": "Congo (DRC)"},
    {"iso3": "CRI", "name": "Costa Rica"},
    {"iso3": "CUB", "name": "Cuba"},
    {"iso3": "DNK", "name": "Denmark"},
    {"iso3": "ECU", "name": "Ecuador"},
    {"iso3": "EGY", "name": "Egypt"},
    {"iso3": "ETH", "name": "Ethiopia"},
    {"iso3": "FIN", "name": "Finland"},
    {"iso3": "FRA", "name": "France"},
    {"iso3": "DEU", "name": "Germany"},
    {"iso3": "GHA", "name": "Ghana"},
    {"iso3": "GRC", "name": "Greece"},
    {"iso3": "GTM", "name": "Guatemala"},
    {"iso3": "HND", "name": "Honduras"},
    {"iso3": "IND", "name": "India"},
    {"iso3": "IDN", "name": "Indonesia"},
    {"iso3": "IRN", "name": "Iran"},
    {"iso3": "IRQ", "name": "Iraq"},
    {"iso3": "IRL", "name": "Ireland"},
    {"iso3": "ISR", "name": "Israel"},
    {"iso3": "ITA", "name": "Italy"},
    {"iso3": "JPN", "name": "Japan"},
    {"iso3": "KEN", "name": "Kenya"},
    {"iso3": "MDG", "name": "Madagascar"},
    {"iso3": "MYS", "name": "Malaysia"},
    {"iso3": "MEX", "name": "Mexico"},
    {"iso3": "MAR", "name": "Morocco"},
    {"iso3": "MOZ", "name": "Mozambique"},
    {"iso3": "MMR", "name": "Myanmar"},
    {"iso3": "NPL", "name": "Nepal"},
    {"iso3": "NLD", "name": "Netherlands"},
    {"iso3": "NZL", "name": "New Zealand"},
    {"iso3": "NGA", "name": "Nigeria"},
    {"iso3": "NOR", "name": "Norway"},
    {"iso3": "PAK", "name": "Pakistan"},
    {"iso3": "PAN", "name": "Panama"},
    {"iso3": "PER", "name": "Peru"},
    {"iso3": "PHL", "name": "Philippines"},
    {"iso3": "POL", "name": "Poland"},
    {"iso3": "PRT", "name": "Portugal"},
    {"iso3": "ROU", "name": "Romania"},
    {"iso3": "RUS", "name": "Russia"},
    {"iso3": "ZAF", "name": "South Africa"},
    {"iso3": "KOR", "name": "South Korea"},
    {"iso3": "ESP", "name": "Spain"},
    {"iso3": "LKA", "name": "Sri Lanka"},
    {"iso3": "SWE", "name": "Sweden"},
    {"iso3": "CHE", "name": "Switzerland"},
    {"iso3": "TZA", "name": "Tanzania"},
    {"iso3": "THA", "name": "Thailand"},
    {"iso3": "TUR", "name": "Turkey"},
    {"iso3": "UGA", "name": "Uganda"},
    {"iso3": "UKR", "name": "Ukraine"},
    {"iso3": "GBR", "name": "United Kingdom"},
    {"iso3": "USA", "name": "United States"},
    {"iso3": "VEN", "name": "Venezuela"},
    {"iso3": "VNM", "name": "Vietnam"},
    {"iso3": "ZMB", "name": "Zambia"},
    {"iso3": "ZWE", "name": "Zimbabwe"},
]


def list_countries() -> list[dict[str, str]]:
    """Return list of available countries with ISO3 codes."""
    return list(_COUNTRIES)


def fetch_gadm(
    country_iso3: str,
    admin_level: int = 0,
    admin_name: str | None = None,
) -> gpd.GeoDataFrame:
    """Fetch administrative boundary from geoBoundaries API.

    Parameters
    ----------
    country_iso3 : str
        ISO 3166-1 alpha-3 country code (e.g. "USA", "GBR").
    admin_level : int
        0 = country, 1 = state/province, 2 = district.
    admin_name : str or None
        Filter to a specific admin region by name.

    Returns
    -------
    gpd.GeoDataFrame
        Boundary polygon(s) with CRS EPSG:4326.
    """
    url = f"{_GEOBOUNDARIES_API}/{country_iso3}/ADM{admin_level}/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    meta = resp.json()
    geojson_url = meta["gjDownloadURL"]

    geojson_resp = requests.get(geojson_url, timeout=60)
    geojson_resp.raise_for_status()

    gdf = gpd.GeoDataFrame.from_features(
        geojson_resp.json()["features"],
        crs="EPSG:4326",
    )

    if admin_name is not None:
        name_col = "shapeName" if "shapeName" in gdf.columns else gdf.columns[0]
        gdf = gdf[gdf[name_col].str.contains(admin_name, case=False, na=False)]
        gdf = gdf.reset_index(drop=True)

    return gdf
```

Update `src/pymarxan/spatial/__init__.py`:

```python
"""Spatial data processing for conservation planning."""

from pymarxan.spatial.gadm import fetch_gadm, list_countries
from pymarxan.spatial.grid import compute_adjacency, generate_planning_grid

__all__ = [
    "compute_adjacency",
    "fetch_gadm",
    "generate_planning_grid",
    "list_countries",
]
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/spatial/test_gadm.py -v`
Expected: 5 PASS

**Step 5: Commit**

```bash
git add src/pymarxan/spatial/gadm.py src/pymarxan/spatial/__init__.py tests/pymarxan/spatial/test_gadm.py
git commit -m "feat(spatial): add GADM admin boundary fetching via geoBoundaries API"
```

---

### Task 2: Implement `wdpa.py` — fetch and apply protected areas

**Files:**
- Create: `src/pymarxan/spatial/wdpa.py`
- Create: `tests/pymarxan/spatial/test_wdpa.py`
- Modify: `src/pymarxan/spatial/__init__.py`

**Step 1: Write the failing tests**

```python
# tests/pymarxan/spatial/test_wdpa.py
"""Tests for WDPA protected area integration."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from pymarxan.models.problem import STATUS_LOCKED_IN, ConservationProblem
from pymarxan.spatial.wdpa import apply_wdpa_status, fetch_wdpa

_SAMPLE_WDPA_RESPONSE = {
    "protected_areas": [
        {
            "id": 1,
            "name": "Test Reserve",
            "designation": "National Park",
            "iucn_category": "II",
            "reported_area": 100.0,
            "geojson": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [0.6, 0], [0.6, 0.6], [0, 0.6], [0, 0]]],
            },
        }
    ]
}


def _make_geo_problem():
    """Create a problem with 4 PUs as a 2x2 grid."""
    gdf = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4],
            "cost": [1.0, 2.0, 3.0, 4.0],
            "status": [0, 0, 0, 0],
        },
        geometry=[
            box(0, 0, 0.5, 0.5),
            box(0.5, 0, 1.0, 0.5),
            box(0, 0.5, 0.5, 1.0),
            box(0.5, 0.5, 1.0, 1.0),
        ],
        crs="EPSG:4326",
    )
    return ConservationProblem(
        planning_units=gdf,
        features=pd.DataFrame({"id": [1], "name": ["f1"], "target": [10.0], "spf": [1.0]}),
        pu_vs_features=pd.DataFrame({"species": [1], "pu": [1], "amount": [20.0]}),
    )


class TestApplyWdpaStatus:
    def test_marks_overlapping_pus(self):
        problem = _make_geo_problem()
        # WDPA polygon covers PU 1 fully (box 0,0,0.5,0.5)
        wdpa = gpd.GeoDataFrame(
            {"name": ["Reserve"], "desig": ["NP"]},
            geometry=[box(0, 0, 0.6, 0.6)],
            crs="EPSG:4326",
        )
        result = apply_wdpa_status(problem, wdpa, overlap_threshold=0.5)
        # PU 1 should be locked in (>50% overlap)
        assert result.planning_units.loc[
            result.planning_units["id"] == 1, "status"
        ].iloc[0] == STATUS_LOCKED_IN

    def test_does_not_mutate_original(self):
        problem = _make_geo_problem()
        wdpa = gpd.GeoDataFrame(
            {"name": ["Reserve"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        result = apply_wdpa_status(problem, wdpa)
        assert all(problem.planning_units["status"] == 0)

    def test_high_threshold_excludes_partial_overlaps(self):
        problem = _make_geo_problem()
        # WDPA covers only a corner of PU 1
        wdpa = gpd.GeoDataFrame(
            {"name": ["Reserve"]},
            geometry=[box(0, 0, 0.2, 0.2)],
            crs="EPSG:4326",
        )
        result = apply_wdpa_status(problem, wdpa, overlap_threshold=0.5)
        assert result.planning_units.loc[
            result.planning_units["id"] == 1, "status"
        ].iloc[0] == 0  # Not locked — overlap < 50%

    def test_custom_status_value(self):
        from pymarxan.models.problem import STATUS_INITIAL_INCLUDE
        problem = _make_geo_problem()
        wdpa = gpd.GeoDataFrame(
            {"name": ["Reserve"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        result = apply_wdpa_status(problem, wdpa, status=STATUS_INITIAL_INCLUDE)
        assert all(result.planning_units["status"] == STATUS_INITIAL_INCLUDE)


class TestFetchWdpa:
    @patch("pymarxan.spatial.wdpa.requests.get")
    def test_fetch_returns_geodataframe(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _SAMPLE_WDPA_RESPONSE
        mock_get.return_value = resp

        gdf = fetch_wdpa(bounds=(0, 0, 1, 1), api_token="test-token")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) >= 1
        assert "name" in gdf.columns

    @patch("pymarxan.spatial.wdpa.requests.get")
    def test_fetch_without_token_raises(self, mock_get):
        resp = MagicMock()
        resp.status_code = 401
        resp.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_get.return_value = resp

        with pytest.raises(Exception):
            fetch_wdpa(bounds=(0, 0, 1, 1))
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/spatial/test_wdpa.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement**

```python
# src/pymarxan/spatial/wdpa.py
"""WDPA protected area integration."""
from __future__ import annotations

import copy

import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import shape

from pymarxan.models.problem import STATUS_LOCKED_IN, ConservationProblem

_WDPA_API = "https://api.protectedplanet.net/v3"


def fetch_wdpa(
    bounds: tuple[float, float, float, float],
    country_iso3: str | None = None,
    api_token: str | None = None,
) -> gpd.GeoDataFrame:
    """Fetch protected areas from Protected Planet API.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) bounding box.
    country_iso3 : str or None
        Optional country filter.
    api_token : str or None
        Protected Planet API token. Required for authenticated access.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: name, desig, iucn_cat, geometry.
    """
    params: dict = {}
    if api_token:
        params["token"] = api_token
    if country_iso3:
        params["country"] = country_iso3

    # Use bbox filter
    minx, miny, maxx, maxy = bounds
    params["with_geometry"] = "true"
    params["per_page"] = 50

    url = f"{_WDPA_API}/protected_areas/search"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    pa_list = data.get("protected_areas", [])

    rows = []
    geometries = []
    for pa in pa_list:
        geojson = pa.get("geojson")
        if geojson is None:
            continue
        geom = shape(geojson)
        if geom.is_empty:
            continue
        # Filter by bounds
        if not geom.intersects(
            __import__("shapely.geometry", fromlist=["box"]).box(*bounds)
        ):
            continue
        rows.append({
            "name": pa.get("name", ""),
            "desig": pa.get("designation", ""),
            "iucn_cat": pa.get("iucn_category", ""),
        })
        geometries.append(geom)

    if not rows:
        return gpd.GeoDataFrame(
            {"name": [], "desig": [], "iucn_cat": [], "geometry": []},
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")


def apply_wdpa_status(
    problem: ConservationProblem,
    wdpa: gpd.GeoDataFrame,
    overlap_threshold: float = 0.5,
    status: int = STATUS_LOCKED_IN,
) -> ConservationProblem:
    """Set PU status for units overlapping protected areas.

    Parameters
    ----------
    problem : ConservationProblem
        Must have GeoDataFrame planning_units.
    wdpa : gpd.GeoDataFrame
        Protected area polygons.
    overlap_threshold : float
        Minimum fraction of PU area that must be covered.
    status : int
        Status to assign (default STATUS_LOCKED_IN=2).

    Returns
    -------
    ConservationProblem
        New problem with updated statuses (does not mutate input).
    """
    result = copy.deepcopy(problem)
    pu_gdf = result.planning_units

    # Dissolve all WDPA polygons into one
    wdpa_union = wdpa.geometry.union_all()

    new_statuses = pu_gdf["status"].values.copy()
    for idx in range(len(pu_gdf)):
        pu_geom = pu_gdf.geometry.iloc[idx]
        pu_area = pu_geom.area
        if pu_area <= 0:
            continue
        intersection = pu_geom.intersection(wdpa_union)
        overlap_ratio = intersection.area / pu_area
        if overlap_ratio >= overlap_threshold:
            new_statuses[idx] = status

    result.planning_units["status"] = new_statuses
    return result
```

Update `src/pymarxan/spatial/__init__.py` to add exports:
```python
from pymarxan.spatial.wdpa import apply_wdpa_status, fetch_wdpa
```

And add to `__all__`:
```python
__all__ = [
    "apply_wdpa_status",
    "compute_adjacency",
    "fetch_gadm",
    "fetch_wdpa",
    "generate_planning_grid",
    "list_countries",
]
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/spatial/test_wdpa.py -v`
Expected: 6 PASS

**Step 5: Commit**

```bash
git add src/pymarxan/spatial/wdpa.py src/pymarxan/spatial/__init__.py tests/pymarxan/spatial/test_wdpa.py
git commit -m "feat(spatial): add WDPA protected area fetch and status overlay"
```

---

### Task 3: Create `gadm_picker` Shiny module

**Files:**
- Create: `src/pymarxan_shiny/modules/spatial/gadm_picker.py`
- Create: `tests/pymarxan_shiny/test_gadm_picker.py`

**Step 1: Write the failing test**

```python
# tests/pymarxan_shiny/test_gadm_picker.py
"""Tests for GADM picker Shiny module."""
from pymarxan_shiny.modules.spatial.gadm_picker import gadm_picker_server, gadm_picker_ui


def test_gadm_picker_ui_callable():
    assert callable(gadm_picker_ui)


def test_gadm_picker_server_callable():
    assert callable(gadm_picker_server)
```

**Step 2: Run to verify failure**

Run: `pytest tests/pymarxan_shiny/test_gadm_picker.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement**

```python
# src/pymarxan_shiny/modules/spatial/gadm_picker.py
"""GADM boundary picker Shiny module."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.spatial.gadm import fetch_gadm, list_countries

try:
    from shinywidgets import output_widget, render_widget
    from pymarxan_shiny.modules.mapping.map_utils import create_geo_map
    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


@module.ui
def gadm_picker_ui():
    countries = list_countries()
    choices = {c["iso3"]: f"{c['name']} ({c['iso3']})" for c in countries}
    map_output = output_widget("boundary_map") if _HAS_IPYLEAFLET else ui.output_ui("boundary_text")
    return ui.card(
        ui.card_header("Administrative Boundaries (GADM)"),
        ui.layout_columns(
            ui.input_selectize("country", "Country", choices=choices),
            ui.input_select("admin_level", "Admin Level", {
                "0": "Country (ADM0)",
                "1": "State/Province (ADM1)",
                "2": "District (ADM2)",
            }),
            col_widths=[6, 6],
        ),
        ui.input_text("admin_name", "Region Name Filter (optional)", value=""),
        ui.input_action_button("fetch", "Fetch Boundary", class_="btn-primary"),
        map_output,
        ui.output_text_verbatim("boundary_info"),
    )


@module.server
def gadm_picker_server(input, output, session, boundary: reactive.Value):

    @reactive.effect
    @reactive.event(input.fetch)
    def _fetch():
        try:
            admin_name = input.admin_name() or None
            gdf = fetch_gadm(
                country_iso3=input.country(),
                admin_level=int(input.admin_level()),
                admin_name=admin_name,
            )
            boundary.set(gdf)
            ui.notification_show(
                f"Fetched {len(gdf)} boundary polygon(s).", type="message",
            )
        except Exception as e:
            ui.notification_show(f"Error fetching boundary: {e}", type="error")

    @render.text
    def boundary_info():
        gdf = boundary()
        if gdf is None:
            return "Select a country and fetch boundaries."
        return f"{len(gdf)} polygon(s) fetched"

    if _HAS_IPYLEAFLET:
        @render_widget
        def boundary_map():
            gdf = boundary()
            if gdf is None:
                return None
            colors = ["#e74c3c"] * len(gdf)
            return create_geo_map(gdf, colors)

    if not _HAS_IPYLEAFLET:
        @render.ui
        def boundary_text():
            gdf = boundary()
            if gdf is None:
                return ui.p("Fetch a boundary to see the preview.")
            return ui.p(f"Boundary: {len(gdf)} polygons")
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan_shiny/test_gadm_picker.py -v`
Expected: 2 PASS

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/spatial/gadm_picker.py tests/pymarxan_shiny/test_gadm_picker.py
git commit -m "feat(shiny): add GADM boundary picker module"
```

---

### Task 4: Create `wdpa_overlay` Shiny module

**Files:**
- Create: `src/pymarxan_shiny/modules/spatial/wdpa_overlay.py`
- Create: `tests/pymarxan_shiny/test_wdpa_overlay.py`

**Step 1: Write the failing test**

```python
# tests/pymarxan_shiny/test_wdpa_overlay.py
"""Tests for WDPA overlay Shiny module."""
from pymarxan_shiny.modules.spatial.wdpa_overlay import wdpa_overlay_server, wdpa_overlay_ui


def test_wdpa_overlay_ui_callable():
    assert callable(wdpa_overlay_ui)


def test_wdpa_overlay_server_callable():
    assert callable(wdpa_overlay_server)
```

**Step 2: Run to verify failure**

Run: `pytest tests/pymarxan_shiny/test_wdpa_overlay.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement**

```python
# src/pymarxan_shiny/modules/spatial/wdpa_overlay.py
"""WDPA protected area overlay Shiny module."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.problem import STATUS_INITIAL_INCLUDE, STATUS_LOCKED_IN, has_geometry
from pymarxan.spatial.wdpa import apply_wdpa_status, fetch_wdpa


@module.ui
def wdpa_overlay_ui():
    return ui.card(
        ui.card_header("Protected Areas (WDPA)"),
        ui.input_text("api_token", "API Token (optional)", value=""),
        ui.layout_columns(
            ui.input_slider("threshold", "Overlap Threshold", min=0.1, max=1.0, value=0.5, step=0.1),
            ui.input_select("status", "Set Status", {
                str(STATUS_LOCKED_IN): "Locked In (2)",
                str(STATUS_INITIAL_INCLUDE): "Initial Include (1)",
            }),
            col_widths=[6, 6],
        ),
        ui.input_action_button("fetch_wdpa", "Fetch & Apply Protected Areas", class_="btn-primary"),
        ui.output_text_verbatim("wdpa_info"),
    )


@module.server
def wdpa_overlay_server(input, output, session, problem: reactive.Value):

    @reactive.effect
    @reactive.event(input.fetch_wdpa)
    def _fetch_and_apply():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first.", type="warning")
            return
        if not has_geometry(p):
            ui.notification_show(
                "Planning units need geometry. Generate a grid first.",
                type="warning",
            )
            return

        try:
            bounds = tuple(p.planning_units.total_bounds)
            token = input.api_token() or None
            wdpa = fetch_wdpa(bounds=bounds, api_token=token)
            if len(wdpa) == 0:
                ui.notification_show("No protected areas found in this region.", type="warning")
                return

            status = int(input.status())
            result = apply_wdpa_status(
                p, wdpa,
                overlap_threshold=input.threshold(),
                status=status,
            )
            n_marked = int((result.planning_units["status"] != p.planning_units["status"]).sum())
            problem.set(result)
            ui.notification_show(f"Marked {n_marked} PUs as protected.", type="message")
        except Exception as e:
            ui.notification_show(f"WDPA error: {e}", type="error")

    @render.text
    def wdpa_info():
        p = problem()
        if p is None:
            return "Load a project to use WDPA overlay."
        n_locked = int((p.planning_units["status"] == STATUS_LOCKED_IN).sum())
        return f"Currently {n_locked} PUs locked in"
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan_shiny/test_wdpa_overlay.py -v`
Expected: 2 PASS

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/spatial/wdpa_overlay.py tests/pymarxan_shiny/test_wdpa_overlay.py
git commit -m "feat(shiny): add WDPA protected area overlay module"
```

---

### Task 5: Integrate new modules into app.py + regression

**Files:**
- Modify: `src/pymarxan_app/app.py`

**Step 1: Add imports and wire up**

Add imports for both new modules:
```python
from pymarxan_shiny.modules.spatial.gadm_picker import gadm_picker_server, gadm_picker_ui
from pymarxan_shiny.modules.spatial.wdpa_overlay import wdpa_overlay_server, wdpa_overlay_ui
```

Add new reactive value for GADM boundary:
```python
gadm_boundary: reactive.Value = reactive.value(None)
```

Add to Data tab:
```python
gadm_picker_ui("gadm"),
wdpa_overlay_ui("wdpa"),
```

Add server calls:
```python
gadm_picker_server("gadm", boundary=gadm_boundary)
wdpa_overlay_server("wdpa", problem=problem)
```

**Step 2: Run full regression**

Run: `pytest tests/ -v --cov --cov-report=term-missing --cov-fail-under=75`
Expected: All tests pass, coverage >= 75%

Run: `ruff check src/ tests/`
Expected: Clean

**Step 3: Commit**

```bash
git add src/pymarxan_app/app.py
git commit -m "feat(app): integrate GADM picker and WDPA overlay into Data tab"
```

---

## Summary

| Task | Description | New Tests |
|------|-------------|-----------|
| 1 | GADM `fetch_gadm` + `list_countries` | 5 |
| 2 | WDPA `fetch_wdpa` + `apply_wdpa_status` | 6 |
| 3 | GADM picker Shiny module | 2 |
| 4 | WDPA overlay Shiny module | 2 |
| 5 | App integration + regression | 0 |
| **Total** | | **~15** |
