# MaPP Feature Porting — Design Document

**Date:** 2026-02-23
**Scope:** Port 7 high-value features from the Marxan Planning Platform (MaPP) cloud application to pymarxan desktop
**Architecture:** New `pymarxan.spatial` subpackage + enhancements to existing models/analysis layers

## Context

The Marxan Planning Platform ([marxan-cloud](https://github.com/Vizzuality/marxan-cloud)) is a cloud-hosted NestJS/React application with 35 API modules and 19 geoprocessing modules. It wraps the C++ Marxan binary and adds GIS data management, collaboration, and cloud infrastructure.

pymarxan already exceeds MaPP in solver diversity (7 native solvers vs 1 binary wrapper), calibration (BLM, SPF, sweep, sensitivity), and analysis (irreplaceability, connectivity metrics). However, MaPP provides significant GIS capabilities that pymarxan lacks: spatial grid generation, admin boundary selection, protected area integration, and GIS format import.

This design ports the 7 most valuable MaPP features while preserving pymarxan's architecture.

### Key Decisions

1. **Data source:** Web APIs for GADM (geoBoundaries API) and WDPA (Protected Planet API). No bundled datasets — keeps package small, requires internet for those features only.
2. **Geometry model:** Upgrade `planning_units` from `pd.DataFrame` to `gpd.GeoDataFrame`. All existing code works unchanged since GeoDataFrame inherits DataFrame. Maps use real shapes instead of synthetic grids.
3. **Package structure:** New `pymarxan.spatial` subpackage rather than extending existing modules. Clean separation of GIS concerns.

---

## Feature 1: Planning Unit Grid Generation

**Module:** `src/pymarxan/spatial/grid.py`

### API

```python
def generate_planning_grid(
    bounds: tuple[float, float, float, float],  # (minx, miny, maxx, maxy)
    cell_size: float,
    grid_type: str = "square",  # "square" | "hexagonal"
    crs: str = "EPSG:4326",
    clip_to: BaseGeometry | None = None,  # Optional clipping polygon
) -> gpd.GeoDataFrame:
    """Generate a planning unit grid as a GeoDataFrame.

    Returns GeoDataFrame with columns: id (int), cost (float, default 1.0),
    status (int, default 0), geometry (Polygon).
    """

def compute_adjacency(
    planning_units: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Compute boundary DataFrame from shared edges between adjacent PUs.

    Returns DataFrame with columns: id1, id2, boundary (shared edge length).
    """
```

### Implementation

- **Square grids:** Tile bounding box with `shapely.box()` cells. Row-major ID assignment.
- **Hexagonal grids:** Flat-top hex tiling using axial coordinates. Convert to Shapely `Polygon` via 6-vertex formula. Offset even/odd rows by half cell width.
- **Clipping:** If `clip_to` is provided, intersect each cell with the polygon. Discard cells with zero intersection area. Retain cells whose centroid falls within the polygon.
- **Adjacency:** Use `geopandas.sjoin` with a small buffer to find touching pairs. Shared boundary length computed via `intersection().length`.

### Shiny Module

`src/pymarxan_shiny/modules/spatial/grid_builder.py`

- Inputs: bounding box (4 numeric fields or draw on map), cell size slider, grid type dropdown (square/hex)
- "Generate Grid" button → creates GeoDataFrame → sets `problem()` reactive value
- Preview: ipyleaflet map showing generated cells
- Status bar: "Generated N planning units"

---

## Feature 2: WDPA Protected Area Integration

**Module:** `src/pymarxan/spatial/wdpa.py`

### API

```python
def fetch_wdpa(
    bounds: tuple[float, float, float, float],
    country_iso3: str | None = None,
    api_token: str | None = None,
) -> gpd.GeoDataFrame:
    """Fetch protected areas from Protected Planet API within bounds.

    Returns GeoDataFrame with columns: name, desig (designation),
    iucn_cat (IUCN category), status_yr, geometry.
    Uses WDPA API v3: https://api.protectedplanet.net/v3/
    """

def apply_wdpa_status(
    problem: ConservationProblem,
    wdpa: gpd.GeoDataFrame,
    overlap_threshold: float = 0.5,
    status: int = STATUS_LOCKED_IN,
) -> ConservationProblem:
    """Set PU status for units overlapping protected areas.

    A PU is marked if the fraction of its area covered by any WDPA
    polygon >= overlap_threshold.
    Returns a new ConservationProblem (does not mutate input).
    """
```

### Implementation

- **API:** GET `https://api.protectedplanet.net/v3/protected_areas/search` with bbox and optional country filter. Paginate results. Parse GeoJSON response into GeoDataFrame.
- **Spatial join:** For each PU, compute `intersection_area / pu_area`. If ratio >= threshold, set status.
- **No API key fallback:** If no token provided, use the public endpoint (rate-limited). Document how to get a free token.

### Shiny Module

`src/pymarxan_shiny/modules/spatial/wdpa_overlay.py`

- API token input (optional, stored in session)
- "Fetch Protected Areas" button (requires problem with geometry)
- Overlap threshold slider (default 0.5)
- Status selector: locked-in (2) or initial-include (1)
- Results: "N PUs marked as protected" notification
- Map overlay: WDPA polygons shown in semi-transparent green

---

## Feature 3: GADM Administrative Boundaries

**Module:** `src/pymarxan/spatial/gadm.py`

### API

```python
def fetch_gadm(
    country_iso3: str,
    admin_level: int = 0,
    admin_name: str | None = None,
) -> gpd.GeoDataFrame:
    """Fetch administrative boundary from geoBoundaries API.

    Uses https://www.geoboundaries.org/api/current/gbOpen/{ISO3}/{ADM}/
    Returns GeoDataFrame with boundary polygon(s).
    """

def list_countries() -> list[dict[str, str]]:
    """Return list of available countries with ISO3 codes.

    Returns list of {iso3: str, name: str} dicts.
    """
```

### Implementation

- **API:** geoBoundaries API is open, no key required. Returns GeoJSON directly.
- **Admin levels:** ADM0 (country), ADM1 (state/province), ADM2 (district). Each level returns progressively finer boundaries.
- **Filtering:** If `admin_name` provided, filter the returned GeoDataFrame by name column.
- **Integration with grid generation:** The returned polygon is used as `clip_to` parameter in `generate_planning_grid()`.

### Shiny Module

`src/pymarxan_shiny/modules/spatial/gadm_picker.py`

- Country dropdown (searchable, populated from `list_countries()`)
- Admin level selector (0/1/2)
- Admin name filter (text input, optional)
- "Fetch Boundary" button → fetches polygon → shows on map
- "Generate Grid from Boundary" button → chains to grid_builder with `clip_to=polygon`

---

## Feature 4: Shapefile/GeoJSON Import

**Module:** `src/pymarxan/spatial/importers.py`

### API

```python
def import_planning_units(
    path: str | Path,
    id_column: str = "id",
    cost_column: str = "cost",
    status_column: str | None = "status",
) -> gpd.GeoDataFrame:
    """Import planning units from shapefile, GeoJSON, or GeoPackage.

    Auto-detects format via geopandas.read_file().
    Renames columns to standard names (id, cost, status, geometry).
    If status_column is None or missing, defaults all PUs to status=0.
    If cost_column is missing, defaults all costs to 1.0.
    """

def import_features_from_vector(
    path: str | Path,
    planning_units: gpd.GeoDataFrame,
    feature_name: str,
    feature_id: int,
    amount_column: str | None = None,
) -> pd.DataFrame:
    """Compute feature amounts per PU via spatial overlay.

    If amount_column is provided, sums that column within each PU.
    If None, uses area of intersection as the amount.
    Returns DataFrame with columns: species, pu, amount.
    """

def import_features_from_raster(
    raster_path: str | Path,
    planning_units: gpd.GeoDataFrame,
    feature_name: str,
    feature_id: int,
    stat: str = "sum",
) -> pd.DataFrame:
    """Compute zonal statistics from raster for each PU.

    Uses rasterio for raster I/O and numpy for statistics.
    stat options: "sum", "mean", "max", "min", "count".
    Returns DataFrame with columns: species, pu, amount.
    """
```

### Implementation

- **Vector import:** `geopandas.read_file()` handles .shp, .geojson, .gpkg, .kml automatically. Column mapping validates that specified columns exist, raises clear errors otherwise.
- **Vector overlay:** `geopandas.overlay(planning_units, features, how="intersection")` then group by PU ID and sum.
- **Raster import:** `rasterio.open()` reads raster. For each PU, `rasterio.mask.mask()` extracts pixels within polygon. Compute stat over masked values. Skip PUs with no data.
- **CRS alignment:** Auto-reproject feature data to match PU CRS if they differ.

### New Dependency

```toml
[project.optional-dependencies]
spatial = ["rasterio>=1.3", "requests>=2.28"]
```

`rasterio` is only needed for raster import. Vector import works with geopandas alone (already a core dep).

### Shiny Module

`src/pymarxan_shiny/modules/spatial/import_wizard.py`

- Tab 1: "Import Planning Units" — file upload, column mapping dropdowns, preview
- Tab 2: "Import Features" — file upload (vector or raster), feature name/ID inputs, preview of amounts
- Both tabs show preview table and map visualization

---

## Feature 5: Custom Cost Surfaces

**Module:** `src/pymarxan/spatial/cost_surface.py`

### API

```python
def apply_cost_from_raster(
    planning_units: gpd.GeoDataFrame,
    raster_path: str | Path,
    aggregation: str = "mean",
) -> gpd.GeoDataFrame:
    """Compute zonal statistics from cost raster, update PU cost column.

    aggregation: "mean" | "sum" | "max" | "min"
    Returns new GeoDataFrame with updated cost column (does not mutate input).
    """

def apply_cost_from_vector(
    planning_units: gpd.GeoDataFrame,
    cost_layer: gpd.GeoDataFrame,
    cost_column: str,
    aggregation: str = "area_weighted_mean",
) -> gpd.GeoDataFrame:
    """Compute cost from vector overlay.

    aggregation: "area_weighted_mean" | "sum" | "max"
    Returns new GeoDataFrame with updated cost column.
    """

def combine_cost_layers(
    planning_units: gpd.GeoDataFrame,
    layers: list[tuple[str, np.ndarray]],
    weights: list[float] | None = None,
) -> gpd.GeoDataFrame:
    """Combine multiple cost arrays with optional weighting.

    Each layer is (name, cost_array) where cost_array is per-PU.
    Layers are min-max normalized before combining.
    If weights is None, equal weighting.
    Returns new GeoDataFrame with updated cost column.
    """
```

### Implementation

- **Raster cost:** Reuses the same zonal statistics pattern as `import_features_from_raster` but writes to the `cost` column instead of creating puvspr rows.
- **Vector cost:** Spatial overlay + area-weighted aggregation. For each PU, sum `(intersection_area / pu_area) * cost_value` across overlapping cost polygons.
- **Multi-layer:** Normalize each layer to [0, 1], apply weights, sum. Store individual layer values as extra columns for transparency.

### Shiny Module

`src/pymarxan_shiny/modules/spatial/cost_upload.py`

- File upload (raster or vector)
- Aggregation method dropdown
- Multi-layer support: add multiple cost files, set weights
- Preview: histogram of cost distribution + map colored by cost
- "Apply Costs" button → updates problem

---

## Feature 6: Per-Scenario Feature Overrides

**Changes to:** `src/pymarxan/analysis/scenarios.py`, `src/pymarxan/models/problem.py`

### API

```python
# New utility function in models/problem.py
def apply_feature_overrides(
    problem: ConservationProblem,
    overrides: dict[int, dict[str, float]],
) -> ConservationProblem:
    """Return a copy of problem with feature targets/SPF overridden.

    overrides maps feature_id -> {field_name: new_value}.
    Valid fields: "target", "spf", "prop".
    Does NOT mutate the original problem.

    Example: {1: {"target": 500}, 3: {"spf": 2.0}}
    """

# Enhanced Scenario dataclass in analysis/scenarios.py
@dataclass
class Scenario:
    name: str
    solution: Solution
    parameters: dict
    feature_overrides: dict[int, dict[str, float]] | None = None

# New method on ScenarioSet
class ScenarioSet:
    def run_with_overrides(
        self,
        name: str,
        problem: ConservationProblem,
        solver: Solver,
        overrides: dict[int, dict[str, float]],
        parameter_overrides: dict | None = None,
        config: SolverConfig | None = None,
    ) -> Scenario:
        """Create scenario by solving with feature overrides applied.

        1. apply_feature_overrides(problem, overrides)
        2. Apply parameter_overrides to copy
        3. Solve
        4. Store scenario with overrides metadata
        """
```

### Implementation

- `apply_feature_overrides` deep-copies the problem, then modifies `features` DataFrame rows matching the override feature IDs.
- The Scenario dataclass gains an optional `feature_overrides` field for provenance tracking.
- `ScenarioSet.compare()` gains a column showing which features were modified in each scenario.
- Backward compatible: existing code that creates Scenarios without overrides is unchanged.

### Shiny Module

Enhancement to existing `results/scenario_compare.py`:

- "New Scenario with Overrides" button
- Feature override editor: table showing current features, editable target/SPF columns
- "Run Scenario" button → solves with overrides → adds to ScenarioSet
- Comparison table shows override indicators

---

## Feature 7: Project Cloning

**Changes to:** `src/pymarxan/models/problem.py`, `src/pymarxan/analysis/scenarios.py`

### API

```python
# New method on ConservationProblem
class ConservationProblem:
    def clone(self) -> ConservationProblem:
        """Deep copy all DataFrames, parameters, and geometry.

        Returns an independent copy that can be modified without
        affecting the original.
        """

# New method on ScenarioSet
class ScenarioSet:
    def clone_scenario(
        self,
        source_name: str,
        new_name: str,
        parameter_overrides: dict | None = None,
        feature_overrides: dict[int, dict[str, float]] | None = None,
    ) -> Scenario:
        """Clone an existing scenario with optional modifications.

        Deep copies the source scenario's solution and parameters.
        Applies any overrides to the copy.
        """
```

### Implementation

- `ConservationProblem.clone()` uses `copy.deepcopy` on all DataFrames and the parameters dict. For GeoDataFrames, this preserves geometry and CRS.
- `ScenarioSet.clone_scenario()` copies the source's Solution (including selected array), parameters, and feature_overrides. Applies new overrides on top.
- Cloned objects are fully independent — no shared mutable state.

### Shiny Module

- **Data tab:** "Clone Project" button → creates deep copy, opens in new state
- **Results tab:** "Clone Scenario" button on each scenario row → opens override editor pre-filled with source values

---

## GeoDataFrame Upgrade Path

### Changes to ConservationProblem

The `planning_units` field type annotation changes from `pd.DataFrame` to `pd.DataFrame` (no annotation change needed — GeoDataFrame IS a DataFrame). The actual runtime type may be either.

### Geometry Detection Pattern

```python
def has_geometry(problem: ConservationProblem) -> bool:
    """Check if planning_units has real spatial geometry."""
    return (
        isinstance(problem.planning_units, gpd.GeoDataFrame)
        and "geometry" in problem.planning_units.columns
        and not problem.planning_units.geometry.is_empty.all()
    )
```

### Map Module Upgrade

All 5 map modules already have the `_HAS_IPYLEAFLET` pattern. The upgrade:

1. Check `has_geometry(problem)` — if True, render real polygons from geometry column
2. If False, fall back to current `generate_grid()` synthetic rectangles
3. No breaking change: existing projects without geometry continue to work

### Solver Compatibility

Solvers only read `id`, `cost`, `status` columns from `planning_units`. The `geometry` column is ignored. No solver changes needed.

---

## New Package Structure

```
src/pymarxan/spatial/
    __init__.py          # Exports all public functions
    grid.py              # Feature 1: PU grid generation
    gadm.py              # Feature 3: GADM admin boundaries
    wdpa.py              # Feature 2: WDPA protected area integration
    importers.py         # Feature 4: Shapefile/GeoJSON/raster import
    cost_surface.py      # Feature 5: Cost surface processing

src/pymarxan_shiny/modules/spatial/
    __init__.py
    grid_builder.py      # Grid generation UI
    gadm_picker.py       # Country/region selector
    wdpa_overlay.py      # Protected area fetcher
    import_wizard.py     # GIS file import wizard
    cost_upload.py       # Cost surface upload
```

## Dependencies

```toml
[project.optional-dependencies]
spatial = ["rasterio>=1.3", "requests>=2.28"]
# geopandas and shapely already in core deps
# requests used for GADM and WDPA API calls
```

## Testing Strategy

- Each spatial module gets its own test file with synthetic GeoDataFrames (small grids, mock polygons)
- API-calling functions (GADM, WDPA) use `unittest.mock.patch` and `responses` library — no real network calls in CI
- Small test shapefiles in `tests/data/spatial/` (< 100 KB)
- Raster tests use in-memory `rasterio.MemoryFile`
- Target: maintain >= 75% coverage
- New marker: `@pytest.mark.spatial` for tests requiring geopandas

## Implementation Phases

| Phase | Features | New Files | Tests |
|-------|----------|-----------|-------|
| 14 | Grid generation (#1) + GeoDataFrame upgrade + adjacency | grid.py, grid_builder.py, map upgrades | ~15 |
| 15 | GADM (#3) + WDPA (#2) + API integration | gadm.py, wdpa.py, gadm_picker.py, wdpa_overlay.py | ~15 |
| 16 | Shapefile/GeoJSON import (#4) + cost surfaces (#5) | importers.py, cost_surface.py, import_wizard.py, cost_upload.py | ~20 |
| 17 | Scenario overrides (#6) + cloning (#7) + integration | scenarios.py changes, problem.py changes | ~12 |

## Out of Scope

- Multi-user authentication and collaboration (cloud-only concern)
- Async job queues / Kubernetes orchestration (cloud infrastructure)
- Server-side tile rendering (we use ipyleaflet client-side)
- PDF/PNG report generation (future phase)
- Legacy Marxan project import wizard (we already read the format natively)
