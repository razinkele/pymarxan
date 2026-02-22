# Phase 9: Remaining Shiny Modules — Complete UI Vision

**Date:** 2026-02-22
**Status:** Design Document
**Goal:** Build all 7 remaining Shiny modules to complete the design doc's UI vision: ipyleaflet maps with synthetic geometry, editable feature table, sensitivity dashboard, and network view.

---

## 1. The 7 Modules

| # | Module | Category | Purpose |
|---|--------|----------|---------|
| 1 | `feature_table` | data | Editable feature targets/SPF with persistence back to problem |
| 2 | `spatial_grid` | mapping | PU map colored by cost/status using ipyleaflet |
| 3 | `frequency_map` | mapping | Selection frequency heatmap across solutions |
| 4 | `comparison_map` | mapping | Side-by-side two-solution comparison |
| 5 | `network_view` | mapping | Connectivity graph overlay on PU grid |
| 6 | `sensitivity` | calibration | Parameter sensitivity dashboard with heatmap |
| 7 | `solution_map` (upgrade) | mapping | Upgrade existing table-based module to ipyleaflet |

### Module Categories in `src/pymarxan_shiny/modules/`

- `mapping/` — spatial_grid, frequency_map, comparison_map, network_view, solution_map (upgrade)
- `data/` — feature_table (new subpackage)
- `calibration/` — sensitivity (new Shiny module, wraps existing backend)

## 2. Synthetic Geometry Generator

All ipyleaflet map modules need PU geometries. Since Marxan datasets don't include spatial data, we generate synthetic grid bounding boxes.

### `src/pymarxan/models/geometry.py`

```python
def generate_grid(n_pu: int, origin: tuple[float, float] = (0.0, 0.0),
                  cell_size: float = 0.01) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Generate grid bounding boxes for n planning units.

    Returns list of (sw_corner, ne_corner) tuples.
    Layout: sqrt(n) columns, ceil rows, left-to-right then bottom-to-top.
    """
```

- Returns `list[tuple[tuple[float, float], tuple[float, float]]]` — each element is `((south, west), (north, east))` suitable for `ipyleaflet.Rectangle(bounds=...)`
- Grid layout: `cols = ceil(sqrt(n_pu))`, `rows = ceil(n_pu / cols)`
- Default origin `(0, 0)` with `cell_size=0.01` degrees (~1km cells)
- Pure function, no side effects, easy to test

## 3. ipyleaflet Map Pattern

All 5 map modules share a common pattern:

```python
from ipyleaflet import Map, Rectangle, basemaps
from shinywidgets import output_widget, render_widget

def module_ui(id: str) -> ui.TagChild:
    ns = ui.NS(id)
    return ui.card(
        ui.card_header("Title"),
        output_widget(ns("map")),
        ui.sidebar(...)  # module-specific controls
    )

def module_server(id: str, ...reactive_inputs...):
    @module
    def server(input, output, session):
        @render_widget
        def map():
            m = Map(center=(0, 0), zoom=12, basemap=basemaps.CartoDB.Positron)
            grid = generate_grid(n_pu)
            for i, bounds in enumerate(grid):
                color = color_fn(values[i])  # module-specific coloring
                rect = Rectangle(bounds=bounds, color=color, fill_color=color, fill_opacity=0.6)
                m.add(rect)
            return m
    return server
```

### Color Schemes Per Module

| Module | Color mapping |
|---|---|
| spatial_grid | Cost → yellow-to-red gradient; Status → categorical (green=locked-in, red=locked-out, gray=available) |
| solution_map | Binary — green=selected, gray=not-selected |
| frequency_map | 0-100% → white-to-blue gradient |
| comparison_map | Green=both, blue=A-only, orange=B-only, gray=neither |
| network_view | Metric value → yellow-to-purple gradient |

### Dependencies

- `ipyleaflet` — map widget
- `shinywidgets` — bridge ipyleaflet into Shiny via `output_widget`/`render_widget`

## 4. Feature Table Editor

### `src/pymarxan_shiny/modules/data/feature_table.py`

**UI:** Shiny `render.data_frame` with `editable=True` displaying features DataFrame (id, name, target, spf).

**Server:**
```python
@render.data_frame
def feature_grid():
    df = problem().features[["id", "name", "target", "spf"]].copy()
    return render.DataGrid(df, editable=True)

@feature_grid.set_patch_fn
def _(*, patch):
    col = patch["column_id"]
    if col in ("target", "spf"):
        return float(patch["value"])
    return patch["value"]
```

**Persistence:** When user edits target or SPF, `set_patch_fn` validates and converts value. A "Save Changes" button writes edits back to `problem().features` DataFrame, making changes available to all downstream modules (solvers, calibration, etc.).

**Read-only columns:** `id` and `name` are not editable (enforced by `set_patch_fn` ignoring changes to those columns).

## 5. Sensitivity Dashboard

### `src/pymarxan_shiny/modules/calibration/sensitivity.py` (Shiny module)

Wraps the existing `src/pymarxan/calibration/sensitivity.py` backend.

**UI Layout:**
- Sidebar: multi-select checkboxes for parameters to vary (BLM, SPF, targets), multiplier range slider (default 0.8-1.2, step 0.1), "Run Sensitivity" button
- Main panel: plotly heatmap showing objective value across parameter combinations, summary table with best/worst configurations

**Server Logic:**
1. On button click, build `SensitivityConfig` from UI inputs
2. Run `run_sensitivity()` in background thread (reuses run_panel threading pattern from Phase 8)
3. Convert `SensitivityResult.to_dataframe()` to plotly heatmap
4. Store results in reactive value for cross-module access

## 6. Network View

### `src/pymarxan_shiny/modules/mapping/network_view.py`

Visualizes connectivity graph using ipyleaflet + synthetic geometry.

**UI Layout:**
- ipyleaflet map with PU rectangles colored by connectivity metric
- Polyline overlays for edges between connected PUs
- Sidebar: dropdown to select metric (degree, betweenness, etc.), threshold slider to filter weak connections

**Server Logic:**
1. Read `problem.boundary` or `problem.connectivity` to build edge list
2. Use `connectivity/metrics.py` to compute per-PU metrics
3. Render PU rectangles with metric-based color scale (yellow-to-purple)
4. Draw polylines between connected PU centroids, opacity scaled by edge weight

## 7. File Changes Summary

### New Files (9)

| File | Purpose |
|---|---|
| `src/pymarxan/models/geometry.py` | Synthetic geometry generator |
| `src/pymarxan_shiny/modules/data/__init__.py` | Data modules package |
| `src/pymarxan_shiny/modules/data/feature_table.py` | Editable feature table |
| `src/pymarxan_shiny/modules/mapping/spatial_grid.py` | PU map with ipyleaflet |
| `src/pymarxan_shiny/modules/mapping/frequency_map.py` | Selection frequency heatmap |
| `src/pymarxan_shiny/modules/mapping/comparison_map.py` | Side-by-side solution comparison |
| `src/pymarxan_shiny/modules/mapping/network_view.py` | Connectivity network overlay |
| `src/pymarxan_shiny/modules/calibration/sensitivity.py` | Sensitivity dashboard (Shiny module) |
| `tests/pymarxan/models/test_geometry.py` | Geometry tests |

### Modified Files (2)

| File | Change |
|---|---|
| `src/pymarxan_shiny/modules/mapping/solution_map.py` | Upgrade from table to ipyleaflet |
| `src/pymarxan_app/app.py` | Wire in all 7 new modules |

### Unchanged

- All 8 solvers, solver ABC, SolverConfig, ProblemCache, ZoneProblemCache
- All Phase 8 modules (run_panel, convergence, progress)
- solver_picker, blm_calibration, spf_calibration modules
- All existing 417 tests
- Core models, I/O, calibration backends, analysis backends

## 8. Testing Strategy

1. All 417 existing tests must continue to pass
2. Geometry tests: grid dimensions, cell count matches n_pu, bounds correctness
3. Feature table tests: edit validation, persistence to DataFrame, read-only column enforcement
4. Map module tests: UI returns tag, server callable, color function correctness
5. Sensitivity tests: config building from UI inputs, result display
6. Network view tests: edge list building, metric coloring, polyline generation
7. Integration tests: modules wire into app.py correctly
