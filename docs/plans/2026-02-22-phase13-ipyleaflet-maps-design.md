# Phase 13: Real ipyleaflet Maps — Interactive Map Widgets for All 5 Modules

**Date:** 2026-02-22
**Status:** Design Document
**Goal:** Replace text-summary map modules with interactive ipyleaflet maps using shinywidgets output_widget/render_widget, while preserving text fallback for headless/CI environments.

---

## 1. Scope

| # | Module | Complexity | Key change |
|---|--------|------------|------------|
| 1 | solution_map | Low | Green/gray rectangles for selected/unselected PUs |
| 2 | spatial_grid | Low | Cost (yellow→red) or status (categorical) rectangles |
| 3 | frequency_map | Low | White→blue heatmap rectangles |
| 4 | comparison_map | Medium | 4-color rectangles (both/A/B/neither) |
| 5 | network_view | Medium | Metric-colored rectangles + polyline edges |

Pattern for all 5: Replace `ui.output_ui("map_content")` / `@render.ui` with `output_widget("map")` / `@render_widget`. Each module keeps its existing text summary as a secondary output alongside the map. Existing color functions reused unchanged.

Fallback: If ipyleaflet isn't available (import fails), each module falls back to the current text summary. Preserves headless/CI compatibility.

## 2. Shared Map Helper

New file: `src/pymarxan_shiny/modules/mapping/map_utils.py`

```python
def create_grid_map(grid, colors, center=None, zoom=12):
    """Create ipyleaflet Map with colored Rectangle layers.

    Parameters
    ----------
    grid : list of ((south, west), (north, east)) tuples
    colors : list of hex color strings (same length as grid)
    center : optional (lat, lon) tuple
    zoom : int, default 12

    Returns
    -------
    ipyleaflet.Map with one Rectangle per PU
    """
```

Takes grid from `generate_grid()` and a list of hex colors. Returns an ipyleaflet.Map with one Rectangle per grid cell. Auto-centers on the grid midpoint if center not provided.

Used by all 5 modules. Network_view additionally adds Polyline layers for edges on top.

## 3. Module Upgrade Pattern

Each module gets the same structural change:

**UI:** `output_widget("map")` + `ui.output_text_verbatim("map_summary")`

**Server:** `@render_widget def map()` builds colors list using existing color functions, calls `create_grid_map()`. Separate `@render.text def map_summary()` keeps the text output for accessibility.

**Fallback:** `@render_widget` wrapped in try/except ImportError — returns None if ipyleaflet unavailable, text summary carries the output alone.

**Tests:** Existing tests unchanged. New tests verify `create_grid_map` produces Map with correct layer count.

## 4. File Changes Summary

### New Files (2)

| File | Purpose |
|------|---------|
| `src/pymarxan_shiny/modules/mapping/map_utils.py` | Shared `create_grid_map()` helper |
| `tests/pymarxan_shiny/test_map_utils.py` | Tests for map helper |

### Modified Files (5)

| File | Change |
|------|--------|
| `src/pymarxan_shiny/modules/mapping/solution_map.py` | `@render_widget` + green/gray rectangles |
| `src/pymarxan_shiny/modules/mapping/spatial_grid.py` | `@render_widget` + cost/status colored rectangles |
| `src/pymarxan_shiny/modules/mapping/frequency_map.py` | `@render_widget` + white→blue heatmap |
| `src/pymarxan_shiny/modules/mapping/comparison_map.py` | `@render_widget` + 4-color comparison |
| `src/pymarxan_shiny/modules/mapping/network_view.py` | `@render_widget` + rectangles + polyline edges |

### Integration Tests (1)

| File | Purpose |
|------|---------|
| `tests/test_integration_phase13.py` | Phase 13 integration tests |

### Unchanged

- All color functions (cost_color, status_color, frequency_color, comparison_color, metric_color)
- All solver, model, I/O, calibration, and analysis code
- app.py wiring (modules keep same function signatures)
- All existing 489 tests
- CI, Makefile, README, pyproject.toml

## 5. Testing Strategy

1. All 489 existing tests must continue to pass
2. `create_grid_map` tests: correct Map type, correct layer count, auto-centering
3. Each module's UI still returns a tag (output_widget produces a valid tag)
4. Each module's server is still callable
5. Fallback: if ipyleaflet import is mocked to fail, modules still render text summaries
6. Integration: app imports cleanly, all map modules importable
