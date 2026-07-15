# S4b — raster-aware Shiny map — design

**Date:** 2026-07-15
**Status:** Approved (brainstorm), pending spec review → implementation plan → design review.
**Scope:** Shiny mapping. Give raster-grid (`GridGeometry`) problems a real map across the
solution-display maps + the connectivity network view. Last raster-grid piece (S1–S4a shipped).

## Motivation

Grid-ingested problems (S2/S3) carry a `GridGeometry` but no vector `geometry` column, so the
Shiny maps fall through to a **synthetic** fake grid (`generate_grid(n_pu)` — cells in an
arbitrary square with no real geography). S4b renders them on their **actual** footprint using
`GridGeometry.cell_bounds()` (reprojected to lat/lon), so a raster solution shows up in the right
place. `has_geometry` stays vector-only (12+ modules read `planning_units.geometry`); grid
support goes through a new predicate + a shared dispatcher.

## Scope

- **`has_grid(problem)`** predicate (`models/problem.py`, exported) — `True` when
  `problem.grid is not None`.
- **`map_utils.build_pu_map(problem, colors, *, zoom=..., max_cells=5000)`** — the shared 3-way
  base-map dispatcher (replaces the inline `if has_geometry … else generate_grid` blocks):
  1. `has_geometry(problem)` → `create_geo_map(problem.planning_units, colors)`.
  2. `has_grid(problem)` → `create_grid_map(_grid_bounds_latlon(problem.grid), colors)`.
  3. else → `create_grid_map(generate_grid(n_pu), colors)` (synthetic, unchanged).
  Returns `None` when ipyleaflet is missing **or** the problem is a grid with `n_pu > max_cells`
  (one `Rectangle` per cell would freeze the browser). The cap applies to the **grid branch only**
  — the vector (`create_geo_map`) and synthetic (`generate_grid`) branches keep their current
  uncapped behaviour (a classic large-`n_pu` Marxan project is a pre-existing concern outside
  S4b's raster scope).

  **Surfacing the cap message.** `@render_widget` renders a *widget*, so it cannot itself show
  text — on `None` the map area simply stays blank. The "grid too large to map (N cells) — use the
  analysis/table views" message is therefore surfaced through each module's **existing summary /
  status `render.text`** (e.g. `frequency_map.map_summary`), which checks the same
  `has_grid(p) and n_pu > max_cells` condition. A tiny helper
  `map_utils.too_large_for_map(problem, max_cells=5000) -> bool` centralizes the predicate so the
  map render (`build_pu_map` → None) and the summary text stay in lockstep. Where a module has no
  summary text, the map is simply blank above the cap.
- **`map_utils.pu_centroids_latlon(problem)`** — companion 3-way helper returning
  `list[(lat, lon)]` for network overlays: geometry → reprojected polygon centroids;
  `has_grid` → `grid.cell_centroids()` reprojected; else → `compute_centroids(generate_grid(n_pu))`.
- **`_grid_bounds_latlon(grid)`** — converts `grid.cell_bounds()` `(minx, miny, maxx, maxy)` per
  cell to `((south, west), (north, east))` in EPSG:4326: grid CRS `None` or already-4326 → use raw
  `y=lat, x=lon`; projected → `pyproj.Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)`
  on the corners (vectorized; cells are tiny so corner-transform is adequate). Same transformer
  reused for the centroids.
- **Refactor** the 5 solution-display maps (`frequency_map`, `solution_map`, `comparison_map`,
  `spatial_grid`, `zonation/zonation_panel`) to call `build_pu_map` + show the too-large message on
  `None`. **`network_view`**: base map via `build_pu_map`, node centroids via
  `pu_centroids_latlon`, then its existing capped polyline overlay; skip (return `None` / message)
  when `build_pu_map` returns `None`.

Out of scope: changing `has_geometry`; the spatial *input* modules (`grid_builder`, `gadm_picker`,
`cost_upload`, `wdpa_overlay` — they build/pick geometry, not display solutions); downsampling large
grids (the cap skips instead, per the brainstorm decision).

## Reprojection detail

`GridGeometry.crs` is a string (e.g. `"EPSG:3035"`) or `None`. `_grid_bounds_latlon`:
```python
bounds = grid.cell_bounds()  # list of (minx, miny, maxx, maxy) in grid CRS
minx, miny, maxx, maxy = (np.array(c) for c in zip(*bounds))
crs = grid.crs
if crs is not None and _epsg(crs) != 4326:
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    west, south = tr.transform(minx, miny)   # arrays; always_xy -> (lon, lat)
    east, north = tr.transform(maxx, maxy)
else:                                        # None or already 4326: y=lat, x=lon
    west, south, east, north = minx, miny, maxx, maxy
return [((s, w), (n, e)) for s, w, n, e in zip(south, west, north, east)]
```
`pyproj` is already available (geopandas depends on it). A `None` CRS is rendered with raw coords
(same as the synthetic path treats arbitrary units) — best effort, no reprojection possible.

## `has_geometry` is untouched

Confirmed: 12+ modules do `if has_geometry(p): <read planning_units.geometry>`. Flipping it would
break them all. S4b adds `has_grid` and routes grid problems through `build_pu_map`; the vector
path is unchanged.

## Testing strategy (TDD)

Tests that build an `ipyleaflet.Map` outside a Shiny session use the
`_allow_widget_outside_session` fixture (CLAUDE.md).

- **`has_grid`:** `True` for a problem with a `grid`, `False` otherwise; `has_geometry` unchanged
  (still `False` for a grid-only problem).
- **`_grid_bounds_latlon`:** a projected grid (`EPSG:3035`) reprojects to plausible EPSG:4326
  `((s,w),(n,e))` boxes (south<north, west<east, in degree range); a `None`/4326 grid passes raw
  coords through; row order matches `cell_bounds()`/PU order.
- **`build_pu_map` three branches:** a vector problem → a Map (GeoJSON layers); a small grid problem
  → a Map with one Rectangle per valid cell; a classic (no geometry/grid) problem → the synthetic
  grid Map (unchanged). ipyleaflet-missing → `None`.
- **Large-grid cap:** a grid problem with `n_pu > max_cells` → `build_pu_map` returns `None`; a
  `max_cells`-sized grid renders.
- **`pu_centroids_latlon`:** grid branch → `len == n_pu`, values in the reprojected range; matches
  `build_pu_map`'s cell order.
- **`too_large_for_map`:** `True` for a grid over the cap, `False` for a small grid / vector /
  classic problem — the predicate the map render and summary text share.
- **Module refactor (guarded, not unit-tested in isolation):** Shiny `@render_widget` functions
  aren't cleanly unit-testable outside a reactive session, so the module refactor is guarded by
  (a) the helper tests above (the map/centroid/predicate logic all lives in `map_utils`, not the
  modules), (b) the **existing module tests staying green** (no vector-path regression), and
  (c) the live check. Each module becomes a thin `build_pu_map(p, colors)` call + a summary-text
  branch on `too_large_for_map`.
- **Live (best-effort):** launch the app, load a small grid problem, screenshot the map showing
  cells (Playwright; the dir-load modal may resist headless automation as in Zonation Phase D — the
  helper unit tests are the hard guard).

**Target:** ~14–18 tests, `make check` green (0 ruff / 0 mypy), coverage ≥ 75%.

## References

`mapping/map_utils.py` (`create_geo_map`/`create_grid_map`), `mapping/{frequency,solution,
comparison,network_view,spatial_grid}_map.py`, `zonation/zonation_panel.py`, `models/geometry.py`
(`generate_grid`), `models/grid.py` (`cell_bounds`/`cell_centroids`). Precedent: Zonation Phase D
panel (`2026-07-14-zonation-phase-d-*`). `has_geometry` gating in `models/problem.py`.
