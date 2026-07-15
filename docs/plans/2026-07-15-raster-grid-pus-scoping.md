# Raster-grid planning units — scoping

**Date:** 2026-07-15
**Status:** Scoping (pre-spec). Decomposition + decisions + recommended first sub-project.
**Origin:** the last big open gap from the 2026-06-12 ecosystem survey (B1) — "the
single biggest functional disadvantage vs. prioritizr/Zonation."

## The gap

pymarxan can already **ingest** raster cost/feature layers as per-PU zonal
statistics (`spatial/cost_surface.py::apply_cost_from_raster`,
`spatial/feature_intersection.py` via `rasterio.mask`), but planning units are
always **vector polygons** — `spatial/grid.py::generate_planning_grid` returns a
GeoDataFrame of `Polygon` cells, and `has_geometry` keys off a shapely `geometry`
column. The gap is the **raster-cell-as-planning-unit** model: use the grid cells
themselves as PUs without materializing a polygon per cell.

## Why this is architectural (not additive like the Zonation phases)

Three core things are vector-geometry-based today and don't scale to raster grids:
1. **Geometry** — `ConservationProblem` carries geometry only as a shapely
   `geometry` column on `planning_units` (`models/problem.py:325`); there is no
   affine/transform/mask field.
2. **Boundary** — `spatial/boundary.py::compute_boundary` uses a shapely STRtree +
   shared-edge-length spatial join (≈O(n²)); prohibitive at raster scale, and
   entirely unnecessary for a *regular* grid where adjacency is known analytically.
3. **Ingestion** — the raster path clips the raster **per polygon**
   (`rasterio.mask` once per PU) — the slow zonal-stat path.

Vectorizing a large raster into one polygon per cell is exactly the scale wall to
avoid — so the feature is upstream of the solvers, in the model + boundary +
ingestion.

## Scale target (decided): **modest first** — thousands of cells

Target grids up to ~tens of thousands of valid cells. This keeps:
- the existing dense `build_pu_feature_matrix` + DataFrame PU model;
- **all** solvers viable, including exact MIP (which does not scale to 1e6 vars);
- no sparse-matrix / `ProblemCache`-rework needed.

Full raster scale (1e5–1e7 cells, sparse matrices, heuristic-solvers-only,
streaming `exactextract`) is a deliberate **later** effort (see S3, deferred).

## Decomposition into sub-projects

- **S1 — Grid-geometry model + analytic boundary (the core; spec first).**
  A `GridGeometry` descriptor (affine transform + grid shape `(nrows, ncols)` +
  a boolean validity mask marking which cells are PUs) carried as a **`kw_only`
  field** `grid: GridGeometry | None = None` on `ConservationProblem` (per the
  project's backward-compatibility rule). The `planning_units` DataFrame stays
  one row per **valid** cell (id/cost/status), with **no** vector geometry column;
  `has_geometry` returns True when `grid is not None`, and cell polygons/centroids
  are derived from the transform on demand (for mapping, SEPDISTANCE). The
  **boundary matrix is generated analytically** from rook adjacency of the grid
  (shared edge = cell size; self-boundary = perimeter for cells on the study-area
  edge) — no shapely, and O(n) instead of the STRtree join.
  *Parity anchor:* on a small fully-valid grid, the analytic boundary must equal
  what `compute_boundary` produces on the equivalent vector grid.

- **S2 — Raster ingestion (spec second, builds on S1).**
  A `from_rasters(...)` constructor: aligned feature rasters (+ optional cost
  raster, status raster) on a common grid → a `ConservationProblem` with a
  `GridGeometry`, a per-valid-cell PU table, and a sparse-ish `pu_vs_features`
  (a row per (cell, feature) where amount > 0). Uses `rasterio`/numpy only
  (already an optional dep). Nodata → excluded cells (the validity mask).
  `exactextract` and misaligned/reprojected rasters are **deferred** (a later
  fast-path for non-aligned layers).

- **S3 — Scale (deferred).** Sparse `pu_vs_features` / masked matrix ops for large
  grids; heuristic-solver-only path (SA/greedy/Zonation; MIP infeasible at 1e6
  vars); `ProblemCache`/delta-model at scale. Out of scope until modest works.

- **S4 — UI/mapping for raster PUs (deferred).** A raster-aware map layer in the
  Shiny app (render cells from the grid transform). The existing `create_grid_map`
  already draws rectangles from bounding boxes, so it partly generalizes.

## Key decisions (settled during scoping)

| Decision | Choice | Why |
|---|---|---|
| Scale target | Modest (thousands) | Keeps dense model + all solvers incl. MIP; tractable first cut |
| PU representation | `GridGeometry` as a `kw_only` field on `ConservationProblem` | Backward-compatible (CLAUDE.md rule); solvers unchanged; composition over a new type |
| Geometry column | None on raster PUs; derive polygons/centroids from the transform | Avoids materializing a polygon per cell (the scale wall) |
| Boundary | Analytic rook adjacency from the grid | O(n) vs shapely STRtree O(n²); the real scale win |
| `exactextract` | Deferred; `rasterio`/numpy aligned fast-path first | Lean deps; aligned rasters are the common case |
| Solver changes | None | They consume the derived DataFrames/matrix; unchanged |

## Recommended first sub-project: **S1**

S1 is the self-contained architectural core — it introduces the `GridGeometry`
model and the analytic boundary (the immediate scale win), is independently
testable against the shapely boundary as a parity anchor, and is what S2's
ingestion builds a problem *into*. Spec S1 next; S2 follows once the model exists.

## Open questions for the S1 spec

- **Cell → PU id scheme.** Row-major cell index over valid cells? A stable
  `(row, col)` → id map must round-trip with the mask so mapping/boundary align to
  `build_pu_feature_matrix` row order (the positional-alignment contract that bit
  the Zonation smoothing review).
- **`has_geometry` semantics.** Extend it to `True` for grid-only problems, or add
  a separate `has_grid`? Downstream map modules gate on `has_geometry` — check they
  degrade correctly when there's a grid but no `geometry` column.
- **CRS.** Carry a CRS on `GridGeometry` (for SEPDISTANCE units + mapping), matching
  the raster's CRS.

## References

2026-06-12 ecosystem survey (`docs/plans/2026-06-12-ecosystem-survey.md`, B1).
Reuses `rasterio` (optional dep). `exactextract` would be a new optional dep,
deferred.
