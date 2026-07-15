# S4a — separation-distance on grid planning units — spec + plan

**Date:** 2026-07-15
**Status:** Approved (brainstorm), streamlined (spec + TDD; one-branch core change).
**Scope:** `solvers/separation.py::get_pu_coordinates` + the geographic-CRS warning in
`solvers/cache.py`. Closes the separation-on-grids limitation the S2 review flagged. S4b (the
raster-aware Shiny map + `has_grid` gating) is a separate, larger effort.

## Motivation

SEPDISTANCE/SEPNUM require per-PU coordinates. `get_pu_coordinates` resolves them from a vector
geometry column or `xloc`/`yloc`, else raises `PUCoordinatesUnavailableError`. A grid-ingested
problem (S2/S3) has neither, but it *does* carry a `GridGeometry` whose `cell_centroids()` are
exactly the `(n_pu, 2)` coordinates needed (in PU order). S4a wires that in, re-enabling
separation on grid problems. It does **not** touch `has_geometry` (which stays vector-only —
12+ Shiny modules gate on it and read `planning_units.geometry`).

## Design

1. **`get_pu_coordinates` grid fallback.** Insert a branch after the `xloc`/`yloc` tier and
   before the final `raise`:
   ```python
   if problem.grid is not None:
       coords = np.asarray(problem.grid.cell_centroids(), dtype=np.float64)
       # NaN guard, consistent with the other tiers (only if the grid origin/cell size
       # were non-finite):
       if np.isnan(coords).any():
           raise PUCoordinatesUnavailableError(
               "grid.cell_centroids() produced NaN coordinates."
           )
       return coords
   ```
   Update the docstring's three-tier fallback → four-tier (geometry → xloc/yloc → grid → raise).

2. **Geographic-CRS warning covers `grid.crs`.** In `cache.py`'s separation block, the
   degrees-vs-metres warning currently reads `getattr(pu_df, "crs", None)` — `None` for a grid
   problem (plain-DataFrame `planning_units`), so a geographic grid (e.g. EPSG:4326, common for
   rasters) would silently give degrees-based separation. Resolve `grid.crs` when the DataFrame
   has none:
   ```python
   crs = getattr(pu_df, "crs", None)
   if crs is None and problem.grid is not None and problem.grid.crs is not None:
       try:
           from pyproj import CRS as _PyprojCRS
           crs = _PyprojCRS.from_user_input(problem.grid.crs)
       except Exception:
           crs = None
   if crs is not None and getattr(crs, "is_geographic", False):
       warnings.warn(...)  # unchanged message
   ```
   pyproj is already available (geopandas depends on it). The `try/except` guards an unparseable
   CRS string.

That's the whole change — separation's `SepState`/delta already consume the resolved `pu_coords`,
so nothing else is needed for it to work on grids.

## Tests (TDD)

- **`get_pu_coordinates` grid fallback:** a `ConservationProblem` with a `grid` field and no
  geometry/`xloc` returns `grid.cell_centroids()` (no raise).
- **Precedence:** with both a geometry column *and* a grid, the geometry centroids win (grid is
  the fallback, not an override); a problem with only `xloc`/`yloc` and a grid uses `xloc`/`yloc`.
- **Cache builds separation on a grid problem:** `ProblemCache.from_problem` on a grid problem
  with a sep-active feature (`sepnum>1`, `sepdistance>0`) sets `separation_active` and populates
  `pu_coords == grid.cell_centroids()` without raising.
- **Geographic grid warns:** a grid problem with `grid.crs="EPSG:4326"` + a sep-active feature
  emits the degrees-vs-metres `UserWarning`; a projected grid (e.g. `EPSG:3035`) does not.
- **Anchor:** `make check` green; the 35.0 parity and existing separation tests untouched.

## References

`solvers/separation.py::get_pu_coordinates` (3-tier fallback), `solvers/cache.py` (separation
precompute + CRS warning), `models/grid.py::GridGeometry.cell_centroids`. Flagged as the S2
"known limitation" (`2026-07-15-raster-grid-s2-review.md`).
