# Raster-grid PUs — S2: raster ingestion (`from_rasters`) — design

**Date:** 2026-07-15
**Status:** Approved (brainstorm), pending spec review → implementation plan
**Scope:** S2 only — build a `ConservationProblem` (with a S1 `GridGeometry`) directly
from aligned rasters. S3 (scale/sparse) and S4 (UI) remain deferred. Builds on S1
(`GridGeometry`, shipped v0.17.0). Scoping: `2026-07-15-raster-grid-pus-scoping.md`.

## Motivation

S1 gave us the grid-cells-as-PUs *model* (`GridGeometry` + analytic boundary). S2 is
the *ingestion* that populates it: aligned feature rasters (+ optional cost / status /
mask rasters) on a common grid → a solvable `ConservationProblem` whose planning units
are the valid grid cells, with **no polygon materialized per cell**. This is the
grid-native counterpart to the existing per-polygon path
(`spatial/feature_intersection.py`, which clips the raster once per PU via
`rasterio.mask` — the slow zonal-stat route). At modest scale (thousands to tens of
thousands of valid cells) the resulting problem uses the existing dense matrix and every
solver, including exact MIP.

## Scope (S2)

- A **pure numpy core** `from_arrays(...)` — aligned numpy arrays + the 4 grid scalars →
  `ConservationProblem`. All ingestion logic lives here; no rasterio.
- A **thin rasterio wrapper** `from_rasters(...)` — reads each raster, derives the
  transform/CRS, normalizes nodata → NaN, delegates to `from_arrays`.
- **One feature container covers both layouts:** `feature_rasters: dict[int, path |
  (path, band)]` — a plain path is a single-band file; a `(path, band)` tuple pulls one
  band from a stack. Separate files and a multi-band stack use the same code path.
- **Validity precedence** for which cells become PUs: explicit `mask` → `cost` non-nodata
  → union of feature non-nodata.
- Builds the `boundary` matrix by default (via S1 `grid.build_boundary`), so BLM works
  out of the box.

Out of scope: reprojection / misaligned rasters (raise, don't warp), `exactextract`
(S2 uses aligned `rasterio`/numpy), sparse/large-scale matrix ops (S3), UI/mapping and
`has_geometry` changes (S4), streaming/windowed reads, per-feature CRS.

## Module layout

```
src/pymarxan/spatial/raster.py    # from_arrays (pure numpy core) + from_rasters (rasterio)
tests/pymarxan/spatial/test_raster.py
```

Both functions live in `spatial/` (the geo-ops layer). `rasterio` is imported at module
top, consistent with the sibling `feature_intersection.py` / `cost_surface.py` (importing
a `ConservationProblem` already pulls in geopandas, so the module is a geo-extras module
regardless). `from_arrays` calls **no** rasterio — its logic is fully exercised with plain
numpy in tests (no GeoTIFF fixtures); only the thin read layer of `from_rasters` needs
`rasterio.MemoryFile`.

## Component — `from_arrays` (the pure core)

```python
def from_arrays(
    feature_arrays: dict[int, np.ndarray],
    *,
    x_min: float,
    y_max: float,
    cell_width: float,
    cell_height: float,
    crs: str | None = None,
    cost_array: np.ndarray | None = None,
    status_array: np.ndarray | None = None,
    mask_array: np.ndarray | None = None,
    feature_names: dict[int, str] | None = None,
    nodata: float | None = None,
    include_boundary: bool = True,
) -> ConservationProblem: ...
```

Takes the 4 grid scalars (matching `GridGeometry`) rather than an `affine.Affine`, so
the core has no rasterio/affine dependency. Semantics:

1. **Shape check.** All provided arrays (`feature_arrays.values()`, `cost_array`,
   `status_array`, `mask_array`) must be 2-D and share one shape `(nrows, ncols)`, else
   `ValueError` naming the offender. `feature_arrays` must be non-empty.
2. **Nodata.** A cell is nodata in an array where `np.isnan(value)` **always**, or where
   `value == nodata` when the `nodata` sentinel is given (for integer arrays that can't
   carry NaN). `from_rasters` normalizes each raster's own nodata to NaN before calling,
   so it never needs the sentinel.
3. **Validity mask (which cells are PUs), by precedence:**
   - if `mask_array` is not None → valid = truthy **and** non-nodata;
   - elif `cost_array` is not None → valid = `cost_array` non-nodata;
   - else → valid = the **union** over `feature_arrays` of non-nodata cells.
   At least one valid cell is required (else `ValueError` — an empty study area is a
   usage error; also surfaces via `GridGeometry`'s own all-False guard).
4. **`GridGeometry`** built from `(x_min, y_max, cell_width, cell_height, valid_mask,
   crs)`.
5. **PU order & ids.** `valid_cells()` row-major order is the PU order; `planning_units`
   ids are `1..n_pu` in that order (the S1 positional-alignment contract — the same order
   `build_pu_feature_matrix` and `build_boundary(1..n_pu)` use).
6. **cost.** `planning_units.cost` = `cost_array` sampled at the valid cells; `1.0` where
   `cost_array` is None or nodata at a valid cell.
7. **status.** `planning_units.status` = `status_array` sampled at the valid cells; `0`
   where None or nodata. Each non-nodata value must be integer-valued and in `{0,1,2,3}`
   (Marxan codes) else `ValueError` — a non-integer status (e.g. `2.7`) raises rather than
   silently truncating.
8. **features table.** id = `sorted(feature_arrays)`; `name` = `feature_names[id]` or
   `f"feature_{id}"`; `target = 0.0`, `spf = 1.0` (set later via the existing feature-
   override machinery). Duplicate/missing `feature_names` keys are tolerated (fallback
   name).
9. **`pu_vs_features` (sparse long form).** For each feature `id` and each valid cell,
   `amount` = the feature array value there; emit a row `{species=id, pu=pu_id,
   amount=value}` **only where `amount > 0` and non-nodata**. (Zero / nodata → no row —
   the natural sparse encoding; the dense `build_pu_feature_matrix` fills those with 0.)
10. **boundary.** If `include_boundary` (default True), `boundary =
    grid.build_boundary(planning_units["id"].to_numpy())`; else `None`.
11. Returns `ConservationProblem(planning_units, features, pu_vs_features,
    boundary=boundary, grid=grid)`.

## Component — `from_rasters` (the rasterio wrapper)

```python
def from_rasters(
    feature_rasters: dict[int, str | Path | tuple[str | Path, int]],
    *,
    cost_raster: str | Path | None = None,
    status_raster: str | Path | None = None,
    mask_raster: str | Path | None = None,
    feature_names: dict[int, str] | None = None,
    include_boundary: bool = True,
) -> ConservationProblem: ...
```

1. **Read.** For each `feature_rasters` value: open the path, read band `b` (a plain path
   → band 1; a `(path, b)` tuple → band `b`; rasterio bands are **1-indexed**) as float.
   Replace the source nodata with NaN (`src.nodata` when set). `cost_raster` /
   `status_raster` / `mask_raster` are single-band (band 1).
2. **Reference grid.** Take the transform/shape/CRS from the **first** feature raster:
   `x_min = transform.c`, `y_max = transform.f`, `cell_width = transform.a`,
   `cell_height = -transform.e`, `crs = src.crs.to_string()` (or `None` if unset).
   **Guard axis-aligned north-up:** reject a rotated/sheared transform (`transform.b` or
   `transform.d` non-zero, within tolerance) and a non-north-up transform
   (`transform.e >= 0`, i.e. rows not top-down) with a clear `ValueError` — `GridGeometry`
   is axis-aligned with `y_max` at the top, and silently ignoring `b`/`d` would misplace
   every cell. (Warping/flipping is deferred; this fails loudly instead.)
3. **Alignment (reprojection deferred).** Every other raster (features, cost, status,
   mask) must have the **same** transform, shape, and CRS as the reference, else
   `ValueError` naming the mismatched input. Transforms are compared with a small
   tolerance (e.g. `Affine.almost_equals` / `np.allclose` on the 6 coefficients), since
   two independently-written rasters describing the same grid can differ by FP epsilon;
   shape is compared exactly; CRS via rasterio's own `CRS` equality (`src.crs ==
   ref_crs`, not string comparison, so an EPSG-code and an equivalent-WKT serialization
   of the same CRS don't falsely mismatch). No warping.
4. **Delegate** the assembled arrays + grid scalars to `from_arrays` (nodata already NaN,
   so no sentinel needed).

## `ConservationProblem` interaction

S2 constructs the existing `ConservationProblem` — no model change (S1 already added the
`grid` field). The resulting problem:
- passes `validate()` (S1's grid branch checks `grid.n_pu == len(planning_units)`, which
  holds by construction);
- has `boundary` populated (so BLM/`compute_boundary`-shaped consumers work);
- `has_geometry(problem)` returns **False** (no vector `geometry` column — grid-aware
  mapping is S4); solvers don't call it.

## Testing strategy (TDD)

Core (`from_arrays`, plain numpy — no rasterio):
- **Basic build:** a `3×3` all-valid grid with 2 feature arrays → 9 PUs, ids `1..9`
  row-major, `build_pu_feature_matrix()` equals the stacked input arrays at the valid
  cells (the round-trip anchor).
- **Validity precedence:** (a) NaN in a feature array + no cost/mask → those cells
  dropped (feature-union); (b) a `cost_array` with a NaN cell that a feature covers →
  that cell dropped (cost footprint wins over union); (c) a `mask_array` → mask wins over
  both cost and features.
- **Sparse amounts:** `amount == 0` and nodata cells produce **no** `pu_vs_features`
  row; positive values do.
- **cost / status defaults:** no `cost_array` → all cost `1.0`; a `status_array` with a
  `2` (locked-in) → that PU's status is `2`; an out-of-range status (e.g. `7`) raises.
- **Alignment raise:** a feature array of a different shape raises `ValueError`.
- **boundary wiring:** `include_boundary=True` → `boundary` equals
  `grid.build_boundary(1..n_pu)`; `False` → `boundary is None`.
- **feature_names:** provided names flow into `features.name`; missing → `feature_{id}`.
- **empty study area:** an all-nodata feature (no cost/mask) raises `ValueError`.

Wrapper (`from_rasters`, `rasterio.MemoryFile`):
- **Single-band files:** a dict of two in-memory single-band rasters → same problem as
  the equivalent `from_arrays` call.
- **Multi-band `(path, band)`:** a 2-band stack addressed as `{1: (p,1), 2: (p,2)}`
  builds the same problem as two separate files.
- **Nodata → NaN:** a raster with `nodata=-9999` drops those cells (validity + amount).
- **Transform / CRS mismatch:** a second raster with a different transform (or CRS)
  raises `ValueError`.
- **Non-axis-aligned / non-north-up:** a rotated/sheared transform (`b` or `d` non-zero)
  and a south-up transform (`e >= 0`) each raise `ValueError`.
- **Non-integer status:** a status raster with a `2.7` value raises `ValueError`.

**Target:** ~16–20 tests, `make check` green (0 ruff / 0 mypy), coverage ≥ 75%.

## Parity note

S2 adds an ingestion path only — no solver or objective math changes. The 35.0 min-set
anchor is untouched. The round-trip test (input arrays == `build_pu_feature_matrix` at
valid cells) plus the S1 boundary parity guarantee the constructed problem is faithful,
and the resulting problem is solvable by the existing MIP/SA/greedy solvers unchanged.

## References

Scoping: `2026-07-15-raster-grid-pus-scoping.md` (S2). S1 model: `models/grid.py`
(`GridGeometry`, `build_boundary`), shipped v0.17.0. Precedent for a module-level
ingestion returning a problem: `rivers/io.py::from_hydrorivers`. Existing raster reads:
`spatial/feature_intersection.py`, `spatial/cost_surface.py` (rasterio at module top).
`rasterio` is an already-declared optional dependency; `exactextract` (deferred) would be
a new one.
