# Raster-grid PUs — S1: GridGeometry model + analytic boundary — design

**Date:** 2026-07-15
**Status:** Approved (brainstorm/scoping), pending implementation plan + review
**Scope:** S1 only — the core grid-geometry model and analytic boundary. S2
(raster ingestion), S3 (scale), S4 (UI) remain deferred. See the scoping doc
`docs/plans/2026-07-15-raster-grid-pus-scoping.md`.

## Motivation

Raster-grid planning units (the last big survey gap, B1) need a way to treat grid
**cells as PUs without materializing a polygon per cell**, and to compute the
boundary matrix (for BLM) analytically instead of via a shapely spatial join. S1
delivers that core: a `GridGeometry` descriptor + an analytic rook-adjacency
boundary. S2's raster ingestion builds a `ConservationProblem` *into* this model;
S1 is the self-contained, independently-testable foundation.

## Scope (S1)

- A `GridGeometry` dataclass: affine grid (origin + cell size) + shape + a boolean
  **validity mask** (which cells are PUs), + CRS.
- A **`kw_only` field** `grid: GridGeometry | None = None` on `ConservationProblem`
  (backward-compatible per the project rule; `copy_with` already forwards it).
- Methods to derive per-PU geometry on demand (centroids, cell bounds) and to
  **build the boundary DataFrame analytically** from rook adjacency.
- **Parity anchor:** the analytic boundary equals `compute_boundary` (shapely) on
  the identical cells.

Out of scope: raster ingestion (S2), sparse/large-scale (S3), map/UI integration
and `has_geometry` changes (S4), hexagonal grids, rotated/skewed transforms,
SEPDISTANCE-on-grid.

## Module layout

```
src/pymarxan/models/grid.py       # GridGeometry (pure numpy/pandas, no shapely)
src/pymarxan/models/problem.py    # + grid kw_only field
tests/pymarxan/models/test_grid.py
```

`GridGeometry` lives in `models/` (core) and is **pure numpy/pandas** — no shapely
or rasterio. The standalone class is geo-free (usable at scale); note that a
`ConservationProblem` *carrying* a grid still needs geopandas, since `problem.py`
hard-imports it at module level for `has_geometry`. (The parity *test* uses shapely
to build the comparison grid, but the class does not.) Its docstring cross-references
the two sibling modules it could be confused with: `spatial/grid.py`
(`generate_planning_grid` — the materialized-vector square grid) and
`models/geometry.py` (`generate_grid` — synthetic bboxes for the map). `GridGeometry`
is the analytic, polygon-free counterpart.

## Component — `GridGeometry` (`models/grid.py`)

A north-up, axis-aligned grid (no rotation — the normal raster case):

```python
@dataclass(eq=False)   # eq=False: the numpy mask field breaks the auto __eq__
class GridGeometry:
    x_min: float          # left edge of column 0
    y_max: float          # top edge of row 0 (top-down, raster convention)
    cell_width: float     # > 0
    cell_height: float    # > 0
    mask: np.ndarray      # (nrows, ncols) bool; True = a valid cell (a PU)
    crs: str | None = None
```

- **`__post_init__`:** `mask` is 2-D and boolean; `cell_width`/`cell_height` > 0;
  at least one `True` in the mask (else `ValueError` — a grid with no PUs is a
  usage error). Store `shape = mask.shape`.
- **`n_pu -> int`** = `int(mask.sum())`.
- **Row-major valid-cell order is the PU order.** `valid_cells() -> list[(r,c)]`
  returns the valid `(row, col)` pairs in **row-major** order (row 0 = top). This
  order **is** the planning-unit order: PU DataFrame row `i` ↔ the `i`-th valid
  cell ↔ the `i`-th row of `build_pu_feature_matrix`. (This is the positional
  alignment contract that the Zonation-smoothing review flagged — stated and
  tested here.)
- **`cell_centroids() -> np.ndarray`** — `(n_pu, 2)` array of `(x, y)` centroids in
  PU order: `x = x_min + (c + 0.5)·cell_width`, `y = y_max − (r + 0.5)·cell_height`.
  (For SEPDISTANCE / future mapping.)
- **`cell_bounds() -> list[tuple[float,float,float,float]]`** — `(minx, miny, maxx,
  maxy)` per PU in PU order (for mapping and the parity test's comparison grid).
- **`build_boundary(pu_ids: np.ndarray) -> pd.DataFrame`** — the analytic boundary
  matrix. `pu_ids` is aligned to valid-cell order (defaults to `1..n_pu`);
  `build_boundary` validates `len(pu_ids) == n_pu` (else `ValueError`) — the count
  is checkable, the *order* alignment is the documented contract (as with the
  Zonation-smoothing coords contract, it can't be verified). For
  each valid cell, consider its **right** neighbor (shared vertical edge, length =
  `cell_height`) and **down** neighbor (shared horizontal edge, length =
  `cell_width`) — only these two, so each shared edge is emitted **once**, as
  `{id1 = this cell, id2 = the right/down neighbor}` (this mirrors `compute_boundary`'s
  positional `(ids[i], ids[j])` pairing; with the default sequential `1..n_pu` ids the
  neighbor is always the higher id, so `id1 < id2` — a property of the default ids, not
  a hard invariant a caller can rely on for arbitrary `pu_ids`):
  - if the neighbor is valid → a row `{id1, id2, boundary=edge_length}` and add
    `edge_length` to both cells' shared total;
  - self-boundary per cell = `2·(cell_width + cell_height) − shared_total`, emitted
    as an `{id1==id2, boundary=self}` row when `> 1e-10`.
  Returns columns `["id1", "id2", "boundary"]` — the exact shape `compute_boundary`
  produces, so a grid problem's `boundary` is a drop-in for the solvers/BLM.

## `ConservationProblem` change

Add one `kw_only` field (alongside the existing `probability`/`connectivity`):

```python
    grid: GridGeometry | None = field(default=None, kw_only=True)
```

Both derived-copy paths preserve `grid` for free: `copy_with` forwards every
dataclass field (`dataclass_fields`), and `clone()` is `copy.deepcopy(self)` (deep
-copies the whole object, including the numpy mask) — neither enumerates fields
explicitly, so no update is needed there.

`validate()` gains one `grid` branch, mirroring the existing `boundary`/`probability`
/`connectivity` branches: when `grid is not None`, assert `grid.n_pu ==
len(planning_units)` (append an error otherwise). This is the checkable half of the
positional-alignment contract — it catches an S2 ingestion that builds the grid from a
different mask than `planning_units` (the row *order* stays a documented, uncheckable
contract). `GridGeometry` is also re-exported from `models/__init__.py` for
discoverability parity with `ConservationProblem`.

No other model change in S1 — `has_geometry` stays vector-only (grid-aware mapping is
S4), and solvers are untouched (they read the derived `boundary`/matrix DataFrames).
(`GridGeometry` is treated as immutable — don't mutate `mask` after construction, since
`n_pu` / `valid_cells` recompute from it and would then disagree with `planning_units`.)

## Testing strategy (TDD, parity-anchored)

- **Construction + validation:** a valid `GridGeometry` builds; a non-2-D or
  non-bool mask, a `cell_width <= 0`, and an all-`False` mask each raise
  `ValueError`.
- **`valid_cells` row-major order + `n_pu`:** on a `3×3` mask with two cells masked
  out, `n_pu == 7` and `valid_cells()` lists the valid `(r,c)` top-down,
  left-to-right.
- **`cell_centroids` / `cell_bounds`:** hand-computed for a `2×2` grid
  (`x_min=0, y_max=2, cell=1`): centroids `(0.5,1.5),(1.5,1.5),(0.5,0.5),(1.5,0.5)`;
  bounds match.
- **Non-square cells (guards the direction→edge mapping).** With `cell_width=2,
  cell_height=3` (and a non-zero origin), assert centroids/bounds scale by the right
  axis, and that `build_boundary` gives horizontal-neighbor edges = `cell_width` and
  vertical-neighbor edges = `cell_height` (**not** swapped). This is the one test that
  catches a width/height swap or a dropped `*cell_width`, since every `w==h==1` test is
  blind to it.
- **Analytic boundary == shapely (the parity anchor).** Build a fully-valid `3×3`
  `GridGeometry`; construct a `GeoDataFrame` of `box(*b)` from `cell_bounds()` with
  ids `1..9`; assert `build_boundary(1..9)` equals `compute_boundary(gdf)` after
  sorting both by `(id1, id2)` (boundary values `approx`). Also: a **corner-masked**
  grid (exposed-edge self-boundary), a **center-masked** `3×3` (`mask[1,1]=False` — a
  fully-surrounded masked-out neighbor, the cleaner "next to a masked cell" case), and
  a **non-square** (`w != h`) grid — each equal to shapely, proving the self-boundary
  and edge-length handling match.
- **Hand-computed `2×2`:** `build_boundary` gives 4 shared-edge rows (each
  `cell_size`) + 4 self rows (each `2·cell_size`), matching the derivation.
- **`build_boundary` guards:** `len(pu_ids) != n_pu` raises `ValueError`; **duplicate**
  `pu_ids` raises `ValueError` (a standalone public method must not silently corrupt the
  shared-length accumulation).
- **Single valid cell:** `build_boundary([1])` gives one self row = the full
  perimeter, no shared rows.
- **`copy_with`/`clone` preserve `grid`:** `problem.copy_with(...)` keeps the same
  `grid`; `clone()` deep-copies it; a plain `ConservationProblem` has `grid is None`.
- **`validate()` grid branch:** a problem whose `grid.n_pu != len(planning_units)`
  yields a validation error; a matching one yields none.

**Target:** ~15 tests, `make check` green (0 ruff / 0 mypy), coverage ≥ 75%.

## Parity note

S1 adds a model field + a new boundary generator; it changes **no** solver or
objective math, and the analytic boundary is anchored to equal the existing
shapely `compute_boundary`. The 35.0 min-set anchor is untouched; `make check`
(which runs the parity tests) confirms it.

## References

Scoping: `docs/plans/2026-07-15-raster-grid-pus-scoping.md`. Boundary parity vs.
`spatial/boundary.py::compute_boundary`. Backward-compat rule: CLAUDE.md
("new optional fields must be `kw_only=True` with defaults").
