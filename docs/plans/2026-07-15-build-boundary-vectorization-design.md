# `GridGeometry.build_boundary` vectorization — design

**Date:** 2026-07-15
**Status:** Approved (brainstorm), pending spec review → implementation plan
**Scope:** Vectorize the S1 `GridGeometry.build_boundary` Python loops with numpy, so
`include_boundary` scales to the million-cell grids S3c can ingest. Companion to the
raster-grid effort (S1–S3a shipped, v0.17–v0.20). One method, one file
(`models/grid.py`); no signature or semantic change.

## Motivation

`GridGeometry.build_boundary` (`models/grid.py`) is a per-cell Python loop: a `cell_to_id`
dict of `n` tuple keys, a per-cell right+down neighbor loop appending to a `list` of `~3n`
dicts, a per-cell self-boundary loop, and a final `pd.DataFrame(list_of_dicts)`. Fine at
S1/S2 modest scale, but at the million-cell scale S3c ingests it is slow (minutes) and
memory-spiky (a multi-million-element dict list) — which is exactly why S3c defaults
`include_boundary=False` on the windowed path. This replaces the loops with O(n) numpy
array ops so BLM scales; S3c can then build the boundary at scale.

## Scope

Rewrite the body of `GridGeometry.build_boundary` only. **Unchanged:** the signature
(`build_boundary(pu_ids=None) -> pd.DataFrame`), the `len(pu_ids) == n_pu` and uniqueness
guards, the columns `["id1", "id2", "boundary"]`, the `> 1e-10` self-boundary threshold,
and the output **multiset** of rows (row *order* may change — see below). Out of scope:
`compute_boundary` (shapely), the S3c windowed `include_boundary` default (a separate call
site decision), diagonal/queen adjacency.

## The vectorized algorithm

`mask` is the `(nrows, ncols)` bool validity grid; `pu_ids` (length `n_pu`, aligned to
`valid_cells()` row-major order) are the planning-unit ids.

**Guards + setup use numpy, not `valid_cells()`.** The rewrite must *not* call
`self.valid_cells()` (it materializes a Python list of `n` `(r,c)` tuples — itself a
million-element scale bottleneck). Instead: `flat_valid = np.flatnonzero(mask.reshape(-1))`,
`n = int(flat_valid.size)` (== `self.n_pu`); the `pu_ids` default (`np.arange(1, n+1)`),
the `len(pu_ids) == n` guard, and the uniqueness guard use that `n`. Prefer
`len(np.unique(pu_ids)) != n` over the `set(...)` form for the uniqueness check to keep it
vectorized.

1. **id grid.** Build `id_grid` `(nrows, ncols)` with `pu_ids` scattered at the valid
   cells in row-major order (invalid cells hold an unused 0):
   ```python
   id_grid = np.zeros(mask.shape, dtype=np.int64)
   id_grid.reshape(-1)[np.flatnonzero(mask.reshape(-1))] = pu_ids
   ```
   Cell validity is always taken from `mask` (never from a sentinel), so invalid cells'
   `id_grid` values are never read. `id_grid` is a fresh C-contiguous `zeros`, so
   `id_grid.reshape(-1)` is a writable view; `reshape(-1)` and `flatnonzero` both flatten in
   **C-order (row-major)**, the same order as `valid_cells()`'s `np.nonzero`, so `pu_ids[i]`
   lands on the `i`-th valid cell regardless of `mask`'s memory layout.

2. **Right edges** (each valid cell and its valid right neighbor share a vertical edge of
   length `cell_height`):
   ```python
   both = mask[:, :-1] & mask[:, 1:]
   r_id1 = id_grid[:, :-1][both]
   r_id2 = id_grid[:, 1:][both]
   ```

3. **Down edges** (valid cell + valid down neighbor share a horizontal edge of length
   `cell_width`):
   ```python
   both = mask[:-1, :] & mask[1:, :]
   d_id1 = id_grid[:-1, :][both]
   d_id2 = id_grid[1:, :][both]
   ```

4. **Self-boundary** via exposed sides. A cell's left/right sides are vertical edges of
   length `cell_height`, its top/bottom sides horizontal edges of length `cell_width`; a
   side is *exposed* when the neighbor on that side is out-of-grid or invalid. Build the
   four "has a valid neighbor on this side" grids by shifting `mask`:
   ```python
   has_left = np.zeros_like(mask);  has_left[:, 1:]  = mask[:, :-1]
   has_right = np.zeros_like(mask); has_right[:, :-1] = mask[:, 1:]
   has_up = np.zeros_like(mask);    has_up[1:, :]   = mask[:-1, :]
   has_down = np.zeros_like(mask);  has_down[:-1, :] = mask[1:, :]
   self_grid = (
       (2 - has_left.astype(np.int64) - has_right.astype(np.int64)) * self.cell_height
       + (2 - has_up.astype(np.int64) - has_down.astype(np.int64)) * self.cell_width
   )
   self_vals = self_grid.reshape(-1)[np.flatnonzero(mask.reshape(-1))]  # PU order
   keep = self_vals > 1e-10
   s_ids = np.asarray(pu_ids)[keep]
   s_vals = self_vals[keep]
   ```
   This is algebraically identical to the loop's `perimeter - shared` — **identical up to
   floating-point rounding** (~1e-14 rel; the two forms regroup the IEEE-754 additions
   differently, far below the `1e-10` emit threshold): the loop's `shared[cell]` accumulates
   `cell_height` for each valid horizontal neighbor (left *and* right, since the left edge is
   emitted by the left cell but added to both) and `cell_width` for each valid vertical
   neighbor, so `perimeter - shared = 2(w+h) - (has_left+has_right)·h - (has_up+has_down)·w
   = (2-has_left-has_right)·h + (2-has_up-has_down)·w`.

5. **Assemble.**
   ```python
   id1 = np.concatenate([r_id1, d_id1, s_ids])
   id2 = np.concatenate([r_id2, d_id2, s_ids])
   boundary = np.concatenate([
       np.full(r_id1.size, self.cell_height),
       np.full(d_id1.size, self.cell_width),
       s_vals,
   ])
   return pd.DataFrame({"id1": id1, "id2": id2, "boundary": boundary})
   ```
   Empty sub-arrays concatenate cleanly (a single cell → no edges, one self row = full
   perimeter). `id1`/`id2` are int64 (from `id_grid`/`pu_ids`); `boundary` is float64.

## Correctness / parity

The output **multiset** of rows is identical to the loop version; only the row *order*
changes (all right-edges, then all down-edges, then all self-rows — vs the loop's
per-cell right-then-down-then-self). Nothing depends on row order: `ProblemCache` and
`compute_boundary`-shaped consumers accumulate order-independently, and every existing
`build_boundary` test is order-independent (the shapely-parity tests `sort_values(["id1",
"id2"])`; the hand-computed tests check sets/counts; the single-cell test has one row). The
`id1 < id2` property for default `1..n` ids is preserved (right/down neighbors have higher
row-major ids). The existing S1 parity anchor (`build_boundary == compute_boundary` on
full / corner-masked / center-hole / non-square grids) is the guard; a new test asserts the
vectorized output equals a reference-loop implementation on random masks (multiset, sorted).

## Testing strategy (TDD)

- **Existing S1 tests stay green** (unchanged): the shapely-parity set, hand-computed 2×2,
  non-square, single-cell, len/uniqueness guards, center-hole.
- **Vectorized == reference loop:** a small in-test reference re-implements the old
  per-cell loop; on several random masks (incl. holes, a full grid, a 1×N strip, non-square
  cells, arbitrary non-sequential `pu_ids`), `build_boundary` equals the reference after
  `sort_values(["id1","id2"]).reset_index(drop=True)` (`assert_frame_equal`, `check_dtype
  =False`).
- **Scale smoke:** `build_boundary` on a moderately large grid (e.g. `200×200 = 40k` cells)
  returns the correct row count and runs fast — a supplementary check that no per-cell
  Python loop (e.g. a leftover `valid_cells()` / dict build) survives. Not a hard perf gate;
  a `bench` marker at larger scale is optional.
- **Anchor:** `make check` green (0 ruff / 0 mypy), coverage ≥ 75%.

## References

`models/grid.py::GridGeometry.build_boundary` (S1, shipped v0.17.0); parity vs
`spatial/boundary.py::compute_boundary`. Enables S3c windowed `include_boundary` at scale
(S3c, v0.19.0). Design docs: `2026-07-15-raster-grid-s1-*`, `...-s3c-*`.
