# Raster-grid PUs — S3c: windowed raster ingestion — design

**Date:** 2026-07-15
**Status:** Approved (brainstorm), pending spec review → implementation plan
**Scope:** S3c only — windowed reads in `from_rasters` so large rasters ingest without
loading full arrays. Builds on S2 (`spatial/raster.py`, shipped v0.18.0). The other S3
pieces — **S3a** (sparse `ProblemCache` matrix, the SA/greedy memory wall) and **S3b**
(MIP-at-scale guard) — are deferred; order is S3c → S3a → S3b. Scoping:
`2026-07-15-raster-grid-pus-scoping.md`.

## Motivation

S2's `from_rasters` reads each raster fully (`src.read(band)` → a `(H×W)` float array),
then hands full arrays to `from_arrays`. For a large grid this is the **first** scale
ceiling: a 10M-cell × 50-feature stack is ~4 GB of dense reads and OOMs *at ingestion*,
before any solver runs. S3c removes that ceiling by reading in **windows** (tiles), so
peak memory is `O(one window + the bool validity mask + the sparse result)` instead of
`O(H×W×n_feat×8)`. It touches only reading — not the objective/delta hot loop — so there
is no 35.0-parity risk. (It does not by itself let SA/greedy run at scale; that is S3a.
It does immediately help the MIP solver, which already reads `pu_vs_features` sparsely.)

## Scope (S3c)

- A **windowed two-pass path** in `from_rasters`, selected by a new
  `window_size: int | "auto" | None` parameter, producing the **same**
  `ConservationProblem` as the full-array path.
- Memory target: rasters whose dense feature stack would be multi-GB but whose `(H×W)`
  **bool** validity mask (1 byte/cell — ~100 MB at 10⁸ cells) and sparse tables fit in
  workstation RAM.

Out of scope: the sparse `ProblemCache`/solver path (S3a), the MIP guard (S3b),
out-of-core / memory-mapped masks (past ~10⁸ cells), exactextract, reprojection,
per-feature CRS. The pure `from_arrays` core is unchanged.

## The switch

`from_rasters(..., window_size: int | Literal["auto"] | None = "auto")`:
- `None` → the current full-array path (read each raster whole, delegate to
  `from_arrays`). Backward-compatible behaviour.
- `int` → windowed path with square tiles of that side length.
- `"auto"` (default) → estimate the dense size `H · W · n_feat · 8` bytes from the first
  raster's shape + feature count (metadata only, no read); use windowed (default tile,
  a module constant, e.g. 1024) when it exceeds a threshold (`_WINDOW_AUTO_BYTES`, e.g.
  512 MiB), else the full path. So small rasters keep the exact S2 behaviour and large
  ones auto-switch.

`include_boundary` becomes `bool | None = None`: `None` resolves per path — `True` on the
full-array path (unchanged S2 behaviour), `False` on the windowed path (the Python-loop
`build_boundary` is a scale bottleneck — see "Boundary at scale"). An explicit `True`/
`False` is always honoured, so a windowed caller can still opt into the boundary.

## The windowed algorithm (two passes)

All rasters are opened once via `contextlib.ExitStack` (open `DatasetReader`s held for
both passes — no per-window reopen), with a **metadata-only** alignment check (transform
tolerant per S2's cell-size-scaled `_transforms_close`, shape + CRS exact) and the S2
`_require_north_up` guard on the reference (first feature) raster. Tiles cover `(H, W)`
in row-major order with side `tile`; a `rasterio.windows.Window(col_off, row_off, w, h)`
per tile (edge tiles are **partial** — width/height clamped to the raster bounds); band
reads are `src.read(band, window=win)`. Each window read **normalizes the source nodata
to NaN** (`np.where(arr == src.nodata, np.nan, arr)`, exactly as S2's `_read`), so
`_nodata_mask` operates on NaN uniformly and matches the full-array path bit-for-bit.

**Pass 1 — build the validity mask.** Preallocate `mask = np.zeros((H, W), bool)`. For
each tile, compute window validity by the S2 precedence and write it into `mask` at the
tile's slice:
- `mask_raster` → read its window: `(m != 0) & ~nodata`;
- elif `cost_raster` → read its window: `~nodata`;
- else feature-union → OR of `~nodata` across every feature window.

Then `flat_valid = np.flatnonzero(mask.ravel())` — an **`(n_pu,)`** int array (sized by
*valid cells*, a few MB, **not** `H×W`), sorted, whose position `i` is PU id `i+1`. This
is what reconciles block-wise reads with the **global row-major PU order** without
materializing a full `(H×W)` index. `n_pu == flat_valid.size`; raise if `0`.

**Pass 2 — sample cost/status + emit feature rows.** `cost_vals = ones(n_pu)`,
`status_vals = zeros(n_pu, int)`. For each tile with any valid cell:
- local valid positions `vr, vc = np.nonzero(mask[tile_slice])`; global flat indices
  `gflat = (row_off + vr) · W + (col_off + vc)`;
- **PU indices** `pu_idx = np.searchsorted(flat_valid, gflat)` — each processed cell is
  valid so `gflat ∈ flat_valid`, and `searchsorted` returns its exact row-major rank
  (0-based); PU id `= pu_idx + 1`. Each valid cell lives in exactly one tile → no
  overwrite;
- **cost**: if `cost_raster`, `cost_vals[pu_idx] = window values` with nodata → `1.0`
  (and the S2 "nodata cost at a valid cell" warning accumulates a count across tiles,
  emitted once at the end);
- **status**: if `status_raster`, validate integer-valued in `{0,1,2,3}` (S2 rule) and
  `status_vals[pu_idx] = iv`;
- **features**: for each feature, read its window, `keep = ~nodata & (amount > 0)`, append
  a `(species, pu_ids[keep], amount[keep])` frame.

Finally build `GridGeometry(x_min, y_max, cell_width, cell_height, mask, crs)` from the
reference transform, `planning_units` (`id 1..n_pu`, `cost_vals`, `status_vals`),
`features` (ids = sorted feature keys, placeholder `target=0.0`/`spf=1.0`),
`pu_vs_features` (concat of the frames), the analytic boundary via
`grid.build_boundary(pu_ids)` when `include_boundary` (O(n_pu)), and return the
`ConservationProblem`.

**Re-reads (I/O, not memory).** Building the mask in pass 1 and sampling in pass 2 means
some rasters are read twice: feature-union validity re-reads every feature (pass 1 mask +
pass 2 amounts); cost-driven validity re-reads the cost raster (pass 1 mask + pass 2
values). `mask`-driven validity reads each feature/cost only once (pass 2). This is an I/O
cost, not a memory one — peak memory is unchanged.

**Boundary at scale (a real limitation).** With `include_boundary=True` the path calls S1's
`grid.build_boundary`, which is a **Python loop over every valid cell** (a `dict` of `n_pu`
tuple keys + a list of ~`3·n_pu` row dicts before the DataFrame) — fine at S1/S2 modest
scale, but slow and memory-spiky at the million-cell scale S3c targets, and it partly
undermines S3c's goal. S3c therefore treats this honestly: `include_boundary` stays a
parameter (default `True`, so small/auto-full problems are unchanged), but the windowed
path **defaults `include_boundary` to `False` when `window_size` resolves to windowed**
(BLM off unless explicitly requested), and the docstring notes that requesting the boundary
on a very large grid is expensive until `build_boundary` is vectorized — a deferred
companion (natural alongside S3a). A multi-band stack (`{1:(p,1), 2:(p,2)}`) opening the
same path twice via the `ExitStack` is a negligible, accepted duplication.

## Memory model

Peak memory ≈ the `(H×W)` bool mask + `flat_valid` (`n_pu` int64) + the growing sparse
`pu_vs_features` (nnz) + one window per open raster. Independent of `n_feat × H × W`. The
bool mask is the largest fixed structure (~100 MB at 10⁸ cells); beyond that is
out-of-scope (S3-later). `searchsorted` per tile is `O(k log n_pu)` for `k` window-valid
cells — negligible vs the reads.

## Parity anchor (the S3c correctness property)

The windowed path must produce a `ConservationProblem` **semantically identical** to the
full-array path: same `GridGeometry.mask`, same `planning_units` (ids/cost/status), same
`features`, same `build_pu_feature_matrix()`, and the same `pu_vs_features` **after
sorting by `(species, pu)`** (row *order* differs — windowed emits per-tile-per-feature,
`from_arrays` emits per-feature-row-major — but the multiset of rows is identical, which
is all the model consumes: `build_pu_feature_matrix` sums, `validate` is order-free).
Verified on small `MemoryFile` rasters with a tiny forced `window_size` (e.g. `2` over a
`5×5` grid, so tiles straddle rows and exercise the row-major-rank mapping).

## Testing strategy (TDD)

- **Windowed == full (the anchor).** A `5×5` grid, 2 feature rasters (+ a masked cell):
  `from_rasters(..., window_size=2, include_boundary=True)` yields the same PU ids,
  `mask`, `build_pu_feature_matrix()`, `cost`/`status`, `boundary`, and `pu_vs_features`
  (sorted by `species,pu`) as `from_rasters(..., window_size=None, include_boundary=True)`.
  (`include_boundary` is forced equal on both sides so the default-resolution difference
  — windowed→None vs full→built — doesn't confound the comparison.) The core guarantee.
- **PU-order mapping across tiles.** With `window_size=2` on a `5×5` grid, assert PU ids
  are `1..n` in global row-major order (a cell in tile (0,1) has a *lower* id than a cell
  in tile (1,0)), proving `searchsorted(flat_valid, gflat)` reproduces row-major rank.
- **Validity precedence, windowed.** mask-, cost-, and feature-union-driven validity each
  match the full path under windowing (including a nodata hole that spans a tile edge).
- **cost/status, windowed.** A nodata-cost cell under a mask warns and defaults to 1.0; a
  status `2` cell keeps status 2; an out-of-range/non-integer status raises — same as S2.
- **`"auto"` switch.** A tiny raster with `window_size="auto"` takes the full path and
  equals `window_size=None`; a monkeypatched-low `_WINDOW_AUTO_BYTES` forces the windowed
  path and still equals the full path (no separate behaviour).
- **Alignment/guards still fire under windowing.** A transform/CRS mismatch and a
  rotated/non-north-up reference raster raise before any window is read (metadata check).
- **`window_size` larger than the grid** degenerates to a single tile == full path.
- **`include_boundary` resolution:** windowed path defaults `boundary` to `None` (not
  built); windowed with explicit `include_boundary=True` builds it and it equals the full
  path's; full path still defaults to a built boundary.

**Target:** ~11–15 tests, `make check` green (0 ruff / 0 mypy), coverage ≥ 75%.

## Parity note

S3c changes ingestion only — no solver/objective math. The 35.0 min-set anchor is
untouched. The windowed==full anchor guarantees the ingested problem is byte-for-byte the
same one the S2 path (already round-tripped against the input arrays) produces.

## References

Scoping: `2026-07-15-raster-grid-pus-scoping.md` (S3). S2: `spatial/raster.py`
(`from_arrays`/`from_rasters`, `_read`, `_transforms_close`, `_require_north_up`,
`_nodata_mask`), shipped v0.18.0. rasterio windowed reads:
`rasterio.windows.Window` + `src.read(band, window=...)`. `contextlib.ExitStack` for
holding open datasets. `scipy` (already a dep) is not needed here — S3c is numpy + rasterio.
