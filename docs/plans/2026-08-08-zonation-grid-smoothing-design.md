# Grid-convolution distribution smoothing at raster scale — design spec

**Date:** 2026-08-08
**Status:** draft (pending loop review + multi-agent design review)
**Provenance:** the deferral named in `rank_removal`'s own smoothing-cap error message
("raster-scale distribution smoothing (grid convolution) is a planned follow-up",
deferred at v0.31.0). Brainstorm approved: column-wise truncated convolution, new
`GridSmoothingSpec`, no 50k cap on the grid path, allclose-not-bitwise contract.

## 1. Problem

`SmoothingSpec` builds a dense n×n kernel — impossible beyond ~50k PUs (the shipped
cap). Grid problems (`problem.grid: GridGeometry`, v0.17+) have regular geometry
that turns the same negative-exponential kernel into a stationary 2-D convolution,
making smoothed CAZ/ABF feasible at ~1M cells. This is Zonation's native
"distribution smoothing" setting (2-D dispersal kernel over a raster; the Zonation
manual's convention `alpha = 2 / dispersal distance` — science review to verify and
decide whether we document it).

## 2. Goals / non-goals

**Goals**

- `GridSmoothingSpec(alpha, truncate=1e-9)` accepted by the existing
  `rank_removal(smoothing=...)` / `ZonationSolver(smoothing=...)` parameters
  (signatures widen to `SmoothingSpec | GridSmoothingSpec | None`); requires
  `problem.grid`, no PU-count cap.
- Math equivalence to the vector path: on grids where the truncation radius covers
  every pairwise distance, grid smoothing equals
  `SmoothingSpec(alpha, coords=grid.cell_centroids())` to `rtol≈1e-10`
  (**allclose, not bitwise** — convolution regroups sums; the established FP-contract
  discipline applies).
- Exact mass conservation over valid cells at ANY truncation (the normalizer uses
  the same truncated kernel, so each source's outgoing weights sum to exactly 1
  over its reachable valid cells — up to FP).
- Sparse-in, sparse-out: never materialize a dense `(n_pu, n_feat)` matrix; per
  feature column, work on one `(nrows, ncols)` grid plane. Smoothed columns keep
  exact compact support (source support ⊕ kernel footprint), so localized features
  stay sparse and the v0.31 engine's dirty-set locality survives smoothing.

**Non-goals (explicit)**

- No change to `SmoothingSpec` or its 50k cap (vector problems keep the exact
  dense-kernel path).
- No Shiny/UI exposure (the solver pass-through comes free; UI is a later phase).
- No other kernel shapes (negative-exponential only, matching the vector path;
  Gaussian etc. are future extensions).
- No edge-removal interaction (still the documented non-goal).

## 3. The math (normative)

Let `K(dx, dy) = exp(-alpha * sqrt((dx*cell_width)^2 + (dy*cell_height)^2))`,
truncated to the square window where `K >= truncate` (window radius in cells:
`rx = ceil(-ln(truncate) / (alpha*cell_width))`, similarly `ry`; window
`(2*ry+1, 2*rx+1)`; `K(0,0) = 1` always kept). Let `M` be the validity mask as
float {0,1} and `q_g` a feature column scattered onto the grid (zeros off-support
and at invalid cells).

Window radii are **clipped to the grid**: `rx = min(ceil(-ln(truncate) /
(alpha*cell_width)), ncols-1)` and likewise `ry` — offsets beyond the grid extent
contribute nothing, and an unclipped tiny-alpha radius would allocate an absurd
kernel (loop-review fix). **Truncation is window-bounded only**: every entry
inside the `(2ry+1, 2rx+1)` rectangle is kept (no per-entry zeroing of corner
values below `truncate`) — so when the window covers the whole grid the kernel is
IDENTICAL to the untruncated vector kernel, which is what makes §6's
grid-vs-vector equivalence exact-in-values rather than approximate.

1. `Z = M ⊛ K` (correlation == convolution; K symmetric) — per-cell sum of
   truncated kernel weights over valid cells. `Z >= 1` wherever `M` is 1 (the
   self-weight), so no zero division at any valid source.
2. `ratio = 0 everywhere; ratio[valid] = q_g[valid] / Z[valid]` — the division is
   masked (loop-review fix: at invalid cells far from any valid cell `Z == 0`,
   and a naive `q_g / Z` would make 0/0 = NaN which the convolution then spreads
   everywhere; `np.divide(..., where=M)` with a zero-initialized output, or
   equivalent, is mandatory). Then `out = ratio ⊛ K` evaluated at valid cells;
   invalid cells discarded.

This is exactly the vector path's `K_colnorm @ q` with the truncated kernel:
out_dest = Σ_src q_src · K(d)/Z_src. Conservation: Σ_dest∈valid out_dest =
Σ_src q_src · (Σ_dest∈valid∩window K / Z_src) = Σ_src q_src, exactly, because
`Z_src` IS that inner sum. Truncation therefore costs redistribution *reach*, not
mass.

**FFT dust and compact support:** FFT-based convolution writes ~1e-16·max noise
everywhere. Restore exact compact support by masking: `reach =
binary_dilation(support(q_g), structure=ones(window))` (scipy.ndimage); zero
everything outside `reach ∧ M`. Values inside are untouched → sparsity is exact,
conservation unharmed (out-of-reach true values are exactly 0 by construction).
Convolution backend: `scipy.signal.oaconvolve` (picks FFT/direct sensibly),
`mode="same"`.

## 4. Structure

- **`connectivity/smoothing.py`** gains the math:
  `smooth_distribution_grid(amounts_flat, grid, alpha, *, truncate=1e-9) ->
  np.ndarray` — takes ONE feature's per-PU amounts (length n_pu, PU order ==
  row-major valid-cell order, the S1 invariant), returns the smoothed length-n_pu
  column. Internals: scatter → Z (computed ONCE per call set — see spec below) →
  convolve → mask → gather.
  To avoid recomputing `Z` and the kernel per column, also expose
  `GridKernel` (private-ish helper class or precomputed tuple) OR give the
  function a multi-column form: `smooth_distribution_grid(amounts, grid, alpha,
  *, truncate)` accepting `(n_pu,)` or `(n_pu, m)` — matching
  `smooth_distribution`'s existing one-or-many contract. **Chosen: the
  one-or-many contract** (mirrors the sibling exactly); kernel + `Z` + dilation
  structure computed once per call, columns looped.
- **`zonation/smoothing.py`** gains the spec:

  ```python
  @dataclass(eq=False)
  class GridSmoothingSpec:
      alpha: float
      truncate: float = 1e-9
  ```

  `__post_init__`: `alpha > 0`, `0 < truncate < 1` else `ValueError`.
  `apply(amounts: np.ndarray, grid: GridGeometry) -> np.ndarray` — thin wrapper
  over `smooth_distribution_grid`, same one-or-many contract.
- **`zonation/rank_removal.py`** dispatch (isinstance) in the smoothing branch:
  - `GridSmoothingSpec`: `problem.grid is None` → `ValueError` ("grid smoothing
    requires a grid problem; use SmoothingSpec for vector problems"). No 50k cap.
    Build the smoothed matrix **column-wise from the existing CSR/CSC** (never a
    dense `(n_pu, n_feat)`): for each feature column (CSC), densify that one
    column to length n_pu (cheap: zeros + scatter from the CSC slice), smooth,
    keep the nonzero (row-index, value) pairs; assemble one CSC from the
    concatenated per-column results, convert `.tocsr()`; `eliminate_zeros()` as
    usual. Then the unchanged engine.
  - `SmoothingSpec`: existing dense path + 50k cap, untouched.
- **`solvers/zonation_solver.py`**: type widening only
  (`SmoothingSpec | GridSmoothingSpec | None`); the `smoothed`/alpha metadata
  markers already duck-type (`.alpha` exists on both).

## 5. Validation and errors

- `GridSmoothingSpec` on a problem without `grid` → `ValueError` (message above).
- `SmoothingSpec` behavior byte-for-byte unchanged, including the 50k cap.
- Amount validation: `_validate_inputs` already runs on raw amounts before any
  smoothing (v0.31 ordering) — unchanged. Smoothed values are nonnegative by
  construction (nonneg inputs × nonneg kernel), preserving the engine's
  nonnegativity invariants (incl. the heap's monotonicity requirement).
- Degenerate windows: huge alpha → minimum radius-1 window per axis (ceil of a
  positive is ≥ 1) with smoothly vanishing off-centre weights — near-identity, no
  special-casing; single-row/-column grids clip that axis' radius to 0. A 1×1
  grid gives the true identity `[[1]]`. All legal, tested.

## 6. Equivalence & FP contract

- **Grid vs vector:** on a small grid whose truncation window covers all pairwise
  offsets, `smooth_distribution_grid(q, grid, alpha)` ==
  `smooth_distribution(q, cdist(centroids), alpha)` to
  `assert_allclose(rtol=1e-10, atol=1e-13)`. NOT bitwise (different summation
  order/grouping). Masked-hole grids included (the vector path only ever sees
  valid cells; the grid path must exclude invalid cells from both `Z` and `out` —
  the `M` factor and the final gather do exactly that).
- **Conservation:** `out.sum() == q.sum()` to `rtol=1e-12`, at aggressive
  truncation too (e.g. `truncate=1e-2`).
- **Engine integration:** with a `GridSmoothingSpec`, `rank_removal` heap==batch
  bitwise at warp=1 (both consume the same smoothed CSR — nothing about the
  v0.32 §7 contract changes). The test-local dense oracle does NOT learn
  GridSmoothingSpec; instead §7.5's self-consistency check compares against a
  problem rebuilt from the manually-smoothed matrix — orders equal exactly (same
  matrix values through the same engine).
- **Compact support:** for a single-cell source, `out`'s support == the kernel
  window ∩ mask, and cells outside are EXACTLY 0.0 (the dilation mask guarantees
  it despite FFT).

## 7. Testing

New file `tests/pymarxan/connectivity/test_grid_smoothing.py` (basename unique)
for the math; zonation additions appended to `test_rank_removal_scale.py`
(dispatch/integration) and `tests/pymarxan/zonation/test_smoothing.py`
(spec-class validation, mirroring SmoothingSpec's tests).

1. Grid-vs-vector allclose (full-window small grid; with and without masked
   holes; anisotropic `cell_width != cell_height`; multi-column `(n_pu, m)`).
2. Conservation at truncate ∈ {1e-9, 1e-2}; single-cell-source compact support
   with exact zeros outside window∩mask.
3. Huge-alpha near-identity (self-weight dominates; window is radius-1);
   single-column grid (one axis radius 0); `truncate` bounds validation; `alpha`
   validation.
4. Dispatch: GridSmoothingSpec + vector problem raises; SmoothingSpec on grid
   problem still works (via coords) — both specs coexist; >50k-cell grid problem
   accepted with GridSmoothingSpec (300×200 mask, trivially sparse feature;
   fast because column-wise).
5. Engine: rank_removal(grid problem, GridSmoothingSpec) heap==batch bitwise
   (warp=1) and equals the manually-smoothed-matrix run (build the smoothed CSR
   in-test via smooth_distribution_grid, feed a problem constructed from it,
   compare orders).
6. ZonationSolver(smoothing=GridSmoothingSpec) end-to-end on a grid problem;
   metadata markers intact.
7. Bench (bench-marked): 300×300 grid, a few localized features, GridSmoothingSpec
   smoothing + rank_removal warp=n//1000 — asserts completion under a generous
   budget and that the smoothed CSR nnz stays ≪ n_pu*n_feat (sparsity claim).
8. `make check` green; parity anchor untouched.

## 8. Performance envelope (claims for review)

- Per column: one `oaconvolve` on the `(nrows, ncols)` plane (`O(G log G)` FFT or
  better), one dilation, scatter/gather `O(G)`. `Z` and kernel once per apply.
  1M-cell grid × 20 features: tens of convolutions of a 1M-cell plane — seconds.
- Memory: a few grid planes at a time (`8 MB` each at 1M cells) + the smoothed
  sparse columns. No n×n kernel, no dense `(n_pu, n_feat)`.
- Smoothed nnz growth: source nnz × window area upper bound; localized features
  with moderate alpha stay far below dense.

## 9. Files touched

- `src/pymarxan/connectivity/smoothing.py` — `smooth_distribution_grid` (+ shared
  kernel/Z helper internals).
- `src/pymarxan/zonation/smoothing.py` — `GridSmoothingSpec`.
- `src/pymarxan/zonation/rank_removal.py` — dispatch branch + type widening;
  docstring paragraph.
- `src/pymarxan/solvers/zonation_solver.py` — type widening only.
- `src/pymarxan/zonation/__init__.py` (+ package `__init__` re-exports as the
  siblings do) — export `GridSmoothingSpec`.
- Tests per §7; `CHANGELOG.md` Added entry → v0.33.0.

## 10. Risks / open questions for review

- The `Z`-normalization direction (column-normalized == per-SOURCE outgoing mass)
  must match `smooth_distribution` exactly — grounding should verify against the
  vector implementation numerically, not by reading.
- `oaconvolve` availability/behavior across the pinned scipy version; `mode="same"`
  centering with even/odd windows (windows are always odd by construction —
  `2r+1`).
- PU-order invariant: scatter/gather must use the S1 row-major valid-cell order
  (`np.flatnonzero(mask.reshape(-1))`) — the same order `build_pu_feature_csr`
  rows follow; a mismatch is silent wrong ranking (the Phase-C review's classic).
- Science: is `alpha = 2/dispersal distance` the right convention note; is
  column-normalized (mass-conserving) smoothing what Zonation actually does, or
  does Zonation use un-normalized kernels (Moilanen 2004/2005; the vector path
  chose normalize=True in Phase C — consistency matters more than either choice,
  but the docs should state what Zonation does).
