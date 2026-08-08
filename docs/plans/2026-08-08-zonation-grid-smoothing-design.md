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
making smoothed CAZ/ABF feasible at ~1M cells. Zonation's native "distribution
smoothing" is exactly this transform in kernel shape (2-D negative-exponential
over the raster, FFT-implemented; Moilanen et al. 2005, doi:10.1098/rspb.2005.3164)
— with one deliberate pymarxan deviation the science review settled from primary
sources: **Zonation accumulates the raw kernel (unnormalized); pymarxan
additionally conserves mass per source** (the Phase-C vector-path choice, kept for
internal consistency). The deviation is ranking-inert wherever a source's
truncated window fits inside the valid mask (CAZ/ABF scores are invariant to
per-feature constant scaling) and differs only near raster edges/mask holes — an
*edge-corrected variant*, documented as such. The manual convention
`alpha = 2 / mean dispersal distance` is verified real (the 2-D kernel's
mean-dispersal identity; Westwood et al. 2020, doi:10.3390/d12020061 uses exactly
2/d) and is documented as convention-not-definition with the inverse-CRS-units
caveat. Zonation 5 itself truncates the kernel tail for tiled FFT (Moilanen et
al. 2022, doi:10.1111/2041-210X.13819) — our window truncation is their own
practice.

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

**FFT dust, compact support, and the nonnegativity clamp:** FFT-based convolution
writes ~1e-16·max noise everywhere. Restore exact compact support by masking:
`reach = support(q_g) ⊕ window` — implemented as a box convolution thresholded at
0.5 (equivalent to `binary_dilation` with a full-window structure but O(G log G);
threshold margin measured at ~11 orders of magnitude); zero everything outside
`reach ∧ M`. Then **clamp: `np.maximum(smoothed, 0.0, out=smoothed)`** —
design-review #1 (HIGH, found independently by two lenses): true in-window corner
values ~`q·truncate^√2` can sink below the FFT noise floor (which scales with
total landscape mass), producing negative dust INSIDE the reach mask —
reproduced at the DEFAULT truncate with a 1e12-amplitude blob plus a remote unit
source. Unclamped negatives would enter the engine unvalidated (raw-amount
validation runs pre-smoothing) and invert the warp=1 heap's monotonicity
invariant → silent heap≠batch divergence. Clipped mass is at FFT-noise scale,
far inside the conservation tolerance (verified). Convolution backend:
`scipy.signal.oaconvolve`, `mode="same"` (windows always odd).

## 4. Structure

- **`connectivity/smoothing.py`** gains the math:
  `smooth_distribution_grid(amounts_flat, grid, alpha, *, truncate=1e-9) ->
  np.ndarray` — takes ONE feature's per-PU amounts (length n_pu, PU order ==
  row-major valid-cell order, the S1 invariant), returns the smoothed length-n_pu
  column. Internals: scatter → Z (computed ONCE per call set — see spec below) →
  convolve → mask → gather.
  One-or-many contract (`(n_pu,)` or `(n_pu, m)`), mirroring the sibling — but
  note (design-review #4): the many-column form materializes dense in/out, so
  the raster-scale integration path deliberately calls `apply` **per column**,
  recomputing kernel + `Z` each call for memory flatness. Overhead is bounded
  ~1.5× in convolutions (3 per column vs 2 + one shared `Z`); a private
  kernel/Z cache helper is deferred until profiling shows the `Z` conv matters.
- **`zonation/smoothing.py`** gains the spec:

  ```python
  @dataclass
  class GridSmoothingSpec:  # no eq=False: two float fields, auto __eq__ works
      alpha: float
      truncate: float = 1e-9
  ```

  `__post_init__`: `alpha > 0`, `0 < truncate < 1` else `ValueError`.
  `apply(amounts: np.ndarray, grid: GridGeometry) -> np.ndarray` — thin wrapper
  over `smooth_distribution_grid`, same one-or-many contract.
- **`zonation/rank_removal.py`** dispatch — POSITIVE isinstance on both spec
  types with a final `else: raise TypeError(f"unsupported smoothing spec: ...")`
  (design-review #5: a negated-isinstance cap guard would silently route any
  future third spec type through the dense cap + dense apply). A fully
  spec-polymorphic `build_smoothed_csr(problem)` method was considered and
  rejected for this phase: SmoothingSpec must stay byte-for-byte unchanged and
  the existing cap tests monkeypatch against `rank_removal` itself. Branches:
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
  smoothing (v0.31 ordering) — unchanged. Smoothed values are nonnegative **by
  construction and clamped against FFT roundoff** (§3's clamp — load-bearing for
  the heap's monotonicity invariant, since post-smoothing values bypass raw
  validation; design-review #1).
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
   with exact zeros outside window∩mask; **nonnegativity regression** with the
   review's high-dynamic-range construction (large plane, 1e12-amplitude blob +
   remote unit source, `(out >= 0).all()`); **truncated-kernel explicit oracle**
   (two-convolution result vs an independently built window-bounded
   column-normalized kernel matrix — pins the formulation beyond the full-window
   case; both verifying lenses measured ≤6.3e-16).
3. Huge-alpha near-identity (self-weight dominates; window is radius-1);
   single-column grid (one axis radius 0); `truncate` bounds validation; `alpha`
   validation.
4. Dispatch: GridSmoothingSpec + vector problem raises; SmoothingSpec on grid
   problem still works (via coords) — both specs coexist; >50k-cell grid problem
   accepted with GridSmoothingSpec (300×200 mask, trivially sparse feature;
   fast because column-wise). The EXISTING `test_smoothing_capped_at_vector_scale`
   gains one assertion pinning the new cap message's GridSmoothingSpec redirect
   (no duplicate test — design-review #6).
5. Engine: rank_removal(grid problem, GridSmoothingSpec) heap==batch bitwise
   (warp=1) and equals the manually-smoothed-matrix run (build the smoothed CSR
   in-test via smooth_distribution_grid, feed a problem constructed from it,
   compare orders).
6. ZonationSolver(smoothing=GridSmoothingSpec) end-to-end on a grid problem;
   metadata markers intact.
7. Bench (bench-marked): 300×300 grid, a few localized features,
   `GridSmoothingSpec(alpha=2.0)` (radius ~11 — design-review #3: the earlier
   alpha=0.5/1e-9 parameters give a radius-42 window and 36%-dense output, which
   cannot witness sparsity) + rank_removal warp=n//1000 — asserts completion
   under budget AND `smoothed nnz < 0.25 * n_pu * n_feat`.
8. `make check` green; parity anchor untouched.

## 8. Performance envelope (claims for review)

- Per column (integration path): three plane convolutions (smoothing, Z, reach —
  Z/kernel recomputed per column, the ~1.5× memory-flatness trade of §4) +
  scatter/gather `O(G)`. Measured by grounding review: 0.35–0.61 s per 1M-cell
  column; 20 columns 6.3 s; the Task-4 bench emulation 18.5 s vs its 120 s
  budget.
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
- `src/pymarxan/zonation/__init__.py` — export `GridSmoothingSpec` beside
  `SmoothingSpec`. No top-level or connectivity re-export: the siblings have
  none (verified — `pymarxan/__init__.py` exports only `__version__`).
- Tests per §7; `CHANGELOG.md` Added entry → v0.33.0.

## 10. Review outcome (2026-08-08)

Reviewed by the four-perspective workflow (`wf_f313e9e2-f7d`); synthesis in
`2026-08-08-zonation-grid-smoothing-review.md`. All prior open questions closed:

- `Z`-direction: **verified numerically** — the vector oracle is per-SOURCE
  outgoing normalization (the per-destination reading differs by 0.19 and breaks
  conservation); the sibling `smooth_distribution` docstring stated it backwards
  and gets a one-line drive-by fix.
- `oaconvolve`: present in the pinned scipy 1.17.1; `mode="same"` verified incl.
  window-larger-than-plane; reach threshold margin ~11 orders.
- PU-order invariant: verified (`flatnonzero(mask.ravel())` == valid_cells
  row-major == CSR rows, bitwise).
- Science: Zonation smooths UNNORMALIZED (ours is a documented edge-corrected
  variant, ranking-inert away from edges/holes); alpha = 2/d convention verified
  and documented; Zonation 5's own kernel truncation cited.
- NEW from review (the one HIGH): FFT negative dust inside the reach mask →
  the §3 clamp, regression-tested.
