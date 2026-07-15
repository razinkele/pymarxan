# Raster-grid PUs — S2 multi-agent design review — synthesis

**Date:** 2026-07-15
**Reviewed:** `2026-07-15-raster-grid-s2-design.md` + `...-s2-implementation.md`
**Lenses:** Architect, Codebase-grounding, Independent re-design. (Scientific-accuracy
skipped — S2 is data ingestion/geometry, no literature claims.)

## Verdict

**Architecturally sound; one HIGH correctness bug in the alignment tolerance, else refinements.**
Grounding verified all 7 factual claims live in the shiny env (constructor field order +
kw_only `grid`; `build_pu_feature_matrix` row/col order + summing, round-trip anchor holds;
`GridGeometry` signature + `np.nonzero` order matching `valid_cells()`; affine/rasterio API
incl. rotated-transform round-trip; no import cycle; `@pytest.mark.spatial` registered;
nodata `==` comparison). The independent re-design converged on every hard call
(transform→grid mapping incl. `cell_height=-e` and south-up rejection, validity precedence,
nodata-vs-0, positional alignment) — strong signal — and two of the plan's choices were
judged better than the reviewer's instinct.

## Findings folded in

- **HIGH — `_transforms_close` tolerance too tight (false-rejects aligned rasters).** The
  absolute `tol=1e-9` on all six affine coefficients is ~1 ULP at a projected origin
  (EPSG:3035 easting 4.3e6 → ULP ≈9.3e-10; web-mercator ~2e7 → 3.7e-9 > tol). Two truly
  co-registered rasters whose origin was recomputed by different tooling differ by a few
  ULPs and get rejected — contradicting the design's own tolerance rationale, hidden
  because every test uses origin 0. *Folded:* scale the tolerance to cell size (compare
  each coefficient within `tol * max(|cell_width|, |cell_height|, 1)`), + a large-origin
  (~4.3e6) regression test that must NOT raise on a ~1e-6 m difference and MUST raise on a
  half-pixel shift.

- **MEDIUM — cost-nodata inside an explicit mask silently defaults to 1.0.** Reachable via
  the `mask`+`cost` combination: a masked-in cell with nodata cost silently gets `1.0`,
  mixing real and magic costs. *Folded:* `warnings.warn` naming the count of valid cells
  that fell back to default cost, + a test. (In the cost-footprint case those cells are
  already excluded, so the warning only fires for the genuine mask+cost hole.)

- **MEDIUM — design overclaims "solvers don't call `has_geometry`."** `separation.py` calls
  it; a grid problem carries centroids via `grid.cell_centroids()` but separation's
  coordinate resolver can't reach them, so an active separation feature would raise. Inert
  in practice — S2's `features` table has no `sepdistance`/`sepnum` columns, so separation
  is off unless the user adds them. *Folded:* corrected the design claim; documented that
  separation on grid problems needs a `cell_centroids()` fallback (a later/S4 solver
  change), so it's explicitly unsupported for now rather than silently broken.

- **LOW — test/coverage + doc:** added `assert problem.validate() == []` to the round-trip
  test; a holey-mask cross-layer test (non-symmetric validity mask + distinct cost +
  distinct per-feature values, to catch a cross-layer transpose/misindex); an all-None-CRS
  build test; a docstring note that feature amounts must be non-negative (values `<= 0` are
  dropped, matching the sparse `> 0` rule) and that `target=0.0`/`spf=1.0` are placeholders
  set later; and switched the south-up test transform to a non-identity `Affine(1,0,100,
  0,1,0)` (the identity-flip emits a `NotGeoreferencedWarning` and is ambiguous).

## Noted, no action (with reason)

- `from_arrays` co-located with a top-level `rasterio` import isn't importable in a
  rasterio-less env — but the siblings (`feature_intersection.py`/`cost_surface.py`) already
  top-import rasterio and `ConservationProblem` hard-requires geopandas, so `spatial/` is a
  geo-extras module regardless. The real benefit (rasterio-free *logic*, unit-tested with
  plain numpy) stands. Top-import kept for consistency; design wording already accurate.
- Single global `nodata` sentinel in `from_arrays` (per-array nodata is a `from_rasters`-only
  capability via NaN normalization): documented as a convenience, NaN is canonical. No change.
- Multi-band stack re-opens the file per band: negligible at S2 scale. No change.
- `from_*` (not `read_*`/`load_*`) naming: correct — no write-side pair, unlike the `io/`
  format readers; added a one-line design note so it isn't "fixed" later.
