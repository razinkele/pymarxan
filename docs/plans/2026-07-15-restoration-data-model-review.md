# restoration data model — design review synthesis

**Date:** 2026-07-15
**Lenses:** Codebase-grounding (RAN 17/17 Task-1 tests), Architect + Independent-redesign. (No
scientific lens — the science is settled in the MESH review; this piece is data-model + ingestion.)

## Verdict

**Design sound, ready to build — no CRITICAL/HIGH.** Grounding ran the verbatim Task-1 code (17/17
pass), verified the row-major PU-order contract on a non-rectangular masked grid, confirmed every
reused S2 helper signature, the rasterio-free import contract, and parity untouched. The architect
independently arrived at the same model shape (standalone `@dataclass(eq=False)` composing
`GridGeometry`; full-length bool `restored ⊆ restorable`; the same method set; rasterio-free core).
Fold the two MEDIUMs + polish below.

## Fixes to fold

- **M1 (MEDIUM) — the validity mask silently sets `A_total` and silently drops restorable/cost data.**
  `from_arrays` defaults the study region to `existing_habitat`'s non-nodata footprint, which becomes
  `GridGeometry.mask` and hence `compute_mesh`'s `A_total = n_pu·cell_area`. A `restorable` (or
  `cost`) cell that is nodata in `existing_habitat` is silently excluded — un-restorable *and*
  uncounted. *Fix:* after building `valid`, **warn** when `restorable`/`cost` carry non-nodata data
  outside `valid` (dropped candidates); document that the validity mask sets `A_total`; add a
  `mask_raster=` passthrough to `from_rasters` (symmetry with `from_arrays(mask_array=)`, mirroring
  S2's `from_rasters` mask support). *(Not adding a `study_region="union"` option — `mask_array` /
  `mask_raster` already let a caller override; YAGNI.)*
- **M2 (MEDIUM) — duplicated `_nodata_mask` + reach into `spatial.raster` privates.** The full fix
  (promote `_nodata_mask`/`_read`/`_read_aligned`/`_require_north_up` into a rasterio-free-importable
  `spatial/_raster_io.py`) is a refactor of working S2 code — out of scope for a data-model piece and
  a real regression risk. *Lighter mitigation (taken):* keep the inline `_nodata_mask` + lazy private
  import, and add a **regression test** that pins (a) the reused `spatial.raster` helper signatures
  are callable as used and (b) the inline `_nodata_mask` matches `spatial.raster._nodata_mask`
  behaviourally (NaN + sentinel + int-dtype). Note the shared-module extraction as a deferred
  cleanup.
- **L1 (accept) — use `spatial.raster._read_aligned` directly.** Grounding confirmed it exists with
  exactly `_read_aligned(path, band, label, ref_tf, ref_shape, ref_crs) -> ndarray` (raster.py:246).
  Delete the plan's bespoke `_read_aligned_layer` wrapper.
- **L2 (accept) — add `restorable_indices` property** (`np.flatnonzero(self.restorable)`) — the
  greedy/SA optimizer's candidate move-set; let the data model own the contract.
- **CRS (accept, grounding nit)** — `from_rasters` uses `crs.to_string()` (not `str(crs)`) for S2
  consistency / a round-trippable CRS string.
- **L3 (accept as doc) — hot-loop validation overhead.** `_check_restored` allocates `~restorable`
  per call; `ndimage.label` dominates so it's minor. Document that a tight optimizer loop should call
  `compute_mesh(grid, existing_habitat | restored)` directly (skipping the public checked methods).
  No unchecked method yet (YAGNI until the optimizer lands).

## Declined (with rationale)

- **N1 — rename `habitat_mask` → `habitat_map`.** Kept: the method returns *the habitat mask*
  (`compute_mesh`'s exact input), so the name is accurate; method-vs-param "collision" is in
  different scopes and not a real hazard.
- **N2 — drop the `arr is None` branch in `validate`.** Kept as harmless defensive code (consistent
  with the length-loop over all three arrays).

## Not changed (verified fine)

Standalone model vs reusing `ConservationProblem` (decoupling justified — restoration has cell states
+ a landscape objective, no features/targets/`pu_vs_features`; `GridGeometry` reused by composition,
no infra to re-sync); full-length bool `restored ⊆ restorable` (direct `compute_mesh` input, optimal
for a bit-flipping optimizer); `cost: np.ndarray | None` + `__post_init__` fill (single assert site,
matches repo idiom); rasterio-free core; `validate()->list[str]` (matches `ConservationProblem`).
