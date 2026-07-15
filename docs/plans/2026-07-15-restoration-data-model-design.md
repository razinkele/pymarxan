# restoptr — restoration data model (`RestorationProblem`) — design

**Date:** 2026-07-15
**Status:** Approved (brainstorm). Pending spec self-review → user review → spec-review loop →
writing-plans → multi-agent design review → TDD.
**Scope:** The second `pymarxan.restoration` piece — a standalone **`RestorationProblem`** data
model (grid + existing-habitat / restorable / cost cell states), its `habitat_mask(restored)` bridge
to `compute_mesh`, `validate()`, and `from_arrays` / `from_rasters` ingestion. Makes MESH
*actionable*: `raster → RestorationProblem → restore_mesh(plan)`. The budget/min-max-restore
constraints and the MESH-maximizing optimizer are the next piece. Named competitor: **restoptr**
(`restopt_problem(existing_habitat, restorable_habitat)`).

## Motivation

`compute_mesh` (v0.28.0) measures the fragmentation of a *given* habitat map, but there is no way to
express a **restoration decision** — which non-habitat cells to convert to habitat. restoptr's data
model is exactly this: an existing-habitat raster + a restorable-area raster + cost, over which an
optimizer chooses cells to restore to improve a landscape index. This piece adds that model (the
optimizer follows), decoupled from `ConservationProblem` (restoration has cell *states* and a
landscape-index objective, not features/targets/`pu_vs_features`) — matching how `RiverNetwork` /
`PhylogeneticTree` / `GridGeometry` are standalone domain models.

## The model

`src/pymarxan/restoration/problem.py`:

```python
@dataclass(eq=False)   # numpy fields break the auto __eq__ (repo convention, cf. GridGeometry)
class RestorationProblem:
    grid: GridGeometry                  # raster landscape; grid.mask = study region; A_total for MESH
    existing_habitat: np.ndarray        # (n_pu,) bool — cells already habitat
    restorable: np.ndarray              # (n_pu,) bool — decision candidates (can be restored)
    cost: np.ndarray | None = None      # (n_pu,) float — per-cell restoration cost; None → uniform 1.0
```

All arrays are `(grid.n_pu,)` in the grid's **row-major valid-cell (== PU) order** — the same
contract `compute_mesh(grid, habitat_mask)` uses. `__post_init__` coerces `existing_habitat` /
`restorable` to bool and, when `cost is None`, sets `cost = np.ones(n_pu)` (else coerces to float).
The field type stays `np.ndarray | None`, but after `__post_init__` `cost` is always an ndarray.

**Methods:**
- `n_pu -> int` (property) — `grid.n_pu`.
- `habitat_mask(restored: np.ndarray) -> np.ndarray` — the bridge to the measure: returns
  `existing_habitat | restored`. Validates `restored` is bool of length `n_pu` and a **subset of
  `restorable`** (`(restored & ~restorable).any()` → `ValueError` — you cannot restore a
  non-restorable cell).
- `baseline_mesh(**mesh_kwargs) -> MeshResult` — `compute_mesh(grid, existing_habitat, **kwargs)`,
  the pre-restoration fragmentation baseline.
- `restore_mesh(restored, **mesh_kwargs) -> MeshResult` — `compute_mesh(grid,
  habitat_mask(restored), **kwargs)`, the post-restoration MESH the optimizer maximizes.
- `restoration_cost(restored: np.ndarray) -> float` — `float(cost[restored].sum())` (validates
  `restored` shape/subset as `habitat_mask` does).
- `validate() -> list[str]` — returns an error list (does **not** raise, mirroring
  `ConservationProblem.validate`): array lengths == `n_pu`; `existing_habitat` and `restorable`
  **disjoint** (an already-habitat cell can't be "restored"); `cost` all finite and `>= 0`.

## Ingestion

Two classmethods. The core (`RestorationProblem` + `from_arrays`) stays **rasterio-free** so
`import pymarxan.restoration` works without rasterio (as `compute_mesh` does); `from_rasters` lazily
imports the S2 rasterio helpers.

- `RestorationProblem.from_arrays(existing_habitat_2d, restorable_2d, *, cost_2d=None, x_min, y_max,
  cell_width, cell_height, crs=None, mask_array=None, nodata=None) -> RestorationProblem`
  — pure numpy. **Validity (study region) precedence:** explicit `mask_array` (non-zero, non-nodata)
  → else the **non-nodata footprint of `existing_habitat`** (the habitat layer defines the landscape
  extent; value `0` = in-landscape non-habitat, nodata = outside). Build `GridGeometry(x_min, y_max,
  cell_width, cell_height, valid_mask, crs)`. Flatten each layer to PU order (`arr[rows, cols]` on
  `np.nonzero(valid)`): **binarize** `existing_habitat`/`restorable` as `> 0` (nodata → False), cost
  → float with nodata → default `1.0` (+ a warning, as S2 does). Shape-check all layers share one
  2-D shape.
- `RestorationProblem.from_rasters(existing_habitat, restorable, *, cost=None, band=1, crs=None)
  -> RestorationProblem` — rasterio wrapper reusing S2's `_read` / `_check_align` /
  `_transforms_close` / `_require_north_up` (lazy `from pymarxan.spatial import raster`): read
  `existing_habitat` as the reference grid, align `restorable` / `cost` to it (shape + transform +
  CRS), derive `x_min`/`y_max`/`cell_width`/`cell_height` from the reference transform, delegate to
  `from_arrays`. Single-band; north-up axis-aligned only (guarded by `_require_north_up`).

## Edge cases / validation

- **`existing_habitat` ∩ `restorable` overlap** → `validate()` error (data inconsistency; a
  done-habitat cell isn't restorable). Ingestion does not silently drop it — faithful ingestion,
  flagged by `validate()`.
- **`restored ⊄ restorable`** (or wrong length / non-bool) → `habitat_mask` / `restoration_cost`
  raise `ValueError`.
- **`cost=None`** → uniform `1.0`; **nodata cost on a valid cell** → `1.0` + warning (S2 parity).
- **Empty validity mask** → `ValueError` (no valid cells), as S2.
- **from_rasters misalignment** (shape / transform / CRS mismatch, rotated/sheared) → `ValueError`
  via the reused S2 guards.

## Testing strategy (TDD)

- **Dataclass + validate:** disjoint existing/restorable passes; overlapping → error; wrong-length
  array → error; negative/NaN cost → error; `cost=None` → uniform 1.0.
- **habitat_mask bridge:** `existing | restored` correct; `restored` not a subset of `restorable`
  → `ValueError`; wrong length/dtype → `ValueError`; a hand example on a small grid where
  `restore_mesh(plan)` equals `compute_mesh(grid, existing|plan)`.
- **baseline vs restore MESH:** on a small grid, restoring a bridging cell raises MESH above
  `baseline_mesh` (ties into the measure's monotonicity); `restoration_cost` sums the right cells.
- **from_arrays:** validity precedence (explicit `mask_array` vs existing-habitat footprint);
  binarization (`>0`); cost default + nodata→1.0 warning; row-major PU-order mapping on a masked
  (non-rectangular) grid; empty-mask → error.
- **from_rasters:** write tiny GeoTIFFs (rasterio, shiny env), round-trip to a `RestorationProblem`
  matching the `from_arrays` result; misaligned/rotated raster → `ValueError`. (marker: `spatial`.)

**Target:** ~16–22 tests, `make check` green, parity 35.0 untouched (pure new subpackage, no solver
change).

## Out of scope (deferred, next pieces)

- **Budget / min-max-restore constraints** — belong with the optimizer (restoptr's
  `add_restorable_constraint` / `add_budget_constraint`). The data model carries `cost`; the
  optimizer applies the budget.
- **MESH-maximizing optimizer** — greedy "restore the restorable cell that most raises MESH until
  budget", then SA (not MIP — MESH is a convex quadratic over combinatorial patches). The next
  piece; `restore_mesh` / `restoration_cost` are its evaluation hooks.
- **IIC / PC** landscape-connectivity indices; restoptr's amount-based (fractional) restorable areas
  (we binarize); non-grid (vector) restoration.

## References

- restoptr: Justeau-Allaire et al. (2023) *Restoration Ecology* doi:10.1111/rec.13910;
  Justeau-Allaire et al. (2021) *J. Appl. Ecol.* doi:10.1111/1365-2664.13803 — `restopt_problem`
  (existing habitat + restorable area) is the model this mirrors.
- Reuses `pymarxan.models.grid.GridGeometry`, `pymarxan.restoration.compute_mesh` (v0.28.0), and the
  S2 raster helpers in `pymarxan.spatial.raster` (`_read` / `_check_align` / `_transforms_close` /
  `_require_north_up` / `_nodata_mask`).
