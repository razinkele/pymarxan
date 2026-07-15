# restoptr — MESH (effective mesh size) measure — design

**Date:** 2026-07-15
**Status:** Approved (brainstorm). Pending spec self-review → user review → writing-plans →
multi-agent design review (scientific lens is load-bearing — the MESH formula) → TDD.
**Scope:** The first `pymarxan.restoration` piece — a **landscape-index measure**:
`compute_mesh`, the effective mesh size of a habitat map on a raster grid. The scientifically
load-bearing evaluation core of restoptr-style restoration planning; the restoration data model
and a MESH-maximizing optimizer are deferred follow-on pieces. Named-competitor gap: **restoptr**
(Justeau-Allaire et al. 2021, *J. Appl. Ecol.* doi:10.1111/1365-2664.13803; package paper
*Restoration Ecology* doi:10.1111/rec.13910).

## Motivation

Marxan/pymarxan selects reserves by representation; it has no notion of **landscape
fragmentation** — whether the selected/existing habitat forms a few large patches or many small
ones. Ecological restoration planning (restoptr) optimizes exactly this: which cells to restore so
the habitat becomes *less fragmented / better aggregated*, measured by a landscape index. **MESH
(effective mesh size, Jaeger 2000)** is restoptr's flagship index and the natural first measure —
it rewards fewer, larger habitat patches. This closes the restoration gap and adds a genuinely new
landscape metric to the library.

## Scientific model — MESH (to be verified in design review)

For a raster study region of total area `A_total`, partition the **habitat** cells into **patches**
(maximal connected components under a chosen adjacency), with patch areas `A_1, …, A_n`. The
**effective mesh size** is
```
MESH (m_eff) = (1 / A_total) · Σ_{i=1}^{n} A_i²
```
(Jaeger 2000, "Landscape division, splitting index, and effective mesh size", *Landscape Ecology*
15:115–130). Equivalent forms: the **degree of landscape division** `D = 1 − Σ (A_i/A_total)²`
and **coherence** `C = Σ (A_i/A_total)² = m_eff / A_total`. We report `m_eff` (area units).

**Load-bearing details (flagged for the scientific-accuracy design-review lens, the way the raptr
`1 − WSS/TSS` denominator was):**
- **`A_total` is the total *landscape* area (every valid cell in the study region), NOT the total
  habitat area.** This is Jaeger's denominator; using habitat area instead changes the meaning and
  the scale. Here `A_total = n_valid_cells · cell_area`.
- **Patches are connected components of *habitat* cells only.** A_i = (cells in patch) · cell_area.
- **Adjacency:** rook (4-neighbour) by default — this matches **restoptr's deliberate choice**
  (`lsm_c_mesh(..., directions = 4)`) and `GridGeometry.build_boundary`'s rook edges. Note it is
  *not* the field-wide default: landscapemetrics / FRAGSTATS default to queen (8-conn), so numeric
  cross-checks against those tools require setting their `directions=4` explicitly. queen
  (8-neighbour) is exposed as an option.
- **CUT vs CBC variants:** Jaeger (2000) gave the cutting-out (CUT) form `m_eff = ΣA_i²/A_total`;
  the cross-boundary-connection (CBC) correction is **Moser et al. (2007)**
  (doi:10.1007/s10980-006-9023-0), for patches spanning *reporting-unit* boundaries. For a
  self-contained study region (one landscape, no sub-unit boundaries) CUT is exact and CBC coincides
  with it — that is what restoptr uses. We compute the CUT form; CBC is out of scope. *(Verified in
  design review against landscapemetrics `lsm_c_mesh`, which restoptr calls.)*

Range: no habitat → `MESH = 0`; a single patch covering all valid cells → `MESH = A_total` (max).
More/larger aggregated patches → higher MESH.

## API

New subpackage `src/pymarxan/restoration/`:

```python
@dataclass(eq=False)               # numpy patch_areas field breaks the auto __eq__ (repo convention)
class MeshResult:
    mesh: float                 # effective mesh size m_eff (area units)
    n_patches: int              # number of habitat patches
    patch_areas: np.ndarray     # (n_patches,) area per patch, descending
    total_area: float           # A_total = n_valid_cells · cell_area
    # cheap canonical Jaeger companions (properties):
    #   coherence  C = mesh / total_area = Σ(A_i/A_total)²   ∈ [0, 1]
    #   division   D = 1 − C                                  ∈ [0, 1]

def compute_mesh(
    grid: GridGeometry,
    habitat_mask: np.ndarray,       # (n_pu,) bool over valid cells, PU (row-major) order
    *,
    connectivity: str = "rook",     # "rook" (4-conn) | "queen" (8-conn)
    cell_area: float | None = None, # default: grid.cell_width · grid.cell_height
) -> MeshResult
```

`habitat_mask[i]` is habitat/not for the i-th valid cell, in the same row-major order as PUs
(so a `ConservationProblem` built from the grid can pass a boolean feature/threshold column
straight through; a problem-level convenience wrapper is deferred to the restoration data model).

## Implementation approach

Use **`scipy.ndimage.label`** (scipy is already a dependency — S3a) — the standard, fast raster
connected-components tool — rather than the vector-PU BFS in
`constraints/contiguity.py::count_connected_components` (which returns only a *count*, not patch
sizes, and is keyed on a boundary DataFrame; the raster tool is the right fit here).

1. Coerce and validate: `habitat_mask = np.asarray(habitat_mask).astype(bool)`; require
   `len(habitat_mask) == grid.n_pu` (else `ValueError`) and `connectivity in {"rook", "queen"}`
   (else `ValueError`). Lift onto the 2-D grid: `hab2d = np.zeros(grid.shape, bool);
   hab2d[grid.mask] = habitat_mask`. (Boolean-mask assignment fills the `mask==True` positions in
   row-major order — exactly the valid-cell/PU order, per S1/S2.)
2. `structure` = rook cross (`[[0,1,0],[1,1,1],[0,1,0]]`) or queen (`np.ones((3,3))`).
3. `labels, n = scipy.ndimage.label(hab2d, structure)`; patch cell-counts =
   `np.bincount(labels.ravel())[1:]` (drop background label 0).
4. `cell_area = cell_area or grid.cell_width * grid.cell_height`;
   `patch_areas = counts · cell_area` (sorted descending);
   `A_total = grid.n_pu · cell_area`;
   `mesh = float((patch_areas ** 2).sum() / A_total)` (0.0 when no habitat).

Pure numpy + one scipy call; `O(n_cells)`. No solver, no ConservationProblem dependency — a clean,
independently testable measure like `compute_space_held` / `compute_phylogenetic_diversity`.

## Edge cases

- **No habitat cells** → `n_patches=0`, `patch_areas=[]`, `mesh=0.0`.
- **`cell_area <= 0`** → `ValueError` (an explicit override bypasses `GridGeometry`'s own
  `cell_width/height > 0` check; without this it would silently yield `mesh=0.0` / negative areas).
  This is the only reachable `A_total == 0` path — an all-False mask (`n_pu == 0`) is already
  rejected by `GridGeometry.__post_init__`, so a degenerate empty grid can't be constructed. The
  `total_area > 0` guard on the division is kept as harmless defence.
- **All valid cells are habitat & connected** → `n_patches=1`, `mesh = A_total` (maximum).
- **`habitat_mask` length ≠ `grid.n_pu`** → `ValueError`.
- **`connectivity` not in `{"rook", "queen"}`** → `ValueError`.
- **`cell_area=1.0`** → MESH in cells² (unit-cell convenience).
- Rook vs queen changes patch membership (diagonal-only touching cells): queen merges them, rook
  does not — asserted in tests.

## Testing strategy (TDD)

- **Hand-computed 3×3 grid** (cell_area=1, A_total=9): all-habitat one patch → MESH = 9²/9 = 9;
  two separated 1-cell patches → MESH = (1+1)/9 = 0.222; a 4-cell L patch + isolated cell →
  (16+1)/9 = 1.889.
- **Bounds/monotonicity:** empty → 0; full-connected → A_total (max); merging two patches (adding a
  bridging cell) never *decreases* MESH; a single large patch beats the same #cells fragmented.
- **Adjacency:** two cells touching only diagonally → 2 patches under rook, 1 under queen.
- **cell_area:** MESH scales linearly with `cell_area` (units) — cell_area=k → MESH×k vs cells².
- **Grid mapping:** a masked (invalid-cell / non-rectangular) grid maps `habitat_mask` to the
  correct 2-D positions (row-major); a non-square grid (nrow≠ncol) is handled.
- **Validation:** wrong-length mask raises `ValueError`; `connectivity="diagonal"` (unknown) raises
  `ValueError`.

**Target:** ~10–14 tests, `make check` green, parity 35.0 untouched (pure new subpackage, no solver
change). Scientifically validated by the design-review scite lens (MESH formula + `A_total`
denominator) before merge.

## Out of scope (deferred, own pieces)

- **IIC / PC** connectivity indices (patch graph + topological distances; Pascual-Hortal & Saura
  2006 / Saura & Pascual-Hortal 2007) — the natural second measure piece. **Home decided:** they
  live in `pymarxan.restoration` (restoration landscape-pattern indices, restoptr's family), with
  their own `compute_*` + `*Result` (patch graph + per-node dIIC/dPC importances — NOT sharing
  `MeshResult`). This is distinct from `pymarxan.connectivity` (circuit/graph *flow*: Omniscape,
  climate velocity, smoothing); the `restoration/__init__.py` docstring records the split.
- **Restoration data model** (habitat / restorable / available cell states on a grid) + a
  problem-level `compute_mesh` convenience.
- **MESH-maximizing optimizer** (greedy "restore the restorable cell that most raises MESH until
  budget", then SA/MIP) — the enforcement follow-on, mirroring raptr measure → SpaceState.
- Jaeger's cross-boundary (CBC) variant; other landscape indices (splitting index, NP, ENN).
- restoptr's exact constraint-programming (Choco) solver — pymarxan will use greedy/heuristic.

## References

- Jaeger (2000) "Landscape division, splitting index, and effective mesh size: new measures of
  landscape fragmentation." *Landscape Ecology* 15(2):115–130, doi:10.1023/A:1008129329289 —
  effective mesh size (the CUT form). **✓ verified (design review).**
- Moser et al. (2007) *Landscape Ecology* 22(3):447–459, doi:10.1007/s10980-006-9023-0 — the CBC
  boundary correction (out of scope). **✓ verified.**
- Justeau-Allaire et al. (2021) *J. Appl. Ecol.* doi:10.1111/1365-2664.13803 — constrained
  optimization of landscape indices for restoration. **✓ verified.**
- restoptr package: Justeau-Allaire et al. (2023) *Restoration Ecology* doi:10.1111/rec.13910; its
  MESH is `landscapemetrics::lsm_c_mesh(directions = 4)` — same total-landscape denominator.
  **✓ verified.**
- Reuses `pymarxan.models.grid.GridGeometry` (raster grid + rook adjacency, S1–S4) and
  `scipy.ndimage.label` (the raster connected-components tool — deliberately distinct from
  `constraints/contiguity.count_connected_components`, which is a vector-PU BFS, count-only, over a
  boundary DataFrame).
