# restoptr MESH measure — design review synthesis

**Date:** 2026-07-15
**Lenses:** Scientific-accuracy (scite/landscapemetrics), Codebase-grounding (RAN the code),
Architect + Independent-redesign.

## Verdict

**Design sound, science VERIFIED, no CRITICAL/HIGH.** The scientific lens confirmed the formula and
the parity-critical `A_total` denominator against **landscapemetrics `lsm_c_mesh`** — the R function
**restoptr itself calls** (`lsm_c_mesh(..., directions = 4)`): `MESH = Σa_ij²/A`, `A` = *total
landscape area*, not habitat area. Grounding RAN all 11 tests verbatim — every hand-computed value
reproduces (9.0; 2/9; 17/9; rook 2 vs queen 1; ×4 scaling; 2×3-cell → 24; centre-hole ring → 8;
both `ValueError`s) — and proved the `hab2d[grid.mask]=habitat_mask` row-major mapping order-correct
(== `valid_cells()` == `np.nonzero(mask)` == PU order). Independent redesign converged on the same
design (same denominator, `MeshResult` not a bare float, greedy/SA-not-MIP). Fold the items below.

## Fixes to fold (all MEDIUM/LOW)

- **MEDIUM (grounding + architect) — `MeshResult` must be `@dataclass(eq=False)`.** The numpy
  `patch_areas` field breaks the auto `__eq__` (`ndarray==ndarray` → ambiguous-truth `ValueError`) —
  the repo has hit this twice (`GridGeometry`, `SmoothingSpec`). Latent (no test does `==`). Add
  `eq=False` + the one-line rationale comment.
- **MEDIUM (architect) — record the IIC/PC home now.** `connectivity/` already exists (Omniscape,
  velocity — circuit/graph flow). The deferred IIC/PC are "connectivity indices" by name but are
  **restoration landscape-pattern indices** (patch-graph structural, restoptr's family) → they live
  in `restoration/`, with their own `compute_*` + `*Result` (not sharing `MeshResult`). Record the
  split rule in `restoration/__init__.py`: *restoration = landscape-pattern/fragmentation indices for
  restoration planning (MESH now, IIC/PC later); distinct from `connectivity` = circuit/graph flow.*
- **LOW (architect) — validate `cell_area > 0`.** An explicit `cell_area <= 0` bypasses
  `GridGeometry`'s own `cell_width/height > 0` check and silently returns `mesh=0.0` (or negative
  areas). Raise `ValueError`, matching the function's validation-strict style. + a test.
- **LOW (architect) — `n_patches` single source.** Derive `n_patches = int(patch_areas.size)` (==
  scipy's `n`, since labels are contiguous 1..n) so there's one source of truth.
- **LOW (scientific) — CBC attribution.** The CUT/CBC dichotomy and the CBC correction are **Moser
  et al. (2007)** (doi:10.1007/s10980-006-9023-0), not Jaeger (2000) (who gave the cutting-out form).
  Reword; add the Moser ref.
- **LOW (scientific) — rook is restoptr's choice, not the universal default.** landscapemetrics /
  FRAGSTATS default to queen (8-conn); restoptr deliberately passes `directions = 4`. Keep rook
  default (matches restoptr + `build_boundary`), but drop any "standard default" implication and note
  numeric cross-checks vs FRAGSTATS need explicit `directions`.
- **LOW (scientific, cosmetic) — citations.** Jaeger 2000 full title has the subtitle "*: new
  measures of landscape fragmentation*"; DOI `10.1023/A:1008129329289`. The restoptr *package* paper
  (rec.13910) is **2023** (the 2021 paper is the separate J. Appl. Ecol. New Caledonia study, 13803 —
  the CHANGELOG's "Justeau-Allaire et al. 2021" for 13803 is fine).
- **LOW (architect, optional → ACCEPTED) — expose `coherence`/`division` as `@property`.** The
  canonical Jaeger trio, free algebra (`C = mesh/total_area`, `D = 1 − C`); makes the measure more
  complete for comparing landscapes of different total area. Cheap; add + one test.
- **LOW (architect + grounding, cosmetic) — docstring cross-ref + test rename.** Note `mesh.py` uses
  `scipy.ndimage.label` (raster CC) deliberately distinct from
  `constraints/contiguity.count_connected_components` (vector-PU BFS, count-only). Rename
  `test_l_patch_plus_isolated_cell` (it's a 2×2 block, not an L).
- **Note (scientific) — hectare docstring.** restoptr/landscapemetrics report MESH in hectares
  (`·1/10000`); `compute_mesh` is unit-agnostic via `cell_area` — docstring should note that
  reproducing restoptr's numeric values needs `cell_area` in hectares.

## Not changed (verified fine)

`scipy.ndimage.label` over the vector `count_connected_components` (right raster tool; the two
operate on genuinely different structures); `(grid, habitat_mask)` boundary (decoupled from
`ConservationProblem`, forward-right for the restoration model); deferring the optimizer and NOT
attempting a MIP (MESH is a convex quadratic over combinatorial patch membership — CP/greedy/SA
territory, which is why restoptr uses Choco); packaging (hatch auto-includes the new subpackage).
