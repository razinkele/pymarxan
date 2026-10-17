# Stage 4 (v0.42.0) — 4-lens design review synthesis

**Date:** 2026-10-17 · **Reviewed:** spec `2026-10-17-stage4-proportion-decisions-design.md`,
plan `2026-10-17-stage4-proportion-decisions-plan.md` (8b17029), prototype `2c0fe48` ·
**Lenses:** architect, codebase grounding (executed code), scientific accuracy, independent
re-design. Raw findings and probe scripts are in the session scratchpad (`stage4/review/`).

## Verdict

- **Architect:** the design fits pymarxan's architecture:
  - layer direction is kept;
  - the `Solver`/`Solution` shape is unchanged;
  - the Stage 3 builder split is respected;
  - the SA and `ProblemCache` paths are untouched;
  - there are no import cycles.
- **Grounding:** replayed every task on a fresh worktree; each task's tree matched the prototype commit, and the full suite passed. The one blocking defect was five plan diffs missing their final context line.
- **Science:** confirmed the prioritizr mapping and the boundary equivalence.
- **Independent re-design:** agreed with the core choices: the dtype discriminator, the evaluator forms, and continuous connectivity auxiliaries.

All accepted findings are applied to the prototype (`3d0e59f`) and to the regenerated plan.

## Findings and rulings

| # | Sev | Lens | Finding | Ruling |
|---|---|---|---|---|
| 1 | HIGH | grounding | 5 of 49 plan diffs lose their final blank context line; `git apply` fails (generator `rstrip`) | **Fixed**: generator keeps trailing context; plan replayed with strict `git apply --whitespace=nowarn` |
| 2 | HIGH | re-design (+ grounding LOW) | `tol = 1e-6 + 1e-7·|R|` lets an empty fractional plan "meet" every target below ~1e-6 — a silently violated target. A relative-only rule passed 600/600 LP rows at scales 1e-6…1e8 (CBC) | **Accepted (R1)**: `tol(R) = 1e-7·|R|`; small scales added to tests; empty-plan test. Spec §5.3 amended |
| 3 | HIGH | architect (+ re-design MEDIUM) | `build_solution` accepts NaN, 2.0, −0.5, float32; int 0/1 arrays are fancy-indexed (pre-existing: cost 75 instead of 35) | **Accepted (R2)**: `_coerce_selected` input contract; int 0/1 → bool (CHANGELOG *Fixed*) |
| 4 | HIGH | science | Manual example binarises an LP optimum into a plan missing every target, with no caveat | **Accepted (R8/R9)**: example prints `all_targets_met`; caveats; points to binary/semicontinuous |
| 5 | MEDIUM | science | Locked-in fixed at `upper` understates existing (fully protected) reserves | **User decision (R3)**: locked-in = 1.0 in every mode; `upper` caps free PUs only; presolve follows (R4). Spec §3.2 amended |
| 6 | MEDIUM | science | "Partial protection" framing conflicts with the literature (reduced-effectiveness zones); Σ r·x assumes even spread within a unit | **Accepted (R9)**: docs define x as a share of area/effort, state the assumption, point to zones |
| 7 | MEDIUM | re-design | Same plan, different `targets_met` by dtype; `binarize_solution` can flip met → unmet at exact-target ties | **Accepted as documented contract (R8)**: binary rule unchanged; pinned test of the flip |
| 8 | MEDIUM | architect | Private `_compute_achieved` / `_require_binary_solver` imported across modules and into Shiny | **Accepted (R7)**: public `compute_held`, `require_binary_solver` |
| 9 | MEDIUM | architect | `presolve_check` accepts invalid `decisions/upper/lower` | **Accepted (R6)**: leaf `solvers/_decisions.py`; `presolve_check` validates |
| 10 | MEDIUM | architect | Raw-array public APIs (`constraints/*`, `objectives/*` `evaluate`/`penalty`) treat x = 0.1 as selected | **Documented, not guarded**: SA hot paths, unreachable from MIP solutions; listed as binary-only in the manual |
| 11 | LOW | re-design | Upward snap to `upper` moves genuine simplex values; can breach COSTBUDGET | **Accepted (R5)**: no upward snap to `upper`. Spec §4 amended |
| 12 | LOW | science | Tolerance looser than prioritizr's eval (1e-10); boundary equivalence only for prioritizr's simple formulation; fractional boundary is a surrogate; frequency meaning for fractional / mixed sets | **Accepted (R9)**: documented |
| 13 | LOW | grounding | RED-step expected messages name the wrong symbols (Tasks 4, 9) | **Fixed** in the regenerated plan (re-measured) |
| 14 | LOW | grounding | Badly scaled amounts (amount/target > ~1e9) hit solver tolerances (CBC returns x = 0, honestly reported unmet) | **Documented** in the manual |
| 15 | LOW | architect | Rule duplication (dtype test, tolerant met/shortfall, conflict list), extraction plumbing, Σx/format idioms | **Not adopted** (cosmetic; each site is small and tested) |
| 16 | LOW | architect / re-design | `lower=None` default with prioritizr hint; record `decisions/upper/lower` in metadata; threshold default from `upper` | **Not adopted**: metadata would change binary output; the error message already names the valid ranges; threshold documented (R8) |
| 17 | LOW | grounding | Completeness regex misses `getattr(..., "selected")` | **Not adopted** (no such access exists; regex documented as the contract) |

None of the findings is Marxan-parity-critical. Binary results are unchanged: every golden
(stage1 v038, stage3 v040, zones v037, stage4 v041) passes without regeneration at `3d0e59f`.
