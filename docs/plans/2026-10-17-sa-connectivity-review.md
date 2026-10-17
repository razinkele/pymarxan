# SA optimises connectivity (v0.44.0) — 4-lens design review synthesis

**Date:** 2026-10-17 · **Reviewed:** spec `2026-10-17-sa-connectivity-design.md`, plan
`2026-10-17-sa-connectivity-plan.md` (`c0bb05d`), prototype `c9f076c` (rulings applied at `f71ef1c`) · **Lenses:** architect,
codebase grounding (executed code), scientific accuracy, independent re-design. Raw findings and
probe scripts are in the session scratchpad (`sacon/review/`).

## Verdict

No CRITICAL or HIGH finding from any lens.

- **Architect:** approach A fits the `ProblemCache` rules; SA and iterative improvement need no
  call-site change; layering is clean; the new defaulted fields break no other cache user.
  Measured delta cost ×1.154 at degree 4, ×0.991 with connectivity off.
- **Grounding:** every code claim checked out against `file:line`.
  - **Fuzz:** 300 adversarial problems (duplicate, reversed, self and unknown rows; negative
    values; mixed id dtypes) gave a worst delta / full-term error of 8e-14.
  - **SA end to end:** 120 seeded runs with BLM, COSTTHRESH and locks showed no
    reported-vs-tracked mismatch.
  - **Mutations:** all 20 were killed by the plan's tests.
  - **Overhead with connectivity off:** none measurable against `06e0be5`.
- **Science:** the sign, the once-per-pair counting and the source-without-sink rule match
  prioritizr's implementation and Marxan's `ASYMMETRICCONNECTIVITY` bookkeeping; the spec §5
  delta equals Marxan's annealing bookkeeping.
- **Independent re-design:** arrived at the same design (CSR arrays, O(degree) delta, free off
  gate).
  - **Fuzz:** 300 problems, within 3e-14.
  - **Cross-solver check on a 6-PU fixture:** brute force, the MIP and seeded SA agree
    (−22.5 symmetric, 21.0 asymmetric).

## Findings and rulings

| # | Sev | Lens | Finding | Ruling |
|---|---|---|---|---|
| 1 | MEDIUM | architect (+ grounding, re-design LOW) | Asymmetric delta uses `~selected[nbrs]`: wrong for 0/1 int64 / uint8 selections (measured 10.0 vs 0.0) while the full objective casts to bool — silent disagreement | **R1**: precomputed `conn_out_rowsum`; delta `rowsum − dot(val, sel)`; int64 / uint8 selection tests |
| 2 | MEDIUM | grounding | Symmetric half of the headline SA test passes on the old code (seed 42 plan has connectivity 0); behaviour seed 123 never fails | **R2**: assert a non-zero connectivity term; keep only seeds that fail on the old code; spec §7 corrected |
| 3 | MEDIUM | architect, re-design | Connectivity semantics live in three hand-written copies (evaluator, MIP, cache); the MIP already differs (KeyError on unknown ids; negative asymmetric values unbounded) | **R4**: cross-copy test (brute force = MIP = cache, seeded SA reaches it). Shared normaliser and the MIP edge cases → backlog |
| 4 | MEDIUM | science | "These are prioritizr's penalties" holds only for clean input (prioritizr keeps the id1 > id2 row of a conflicting pair, sums repeated symmetric rows, rewards an asymmetric self-row, errors on unknown ids) | **R5**: manual wording qualified, pymarxan's rules stated |
| 5 | MEDIUM | science | Solver list omits the Zonation solver and the external Marxan binary (report only); docs should distinguish this term from Marxan's boundary-file connectivity × BLM | **R5** |
| 6 | MEDIUM | re-design | CHANGELOG "equals the MIP's term" is too broad given the MIP's edge-case crashes | **R5**: qualified |
| 7 | LOW | architect, grounding | Full-objective term rebuilds an O(edges) row array per call (slows ITIMPTYPE 3 swaps on large tables) | **R3**: precomputed row index |
| 8 | LOW | architect, science | CHANGELOG omits downstream effects (BLM calibration, replacement cost, auto STARTTEMP) and that plans may gain units / cost | **R5** |
| 9 | LOW | re-design | Merged asymmetric out/in arrays would halve the asymmetric cost (+45% vs +21% symmetric); bench lacks the asymmetric case | **Not adopted** (acceptable cost); asymmetric case added to the bench's printed output |
| 10 | LOW | architect | 9-element tuple / integer `conn_mode` codes | **Not adopted** (cosmetic, private) |
| 11 | LOW | architect, re-design | II swap / two-step passes have no connectivity test | **Not adopted**: ITIMPTYPE 3 uses the full objective, covered by full = evaluator |
| 12 | LOW | grounding | String PU ids already split SA's reported vs tracked objective (pre-existing) | Backlog |
| 13 | LOW | science | Marxan's greedy heuristics include BLM·connection cost; pymarxan's ignore `ASYMMETRICCONNECTIVITY` in the boundary file | Backlog |
| 14 | LOW | architect | Zone SA full vs delta may disagree on self-pairs (by reading) | Backlog — confirmed by probe: a self-row gives delta +2v when the unit leaves a zone, full difference 0 |
| 15 | LOW | grounding | Plan says 20 bench tests deselected; it is 21 after Task 3 | **R6**: fixed |

None of the findings is Marxan-parity-critical: `CONNECTIVITY_WEIGHT` is a pymarxan/prioritizr
extension. No golden changes, and the 35.0 anchor holds.
