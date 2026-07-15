# `build_boundary` vectorization — multi-agent design review — synthesis

**Date:** 2026-07-15
**Reviewed:** `2026-07-15-build-boundary-vectorization-{design,implementation}.md`
**Lenses:** Codebase-grounding+parity (RAN vectorized-vs-loop-vs-shapely), Independent re-design.
(Architect lens skipped — a single-method, no-signature-change, no-new-abstraction perf refactor.)

## Verdict

**Ship it — no CRITICAL/HIGH/MEDIUM.** The grounding agent implemented the plan's vectorized
`build_boundary` verbatim and proved it a bit-for-bit sorted-multiset match to both the original
loop and shapely `compute_boundary` across a full battery (full grid, random holes, 1×N / N×1
strips, center-hole, single cell, non-square cells, arbitrary non-sequential `pu_ids`), then
monkeypatched it onto the real `GridGeometry` and ran the suite (54 passed). The independent
re-design derived the same vectorization line-for-line (right→`cell_height`, down→`cell_width`,
all four `has_*` shifts, the `perimeter−shared` = exposed-side algebra, id1/id2 orientation) and
ran 30+ masks incl. Fortran-order / non-contiguous / `1e6`-scale — all multiset-identical.
`reshape(-1)` writable-view + C-order invariant empirically confirmed. Scale: 200×200 in ~9 ms
(vs minutes for the loop). Parity/35.0 untouched (no solver math).

## Findings folded in

- **LOW — "algebraically identical" ≠ bit-identical.** The exposed-side form `(2−nl−nr)·h +
  (2−nu−nd)·w` and the loop's `2(w+h) − Σedges` regroup differently in IEEE-754 (measured ~7e-15
  abs / 4e-14 rel — far below the `1e-10` emit threshold and `assert_frame_equal`'s default
  tolerance, and every test uses bit-exact integer cell sizes). Not a defect, but the "identical"
  wording could invite a future `check_exact=True` tightening that would then flake on ULP noise.
  *Folded:* softened the design wording to "identical up to floating-point rounding", and added a
  reference-loop test case with a **non-integer** cell size (`w=0.3, h=0.7`) so the tolerant path
  is actually exercised.

- **LOW — no non-C-contiguous mask test.** The subtle "row-major regardless of memory layout"
  claim was empirically verified by both agents but untested. *Folded:* added a **Fortran-order**
  (`np.asfortranarray`) mask case to the reference-loop test, guarding against a future refactor
  to a layout-sensitive indexing.

## Noted, no action

- The uniqueness-guard switch (`set(...tolist())` → `len(np.unique(...))`) is behaviour-equivalent
  and the message (`"pu_ids must be unique"`) still satisfies the existing `match="unique"` test.
- `test_boundary_wired_and_toggle` compares `build_boundary` outputs without sorting, but both
  sides come from the same `build_boundary` with the same ids → identical order → self-consistent
  under the row-order change (grounding-confirmed).
