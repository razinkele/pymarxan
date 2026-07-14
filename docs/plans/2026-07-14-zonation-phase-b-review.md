# Zonation Phase B design review — synthesis

**Date:** 2026-07-14
**Reviewed:** `2026-07-14-zonation-phase-b-design.md` + `...-implementation.md`
**Method:** four perspectives (architect, codebase-grounding [ran greps against
real code], independent re-design, scientific/consistency).

## Verdict

**Approve after fixing the findings below.** The two central judgment calls
(one deterministic length-1 solution over copies/portfolio; `top_fraction` as the
reserve knob with targets-not-guaranteed) were independently endorsed by the
re-designer. But three agents independently caught a blocker, and the re-designer
found a real correctness gap the plan missed.

## Findings absorbed

### CRITICAL — `SolverRegistry.get()` does not exist; the method is `create()`
`registry.py` exposes `register` / `create(name)` / `list_solvers` /
`available_solvers` — **no `get`**. The plan uses `.get(` in 5 spots (design +
impl Global Constraints + the Task-2 registry test + its expected-failure line).
Executed verbatim, `get_default_registry().get("zonation")` raises
`AttributeError` **before and after** registration, so the test never goes green
and `make check` can't pass. `create()` instantiates no-arg via
`self._solvers[name]()` and raises `KeyError` on an unknown name, so the plan's
"raises KeyError" narrative becomes correct once fixed. **Fix:** `.get` →
`.create` everywhere.

### MEDIUM — locks not enforced on the thresholded reserve
`top_fraction` selects purely by rank. `rank_removal` ranks locked-out cells
*lowest* and locked-in *highest*, so at moderate fractions locks are respected
incidentally — but a high `top_fraction` (notably the tested `1.0`, or any
fraction exceeding the non-locked-out share) sweeps **locked-out** PUs into the
reserve, and a pathologically low fraction could drop a **locked-in** PU. Every
other solver treats lock-out/in as a hard constraint (MIP `x==0`/`x==1`; SA/greedy
honor `status`). **Fix:** after building `selected`, apply
`selected[status==LOCKED_OUT]=False` and `selected[status==LOCKED_IN]=True`; add a
test with a locked-out PU at `top_fraction=1.0`. The `top_fraction=1.0` test
passed only because the fixture had no locked-out PUs (this path was untested).

### MEDIUM — document the `num_solutions` divergence from MIPSolver
The plan returns length-1 regardless of `num_solutions` (correct — copies would
fake variety to `compute_selection_frequency`, which divides by
`len(solutions)`). But `MIPSolver` (also deterministic) pads to `num_solutions`
identical deep copies (`mip_solver.py:577`), so the two now have opposite
contracts. Length-1 is the better call, but the docstring/CHANGELOG should note
the deliberate divergence so a caller expecting `len==num_solutions` isn't
surprised. **Fix:** one docstring line.

### MEDIUM — make the `supports_*` inherit-True decision explicit
Zonation is blind to PROBMODE 3 / TARGET2 / SEPNUM during ranking, but
`build_solution` reports those gaps post-hoc — exactly `HeuristicSolver`'s
contract, which keeps `supports_probmode3/clumping/separation` at their `True`
defaults. Keeping the defaults is **correct** (overriding to False would make
Zonation less capable than the heuristic it resembles and wrongly hide it from a
capability-filtering dispatcher). **Fix:** a one-line docstring note + a light
assertion in the ABC-surface test so a future maintainer doesn't flip them.

### MEDIUM (scientific) — `top_fraction` is a PU-count share, not area/budget
`top_fraction(f)` selects `ceil(f·n_pu)` PUs *by count*. The docs call this
"share of the landscape," which is raster-Zonation phrasing where equal-area
cells make count = area. For pymarxan's vector PUs of unequal area/cost,
top-30%-by-count ≠ 30% of area ≠ 30% of budget — and the 30×30 / Aichi targets
are area-based. Phase A already records both `prop_landscape_remaining` (count)
and `prop_cost_remaining` (budget) in the curves for exactly this reason.
**Fix (doc-only):** call `top_fraction` a "fraction of planning units (by count)"
and note that budget-based thresholding should read
`performance_curves["prop_cost_remaining"]`.

## Not absorbed (with reason)
- **Match MIP's copy-padding** — rejected; length-1 is the honest contract
  (copies degrade selection-frequency analysis). Documented instead.
- **Nested-threshold portfolio** — rejected in the spec; callers reslice via
  `result.top_fraction(f)` on the metadata ranking.
- **`config.metadata["top_fraction"]` per-run override** (independent LOW) —
  deferred; the registry default + direct construction + metadata-reslice cover
  the need. Revisit if Phase D needs a per-run knob.
- **A pinning test for `supports_*`** as a *separate* test — folded into the
  existing ABC-surface test instead of a new one.
