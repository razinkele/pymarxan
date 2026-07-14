# Zonation Phase D design review — synthesis

**Date:** 2026-07-14
**Reviewed:** `2026-07-14-zonation-phase-d-design.md` + `...-implementation.md`
**Method:** three perspectives (architect, codebase-grounding [confirmed all
anchors PASS], independent re-design).

## Verdict

**Approve after fixing the findings.** Grounding verified every wiring anchor,
signature, and line number is correct; the reactive pattern, ipyleaflet fallback,
matplotlib choice, and `active_solver`-is-the-only-run-flow-seam are all faithful
copies of `blm_explorer`/`frequency_map`/`rivers_panel`. But the two design
lenses converged on a real HIGH (stale viz), a real MEDIUM error gap, and several
testability/UX MEDIUMs.

## Findings absorbed

### HIGH — stale `_result` on project reload (silent wrong output)
`_compute` is `@reactive.event(input.rank)`, so reading `problem()` inside it
creates no dependency — loading a new project does **not** clear `_result`. But
`map()` reads `problem()` directly, so on a new load it re-renders the new PU ids
against the **old** `priority_rank`; unmatched ids `.get(pid, 0.0)`-default to
lightest → a plausible-looking but stale/blank map, while `curves()` shows the
previous project entirely. **Fix:** a `@reactive.effect @reactive.event(problem)`
that `_result.set(None)`, so outputs return to the "click Rank" state on load.

### MEDIUM — no error guard around `rank_removal` (cost ≤ 0 raises silently)
The tab calls `rank_removal(p, rule=...)` with default `use_cost=True`, which
**raises** `ValueError` if any PU cost ≤ 0 (legal in Marxan). `_compute` (an
effect) has no try/except (unlike `blm_explorer._run`), so the raise escapes on
the main thread: Rank fails, no message. **Fix:** wrap the compute in try/except
→ `ui.notification_show(..., type="error")`.

### MEDIUM — the run-flow config contract is only Playwright-covered; make it unit-testable
The picker writes `zonation_rule`/`zonation_top_fraction`; `active_solver` reads
them — a string contract a typo breaks silently to defaults, and `active_solver`
is an unimportable app.py closure. **Fix:** extract a pure
`zonation_solver_from_config(config_dict) -> ZonationSolver` helper (in
`zonation_panel.py`), have `active_solver` call it, and unit-test the mapping +
defaults. Clamp `top_fraction` there (`min(1.0, max(0.05, ...))`) — this also
fixes the LOW where a typed `top_fraction > 1` from the picker raises *before*
`run_panel`'s try.

### MEDIUM — brittle picker test → assert on the rendered UI
`inspect.getsource(...) contains '"zonation"'` passes on any occurrence. **Fix:**
`@module.ui` renders session-free, so assert
`"Zonation (rank-removal)" in str(solver_picker_ui("t"))` — checks the choice
actually reached the UI tree.

### MEDIUM — add the `solver_info` zonation branch
The picker's `solver_info` (solver_picker.py:268-310) shows a per-solver
description; Zonation would render blank. **Fix:** add `elif st == "zonation"` with
a description + the "ranks by biological loss; may leave targets unmet at low
top_fraction by design" note (so a picker user isn't confused by unmet targets in
Results).

### MEDIUM — drop the half-wired `top_fraction` slider from the tab
On the tab, `top_fraction` only changes the summary text; it never touches the
map (which shows the *continuous* rank), so it invites an expectation it doesn't
meet. Both the independent design and the reserve/rank split say thresholding
belongs to the picker path, not the continuous-surface tab. **Fix:** drop the
slider + the top-fraction summary from the tab; keep the map + curves + a fixed
top-20 priority table + a simple "RULE: ranked N PUs" readout.

### MEDIUM — decouple the palette (don't cross-import a sibling UI module)
Importing `frequency_color` from `frequency_map` pulls that whole UI module in.
`ocean_palette` already holds the constants and says "every viz module should
import from here." **Fix:** relocate `frequency_color` into `ocean_palette`;
`frequency_map` and `zonation_panel` both import it from there.

### LOW — absorbed
- Echo `res.rule` in the curve plot title (so a changed dropdown vs. the ranked
  rule is visible).
- Guard the matplotlib legend (only call `ax.legend()` when there are `feat_`
  series) to avoid the empty-legend warning on a feature-less problem.
- `ui.notification_show("Ranking…")` before the compute for slow-rank feedback.

## Not absorbed (with reason)
- **Seed the tab from `current_solution.metadata`** after a picker run
  (independent MEDIUM) — deferred; the tab computing its own ranking is simpler
  and the two entry points are intentionally independent (a one-line help note
  covers the surprise).
- **Refactor ALL of `active_solver` into `build_solver`** — over-scoped; the
  targeted `zonation_solver_from_config` helper covers the new branch.
- **A `use_cost` toggle on the tab** — deferred; the error guard handles the
  cost≤0 case gracefully for the first cut.

## Test count
Was +5. Now +6: the two `rank_to_colors` tests, `performance_curve_frame`,
the module smoke, the rendered-UI picker test (replaces getsource), and the new
`zonation_solver_from_config` test.
