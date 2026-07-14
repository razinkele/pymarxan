# Zonation Phase D — Shiny panel + solver-picker wiring design

**Date:** 2026-07-14
**Status:** Approved (brainstorm), pending implementation plan + review
**Scope:** Phase D — the final Zonation phase. A standalone "Zonation" Shiny tab
(priority-rank map + performance-curve plot) plus a solver-picker entry so
Zonation is also runnable in the main run flow. This completes Zonation
end-to-end across the stack (core → solver → smoothing → UI).

## Motivation

Phases A–C shipped the Zonation engine, the `ZonationSolver` adapter, and
distribution smoothing — all Python-only. Phase D surfaces the paradigm's value
(a continuous priority map + per-feature performance curves) in the Shiny app,
the way the rivers feature was finished. Two complementary entry points: a
dedicated tab for the rich ranking visualization, and a solver-picker choice so
Zonation runs through the normal run→results flow like any other solver.

## Architecture & reused patterns

This is UI work in `pymarxan_shiny`; it follows established module conventions
(no new patterns invented):
- **`rivers_panel`** (`modules/rivers/rivers_panel.py`) — the template for a
  domain panel: a `@module.ui` / `@module.server` pair taking a reactive value,
  with `@render.plot` + `import matplotlib.pyplot as plt` (server-side
  matplotlib — **never** plotly-via-`render.ui`, the hard-won lesson).
- **`frequency_map`** (`modules/mapping/frequency_map.py`) — the template for a
  0–1-per-PU choropleth: `frequency_color(0–1) -> hex` + `create_geo_map`
  (`has_geometry`) / `create_grid_map` (grid fallback). Priority rank is 0–1 per
  PU, so the priority-rank map is this pattern reskinned.
- **`solver_picker`** (`modules/solver_config/solver_picker.py`) — a hardcoded
  `solver_choices` dict + `panel_conditional` config sub-panels + a
  `reactive.effect` that assembles `solver_config`.
- **`run_panel`** (`modules/run_control/run_panel.py`) — `_run_solver` maps
  `solver_type` → solver construction + `SolverConfig`.

## Component 1 — the Zonation panel (`modules/zonation/zonation_panel.py`)

A `@module.ui` / `@module.server` pair taking the reactive
`problem: reactive.Value[ConservationProblem | None]`.

**Pure, unit-testable helpers (module-level functions, tested directly):**
- `rank_to_colors(priority_rank: dict[int, float], pu_ids: list[int]) -> list[str]`
  — map each PU's rank (0–1] to a hex color, aligned to `pu_ids` order. Reuses
  `frequency_color` (white → deep-navy, high rank = dark) for app consistency.
- `performance_curve_frame(result: ZonationResult) -> pd.DataFrame` — the tidy
  data the plot draws (already the `result.performance_curves` DataFrame;
  a thin accessor/validator so the plot code and a test share one source).

**`zonation_panel_ui()`** — a `ui.card`:
- header + short description;
- `ui.input_select("rule", "Removal rule", {"caz": "Core-area (CAZ) — rarity",
  "abf": "Additive benefit (ABF) — richness"})`;
- `ui.input_slider("top_fraction", "Top priority fraction", 0.0–1.0, value=0.3)`;
- `ui.input_action_button("rank", "Rank")` — computing the ranking is O(n²), so
  it runs on an explicit click, not on every input tick;
- the priority-rank map (`@render_widget` output), the performance-curve plot
  (`ui.output_plot`), and a top-priority readout/table (`ui.output_text` +
  `ui.output_data_frame`).

**`zonation_panel_server(input, output, session, problem)`:**
- `@reactive.calc` `_result()` — gated on `input.rank()` (the button) and reading
  `input.rule()`; returns `rank_removal(problem(), rule=input.rule())` or `None`
  when no problem is loaded. (Smoothing is not exposed in the UI — see Out of
  scope.)
- `@render_widget` map — `rank_to_colors` → `create_geo_map(pu, colors)` when
  `has_geometry(problem())`, else `create_grid_map` on a synthetic grid, else a
  "load a spatial project to see the map" message. Follows the existing map
  modules' `has_geometry` gate (the map-crash lesson).
- `@render.plot` curves — matplotlib: x = `prop_landscape_remaining` (1→0),
  y = each `feat_<id>` retained proportion (one line per feature), plus labels.
- top-priority readout — how many PUs are in `result.top_fraction(input.top_fraction())`
  and their share; the table lists the top-ranked PU ids + ranks. `top_fraction`
  changes only this (cheap), not the ranking.

## Component 2 — solver-picker + run-panel wiring

- **`solver_picker_ui`:** add `solver_choices["zonation"] = "Zonation
  (rank-removal)"` and a `panel_conditional("input.solver_type === 'zonation'",
  ...)` with a rule select (`zonation_rule`) and a `top_fraction` numeric
  (`zonation_top_fraction`).
- **`solver_picker_server`:** extend the `reactive.effect` to read those inputs
  into `solver_config` (`"zonation_rule"`, `"zonation_top_fraction"`) when
  `solver_type == "zonation"`.
- **`run_panel._run_solver`:** add a branch that builds
  `ZonationSolver(rule=config_dict.get("zonation_rule", "caz"),
  top_fraction=config_dict.get("zonation_top_fraction", 0.3))` and runs it,
  producing a `Solution` (the thresholded reserve) that flows to the generic
  Results view. Confirm the exact construction/threading path against the shipped
  `run_panel` when planning.

## Component 3 — app wiring (`pymarxan_app/app.py`)

Import `zonation_panel_ui` / `zonation_panel_server`; add a `ui.nav_panel(
"Zonation", zonation_panel_ui("zonation"))` near the Rivers tab; call
`zonation_panel_server("zonation", problem)` in the server, passing the existing
reactive `problem`.

## Testing strategy

Shiny modules are lightly covered by convention (the `src/pymarxan_app/` app is
excluded from coverage; module servers need a session to exercise fully). Follow
`tests/pymarxan_shiny/test_rivers_panel.py`:
- **Pure helpers (real unit tests):** `rank_to_colors` maps rank 1.0 → the
  darkest color and rank→low → light, aligned to `pu_ids`; `performance_curve_frame`
  returns the expected columns.
- **Module smoke:** importing `zonation_panel_ui`/`_server` and constructing the
  UI does not raise (mirrors the rivers-panel test).
- **Picker choice present:** `"zonation"` is in the `solver_choices` the picker
  builds (or the UI contains the option).
- **Run-panel branch:** with `solver_config = {"solver_type": "zonation",
  "zonation_rule": "caz", "zonation_top_fraction": 0.5}`, the solver built is a
  `ZonationSolver` and solving a tiny problem returns a `Solution` whose
  `metadata["solver"] == "zonation"`. (Test the construction helper directly if
  `_run_solver` is not unit-callable; otherwise a focused reactive test.)
- **Live verification (execution-time, optional but recommended):** drive the
  app with Playwright, open the Zonation tab, confirm 0 console errors and that
  the performance-curve plot + map render — as the rivers tab was verified.

**Target:** ~8–12 tests (mostly the pure helpers + smoke + wiring), `make check`
green (0 ruff / 0 mypy), coverage ≥ 75% (Shiny modules stay low-coverage per
convention; the pure helpers carry the real assertions).

## Out of scope (YAGNI, Phase D)

- **Exposing Phase C smoothing in the UI** — needs geometry-derived coords (the
  deferred `SmoothingSpec.from_problem`), plus alpha/CRS controls. The panel runs
  unsmoothed CAZ/ABF; smoothing stays a Python-API feature for now.
- Warp / weights / cost controls in the UI (defaults only).
- A zonation-specific branch of the *generic* Results view (the dedicated tab
  carries the rich viz; the picker path uses the standard reserve view).

## Parity note

Pure UI + wiring; no Marxan-solver/objective math changes. The 35.0 anchor is
untouched; `make check` (which runs the parity tests) plus a `marxan-parity-check`
confirm no regression.

## References

Zonation lineage (Moilanen 2005 / Lehtomäki & Moilanen 2013) — scite-verified in
Phase A. UI reuses `frequency_map`, `rivers_panel`, `create_geo_map` conventions.
