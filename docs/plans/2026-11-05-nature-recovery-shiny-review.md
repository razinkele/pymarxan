# Nature recovery Shiny tab (v0.50.0) — four-lens design review (2026-11-05)

Artifact reviewed: `docs/plans/2026-11-03-nature-recovery-workflow-design.md` §4 (the v0.50.0 outline),
against the shipped v0.49.0 core (`cfaf436`). Lenses: architect, codebase grounding (executed every
claim: headless playwright against the laguna app, end-to-end core drives on `tests/data/simple` and
prioritizr's 1751-unit Marxan project, a scale sweep to 20 000 units), scientific accuracy (scite; 15
papers retrieved, no editorial notices), independent re-design (designed before reading §4).
Full lens reports and probe scripts are archived outside the repo at
`~/.claude/projects/-home-razinka-marxan/be611b2f-c158-4a82-8be3-dc66e22357b8/backup/v050-review/`
(`architect.md`, `grounding.md`, `science.md`, `redesign.md`, `grounding/probe*.py|.out`).

Verdict: §4 names the right constraints (own state, own solve/results path, new-object writes, one
module signature for both apps, size guard) but is not yet a design. More urgently, the review found
**three defects in the live app and core that exist today, independent of the tab**.

## P — defects in shipped code (found by this review; not part of the tab)

| # | Defect | Lenses | Evidence (controller re-checked P1, P2) |
|---|---|---|---|
| P1 | **laguna's Run and every Calibrate tab crash.** `pymarxan_shiny/app.py:194` imports `get_solver` from `pymarxan.solvers`; that name has never existed (only commit touching it: 9e09575, Phase 4a). Clicking Run raises `ImportError` and disconnects the session. No app-level test loads the laguna app. | grounding | `python -c "from pymarxan.solvers import get_solver"` → ImportError (controller); playwright trace `run_panel.py:78 → app.py:194`. `pymarxan_app/app.py:273-302` builds solvers differently and works. |
| P2 | **laguna has no MIP backend.** Shiny Server runs the app as `shiny` (`su --login … shiny`, env python, no `~shiny/.local`), so it imports conda PuLP 2.8.0, whose bundled CBC is unavailable (`PULP_CBC_CMD().available() == False`). `_backends._make_pulp_solver` knows only PULP_CBC_CMD/HiGHS/Gurobi; the env's working `cbc` **2.10.13** (`/opt/micromamba/envs/shiny/bin/cbc`) is never used. `ZoneMIPSolver.available()` returns `True` unconditionally. A `COIN_CMD` fallback to that binary also removes the CBC 2.10.3 "worse plan reported Optimal" bug. | architect, re-design, grounding | Controller: `python -s` → pulp 2.8.0, CBC avail False; `cbc` binary reports 2.10.13; laguna process is `su … shiny … python3 SockJSAdapter.py`. Not observed in the live process. |
| P3 | **A time-limited MIP reports `status = "Optimal"`.** PuLP's `sol_status` says "Solution Found", but `zones/mip_solver.py:341` records only `LpStatus[model.status]`; the single-zone `mip_solver.py:861` does the same (probably the same defect; verify). `RecoveryResult` has no status at all. | grounding (executed), re-design | Probe: 500 units, BLM 1, limit 10 s / 30 s → `status=Optimal sol_status=Solution Found`. |

## CRITICAL (tab design)

| # | Finding | Lenses |
|---|---|---|
| C1 | **A solve can stall every user.** laguna runs one process per app (`shiny-server.conf:159-163`, no utilization scheduler). The only background pattern is an unbounded daemon thread per click with no cancel (`run_panel.py:121-151`); calibration/zonation modules solve synchronously in the event loop. The targets-mode MIP hits any time limit at modest sizes (below); the heuristic is pure Python and holds the GIL. The tab needs a process-wide concurrency limit, server-clamped time limits, and no synchronous solves. | architect, grounding, re-design |

## HIGH

| # | Finding | Lenses |
|---|---|---|
| H1 | **Solve-time evidence for the guard.** Targets MIP reaches a 120 s limit at 200 units with BLM 1, at 2 000 units without BLM, on prioritizr (1751 × 17); budget mode stays ≤ 12 s up to 12 500 units; inputs/check/build/evaluate/sensitivity < 1 s at 20 000 × 5. The core's `RECOVERY_MIP_SIZE_WARNING = 50 000` decisions is far too high for a server and arrives via `warnings.warn` mid-solve. **The time limit is the real guard**, BLM > 0 is the strongest cost driver. | grounding, architect, re-design |
| H2 | **The heuristic is not a fallback on a server.** It uses the project's NUMITNS (Marxan default 1e6) × 10 runs: 130 s on 6 units, 510 s on prioritizr, 610 s for 2 000 iterations at 5 000 units (zone II re-scans; review M8 of 2026-11-03 confirmed). Its plans cost 1.4–3.5× a MIP stopped at 5 s. | grounding, re-design, architect |
| H3 | **Every species becomes a "habitat" and the project's targets are silently replaced.** `habitat_ids()` = all base features; on `tests/data/simple` targets 30/20/15 become 0.3 × existing = 18/13.5/9.9. Anti-goal "silent defaults". Needs a habitat selection and protect targets prefilled from the project. | re-design (executed), grounding |
| H4 | **Real projects with zero-cost units are refused** ("restore cost must be > 0", prioritizr has 4) and §4 gives no remedy. A per-unit file `pu, p_success, restore_cost` (the existing `recovery_units.csv` shape) fixes this, carries per-unit p (which does not fit the long potential CSV), and supplies the per-unit costs the manual already tells users to give (USER_MANUAL §13). | grounding, re-design |
| H5 | **Results must be tied to their inputs.** §4 does not say what happens when inputs change or a new project loads after a solve. Store one immutable bundle per solve (inputs, problem, mapping, solution, result); show it only while it matches the current inputs; snapshot the base explicitly and show a banner when the loaded project changes; drop late-finishing jobs. | architect, re-design, grounding |
| H6 | **Expected-value targets read as "met".** The solver drives the expected restored amount *to* the target, so the chance of actually reaching it is roughly even; "met on average" under-warns. State it in frequencies; tag zero-margin targets; offer a safety-margin input. Never show a "probability met" under independence without an "optimistic" label. (Riera et al. 2023 for number formats — health domain, transfer assumed.) | science |
| H7 | **Single scalar p**: success is heterogeneous by ecosystem/site and often measured after 1–2 years with publication bias (Bayraktarov et al. 2016, marine coastal; no Lithuanian/terrestrial default found). Slider 0.05–1.0, illustrative 0.5 labelled "not evidence", p = 1 amber, per-unit file optional; per-habitat p deferred and disclosed. | science, re-design |
| H8 | **Sensitivity table misleads at the cap** (p × 1.25 capped at 1): show effective success rates, grey capped rows, state a break-even rate, label "same plan re-scored". | science |
| H9 | **Budget mode drops boundary silently and `total_cost` omits the boundary term.** Banner on `boundary_dropped`, grey BLM controls in budget mode, show action cost and boundary penalty separately in both modes. | science, grounding |
| H10 | **No re-entry path.** §4 exports a recovery project but cannot load one (`load_recovery_project` already exists). | re-design, architect |

## MEDIUM

- M1 One module `recovery_panel_ui(id)` / `recovery_panel_server(id, *, problem)`; logic in pure helper files tested without a session (the repo has no session harness); no nested modules; it never writes `problem`, `current_solution`, `all_solutions`, `solver_config` (architect, grounding).
- M2 Layout: one page (sidebar inputs, main = live check → results → export) instead of a 5-step wizard — check/build are ≤ 0.4 s at 10 000 units, so "Build" is not a step (re-design).
- M3 Core additions: `read_potential` / `read_unit_values` (accept `species` alias and habitat names), `recovery_problem_size`; the tab calls `check_recovery_inputs` itself, because `warnings.warn` is process-global and shows once per process (architect, grounding).
- M4 Map: `build_pu_map(recovery_base, colors)` (the built ZonalProblem has no geometry); 4-category colour-blind-safe palette in `ocean_palette.py` with labels/legend, "no action" visible grey; polygon cap for vector bases (architect, grounding, science).
- M5 BLM: own control, default 0, inherited value shown; projects routinely carry BLM 1 and MIP_TIME_LIMIT 300 (grounding).
- M6 Caveat: two tiers (plain banner + expandable `RECOVERY_CAVEAT`); placeholder list generated from provenance (p, protect/restore %, cost factor missing today); soften "usually … for decades" to "often, depending on habitat and measure"; η, g labels (g = 1 is the optimistic default, amber chip) (science).
- M7 Budget framing: say "reduces total relative shortfall (all targets weighted equally)", not "optimal"; protection credits existing habitat without a loss counterfactual; costs in "model cost units" unless user-supplied (science).
- M8 Worked example: one source in the core (`pymarxan.recovery.example_inputs()`), shared with `examples/nature_recovery.py` (re-design).
- M9 Export: one zip = `save_recovery_project` files + plan/habitat/sensitivity CSVs + README with caveat and placeholder list; per-session temp dir cleaned on `session.on_ended`; use the non-deprecated download API (all).
- M10 Dead `zone_problem` and the Zones subtab: remove from both apps (and the `zone_sa` picker entry, a dead end without a zone loader), or feed read-only from the tab (architect prefers wiring, re-design and grounding prefer removal).

## LOW

Empty potential / p = 0 builds a protection-only plan — refuse to solve; "expected restored (ha)" not "restored"; vocabulary from the LT stakeholder deck; Lithuanian strings need a native-speaker review; laguna imports modules under two names — a process-wide semaphore must live in a file imported only as `pymarxan_shiny.modules.…`; app-level test for **each** app evaluating solver factories (would have caught P1); playwright is installed for one e2e test.

## Conflicts between lenses and the controller's ruling

- Architect M2 says the MIP status can be read from `Solution.metadata["status"]`; grounding shows that value is wrong at the time limit (P3). **Grounding wins**: the core must record `sol_status`.
- Execution model: architect wants a child process (real cancel); re-design a thread + semaphore. If the heuristic is not offered in the app (H2), the remaining work is CBC, which already runs as a subprocess and stops at its clamped time limit; a thread + process-wide semaphore + clamped limit is then enough. Child process only if the heuristic stays.
- Scenario comparison: science recommends a 3-run shelf (Adams et al. 2016; Ban et al. 2009 support scenario comparison in participatory planning); re-design defers. User decision.

## References (retrieved through scite by the science lens)

Adams, Pressey & Álvarez-Romero 2016 (10.1371/journal.pone.0158350); Ban, Picard & Vincent 2009
(10.1111/j.1523-1739.2009.01185.x); Bayraktarov et al. 2016 (10.1890/15-1077); Crouzeilles et al. 2017
(10.1126/sciadv.1701345); Fraschetti et al. 2021 (10.3389/fmars.2021.626843); Holl & Aide 2011
(10.1016/j.foreco.2010.07.004); Kockel et al. 2020 (10.1371/journal.pone.0233339); Levin et al. 2013
(10.1016/j.biocon.2012.08.032); Meli et al. 2017 (10.1371/journal.pone.0171368); Moreno-Mateos et al. 2012
(10.1371/journal.pbio.1001247); Pezzati et al. 2018 (10.1021/acs.est.8b01501); Reid, Fagan & Zahawi 2018
(10.1126/sciadv.aas9143); Ren et al. 2017 (10.1002/ldr.2728); Riera et al. 2023 (10.1186/s12961-023-01017-2).

## Decisions

Recorded with the user's answers in the revised spec §4.
