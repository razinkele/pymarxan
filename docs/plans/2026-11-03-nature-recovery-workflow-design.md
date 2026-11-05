# Nature recovery workflow — design (rev 2, 2026-11-03)

Status: §4 **rev 3 (2026-11-05)** after its own four-lens review (`docs/plans/2026-11-05-nature-recovery-shiny-review.md`) and user decisions E1–E4; §1–§3 unchanged from rev 2.
Previous: **rev 2 after the four-lens review** (`docs/plans/2026-11-03-nature-recovery-workflow-review.md`,
findings C1–C3, H1–H10, M1–M10) **and user decisions D1–D4** (2026-11-03). Awaiting written-spec
approval. Rev 1 was drafted without a brainstorming interview; remaining choices that are not a
user decision or a review finding are marked *(my call)*.

Inputs: design note `docs/plans/2026-11-03-nature-recovery-design-note.md` (Baker et al. 2025,
*BioScience* 75(6):472–489, doi:10.1093/biosci/biaf030); sibling project ROWER_pymarxan_DST
(`restoration_zones.build_zonal_problem`; reviewed passive spec
`docs/superpowers/specs/2026-09-29-passive-restoration-design.md`, user decisions U1–U3);
pymarxan v0.48.0.

## User decisions (2026-11-03)

- **D1 — two releases.** **v0.49.0 = core** (zone-SA fix, recovery builder, evaluation, I/O,
  zone-solver registration). **v0.50.0 = the "Nature recovery" Shiny tab.**
- **D2 — passive recovery included, ROWER defaults:** cost factor 1.0, η 0.5 (labelled
  placeholders, per-habitat η), passive cost ≥ active cost rejected, and the scalar-η consequence
  stated (§3.6).
- **D3 — scheduling and MESH dropped from v1.** Time lag enters as an optional per-habitat
  maturity factor (§3.4); uncertainty as success-rate re-scoring (§3.7). Scheduling and MESH are
  later, separately designed items.
- **D4 — fix zone SA in pymarxan** so it can choose "no action" (§2).

## 1. Intent

Planners increasingly decide **where to restore**, not only where to protect, and how the two
share a budget. pymarxan has the mathematics — Marxan-with-Zones problems with per-zone costs,
contributions, overall targets and unassigned ("no action") units, exact and heuristic zone
solvers — but no way to use it for recovery planning, and the app cannot solve zone problems at
all today (review H1). This work makes recovery planning a supported, honest workflow: a planner
with a Marxan project plus estimates of where habitat could be restored, how likely restoration is
to succeed, and what it costs, gets which units to protect, restore actively, leave to passive
recovery, or leave alone; how far each habitat's protection and restoration targets are met in
**expected** terms; the cost of each action; and how robust the plan is to lower success rates.

Success for v0.49.0: the core API does this from Python, exactly (zone MIP) and heuristically,
with results that can be read back per habitat after a save/reload. Success for v0.50.0: the
same in the app on laguna.ku.lt without code.

## 2. Prerequisite fix: zone SA can choose "no action" (v0.49.0, D4)

`ZoneSASolver` draws start zones and proposals only from `rule.allowed`
(`zones/solver.py:18-25, 96-100, 138-144`), so it never leaves a unit unassigned even when its
rule has `may_be_unassigned=True` — it then protects/restores every free unit (review C1: SA
`[1,1,1,1]` cost 4.0 vs MIP `[1,0,0,0]` 1.0). Fix: for units with `may_be_unassigned`, include
0 in the proposal options (and in the start-zone draw). This is a **behaviour change** for zone
SA on every zone problem with unlocked, unrestricted units (CHANGELOG `Changed`); the per-flip
delta must handle moves to and from 0 (zone 0 pays nothing and contributes nothing), and the
bench marker must show no per-flip regression. Also fix the solver-picker help text that claims
zone SA can leave units unassigned (review LOW). Not in scope: the zone heuristic's inability to
move a unit back to 0, and II's default ITIMPTYPE 0 being a no-op — both documented; recovery
does not use them as stand-alone solvers.

## 3. Core: `pymarxan.recovery` (v0.49.0)

New top-level package, modelled on `rivers/` / `phylo/` (review M1) *(names my call)*:
`inputs.py` (`RecoveryInputs`, `PassiveAction`), `builder.py` (`build_recovery_problem`,
`check_recovery_inputs`), `evaluate.py` (`RecoveryResult`, `evaluate_recovery`,
`success_sensitivity`), `io.py` (read/write pair), exported from `__init__` with `__all__`. Not in
`zones/` (MarZone parity surface) nor `restoration/` (pattern indices).

### 3.1 Inputs — `RecoveryInputs` (standalone model, review M2)

`@dataclass(eq=False)` with `validate() -> list[str]`, like `RestorationProblem`. Not a
`ConservationProblem` subclass (it is not solvable).

| Field | Meaning | Default |
|---|---|---|
| `base` | `ConservationProblem`: planning units (id, cost, status), features = habitats, `pu_vs_features` = **existing** habitat amount per unit | required |
| `potential` | long table `pu, feature, amount`: the **additional** habitat amount restoration would add to the unit (incremental, not the total afterwards — review H2) | required |
| `p_success` | probability that active restoration succeeds, per unit (column) or one scalar, in [0, 1] | required |
| `restore_cost` | per-unit active restoration cost; else `restore_cost_factor × cost`; optional per-habitat additive implementation cost (review M6) | factor 2.5 *(ROWER placeholder)* |
| `passive` | optional `PassiveAction(efficacy=0.5, efficacy_by_habitat=None, cost_factor=1.0)` — passive success = η_h × p_success; passive cost = `cost_factor × cost` (D2) | off |
| `maturity` | optional per-habitat factor g_h ∈ (0, 1]: share of an existing unit's value a restored unit provides at the assessment horizon (review H6) | 1.0 for every habitat *(labelled: "no time lag modelled")* |
| `horizon_years` | assessment horizon H at which p, η and g are judged; recorded and displayed, not used in the maths | none (must be stated when g ≠ 1) *(my call)* |
| targets | per habitat: `protect_prop` of the existing amount; restore either `restore_prop` of the **maximum expected restorable amount** or an absolute `restore_target` **in expected units** (review H3) | 0.3 / 0.2 *(ROWER placeholders, "policy-style percentages, not ecological thresholds")* |
| `spf` | penalty factor; default scaled per feature so that a unit of shortfall costs more than the dearest way of supplying it (review H7) *(scaling rule left to the plan; must make heuristics ≥ MIP on the anchors)* | scaled |

`validate()` / `check_recovery_inputs` report: unknown unit or habitat ids; negative amounts or
costs; p outside [0, 1]; η or g outside (0, 1]; restore cost ≤ 0 (needed so restoring a
zero-potential unit is strictly worse than no action, review M3); passive `cost_factor ×
cost ≥ restore cost` in any unit, worded as "passive is never chosen" (ROWER); habitats with no
potential (warning); and restore targets above the maximum deliverable — via the existing
`diagnose_zone_infeasibility`, which already works in expected units (review H3).

### 3.2 Zones (review C2)

| Zone id | Name | Zone cost | |
|---|---|---|---|
| — (unassigned) | **no action** | 0 | implicit state, not a listed zone |
| 1 | protect | unit cost | listed first, so `min(zone_ids)` = protect |
| 2 | restore (active) | `restore_cost` | |
| 3 | restore (passive) | `passive.cost_factor × cost` | only when `passive` is given |

`MANDATORYALLOCATION` stays 0, so free units may take no action. **No puzone rows for
restorability** (review C3): a unit with no potential simply gains nothing from a restore zone,
and positive restore cost makes no action strictly better (zero potential = not restorable).

Status handling (review M1): status 3 (locked out) → no action (existing rule). Status 2 (locked
in): without potential → explicit `pu_locks` row into protect; **with potential → puzone {1, 2[,
3]}** (a puzone with ≥ 2 zones forbids unassigned, which is exactly "this protected area is kept
— protected or restored — and never dropped"). Status 1 (initial include) is treated as 0
*(my call)*. The builder asserts protect is the lowest zone id.

### 3.3 Features, contributions and targets (reviews H2, H3, M2)

Each habitat `h` becomes two features with an explicit, returned mapping (review M4); ids
`2h − 1` / `2h` as ROWER (the anchor ports directly) — the builder must handle non-contiguous
habitat ids and reject collisions; exported names `<habitat>_existing` / `<habitat>_potential`.

| Feature | Amount per unit | protect | active | passive |
|---|---|---|---|---|
| `h_existing` | existing amount | 1 | **1** | **1** |
| `h_potential` | `potential × p_success` | 0 | g_h | η_h · g_h |

Restoring a unit keeps the credit for habitat already there (review H2). All targets are
**MarZone overall targets** (one code path, unchanged when passive toggles — review M2):
`h_existing` target = `protect_prop × Σ existing`; `h_potential` target in expected units =
`restore_prop × Σ_restorable potential × p × g_h` (the most active restoration could deliver), or
the absolute expected target. Contributions of 0 keep protect from counting toward restoration.

Note: with separate protect and restore targets the user's proportions fix the mix of outcomes;
units compete only through exclusive actions and cost. Genuine protect-vs-restore competition
for one budget is the **budget mode** (§3.5) (review M7).

### 3.4 Time lag

The maturity factor g_h (§3.1) is the v1 representation of restoration gaps (Baker et al. Fig.
2c): it scales what a restored unit counts for at horizon H. Default 1 is displayed as "no time
lag modelled". Results never add existing-protected and expected-restored amounts into one
"habitat secured" total (science H4).

### 3.5 Solving

`build_recovery_problem(inputs) -> (ZonalProblem, RecoveryMapping)` *(signature my call)*; the
problem is an ordinary `ZonalProblem`, so every zone solver works. Two modes:

- **Cheapest plan meeting the targets** — minimum set (zone MIP; BLM allowed).
- **Best plan within a budget** — zone MIP `min_shortfall` (or `max_weighted_features`) with
  `COSTBUDGET` (`zones/mip_solver.py:82, 251`); boundary terms are dropped in budgeted objectives
  and results say so (review H8).

Default solver: zone MIP; fallback (no MIP backend, or large problems): zone SA (after §2) followed
by iterative improvement with ITIMPTYPE ≥ 2 via `RunModePipeline` *(my call: RUNMODE 2,
ITIMPTYPE 3)*. A size guard warns above a unit × zone count to be set in the plan (zone MIP with
~4 binaries per unit; review M8).

**Correctness anchors (tests):** ROWER's 4-unit anchor (zone MIP unique optimum `[1,0,0,2]`, cost
6.0 — reproduced on 0.48.0); the grounding lens's passive cases (b1–b3: shared overall restore
target, MIP 6.0 / 1.6 / 3.9); a unit with both existing habitat and potential that restore keeps;
status-2 with/without potential; an infeasible restore target caught by the check. Heuristic
results must be ≥ the MIP optimum or report the target shortfall (review H7).

### 3.6 Passive recovery (D2)

Passive success = η_h × active p (ROWER U3: one η per habitat, default `efficacy`). Defaults cost
factor 1.0, η 0.5, both labelled placeholders with `provenance: placeholder` (ROWER). Stated
consequence (ROWER, science H1): with scalar η and costs proportional to unit cost, the cost per
expected unit of passive vs active is `cost_factor_passive / (η_h × cost_factor_active)` in every
unit, so for a given habitat the plan prefers the same action everywhere — a model artefact;
real active/passive choices are site-dependent (Holl & Aide 2011). The help text cites η > 1 only
as "reported but contested" (Crouzeilles et al. 2017 vs Reid et al. 2018).

### 3.7 Results and uncertainty — `RecoveryResult`

From a zone `Solution` + `RecoveryMapping`: action per unit; per habitat the existing amount
protected, the expected amount restored (active and passive separately), their targets and
shortfalls (shortfall first, review H7); cost by action; which mode and solver, whether boundary
terms were dropped. `success_sensitivity(result, multipliers=(0.5, 0.75, 1.0, 1.25))` re-scores
the fixed plan with p scaled (capped at 1) — no re-solve — and reports, per habitat, the lowest
multiplier at which the restore target is still met (review M10).

The always-shown caveat (science lens wording): "Restored amounts are expected values: potential
× probability that restoration succeeds (passive: × efficacy η), judged at horizon H — often only
1–2-year establishment survival. Targets are met on average, not with any stated probability: a
target met by many low-probability units may well fall short in practice. Restored habitat
usually stays below the quality of existing habitat for decades; [no time lag is modelled | a
maturity factor of … is applied]. Costs, η and the maturity factor are placeholders unless
replaced with local evidence. This plan is a decision aid for stakeholder discussion, not a
prescription."

### 3.8 Persistence (review H10)

- `save_zone_project(problem, dir)` in `pymarxan.zones.writers`, symmetric with
  `load_zone_project` (the composition is verified to round-trip: grounding lens).
- `write_recovery_inputs` / `read_recovery_inputs`: `potential.csv`, `recovery_units.csv`
  (p_success, restore_cost), `recovery.json` (mapping habitat → feature ids, zone ids → actions,
  η, g, H, cost factors, targets, provenance flags).
- `save_recovery_project` = the zone project + the sidecar; a reload can rebuild the problem and
  split results back per habitat. Exported zone files are **reproducible in pymarxan**; MarZone
  C++ has no "unassigned" state, so they are not guaranteed to reproduce there (stated in docs).

### 3.9 Zone solver availability

Register `zone_mip` (and the RUNMODE pipeline on zone problems) in `solvers/registry.py` so the
v0.50.0 tab and Python users can select them (review H1/H3). The single-zone picker is untouched
in v0.49.0.

## 4. Shiny "Nature recovery" tab (v0.50.0 — rev 3 after its own review)

Review: `docs/plans/2026-11-05-nature-recovery-shiny-review.md` (P1–P3, C1, H1–H10, M1–M10).
User decisions 2026-11-05: **E1** v0.49.1 hotfix for the laguna `get_solver` crash only (P1); the MIP
backend fallback (P2) and the MIP status fix (P3) are v0.50.0 core prerequisites. **E2** the app
offers the zone MIP only; the heuristic stays a Python-API option. **E3** success probability starts
at 0.5 with a placeholder badge. **E4** v0.50.0 includes a safety-margin input, a scenario shelf
(last 3 runs), English + Lithuanian strings for the tab, and loading a saved recovery ZIP.

### 4.1 Core prerequisites (UI-free, TDD, in v0.50.0)

- **MIP backend (P2).** `solvers/_backends.py`: when `PULP_CBC_CMD` is unavailable, fall back to
  `pulp.COIN_CMD` with a `cbc` found on `PATH` or in `sys.prefix/bin`; `MIPSolver.available()` and
  `ZoneMIPSolver.available()` report the truth. Acceptance: a MIP solve works with the shiny env's
  site-packages only (`python -s`, as user `shiny` sees it), and on laguna after deployment.
- **MIP status (P3).** Both MIP solvers record `proven_optimal` (from `model.sol_status`),
  `time_limit_hit`, `time_limit_s` and wall time in `Solution.metadata`; `RecoveryResult` gains
  `status`, `proven_optimal`, `boundary` (BLM × boundary term) and `objective`, so the tab reports
  action cost and boundary penalty separately (H9).
- **Readers.** `recovery/io.py`: `read_potential(src) -> DataFrame` (columns `pu, feature, amount`;
  `species` accepted for `feature`; habitat **names** mapped to ids through `base.features.name`;
  clear messages for missing columns, duplicate rows, non-numeric amounts) and
  `read_unit_values(src) -> DataFrame` (`pu[, p_success][, restore_cost][, passive_cost]`, either
  value column optional; the `recovery_units.csv` shape). `read_recovery_inputs` reuses both.
- **Per-unit passive cost.** `RecoveryInputs.passive_cost: pd.Series | None` (the v0.49 deferral),
  so zero-cost units are not refused when passive is on (H4).
- **Size.** `recovery_problem_size(problem) -> RecoverySize(units, zones, decisions, boundary_rows)`;
  no policy in the core.
- **Example.** `pymarxan.recovery.example_inputs() -> RecoveryInputs` — the floodplain scenario,
  which `examples/nature_recovery.py` then imports (one source, M8).

### 4.2 Module shape and state

One package `pymarxan_shiny/modules/recovery/`, one public signature for both apps:
`recovery_panel_ui(id)` / `recovery_panel_server(id, *, problem)`. It reads `problem`; it never writes
`problem`, `current_solution`, `all_solutions` or `solver_config`, and does not use `solver()`.
Logic lives in pure helper files (no Shiny import at module level) tested without a session:
settings → `RecoveryInputs`, guard decision, action colours and legend, result tables, export
directory, strings. Package-absolute imports throughout (laguna loads `modules.*` under a second
name). Placed as a top-level "Nature recovery" nav panel in both apps.

The dead `zone_problem` value, the Zones subtab (`zone_config`) and the `zone_sa` choice in the
global solver picker (a dead end: no loader produces a zone problem) are **removed** from both apps
*(my call, M10; wiring a read-only view adds UI that no user flow needs)*.

State (all writes build new objects):

| name | kind | written by |
|---|---|---|
| `recovery_base` | Value | "Use loaded project", "Load recovery ZIP", "Load worked example" — an explicit snapshot, never followed reactively |
| `potential`, `unit_values` | Value[DataFrame \| None] | uploads, via the core readers |
| `settings` | Value[frozen `RecoverySettings`] | widgets, debounced; `dataclasses.replace` |
| `inputs` | Calc → `RecoveryInputs` + `RecoveryCheck` | `check_recovery_inputs`, called by the tab (warnings are process-global, M3) |
| `run` | Value[`RecoveryRun`] | solve completion: one frozen bundle (inputs, problem, mapping, solution, result, sensitivity, label) |
| `shelf` | Value[tuple[`RecoveryRun`, …]] (≤ 3) | "Keep this plan"; oldest dropped; holds the bundle minus the built problem |

Staleness (H5): if `problem() is not recovery_base()`, an amber banner offers "Use the loaded project"
(ids in the uploads may no longer match). If `run.inputs is not inputs()`, results are greyed with
"Inputs changed since this plan — solve again". A solve that finishes after its inputs changed is
shown greyed, never as current.

### 4.3 Layout and flow (one page, M2)

`layout_sidebar`: inputs on the left, consequences on the right; no wizard.

Sidebar, top to bottom:
1. **Start**: Use loaded project · Load recovery ZIP (`load_recovery_project`) · Load worked example.
   Language EN/LT radio (tab-local).
2. **Restoration data**: upload potential (incremental amounts — the help says so); optional upload of
   per-unit values; template downloads pre-filled with the base's unit and habitat ids.
3. **Habitats and targets**: one editable row per base feature — include (default: features that
   appear in the potential file; H3) | name | protect % (prefilled from the project's own targets
   — `prop`, else `min(1, target / existing)` — labelled "from project"; 30 % placeholder only
   where the project has none) | restore % | maturity factor g | passive η. Excluded features are
   dropped from the base handed to the core (`copy_with`).
4. **Success and costs**: success probability slider 0.05–1.00 step 0.05, **0.5 with a placeholder
   badge** (E3), amber note at 1.0, overridden per unit by the uploaded file; active restore cost
   factor (2.5, placeholder) unless per-unit costs are uploaded; passive recovery off by default →
   cost factor 1.0, η 0.5 (placeholders); horizon (required when any g ≠ 1).
5. **Plan**: mode — *Cheapest plan that meets the targets* / *Best plan within a budget* (budget
   objective `min_shortfall` only; budget field shows the least possible cost and, once known, the
   cheapest target-meeting cost); **safety margin** "plan for X % above each restore target" (0, 10,
   20, 30 %; scales `restore_target` in the settings layer, recorded in provenance; E4); *Advanced*:
   BLM (own control, **default 0**, project value shown; greyed with the reason in budget mode; M5),
   MIP time limit (slider up to the server cap).
6. **Solve** (disabled while errors exist, while this session is solving, or when no MIP backend is
   available — with the reason) and **Cancel** (stops waiting; see §4.4).

Main pane, top to bottom:
1. **Check** (live, debounced): units, habitats, zones, decisions vs the server cap; per habitat
   existing total, maximum expected restorable, targets; errors in red (block Solve), warnings in
   amber, verbatim from `RecoveryCheck`; the placeholders still in use (from provenance). An empty
   potential or every `restore_max = 0` is an error in the tab ("nothing to restore"), not a warning.
2. **Results** (after a solve):
   - Caveat, two tiers (M6): an always-visible plain banner — expected value = amount × chance; *the
     plan is built to just reach each target on average, so about half the time it would fall short
     unless a safety margin is used* (H6); restored habitat may take years to match existing; the
     placeholders in use (generated, not fixed text) — and an expandable "Details" with
     `RECOVERY_CAVEAT`.
   - Headline: action cost and boundary penalty separately ("model cost units" unless costs came from
     the user); cost share by action; solver status in words — "cheapest plan (proven)" or "best plan
     found in N s; not proven cheapest" (P3); in budget mode "reduces total relative shortfall, all
     targets weighted equally" and the `boundary_dropped` banner (H9, M7).
   - Per-habitat table, shortfall first: existing protected vs target; expected restored, active and
     passive, vs target; never summed; "short by X (Y %)" text, not colour alone; a "no margin" tag
     where the expected amount equals the target.
   - Action map: `build_pu_map(recovery_base, colors)`; four colour-blind-safe categories with a legend
     (protect / restore actively / restore passively / no action — visible grey); title "Where the
     plan places actions"; hover with unit, action, cost, existing and expected restored amount;
     polygon cap with a table fallback.
   - Robustness (H8): "This plan, if success turns out different — the same plan re-scored": rows by
     **effective** success rate, rows where the cap at 100 % binds greyed, and per habitat the lowest
     success rate at which the restore target is still met ("met down to 62 %; you assumed 80 %").
   - Per-unit action table.
3. **Scenario shelf** (E4): "Keep this plan" stores the current run (≤ 3 per session, named by the
   user or auto-labelled with mode/p/margin); a side-by-side summary — cost, boundary, per-habitat
   shortfalls, key inputs — and a map toggle between kept plans.
4. **Export**: Recovery ZIP = `save_recovery_project` files + `plan_actions.csv`, `habitats.csv`,
   `sensitivity.csv` + `README.txt` with the caveat and placeholder list; single CSV downloads; a
   per-session temp dir removed on `session.on_ended`; current download API (not the deprecated
   `render.download`).

### 4.4 Execution on the shared server (C1, H1)

- MIP only (E2). A solve runs in a worker thread (CBC itself is a subprocess); results return to the
  session through the poll/`ExtendedTask` pattern; reactive values are written only on the main loop.
- One solve per session; a **process-wide** `threading.BoundedSemaphore` (default 2) shared by all
  sessions; a session that cannot acquire shows "queued — another plan is being computed". The
  semaphore lives in a module imported only by its package path.
- Server limits, one constant set in the helper module, overridable by environment variables
  (`PYMARXAN_RECOVERY_*`) so laptops are not held to laguna's: MIP time limit **clamped** to ≤ 120 s
  (default 60 s) whatever the project's `MIP_TIME_LIMIT`; targets mode with BLM > 0 refused above 500
  units (offer BLM 0 or budget mode); everything refused above 20 000 units or 60 000 unit × zone
  decisions. Thresholds come from the review's probes and are re-checked on laguna before release.
- Cancel only detaches the session (results of that solve are discarded, the button re-enables);
  the semaphore slot is released when the worker thread exits, so concurrent CBC runs stay bounded;
  CBC ends at its clamped time limit at the latest *(my call: killing CBC early needs a child
  process, not worth it with a clamp)*.
- Errors from `RecoveryInputError`/`ValueError` are shown verbatim (already planner-readable);
  anything else shows a generic message and is logged, never a traceback.

### 4.5 Language (E4)

All tab strings live in one `strings.py` table keyed by id with `en` and `lt` values; the tab's
EN/LT radio selects them; numbers rounded for display ("X of Y ha"). The rest of the app stays
English *(my call: no app-wide i18n in v0.50)*. The Lithuanian text must be reviewed by the user or a
native speaker before release, and should reuse the LT stakeholder deck's terms
(`docs/presentations/`).

### 4.6 Tests and acceptance

Pure helpers by ordinary pytest; source-text pins for wiring; an app-level test for **each** app that
imports it and evaluates its solver factory for every picker choice (would have caught P1) and that the
recovery tab is present; one playwright e2e test (worked example → solve → results table, marked
`integration`/`slow`). Acceptance on laguna as user `shiny`: worked example solves with the MIP, a
1751-unit project hits the time-limit banner, two concurrent sessions queue rather than stall.
Manual section "Nature recovery in the app" and CHANGELOG ship with v0.50.0.

## 5. Goals, non-goals, anti-goals

**Goals:** §2 and §3 in v0.49.0; §4 in v0.50.0.

**Non-goals:** scheduling restoration over periods (`pymarxan.temporal` models opportunity loss
and counts never-acted units as retained value — review H5; a restoration scheduler needs its own
design); MESH reporting (review H9); conditional spatial effects (Baker Fig. 3); scenario
robustness UI over `pymarxan.analysis.robustness`; chance constraints for zones (zone solvers
ignore PROBMODE); fixing the zone heuristic's no-move-to-0 and II's ITIMPTYPE-0 behaviour.

**Anti-goals:** presenting plans as guarantees (caveat always shown; shortfalls first);
silent ecological defaults (every placeholder labelled with provenance); changing existing
single-zone behaviour or existing zone results other than the documented zone-SA fix (§2);
ROWER-specific code or terms in pymarxan.

## 6. Constraints

Core UI-free and deterministic (problem-hash test over sorted DataFrames, excluding object
identity); existing tests stay green apart from deliberate zone-SA expectations; Python 3.11+,
type hints, strict TDD, ruff, mypy; zone-SA per-flip bench not regressed; ROWER's anchor and the
review's executed cases become tests.

## 7. Delivery

**v0.49.0 (core):** R1 zone SA "no action" fix (§2) + bench + picker text; R2 `RecoveryInputs`,
checks, builder, mapping (§3.1–3.3, §3.6) with anchor tests; R3 solving modes, solver registration,
spf scaling, heuristic ≥ MIP tests (§3.5, §3.9); R4 `RecoveryResult`, `success_sensitivity`,
caveat text (§3.7); R5 persistence (§3.8); R6 docs (manual section for the Python API),
CHANGELOG. **v0.49.1:** laguna `get_solver` crash fix + app-level tests (§4, E1). **v0.50.0 (Shiny):** §4.1 core prerequisites, then §4.2–4.6.

Process: this spec → prototype-first plan for v0.49.0 → plan review → SDD → release.

## 8. Left to the plan

v0.50.0: exact `RecoverySettings` fields and their mapping to `RecoveryInputs`; the editable-grid
widget for §4.3 item 3; colour values for the four actions (`ocean_palette.py`); shelf memory per
run; the Lithuanian review workflow (who reviews, when); file names inside the export ZIP;
ExtendedTask vs thread-and-poll for §4.4; test fixtures beyond the worked example and prioritizr.
