# Nature recovery workflow — design (draft, 2026-11-03)

Status: **draft for review.** Written without a brainstorming interview (user: "skip the
questions, draft the spec"); every choice below not taken from an earlier user decision is
marked *(my call)* and is open to change. Inputs: the design note
`docs/plans/2026-11-03-nature-recovery-design-note.md` (from Baker et al. 2025, *BioScience*
75(6):472–489, doi:10.1093/biosci/biaf030), the sibling project ROWER_pymarxan_DST (its
`restoration_zones.build_zonal_problem` and its reviewed passive-restoration spec,
`docs/superpowers/specs/2026-09-29-passive-restoration-design.md`), and pymarxan v0.48.0.

## 1. Intent

Conservation planners increasingly have to decide **where to restore**, not only where to
protect, and how restoration competes with protection for one budget. pymarxan already has
every piece of the mathematics: Marxan-with-Zones problems with per-zone costs, contributions,
targets and "no action" (zone 0) units, exact and heuristic zone solvers, restoration pattern
indices (MESH) and dynamic scheduling. What is missing is a **workflow a planner can use in the
app**: turn an ordinary project plus a few recovery layers into a protect-or-restore problem,
solve it, and read the result honestly.

Success: a planner with a Marxan project and three extra pieces of information (where each
habitat could be restored and how much, how likely restoration is to succeed there, and what it
costs relative to protection) gets, in the app and without writing code, a map of which units
to protect, which to restore actively, which to leave to passive recovery and which to leave
alone; how far each habitat's protection and restoration targets are met; what each action
costs; and, optionally, an order in which to do the restoration over several budget periods.

## 2. Goals, non-goals, anti-goals

**Goals**

- G1. A pure core builder (`pymarxan.recovery`) that turns a `ConservationProblem` plus recovery
  inputs into a `ZonalProblem` — the ROWER formulation made generic (§3).
- G2. A "Nature recovery" Shiny tab: load recovery inputs, set targets and action parameters,
  build, solve with the existing zone solvers, view results (§4).
- G3. Recovery-specific results: per-action map and table, per-habitat target table split into
  existing (protected) and expected restored amounts, cost by action, MESH before / after when
  the project is a raster grid (§4.4).
- G4. Optional scheduling of the chosen restoration over budget periods with the existing
  `pymarxan.temporal.dynamic_reserve_greedy` (§4.5).
- G5. Export the built problem as a standard MarZone project (reproducible outside the app) and
  the results as CSV.
- G6. Documentation: a manual section, help pages, a worked example dataset.

**Non-goals (this release)**

- New optimisation mathematics: no new solver, objective or constraint type. Everything solves as
  an ordinary `ZonalProblem`.
- Conditional spatial effects (Baker et al. Fig. 3: an action's value depending on neighbouring
  actions) — a research item.
- Scenario robustness UI over `pymarxan.analysis.robustness` — its own panel, next on the roadmap.
- The MESH-maximising restoration optimiser (`greedy_mesh_restore`) as a planning mode — MESH is
  reported, not optimised, here *(my call: it answers a different question — connectivity of one
  habitat mask — and would need its own inputs)*.
- Chance constraints (PROBMODE 3–5) in zone problems — zone solvers ignore them (v0.48.0).
  Success probability enters as an **expected amount** (§3.3), as in ROWER.
- Two-stage stochastic scheduling (`two_stage_reserve_mip`) in the UI.

**Anti-goals (would make this a failure even if it works)**

- Presenting a plan as an outcome guarantee. Every result view states that restored amounts are
  *expected* amounts (potential × probability of success) and that the plan is a decision aid
  (ROWER register action A6; Baker et al.: outputs "are not intended to be prescriptive").
- Silent ecological defaults. Every parameter with a default (cost factor, passive efficacy) is
  shown in the UI with its value and a note that it is a placeholder to replace with local
  evidence.
- Changing the behaviour of existing single-zone or zone workflows, files or results.
- ROWER-specific code or terminology leaking into pymarxan (pressure layers, pilot names,
  evidence packages stay in ROWER).

## 3. Core: `pymarxan.recovery`

### 3.1 Inputs

A `RecoveryInputs` dataclass *(my call on names)*:

| Field | Meaning | Default |
|---|---|---|
| `base` | the `ConservationProblem`: planning units (id, cost, status), features (habitats), `pu_vs_features` = **existing** habitat amount per unit | required |
| `potential` | long table `pu, feature, amount`: habitat amount a unit would hold if restored | required |
| `p_success` | per-unit probability that active restoration succeeds, in [0, 1] | required (a scalar is broadcast) |
| `restore_cost` | per-unit cost of active restoration; else `restore_cost_factor × cost` | factor 2.5 (ROWER placeholder) |
| `passive` | optional passive-recovery action: `efficacy` η in (0, 1] (passive success = η × p_success), optional per-habitat overrides, `passive_cost_factor` | off; η 0.5, factor 0.2 when on *(my call for the factor)* |
| `restorable` | optional per-unit mask; a unit outside it cannot take a restoration action | all units with any potential |
| targets | per habitat: `protect_prop` of existing amount, and either `restore_prop` of potential or an absolute `restore_target` | 0.3 / 0.2 (ROWER placeholders) |
| `spf` | penalty factor for every split feature | 10 |

Units with `status` 2 (locked in) are locked into **protect**; status 3 (locked out) take no
action. Existing protected areas in poor condition (design note row 6) are handled by leaving
them `status` 0 with potential > 0, so the solver may choose restore or protect *(my call;
a dedicated "degraded protected area" status is deferred)*.

### 3.2 Zones

| Zone id | Name | Cost | Allowed where |
|---|---|---|---|
| 0 | no action | 0 | everywhere (except status 2) |
| 1 | protect | unit cost | everywhere |
| 2 | restore (active) | `restore_cost` | `restorable` units |
| 3 | restore (passive) — only when `passive` is on | `passive_cost_factor × cost` | `restorable` units |

"Allowed where" uses the existing per-unit zone rules (`puzone` / `PUZoneRule`, v0.38), with
zone 0 allowed (`may_be_unassigned`) so a unit can take no action.

### 3.3 Features, contributions and targets

Each habitat `h` becomes two features (ROWER ids: `2h − 1`, `2h`) *(the id scheme is the
builder's choice; it must be deterministic and invertible for reporting)*:

| Feature | Amount per unit | Contribution: protect | active | passive |
|---|---|---|---|---|
| `h_existing` | existing amount (from `pu_vs_features`) | 1 | 0 | 0 |
| `h_potential` | potential × p_success | 0 | 1 | η_h |

Zone targets: `protect` × `h_existing` = `protect_prop × Σ existing`; the restore target on
`h_potential` is shared by the active and passive zones through MarZone **overall** targets
(feature target with zone contributions, v0.36) when passive is on, and is a zone target on
zone 2 otherwise. Restore targets are stated in **raw potential units** and met in expected
units (`Σ selected potential × p × contribution`) — the user sees both.

The builder validates its inputs and returns errors as a list (like `validate()`): unknown
unit or habitat ids, negative amounts, p outside [0, 1], η outside (0, 1], restore targets
above what all restorable units could deliver (warning, not error), habitats with no potential.

### 3.4 Results

`RecoveryResult` *(my call)* from a zone `Solution`: action per unit; per habitat the existing
amount protected, the expected amount restored (active and passive), targets and shortfalls;
cost by action; and, for grid projects, MESH of the habitat mask before and after (existing ∪
units restored, active or passive, as habitat; per habitat when the potential is per habitat —
*builder to decide whether MESH is per habitat or on the union*).

### 3.5 Scheduling

`schedule_recovery(result, budgets, loss_prob=None)` wraps `dynamic_reserve_greedy` over the
units chosen for restoration: value = the unit's expected restored contribution summed over
habitats (weights equal, *my call*), cost = its action cost, `loss_prob` = the per-period chance
the opportunity is lost if not acted on (default: the project's Probability 1D `prob` if present,
else a single user-set value). Output: the period in which each unit is restored and the expected
retained value.

## 4. Shiny: "Nature recovery" tab

A new tab in both apps (`pymarxan_shiny/app.py`, which laguna.ku.lt serves, and
`pymarxan_app/app.py`), from new modules under `pymarxan_shiny/modules/recovery/`, following the
existing pattern (pure helpers unit-tested, thin reactive wiring; every write builds a new
problem — v0.48.0 review finding).

1. **Inputs.** Uses the project loaded in the Data tab as the base. Upload a potential table
   (CSV, `pu, feature, amount`) and a per-unit table (CSV, `pu, p_success` and optional
   `restore_cost`, `restorable`). A summary shows how many units are restorable and the total
   potential per habitat; builder errors and warnings are listed.
2. **Actions and targets.** Protect and restore targets per habitat (proportions, with an
   absolute-target option), restore cost factor, passive action on/off with η (default and per
   habitat) and its cost factor. Placeholder defaults are labelled as such.
3. **Build and solve.** "Build recovery problem" shows the resulting zones / features / targets
   summary; "Solve" runs a zone solver (MIP when available, else zone SA + iterative
   improvement — *my call*) through the existing run machinery.
4. **Results.** Map coloured by action (no action / protect / active / passive); target table per
   habitat with existing-protected and expected-restored columns; cost by action; MESH before /
   after for grid projects; the expected-amount caveat always visible.
5. **Schedule (optional).** Per-period budgets → restoration order map and table.
6. **Export.** MarZone project files (zones, zone costs, contributions, targets, puvspr, spec,
   pu, input.dat) and result CSVs.

The tab works on the laguna deployment without extra dependencies; MESH needs a grid project and
hides otherwise.

## 5. Constraints

- The core stays UI-free; the builder is a pure function of its inputs (deterministic output for
  the same inputs — ROWER pins a problem hash; a similar test is expected).
- No behaviour change to existing modules; existing zone and single-zone tests stay green.
- Python 3.11+, `from __future__ import annotations`, full type hints, strict TDD, ruff, mypy.
- Correctness anchors: ROWER's 4-unit anchor (unique optimum `[1, 0, 0, 2]`, cost 6.0) is
  reproduced by the generic builder; a hand-checkable passive example (an η that makes passive
  cheaper per expected unit wins, and the reverse) is pinned.
- Performance: the builder is O(units × habitats); building a 100k-unit grid problem must not
  dominate the solve.

## 6. Decided vs left to the builder

**Decided (by earlier user decisions or project practice):** zones as actions (design note,
user-approved 2026-11-03); success as expected amount, not a chance constraint (zone solvers
ignore PROBMODE); generic core + thin Shiny modules; outputs never presented as guarantees.

**Marked *(my call)* above, open to review:** module and type names; the passive cost factor
0.2; the solver default; equal habitat weights in scheduling; the no-new-status treatment of
degraded protected areas; MESH reported but not optimised.

**Left to the builder:** feature id scheme; file formats of the two uploads beyond the stated
columns (delimiter detection as elsewhere); MESH per habitat vs union; whether `RecoveryResult`
lives in `pymarxan.recovery` or `pymarxan.analysis`; the exact layout of the tab; the example
dataset (small, synthetic, grid-based so MESH shows).

## 7. Delivery (proposed)

One minor release (v0.49.0), in order: R1 core builder + checks + anchor tests; R2 results +
MESH; R3 scheduling wrapper; R4 Shiny inputs / build / solve; R5 Shiny results / schedule /
export; R6 example dataset, manual, help, CHANGELOG. Process as usual: spec review →
prototype-first plan → review → SDD → release.
