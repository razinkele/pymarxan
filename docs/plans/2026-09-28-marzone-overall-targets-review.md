# MarZone overall targets — four-lens design review (synthesis)

**Date:** 2026-09-28
**Reviewed artifact:** `2026-09-28-marzone-overall-targets-design.md` at commit `fef0944` (revision 1)
**Method:** Workflow `wf_b3b8001c-a12`, 10 agents: architect, executing grounding (built the §5
anchor against the current solvers, replayed `reserve.hpp:148-182` on `tests/data/zones`, ran SA/II
probes, checked every citation), scientific accuracy (Watts 2009 full text via scite plus the
fetched C++), independent re-design (wrote its own design before opening the spec), then six
adversarial refuters on the serious findings. Five of six confirmed by execution; one refuted as
"caused by the spec" but confirmed as a pre-existing bug.

**Outcome.** The core is right and was confirmed by execution: the current code solves the anchor
at (2, 2, 2) cost 3 with the overall target ignored; the independent oracle gives the unique
optimum (1, 2, 2) at 7, runner-up 8; MarZone's accumulation on the zones fixture matches the
spec's reading of the C++ (overall contribution-weighted, zone raw). The re-design converged on
the two-accumulator architecture, the parameter as the compatibility switch, and the
`targets_met` unification. What changes in revision 2 is everything at the seams: the anchor
must pin SPF, the heuristic's early exit must go, one shared solution builder must replace five
divergent ones, the contribution default must follow MarZone, and the summary writer must be
rewritten on the shared helpers.

## Findings and resolutions

### HIGH (all confirmed)

| # | Finding | Lenses | Resolution in rev 2 |
|---|---|---|---|
| H1 | Anchor never pins SPF; at the default 1.0 the penalised objective's minimum is the infeasible (2, 2, 2) at 6.0, so "heuristics ≥ 7 with every target met" is false. Threshold is spf > 4/3. | grounding, science, re-design | §5 pins `spf = 10` for the heuristic anchor and states the threshold; the oracle enumerates the penalised objective and asserts its argmin equals the hard optimum; a second assertion at spf = 1 pins the penalty arithmetic itself. |
| H2 | `ZoneHeuristicSolver` breaks its greedy loop as soon as zone targets are met; on the anchor it stops at (2, 0, 0) cost 1 with the overall target unmet. | architect, re-design | The early exit is removed (the loop already ends when no move improves the objective); a heuristic test on the anchor asserts both tiers met and cost ≥ 7. |
| H3 | Five `Solution(...)` construction sites with three different hand-rolled objective sums (some omit connectivity, one omits zone boundary); an anchor with BLM 0 cannot catch a site that forgets the overall term. | architect | New shared `build_zone_solution` in `objective.py`; all four solvers and `ZoneIISolver.improve` use it; a test asserts `sol.objective == compute_zone_objective(...)` for every solver on the fixture with BLM > 0 and boundary present. |
| H4 | Missing contribution rows default to 1.0 in pymarxan; MarZone zero-fills whenever a contribution file is supplied (`zones.hpp:619`) and uses all-ones only without one (`:651-668`). Load-bearing once the overall target is contribution-weighted. | science, re-design (grounding as MEDIUM) | Default becomes 0.0 when `zone_contributions` is supplied, 1.0 when absent; `validate()` lists missing pairs; CHANGELOG migration line; fixture re-pins recorded. |
| H5 | `write_zone_summary` computes per-zone contribution-weighted amounts against the overall target; matches neither tier. Refuter: the mismatch predates the spec (already disagrees with `check_zone_targets` today), so it is a pre-existing bug, not a spec consequence. | architect | Rewritten on the shared helpers with two row groups (overall per feature, zone per (zone, feature)); pinned against the objective module. |

### MEDIUM (accepted)

| # | Finding | Resolution |
|---|---|---|
| M1 | The "nnz trick" needs a CSR the zone cache lacks; dense row ops are faster at bench scale; an ungated overall term doubles per-flip cost (+109 % measured) and the absolute-budget bench cannot see it; an O(1) `contrib_differs[old][new]` gate costs +6 % and makes every default-contribution move free (as `reserve.hpp:393`). | Dense row ops; precomputed gate; comparative per-flip bench (targets zeroed vs populated, ratio ≤ 1.05 under default contributions). |
| M2 | Flag-dependent `held_per_zone` semantics and two lock-stepped arrays invite drift. | `held_per_zone` always raw; `ZONETARGETCONTRIB` applies a `zone_target_weight` matrix inside the penalty and delta only; both arrays bundled in a `ZoneHeld` state object that is a required keyword argument; identity Σ_z c[z,f]·held_raw[z,f] == held_overall[f] tested in both modes. |
| M3 | §4.5 re-proposes reader/writer code that already exists (`read_input_dat` auto-parses any key; `save_project` writes every parameter). | Replaced by a round-trip test plus `validate()`/cache rejection of values outside {0, 1}. |
| M4 | Contribution lookup has five-plus copies in two key orders; the `objectives/` copies are keyed backwards and silently use 1.0; `ZonalObjective` already formulates eq. 6 but is unwired. | `ZonalProblem.contribution_lookup()` + matrix builder shared by objective, MIP, cache, writers; the `objectives/` zone methods route through `compute_overall_achieved`. |
| M5 | `targets_met` unification incomplete: heuristic/II carry tuple-keyed zone dicts in `targets_met` and no metadata; consumers index by feature id (`calibration/spf.py:59` would crash on tuples); `all_targets_met` semantics for zone runs undefined. | All four solvers: `targets_met = check_overall_targets` (int keys) and `metadata["zone_targets_met"]` in the `z{z}_f{f}` form; `all_targets_met` documented as overall-only; base.py comment updated; agreement test across the four solvers. |
| M6 | "Unassigned = available zone, documented as equivalent" holds only under Watts' four conditions (zero cost, zero contribution, no zone target, matching boundary); MarZone's default available zone contributes 1. | §8 rewritten with the four conditions and a named deviation for projects without a contribution file. |
| M7 | `zonetarget.dat` `targettype` column: MarZone reads by position with types 0–3; pymarxan ignores the column, so a type-2 occurrence count would be read as area. | Reader accepts optional `targettype`: 0 as now, 1 → proportion of the feature's total raw amount, 2/3 → ValueError naming the row. The native MarZone column dialect stays a follow-on. |
| M8 | `ZoneIISolver` makes no moves unless ITIMPTYPE is set. | Anchor tests set `ITIMPTYPE = 3`; a separate test documents the unchanged-start behaviour at 0. |
| M9 | SPF × absolute shortfall is an approximation of MarZone's `spf × spec.penalty × Σ(shortfall/target)` (`reserve.hpp:816-872`, delta `:524`), and its unit-dependence mixes the two tiers. | Labelled "approximation" in §3.3; follow-on names `compute_baseline_penalty` as the existing building block. |
| M10 | MISSLEVEL enters MarZone only in `CountMissing` (reporting); pymarxan scales the penalty and MIP constraint too. | Stated as a pymarxan convention in §3 and in the VALIDATION.md subsection. |
| M11 | `target2`/clumping features take a different path in MarZone's accumulator. | `validate()` error for `target2 > 0` on zone problems; §8 follow-on. |

### LOW (accepted)

Citation fixes (`mip_solver.py:471-473` and the duplicate at `:410-412`; `solvers/cache.py:674-687`;
`reserve.hpp:193-275`); the redundant MIP slack: overall targets are added as hard constraints
only, existing zone-target slack left untouched and the hard-vs-soft distinction documented;
CHANGELOG uses the project's bold **BREAKING (zone targets only)** marker as in v0.6.0; test
docstrings that justify results with contribution-weighted zone values are rewritten with raw
numbers; 0.36.0 is consistent with project precedent.

## Rulings

- **Anchor SPF = 10** (heuristic tier), spf = 1 kept for the MIP/oracle tier: margin over the 4/3
  threshold, and the spf = 1 case is turned into an explicit penalty-arithmetic assertion.
  Cost if wrong: none for correctness; a different value only moves the margin.
- **Heuristic early exit removed** rather than extended: the only place the two tiers could drift
  apart again. Cost if wrong: a few more greedy iterations on problems whose zone targets are
  met early.
- **Contribution default follows MarZone** (0 when a table is supplied). Cost if wrong: projects
  with partial `zonecontrib.dat` files lose credit for unlisted zones; that is what MarZone does
  and the CHANGELOG says so.
- **Overall targets are hard constraints in the MIP with no slack term**; zone-target slack is out
  of scope. Cost if wrong: none observable (slack is provably zero under the hard constraint).

## Probes worth keeping

The grounding scripts under the scratchpad (`anchor.py`, `oracle_table.py`, `marzone_accum.py`,
`sa_run.py`, `ii2.py`, `flip_cost*.py`) are the seeds for the plan's tests: the oracle table,
the C++ accumulation cross-check, and the comparative per-flip bench.
