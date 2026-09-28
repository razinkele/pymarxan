# MarZone overall feature targets and raw zone targets — design

**Date:** 2026-09-28 (revision 2, after the four-lens review in
`2026-09-28-marzone-overall-targets-review.md`)
**Issue:** razinkele/pymarxan #2 (surfaced by the ROWER_pymarxan_DST spec review); also closes
the shortfall-helper half of #6
**Status:** reviewed; awaiting user approval, then `writing-plans`, then subagent-driven execution
**Target version:** 0.36.0

## 1. Problem

pymarxan's four zone solvers (`ZoneMIPSolver`, `ZoneSASolver`, `ZoneHeuristicSolver`,
`ZoneIISolver`) deviate from Marxan with Zones (MarZone; Watts et al. 2009,
doi:10.1016/j.envsoft.2009.06.005) in three ways, verified against the MarZone C++ source
(`Marxan-source-code/marzone`, `reserve.hpp`, `zones.hpp`) and by executing pymarxan:

1. **Overall feature targets are not enforced.** MarZone evaluates each species' `spec.dat`
   target on the contribution-weighted amount summed over every assigned zone:
   `speciesAmounts[isp].amount += amount × GetZoneContrib(...)` (`reserve.hpp:158-171`), Watts
   eq. 6, Σ_i Σ_k a_ij · ca_jk · x_ik ≥ t1_j. pymarxan's zone MIP enforces only `zone_targets`
   (`_add_zone_target_constraints`); `features.target` is reported post hoc by `check_targets`
   over any-zone selection without contributions (`mip_solver.py:498-499`) and is not a
   constraint. The SA cache penalty, the heuristic and the II solver score only zone targets.
   Executed: on the §5 anchor the current MIP returns (2, 2, 2) at cost 3 with `targets_met
   {1: True}` although the contribution-weighted amount is 12 < 15.
2. **Zone targets apply contributions.** MarZone accumulates zone-target amounts from raw
   `puvspr` amounts (`reserve.hpp:164`; delta path `:457, :464`), Watts eq. 7. pymarxan
   multiplies zone-target amounts by the contribution (`mip_solver.py:410-412` and `:471-473`,
   `objective.py:151-153`, `zones/cache.py:270`).
3. **Missing contribution rows default to 1.0.** MarZone zero-fills the contribution array
   whenever a contribution file is supplied and sets only the listed pairs (`zones.hpp:619-627`);
   the all-ones default applies only when no file exists (`zones.hpp:651-668`). pymarxan
   defaults unlisted (feature, zone) pairs to 1.0 everywhere.

Deviation 1 blocks any zoning in which several action zones share one feature target (the
ROWER active-versus-passive restoration split). Deviations 2 and 3 silently change results for
projects that pair `zone_contributions` with `zone_targets`; deviation 3 becomes load-bearing
the moment overall targets are contribution-weighted.

## 2. Decisions

| Decision | Choice | Rejected |
|---|---|---|
| Scope | Overall targets in all four zone solvers; zone targets on raw amounts; contribution default per MarZone; `targettype` column read (0, 1) and rejected (2, 3) | Overall targets only; occurrence targets; `zonetarget2.dat`; MarZone-native column dialect |
| Default | MarZone semantics; `ZONETARGETCONTRIB=1` restores contribution-weighted zone targets | Old behaviour by default |
| Architecture | Two accumulators in one `ZoneHeld` state (`per_zone` raw, `overall` contribution-weighted), mirroring MarZone's `zoneSpec` / `speciesAmounts`; one shared solution builder for all solvers | Flag-dependent `held_per_zone` semantics; synthetic zone rows; pluggable objective object |
| Penalty form | SPF × absolute shortfall for both tiers — an **approximation** of MarZone's `spf × spec.penalty × Σ(shortfall / target)` (`reserve.hpp:816-872`, delta `:524`); it drops the per-feature baseline penalty and the proportional normalisation, so the two tiers' terms are in habitat units, not comparable across features | MarZone's proportion-met scaling (follow-on, §8) |
| MIP form | Overall targets are hard constraints; existing zone-target slack left as is | Adding a slack for overall targets (provably zero under the hard constraint) |

## 3. Semantics

### 3.1 Contributions

`ZonalProblem.contribution_lookup() -> dict[tuple[int, int], float]` keyed `(feature, zone)`,
and `contribution_matrix()` of shape `(n_zones + 1, n_feat)` with row 0 (unassigned) all zero,
are the single source for every consumer (objective, both MIP builders, the cache, the writers,
the `objectives/` zone methods that today key the lookup backwards and silently use 1.0).
Default for an unlisted pair: **1.0 when `zone_contributions` is `None`; 0.0 when a table is
supplied** (MarZone `zones.hpp:619` / `:651`). `contribution_gaps()` lists unlisted pairs and
the loader warns, so a partial table is visible; `validate()` does not report them (advisory,
plan review M1).

### 3.2 Overall feature targets

- Source: `features.target` (proportional targets are resolved into it at read time by
  `io.readers._resolve_prop_targets`). Rows with `target <= 0` are inert. `targetocc` and
  `target2` are out of scope; `validate()` errors on `target2 > 0` in a zone problem.
- Achieved: `A_f = Σ_i amount[i, f] × contribution[f, z_i]` over PUs with `z_i > 0`.
- Met when `A_f ≥ target_f × MISSLEVEL`. Penalty `spf_f × max(0, target_f × MISSLEVEL − A_f)`.
- MIP: hard constraint `A_f ≥ target_f × MISSLEVEL` for every feature with target > 0.

### 3.3 Zone targets

- Achieved: `Z_kf = Σ_{i: z_i = k} amount[i, f] × w[k, f]` where `w` is the zone-target weight
  matrix: all ones by default; `= contribution_matrix()` when `ZONETARGETCONTRIB == 1`.
- Met when `Z_kf ≥ zone_target_kf × MISSLEVEL`; penalty `spf_f × max(0, …)`; MIP hard constraint
  plus the existing slack, unchanged apart from the weight.
- `ZONETARGETCONTRIB` lives in `problem.parameters` (auto-parsed from `input.dat` and
  auto-written by `save_project`; no reader/writer code). `validate()` and the cache reject
  values outside {0, 1}. It is a pymarxan extension MarZone does not read.
- `zonetarget.dat` may carry an optional `targettype` column: 0 → as now; 1 → target × the
  feature's total raw amount (`zones.hpp:149-150`); 2 or 3 → `ValueError` naming the row
  (occurrence targets unsupported).

### 3.4 MISSLEVEL

pymarxan applies MISSLEVEL to the met test, the penalty and the MIP constraint for both tiers,
as it already does for zone targets and in the single-zone cache. MarZone applies it only when
counting missing features (`CountMissing`, `reserve.hpp:193-275`); its penalties use the raw
target. This is a documented pymarxan convention, not parity.

### 3.5 Objective and reporting

`objective = zone_cost + BLM × standard_boundary + zone_boundary + overall_penalty +
zone_penalty + connectivity`. Every zone solver returns a `Solution` built by one shared
`build_zone_solution(problem, assignment, blm, *, solver_name, run=None)` in `objective.py`
(promoted from the MIP's private builder), so:
- `targets_met = check_overall_targets(...)` (`dict[int, bool]`, feature ids) for all four
  solvers; `all_targets_met` therefore means "overall targets met" for zone runs, documented in
  `solvers/base.py`;
- `metadata["zone_targets_met"]` in the `"z{zone}_f{feature}"` form for all four solvers;
- `metadata["overall_penalty"]`, `metadata["zone_penalty"]`; `penalty` and `shortfall` sum both
  tiers;
- `objective` equals `compute_zone_objective(...)` by construction.

The MIP is hard-constrained and returns `[]` when infeasible; the heuristics minimise the
penalised objective and return the best assignment found, feasible or not.

## 4. Changes by module

### 4.1 `zones/model.py`
`contribution_lookup()`, `contribution_matrix()`, `zone_target_weight_matrix()`; `validate()`
additions (parameter domain, `target2`, unlisted contribution pairs as advisory messages).

### 4.2 `zones/objective.py`
- `compute_overall_achieved`, `check_overall_targets`, `compute_overall_penalty`,
  `compute_overall_shortfall`, `compute_overall_shortfalls` (per feature),
  `compute_zone_shortfalls` (per (zone, feature); issue #6 part 1).
- `_compute_zone_achieved` uses the weight matrix (raw by default).
- `compute_zone_objective` adds `overall_penalty`.
- `build_zone_solution` (shared).

### 4.3 `zones/mip_solver.py`
- `_add_overall_target_constraints`; each achieved expression built once and reused where the
  penalty and constraint builders duplicate it today.
- Zone-target expressions use the weight matrix.
- Uses `build_zone_solution`; merges its `solver`/`status`/`mip_backend` keys as in v0.35.

### 4.4 `zones/cache.py` (`ZoneProblemCache`)
- `held_per_zone` is **always raw**. New `ZoneHeld` dataclass `(per_zone, overall)` returned by
  `compute_held(assignment)` and mutated by `update_held(state, idx, old_col, new_col)`.
  `compute_full_zone_objective` and `compute_delta_zone_objective` take `held: ZoneHeld` as a
  required keyword argument, so stale callers fail loudly.
- `overall_target_vector` (n_feat) = `features.target × MISSLEVEL` (0 where inert);
  `zone_target_weight` (n_zones+1, n_feat).
- Overall delta: dense row ops on the PU's row (no CSR), gated by a precomputed boolean
  `contrib_differs[old_col, new_col]` (any feature whose contribution differs between the two
  zones); under default contributions every zone-to-zone move skips the overall term, and the
  term is skipped entirely when `overall_target_vector` is all zero.
- Invariant tested in both modes: `Σ_z contribution[z, f] × held.per_zone[z, f] == held.overall[f]`.

### 4.5 Solvers
- `solver.py` (SA): carry `ZoneHeld`; all Solution construction through `build_zone_solution`.
- `heuristic.py`: the early exit on zone targets is **removed** (the loop ends when no move
  improves the objective); greedy scoring uses the combined penalty.
- `iterative_improvement.py` and `ZoneIISolver.improve`: shared builder; ITIMPTYPE semantics
  unchanged (0 returns the start assignment).

### 4.6 `zones/writers.py`
`write_zone_summary` rewritten on the shared helpers: an overall block (feature, target,
achieved, times_met) and a zone block (zone, feature, zone_target, achieved, times_met).

### 4.7 `zones/readers.py`
Optional `targettype` in `read_zone_targets` as in §3.3.

### 4.8 Docs
- `docs/VALIDATION.md`: "MarZone targets" subsection: the anchor, the C++ line references, and
  three named deviations (MISSLEVEL in penalties; unassigned-vs-available zone; penalty form).
- CHANGELOG: `### Changed` with `**BREAKING (zone targets only): zone targets now accumulate
  raw amounts; unlisted contribution pairs default to 0 when a contribution table is supplied;
  set ZONETARGETCONTRIB 1 to restore v0.35 zone-target behaviour**`, plus `### Added`.

## 5. Correctness anchor (hand-computed, confirmed by an independent oracle in review)

Three PUs, two zones, one feature. Amount 10 in every PU. Contributions listed for every pair:
zone 1 → 1.0, zone 2 → 0.4. Zone costs: zone 1 = (5, 6, 7), zone 2 = (1, 1, 1). Overall target
15. Zone target (zone 2, feature 1) = 10 raw. MISSLEVEL 1, BLM 0, no boundary.

| Assignment (PU1, PU2, PU3) | A_f | Z_2f (raw) | Feasible | Cost |
|---|---|---|---|---|
| (2, 2, 2) | 12 | 30 | no | 3 |
| (0, 2, 2) | 8 | 20 | no | 2 |
| (1, 2, 0) | 14 | 10 | no | 6 |
| **(1, 2, 2)** | **18** | **20** | **yes** | **7** |
| (2, 1, 2) | 18 | 20 | yes | 8 |
| (2, 2, 1) | 18 | 20 | yes | 9 |
| (1, 1, 2) | 24 | 10 | yes | 12 |

Unique feasible optimum **(1, 2, 2) at cost 7**, runner-up 8. Three behaviours are pinned:

1. **Current code** (before this change): (2, 2, 2) at cost 3, `targets_met {1: True}`.
2. **New semantics, MIP**: (1, 2, 2) at 7; `targets_met {1: True}`; `zone_targets_met
   {z2_f1: True}`.
3. **`ZONETARGETCONTRIB=1`**: zone 2 needs 3 PUs (3 × 4 = 12 ≥ 10), leaving A_f = 12 < 15:
   the MIP returns `[]`; the heuristics return a best-effort assignment with at least one tier
   unmet.

**SPF is part of the anchor.** The heuristics minimise `cost + spf × shortfall`, and at spf = 1
the penalised minimum is the infeasible (2, 2, 2) at 6.0 (threshold: spf > 4/3). The heuristic
tier of the anchor therefore uses **spf = 10** (penalised (2, 2, 2) = 33; the penalised argmin
equals the hard optimum), and a separate assertion at spf = 1 pins the penalty arithmetic
(objective 6.0 for (2, 2, 2), overall target unmet). II tests set `ITIMPTYPE = 3`.

The oracle enumerates all 27 assignments with the §3 formulas written without pymarxan,
computing both feasibility and the penalised objective, and asserts the penalised argmin equals
the hard optimum at spf = 10.

The single-zone anchor (`tests/data/simple`, 35.0) must not move. The zones fixture gets a
MarZone-accumulation cross-check: a 15-line reimplementation of `reserve.hpp:148-182` (for
`z_i > 0`; `z_i = 0` skipped by construction) equals `compute_overall_achieved` and the raw
per-zone amounts for three fixed assignments.

## 6. Tests

Mirror `src/`. Every new function has a test that failed first. Existing tests whose pinned
values change under raw zone targets or the contribution default (`test_zone_cache.py:150-160`
4.0 / 2.1 pins, `test_objective.py` docstrings quoting weighted values) are re-pinned with the
old value, the new value and the reason recorded in the commit.

- `test_model.py`: contribution lookup/matrix defaults (1.0 without a table, 0.0 for unlisted
  pairs with one); `validate()` messages; parameter domain.
- `test_objective.py`: overall achieved / met / penalty / shortfalls on the anchor; zone
  achieved raw by default and weighted under the parameter; `compute_zone_objective` includes
  both terms; MISSLEVEL scaling, met flags asserted separately from penalties;
  `build_zone_solution` fields.
- `test_zone_mip_overall.py`: the three anchor behaviours; `targets_met` feature-keyed;
  metadata keys; hard-constraint infeasibility returns `[]`.
- `test_zone_cache.py`: `held.per_zone` raw in both modes; identity invariant in both modes;
  delta equals full recomputation over random flips (`abs=1e-10`) in both modes; the
  `contrib_differs` gate skips the overall term under default contributions.
- Solver tests: SA (spf 10) reaches cost ≥ 7 with both tiers met; heuristic on the anchor ends
  with both tiers met (regression for the removed early exit); II with ITIMPTYPE 3 likewise,
  plus the ITIMPTYPE 0 unchanged-start test; all four solvers agree on `targets_met` keys and
  `metadata["zone_targets_met"]` keys; `sol.objective == compute_zone_objective(...)` for every
  solver on the fixture with BLM > 0 and boundary present.
- `test_writers.py`: `write_zone_summary` rows equal the objective module's numbers.
- `test_readers.py`: `targettype` 0 and 1 resolved, 2 and 3 rejected; `ZONETARGETCONTRIB`
  round-trips `load_zone_project → save_project → load_zone_project` as int.
- Bench (`bench` marker, comparative): per-flip delta with `features.target` populated vs
  zeroed on the generated bench problem under default contributions, ratio ≤ 1.05.
- `tests/test_examples.py` (parity harness) unchanged at 35.0 / 43.0 / 45.0.

## 7. Review record

Four-lens review run 2026-09-28 (`wf_b3b8001c-a12`); five HIGH findings confirmed by execution
and absorbed (SPF-pinned anchor, heuristic early exit removed, shared solution builder,
MarZone contribution default, summary writer rewritten); eleven MEDIUM accepted. Synthesis in
`2026-09-28-marzone-overall-targets-review.md`.

## 8. Out of scope / follow-ons

- Occurrence targets (`targetocc`; zone target types 2–3), `zonetarget2.dat`, and the
  MarZone-native column dialect (`zoneid,speciesid,fraction/multiplier`; positional reads,
  `zones.hpp:878-920`).
- MarZone's penalty form: `spf × penalty_f × Σ(shortfall / target)` (`reserve.hpp:816-872`,
  delta `:524`) with the per-feature baseline `penalty_f` computed once in
  `marzone.cpp:733-832` (`CalcPenalties`: the cheapest planning units on raw amounts needed to
  reach `max(target_f, Σ_k zone_target_fk)`, scaled up when unreachable);
  `compute_baseline_penalty` already exists as the building block. (`reserve.hpp:750-777`
  `GreedyPen` is the greedy heuristic's move score, not this baseline.)
- `target2` / clumping features in zone problems (`reserve.hpp:172-189`).
- Lock-into-a-named-zone status (issue #6 part 2).
- **Unassigned versus MarZone's "available" zone.** MarZone has no unassigned state. Watts et
  al. (2009) state that Marxan with Zones reduces to Marxan under three conditions: two zones,
  the unreserved zone contributing 0 and the reserved zone 1 for every feature; the unreserved
  zone costing 0 in every planning unit; and no zone-specific targets. A fourth, implicit
  condition — the zone connectivity matrix must reduce to Marxan's boundary term — is
  pymarxan's own statement. pymarxan's zone 0 satisfies the first three by construction, so a
  MarZone project that lists such an available zone reproduces; one **without** a contribution
  file (available zone contributing 1, `zones.hpp:658-662`) does not. Named deviation in
  VALIDATION.md.

## 9. References

- Watts, M. E., et al. (2009). Marxan with Zones. *Environmental Modelling & Software*, 24,
  1513–1521. https://doi.org/10.1016/j.envsoft.2009.06.005 (eq. 5–7; the reduction-to-Marxan
  conditions)
- MarZone C++: https://github.com/Marxan-source-code/marzone — `reserve.hpp` 148–190
  (accumulation), 193–275 (`CountMissing`, MISSLEVEL), 393 (contribution gate in the delta),
  751–775 (`GreedyPen`), 816–872 (objective penalty); `zones.hpp` 96–160 (`BuildZoneTarget`,
  target types), 619–627 and 651–668 (contribution defaults), 878–920 (`LoadZoneTarget`).
