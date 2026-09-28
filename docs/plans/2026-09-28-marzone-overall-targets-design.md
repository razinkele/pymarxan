# MarZone overall feature targets and raw zone targets — design

**Date:** 2026-09-28
**Issue:** razinkele/pymarxan #2 (surfaced by the ROWER_pymarxan_DST spec review)
**Status:** approved in brainstorm; awaiting written-spec review, then `writing-plans`, then the
four-lens Workflow design review, then subagent-driven execution
**Target version:** 0.36.0

## 1. Problem

pymarxan's four zone solvers (`ZoneMIPSolver`, `ZoneSASolver`, `ZoneHeuristicSolver`,
`ZoneIISolver`) deviate from Marxan with Zones (MarZone; Watts et al. 2009,
doi:10.1016/j.envsoft.2009.06.005) in two ways, both verified against the MarZone C++ source
(`Marxan-source-code/marzone`, `reserve.hpp`, `zones.hpp`) during the ROWER review:

1. **Overall feature targets are not enforced.** MarZone evaluates each species' `spec.dat`
   target on the contribution-weighted amount summed over every assigned zone:
   `speciesAmounts[isp].amount += amount × GetZoneContrib(...)` (`reserve.hpp:158-171`), i.e.
   Watts eq. 6, Σ_i Σ_k a_ij · ca_jk · x_ik ≥ t1_j. pymarxan's zone MIP enforces only
   `zone_targets` (`_add_zone_target_constraints`); `features.target` is reported post hoc by
   `check_targets` over any-zone selection without contributions (`mip_solver.py:498-499`) and
   is not a constraint. The SA cache penalty (`ZoneProblemCache._compute_zone_penalty`) and the
   heuristic / II solvers likewise score only zone targets.
2. **Zone targets apply contributions.** MarZone accumulates zone-target amounts from raw
   `puvspr` amounts: `zoneSpec[...].amount += pu.puvspr[ism].amount` (`reserve.hpp:164`; delta
   path `:457, :464`), Watts eq. 7. pymarxan multiplies zone-target amounts by the contribution
   (`mip_solver.py:465-470`, `objective.py:151-153`, `cache.py:270`).

Deviation 1 blocks any zoning in which several action zones share one feature target (the
ROWER active-versus-passive restoration split). Deviation 2 silently changes results for every
project that pairs `zone_contributions` with `zone_targets`.

## 2. Decisions taken in the brainstorm

| Decision | Choice | Rejected |
|---|---|---|
| Scope | Overall targets in all four zone solvers + zone targets on raw amounts | Overall targets only; full MarZone target model (target types 0–3, `zonetarget2.dat`, occurrence targets) |
| Default | MarZone semantics (raw zone targets) by default; `ZONETARGETCONTRIB=1` restores the old behaviour | Old behaviour by default with opt-in parity |
| Architecture | Two accumulators in `ZoneProblemCache` (`held_zone` raw, `held_overall` contribution-weighted), mirroring MarZone's `zoneSpec` / `speciesAmounts` | Synthetic zone rows (cannot express a cross-zone sum); pluggable objective object |
| Penalty form | SPF × shortfall for overall targets, the form the zone solvers already use for zone targets | MarZone's proportion-met scaling (a separate, deeper parity item) |

## 3. Semantics

### 3.1 Overall feature targets

- Source: `features.target` (proportional targets are already resolved into this column at read
  time by `io.readers._resolve_prop_targets`, so no further resolution here). Rows with
  `target <= 0` are inert. `targetocc` is out of scope.
- Achieved for feature f under assignment z:
  `A_f = Σ_i amount[i, f] × contribution[f, z_i]` over PUs with `z_i > 0`, where a missing
  contribution row defaults to 1.0 (unchanged convention) and unassigned PUs contribute 0.
- Met when `A_f ≥ target_f × MISSLEVEL` (MISSLEVEL default 1.0, same as everywhere else).
- Penalty term: `spf_f × max(0, target_f × MISSLEVEL − A_f)`.
- MIP: hard constraint `A_f ≥ target_f × MISSLEVEL` plus the slack-penalty term, exactly as
  zone targets are treated today (constraint and penalty; the penalty is redundant at any
  feasible point but keeps the objective form uniform).

### 3.2 Zone targets

- Achieved for (zone k, feature f): `Z_kf = Σ_{i: z_i = k} amount[i, f]` — raw amounts.
- Met when `Z_kf ≥ zone_target_kf × MISSLEVEL`; penalty `spf_f × max(0, …)`; MIP hard
  constraint plus slack, as today, minus the contribution factor.
- **Compatibility parameter `ZONETARGETCONTRIB`** (int, `problem.parameters`; read from
  `input.dat` like `MISSLEVEL`; default 0). When 1, `Z_kf` uses
  `amount × contribution[f, k]` as in v0.35 and earlier. The parameter affects zone targets
  only; overall targets are always contribution-weighted.

### 3.3 Objective

`objective = zone_cost + BLM × standard_boundary + zone_boundary + overall_penalty +
zone_penalty + connectivity`. The two penalties are reported separately in
`Solution.metadata` (`overall_penalty`, `zone_penalty`) and summed into `Solution.penalty`.

## 4. Changes by module

### 4.1 `zones/objective.py`
- `_contribution_lookup(problem)` shared helper (the three copies in objective/mip_solver
  collapse into it).
- `compute_overall_achieved(problem, assignment) -> dict[int, float]`.
- `check_overall_targets(problem, assignment) -> dict[int, bool]` (MISSLEVEL applied).
- `compute_overall_penalty(problem, assignment) -> float` and
  `compute_overall_shortfall(problem, assignment) -> float`.
- `_compute_zone_achieved` honours `ZONETARGETCONTRIB` (raw by default).
- `compute_zone_objective` adds `overall_penalty`.
- New public `compute_zone_shortfalls(problem, assignment) -> dict[tuple[int, int], float]`
  and `compute_overall_shortfalls(problem, assignment) -> dict[int, float]` (issue #6 part 1;
  cheap once the accumulators exist).

### 4.2 `zones/mip_solver.py`
- `_add_overall_target_constraints` and the overall slack term in `_build_penalty_expr`.
- Zone-target expressions drop the contribution factor unless `ZONETARGETCONTRIB == 1`.
- `_build_zone_solution`: `targets_met = check_overall_targets(...)` replaces the raw
  any-zone `check_targets` call; `penalty` and `shortfall` include both terms; metadata gains
  `overall_penalty` / `zone_penalty`.

### 4.3 `zones/cache.py` (`ZoneProblemCache`)
- Existing `contribution_matrix` (n_zones+1, n_feat) stays.
- `held_per_zone` becomes **raw** (drop the contribution multiply in `compute_held_per_zone`
  and `update_held_per_zone`) unless `ZONETARGETCONTRIB == 1`, in which case the old
  multiply applies. Store the flag on the cache (`zone_target_contrib: bool`).
- New `held_overall` (n_feat): `Σ_i pu_feat[i] × contribution_matrix[z_col_i]`, with
  `update_held_overall(held_overall, idx, old_col, new_col)` applying
  `pu_feat[idx] × (contribution[new] − contribution[old])`.
- New `overall_target_vector` (n_feat) = `features.target × MISSLEVEL`, zero where inert.
- `_compute_zone_penalty` → two terms; `_penalty_delta` gains the overall delta, computed
  only over the PU's nonzero feature columns (same trick as `cache.py:670-686` in the
  single-zone cache).
- `compute_full_zone_objective` / `compute_delta_zone_objective` signatures gain
  `held_overall` (keyword, so callers update explicitly).

### 4.4 `zones/solver.py`, `zones/heuristic.py`, `zones/iterative_improvement.py`
- SA: carry `held_overall` alongside `held_per_zone`, pass both to the delta path, apply
  both updates on acceptance. Report `targets_met = check_overall_targets(...)` (today `{}`).
- Heuristic / II: use the shared objective functions; heuristic's greedy step scores the
  combined penalty; both report the overall `targets_met`.

### 4.5 Readers / writers
- `io.readers` accepts `ZONETARGETCONTRIB` in `input.dat` (integer). Writers emit it when
  present. No new files; `zonetarget.dat` format unchanged.

### 4.6 Docs
- `docs/VALIDATION.md`: a "MarZone targets" subsection with the anchor below and the C++ line
  references. CHANGELOG `### Changed` (behaviour change, migration line) and `### Added`.

## 5. Correctness anchor (hand-computed)

Three PUs, two zones, one feature. Amount 10 in every PU. Contributions: zone 1 → 1.0,
zone 2 → 0.4. Zone costs: zone 1 = (5, 6, 7), zone 2 = (1, 1, 1). Overall target 15.
Zone target: (zone 2, feature 1) = 10 raw. MISSLEVEL 1, BLM 0.

| Assignment (PU1, PU2, PU3) | A_f (overall) | Z_2f (raw) | Feasible | Cost |
|---|---|---|---|---|
| (2, 2, 2) | 12 | 30 | no (12 < 15) | 3 |
| (0, 2, 2) | 8 | 20 | no | 2 |
| (1, 2, 0) | 14 | 10 | no (14 < 15) | 6 |
| **(1, 2, 2)** | **18** | **20** | **yes** | **7** |
| (2, 1, 2) | 18 | 20 | yes | 8 |
| (1, 1, 2) | 24 | 10 | yes | 12 |
| (1, 1, 0) | 20 | 0 | no (zone target) | 11 |

Unique optimum **(1, 2, 2) at cost 7**; runner-up 8. Under the pre-change semantics the same
tables give (2, 2, 2) at cost 3, because the overall target is ignored and the
contribution-weighted zone target (3 × 4 = 12 ≥ 10) is met. Under `ZONETARGETCONTRIB=1` with
overall targets enforced the problem is infeasible (zone 2 needs all three PUs, leaving A_f =
12): the MIP returns `[]` and the SA reports the overall target unmet. All three outcomes are
asserted by tests, and the enumeration over 3³ = 27 assignments uses an oracle written without
pymarxan.

The single-zone anchor (`tests/data/simple`, cost 35.0) must not move; the zone fixture
`tests/data/zones` gets a MarZone-accumulation cross-check: for a fixed assignment, `A_f` and
`Z_kf` computed by a 10-line reimplementation of `reserve.hpp:158-171` equal the objective
module's values.

## 6. Tests

Mirror `src/`. Every new function has a test that failed first.

- `test_objective.py`: overall achieved / met / penalty / shortfall on the anchor; zone
  achieved raw by default and contribution-weighted under the parameter; `compute_zone_objective`
  includes the overall term; MISSLEVEL scaling for both.
- `test_zone_mip.py` / new `test_zone_mip_overall.py`: the anchor's unique optimum, the
  parameter's infeasible outcome (`[]`), `targets_met` is the overall check, metadata carries
  both penalties, existing tests re-pinned where zone-target semantics changed (each change
  recorded with the old and new value and why).
- `test_zone_cache.py`: `held_per_zone` raw (the pinned 2.1 / 4.0 style values are re-pinned
  with rulings), `held_overall` full vs incremental agreement over random flips (`abs=1e-10`),
  delta path equals full recomputation; parameter path preserves the old pins.
- `test_solver.py` (SA), heuristic, II: on the anchor, SA/heuristic/II cost is at or above 7
  with every target met; a heuristic below the MIP is a bug, never a better solver.
- MarZone accumulation cross-check on `tests/data/zones`.
- `tests/test_examples.py` (parity harness) unchanged at 35.0 / 43.0 / 45.0.

## 7. Review before implementation

Four-lens Workflow review on this spec: architect (cache/delta invariants, signature changes,
the three-copy contribution lookup), grounding (must EXECUTE the anchor through the current
objective functions to confirm the "before" numbers, and run the C++-derived accumulation on
the fixture), science (Watts 2009 eq. 5–7 and the C++ lines; whether SPF × shortfall for
overall targets is an acceptable stand-in for MarZone's proportion-met penalty in this phase),
independent re-design.

## 8. Out of scope / follow-ons

- Zone target types 0–3 and `zonetarget2.dat` (`zones.hpp:147-154`, `LoadZoneTarget`).
- Occurrence targets (`targetocc`, zone occurrence targets).
- MarZone's proportion-met penalty scaling and `spec.penalty` cost-based penalty.
- Lock-into-a-named-zone status (issue #6 part 2).
- Watts eq. 5's mandatory "available" zone (`Σ_k x_ik = 1`); pymarxan keeps `≤ 1` with
  unassigned = available, documented as equivalent.

## 9. References

- Watts, M. E., et al. (2009). Marxan with Zones. *Environmental Modelling & Software*, 24,
  1513–1521. https://doi.org/10.1016/j.envsoft.2009.06.005 (eq. 5–7)
- MarZone C++: https://github.com/Marxan-source-code/marzone — `reserve.hpp` 150–190
  (accumulation), 196–270 (targets-met counting with MISSLEVEL), `zones.hpp` 96–160
  (`BuildZoneTarget`, target types), 878–920 (`LoadZoneTarget`).
