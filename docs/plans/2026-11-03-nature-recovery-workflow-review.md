# Nature recovery workflow — four-lens design review (2026-11-03)

Draft reviewed: `docs/plans/2026-11-03-nature-recovery-workflow-design.md` (commit a08615f).
Lenses: architect, codebase grounding (executed the formulation against pymarxan 0.48.0 with
exhaustive brute force on 4-unit problems), scientific accuracy (scite; 20 papers retrieved),
independent re-design (designed from the requirements before reading the draft). The controller
re-ran the decisive probes (zone-solver test, grounding cases a/b1) and checked the ROWER
passive spec; results below are confirmed unless marked.

Verdict: the core idea — Marxan-with-Zones zones as recovery actions, success as an expected
amount, a generic core builder — is sound and grounded (Baker et al. 2025; Watts et al. 2009;
Wilson et al. 2011; ROWER anchor reproduces: zone MIP `[1,0,0,2]` cost 6.0). The draft's
**model details, defaults and app integration are wrong in several places**, and its scope is
too large for one release.

## CRITICAL

| # | Finding | Lenses | Evidence |
|---|---|---|---|
| C1 | `ZoneSASolver` never proposes zone 0, so it cannot output "no action"; bare zone SA protects/restores every free unit. II with the default ITIMPTYPE 0 is a no-op returning the all-protect seed; the zone heuristic never moves a unit back to 0. The draft's fallback "zone SA + iterative improvement" is wrong. | architect, grounding (confirmed) | `zones/solver.py:18-25, 96-100, 138-144`; probe: SA `[1,1,1,1]` 4.0 vs MIP `[1,0,0,0]` 1.0; anchor SA 8.0 vs 6.0 |
| C2 | Listing "0 no action" as a zone conflicts with the model (zone 0 is the implicit unassigned state; a listed id 0 needs zone_costs rows and makes status-2 lock into it). | architect | `zones/model.py:120-122, 273, 294-295`; `zones/cache.py:53-67` |
| C3 | The restorable mask via puzone is impossible: a puzone unit cannot be unassigned, zone-0 rows are rejected, a single-zone puzone is a validate() error. Every restorable unit would be forced to act (2.6 vs 1.6). | architect, grounding (executed) | `zones/model.py:258-267, 290-292, 345-352, 409-416`; pu-locks design `2026-10-11…:5` |

## HIGH

| # | Finding | Lenses |
|---|---|---|
| H1 | **No zone path exists in either app**: `zone_problem` is created but never set; no zone loader in Shiny; registry offers only `zone_sa`; run panel solves `problem()`; no module reads `zone_assignment`. "Solve through the existing run machinery" is false; a dedicated recovery solve/results path, `zone_mip` registration and base/built state separation are new work. (confirmed by controller grep) | all three code lenses |
| H2 | **Restoring wipes existing habitat**: `h_existing` contribution 0 under restore zones drops a unit's existing amount when it is restored; ROWER avoided it only because its inputs are disjoint. Credit `h_existing` 1 under every action zone and define `potential` as the **incremental** amount restoration adds. | science, re-design |
| H3 | **Mixed target units**: restore targets raw (prop × Σ potential), achievement expected (Σ potential × p × c); any restore_prop above the amount-weighted mean p is infeasible (executed: target 14 vs max 13.6 → MIP infeasible). State restore targets in expected units; reuse `diagnose_zone_infeasibility` (already correct) for the warning. | grounding, science, re-design |
| H4 | **Passive cost factor 0.2 contradicts the user's ROWER decision U2 (cost_factor 1.0, η 0.5)** and makes passive ~6× cheaper per expected unit than active everywhere (and cheaper than protection); ROWER also rejects `passive cost ≥ active cost`. (confirmed in ROWER spec) Literature: Strassburg 2019/2020, Possingham 2015, Zahawi 2014. | science, re-design |
| H5 | **Scheduling is mis-modelled**: `dynamic_reserve_greedy` counts a never-acted unit as retained value (8.04 vs 4.8 executed); its loss model is opportunity loss, while the draft defaults `loss_prob` to Probability 1D `prob` = P(action fails), double-counting failure and inverting McBride et al. 2007's guidance; neither temporal function models restoration lag. | grounding, science, re-design, architect |
| H6 | **Restoration gaps / time lag not modelled** (Baker Fig. 2c; Rey Benayas 2009; Moreno-Mateos 2012; Jones 2018). Cheap fix: optional per-habitat maturity/gap factor g_h ∈ (0, 1] on restore contributions (active g_h, passive η_h·g_h), plus an assessment horizon H; never sum existing and expected-restored into one total. | science, re-design |
| H7 | **Heuristics report plans below the MIP optimum with the restore target unmet** (b1: II 3.9 vs MIP 6.0; confirmed). spf × absolute shortfall is unit-dependent, so spf 10 is uncalibrated. Results must lead with shortfall; spf must be scaled (e.g. relative to cost per expected unit) or calibrated; tests pin heuristic ≥ MIP. | grounding |
| H8 | **"One budget" not addressed**: only min-set. Add a budget mode (zone MIP `min_shortfall` / `max_weighted_features` + `COSTBUDGET`; UI notes boundary terms are dropped). | re-design (verified `zones/mip_solver.py:82, 251`) |
| H9 | **MESH can never appear in the app** (no Shiny path creates grid projects; `load_zone_project` drops `grid`); on the union of habitats it is ecologically meaningless; the "after" mask treats restoration as certain. | architect, grounding, science, re-design |
| H10 | **Persistence**: no `save_zone_project`; MarZone export loses raw potential, p, η and the habitat↔feature map, so results can't be split back; MarZone C++ has no "unassigned" concept → say "reproducible in pymarxan". Add `save_zone_project` and a recovery sidecar (inputs + mapping). | architect, grounding |

## MEDIUM

- M1 Status 2 → protect holds only because protect is `min(zone_ids)`; write explicit locks. For status-2 units **with potential**, allow {protect, restore} (puzone with ≥2 zones forbids unassigned — exactly the wanted rule), so a legal PA is never dropped (re-design, science M6).
- M2 Always use overall targets (one code path; results unchanged when passive toggles) (architect, re-design).
- M3 Restorability by zero potential works (c4) but SA may still place a restore zone on a zero-potential unit; with C1 fixed, restore cost > 0 makes it dominated; also require restore cost > 0.
- M4 Feature-id scheme and habitat mapping must be explicit output (`RecoveryMapping`), not left to the builder.
- M5 Other tabs can corrupt the built problem (Features/Cost/Probability tabs `copy_with` the ZonalProblem); keep `recovery_base` and a derived built problem.
- M6 Restore cost as one multiplier of land cost is weak (Bayraktarov 2016: costs differ by orders of magnitude between ecosystems); support a per-habitat additive implementation cost, factor as fallback.
- M7 Separate protect/restore targets fix the mix; "competes for one budget" only really holds in budget mode (H8); soften §1 accordingly (science M1).
- M8 Zone heuristic/II evaluate the full objective per move (O(n²) per pass); at large n only MIP scales until C1 is fixed; add a size/time-limit guard for laguna.
- M9 Reuse `RestorationProblem.habitat_mask` / `compute_mesh` if MESH is kept anywhere.
- M10 Uncertainty beyond a caveat: re-score the fixed plan at scaled success rates (p × 0.5…1.25) and report the lowest multiplier at which each target is still met (no re-solve) (re-design).

## LOW

- Placeholders 0.3 / 0.2 targets fine if labelled policy-style percentages; η 0.5 needs `provenance: placeholder` as in ROWER; η > 1 only as "reported but contested" (Crouzeilles 2017 vs Reid 2018).
- Export readable feature names `<habitat>_existing` / `_potential`.
- Solver picker help says zone SA can leave units unassigned — false; fix the text.
- Baker et al. claims verified, no overstatement (science L3).
- ROWER docstring "overall targets are issue #2" is stale (shipped in v0.36).

## Proposed caveat wording (science lens)

"Restored amounts are expected values: potential × probability that restoration succeeds
(passive: × efficacy η), judged at horizon H — often only 1–2-year establishment survival.
Targets are met on average, not with any stated probability: a target met by many
low-probability units may well fall short in practice. Restored habitat usually stays below the
quality of existing habitat for decades; [no time lag is modelled | a gap factor g of … is
applied]. Costs, η and the gap factor are placeholders unless replaced with local evidence. This
plan is a decision aid for stakeholder discussion, not a prescription."

## Decisions needed from the user

D1 scope split; D2 passive in v1; D3 scheduling and MESH in v1; D4 fix zone SA "no action" in
pymarxan vs work around it. (Recorded with answers in the revised spec.)

## References (retrieved through scite by the science lens)

Baker et al. 2025 (10.1093/biosci/biaf030); Bayraktarov et al. 2016 (10.1890/15-1077); Costello &
Polasky 2004 (10.1016/j.reseneeco.2003.11.005, metadata only); Crouzeilles et al. 2017
(10.1126/sciadv.1701345); Game et al. 2008 (10.1890/07-1027.1); Holl & Aide 2011
(10.1016/j.foreco.2010.07.004); Jaeger 2000 (10.1023/A:1008129329289); Jones et al. 2018
(10.1098/rspb.2017.2577); Justeau-Allaire et al. 2023 (10.1111/rec.13910); McBride et al. 2007
(10.1111/j.1523-1739.2007.00832.x); Moreno-Mateos et al. 2012 (10.1371/journal.pbio.1001247);
Moser et al. 2007 (10.1007/s10980-006-9023-0); Possingham et al. 2015
(10.1371/journal.pbio.1002052); Reid et al. 2018 (10.1126/sciadv.aas9143); Rey Benayas et al. 2009
(10.1126/science.1172460); Strassburg et al. 2019 (10.1038/s41559-018-0743-8, **has correction**);
Strassburg et al. 2020 (10.1038/s41586-020-2784-9, **has correction**); Watts et al. 2009
(10.1016/j.envsoft.2009.06.005); Wilson et al. 2011 (10.1111/j.1365-2664.2011.01975.x); Zahawi
et al. 2014 (10.1111/rec.12098).
