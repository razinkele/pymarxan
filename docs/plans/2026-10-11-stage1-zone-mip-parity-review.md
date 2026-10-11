# Stage 1 (v0.39.0) zone-MIP parity — 4-lens design review

Spec under review: `docs/plans/2026-10-11-stage1-zone-mip-parity-design.md` (rev 1, `06fd0e8`).
Lenses: architect, codebase grounding (executed under the `shiny` env, CBC only), scientific
accuracy (prioritizr 9.0.1 source @ f2f5628, MarZone C++ source, Watts et al. 2009
doi:10.1016/j.envsoft.2009.06.005 — scite: exists, no notices), independent re-design.

Findings marked **[n×]** were raised independently by n lenses. **P** = parity-critical.

## Confirmed sound (no change)

- D1 one term per target, D2 single total budget, D7 relative shortfall: match prioritizr source
  (`rcpp_apply_min_shortfall_objective.cpp`, `..._min_largest_shortfall_...`).
- D3 single problem parameter for mandatory allocation (better than the roadmap's kwarg+param).
- `pu_zone_rules()` is the right seam: grounding emulated MANDATORYALLOCATION=1 via pu_zones and
  MIP/SA/heuristic/II all returned zero rule violations — "no solver code changes" holds.
- Proposed zone formulations are linear and feasible; prototype builders written to the spec
  table matched brute-force enumeration at every budget tested.
- Every code reference in the spec exists as described (table in grounding report).
- `read_input_dat`/`write_input_dat` round-trip unknown keys; LP capture by patching
  `LpProblem.solve` works.

## CRITICAL

None.

## HIGH

| # | Finding | Lenses | Resolution |
|---|---|---|---|
| H1 | **Existing `pymarxan.objectives` package** (`Objective`/`ZonalObjective` ABCs with `build_zone_mip_*`, `MinShortfallObjective` etc.) has no consumers; its `min_shortfall` is absolute/unweighted, reads `BUDGET` not `COSTBUDGET`, and its zone builders create slacks with no defining constraints. Shiny `objective_selector` writes `cfg["objective_type"]`, read by nothing. D5 adds a third mechanism without mentioning it. | grounding | **User decision** (see below). |
| H2 **P** | **Zero-occurrence features silently met** in core `MIPSolver`: target constraints only inside `if fid in feat_groups.groups` (mip_solver.py:474-478). Executed: target-100 feature with no rows → no slack constraint, max shortfall reported 18 not 100. New core `min_shortfall` would inherit it; zone side is correct (empty lpSum → s ≥ R), so same-named objectives would disagree. | [3×] arch, grounding, redesign | One `TargetTerm` rule in both solvers: every target>0 feature is a term, A_t = 0 if absent. Fix core for all slack objectives; test in both. |
| H3 **P** | **Relative divisor inconsistent**: zone table divides by R_t = target×MISSLEVEL, MIPSolver section by `target_j`; core slacks already measure against target×MISSLEVEL. MISSLEVEL=0 or target≤0 → divide-by-zero; skip stated only for `min_shortfall`. | arch, grounding | R_t = target×MISSLEVEL in both; skip R_t ≤ 0 in **every** objective; test with MISSLEVEL<1. |
| H4 | **`min_penalties` big-M not lexicographic**: M = Σ max cost + 1 only dominates when w·δs ≥ 1. Fractional amounts / SPF<1 / negative costs (which P1 allows) let cost beat shortfall. Existing MIPSolver docstring claim (mip_solver.py:432) is false for non-integral data. | redesign | Two-pass lexicographic solve (min Σw·s; then min cost s.t. Σw·s ≤ opt(1+ε)+ε) in zone MIP; failing test with 0.01 amounts on core, fix core too (or log as bug). |
| H5 **P** | **`min_penalties` / objective names ≠ prioritizr v9.** v9 renamed `max_features`→`max_n_targets_met`, `max_utility`→`max_wtd_sum`; our `max_weighted_features` = `max_n_targets_met`+`add_feature_weights` (NOT `max_wtd_sum`). prioritizr `min_penalties` = hard targets, zero cost coefficients, objective is boundary/connectivity penalties only — near-opposite of ours. | [2×] science, redesign | Drop "same names as prioritizr"; add a name-mapping table; document `min_penalties` as pymarxan-specific (rename candidate for Stage 3 with deprecated alias). Optional alias `max_n_targets_met`. |
| H6 | **Budget feasibility floor wrong** (P9 and `_warn_no_solution`): empty selection is always feasible, so "budget < cheapest PU" is not a problem. Real infeasibility: COSTBUDGET < Σ over must-assign PUs of min allowed-zone cost (locks, status 2, and every eligible PU under MANDATORYALLOCATION=1). Executed: anchor with mandatory on has floor 3, B=2 infeasible with no locks. `_warn_no_solution` takes no objective and always prints "Unreachable targets". | [3×] arch, grounding, redesign | One `min_forced_spend(problem)` helper used by P9 and `_warn_no_solution(problem, status, objective)`. |
| H7 | **Boundary variables built for every objective**: `_ZoneModelContext` pre-builds `boundary_expr`/`zone_boundary_expr`, adding `y_std`, `bnd_abs` and O(n_bnd·Z²) binary `w` vars even for budgeted objectives that drop them. | arch, grounding | Lazy cached builders on the context; `min_set` calls them in today's order (golden). |
| H8 | **35.0 anchor false at fixture BLM**: `tests/data/simple` has BLM 1.0 → MIP cost 38, objective 43. 35.0 needs BLM=0 (as `validate_marxan_parity.py:68` sets). | [2×] grounding, redesign | Tests 4 & 6 state BLM=0. |

## MEDIUM

| # | Finding | Lenses | Resolution |
|---|---|---|---|
| M1 | MANDATORYALLOCATION + status-3 warning unreachable: `_pu_lock_issue_list` returns early when no lock tables (zones/model.py:296-297). | [3×] | Emit from `warn_zone_input_issues` / before the early return. |
| M2 | Presolve runs twice (`solve()` and `build_model()`, which `solve()` calls). | [3×] | Run once, in private `_build`. |
| M3 | `build_model` must be stateless (solver instances reused: portfolio_cuts.py:63, replacement_cost.py:65, app.py:266) and must own all gating (`_warn_if_large_mip`, PROBMODE/clump/sep NotImplementedErrors, zone TypeErrors). | arch | Private `_build(problem, config) -> _BuiltModel`; public `build_model` returns `.model`. Dedupe `resolved_backend` into `_backends.py`. |
| M4 | Layering/import cycles: `models/presolve.py` would import zones (upward); `pymarxan.zones` imports `solvers.base`; top-level `pymarxan` exports only `__version__`. | [2×] | `ConservationProblem.presolve_issues()` overridden by `ZonalProblem`; thin `presolve_check()` in `pymarxan.solvers`; function-local imports in `write_problem`; no top-level exports. |
| M5 | `write_problem`: PuLP has no LP reader (only `fromMPS`); both formats drop objective constants; `write_problem(ZonalProblem, MIPSolver())` silently exports a zone-blind model; name breaks `read_*`/`write_*` symmetry. | arch, grounding, redesign | Test: `.mps` via `fromMPS`, `.lp` via CBC binary; assert objective constant = 0; raise on zonal+core solver; keep prioritizr name, document exception (or rename — user call, default keep). |
| M6 | Relative-vs-absolute test needs a discriminating fixture: anchor optima identical at all budgets. **simple, BLM 0, COSTBUDGET ∈ {12,13,14}**: absolute → {4} (max abs 18), relative → {6} (max rel 2/3), both unique. Avoid B=20 (tie). | grounding | Use that fixture; record v0.38 golden {4} at B=12 before the change. |
| M7 | Tie-heavy objectives: MIP and enumeration return different tied assignments (max_features B=2 has 6 optima). `Solution.objective` is MarZone-recomputed. | grounding, redesign | Compare objective values via `metadata["mip_objective_value"]`; add seeded random-instance enumeration property test (5 PU × 3 zones × 3 features, ~10 seeds, all objectives, mandatory on/off). |
| M8 | presolve_check vs prioritizr: prioritizr uses **absolute** thresholds (1e6 / 1e-6) on compiled model coefficients, flags budget > total cost, all-locked-in / all-locked-out separately, single feature, ≥50% PUs with no features; runs by default. Spec's ratio checks are not prioritizr's; data-only scan misses 1/R_t and M·w coefficients. | science, redesign | Describe as "inspired by"; add prioritizr's checks plus a built-model coefficient scan (|coef| > 1e6, nonzero range) and NaN / negative amounts. Default stays off (documented deviation). |
| M9 | MANDATORYALLOCATION doc note misleading: MarZone has no such key; pymarxan zone 0 stands in for MarZone's available zone, so `0` reproduces MarZone unless the project lists the available zone explicitly. | [3×] | Reword §2 per science lens; `save_project` round-trip test. |
| M10 | "Status 3 wins" deviates from prioritizr (Σx = 1 for every PU → infeasible if locked out everywhere; NA-cost-everywhere drops the PU). prioritizr requires ≥ 2 zones. | science | Keep design; document as deliberate deviation in §2 + VALIDATION.md; state single-zone behaviour. |
| M11 | SA already never leaves a free PU at zone 0 (zones/solver.py:89-93, 133-138) → MANDATORYALLOCATION=0 already differs across solvers. | arch | Document in VALIDATION.md. |
| M12 | Shared builder module: four copies of the core budget check, semantics already drifting. | arch | Term-based builders in solver-agnostic `solvers/_mip_objectives.py`, used by zone MIP and core `min_shortfall`; `min_set` stays solver-specific. (Ties into H1.) |
| M13 | BLM / zone-boundary costs silently dropped for non-`min_set` objectives. | redesign | Warn once when BLM>0 or `zone_boundary_costs` set. |
| M14 | `MANDATORYALLOCATION` parsing: spec silently treats bad values as 0; `read_input_dat` yields `1.0` float. | arch, grounding, redesign | `mandatory_allocation()` accessor raising on bad values (mirrors `zone_target_contrib()`), accepts 0/1/"1"/1.0; zero-zone edge case. |
| M15 | API asymmetry: D7 applies to both solvers but `shortfall=` only on `MIPSolver`. | arch | Add `shortfall=` to `ZoneMIPSolver`; derive allowed objectives from builder-dict keys. |

## LOW

- `metadata["mip_objective"]` clashes with core `metadata["objective"]` (= name) → `mip_objective_value` in both; natural sign. (arch, redesign)
- Golden `min_set` dump: compare name→expression maps, not `writeLP` bytes. (arch, grounding)
- Weights: prioritizr weights are per target row (feature×zone matrix) and targets may span any zone subset; SPF reused as weight. Document. (science)
- Missing zone cost = 0 (free) vs prioritizr NA = forbidden. Document. (science)
- P9 "nonzero" zone-cost filter false-alarms on legitimately free zones. (grounding)
- Core P6 should subtract locked-out amounts. (grounding)
- Performance wording: zone P6 reuses Python-loop `diagnose_zone_infeasibility`, not "numpy". (grounding)
- `presolve_check=` kwarg shadows function name → `check=`. (grounding, redesign)
- Core `max_weighted_features` reads `features["spf"]` directly (KeyError if absent); use defaulting lookup. (redesign)
- Stable issue codes ("P6: …" prefix). (redesign)
- `docs/TUTORIAL.md:159` describes `min_largest_shortfall` as absolute — add to D7 change list. (grounding)
- Roadmap stale: lists `min_shortfall` as already at parity; kwarg/`check=` superseded by D3/D4. (arch)

## User decisions required before rev 2

1. **H1 — existing `pymarxan.objectives`:** (a) build on it (fix its zone builders, switch to COSTBUDGET, make it the shared module from M12), (b) supersede and deprecate it in v0.39, or (c) leave it and just document.
2. **H4 — `min_penalties` lexicographic:** two-pass solve (exact, ~2× solve time) vs. data-scaled big-M (single solve, numerically fragile).
3. **M5 — name:** keep prioritizr's `write_problem` vs. `export_model`.
