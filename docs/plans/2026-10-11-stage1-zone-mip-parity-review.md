# Stage 1 (v0.39) — 4-lens design review synthesis

Reviewed: `2026-10-11-stage1-zone-mip-parity-design.md` rev 1 (`06fd0e8`). Workflow
`wf_148378f8-b6d` ran 11 agents with 0 errors:
- **Architect.**
- **Grounding:** executed prototypes, 154/154 checks against brute-force enumeration.
- **Parity:** prioritizr v9.0.1 at `f2f5628`, and MarZone at `85082e3`.
- **Independent re-design.**

Every HIGH finding was adversarially verified.

**Verdict.** The core design holds. Prototypes of the five zone objectives matched enumeration in
every case, D1/D2/D7 match prioritizr exactly, and the build/solve split is mechanical. **One
HIGH survived:** pymarxan's `min_penalties` is a different objective from prioritizr's
`add_min_penalties_objective`. The other HIGHs were real but downgraded to MEDIUM. Spec patched
to rev 2.

| # | Sev (verified) | Lens | Finding | Rev 2 disposition |
|---|---|---|---|---|
| 1 | HIGH (P) | parity, redesign | `min_penalties` shares prioritizr's name but not its semantics (prioritizr: minimise penalties, hard targets, optional budget) | Zone `min_penalties` dropped from Stage 1; core docstring states the difference; naming settled in Stage 3 |
| 2 | HIGH→MED (P) | grounding, redesign, architect | Core MIP silently gives ghost features (no rows) zero shortfall and free `feat_met`; new `min_shortfall` would copy it | Ghost features become terms with A = 0 in every new or relative objective; `max_features` ghost bug fixed ("Fixed"); absolute path byte-identical |
| 3 | HIGH→MED | architect | Circular imports: presolve in `models/` with zone imports, `write_problem` importing `ZoneMIPSolver` at module level (both reproduced) | Both in `solvers/` with lazy zone imports; lazy `pymarxan.__getattr__`; subprocess import test |
| 4 | HIGH→MED | architect | Builder shape not composable for Stage 3 (duplicate names, budget inside builders, zone-only location) | Solver-agnostic `_mip_objectives.py`: `TargetTerm`, `ObjectiveContext` with create-once slack/met and name prefix, `ObjectiveTerm(expr, sense)`; solver adds the budget once; composition test |
| 5 | HIGH→MED (P) | redesign | Relative denominator ambiguous; divide by zero at target 0 / MISSLEVEL 0 | R = target × MISSLEVEL; skip R ≤ 0 in both relative objectives, both solvers; tests |
| 6 | HIGH→MED (P) | parity | "MANDATORYALLOCATION 1 is MarZone behaviour" only holds when the project's zones include MarZone's available zone | VALIDATION note reworded; no AVAILABLEZONE key at 85082e3 |
| 7 | MED | architect, grounding | Presolve warns twice and stacklevels break when `solve()` calls `build_model()` | Private `_build` with no warnings; each public method runs the preamble once; stacklevel tests |
| 8 | MED | architect, grounding, redesign | Status-3 + mandatory warning never fires without lock tables (early return) | Computed outside the early return; test without tables |
| 9 | MED | grounding, redesign | Budget infeasibility under mandatory allocation has no correct message; P9 misses it | `min_forced_cost` helper for the MIP message and P9a; P9b keeps "nothing can be selected" |
| 10 | MED | grounding | Round-trip "35.0" wrong: `tests/data/simple` has BLM 1 (43.0); PuLP 3.3 has no `fromLP` | BLM = 0 for 35.0; BLM = 1 compares with in-process value; `.lp` via the CBC binary |
| 11 | MED | architect | `shortfall=` only on MIPSolver; no test detects the default change | Keyword on both solvers; recorded in metadata; core fixture with relative ≠ absolute and v0.38 golden |
| 12 | MED | parity | prioritizr ≥ 9 renamed `max_features` to `max_n_targets_met` | Accepted as an alias in both solvers |
| 13 | MED | parity | Budgeted objectives silently drop BLM / zone boundary; prioritizr applies them | One `UserWarning` when they are present; follow-up recorded |
| 14 | MED | redesign | Mandatory/weighted tests could not tell right from wrong | Discriminating fixtures (heuristic/II ITIMPTYPE 1/MIP zero-target; 2-feature SPF-5 variant) |
| 15 | LOW | various | spf column optional (KeyError); canonical-model golden instead of raw LP text; accessor for parameter typing; P1 covers all numeric inputs; presolve runs `validate()` first; flag identity check; objective-constant warning on export; big-M approximate and negative-cost bound; NA-cost note; status-3 exemption labelled deliberate; ITIMPTYPE 1 note; TUTORIAL update | All accepted in rev 2 |

**Confirmed facts:**
- prioritizr min_shortfall = Σ w·y with Σ r·x + t·y ≥ t, y ∈ [0, 1], so relative.
- prioritizr min_largest_shortfall is unweighted and relative.
- No cost tie-break in either (the v9 NEWS removed it from max_n_targets_met).
- A scalar budget is one row over all zone × PU pairs.
- MarZone never leaves a PU without a zone.
- `read_input_dat` keeps unknown keys as int or float.
- PuLP 3.3.0 is installed; HiGHS is not available in the shiny env.

## Second pass (independent 4-agent review of rev 1, 2026-10-11 eve)

A second, independently run architect, grounding (executed, CBC), science (prioritizr 9.0.1 @
f2f5628, MarZone, Watts et al. 2009 doi:10.1016/j.envsoft.2009.06.005, scite-checked: exists,
no notices) and re-design pass reviewed rev 1. Rev 2 already absorbed most of its findings: ghost
features, the R = target × MISSLEVEL divisor, import cycles, the double presolve, the status-3
early return, `min_forced_cost`, BLM = 0 for 35.0, the missing `fromLP`, `shortfall=` on both
solvers, the `max_n_targets_met` alias, the boundary-ignored warning, the MarZone wording, and
the zone `min_penalties` naming. The new findings, and the user's decisions on them, are below.

| # | Sev | Lens | Finding | Disposition (rev 3) |
|---|---|---|---|---|
| S1 | HIGH | grounding | **Existing `pymarxan.objectives` package.** It has `Objective`/`ZonalObjective` ABCs with `build_zone_mip_*` and `MinShortfallObjective`/`MaxCoverage`/`MaxUtility`/`MinSet`, but no consumers outside its own tests. Its `min_shortfall` is absolute and unweighted and reads `BUDGET`, not `COSTBUDGET`. Its zone builders create slacks without defining constraints. Shiny `objective_selector` writes `cfg["objective_type"]`, which nothing reads. Rev 2 adds a third mechanism without mentioning it. | **User: supersede + deprecate.** `_mip_objectives.py` is the single mechanism. `pymarxan.objectives` emits a `DeprecationWarning` on import in v0.39 and is removed later. The dead selector wiring is noted for a Shiny follow-up. CHANGELOG "Deprecated". |
| S2 | HIGH (P) | redesign | **Core `min_penalties` big-M is not lexicographic.** M = Σ cost + 1 dominates only when w·δs ≥ 1. With fractional amounts, SPF < 1, or negative costs (which P1 permits), cost can beat shortfall. The docstring claim at mip_solver.py:432 is false for non-integral data. | **User: two-pass solve.** Pass 1 minimises Σ w·s. Pass 2 adds Σ w·s ≤ opt·(1+1e-9) + 1e-9 and minimises cost. Scope (user to confirm): fix core `min_penalties` in Stage 1 as "Fixed", with a failing 0.01-amount test first. Zone `min_penalties` stays in Stage 3. |
| S3 | MED | science, redesign | **`presolve_check` is not prioritizr's check.** prioritizr `run_presolve_check.R` uses **absolute** thresholds (> 1e6, nonzero < 1e-6) on the compiled model's objective/constraint coefficients, feature data and budget. It also flags budget > total cost, all locked in and all locked out (separately), a single feature, and ≥ 50 % of PUs with no feature data, and it runs by default. Rev 2's ratio checks are reasonable but different, and a data-only scan misses the 1/R_t coefficients the relative objectives create. | Docstring: "inspired by prioritizr's presolve_check; checks differ". Add P10 (absolute coefficient scan of the built model, \|coef\| or rhs > 1e6 or nonzero < 1e-6, flag path only), P11 (budget ≥ total cost: budget non-binding), and split P7 into all-in / all-out. Default off is documented as a deviation. |
| S4 | MED | science | **Mandatory allocation on a single-zone `ZonalProblem`.** prioritizr requires ≥ 2 zones; rev 2 would force every eligible PU into the one zone. | State this in §2: allowed, forces selection of every non-status-3 PU; documented deviation. |
| S5 | MED | redesign | **Fixture breadth.** The enumeration oracles run on 3–6 PU hand fixtures only. | Add a seeded random-instance property test: 5 PU × 3 zones × 3 features, 10 seeds, every budgeted objective, mandatory on and off, comparing objective values. |
| S6 | LOW | architect | `metadata["mip_objective"]` (value) sits next to core `metadata["objective"]` (name). | Rename to `mip_objective_value` in both solvers. |
| S7 | LOW | grounding | Core P6 ignores status-3 PUs. | Subtract locked-out amounts. |
| S8 | LOW | arch, grounding, redesign | `write_problem` breaks the `read_*`/`write_*` pairing. | **User: keep `write_problem`** (prioritizr name); docstring notes the deliberate exception. |
| S9 | MED (new bug, executed) | synthesis | While executing S2, the following turned up. A **zero-cost PU with no features and no boundary** never enters the PuLP model, so its `varValue` is `None`. `MIPSolver`'s `has_values = all(... is not None)` (mip_solver.py:573) then makes `solve()` return `[]` for every objective, `min_set` included. The S2 fixture with PU2 cost 0 gives 0 solutions. `ZoneMIPSolver` is immune: it uses `any(...)` and reads `None` as 0 (zones/mip_solver.py:201, 211). | Fixed in Stage 1, copying the zone rule, with a failing test first. |

S2 counter-example, executed on v0.38. Fixture: PU1 cost 100 holding 0.01 of feature a; PU2 cost
1; target 0.01; BLM 0. With M = 102, `min_penalties` selects nothing and the target is unmet,
while `min_set` selects PU1. User decisions: S1 supersede and deprecate; S2 two-pass solve,
fixing the core solver in Stage 1; S8 keep `write_problem`. Spec rev 3 absorbs S1–S9.
