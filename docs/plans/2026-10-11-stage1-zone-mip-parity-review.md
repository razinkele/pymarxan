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
