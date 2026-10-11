# Stage 1 (v0.39.0): Zone-MIP parity and solver tooling — design

**Status:** rev 2. Rev 1 was approved section by section in chat on 2026-10-11 and then patched
after the 4-lens design review (`2026-10-11-stage1-zone-mip-parity-review.md`). **Roadmap:**
`docs/plans/2026-10-11-prioritizr-parity-roadmap.md` Stage 1. **Baseline:** v0.38.0 (`e5a066c`).

## Goal

Zone problems get the budgeted objectives that single-zone problems have, plus `min_shortfall`,
with prioritizr semantics (shortfall relative to target, one term per target) and prioritizr ≥ 9
names accepted as aliases. Users can force every planning unit into a zone. Both MIP solvers can
export their model, and a pre-solve check warns about data problems before solving.

## Decisions

| # | Question | Decision |
|---|---|---|
| D1 | What counts as one "target met" for zone objectives | **Each target separately**: every `(zone, feature)` target and every overall feature target is its own term (prioritizr zones; confirmed against prioritizr `rcpp_add_rij_data.cpp:35-40`). SPF counts once per term |
| D2 | Budget | **One total budget**: `COSTBUDGET` caps Σ zone cost (prioritizr's scalar budget; a missing `zone_costs` pair costs 0, which is MarZone's convention, not prioritizr's NA = unavailable) |
| D3 | Mandatory allocation switch | **Problem parameter** `MANDATORYALLOCATION`, applied in `pu_zone_rules()`, so all four zone solvers honour it |
| D4 | Presolve check delivery | **Function + opt-in solver flag**: `presolve_check(problem)`; `presolve_check=True / "strict"` on both MIP solvers; default off |
| D5 | Objective-builder structure | **Builders in a solver-agnostic module** `src/pymarxan/solvers/_mip_objectives.py` (imports only pulp/stdlib). `ZoneMIPSolver` uses them now; `MIPSolver` moves onto them in Stage 3. Builders return an expression plus its sense, and never add the budget or hard targets themselves (review: composable for Stage 3) |
| D6 | Objective set | Zone MIP: `min_set`, `max_features`, `max_weighted_features`, `min_shortfall`, `min_largest_shortfall`. Core MIP gains `min_shortfall`. Both accept the prioritizr ≥ 9 alias `max_n_targets_met` (→ `max_features`). **Zone `min_penalties` dropped from Stage 1** (review HIGH: pymarxan's `min_penalties` is not prioritizr's objective of the same name; Stage 3 settles the naming). The core `min_penalties` is unchanged, and its docstring now states the difference |
| D7 | Shortfall units | **Relative** (shortfall / *R*, prioritizr) for `min_shortfall` and `min_largest_shortfall` in both solvers. Both solvers take `shortfall="relative" \| "absolute"`; `"absolute"` on `MIPSolver` restores v0.38 `min_largest_shortfall` exactly |

## 1. Objectives

### Target terms (D1)

A shared frozen dataclass in `_mip_objectives.py`:

```python
@dataclass(frozen=True)
class TargetTerm:
    key: str        # unique name stem: "z2_f5" (zone), "o_f5" (overall), "f5" (core feature)
    kind: str       # "zone" | "overall" | "feature"
    achieved: pulp.LpAffineExpression   # A_t(x); lpSum([]) when the feature has no rows
    required: float # R_t = target × MISSLEVEL
    weight: float   # SPF of the feature (default 1.0 when the column is absent)
```

- **Zone problems.** `_zone_target_terms(problem, x)` returns one term per `(z, f)` in
  `zone_target_lookup()`, using today's `_zone_achieved_exprs`, and one term per feature with
  target > 0, using today's overall-target expression (Watts et al. 2009 eq. 6). The skip rule
  for building terms is today's: raw target ≤ 0 is skipped.
- **Core problems.** Only the new shortfall code in `MIPSolver` builds terms: one per feature
  with target > 0.
- **Features with no `pu_vs_features` rows** ("ghost features") still produce a term, with
  `achieved = lpSum([])`. Under every budgeted objective they are therefore never met and have
  full shortfall (prioritizr semantics; review MEDIUM, parity). `presolve_check` P5 reports them.
- **Relative objectives.** Terms with *R_t* ≤ 0 (for example MISSLEVEL = 0) are skipped by both
  relative objectives in both solvers, so there is no division by zero.

### Builder contract (D5)

```python
@dataclass
class ObjectiveContext:
    model: pulp.LpProblem
    cost_expr: pulp.LpAffineExpression
    terms: list[TargetTerm]
    prefix: str = ""     # prepended to every variable/constraint name a builder creates
    def slack(self, term) -> pulp.LpVariable   # create-once: s ≥ 0, s ≥ R − A (constraint "{prefix}short_{key}")
    def met(self, term) -> pulp.LpVariable     # create-once: binary y, A ≥ R·y (constraint "{prefix}met_{key}")

@dataclass(frozen=True)
class ObjectiveTerm:
    expr: pulp.LpAffineExpression
    sense: int           # pulp.LpMinimize | pulp.LpMaximize, in the objective's natural units
```

| Builder | Uses | Returns |
|---|---|---|
| `max_features` | `met(t)` for all terms | Σ *y_t*, maximise |
| `max_weighted_features` | `met(t)` | Σ *w_t*·*y_t*, maximise |
| `min_shortfall(shortfall)` | `slack(t)` | relative: Σ *w_t*·*s_t* / *R_t*; absolute: Σ *w_t*·*s_t*; minimise |
| `min_largest_shortfall(shortfall)` | `slack(t)`, plus one `m` with *m* ≥ *s_t* / *R_t* (or *s_t*) | *m*, minimise. Unweighted, as in prioritizr |

- The solver sets the model objective: `+expr` for minimise, `−expr` for maximise.
- The solver adds `cost_expr <= COSTBUDGET` once (`"cost_budget"`).
- `min_set` is not a builder in Stage 1: it keeps today's code path byte-identical.
- One test composes `min_shortfall` and `max_features` in one model, to show names cannot collide.

### `ZoneMIPSolver`

- **Constructor:** `ZoneMIPSolver(objective="min_set", shortfall="relative", presolve_check=False, …)`.
  Names are validated with `_validate_mip_strategy`, and `max_n_targets_met` is an alias.
- **`min_set`:** byte-identical to v0.38, including the always-zero penalty slack. This is pinned
  by a canonical-model golden (Testing 4).
- **Budgeted objectives:**
  - They require `COSTBUDGET`; without it they raise `ValueError` naming the objective.
  - They omit the hard targets and the BLM / zone-boundary terms. When BLM > 0 or any
    `zone_boundary_cost` is nonzero, the solver emits one `UserWarning`: "ZoneMIPSolver: objective
    '<name>' ignores boundary penalties (BLM / zone boundary costs); prioritizr applies them —
    planned follow-up".
  - Budget infeasibility gets a message from the shared helper `min_forced_cost(problem)` (§2):
    "the planning units that must be assigned cost at least X > COSTBUDGET Y". This replaces the
    target diagnostic for budgeted objectives.
- **Metadata:** `Solution.objective` is recomputed by `build_zone_solution`, as today. The
  builder's own objective value goes in `Solution.metadata["mip_objective"]`, and the units in
  `metadata["shortfall"]`.

### `MIPSolver`

- **New `min_shortfall`**, built from `TargetTerm`s and the shared builder. Unlike v0.38, ghost
  features are included and spf defaults to 1.0. It requires `COSTBUDGET`.
- **New `shortfall="relative"` keyword**, which applies to `min_shortfall` and
  `min_largest_shortfall`.
  - Relative `min_largest_shortfall` uses the shared builder, so ghost features count and terms
    with *R* ≤ 0 are skipped.
  - `shortfall="absolute"` keeps v0.38's code path exactly, ghost-feature behaviour included.
- **`max_n_targets_met`** is accepted as an alias for `max_features`.
- **Fixed:** in `max_features` and `max_weighted_features`, a ghost feature with target > 0 no
  longer counts as met for free (v0.38 left its `feat_met` unconstrained). Its `feat_met` is
  fixed to 0.
- **Unchanged:** the core `min_penalties`. Its docstring now says it is "min Σ SPF·shortfall,
  then cost (big-M; approximately lexicographic) — not prioritizr's `add_min_penalties_objective`".
- **CHANGELOG:**
  - "Changed": `min_largest_shortfall` is relative by default.
  - "Fixed": ghost features in `max_features` and `max_weighted_features`.
  - TUTORIAL §5 gets the new objective row and the relative note.

## 2. Mandatory allocation (D3)

- **Accessor:** `ZonalProblem.mandatory_allocation() -> bool` reads `MANDATORYALLOCATION` from the
  parameters.
  - Values 0, 1, 0.0 and 1.0 are accepted, as `read_input_dat` gives int or float.
  - Anything else is a `validate()` error, and the accessor returns False. That includes bool,
    str and 2.
  - `validate()` and `pu_zone_rules()` both use the accessor.
- **Rules:** when the parameter is on, the final ("free") row of `pu_zone_rules()` gives status
  0/1 PUs `PUZoneRule(allowed=all, may_be_unassigned=False, start=<unchanged>)`.
- **Status 3 wins.** This is a deliberate pymarxan choice, the same as prioritizr dropping
  all-NA PUs. The issue "N planning units are locked out (status 3) while MANDATORYALLOCATION is
  1; they stay unassigned" is computed outside `_pu_lock_issue_list`'s no-tables early return, so
  it fires without lock tables. It is a warning only (a `ZoneLockWarning` through
  `warn_zone_input_issues`).
- **Shared helper:** `min_forced_cost(problem) -> float`, in `zones/diagnostics.py`, returns
  Σ over must-assign rules of the cheapest allowed zone cost, plus min(0, cheapest cost) for
  optional PUs. Core problems use Σ locked-in costs. The MIP budget message and presolve P9a
  both use it.
- **Solvers:** no solver code changes.
  - `ZoneSASolver` already never leaves a non-locked-out PU unassigned.
  - With the parameter on, the MIP, heuristic and II agree with SA.
  - II at ITIMPTYPE 0/1 only reproduces its seed for must-assign PUs. This is documented, and
    the tests use ITIMPTYPE 2/3 for II.
- **MarZone note (VALIDATION.md):** MarZone keeps every PU in one of its zones, and its zone
  index 0 (first zone, usually "available") is the do-nothing state. `MANDATORYALLOCATION 1`
  reproduces MarZone only when the project's zones include that available zone. When
  pymarxan's zone 0 stands in for it, leave the parameter at 0. There is no AVAILABLEZONE input
  key at 85082e3.

## 3. `write_problem`

- **Build/solve split.** Each MIP solver gets a private `_build(problem, config) -> _BuiltModel`
  that emits no warnings, and two public entry points:
  - `build_model(problem, config=None) -> pulp.LpProblem`
  - `solve(problem, config=None) -> list[Solution]`

  Each public method runs the preamble once: the type check, `raise_if_separation_active`,
  `warn_zone_input_issues`, `_warn_if_large_mip`, the presolve check and the budget-penalty
  warning. It then calls `_build`, so warnings fire once with unchanged stacklevels.

  `_BuiltModel` carries `model`, `x`, `pu_ids` (zone: plus `zone_ids`) and `blm`. Everything
  else (`probmode`, the TARGET2/sep flags) is recomputed in `solve()`. Strategy gating raises
  live in `_build`, so `build_model` refuses the same configurations `solve` does. This is a pure
  refactor: every existing test passes, and MIP 35.0 holds.
- **API.** `pymarxan.solvers.write_problem(problem, path, *, solver=None, config=None) -> Path`
  lives in `src/pymarxan/solvers/export.py`, with `ZoneMIPSolver` / `ZonalProblem` imported
  inside the function.
  - **Format:** `.lp` uses `writeLP` and `.mps` uses `writeMPS`. Any other suffix raises
    `ValueError`.
  - **Default solver:** `ZoneMIPSolver()` for a `ZonalProblem`, `MIPSolver()` otherwise.
  - **Errors:** a solver without `build_model` raises `TypeError`. So does a solver whose
    `supports_zones()` does not match the problem type.
  - **Objective constant:** a nonzero `model.objective.constant` triggers a `UserWarning`,
    because LP/MPS writers drop it.
  - **Not exported:** time limit and gap. No-good cuts are included only when that `config` is
    passed.

## 4. `presolve_check` (D4)

- **Location.** `src/pymarxan/solvers/presolve.py`, which imports no zone code at module level
  (zone imports are inside functions). This avoids the circular imports the review reproduced.
  - `presolve_check(problem) -> list[str]` and `class PresolveWarning(UserWarning)`.
  - Exported from `pymarxan.solvers` and, through a lazy module `__getattr__`, from `pymarxan`.
  - A subprocess import test covers `pymarxan`, `pymarxan.solvers`, `pymarxan.zones` and
    `pymarxan.solvers.presolve`, each imported first in a fresh interpreter.
- **Behaviour.** It never raises. It runs `problem.validate()` first. If there are validation
  errors, it returns the single message "problem fails validate(): <n> error(s); fix those first"
  and skips the numeric checks. Each message names the entity and the numbers involved.

| # | Condition | Example message |
|---|---|---|
| P1 | non-finite value in any numeric input (cost, zone cost, amount, target, boundary, contribution, BLM, COSTBUDGET), or a negative cost | "PU 7 has cost -3.0 (negative)" |
| P2 | max/min nonzero cost ratio > 1e6 | "costs span 1.2e+07× (min 0.01, max 1.2e+05): solvers may lose precision" |
| P3 | the same for nonzero amounts | "amounts span …" |
| P4 | the same for nonzero boundary values | "boundary lengths span …" |
| P5 | feature with zero total amount and target > 0 | "feature 4 occurs in no planning unit but has target 10" |
| P6 | target × MISSLEVEL above the total amount (zones: reuse `diagnose_zone_infeasibility`) | "feature 2 target 50 exceeds its total amount 30" |
| P7 | every PU locked (core status 2/3; zones: every rule forced or empty) | "every planning unit is locked: nothing to optimise" |
| P8 | BLM > 0 and BLM × max boundary vs median nonzero cost ratio > 1e6 or < 1e-6 | "BLM × boundary is 3.1e+07× the median cost: boundary will dominate" |
| P9a | `COSTBUDGET` < `min_forced_cost(problem)` | "COSTBUDGET 5 is below the cost that locked / must-assign units force (12): infeasible" |
| P9b | `COSTBUDGET` below the cheapest selectable PU or zone option | "COSTBUDGET 5 is below the cheapest planning unit (8): nothing can be selected" |

- **Solver flag.** `presolve_check` accepts only `True`, `False` or `"strict"`, checked by
  identity (so 0 and 1 are rejected); anything else is a `ValueError` at construction.
  - `True` emits one `PresolveWarning` joining the messages, at the caller's frame.
  - `"strict"` raises `ValueError`.

## Out of scope (with destination)

- Per-zone budgets: later, if asked.
- Penalties in budgeted objectives: follow-up, with a warning until then.
- Zone `min_penalties` and prioritizr-semantics `min_penalties`: Stage 3.
- `MIPSolver` fully on builders: Stage 3.
- Per-(zone, feature) weights: later.
- An exact lexicographic replacement for the big-M in `min_penalties`: Stage 3 hierarchical.
- Non-MIP model export.

## Testing

Fixture numbers below come from the review's executed prototypes (154/154 checks agreed with
enumeration).

1. **Zone objective oracle.** On `make_anchor_problem()`, the PU4 variant, and spf ∈ {1, 3},
   brute-force enumeration over {0,1,2}^n must equal the MIP optimum for every budgeted
   objective and budget in {0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 13}. Compare objective values; compare
   assignments only where the argmin is unique:
   - B = 3: (2,2,2) for `min_shortfall` (0.2 at spf 1, 0.6 at spf 3) and for
     `min_largest_shortfall` (0.2);
   - B = 7: (1,2,2) for every objective;
   - B = 6: value only ((1,0,2) and (1,2,0) tie at 0.0667).
2. **Target unit (D1).** The anchor with B ∈ [1, 6] gives `max_features` = 1. Per-feature
   semantics would give 0.
3. **Weighted.** Add feature 2 to the anchor: target 10, spf 5, amount 10 in PU3 only,
   contribution {z1: 1.0, z2: 0.0}. At B = 7, `max_features` gives unique (1,2,2) with value 2,
   and `max_weighted_features` gives unique (0,0,1) with value 5.
4. **`min_set` unchanged.** A canonical-model golden (per constraint name: sense, rhs, sorted
   (var, coef); sorted objective coefficients; variable bounds/categories) of the zone anchor and
   `tests/data/zones`. It is captured on v0.38 (`e5a066c`) by patching `pulp.LpProblem.solve`,
   before any change, and must stay equal. All existing zone/MIP tests pass and
   `tests/data/simple` = 35.0.
5. **Relative vs absolute (core).** 3 PUs at cost 1, `COSTBUDGET` 1. Feature a: target 100,
   PU1 = 60, PU3 = 30. Feature b: target 10, PU2 = 8, PU3 = 5.
   - Relative `min_largest_shortfall` selects PU3 (0.7).
   - `shortfall="absolute"` selects PU1 (40), equal to v0.38's [1,0,0], recorded before the
     change.
   - For `min_shortfall`, assert values: the fixture ties.
6. **Cross-solver oracle.** A 1-zone `ZonalProblem` built from a core problem (zone cost = PU
   cost, contribution 1, no zone targets) gives the same `metadata["mip_objective"]` as
   `MIPSolver` for `max_features`, `min_shortfall` and `min_largest_shortfall` (relative). Run it
   with and without a ghost feature, and at MISSLEVEL 1 and 0.9.
7. **Edge cases.**
   - A target-0 feature and MISSLEVEL = 0 under relative shortfall: no crash, term ignored.
   - No spf column: weights are 1.0.
   - A ghost feature in core `max_features` is not counted.
   - `min_shortfall` and `max_features` composed in one model: no name collision.
   - A budget objective with BLM > 0 emits one boundary-ignored warning.
8. **Mandatory allocation.** Use a 6-PU fixture: the anchor plus PU4 (status 0), PU5 (status 3)
   and PU6 (status 1).
   - With the parameter on, every solver (II at ITIMPTYPE 2 and 3) gives
     `rule_violations == []`, and only PU5 is in zone 0. The MIP optimum = enumerated 10.5 at
     (1,2,2,2,0,2).
   - Detection tests:
     - heuristic on `tests/data/zones`: today's all-zero seed becomes all non-zero;
     - II ITIMPTYPE 1: the removal pass must not unassign;
     - MIP on a zero-target fixture: all zone 0 becomes the cheapest zones.
   - SA is a regression guard only.
   - The status-3 warning fires with no lock tables, exactly once per solve.
   - Accessor values: 1, 1.0 accepted; "1", True, 2 rejected.
   - The golden tests pass with the parameter off.
   - Budget infeasibility with the parameter on gives the `min_forced_cost` message.
9. **write_problem.**
   - `tests/data/simple` with BLM = 0 → `.mps` (`LpProblem.fromMPS` + CBC) and `.lp` (the CBC
     binary on the file; PuLP 3.3 has no `fromLP`) both give 35.0.
   - With BLM = 1, the round-trip equals the in-process objective (43.0).
   - The zone anchor round-trip gives 7.0, and a `pu_zones` bound survives.
   - Error cases: bad suffix, non-MIP solver, solver/problem mismatch.
   - Stacklevel tests: `w.filename == __file__` for `PresolveWarning`, the large-MIP warning and
     `ZoneLockWarning`, raised from both `solve()` and `build_model()`.
10. **presolve_check.**
    - One triggering test per check, P1–P9b.
    - `tests/data/simple`, `tests/data/zones` and the anchor give `[]`.
    - A `validate()` short-circuit test.
    - Flag tests: one warning, `"strict"` raises, off stays silent, values 0/1 rejected.
    - The subprocess import test.
11. **Gate.** `make check` green, parity MIP 35.0, every v0.37/v0.38 golden test green, benches
    non-regressing.

## Performance

- `min_set` is unchanged.
- The budgeted objectives add *O(#terms)* variables.
- `presolve_check` is *O(n_pu + rows)* and runs only on request.
- `_build` adds no work.
