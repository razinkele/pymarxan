# Stage 1 (v0.39.0): Zone-MIP parity and solver tooling — design

**Status:** rev 1, approved section by section in chat 2026-10-11. **Roadmap:**
`docs/plans/2026-10-11-prioritizr-parity-roadmap.md` Stage 1. **Baseline:** v0.38.0 (`e5a066c`).

## Goal

Zone problems get the same exact objectives as single-zone problems, under the same names as
prioritizr. Users can force every planning unit into a zone. Both MIP solvers can export their
model, and a pre-solve check warns about problems before solving.

## Decisions (made in chat)

| # | Question | Decision |
|---|---|---|
| D1 | What counts as one "target met" for zone objectives | **Each target separately**: every `(zone, feature)` target and every overall feature target is its own term (prioritizr zones semantics) |
| D2 | Budget | **One total budget**: `COSTBUDGET` caps Σ zone cost, the same parameter as `MIPSolver` |
| D3 | Mandatory allocation switch | **Problem parameter** `MANDATORYALLOCATION`, applied in `pu_zone_rules()`, so all four zone solvers honour it |
| D4 | Presolve check delivery | **Function + opt-in solver flag**: `presolve_check(problem)`; `presolve_check=True / "strict"` on both MIP solvers; default off |
| D5 | Objective-builder structure | **One builder function per objective** in `zones/mip_solver.py`, chosen by dict lookup. This tries the pattern Stage 3 will apply to `MIPSolver` |
| D6 | Objective set | Same names in both MIPs: `min_set`, `max_features`, `max_weighted_features`, `min_shortfall` (new in both), `min_largest_shortfall`, `min_penalties` |
| D7 | Shortfall units | **Relative** (shortfall / target, prioritizr) for `min_shortfall` and `min_largest_shortfall` in both solvers. `MIPSolver` gains `shortfall="relative" \| "absolute"`; `"absolute"` restores v0.38 `min_largest_shortfall` |

## 1. Zone-MIP objectives

### Target terms (D1)

The zone MIP has two kinds of target. Both become terms *t* with an achieved expression
*A_t(x)*, a requirement *R_t*, and a weight *w_t*:

- **Zone targets:** one term per `(z, f)` in `zone_target_lookup()`.
  - *A_t* is today's `_zone_achieved_exprs[(z, f)]`: raw amounts, contribution-weighted under
    `ZONETARGETCONTRIB=1`.
  - *R_t* = target × `MISSLEVEL`.
- **Overall targets:** one term per feature with `target > 0`.
  - *A_t* = Σ_i Σ_z amount(i, f) · contribution(f, z) · x(i, z), as in today's
    `_add_overall_target_constraints` (Watts et al. 2009 eq. 6).
  - *R_t* = target × `MISSLEVEL`.
- **Weights:** *w_t* is the feature's SPF (`_spf_lookup`, default 1.0).
- **Skipped terms:** terms with *R_t* ≤ 0 are skipped. This matches today's `t <= 0: continue`
  and the `zone_target_lookup` target > 0 filter.

One helper, `_zone_target_terms(problem, x) -> list[TargetTerm]`, builds this list. It replaces
the separate expression-building in today's `_add_zone_target_constraints` and
`_add_overall_target_constraints`. The constraint names stay `zone_target_{z}_{f}` and
`overall_target_{f}`.

### Builders (D5, D6)

Each objective has a builder:

```python
_ZoneObjective = Callable[[_ZoneModelContext], pulp.LpAffineExpression]
```

`_ZoneModelContext` (a small dataclass) carries:
- `model`, `x`, `pu_ids`, `zone_ids`;
- `cost_expr`, `boundary_expr`, `zone_boundary_expr`;
- `terms`, `problem`.

Each builder adds its own variables and constraints and returns the expression to minimise.
`ZoneMIPSolver(objective="min_set")` validates its name with the same
`_validate_mip_strategy` as `MIPSolver`.

| Objective | Variables / constraints | Minimised expression | Budget |
|---|---|---|---|
| `min_set` (default) | hard *A_t* ≥ *R_t* for every term | cost + BLM·boundary + zone boundary | — |
| `max_features` | binary *y_t*; *A_t* ≥ *R_t*·*y_t* | −Σ *y_t* | required |
| `max_weighted_features` | the same | −Σ *w_t*·*y_t* | required |
| `min_shortfall` | *s_t* ≥ 0; *s_t* ≥ *R_t* − *A_t* | Σ *w_t*·*s_t* / *R_t* | required |
| `min_largest_shortfall` | *s_t* as above; *m* ≥ *s_t* / *R_t* | *m* | required |
| `min_penalties` | *s_t* as above | M·Σ *w_t*·*s_t* + cost, with M = Σ_i max_z cost(i, z) + 1 | — |

- **`min_set`:** byte-identical to v0.38. Today's always-zero penalty slack
  (`_build_penalty_expr`) stays in `min_set` only, so nothing about v0.38 changes.
- **Budget:** cost ≤ `COSTBUDGET`, where cost is the zone-cost expression Σ cost(i,z)·x(i,z).
  Pairs missing from `zone_costs` cost 0, as today. A missing budget raises `ValueError`, with
  the same message as `MIPSolver` but naming the objective.
- **Dropped terms:** non-`min_set` objectives omit the hard target constraints and the BLM /
  zone-boundary terms, as `MIPSolver` does. Adding penalties to budgeted objectives (as
  prioritizr does) is a follow-up, not in this stage.
- **Infeasibility:** with no hard targets, the budgeted objectives are infeasible only when
  locks cost more than the budget. `_warn_no_solution` adds "the locked planning units cost more
  than COSTBUDGET" in that case. This is the only new message.
- **Reported objective:** `Solution.objective` is recomputed by `build_zone_solution`, as today.
  Builders report their own objective value in `Solution.metadata["mip_objective"]`.

### `MIPSolver` changes (D6, D7)

- **New `min_shortfall` objective**, with the same terms as the per-feature slacks of
  `min_largest_shortfall`. It minimises Σ spf_j · slack_j / target_j and needs `COSTBUDGET`.
  Features with target ≤ 0 are skipped.
- **New keyword `shortfall: str = "relative"`**, validated against `("relative", "absolute")`.
  - Under `"relative"`, `min_largest_shortfall` constrains `t ≥ slack_j / target_j`.
  - `"absolute"` keeps v0.38's `t ≥ slack_j` exactly.
  - The keyword applies to `min_shortfall` too. Absolute there means Σ spf·slack.
- **CHANGELOG "Changed":** `MIPSolver(objective="min_largest_shortfall")` now minimises the
  largest *relative* shortfall. Pass `shortfall="absolute"` for the v0.38 behaviour.

## 2. Mandatory allocation (D3)

- **Parameter:** `MANDATORYALLOCATION` (input.dat or `problem.parameters`).
  - `0` is the default and leaves v0.38 behaviour unchanged.
  - `1` forces every eligible PU into a zone.
  - Any other value is a `validate()` error. `pu_zone_rules()` then treats it as `0`, so a
    solver never crashes on it.
  - `read_input_dat` already keeps unknown keys; verify this in the grounding review.
- **Rules:** in `ZonalProblem.pu_zone_rules()`, the last ("free") precedence row changes when
  the parameter is 1. Status 0 and 1 PUs with no lock or puzone get
  `PUZoneRule(allowed=all zones, may_be_unassigned=False, start=<unchanged>)`: status 1 starts
  at `min(zone_ids)`, status 0 has no start. Pulock, puzone and status-2 PUs are already
  must-assign.
- **Status 3 wins:** a status-3 PU stays unassigned. `pu_lock_issues()` adds one warning-only
  issue: "N planning units are locked out (status 3) while MANDATORYALLOCATION is 1; they stay
  unassigned". It is not a `validate()` error.
- **Solvers:** no solver code changes. MIP uses `assign == 1`. SA already never proposes 0. The
  heuristic seeds must-assign PUs. In II, the removal pass skips them and the swap pass drops
  0 for them. All of this comes from the v0.38 rule machinery.
- **MarZone note** (`docs/VALIDATION.md`): MarZone assigns every PU to a zone, so `1` is MarZone
  behaviour. pymarxan keeps `0` as the default for compatibility.

## 3. `write_problem`

- **Build/solve split:** both MIP solvers get `build_model(problem, config=None) ->
  pulp.LpProblem`, which holds everything from variable creation to the last constraint.
  `solve()` = `build_model` → solve → extract. This is a pure refactor: every existing test and
  the 35.0 anchor must pass unchanged. Any state `solve()` needs after the solve (variable dicts,
  backend) comes back through a small private result or attributes. The implementation plan
  pins the exact shape.
- **API:** `pymarxan.solvers.write_problem(problem, path, *, solver=None, config=None) -> Path`.
  - **Format:** `.lp` uses `model.writeLP(path)` and `.mps` uses `model.writeMPS(path)`. Any
    other suffix raises `ValueError`.
  - **Default solver:** `ZoneMIPSolver()` for a `ZonalProblem`, `MIPSolver()` otherwise.
  - **Errors:** a solver without `build_model` raises `TypeError` ("write_problem supports MIP
    solvers only").
  - **Not exported:** the time limit and gap. No-good cuts from `config.metadata` are included
    only when that `config` is passed.

## 4. `presolve_check` (D4)

- **API:** `pymarxan.presolve_check(problem) -> list[str]`, in a new module
  `src/pymarxan/models/presolve.py`, exported from `pymarxan`. It never raises; any internal
  error becomes a message. Each message names the entity and the numbers involved.
- **Checks** (for a `ZonalProblem`, PU cost means zone costs and targets mean zone and overall
  targets):

| # | Condition | Example message |
|---|---|---|
| P1 | non-finite or negative cost | "PU 7 has cost -3.0 (negative)" |
| P2 | max/min nonzero cost ratio above 1e6 | "costs span 1.2e+07× (min 0.01, max 1.2e+05): solvers may lose precision" |
| P3 | same for nonzero `pu_vs_features` amounts | "amounts span …" |
| P4 | same for nonzero boundary values | "boundary lengths span …" |
| P5 | a feature with zero total amount | "feature 4 occurs in no planning unit" |
| P6 | target above the total amount. Core: target × MISSLEVEL > Σ amount. Zones: reuse `diagnose_zone_infeasibility` | "feature 2 target 50 exceeds its total amount 30" |
| P7 | every PU locked in or out (core status 2/3; zones: every rule `forced` or empty) | "every planning unit is locked: nothing to optimise" |
| P8 | BLM > 0 and BLM × max boundary vs median nonzero cost ratio above 1e6 or below 1e-6 | "BLM × boundary is 3.1e+07× the median cost: boundary will dominate" |
| P9 | `COSTBUDGET` set and below the cheapest selectable PU (core: min cost of non-locked-out PUs; zones: min over allowed nonzero zone costs) | "COSTBUDGET 5 is below the cheapest planning unit (8)" |

- **Solver flag:** `MIPSolver` / `ZoneMIPSolver(presolve_check: bool | str = False)`.
  - `True` runs the check at the start of `solve()` and of `build_model()`. If there are
    messages, it emits one `PresolveWarning(UserWarning)` that joins them, with `stacklevel`
    pointing at the caller.
  - `"strict"` raises `ValueError` with the joined messages.
  - Any other value is a `ValueError` at construction.
  - `PresolveWarning` is exported from `pymarxan`.

## Out of scope

- Per-zone budgets.
- Penalties (BLM / zone boundary) in budgeted objectives.
- Objective builders for `MIPSolver`: that is Stage 3. Only `min_shortfall` and the `shortfall`
  keyword are added here, in the existing `if/elif` style.
- Zone SA / heuristic / II objectives other than the MarZone penalty objective.
- Writing models for non-MIP solvers.
- MarZone `AVAILABLEZONE` semantics, beyond the doc note.

## Testing

1. **Zone objective anchors.** On the MarZone anchor (`tests/pymarxan/zones/marzone_anchor.py`)
   and the PU4 variant, enumerate every assignment by brute force for each objective and budget.
   The MIP optimum must equal the enumerated optimum. Use at least two budgets per budgeted
   objective, one of which forces a trade-off.
2. **Target unit (D1).** A fixture where one feature has an overall target and a zone target,
   and the budget pays for only one of them. `max_features` = 1, not 0 and not 2.
3. **Relative vs absolute (D7).** A fixture where the relative and absolute
   `min_largest_shortfall` optima differ. Both solvers agree with enumeration, and
   `shortfall="absolute"` reproduces v0.38's choice. Record that value on v0.38 code before the
   change, as in v0.38's golden practice.
4. **`min_set` unchanged.**
   - Every existing zone MIP test passes.
   - A golden check: a constraint/objective dump of `build_model()` on the anchor equals a v0.38
     export recorded before the refactor. v0.38 has no `build_model`, so capture the `LpProblem`
     at its `model.solve(...)` call, for example by patching `pulp.LpProblem.solve`.
   - `tests/data/simple` = 35.0.
5. **Mandatory allocation.**
   - With the parameter on, every solver's output has `rule_violations == []`, and only status-3
     PUs are in zone 0.
   - The MIP optimum with it on is ≥ the optimum with it off, and equals the enumerated
     optimum on the anchor.
   - With it off, the v0.37 / v0.38 golden tests pass unchanged.
   - A value of `2` is a `validate()` error, and solvers treat it as off.
6. **write_problem.**
   - `tests/data/simple` exported to `.lp` and `.mps`, read back with PuLP and solved with CBC,
     gives 35.0.
   - The zone anchor round-trip gives the solver's objective.
   - Error cases: a bad suffix and a non-MIP solver.
7. **presolve_check.**
   - Each of P1–P9 has a test that triggers it.
   - `tests/data/simple` and the zone fixtures give `[]`.
   - Flag tests cover: warn once, `"strict"` raising, off staying silent, and a bad flag value.
8. **Gate.** `make check` green, parity MIP 35.0, every golden test green.

## Performance

- `min_set` builds the same model as v0.38.
- The other objectives add *O(#terms)* variables, which is small next to *n_pu × n_zones*.
- `presolve_check` is *O(n_pu + n_rows)*, using numpy reductions over the existing tables, and
  only runs when asked.
- No bench changes are needed. The existing zone and MIP benches must stay non-regressing.
