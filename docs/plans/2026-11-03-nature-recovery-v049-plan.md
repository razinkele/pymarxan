# Nature recovery core (v0.49.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `pymarxan.recovery` — protect / restore actively / restore passively / no action as a Marxan-with-Zones problem with expected restored amounts — plus the zone-SA "no action" fix, zone-project saving and `zone_mip` registration (spec §2, §3, §7 R1–R6).

**Architecture:** `RecoveryInputs` (standalone model) → `build_recovery_problem` → an ordinary `ZonalProblem` (zones 1/2/3, unassigned = no action, habitat `h` → features `2h−1` / `2h`, overall targets in expected units) plus a `RecoveryMapping`. `solve_recovery` picks the zone MIP (targets or budget mode) or `RunModePipeline` RUNMODE 2; `evaluate_recovery` reads the zone solution back per habitat; a CSV/JSON sidecar beside a saved zone project makes it reloadable. No Shiny code (v0.50.0).

**Tech Stack:** Python ≥ 3.11, pandas, NumPy, PuLP (CBC), pytest under the `shiny` micromamba env, ruff, mypy.

**Spec:** `docs/plans/2026-11-03-nature-recovery-workflow-design.md` (rev 2, approved); review `docs/plans/2026-11-03-nature-recovery-workflow-review.md`.

**Reference prototype:** local branch `proto/v0.49-recovery` (worktree `/tmp/claude-1000/-home-razinka-marxan/be611b2f-c158-4a82-8be3-dc66e22357b8/scratchpad/proto-rec`, not pushed). One commit per task, each green:

| Task | Commit |
|---|---|
| 1 (R1) zone SA no action | `5d61b06` |
| 2 (R2) inputs + builder | `3020937` |
| 3 (R3) solving + registry | `9dd671a` |
| 4 (R4) results | `c71b6c3` |
| 5 (R5) persistence | `b01e76c` |
| 6 (R6) docs | `a67f4b9` |

Implementers may `git show <sha> -- <path>` for reference; the controller compares each task's diff with the prototype (`git diff <sha>~1 <sha>`).

## Global Constraints

- Tests run with `/opt/micromamba/envs/shiny/bin/pytest` and `PYTHONPATH=src:.` (tests import `tests.benchmarks.conftest`); never `.venv` (no rasterio/ipyleaflet).
- Python 3.11 floor; `from __future__ import annotations` in every file; full type hints.
- Strict TDD: failing test first, then code.
- `~/.local/bin/ruff check src tests examples` (E, F, I, UP; line length 99) — **ruff the test files at every commit**; `/home/razinka/marxan/.venv/bin/mypy src/pymarxan src/pymarxan_shiny --ignore-missing-imports` clean.
- Benches are comparative only (ratio vs a baseline in the same run), never absolute wall-clock; run `-m bench` deliberately.
- New `ConservationProblem` fields would be `kw_only=True` with defaults — this plan adds none (`RecoveryInputs` is standalone, spec §3.1).
- I/O in `read_*`/`write_*` pairs; `save_zone_project` is the inverse of `load_zone_project`.
- No ROWER terms or code in `src/` (ROWER is named only in test comments as the anchor's provenance).
- The caveat is spec §3.7 verbatim; the only variable part is the bracket `[no time lag is modelled | a maturity factor of … is applied]` (Task 4).
- Placeholders keep their labels: restore cost factor 2.5, protect 0.3, restore 0.2, passive η 0.5 and cost factor 1.0 are reported as `"placeholder"` in `RecoveryMapping.provenance` when left at default.
- Correctness anchor untouched: `examples/validate_marxan_parity.py` still reports MIP 35.0.
- Existing single-zone behaviour unchanged; the only changed existing results are zone-SA results (Task 1, CHANGELOG `Changed`).
- PuLP 3.3.x emits `DeprecationWarning`s: tests that turn warnings into errors must ignore `DeprecationWarning`.

## Review Focus

1. **Zero-cost units** (public land, already-owned parcels) with the default `restore_cost_factor` give restore cost 0 → `validate()` error naming the units, not a silent tie between restore and no action. Test `test_zero_cost_unit_makes_factor_restore_cost_zero` (Task 2).
2. **Non-integral or non-positive habitat ids** (`2.5`, `0`) cannot map to `2h−1`/`2h` → error "must be a positive integer". Test `test_habitat_ids_must_be_positive_integers` + `test_non_integral_habitat_id_is_rejected` (Task 2).
3. **All potential in locked-out (status 3) units** → the habitat's restore target is 0 with a "no restoration potential" warning, not an infeasible build. Test `test_potential_only_in_locked_out_units_warns` (Task 2).
4. **Budget below what locked-in units must cost** → `ValueError` naming the floor ("least cost any plan must pay"), not a generic "targets compete". Test `test_budget_below_the_locked_in_cost_names_the_floor` (Task 3).
5. **Saving over an older project folder** (a previous save had `pulock.dat`/`zonecontrib.dat`) → the reload must not pick up stale tables. Test `test_saving_without_a_table_removes_its_stale_file` (Task 5).

---

### Task 1: Zone SA can choose "no action" (R1, spec §2)

**Files:**
- Modify: `src/pymarxan/zones/solver.py` (new `_proposal_options`; use it for proposals and the start draw)
- Modify: `src/pymarxan/zones/cache.py` (`_boundary_delta_add/_remove` take the assignment, not a mask)
- Modify: `src/pymarxan_shiny/modules/solver_config/solver_picker.py` (zone_sa help text)
- Modify: `tests/pymarxan/zones/test_zone_golden_v037.py` (regenerate 3 SA entries + dated header note)
- Modify: `tests/benchmarks/bench_zone_pu_restrictions.py` (`_opts` uses `_proposal_options`)
- Create: `tests/pymarxan/zones/test_zone_sa_no_action.py`, `tests/benchmarks/bench_zone_sa_no_action.py`
- Modify: `CHANGELOG.md` (`[Unreleased]` → `### Changed`)

**Interfaces:**
- Produces: `_proposal_options(rules: list[PUZoneRule], zone_ids: list[int]) -> list[np.ndarray]` in `pymarxan.zones.solver` (per PU: `[0, *allowed]` when `rule.may_be_unassigned`, else `allowed`; unrestricted PUs share one array per kind). `_propose` signature unchanged.

- [ ] **Step 1: Write the failing tests** in `test_zone_sa_no_action.py` (fixture: 4 PUs cost 1, zones 1/2 with zone cost = zone id, one feature target 1 held only by PU 1, `NUMITNS 20000, NUMTEMP 100`):

```python
def test_sa_leaves_unneeded_units_unassigned():  # seed 1
    assert sol.zone_assignment.tolist() == [1, 0, 0, 0] and sol.cost == pytest.approx(1.0)
def test_sa_matches_mip_on_c1_probe():  # seeds 0..4: SA cost == MIP cost
def test_options_include_zero_only_when_unassigned_is_allowed():
    # statuses (0, 2, 3, 0), pu_zones {4: {1, 2}}
    assert opts[0].tolist() == [0, 1, 2] and opts[3].tolist() == [1, 2]
    assert set(fixed_zones(rules)) == {1, 2}
def test_mandatory_allocation_never_proposes_zero():  # MANDATORYALLOCATION 1: no 0 anywhere
def test_start_draw_includes_no_action():  # NUMITNS 0, seeds 0..9: some start has a 0
def test_tracked_objective_equals_full_objective_after_every_flip():
    # make_zone_problem(40, 5, 3, density 0.4, seed 7) + contributions (1.0/0.5/0.2) +
    # chain connectivity, BLM 0.7, CONNECTIVITY_WEIGHT 0.3; 400 random flips over zones 0..3
    # via ZoneProblemCache: tracked == compute_zone_objective (rel 1e-9) after each flip;
    # > 50 moves to and > 50 from zone 0
```

- [ ] **Step 2: Run** `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones/test_zone_sa_no_action.py -q --no-cov` — Expected: FAIL (`ImportError: _proposal_options`).
- [ ] **Step 3: Implement** `_proposal_options` and use it in `ZoneSASolver.solve` for `opts` (the start draw already samples `opts`). Change `_boundary_delta_add/_remove(idx, assignment)` to test `assignment[j] > 0` per neighbour — building `assignment > 0` per call made every move to/from 0 O(n_pu). Picker text: "Each planning unit is assigned to a zone or left unassigned (no action), unless MANDATORYALLOCATION is 1 or the unit is locked or restricted to zones (pulock / puzone / status 2)."
- [ ] **Step 4: Run** zone tests: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q --no-cov -m "not bench" -k "zone or Zone or run_mode or phase3 or phase20 or capability or picker"` — Expected: only `test_zone_golden_v037.py::test_matches_v037[small-sa-None-None|large-sa-None-None|large-sa-10.0-None]` fail. Regenerate with `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python tests/pymarxan/zones/test_zone_golden_v037.py`, replace **only** the three SA entries (small SA becomes `((0, 1, 0, 0), 184.0)` ×2, was 318.8) and add a dated `2026-11-03 (v0.49.0)` header bullet saying why; heuristic/II/improve values must be byte-identical. MANDATORYALLOCATION / puzone / lock tests must not move. Re-run: PASS.
- [ ] **Step 5: Bench** — create `bench_zone_sa_no_action.py`: `test_no_action_delta_does_not_scale_with_n_pu` (per to-zero delta, 20 000 vs 1 000 PUs, n_feat 10, density 0.1: ratio ≤ 1.5) and `test_no_action_delta_close_to_zone_swap_delta` (1 000 PUs, 50 features: to-zero / zone-swap ≤ 1.5). Run `-m bench` on the four zone bench files on `main` and on the branch; compare `bench_zone_pu_restrictions` printed ratios (main: 1.014 / 1.056 / 0.894; prototype: 1.078 / 1.042 / 0.893) and durations. Check the guard bites: with the old mask code the 20k/1k ratio was 1.656 (FAIL).
- [ ] **Step 6: CHANGELOG** `[Unreleased] ### Changed` — "Zone SA can choose 'no action'" entry: behaviour change on every zone problem with unlocked, unrestricted units; seeded runs take new paths.
- [ ] **Step 7: Commit** `git add src/pymarxan/zones/solver.py src/pymarxan/zones/cache.py src/pymarxan_shiny/modules/solver_config/solver_picker.py tests/pymarxan/zones/test_zone_sa_no_action.py tests/pymarxan/zones/test_zone_golden_v037.py tests/benchmarks/bench_zone_pu_restrictions.py tests/benchmarks/bench_zone_sa_no_action.py CHANGELOG.md && git commit -m "fix(zones): zone SA proposes no action (zone 0) for units that may stay unassigned"`

### Task 2: `RecoveryInputs`, checks and `build_recovery_problem` (R2, spec §3.1–3.3, §3.6)

**Files:**
- Create: `src/pymarxan/recovery/__init__.py`, `src/pymarxan/recovery/inputs.py`, `src/pymarxan/recovery/builder.py`
- Create: `tests/pymarxan/recovery/__init__.py`, `tests/pymarxan/recovery/recovery_anchor.py`, `tests/pymarxan/recovery/test_recovery_builder.py`, `tests/pymarxan/recovery/test_recovery_inputs.py`

**Interfaces:**
- Produces (`inputs.py`): `PassiveAction(efficacy: float = 0.5, efficacy_by_habitat: Mapping[int, float] | None = None, cost_factor: float = 1.0)` frozen dataclass with `efficacy_for(habitat) -> float`; `RecoveryInputs` `@dataclass(eq=False)` fields `base: ConservationProblem, potential: pd.DataFrame (pu, feature, amount), p_success: float | pd.Series (indexed by unit id), restore_cost: pd.Series | None = None, restore_cost_factor: float = 2.5, restore_cost_by_habitat: Mapping[int, float] | None = None, passive: PassiveAction | None = None, maturity: Mapping[int, float] | None = None, horizon_years: float | None = None, protect_prop: float | Mapping[int, float] = 0.3, restore_prop: float | Mapping[int, float] = 0.2, restore_target: Mapping[int, float] | None = None, spf: float | None = None`; methods `validate() -> list[str]`, `provenance() -> dict[str, str]`, views `habitat_ids()` (ascending), `habitat_names()`, `unit_ids()` (base order), `unit_costs()`, `unit_statuses()`, `p_by_unit()`, `potential_matrix()`, `existing_matrix()` (units × habitats), `restore_cost_by_unit()` (incl. implementation cost per unit of raw potential), `passive_cost_by_unit() -> np.ndarray | None`, `maturity_for(h)`, `protect_prop_for(h)`, `restore_prop_for(h)`.
- Produces (`builder.py`): constants `ZONE_NO_ACTION=0, ZONE_PROTECT=1, ZONE_RESTORE_ACTIVE=2, ZONE_RESTORE_PASSIVE=3`, `ACTION_NAMES = {0: "no_action", 1: "protect", 2: "restore_active", 3: "restore_passive"}`; `existing_feature_id(h) = 2h−1`, `potential_feature_id(h) = 2h`; `RecoveryInputError(ValueError)` with `.errors: list[str]`; `RecoveryInputWarning(UserWarning)`; `RecoveryCheck(NamedTuple: errors, warnings)`; `RecoveryMapping` (`@dataclass(eq=False, frozen=True)`: `habitats` DataFrame [habitat, name, existing_feature, potential_feature, existing_total, existing_target, restore_max, restore_target, maturity, efficacy], `units` [pu, cost, status (as given), p_success, restore_cost, passive_cost, restorable], `potential` [pu, habitat, amount], `zones: dict[int, str]`, `passive: bool`, `horizon_years: float | None`, `spf_scaled: bool`, `provenance: dict[str, str]`, `action_names(assignment) -> list[str]`); `check_recovery_inputs(inputs) -> RecoveryCheck`; `build_recovery_problem(inputs) -> tuple[ZonalProblem, RecoveryMapping]`.

- [ ] **Step 1: Write the anchor fixture** `recovery_anchor.py`: `anchor_base(status=(0,0,0,0))` (units 1–4, cost `[1, 1, 1, 2]`, habitat 1 "wetland", existing PU1 10, PU2 6, features carry `target`/`spf` — base `validate()` requires them), `anchor_inputs(status=..., **kw)` (potential PU3 8, PU4 12; p `{1: .5, 2: .5, 3: .5, 4: .8}`; protect 0.5; restore_prop 0.45; factor 2.5), `brute(problem)` (all rule-respecting assignments meeting every overall target, cheapest first, cost via `compute_zone_cost`), `fingerprint(problem)` (sha256 over every table sorted by key columns with sorted columns via `pd.util.hash_pandas_object(index=False)`, plus `json.dumps(parameters, sort_keys=True)`).
- [ ] **Step 2: Write the failing builder tests** (`test_recovery_builder.py`), MIP via `ZoneMIPSolver(mip_backend="cbc")` with `MIP_GAP 0.0`:

```python
test_anchor_tables:  zones [1, 2] ["protect", "restore_active"]; MANDATORYALLOCATION 0
  features ids [1, 2], names ["wetland_existing", "wetland_potential"], targets [8.0, 0.45*13.6]
  pv {(1,1): 10, (1,2): 6, (2,3): 4.0, (2,4): 9.6}
  contributions {(1,1): 1, (1,2): 1, (2,1): 0, (2,2): 1}
  zone costs {(1,1): 1, (1,2): 3.5, (2,1): 1, (2,2): 3.5, (3,1): 1, (3,2): 2.5, (4,1): 2, (4,2): 5}
  pu_locks/pu_zones None; restore_max 13.6; restorable [F, F, T, T]
test_rower_anchor_unique_optimum[{} | restore_target {1: 9.0}]: brute[0] == (6.0, (1,0,0,2)), brute[1] > 6.0, MIP == ([1,0,0,2], 6.0)
PASSIVE_CASES = [(0.2, 9.0, [1,0,0,2], 6.0), (0.2, 6.5, [1,0,3,3], 1.6), (0.2, 8.5, [1,0,2,3], 3.9),
                 (1.0, 9.0, [1,0,0,2], 6.0), (1.0, 6.5, [1,0,3,3], 4.0), (1.0, 8.5, [1,0,2,3], 5.5)]
test_passive_cases: PassiveAction(efficacy=0.5, cost_factor=cf), restore_target {1: t}; contrib (1,3)=1, (2,3)=0.5; unique brute optimum == MIP
test_existing_habitat_is_kept_when_restored: U1 existing 10 + potential 8, U2 existing 10, U3 potential 8, p .5, protect .5, restore_target 4 -> ([2,0,0], 2.5) (3.5 if restore wiped existing)
test_status_2_without_potential_is_locked_to_protect: status (0,2,0,0), restore_prop 0 -> pu_locks [[2,1]], MIP ([1,1,0,0], 2.0)
test_status_2_with_potential_is_kept_protected_or_restored: status (0,0,2,0), passive -> pu_zones [[3,1],[3,2],[3,3]], rule {1,2,3} not unassigned, cost 2.0, PU3 in (1, 3)
test_status_3_units_are_excluded_from_restore_max: status (0,0,0,3), restore_prop 1.0 -> restore_max 4.0, ([1,0,2,0], 3.5)
test_status_1_is_treated_as_0: built status [0,0,0,0], MIP ([1,0,0,2], 6.0)
test_unit_without_potential_is_never_restored: potential only PU4, passive cf .2, target 4 -> brute[0] (1.4, (1,0,0,3)); zc (3,2)=3.5, (3,3)=1.2
test_non_contiguous_habitat_ids: habitats 10 "fen", 3 "bog" -> feature ids [5, 6, 19, 20], names bog_*, fen_*
test_build_is_deterministic_and_row_order_free: fingerprints equal across two builds and shuffled potential rows
```

The PASSIVE_CASES optima were brute-forced on the semantics independently of the builder (cost factor 0.2 reproduces the review's b1–b3 6.0/1.6/3.9; the D2 default 1.0 gives 6.0/4.0/5.5 — the spec's §3.5 numbers assumed 0.2).

- [ ] **Step 3: Write the failing input tests** (`test_recovery_inputs.py`): `test_anchor_is_valid_and_has_no_warnings` (`check == ([], [])`); parametrised `test_validate_reports` with these (kw → substring) pairs: unknown unit 9 → "potential names unknown units 9"; unknown habitat 7; amount −1 → "potential amounts must be finite and >= 0"; duplicate rows → "duplicate (pu, feature)"; p 1.2 → "p_success must be in [0, 1]"; Series missing unit 4 → "p_success has no value for units 4"; η 0 → "passive efficacy must be in (0, 1]"; η_by_habitat 1.5; maturity 0 → "maturity[1] = 0.0 must be in (0, 1]"; maturity 0.6 without horizon → "horizon_years must be stated"; restore_cost 0 in a unit → "restore cost must be > 0 in every unit"; passive cost_factor 2.5 → "passive recovery is never chosen there"; protect 1.5; restore_target for habitat 2 → "restore_target names unknown habitats 2"; spf 0 → "spf must be > 0". Plus `test_zero_cost_unit_makes_factor_restore_cost_zero` (cost `[0, 1, 1, 2]` → error with "units 1"), `test_habitat_ids_must_be_positive_integers` (id 0), `test_non_integral_habitat_id_is_rejected` (id 2.5), `test_infeasible_restore_target_is_an_error_naming_the_habitat` (target 14 → one error starting "restore target for habitat 'wetland' (expected units)" containing "is 14 but at most 13.6 is reachable"; `build_recovery_problem` raises `RecoveryInputError` with the same `.errors`), `test_restore_prop_one_is_feasible_by_construction`, `test_habitat_without_potential_warns` (exact text "habitat 'wetland' has no restoration potential (with p_success > 0 in a unit that may act); its restore target is 0"; `pytest.warns(RecoveryInputWarning)`), `test_potential_only_in_locked_out_units_warns` (status (0,0,3,3) → same warning, errors []), `test_restore_cheaper_than_protect_on_existing_habitat_warns` (potential on PU1 2.0, restore_cost PU1 0.5 → one warning containing "units 1"), `test_provenance_marks_defaults_as_placeholders`, `test_build_has_no_warnings_on_the_anchor`.
- [ ] **Step 4: Run** `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/recovery -q --no-cov` — Expected: FAIL (`ModuleNotFoundError: pymarxan.recovery`).
- [ ] **Step 5: Implement** `inputs.py` and `builder.py` (export every name above from `__init__` with a sorted `__all__`). Decisions the tests do not spell out:
  - Errors stop early: base `validate()` errors (prefixed `"base: "`) and bad habitat ids return before the rest; cost-derived checks (restore/passive cost > 0, passive ≥ active) run only when everything else is valid.
  - `restorable` = any `potential × p > 0`. Restore zones on non-restorable units cost **unit cost + restore cost** ("surcharge"): with H2 (existing credited under every action) a passive cost factor ≤ 1 would otherwise tie or beat protection on units with no potential.
  - `restore_max = g_h × Σ_{status ≠ 3} potential × p`; existing target = `protect_prop × Σ existing` over **all** units (Marxan proportion convention).
  - Built status: 1 → 0, 2 and 3 kept; status-2 units get a `pu_locks` row (zone 1) if not restorable, else `pu_zones` rows for every listed zone.
  - Parameters: copy `BLM, NUMITNS, NUMTEMP, COOLING, MIP_TIME_LIMIT, MIP_GAP` from base when present, `BLM` default 0.0, then force `MANDATORYALLOCATION 0`, `MISSLEVEL 1.0`; boundary copied; probability/connectivity dropped.
  - Frames sorted (features by habitat id, puvspr by `(species, pu)`), rows with amount 0 omitted.
  - `check_recovery_inputs`: infeasibility messages from `diagnose_zone_infeasibility` on the assembled problem, prefixed by habitat (regex on `feature (\d+)`). `build_recovery_problem` asserts the built problem's `validate() == []`.
  - spf in this task: `inputs.spf` or `1.0`; Task 3 replaces `None` with the scaled rule.
- [ ] **Step 6: Run** the recovery tests — Expected: PASS (42). ruff + mypy clean.
- [ ] **Step 7: Commit** `git add src/pymarxan/recovery tests/pymarxan/recovery && git commit -m "feat(recovery): RecoveryInputs, checks and build_recovery_problem"`

### Task 3: Solving modes, scaled spf, registry (R3, spec §3.5, §3.9)

**Files:**
- Create: `src/pymarxan/recovery/solve.py`, `tests/pymarxan/recovery/test_recovery_solve.py`
- Modify: `src/pymarxan/recovery/builder.py` (`SPF_SCALE`, `scaled_spf`, default spf), `src/pymarxan/recovery/__init__.py`, `src/pymarxan/solvers/registry.py`

**Interfaces:**
- Consumes: Task 2's `build_recovery_problem`, `PassiveAction`, `RecoveryInputs`.
- Produces: `SPF_SCALE = 10.0`; `scaled_spf(problem: ZonalProblem) -> dict[int, float]`; `solve_recovery(problem: ZonalProblem, *, mode: str = "targets", budget: float | None = None, method: str = "mip", budget_objective: str = "min_shortfall", config: SolverConfig | None = None) -> Solution` with `metadata["recovery_mode"|"recovery_method"|"boundary_dropped"]`; `RECOVERY_MIP_SIZE_WARNING = 50_000`; `RecoverySizeWarning(UserWarning)`; registry name `"zone_mip"`.

- [ ] **Step 1: Write the failing tests** (module-level `filterwarnings("ignore::DeprecationWarning")`; `FAST = {"NUMITNS": 20_000, "NUMTEMP": 200}`):

```python
test_scaled_spf_is_scale_times_dearest_cost_per_unit_supplied:
    scaled_spf(anchor) == {1: 10 * 3.5 / 6, 2: 10 * 0.625}; features spf equal; SPF_SCALE == 10.0
test_explicit_spf_is_used_as_given: spf=3.0 -> [3.0, 3.0], not mapping.spf_scaled
test_targets_mode_mip_hits_the_anchor_optimum[anchor + 6 PASSIVE_CASES]: cost == optimum, all_targets_met, metadata ("targets", "mip", False)
test_heuristic_on_the_anchors[...] (slow): config SolverConfig(5, seed=1);
    "passive-0.2-9.0": not met, shortfall > 0, cost < optimum; others: met and cost == optimum
test_b1_penalised_minimum_is_infeasible_and_reported (slow): [1,0,2,3], cost 3.9, shortfall 0.2,
    objective == 3.9 + SPF_SCALE * 0.625 * 0.2
test_heuristic_is_pipeline_runmode_2_itimptype_3: spy on RunModePipeline.solve sees RUNMODE 2, ITIMPTYPE 3
test_budget_mode_best_plan_within_budget: passive cf 1.0, target 8.5: budget 4.0 -> cost <= 4, objective "min_shortfall", not met; budget 5.5 -> met, cost 5.5
test_budget_mode_flags_dropped_boundary: chain boundary, BLM 0.5: pytest.warns(UserWarning, "ignores boundary penalties"), boundary_dropped True; targets mode False
test_budget_mode_needs_a_budget_and_the_mip: "budget mode needs a budget", "budget mode is solved by the zone MIP only", "mode must be"
test_targets_mode_raises_when_the_mip_finds_no_plan: tampered target 50 -> ValueError "no plan meets every target"
test_budget_below_the_locked_in_cost_names_the_floor: status (0,2,0,0), restore_prop 0, budget 0.5 -> ValueError "least cost any plan must pay"
test_size_guard_warns_above_the_threshold: monkeypatch RECOVERY_MIP_SIZE_WARNING=7 -> RecoverySizeWarning "8 unit-zone decisions"
test_registry_offers_zone_mip_and_pipeline_on_zone_problems: {"zone_mip","zone_sa","pipeline"} <= list_solvers(); zone_mip -> [1,0,0,2]; pipeline solves the ZonalProblem
test_random_instances_heuristic_is_honest[seed 0..5] (slow): 7 units, 2 habitats, passive cf .5;
    all_targets_met == (shortfall == 0); met -> cost >= _exact_optimum; else shortfall > 0
```

`_exact_optimum` is a vectorised brute force (all `(n_zones+1)^n_pu` assignments via `ZoneProblemCache` matrices), **not** the MIP — see Step 3's CBC note.

- [ ] **Step 2: Run** `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/recovery/test_recovery_solve.py -q --no-cov` — Expected: FAIL (`ImportError: SPF_SCALE`).
- [ ] **Step 3: Implement.** The spf rule (not determined by the tests beyond the anchor values):

```python
# per feature f: SPF_SCALE * max over puvspr rows (u, f, a) and zones z with s = a * contrib(f, z) > 0
#   of zone_cost(u, z) / s ; 1.0 when nothing supplies f
```

Rationale to keep in the docstring: integrality means no finite proportional spf guarantees heuristic feasibility (anchor b1 at cost factor 0.2 needs ≥ 17); on the anchors plus 15 random 8-unit instances SA+II met all targets in 50 % of runs at ×1, 67 % at ×2, 95 % at ×10 (always at the MIP optimum when met), 100 % at ×100 but with gaps up to 4 %. Builder: `inputs.spf is None` → set `features["spf"]` from `scaled_spf` after assembling.
  `solve_recovery`: validate mode/method/budget first; heuristic = `RunModePipeline` on `problem.copy_with(parameters={..., "RUNMODE": 2, "ITIMPTYPE": 3})`, default `SolverConfig(num_solutions=10, seed=0)`, return `min(runs, key=objective)`; MIP path warns `RecoverySizeWarning` when `n_pu × n_zones > RECOVERY_MIP_SIZE_WARNING`; budget mode solves a copy with `spf = 1.0` (equal weights) and `COSTBUDGET`; suppress `ZoneMIPNoSolutionWarning` and raise `ValueError` instead — targets mode: `"no plan meets every target: " + diagnose_zone_infeasibility(...)`, budget mode: `"no plan within the budget: COSTBUDGET {b} is below the least cost any plan must pay ({min_forced_cost})"` when that holds. `boundary_dropped = objective != "min_set" and boundary is not None and BLM != 0`. Registry: `reg.register("zone_mip", ZoneMIPSolver)` ("pipeline" already dispatches on `ZonalProblem`; single-zone picker untouched).
  **CBC note for the reviewer:** PuLP's bundled CBC 2.10.3 returns a suboptimal "Optimal" (8.405 vs 6.755) on random instance seed 2 — its `preprocess` step; conda CBC 2.10.13 and `-preprocess off` are correct. Do not compare heuristics with the MIP on random data.
- [ ] **Step 4: Run** `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/recovery tests/pymarxan/solvers/test_registry.py tests/test_integration_phase4.py tests/test_integration_phase5.py tests/test_integration_phase6.py -q --no-cov` — Expected: PASS.
- [ ] **Step 5: Commit** `git add src/pymarxan/recovery src/pymarxan/solvers/registry.py tests/pymarxan/recovery/test_recovery_solve.py && git commit -m "feat(recovery): solve_recovery targets/budget modes, scaled spf, zone_mip in registry"`

### Task 4: `RecoveryResult`, `evaluate_recovery`, `success_sensitivity`, caveat (R4, spec §3.7)

**Files:**
- Create: `src/pymarxan/recovery/evaluate.py`, `tests/pymarxan/recovery/test_recovery_evaluate.py`
- Modify: `src/pymarxan/recovery/__init__.py`

**Interfaces:**
- Consumes: `RecoveryMapping`, `ACTION_NAMES`, zone constants (Task 2); `solve_recovery` (Task 3).
- Produces: `RECOVERY_CAVEAT: str` (template with `{maturity}`), `NO_TIME_LAG = "no time lag is modelled"`, `DEFAULT_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25)`, `recovery_caveat(mapping) -> str`; `RecoveryResult` `@dataclass(eq=False)`: `assignment: np.ndarray, actions: pd.DataFrame [pu, zone, action], habitats: pd.DataFrame, cost_by_action: dict[str, float], total_cost: float, targets_met: bool, mode: str | None, method: str | None, solver: str | None, boundary_dropped: bool, caveat: str, mapping: RecoveryMapping`; `evaluate_recovery(solution: Solution, problem: ZonalProblem, mapping: RecoveryMapping) -> RecoveryResult`; `success_sensitivity(result: RecoveryResult, multipliers: Sequence[float] = DEFAULT_MULTIPLIERS) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests:**

```python
test_caveat_is_the_spec_text: RECOVERY_CAVEAT.replace("{maturity}", "[no time lag is modelled | a maturity factor of … is applied]") == spec §3.7 text (pasted verbatim)
test_anchor_result: actions ["protect","no_action","no_action","restore_active"]; cost_by_action {"protect": 1.0, "restore_active": 5.0}; total 6.0; targets_met;
    (mode, method, solver) == ("targets", "mip", "Zone MIP (PuLP)"); existing_kept 10, existing_target 8, restored_active 9.6, restored_passive 0, restore_target 6.12, shortfalls 0;
    caveat == RECOVERY_CAVEAT.format(maturity=NO_TIME_LAG)
test_table_leads_with_shortfall_and_never_sums_existing_and_restored: plan [1,0,2,3], passive cf .2, target 9:
    columns[:4] == ["habitat","name","restore_shortfall","protect_shortfall"]; no column containing "total"/"secured"; restore_shortfall 0.2, not met
test_rows_with_the_largest_relative_shortfall_come_first: fen/bog, plan [1,1] -> names ["bog","fen"]
test_existing_habitat_counts_when_restored: shared-unit inputs -> ["restore_active","no_action","no_action"], kept 10, restored 4
test_passive_amounts_and_cost_by_action: [1,0,2,3], passive default, target 8.5 -> active 4.0, passive 4.8, {"protect": 1, "restore_active": 2.5, "restore_passive": 2.0}
test_success_sensitivity_rescored_without_resolving: restored_at_0.5 4.4, _0.75 6.6, _1 8.8, _1.25 11.0 (PU4 p capped), lowest_multiplier_met 1.0; m=1 == active + passive
test_success_sensitivity_caps_p_at_one_and_reports_nan_when_never_met: (2.0, 0.5) -> restored_at_2 14.0, lowest 2.0; (0.5, 0.75) -> NaN; () -> ValueError "multipliers"
test_maturity_factor_scales_restored_amounts_and_fills_the_caveat: maturity {1: .5}, horizon 10, plan [1,0,0,2] -> restored 4.8, target 0.45*6.8, caveat has "a maturity factor of 0.5 is applied"
test_evaluate_needs_a_zone_solution: zone_assignment None -> ValueError "zone solution"
```

- [ ] **Step 2: Run** `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/recovery/test_recovery_evaluate.py -q --no-cov` — Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** in `evaluate.py`. Amounts come from the built problem (`amount × get_contribution(feature, zone)` per action zone), so they equal what the solvers optimised; existing kept sums the existing feature over every action zone. Met flags use `pymarxan.zones.objective._meets` (target ≤ 0 counts as met). Row order: largest of restore/protect relative shortfall first, ties by habitat id. Maturity clause: all g = 1 → `NO_TIME_LAG`; one habitat → `"a maturity factor of {g:g} is applied"`; several → `"a maturity factor of 0.5 (fen), 1 (bog) is applied"`. Sensitivity: per unit, raw potential × `min(1, p × m)` × (g under active, η·g under passive), multipliers sorted ascending, columns `restored_at_{m:g}`, `lowest_multiplier_met` = first m meeting the target, NaN if none.
- [ ] **Step 4: Run** the recovery tests — Expected: PASS. ruff + mypy.
- [ ] **Step 5: Commit** `git add src/pymarxan/recovery tests/pymarxan/recovery/test_recovery_evaluate.py && git commit -m "feat(recovery): RecoveryResult, evaluate_recovery, success_sensitivity, caveat"`

### Task 5: Persistence (R5, spec §3.8)

**Files:**
- Modify: `src/pymarxan/zones/writers.py` (`save_zone_project`), `src/pymarxan/zones/__init__.py` (export)
- Create: `src/pymarxan/recovery/io.py`; Modify: `src/pymarxan/recovery/__init__.py`
- Create: `tests/pymarxan/zones/test_save_zone_project.py`, `tests/pymarxan/recovery/test_recovery_io.py`
- Modify: `tests/pymarxan/recovery/recovery_anchor.py` (if `fingerprint` is not there yet)

**Interfaces:**
- Consumes: `build_recovery_problem`, `RecoveryMapping`, `PassiveAction`, `RecoveryInputs` (Task 2); `solve_recovery` / `evaluate_recovery` in tests.
- Produces: `save_zone_project(problem: ZonalProblem, project_dir: str | Path) -> None`; in `pymarxan.recovery`: `write_recovery_inputs(inputs, project_dir) -> None`, `read_recovery_inputs(project_dir, base: ConservationProblem) -> RecoveryInputs`, `save_recovery_project(inputs, project_dir) -> tuple[ZonalProblem, RecoveryMapping]`, `load_recovery_project(project_dir) -> RecoveryProject` (`NamedTuple(inputs, problem, mapping)`); file names `potential.csv`, `recovery_units.csv`, `recovery.json`; `FORMAT = "pymarxan-recovery"`, `FORMAT_VERSION = 1`.

- [ ] **Step 1: Write the failing zone tests:** `test_marzone_fixture_round_trips` (`tests/data/zones`: load → save → load; `validate() == []`; zones, zone_costs, contributions, zone targets, zone boundary costs equal after sorting; MIP `(assignment, cost)` equal), `test_locks_and_restrictions_are_named_and_round_trip` (pu_locks {1: 2}, pu_zones {2: {1, 2}} → `input.dat` contains `PULOCKNAME pulock.dat` and `PUZONENAME puzone.dat`; tables and MIP equal), `test_saving_without_a_table_removes_its_stale_file`.
- [ ] **Step 2: Write the failing recovery I/O tests:** `_rich_inputs()` = anchor with status (1, 2, 2, 3), restore_cost `{1: 2, 2: 2.5, 3: 3, 4: 6}`, `restore_cost_by_habitat {1: 0.05}`, `PassiveAction(0.6, {1: 0.4}, 0.5)`, maturity `{1: 0.8}` + horizon 15, protect `{1: 0.4}`, restore `{1: 0.3}`, chain boundary + BLM 0.25. `test_inputs_round_trip[anchor | scalar p 0.6 + restore_target {1: 3} | rich]` (exactly the 3 files; read inputs valid; fingerprints of the rebuilt problems equal; provenance equal); `test_sidecar_contents` (format/version; habitats `[{"habitat": 1, "name": "wetland", "existing_feature": 1, "potential_feature": 2}]`; zones `{"0": "no_action", "1": "protect", "2": "restore_active", "3": "restore_passive"}`; passive `{"efficacy": 0.6, "efficacy_by_habitat": {"1": 0.4}, "cost_factor": 0.5}`; maturity `{"1": 0.8}`, horizon 15.0; units columns `["pu", "status", "p_success", "restore_cost"]`, status `[1, 2, 2, 3]` as given); `test_project_round_trip_rebuilds_problem_mapping_and_results` (fingerprint, mapping frames, MIP assignment and evaluated habitats table equal); `test_saved_zone_project_reproduces_in_pymarxan` (`load_zone_project` of the folder: valid, same MIP cost); `test_rejects_a_foreign_sidecar` ("not a pymarxan-recovery v1 file"); `test_write_refuses_invalid_inputs` ("invalid recovery inputs").
- [ ] **Step 3: Run** both files — Expected: FAIL (ImportError).
- [ ] **Step 4: Implement.** `save_zone_project`: params = problem parameters + `INPUTDIR` default + a `*NAME` key per present table (`ZONESNAME zones.dat`, `ZONECOSTNAME zonecost.dat`, `ZONECONTRIBNAME zonecontrib.dat`, `ZONETARGETNAME zonetarget.dat`, `ZONEBOUNDCOSTNAME zoneboundcost.dat`, `PULOCKNAME pulock.dat`, `PUZONENAME puzone.dat`); delete the key and a stale default-named file for absent tables; `save_project(problem.copy_with(parameters=params), dir)`, then the existing `write_zone*` / `write_pu_lock` / `write_pu_zone` writers. Docstring states the "reproducible in pymarxan, not MarZone C++" rule. Sidecar: `recovery_units.csv` restore_cost = given Series or factor × cost (before implementation cost), JSON `restore_cost_from_factor` decides which on read; `p_success_scalar` restores a scalar exactly; JSON keys are strings — cast to `int` on read. `load_recovery_project`: base = zone project's `pu.dat` (id, cost) + statuses from `recovery_units.csv`, habitats from JSON (`target 0.0`, `spf 1.0`), existing amounts = puvspr rows of the `*_existing` feature ids mapped back to habitat ids, boundary and parameters from the zone project; then `read_recovery_inputs` and `build_recovery_problem`.
- [ ] **Step 5: Run** `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/zones tests/pymarxan/recovery -q --no-cov -m "not slow"` — Expected: PASS.
- [ ] **Step 6: Commit** `git add src/pymarxan/zones/writers.py src/pymarxan/zones/__init__.py src/pymarxan/recovery tests/pymarxan/zones/test_save_zone_project.py tests/pymarxan/recovery && git commit -m "feat(zones,recovery): save_zone_project and recovery project persistence"`

### Task 6: Docs and example (R6)

**Files:**
- Create: `examples/nature_recovery.py`
- Modify: `tests/test_examples.py`, `examples/README.md`, `docs/USER_MANUAL.md` (section 13 "Nature recovery (Python API)" before "pymarxan, Marxan and prioritizr"; renumber TOC 13–16), `README.md` (tree line + feature section), `.github/copilot-instructions.md` (zones row mentions `save_zone_project`; new `recovery/` row), `CHANGELOG.md` (`[Unreleased] ### Added`)

**Interfaces:**
- Consumes: everything exported by `pymarxan.recovery` (Tasks 2–5).
- Produces: `examples/nature_recovery.py` with `build_recovery_inputs() -> RecoveryInputs` and `main() -> None`.

- [ ] **Step 1: Write the failing example tests** (fixture `recovery = _load("nature_recovery.py")`): `test_recovery_example_exact_plan` (inputs valid; 30 units, 3 zones; MIP result targets met, `total_cost == approx(17.7)`, actions == `{"no_action", "protect", "restore_passive"}` — the scalar-η artefact of spec §3.6) and `test_recovery_example_main_runs` (slow; output contains `"cost 17.70"`, `"Saved and reloaded: same plan = True"`, `"not a prescription"`).
- [ ] **Step 2: Run** `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/test_examples.py -k recovery -q --no-cov` — Expected: FAIL (file not found).
- [ ] **Step 3: Write the example** — 6 × 5 floodplain grid (river along column 0), cost `round(1 + 0.4c + 0.1((7r + 3c) mod 5), 2)`; wet meadow (habitat 1) existing `10 − 2c` where `c == 0` or (`c == 1` and r even), potential `8 − c` for `1 ≤ c ≤ 3`; alder carr (habitat 2) existing 6 where `c ≥ 4, r ∈ {0, 1}`, potential 5 where `c ≥ 3, r ≥ 2`; `p = round(0.9 − 0.1c, 2)`; `PassiveAction()`; protect 0.3, restore 0.2. `main()` prints the placeholders, exact plan table, heuristic (`NUMITNS 50000`, `NUMTEMP 500`, 5 runs seed 1), budget `0.7 ×` exact cost, sensitivity, save/reload check, caveat. Prototype output for reference: exact 17.70 (22 no action, 6 passive, 2 protect), heuristic 19.20 met, budget 12.39 → 12.20 with shortfalls.
- [ ] **Step 4: Write the docs.** Manual section headings: Inputs (field table), The built problem (zone table, feature credits, status rules, surcharge), Solving (code block with `build_recovery_problem` / `solve_recovery` (targets, budget, heuristic) / `evaluate_recovery` / `success_sensitivity`; spf rule; size guard), Results (table columns, never summed, sensitivity, the caveat quoted with "no time lag is modelled"), Saving and reloading (MarZone C++ caveat), Things to keep in mind (scalar-η artefact with Holl & Aide 2011; η > 1 "reported but contested", Crouzeilles et al. 2017 vs Reid et al. 2018; separate targets fix the mix — use budget mode; Bayraktarov et al. 2016 on cost spread), References — **only** these verified DOIs: Baker et al. 2025 10.1093/biosci/biaf030; Bayraktarov et al. 2016 10.1890/15-1077; Crouzeilles et al. 2017 10.1126/sciadv.1701345; Holl & Aide 2011 10.1016/j.foreco.2010.07.004; Reid et al. 2018 10.1126/sciadv.aas9143; Watts et al. 2009 10.1016/j.envsoft.2009.06.005. CHANGELOG `Added`: recovery package, `solve_recovery`, `evaluate_recovery`/`success_sensitivity`/caveat, `save_zone_project` + sidecar I/O, registry `zone_mip`, manual + example. (Edit Markdown with a file-based script or the editor — an unquoted shell heredoc expands the backticks.)
- [ ] **Step 5: Run** the example tests, `PYTHONPATH=src /opt/micromamba/envs/shiny/bin/python examples/nature_recovery.py`, ruff (`src tests examples`), mypy — Expected: PASS, output as in Step 3.
- [ ] **Step 6: Commit** `git add examples/nature_recovery.py examples/README.md tests/test_examples.py docs/USER_MANUAL.md README.md .github/copilot-instructions.md CHANGELOG.md && git commit -m "docs(recovery): manual section, example, README, copilot table, CHANGELOG"`

### Final gate (after Task 6)

- [ ] `~/.local/bin/ruff check src tests examples`; mypy as above.
- [ ] `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -p no:cacheprovider -m "not bench"` — all green (prototype HEAD a67f4b9: 4484 passed, 101 skipped, 38 deselected).
- [ ] Zone benches vs `main` (Task 1 Step 5); `python examples/validate_marxan_parity.py` still reports MIP 35.0.
- [ ] `gh run list` after pushing: local green ≠ CI green.

## Decisions taken in the prototype (review these)

- **Surcharge on non-restorable units** (Task 2): restore zones cost protect + restore where expected potential is 0. Needed because spec §3.2's "positive restore cost makes no action strictly better" ignores H2: with existing habitat credited under every action, a passive cost factor ≤ 1 ties or beats protection on units with no potential (brute force showed `(3, 0, …)` ties at cf 1.0).
- **Passive cases b1–b3 re-pinned** (Task 2): spec §3.5's 6.0 / 1.6 / 3.9 assume passive cost factor 0.2 (the probe); D2's default 1.0 gives 6.0 / 4.0 / 5.5. Both pinned.
- **ROWER anchor under rev 2** unchanged: restore target 0.45 × 13.6 = 6.12 expected (or 9.0 absolute) still forces PU 4; `[1, 0, 0, 2]` at 6.0, unique.
- **spf** = 10 × dearest cost per unit supplied; spec's "heuristics ≥ MIP on the anchors" holds only with the "or report the shortfall" clause — b1 at cost factor 0.2 has an infeasible penalised minimum for any factor < ~17.
- **Infeasible targets are build errors** (not warnings), also for budget mode; `restore_max` excludes status-3 units so `restore_prop ≤ 1` is always feasible; existing targets use all units.
- **Budget mode**: MIP only, equal target weights (spf 1), `min_shortfall` default.
- **Size guard** 50 000 unit × zone decisions, advisory.
- **Open, for the user:** PuLP's bundled CBC 2.10.3 can return suboptimal "Optimal" MIPs (preprocess bug, reproduced on a 7-unit recovery instance; affects every pymarxan MIP on pip installs). A follow-up should pass `preprocess off` or prefer a system CBC ≥ 2.10.4; out of scope here.

## Self-review

- **Spec coverage:** §2 → Task 1 (proposal/start, delta to/from 0 test, bench, picker, CHANGELOG Changed). §3 package layout → Tasks 2–5 (`inputs.py`, `builder.py`, `evaluate.py`, `io.py`, plus `solve.py`). §3.1 fields and every listed check → Task 2 (incl. horizon rule, infeasibility via `diagnose_zone_infeasibility`). §3.2 zones, MANDATORYALLOCATION 0, no puzone for restorability, status 1/2/3 → Task 2. §3.3 features, ids, names, credits, expected-unit overall targets → Task 2. §3.4 maturity / never summed → Tasks 2, 4. §3.5 modes, fallback, size guard, anchors, heuristic ≥ MIP or shortfall → Task 3. §3.6 passive defaults, provenance, rejection, stated consequence → Tasks 2, 6. §3.7 result, sensitivity, caveat → Task 4. §3.8 → Task 5. §3.9 → Task 3. §6 determinism hash → Task 2. §7 R6 → Task 6. §4 Shiny and §5 non-goals: not planned.
- **Type consistency:** `build_recovery_problem -> tuple[ZonalProblem, RecoveryMapping]` used by Tasks 3–6; `solve_recovery -> Solution` with the three metadata keys read by `evaluate_recovery`; `evaluate_recovery(solution, problem, mapping)` argument order everywhere; `RecoveryProject(inputs, problem, mapping)`.
- **Proportion:** the plan names files, signatures, test values and the two non-obvious algorithms (spf, surcharge); bodies live in the prototype commits.
