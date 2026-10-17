# PROBMODE 1/2 consistency across solvers (v0.45.0) — design

**Date:** 2026-10-17 · **Status:** approved in conversation; revised after spec-review rounds 1 (C1, I1–I7, M1–M13) 2 (N1–N8) and 3 (clean; R3-M1, R3-M2 applied) and a fresh-eyes final review (F1, F2, M1–M6 applied)
**Follows:** v0.44.0 (SA optimises connectivity). **Precedes:** a separate stage implementing
Marxan's Probability 1D (chance constraint) — out of scope here.

## 1. Background (verified 2026-10-17)

- **Marxan C++** (`Marxan-source-code/marxan`, `probability.cpp`, `score_change.cpp`,
  `input.cpp:148`) has no `PROBMODE` parameter. A `prob` column in `pu.dat` switches on
  Probability 1D: per feature, expected `Σ amount·(1−p)`, variance `Σ amount²·p·(1−p)`,
  Z-score against `ptarget1d`, penalty `PROBABILITYWEIGHTING · Σ shortfall`. A `prob` column
  in `puvspr` switches on Probability 2D (presence polarity), which is pymarxan's PROBMODE 3.
- **pymarxan's PROBMODE 1** (risk premium `γ·Σ p_i·c_i·x_i`) and **PROBMODE 2** (feature
  amounts scaled by `(1 − p_i)`, deterministic) are pymarxan extensions predating Phase 18.
  They read the per-PU probability table (`problem.probability`, `prob.dat`).
- **The solvers disagree** (probe on `tests/data/simple`, selection {2,4,6}, probabilities
  `linspace(0.1, 0.6)`, `PROBABILITYWEIGHTING` 2):

| | MIP | Greedy | SA / II (`ProblemCache`) | Reported (`build_solution`) |
|---|---|---|---|---|
| Default `PROBMODE` with a probability table | 1 | 1 | **0** | 1 |
| PROBMODE 1 premium | ✓ | ✓ (effective cost) | **✗** (44.0 vs reported 69.2) | ✓ |
| PROBMODE 2 `(1 − p)` amounts | ✓ | ✓ | **✗** | **✗** (targets judged on raw amounts) |

  With all probabilities 0.5 under PROBMODE 2 the MIP selects every unit to meet the
  scaled targets, and `build_solution` judges them on raw amounts.

## 2. Decisions (user-approved)

| Decision | Choice |
|---|---|
| Scope | Staged: consistency now (this spec); Marxan Probability 1D as its own later stage. |
| Default mode | With a probability table and no `PROBMODE`, every path uses 1 (the cache changes from 0). |
| PROBMODE 2 "held" | **Expected amount everywhere**: `_compute_achieved` scales by `(1 − p)`, so targets met, shortfall, penalty, `compute_held`, writers, analysis and Shiny all report `Σ amount·(1 − p)`, consistent with the MIP's target rows and `MultiObjectiveMIP`'s met counts. |
| Labelling | Docs call PROBMODE 1/2 pymarxan extensions and say Marxan Probability 1D is not yet implemented. |

## 3. Semantics (exactly the MIP's)

Let `p_i` be the probability of PU `i` from `problem.probability` (missing PU → 0; a PU listed
more than once → the last row wins, as every path's dict does), `c_i` its cost,
`γ = PROBABILITYWEIGHTING` (default 1.0), `mode = PROBMODE` (default 1 when a probability table
is present). Probability values are used as given, without clamping, identically in every path;
`validate()` already reports values outside [0, 1] and gains a duplicate-PU check.

- **mode 1:** objective term `γ · Σ_i x_i · p_i · c_i`. **Off when `γ <= 0`** or no table (the
  MIP and greedy rule; the evaluator changes from `== 0` to `<= 0`). The term is separate from
  cost: `Solution.cost`, `total_cost` and COSTTHRESH stay base cost. A zero or negative cost
  gives a zero or negative premium in every path (consistent today; pinned, not clamped).
- **mode 2:** every feature amount `a_ij` becomes `a_ij · (1 − p_i)` wherever held amounts are
  computed for targets, shortfall, penalty and reporting. No objective term.
- **mode 1 or 2 with no probability table:** a no-op in every path (as today).
- **mode 0 / 3:** unchanged.

## 4. Changes

**`solvers/cache.py` (`ProblemCache`)**
- Read `PROBMODE` with default 1 when `problem.probability` is present (0 otherwise); PROBMODE
  3 behaviour identical.
- mode 1: new defaulted field `prob_cost: (n_pu,) float64` = `γ·p_i·c_i` (zeros when off).
  `compute_full_objective` adds `Σ prob_cost[selected]`; `compute_delta_objective` adds
  `sign · prob_cost[idx]` behind a cheap flag check. `costs` is not modified.
- mode 2: new field `held_csr: csr_matrix | None = field(default=None, compare=False)`, built
  once as a **copy of `pu_feat_csr` with `.data` scaled by `np.repeat(1 − p, np.diff(indptr))`**
  — identical `indptr` / `indices` (a `p_i = 1` row keeps explicit zeros). Do not build it with
  `sp.diags(1 − p) @ pu_feat_csr`: the product drops explicit zeros and silently misaligns the
  indices. `compute_held`, `apply_flip_to_held` and the penalty delta read **both columns and
  amounts** from `self.held_csr` directly. `from_problem` always sets it — to the scaled copy
  when mode 2 is active with a probability table, otherwise to `pu_feat_csr` itself (no copy);
  a hand-built cache (default `None`) gets the same fallback once in `__post_init__` (via
  `object.__setattr__`). The matrix is resolved at build time, so no reader pays a per-flip
  check.
- `pu_feat_csr`, `pu_feat_matrix`, `feat_uses_pu` and `feat_baseline_penalty` stay **raw**, so
  `ClumpState` and `SepState` keep matching their reporting counterparts
  (`compute_clump_penalty_from_scratch` / `evaluate_solution_clumping`,
  `compute_sep_penalty_from_scratch`), which also stay raw. Clump- and separation-active
  features are outside `_det_spf`, so no scaled/raw mix enters one term. `SpaceState` and
  `pymarxan.constraints` do not read cache amounts and are unchanged. **Documented limitation:**
  under PROBMODE 2, clump occupancy, separation occurrence and space weights use raw amounts.

**`solvers/utils.py`**
- `_compute_achieved`: under mode 2 scale by `(1 − p)` (binary and fractional paths). Every
  consumer through `compute_held` / `check_targets` / shortfall / penalty inherits it
  (`build_solution`, `write_mvbest`, `export_summary_csv`, Shiny summary and target-met views).
- `_compute_probmode1_risk_premium`: off for `γ <= 0`.
- Docstrings: `compute_probability_penalty` (drop "handled upstream"), `compute_held`.

**`analysis/representation.py`:** the binary path uses `compute_held` (as the fractional path
does), so `represented_amount` is the expected amount under PROBMODE 2; the denominator
`total_amount` is the expected total `Σ_i a_ij·(1 − p_i)` over all PUs (selecting every PU =
100 %). **`analysis/gap_analysis.py`:** the protected amount is `compute_held` of the locked-in
mask, and its `total_amount` (the `percent_protected` denominator) is the same expected total
as representation's — one shared helper for both modules (its "agrees with build_solution"
comment then holds). The expected-total helper applies **only under PROBMODE 2 with a
probability table**; in every other mode both modules keep `feature_amounts()` exactly as today
(so problems whose `pu_vs_features` has rows for unknown PU ids report unchanged percentages).

**`models/problem.py` `validate()`:** report probability-table PUs listed more than once.

**Docs and UI text:** CHANGELOG `[Unreleased]` (`### Changed`: SA/II plans change on
probability problems, including projects whose `prob.dat` is picked up automatically with no
`PROBMODE` — now mode 1; PROBMODE 2 reported held/met/shortfall/representation/gap now use
expected amounts; `### Fixed`: the mismatches in §1 and negative `PROBABILITYWEIGHTING`);
`docs/USER_MANUAL.md` PROBMODE rows relabelled as pymarxan extensions with the Marxan 1D/2D
mapping, the PROBMODE 2 limitations (clump/separation/space raw; presolve P6, irreplaceability,
Ferrier and prop-target resolution use raw totals); Shiny `probability_config.py` Mode 2 label
"Expected (persistence-adjusted) amounts" and tooltip (pymarxan extension; targets judged on
expected amounts); docstrings of `MultiObjectiveMIP` (its PROBMODE 2 caveat), `ProblemCache`
(probmode comments) and `objectives/base.py` (`effective_amounts`); roadmap: Marxan Probability
1D stage, PROBMODE backlog line removed, new backlog lines for the §7 raw-amount paths.

## 4a. Implementation details (final review)

- **Flags:** `prob_cost: np.ndarray = field(default_factory=lambda: np.zeros(0))` and
  `prob_active: bool = False`; `prob_active` is True iff mode 1, a table and γ > 0. Both
  `compute_full_objective` and `compute_delta_objective` read `prob_cost` only when
  `prob_active`.
- **Typing:** `held_csr: csr_matrix = field(default=None, compare=False, repr=False)  # type:
  ignore[assignment]`; `__post_init__` guarantees it is never `None` after construction, so
  readers need no narrowing.
- **p vector:** `np.zeros(n_pu)` filled by a Python loop over the table rows in order; PU ids not
  in `pu_id_to_idx` are skipped; later rows overwrite earlier ones (last wins).
- **Expected-total helper:** `pymarxan.analysis._totals.expected_feature_totals(problem) ->
  dict[int, float]`. Under PROBMODE 2 with a table it returns `compute_held(problem,
  np.ones(n_pu, dtype=bool))` (rows whose PU is not in `planning_units` excluded; every feature
  id a key, 0.0 when absent); otherwise `{int(f): float(a) for f, a in
  problem.feature_amounts().items()}`. Representation and `gap_analysis` use it for their
  denominators (under mode 2, features with no rows get 0.0 instead of a `KeyError` in
  `to_dataframe`).
- **`validate()` message:** `f"probability lists planning unit IDs more than once:
  {sorted(dups)}"`. MIP presolve runs `validate()` (P0), so the "PU listed twice" fixture is used
  only in the delta / full tests, not MIP tests.
- **Golden fix shape:** `STAGE3_PROBMODE2_FIXED: dict[str, tuple[float, dict[str, bool]]]` maps
  case → (new objective, new targets_met). The test asserts `golden[case]["objective"]` and
  `["targets_met"]` still equal the v0.44 values (54.0 or 41.0 and the stored dict), asserts
  `got` has the new values, and runs `_assert_same` on `selected`, `cost` and
  `mip_objective_value` only.

## 5. Effects

- SA and iterative-improvement plans change on problems with a probability table in mode 1 or
  2, including tables with no `PROBMODE` (now mode 1) — e.g. any project whose input folder has
  a `prob.dat`, which `load_project` attaches automatically.
- PROBMODE 2 reports change to expected amounts: targets met, shortfall, penalty, held,
  mvbest, summary, representation, gap analysis, Shiny.
- Solvers that do not see probabilities but report through `build_solution` (the external
  Marxan binary, the Zonation solver) now have their PROBMODE 2 targets judged on expected
  amounts.
- The MIP and greedy are unchanged (models and plans).
- **Goldens:** the Stage 1 and Stage 4 goldens do not move. The Stage 3 golden's 8
  `probmode2|{max_features,max_weighted_features,min_shortfall,min_largest_shortfall}|{relative,absolute}`
  records pin `build_solution`'s `objective` and `targets_met`, which now use expected amounts
  (objective 54.0→59.1 and 41.0→55.4; min_shortfall / min_largest_shortfall `targets_met`
  {1:T,2:T,3:F}→{1:F,2:F,3:F}). They go in a `STAGE3_PROBMODE2_FIXED` dict in the test (the
  `STAGE3_FIXED` pattern): assert the golden still holds the v0.44 value, assert the new value;
  `selected`, `cost` and `mip_objective_value` still compare against the golden. The JSON is
  not regenerated.
- The 35.0 anchor holds (no probability table).

## 6. Testing

All SA tests in this section set `NUMITNS = 10_000` and `NUMTEMP = 100` in `parameters` and run
`SimulatedAnnealingSolver().solve(p, SolverConfig(num_solutions=10, seed=42))`; consistency
tests use `num_solutions=1, seed=42`. "Best of 10 reaches X" means
`min(s.objective for s in sols) == pytest.approx(X, abs=1e-9)`. Probed hit rates are 1–3 of 10,
deterministic for these seeds; do not change `NUMITNS` or the seeds without re-probing. "Equals"
between reported and tracked objectives means `pytest.approx(rel=1e-9, abs=1e-9)` (float drift
≈ 3e-12). Mark the best-of-10 tests `slow`.

- **SA consistency:** seeded SA's reported `Solution.objective` equals
  `metadata["history"]["best_objective"][-1]` for mode 0, 1, 2 and a table with no `PROBMODE`
  (fails today for mode 1 and the default case; mode 2 agrees today because both sides are raw).
- **Mode-2 oracle:** `ProblemCache.from_problem(p).compute_held(sel)` equals a hand-computed
  `Σ a·(1 − p)`; on a fixture where raw amounts meet a target but expected amounts don't
  (`simple` with p = 0.5 on PUs 4–6 and every `spf ≥ 10 × Σ cost`; at spf 1 the soft optimum
  46.5 misses targets), the feasible optimum is 50.0 at (1,1,1,0,0,0); the best of 10 seeded
  SA runs reaches it, meets every target under expected amounts, and `Solution.targets_met` is
  all True.
- **Delta exactness:** every PU, both directions, mode 1 and 2, delta == full difference.
- **Full = evaluator:** cache full objective == `build_solution(...).objective` on random
  selections (modes 1, 2, default; γ = −1).
- **Fixtures** for the last two include: a PU missing from the table, a PU listed twice, a PU
  with p = 1, a negative-cost PU, MISSLEVEL < 1 and a locked-in PU.
- **Evaluator PROBMODE 2** (`simple`, p = 0.5 on PUs 4–6, PROBMODE 2):
  - (i) `MIPSolver(mip_backend="cbc", objective="max_features")` with `COSTBUDGET` 33:
    `sum(targets_met.values()) == mip_objective_value` (1). Today this reports 2, judged on raw
    amounts: **red first**.
  - (ii) `binarize_solution(p, build_solution(p, np.array([0.75, 1, 0.5, 0, 0, 1.0]), 1.0), 0.75)`
    has `targets_met == {1: True, 2: False, 3: False}` (feature 2 holds 22 raw and 18.5 expected
    against a target of 20; today it reports True): **red first**.
  - (iii) Regression pins, green today and after: the `min_set` plan and the
    `decisions="proportion"` plan report every target met.
- **Cross-solver:** on a ≤ 8-PU fixture with every `spf ≥ 10 × Σ cost` (so the soft optimum meets
  every target — asserted), brute-force optimum of the cache objective == brute-force minimum of
  `build_solution(...).objective` over target-feasible plans == `MIPSolver(mip_backend="cbc")`
  `mip_objective_value`; the best of 10 seeded SA runs reaches it (single seeds do not
  reliably); modes 1 and 2.
- **Combinations (unchanged behaviour pinned):** PROBMODE 2 with clumping, and with separation
  (a fixture with `xloc` / `yloc`; no space targets, for which identity (b) would not hold):
  (a) `ClumpState` penalty after N random flips == `compute_clump_penalty_from_scratch(problem,
  sel)[1]`, `SepState` likewise with `compute_sep_penalty_from_scratch`; (b) SA tracked best ==
  `reported.objective − Σ_{j clump/sep-active} spf_j·shortfall_j + clump_penalty +
  sep_penalty`, shortfall on expected amounts. Both hold at mode 0 today.
- **Representation / gap:** expected amounts and denominator under PROBMODE 2; binary and
  proportion plans of the same units agree.
- **No table:** PROBMODE 1 / 2 with no table is a no-op in cache and evaluator (`prob_cost`
  all zeros, `held_csr is pu_feat_csr`).
- **`held_csr` structure:** same `indptr` / `indices` as `pu_feat_csr`, including a p = 1 row.
- **Bench** (`bench` marker, comparative, the v0.44.0 connectivity-bench pattern): per-flip
  `compute_delta_objective` time on the same problem with no table vs (a) a table with PROBMODE
  0, (b) mode 1 with γ = 0 — both asserted ≤ 1.05 × no-table (with the same timer-noise floor) —
  (c) mode 2, also asserted ≤ 1.05 × no-table (pre-scaled amounts, no extra per-flip work), and
  (d) mode 1 with γ > 0, print-only.
- `make check` green; parity harness unchanged; no golden JSON regenerated.

## 7. Out of scope

Marxan Probability 1D (next stage); reading Marxan's `pu.dat` `prob` column; PROBMODE 3;
zones; fractional SA (SA is binary-only). Raw-amount paths left as they are and documented as
PROBMODE 2 limitations (backlog): presolve P6 reachability, `analysis/irreplaceability.py`,
`analysis/ferrier_importance.py`, relative / prop target resolution against raw totals, and
clump / separation / space amounts (§4).
