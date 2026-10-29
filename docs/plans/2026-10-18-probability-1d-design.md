# Marxan Probability 1D (planned v0.47.0) — design

**Date:** 2026-10-18, source-grounded 2026-10-29, revised after spec review rounds 1–3 (2026-10-29)
· **Status:** design approved in conversation (approach A); written spec under review.
**Next steps:** spec review loop → prototype-first plan → plan review → SDD.
**Follows:** v0.45.0 (PROBMODE 1/2 consistency), whose review established that Marxan C++ has no
`PROBMODE`, that pymarxan's PROBMODE 1/2 are extensions, and that Marxan v4's `probZUT` has an
integer-division bug (`1 / 2`) that inverts its probability penalty.

## 1. Reference: Marxan 3

Source of truth: `Marxan-source-code/marxan`, branch **`marxan3`**, `probability.c`
(`ComputeProbability1D/2D`, `ChangeProbability1D/2D`, `probZUT`), `input.c` (`pu.dat` `prob`,
`spec.dat` `ptarget1d` / `ptarget2d`, default −1), `marxan.c` (score assembly). Not v4.

Per feature `j`, with per-unit probability of **loss** `p_i` from `pu.dat` `prob`, summed over
the **selected** PUs only:

- expected `E_j = Σ_i a_ij·(1 − p_i)`, variance `V_j = Σ_i a_ij²·p_i·(1 − p_i)`;
- `Z_j = (target_j − E_j)/√V_j` if `V_j > 0`, else **`Z_j = 4`** (zero variance);
- `P_j = probZUT(Z_j)` = upper normal tail = `norm.sf(Z_j)`;
- `penalty_j = [ptarget1d_j > P_j] · (ptarget1d_j − P_j)/ptarget1d_j` if `ptarget1d_j > 0`,
  else 0 (−1 = disabled);
- `probability1D = PROBABILITYWEIGHTING · Σ_j penalty_j`. No SPF, no MISSLEVEL.

2D is the same with per-cell **presence** probability `q_ij` (puvspr `prob`): `E = Σ a·q`,
`V = Σ a²·q(1 − q)`, target `ptarget2d`. 1D equals 2D with `q_ij = 1 − p_i` (checked for E and
V). Marxan's total is `cost + connection + penalty + probability1D + probability2D`
(`marxan.c` `computeReserveValue`): **both terms can be on at once.**

`target_j` in Z is the resolved deterministic target (pymarxan: `cache.py` `feat_targets`;
Marxan: `spec[i].target`). One existing difference feeds Z in both modes: for `prop`, pymarxan
resolves `max(target, prop·total)` while Marxan overrides the target with `prop·total`
(`marxan.c:895-899`). Not changed here; noted.

**Zero-variance quirk.** Zero variance gives `Z = 4`, `P ≈ 3.17e-5`, so the feature fails any
`ptarget > 3.17e-5` with penalty ≈ 1. This hits every feature with no uncertainty in the
selection: nothing selected, all `p = 0` (no risk), all `q ∈ {0, 1}`. Consequence (round 1, I7):
a plan holding a feature only in safe PUs scores worse than one using a slightly risky PU
(target 10, ptarget 0.9, weighting 10: safe 10.9996, risky 1.0). Ruling (user, 2026-10-29):
**Marxan's rule by default, with an opt-out** — see `PROBZEROVAR` in §3.

**Activation in Marxan 3:** 1D is on when `pu.dat` has a `prob` column (`input.c`
`readPlanningUnits`, `fProb1D = 1`); 2D when the puvspr header ends in `prob`; no condition on
the targets. Marxan 3 accepts spec columns `ptarget1d` and `ptarget2d` only; bare `ptarget` is
pymarxan's Phase 18 name for the 2D target.

**`probZUT`** is the Ibbetson (CACM Alg. 209) polynomial with `Z_MAX = 5`: exactly 0 / 1 for
|z| ≥ 5; for |z| < 5 within 1.05e-9 of `norm.sf` (measured); 2.87e-7 at the cut-off. pymarxan
keeps `norm.sf`. Parity tests against a transcription compare **P** with tolerance 1e-6 and
assert nothing exact at |z| ≥ 5; penalty differences are P differences ÷ ptarget.

**Not reproduced (Marxan 3 delta bugs):** the SA delta is not the difference of Marxan 3's own
score. (a) `ChangeProbability1D` and `ChangeProbability2D` read `spec[i].ptarget1d/2d` with `i`
= the richness-loop counter, not the species index `isp` (`probability.c:236-249, 329-342`), and
write `spec[isp].Zscore1D` / `probability1D` even for rejected moves. (b) `doChange` does not
update the 1D/2D expected and variance totals for clumping species (`target2 > 0`,
`marxan.c:1995-2025`). (c) Marxan 3's SA keeps running E/V sums and tests `rNewVariance > 0`
(`probability.c:207-212`), so float residue can misclassify zero variance there too. pymarxan
reproduces the **score** (`ComputeProbability1D/2D`); its SA delta equals the difference of its
own full recompute, and its zero-variance test is exact (§3.4) by design.

## 2. Decisions

| Decision | Choice |
|---|---|
| Activation / inputs (2026-10-18) | **Marxan-native.** pu.dat `prob`, spec `ptarget1d`/`ptarget2d`, new **PROBMODE 4 = Probability 1D**, auto-set when `input.dat` gives no PROBMODE (rules in §3). |
| Zero variance (2026-10-18, refined 2026-10-29) | Marxan 3's `Z = 4` rule in both 1D and 2D through one shared function; **opt-out** parameter `PROBZEROVAR` restores "zero variance = met". |
| Approach (2026-10-18) | **A — generalise PROBMODE 3's machinery**: one Z / P / penalty core; 1D and 2D differ only in where E and V come from and in the target column. |
| 1D probability source (2026-10-18) | `pu.dat` `prob`; if absent, `problem.probability` (prob.dat); always probability of loss. |
| 1D and 2D together (2026-10-29) | **Both, as Marxan:** new **PROBMODE 5 = 1D + 2D**, penalties added. (Replaces "warn and use 1D".) |
| SA speed (2026-10-29) | **Fix now:** incremental per-feature E / V for both chance terms, re-scoring only the flipped PU's features, as Marxan does. Speeds up mode 3 as well. |

## 3. Design

### 3.1 Modes and selection

`PROBMODE`: 0 none, 1 cost premium, 2 expected amounts (both pymarxan extensions, unchanged),
**3 = 2D** (chance term from puvspr `prob`), **4 = 1D**, **5 = 1D + 2D**. `validate()` reports
any other value (today a typo silently means "no probability").

One helper, `_chance_terms(problem) -> frozenset[{"1d", "2d"}]` in `solvers/utils.py`, replaces
every `PROBMODE == 3` branch. Sites (round 1, I6): `utils.py:460,716`;
`cache.py:331,623,699,834`; `mip_solver.py:76,479,781` (`_strategy_metadata`, also
`multiobjective.py:649`); `results/target_met.py:52`; `run_control/run_panel.py:250`;
`help/help_content.py:865`; `probability/probability_config.py`; `solvers/base.py:30-33,129`
(`supports_probmode3` keeps its name and means "supports chance terms", modes 3–5; docstring
says so). Zones, Zonation, analysis and presolve have no mode-3 branch today.

**The default lives in `_probmode()`** (not `load_project`), so problems built in code, made by
`copy_with`, or edited in Shiny get it too. An explicit `PROBMODE` always wins. With none:

1. 1D data = `planning_units` has a `prob` column, or `problem.probability` is set; 1D wanted =
   1D data and any `ptarget1d > 0`.
2. 2D wanted = `pu_vs_features` has a `prob` column and any `ptarget > 0` (the in-memory 2D
   target, §3.2).
3. Both → 5; 1D only → 4; 2D only → 3; neither → today's default (1 with `problem.probability`,
   else 0).

A missing `ptarget1d` or `ptarget` column counts as all −1 (features built in code or by
`copy_with` may lack them).

**No premium in modes 3–5 (round 2, I-3).** The PROBMODE 1 cost premium and PROBMODE 2
expected amounts apply only in modes 1 / 2 (`cache.py:969`, `heuristic.py:203`,
`utils.py:457-459`); modes 3–5 use raw costs and raw held amounts. So a project with `prob.dat`
that v0.46 ran as mode 1 and that now selects 4 (prob.dat + `ptarget1d > 0`) loses the premium
and gains the chance term. There is no mode with both; the warnings and the CHANGELOG say so.
`pu.dat` `prob` feeds the 1D term only, never modes 1 / 2 (which read `prob.dat`).

**Purity and diagnostics (round 2, I-4).** `_probmode`, `_chance_terms` and
`_loss_probability_vector` are pure and never warn: they run ~31 times per 10-run solve, per
`copy_with` in sweeps, and on every Shiny render, and `ConservationProblem` is unhashable, so
"once per problem" cannot be tracked on it. One function `_probability_diagnostics(problem) ->
list[str]` returns every message below. `_emit_probability_diagnostics(problem)` issues each as
one `UserWarning`. It is called by `load_project` and at the top of every top-level solve, and is
guarded by a `contextvars.ContextVar` so a solve nested inside another (RunModePipeline's inner
SA / heuristic solves, RUNMODE 4/5's two per run, II, replacement cost's inner solves) stays
silent. Call sites (round 3, I-4; `Solver.solve` is abstract, `base.py:113`, so each emits):
`SimulatedAnnealingSolver`, `IterativeImprovementSolver`, `HeuristicSolver`, `MIPSolver`,
`ZonationSolver`, `MarxanBinarySolver`, `RunModePipeline`, `ZoneSASolver`, `ZoneIISolver`,
`ZoneHeuristicSolver`, `ZoneMIPSolver`, and `MultiObjectiveMIP.solve` (`multiobjective.py:557`,
not a `Solver`; it builds through `mip._build_base` and never reaches `MIPSolver.solve`).
Sweeps (`calibrate_blm`, `calibrate_spf`, `run_sweep`, `run_sensitivity`,
`run_sweep_parallel`) and `compute_replacement_cost` / `generate_portfolio_cuts` call a
top-level solve per point, so they emit once per point. `load_project` then `solve` emitting
the same message twice is intended (load time and solve time are separate moments for a user).
Counts are only observable with warnings recorded or the filter at "always": Python's default
filter shows an identical message once per call site.

**`validate()` stays errors-only (round 3, I-3).** It gains real errors only: an invalid
PROBMODE, and `prob` values that are non-finite or outside [0, 1]. Advisory diagnostics never go
into it: `presolve.py:215-219` disables every presolve check on any `validate()` entry, and
`presolve_check="strict"` then raises from `MIPSolver` / `ZoneMIPSolver`. The Shiny upload
(`upload.py:77,104`) shows `_probability_diagnostics` next to `validate()`.

The "differs from v0.46" message repeats on every load and solve until the project sets
`PROBMODE` explicitly (`_probmode` is pure and `save_project` does not write a resolved mode);
that is intended, and the message names the remedy.

Messages:
- the implicit mode differs from v0.46's (1 / 0): name both modes, say modes 3–5 carry no
  cost premium, and say how to keep the old one (`PROBMODE 1` in `input.dat`, or
  `parameters["PROBMODE"] = 1` in code);
- an explicit PROBMODE of 1, 2 or 3 while 1D is wanted: the 1D targets are ignored; an explicit
  1, 2 or 4 while 2D is wanted: likewise for the 2D targets;
- an explicit PROBMODE 1 or 2 with `pu.dat` `prob` but no `prob.dat`: `pu.dat` `prob` is used by
  Probability 1D only;
- a chance term without its data (below);
- 1D sources disagree (§3.3);
- zone solvers ignore chance terms (below).

**A chance term without its data** (explicit 3 or 5 with no puvspr `prob`; explicit 4 or 5 with
no 1D source): a diagnostic, and that term is off (penalty 0). Marxan cannot reach this state;
keeping it off avoids a constant ≈ PROBABILITYWEIGHTING penalty per feature that no selection can
remove, and keeps `test_probmode_3_with_no_probability_data_returns_zero` green.

**Zones** (`ZonalProblem`, `zones/readers.py:168` calls `load_project`): zone solvers ignore
chance terms; if `_chance_terms` is non-empty, the zone solver's `solve` emits the diagnostic.

### 3.2 Inputs and outputs

- **In-memory names (round 1, C2):** the 2D target stays in `features["ptarget"]` (no rename;
  every existing reader, `copy_with`, scenario overrides and the Shiny editor keep working). The
  1D target is new: `features["ptarget1d"]`, default −1.
- **`read_spec`:** whitelist `ptarget1d`, `ptarget2d`. `ptarget2d` is read into `ptarget` and
  the `ptarget2d` column is then **dropped** (round 2, M-2: no stale twin in `features`,
  `copy_with` or the editor); a file with both `ptarget` and `ptarget2d` raises `ValueError` if
  they differ anywhere. Missing columns default to −1.
- **`write_spec`:** writes `ptarget2d` (from `ptarget`) and `ptarget1d`, each dropped when all −1;
  never writes `ptarget`. A legacy project is re-written in Marxan-native form.
- **`read_pu` / `write_pu`** already keep and round-trip every column (`writers.py:27`); nothing
  to add for `prob`.
- **Fix while here (round 1, M3):** `write_puvspr` drops an all-zero `prob` column
  (`writers.py:76-77`) while the reader defaults a missing one to 1.0 (`cache.py:333`), so
  presence 0 round-trips as presence 1. Drop the column only when it is all 1.
- **`_OVERRIDABLE_FIELDS`** (`problem.py:405`), `_KNOWN_SPEC_COLS` and the Shiny feature table
  (`feature_table.py`) gain `ptarget1d`.

### 3.3 The 1D probability source

One helper `_loss_probability_vector(problem, pu_id_to_idx, n_pu)`:

- `pu.dat` `prob` if the column exists; else `problem.probability` via the existing
  `_probability_vector` (a PU missing from the table has p = 0).
- Both present and **any** PU's values differ (a PU absent from the table counts as 0): a
  diagnostic (§3.1), use `pu.dat`.
- Non-finite values raise `ValueError` naming the PU ids (as `_probability_map` does); values
  outside [0, 1] are reported by `validate()`.
- `validate()` also checks `planning_units["prob"]` and `pu_vs_features["prob"]` (finite, in
  [0, 1]) when present.

E and V for 1D are built **directly from p** with Marxan's expressions, `a(1 − p)` and
`a²·p(1 − p)`, not by forming `q = 1 − p` first (round 1, I9: for p < 1.1e-16, `1 − p` rounds
to 1 and the variance would vanish).

### 3.4 Chance-constraint core (`solvers/probability.py`)

One function scores a feature from `(E, V, nonzero, target, ptarget, zero_var_rule)`:

- **zero variance** = `nonzero == 0`, where `nonzero` is the number of selected cells whose
  variance term, as computed in float, is `> 0`. For a full recompute this equals Marxan's
  `V > 0` (a sum of positive floats is positive); the count makes the incremental path agree on
  the **classification** exactly, with no float threshold (round 1, C1: a relative threshold
  misclassifies legitimate small variances). The running **values** are not exact (round 2,
  I-1): see the resync rules in §3.5.
- zero variance under `PROBZEROVAR = 0` (default, Marxan): `Z = 4`, then the normal path;
  under `PROBZEROVAR = 1`: penalty 0 and shortfall 0 ("met").
- otherwise `Z = (target − E)/√V`, `P = norm.sf(Z)`, penalty and shortfall as in §1.

`compute_zscore_per_feature`, `compute_zscore_penalty` and `evaluate_solution_chance` are
rebuilt on it. **All three current zero-variance skips go** (round 1, I3):
`probability.py:126` (`compute_zscore_penalty`), `probability.py:205` (zeroes
`prob_shortfalls`, which feeds mvbest `Prob_Gap` and the target-met table) and `cache.py:632-634`
(`var <= 0: continue`). `evaluate_solution_chance` reports per term (`1d`, `2d`): E, V, Z, P,
shortfall, and the penalty total; the mvbest / target-met columns gain 1D equivalents
(`Prob1D_Gap` etc.), named in the plan.

`PROBZEROVAR` is a new `input.dat` parameter (0 default, 1 opt-out), read by `load_project`,
written by `save_project`, documented in the manual, and shown in the Shiny probability panel.

### 3.5 Solvers

- **`ProblemCache`** stays stateless with respect to the selection (today
  `compute_delta_objective(idx, selected, held, total_cost, blm)`, `cache.py:710`, takes the
  caller's state). It gains, per chance term, immutable per-cell contribution arrays (E and V
  terms and the per-cell "variance term > 0" flag, CSR-aligned with `held_csr`), and drops its
  internal dense mode-3 delta term (`cache.py:834-850`). Its only callers are SA
  (`simulated_annealing.py:206,273`) and II (`iterative_improvement.py:213,295`).
- **`ChanceState` sidecar** (round 2, I-2), in the `ClumpState` / `SepState` pattern
  (`from_selection`, `delta_penalty`, `apply_flip`, `penalty_total`): per chance term,
  per-feature running `E`,
  `V`, `nonzero`. `delta_penalty(idx)` re-scores only the flipped PU's features and **never
  mutates** state (the analogue of Marxan bug (a)); `apply_flip(idx)` commits. Built
  `from_selection` at SA run start (`simulated_annealing.py:149`) and at the start of each II
  pass (`iterative_improvement.py:188` removal, `:272` addition; the two-step pass alternates
  them, each from a fresh `solution.selected`). The SA temperature-sampling loop
  (`simulated_annealing.py:203-221`) and rejected moves call `delta_penalty` only.
  `_swap_pass_loop` (`iterative_improvement.py:337-362`) flips two PUs in place and calls
  `compute_full_objective`, which keeps a from-scratch chance term; it also serves the initial
  objective. Both terms can be active at once (mode 5); their penalties add.
- **Error-bounded recompute (rounds 2–3).** Running sums lose precision (round 2: V relative
  error 5.7e-4 at amounts 1–1e6; with p = 1e-17 cells beside a = 1e4, p = 0.5 cells, 171 states
  had `nonzero > 0` but running `V ≤ 0`, so `√V` was NaN and the penalty silently 0). Drift
  builds up over many moderate removals and from the initial sum, not only from one large
  cancellation (round 3: a largest-first removal chain at amounts spanning 1e6 gave |delta − full
  diff| 6.8e-6, at 1e8 5.3e-2, and a "one removal cancels > 1 − 1e-8 of V" rule never fired).
  So each feature carries float error bounds since its last recompute:
  `BV += eps·(|V_old| + |ΔV|)`, `BE += eps·(|E_old| + |ΔE|)` per update (eps = machine epsilon).
  The feature is **recomputed from its cells over the current selection** (CSC column of the
  contribution arrays) when `V ≤ 0` while `nonzero > 0`, or `BV > τ·V`, or `BE > τ·√V`
  (τ = 1e-12); after a recompute both bounds reset to `4·eps·|value|`; when `nonzero` reaches 0,
  E and V snap to the exact sums of the remaining zero-variance cells (V = 0). The E bound
  matters because Z amplifies E error by 1/√V. Measured (round 3): every adversarial case
  ≤ 7.4e-14 for τ from 1e-12 to 1e-9, the 1e8 chain included; at 2000 × 20 and 10 000 × 50,
  0–101 recomputes in 4000–6000 flips. No periodic full resync is needed (its jumps were
  ≤ 2e-15), so SA's tracked objective needs no correction term. The rules run inside
  `delta_penalty` on a scratch copy of the touched features, so the no-mutation rule holds.
- **Commit == preview.** `apply_flip(idx, adding)` commits the cached result of the last
  `delta_penalty` when `(idx, adding)` matches, and otherwise computes through the same code
  path; the committed state equals the previewed state bit for bit. (Recomputing at commit
  measured 5.6–6.3×, over the bench bound.)
- **Cost.** Measured on the real layout with the rules above and cached commits (round 3):
  4.9–5.1× mode 0 per flip, against ~300× for today's mode 3. "Per flip" means delta, plus
  commit on an accepted flip, for mode 0 and the chance modes alike.
- **Full objective (round 3, I-5).** `compute_full_objective` no longer includes the chance
  term either, so the cache's own delta == full contract holds for every mode. SA adds
  `ChanceState.penalty_total()` to its initial objective, as it does for `SepState`
  (`simulated_annealing.py:151,188-190`); the swap pass adds a from-scratch chance term; the
  evaluator / `build_solution` keep the full chance term. Counting it in both places is the trap
  this avoids.
- **Reported == tracked:** `Solution.objective` from the evaluator equals SA's and II's tracked
  objective for modes 3, 4, 5 (test).
- **`MIPSolver`:** only `drop` works today (`mip_solver.py:479-492`; `piecewise` / `socp`
  raise `NotImplementedError`, pinned by `test_probmode3_solvers.py:119,125`). Modes 4 and 5 do
  the same: `drop` solves without the chance terms and its post-hoc report includes them
  (including the Z = 4 penalty); `piecewise` / `socp` raise.
- **Greedy** and **Zonation** ignore chance terms during optimisation (as for mode 3) and
  report them afterwards through `build_solution`; each emits a diagnostic saying so, like the
  zone solvers. **Non-binary decisions** are rejected for modes 3–5.
- **`MarxanBinarySolver`** writes the project with `save_project` (`marxan_binary.py:54`), so the
  spec file now carries `ptarget1d` / `ptarget2d`, and pu.dat `prob` already round-trips. A
  Marxan 3 binary then switches its own 1D / 2D terms on from the files (§1), whatever
  pymarxan's PROBMODE says; its diagnostic says so. (Marxan v4's handling not verified.)
- **Shiny probability panel** (round 1, C3; round 2, I-5; required, not optional): the mode
  radio offers 0 ("none"), 1–5 and is initialised from `_probmode(problem)` on load. Shiny cannot
  tell a user's radio change from `ui.update_radio_buttons`, so the mode effect writes
  `PROBMODE` only if `int(input.prob_mode()) != _probmode(problem)`; the weight slider has its own
  effect that writes only `PROBABILITYWEIGHTING` (today `_update_params`,
  `probability_config.py:127-134`, writes both on either input, so moving the slider rewrites
  PROBMODE to the radio's default "1"). The panel shows `PROBZEROVAR`. The feature table edits
  `ptarget1d`.

### 3.6 Existing users (CHANGELOG `Changed`)

1. PROBMODE 3 zero-variance features now pay ≈ PROBABILITYWEIGHTING per feature (Marxan's
   rule); `PROBZEROVAR 1` restores the old result. The entry states the safe-vs-risky inversion (§1).
2. Projects with no `PROBMODE` whose data now selects 3, 4 or 5 (§3.1) change mode; they get a
   warning naming the old mode. This includes Marxan-native projects with pu.dat `prob` +
   `ptarget1d`, or puvspr `prob` + `ptarget2d`, which today warn about an unknown column and run
   with no chance term. A project with `prob.dat` that ran as mode 1 and now selects 3–5 **loses
   the PROBMODE 1 cost premium** (§3.1).
3. `write_spec` writes `ptarget2d` instead of `ptarget`.
4. `write_puvspr` keeps an all-zero `prob` column (it used to drop it, and reloading turned
   presence 0 into presence 1); listed under `Fixed`.

Measured blast radius (rounds 1–2), each test rewritten with a written reason:
- Z = 4 rule: `test_probability_dispatch.py::test_probmode_3_with_no_probability_data_returns_zero`
  (stays green under §3.1's "term off without data") and
  `test_zscore.py::TestZscorePenalty::test_sentinel_variance_means_deterministic_no_penalty`
  (rewritten: zero variance → penalty `(ptarget − norm.sf(4))/ptarget` by default, 0 under
  `PROBZEROVAR 1`).
- §3.2 I/O: `test_io_probability.py::test_puvspr_no_prob_when_all_zero` (asserts the bug being
  fixed) and `test_writers.py::TestWriteSpec::test_roundtrip` (`ptarget1d` joins its drop list).
- §3.5 cache change: `test_cache_probmode3.py::TestProbmode3Delta::test_delta_matches_full_under_probmode3`
  moves to `ChanceState` (delta == difference of `penalty_total`), and
  `tests/benchmarks/bench_prob.py::test_probmode3_delta_under_2500us_at_500pu` is re-pointed at
  `ChanceState` (it would otherwise pass while measuring nothing).
- §3.1 implicit rule: no existing test reaches it (patched into every `_probmode` site: 3778
  passed), so every mode-selection test is new.
- **No golden record uses PROBMODE 3** (stage 1/3/4 JSON checked); none moves.

## 4. Tests

- **Hand example vs a Marxan 3 transcription** (a test-local Python port of
  `ComputeProbability1D/2D` and `probZUT`): p ≠ 0.5 (catches the E = a·p polarity bug), a
  feature with E > target (negative Z), a zero-variance feature with ptarget > P, an
  unselected feature, two features with different ptarget1d; P compared at 1e-6.
- **Mode 4 == mode 3 with q = 1 − p:** shuffled, non-contiguous PU ids; a PU missing from
  prob.dat; `ptarget1d` ≠ `ptarget` on purpose, mapped explicitly in the mode-3 twin (so a
  mode-4 implementation reading the wrong column fails); all p > 1e-16 (the twin forms
  q = 1 − p) and compared at 1e-12 relative.
- **Mode 5 == mode 3 + mode 4** on the same data, with `ptarget1d` ≠ `ptarget` and different
  1D / 2D distributions (catches pooled E / V, or one target column used for both).
- **SA delta == full difference** for modes 3, 4, 5: p forced to 0 or 1 on some cells, flips that
  empty a feature's variance and refill it, ≥ 2 features with different ptarget, both
  `PROBZEROVAR` values, ≥ 5000 random flips mixing accepted and rejected moves and the
  temperature-sampling path (catches state committed inside `delta_penalty`), amounts spread
  ≥ 1e6 with removal of the dominant-variance cell, and p = 1e-17 cells sharing a feature with
  large-variance cells (the resync rules); tolerance 1e-9.
- **Reported == tracked** objective after SA and after II (every ITIMPTYPE, incl. the swap pass),
  modes 3–5; II's removal pass keeps a PU the chance term needs.
- **Mode 5 with `prob.dat`:** held amounts and costs stay raw (catches `_probmode2_active`
  generalised to "mode ≥ 2", or the premium leaking into modes 3–5).
- **Z uses the raw target:** MISSLEVEL ≠ 1 does not change Z (Marxan's `spec.target`).
- **Diagnostics counts** (warnings recorded, filter "always"; problem built in code with
  implicit mode 4): a 10-run SA solve and a 10-run `RunModePipeline` solve (RUNMODE 1 and 4)
  each emit each message once; a 5-point `calibrate_blm` five times; `MultiObjectiveMIP.solve`
  once; `_probmode` and `validate()` never warn, and `validate()` returns no advisory entries
  (so `presolve_check` still runs).
- **Zero-variance rule** in both terms: all p = 0, nothing selected, q ∈ {0, 1}; default and
  opt-out.
- **Tiny p:** p = 1e-17 gives V > 0 (not Z = 4).
- **Mode selection:** each row of §3.1 incl. the v0.46-differs warning, explicit-mode-ignores
  warnings, term-off-without-data warnings, the pu.dat-prob-with-mode-1/2 warning, invalid
  PROBMODE in `validate()`, a problem built in code (no `load_project`) gets mode 4, features
  without `ptarget1d` / `ptarget` columns.
- **Source precedence:** pu.dat vs prob.dat agree (silent) / differ (warn, pu.dat used).
- **I/O:** `ptarget2d` → `ptarget`; `ptarget` + `ptarget2d` disagree → ValueError; legacy
  `ptarget` round trip writes `ptarget2d`; `ptarget1d` round trip; an in-memory `ptarget` set
  after loading is honoured; puvspr all-zero `prob` round trip; `PROBZEROVAR` round trip.
- **MIP:** `drop` with modes 4/5 reports the chance penalty; `piecewise` / `socp` raise.
- **Non-binary decisions** rejected for modes 4/5; **zones** warn.
- **Shiny:** load a problem whose mode is implicit 4, move the weight slider:
  `"PROBMODE" not in p.parameters`; the radio initialises to "4"; a mode-0 problem initialises
  to "0".
- **Bench** (`bench` marker): per-flip cost (delta, plus commit on an accepted flip) of modes
  3/4/5 ≤ 6× mode 0 (comparative, not wall-clock), at 2000 × 20 and 10 000 × 50.
- **Commit == preview:** after `delta_penalty(i)` then `apply_flip(i, adding)`, the state equals
  a fresh `from_selection` of the new selection to 1e-12, and equals the previewed scratch state
  exactly.

## 5. Left to the plan

Exact column names of the new 1D report fields; the CSR layout for per-cell contributions;
docs to update (manual — including dropping the zero-variance caveat on the "prob = 1 − p"
workaround at `USER_MANUAL.md:417-419`, README, copilot instructions — replacing "Probability
1D not yet implemented"); docstrings that cite Marxan v4 or "trivially met"
(`probability.py:8-16,46-48,123-125`, `utils.py:508-509`, `cache.py:619-622`, `base.py:30-33`).
