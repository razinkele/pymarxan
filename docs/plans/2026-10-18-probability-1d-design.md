# Marxan Probability 1D (planned v0.47.0) — design

**Date:** 2026-10-18, source-grounded 2026-10-29, revised after spec review round 1 (2026-10-29)
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
`marxan.c:1995-2025`). pymarxan reproduces the **score** (`ComputeProbability1D/2D`); its SA
delta equals the difference of its own full recompute.

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

When this implicit choice differs from what v0.46 would have chosen (1 / 0), `_probmode` emits
one `UserWarning` per problem naming both modes and how to keep the old one (`PROBMODE 1` in
`input.dat`). An explicit PROBMODE of 1, 2 or 3 while 1D is wanted warns that the 1D targets are
ignored; an explicit 1, 2 or 4 while 2D is wanted warns likewise for the 2D targets.

**A chance term without its data** (explicit 3 or 5 with no puvspr `prob`; explicit 4 or 5 with
no 1D source): `UserWarning`, and that term is off (penalty 0). Marxan cannot reach this state;
keeping it off avoids a constant ≈ PROBABILITYWEIGHTING penalty per feature that no selection can
remove, and keeps `test_probmode_3_with_no_probability_data_returns_zero` green.

**Zones** (`ZonalProblem`, `zones/readers.py:168` calls `load_project`): zone solvers ignore
chance terms; if `_chance_terms` is non-empty, warn once per solve.

### 3.2 Inputs and outputs

- **In-memory names (round 1, C2):** the 2D target stays in `features["ptarget"]` (no rename;
  every existing reader, `copy_with`, scenario overrides and the Shiny editor keep working). The
  1D target is new: `features["ptarget1d"]`, default −1.
- **`read_spec`:** whitelist `ptarget1d`, `ptarget2d`. `ptarget2d` is read into `ptarget`; a
  file with both `ptarget` and `ptarget2d` raises `ValueError` if they differ anywhere, else
  uses them. Missing columns default to −1.
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
- Both present and **any** PU's values differ (a PU absent from the table counts as 0): one
  `UserWarning`, use `pu.dat`.
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
  `V > 0` (a sum of positive floats is positive); the count makes the incremental path agree
  exactly, with no float threshold (round 1, C1: a relative threshold misclassifies legitimate
  small variances).
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

- **`ProblemCache`** keeps, per chance term, per-feature running `E`, `V`, `nonzero` for the
  current selection, and per-cell contribution arrays (CSR, aligned with `held_csr`). The flip
  delta touches only the flipped PU's features: new E/V/nonzero = old ± the cell's
  contribution, re-score those features through the core, return the change in penalty. The
  totals are recomputed from scratch at the start of each run (and whenever the selection is
  set wholesale), so float drift cannot accumulate across runs. This replaces the dense
  per-flip sum over the selection (`cache.py:835-836`; measured 2.9–4.8 ms vs 14 µs for mode 0,
  round 1, I4). Both terms can be active at once (mode 5); their penalties add.
- **Reported == tracked:** `Solution.objective` from the evaluator equals SA's tracked
  objective for modes 3, 4, 5 (test).
- **`MIPSolver`:** only `drop` works today (`mip_solver.py:479-492`; `piecewise` / `socp`
  raise `NotImplementedError`, pinned by `test_probmode3_solvers.py:119,125`). Modes 4 and 5 do
  the same: `drop` solves without the chance terms and its post-hoc report includes them
  (including the Z = 4 penalty); `piecewise` / `socp` raise.
- **Greedy** ignores chance terms (as for mode 3); **non-binary decisions** are rejected for
  modes 3–5.
- **Shiny probability panel** (round 1, C3; required, not optional): the mode radio gains 4 and
  5, is initialised from `_probmode(problem)`, and writes `PROBMODE` only when the radio itself
  changes (today moving the weight slider rewrites PROBMODE to the radio's default "1"). The
  panel shows `PROBZEROVAR`. The feature table edits `ptarget1d`.

### 3.6 Existing users (CHANGELOG `Changed`)

1. PROBMODE 3 zero-variance features now pay ≈ PROBABILITYWEIGHTING per feature (Marxan's
   rule); `PROBZEROVAR 1` restores the old result. The entry states the safe-vs-risky inversion (§1).
2. Projects with no `PROBMODE` whose data now selects 3, 4 or 5 (§3.1) change mode; they get a
   warning naming the old mode. This includes Marxan-native projects with pu.dat `prob` +
   `ptarget1d`, or puvspr `prob` + `ptarget2d`, which today warn about an unknown column and run
   with no chance term.
3. `write_spec` writes `ptarget2d` instead of `ptarget`.

Measured blast radius of the Z = 4 rule (round 1): the full suite fails only
`test_probability_dispatch.py::test_probmode_3_with_no_probability_data_returns_zero` (stays
green under §3.1's "term off without data") and
`test_zscore.py::TestZscorePenalty::test_sentinel_variance_means_deterministic_no_penalty`
(rewritten: zero variance → penalty `(ptarget − norm.sf(4))/ptarget` by default, 0 under
`PROBZEROVAR 1`). **No golden record uses PROBMODE 3** (stage 1/3/4 JSON checked); none moves.

## 4. Tests

- **Hand example vs a Marxan 3 transcription** (a test-local Python port of
  `ComputeProbability1D/2D` and `probZUT`): p ≠ 0.5 (catches the E = a·p polarity bug), a
  feature with E > target (negative Z), a zero-variance feature with ptarget > P, an
  unselected feature, two features with different ptarget1d; P compared at 1e-6.
- **Mode 4 == mode 3 with q = 1 − p:** shuffled, non-contiguous PU ids; a PU missing from
  prob.dat; `ptarget1d` ≠ `ptarget` on purpose, mapped explicitly in the mode-3 twin (so a
  mode-4 implementation reading the wrong column fails).
- **SA delta == full difference** for modes 3, 4, 5: p forced to 0 or 1 on some cells, flips that
  empty a feature's variance and refill it, ≥ 2 features with different ptarget, both
  `PROBZEROVAR` values, ≥ 5000 random flips, tolerance 1e-9.
- **Reported == tracked** objective after SA, modes 3–5.
- **Zero-variance rule** in both terms: all p = 0, nothing selected, q ∈ {0, 1}; default and
  opt-out.
- **Tiny p:** p = 1e-17 gives V > 0 (not Z = 4).
- **Mode selection:** each row of §3.1 incl. the v0.46-differs warning, explicit-mode-ignores
  warnings, term-off-without-data warnings, invalid PROBMODE in `validate()`, a problem built in
  code (no `load_project`) gets mode 4.
- **Source precedence:** pu.dat vs prob.dat agree (silent) / differ (warn, pu.dat used).
- **I/O:** `ptarget2d` → `ptarget`; `ptarget` + `ptarget2d` disagree → ValueError; legacy
  `ptarget` round trip writes `ptarget2d`; `ptarget1d` round trip; an in-memory `ptarget` set
  after loading is honoured; puvspr all-zero `prob` round trip; `PROBZEROVAR` round trip.
- **MIP:** `drop` with modes 4/5 reports the chance penalty; `piecewise` / `socp` raise.
- **Non-binary decisions** rejected for modes 4/5; **zones** warn.
- **Shiny:** moving the weight slider does not change PROBMODE; the radio initialises from a
  loaded mode-4 problem.
- **Bench** (`bench` marker): per-flip cost of modes 3/4 within a small factor of mode 0
  (comparative, not wall-clock), at 2000 × 20 and 10 000 × 50.

## 5. Left to the plan

Exact column names of the new 1D report fields; the CSR layout for per-cell contributions;
docs to update (manual — including dropping the zero-variance caveat on the "prob = 1 − p"
workaround at `USER_MANUAL.md:417-419`, README, copilot instructions — replacing "Probability
1D not yet implemented"); docstrings that cite Marxan v4 or "trivially met"
(`probability.py:8-16,46-48,123-125`, `utils.py:508-509`, `cache.py:619-622`, `base.py:30-33`).
