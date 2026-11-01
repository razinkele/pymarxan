# Chance-term core (v0.47.0) — plan review synthesis

**Date:** 2026-10-31 · **Reviewed:** plan `2026-10-29-chance-core-v047-plan.md` (`231f81a`), replayed task by task on a scratch branch whose trees equal this branch's Task 1–7 commits (`4018440` … `fbce740`, base `135d025`; commit ids below are this branch's) · **Outcome:** the plan's "Review outcome" addendum · **Spec:** `2026-10-18-probability-1d-design.md` · **Reference:** Marxan 3 (`marxan3` branch) `probability.c`, `input.c`, `marxan.c`; Marxan v4 `utils.hpp` · Raw findings, votes and probe scripts were kept outside the repository.

## Lenses

Twelve reviewers worked on the replayed tree:
- **Architect:** the post-plan tree against spec §1, §3.1–3.6, §4 and §6, with per-flip timings from the bench harness.
- **Grounding, one reviewer per task (T1–T7):** claims about pre-existing code; RED/GREEN re-measured on the previous tree and on the task's own tree; mutation runs; edge cases; effects outside release A.
- **Scientific accuracy:** the score, the zero-variance rule, `probZUT` and the references, checked against Marxan 3 and Marxan v4.
- **Independent re-design:** a design written from spec §1–3.6 / §6 A and the `135d025` code before reading the plan, then diffed against it.
- **Numerics:** `ChanceState` error bounds, triggers, snap and nonzero count, against `math.fsum` plus an exact count.
- **Portability:** Windows 11, Python 3.13.12, numpy 2.4.3 on MKL 2025.3 (10 threads), pandas 2.3.3, scipy 1.17.1, PuLP 3.3.2 + CBC, spawn workers.

The 50 raw findings merge into 37 clusters. The 22 clusters at MEDIUM or above were verified adversarially, with 2–3 independent votes each (refute, reproduce, scope). All 22 were confirmed and 6 were re-rated: 3 HIGH→MEDIUM and 3 MEDIUM→LOW. The 15 LOW clusters were not verified. **Severity rule:** verified severity = the majority of the votes; a 1–1 split keeps the raised severity and does not count as a re-rating. The table shows raised→verified, with the votes as H/M/L.

## Verdict

- **The replay is byte-exact.** For every Modify block, the plan's `index a..b` line (the prototype's pre- and post-image blob hashes) matches the replay commit: 32/32 modified files. The 6 new files hash-match the plan text: 6/6. The prototype commits (`5d1ac60` … `fe6bd76`) are not in the review clone, so the comparison is against the hashes the plan records.
- **RED/GREEN hold for all seven tasks.** Measured by the task lenses, previous tree → own tree:
  - T1: 2 failed / 6 passed → 8 passed.
  - T2: 2 failed / 5 passed → 7 passed (spawn).
  - T3: 1 collection error → 13 passed.
  - T4: 1 collection error → 38 passed.
  - T5: 1 collection error → 34 passed (30 in `test_chance_state.py`, 4 in `test_fractional_readers.py`).
  - T6: 4 failed / 91 passed → 95 passed.
  - T7: 18 failed / 47 passed → 65 passed.
- **Nothing outside the chance term moves.**
  - Modes 0/1/2 are bit-identical on 14 problems at `135d025`, `88a2861` and `fbce740` (SA, and II at ITIMPTYPE 1/2/3). This covers selections, full-precision objectives, temperatures and history arrays.
  - The parity example gives MIP 35.0 / SA 43.0 / greedy 45.0, with round trip True (at `4018440`).
  - The plan regenerates no golden.
- **Windows / py3.13.**
  - On `fbce740`, the 198 tests in the 11 touched test files pass, and so do the 6 benches. All 137 tests in the seven chance test files pass (re-design lens).
  - The plan's Run/Gate commands use the repo's Linux paths and were not run as written.
  - No full-suite run was part of this review.
- **Marxan 3 parity.**
  - These match Marxan 3: the E/V sums, the Z = 4 rule, the Heaviside step, the ptarget normalisation, the PROBABILITYWEIGHTING-only total, the raw target (no MISSLEVEL), and the operand order `((a·a)·q)·(1 − q)`.
  - `ndtr` is within 1.05e-9 of `probZUT` for |z| < 5.
  - There is one parity defect, duplicate (pu, species) rows (#1). It entered through the Task 5 ruling (plan line 54) and a test pins it.
  - Out-of-range q diverges from Marxan, but only on invalid input (#5).
- **The incremental core is correct for valid, finite, duplicate-free input.** *Erratum (second review, see the plan's "Review outcome" addendum): not from a start selection that holds no cell of the term. There `from_selection`'s E and V were int64 (`np.bincount` over an empty index), so every flip truncated them: SA at `PROP 0`, II's addition pass from an empty start. This review missed it; `a28f33e` fixes it.*
  - The chance term is counted once on every path, and commit == preview.
  - The triggers follow spec §3.5, the snap is right, and the nonzero count is exact over 18 240 random flips.
  - Task 5 regresses on non-finite input (#2).
  - Tests #3, #4 and #17 do not pin what they claim. Deleting the BE or BV trigger (#7), or either of two wiring sites (#15, #16), fails no test; the wiring-site cases contradict plan line 57's mutation claim.
  - The swap pass is 4–6× faster, with identical objectives.
- **Counts:** no CRITICAL; 4 HIGH (1 parity-critical); 15 MEDIUM; 3 verified LOW; 15 unverified LOW nits. None of the 37 items requires touching Tasks 1–7 (the int64 E/V defect in the erratum above, found later, is in Task 5's code).

## Confirmed findings

Rows are ordered by verified severity. Task 0 means pre-existing or cross-task. Dispositions refer to the follow-up tasks in the Disposition section.

| # | Sev (votes) | Parity | Task | Lens | Finding | Location | Disposition |
|---|---|---|---|---|---|---|---|
| 1 | HIGH (H H H) | **yes** | 5 | architect, T4–T7, science, re-design, numerics | Duplicate (pu, species) puvspr rows: `ChanceCells` puts the last row's `prob` on the summed amount, while the evaluator and Marxan 3 sum E and V row by row. So SA/II optimise a different chance term from the one they report, and can return a plan 51 % worse than the reported optimum. The ruling at plan line 54 and a test pin this behaviour, and every equality fixture deduplicates. | `solvers/cache.py:961-969`; `solvers/probability.py:306-320`; `test_chance_state.py:189-200` | Task 8 |
| 2 | HIGH (H H H) | — | 5 | T5, numerics | A non-finite cell term in an *unselected* PU (blank puvspr `prob`) poisons the recompute matvec (NaN·0 = NaN). No trigger fires on NaN, so the feature's chance penalty stays 0: SA reported − tracked is +26.5 to +36.6, against 0.0 at `135d025`. This is a Task 5 regression; the dense delta read selected rows only. | `chance_state.py:236-238`; `cache.py:957-971` | Task 10 |
| 3 | HIGH (H H M) | — | 5 | T5 | `test_stale_preview_is_never_committed` is vacuous. PUs 1 and 3 of `mixed(4)` share features only through q = 0 cells, so deleting `self._preview = None` still passes it; only private `_preview is None` asserts in another test catch that mutant. | `test_chance_state.py:347-363` | Task 11 |
| 4 | HIGH (M H H) | — | 6 | T6 | `test_removal_pass_keeps_a_pu_the_chance_term_needs` (named in spec §4) passes with the removal pass's chance wiring deleted. The all-selected start is already optimal, and both outer loops return it when a pass makes things worse. | `test_sa_chance.py:116-126`; `iterative_improvement.py:183, 273` | Task 11 |
| 5 | HIGH→MEDIUM (M M M) | invalid input | 4 | T4, science | An out-of-range or NaN puvspr `prob` beside a valid uncertain cell gives nonzero ≥ 1 with V ≤ 0 or NaN, hence Z = NaN and a silent penalty of 0. Marxan 3 tests V > 0 and charges ≈W (q = 1.5: 0.0 vs 9.9996). Adding one valid PU then drops a feature's penalty from ≈W to 0. The ruling at plan line 52 holds only for q ∈ [0, 1]. | `probability.py:154`; `chance_state.py:203-225` | Task 10 |
| 6 | HIGH→MEDIUM (M M M) | — | 2 | T2 | `run_sweep_parallel` still treats every `ValueError` as an infeasible point. The workers now get the whole problem, so `_probability_map`'s non-finite error and `PUCoordinatesUnavailableError` become silently dropped points, while `run_sweep` raises. Task 2's claim that the two "agree" is false for these inputs. | `calibration/parallel.py:29-30, 61-66` | Task 9 |
| 7 | HIGH→MEDIUM (M H M) | — | 5 | T5 | Neither the BE nor the BV recompute trigger is pinned: deleting either leaves every chance test green (124 tests in six files). Targeted layouts then miss the 1e-9 contract by 24–30×. The "E is the exact sum" half of the snap test is also unpinned. | `chance_state.py:204-208`; `test_chance_state.py:266-301, 411-427` | Task 11 |
| 8 | MEDIUM (L M) | — | 1 | T1 | No test writes a frame without `prob`. A writer that adds or reindexes the column passes all 74 related tests, and it switches the 2D term on for a legacy PROBMODE 3 project after save/load (objective 3.0 → 22.9993). | `test_io_probability.py:80-88`; `io/writers.py:78` | Task 11 |
| 9 | MEDIUM (M M) | — | 0 | T2 | `add_connectivity_features` keeps the five-field rebuild that Task 2 removes from the sweep. It drops `probability`, `connectivity`, `grid` and the subclass (seeded objective 34.4 → 8.0), and its "copy_with() is not appropriate" comment is wrong. Pre-existing, and outside the plan. | `connectivity/features.py:62-71` | Task 9 |
| 10 | MEDIUM (L M) | — | 3 | T3 | For mode 3 without a puvspr `prob` column, Task 3 also drops mvbest's `Prob_Gap` and the target-met probability columns, and the MIP notice still promises a post-hoc gap. Plan line 472, the commit message and CHANGELOG:40-41 name only the `Solution` fields. The behaviour itself follows spec §3.4. | `io/writers.py:238-239`; `run_panel.py:250, 259-264` | Task 13; notice defer-v048 |
| 11 | MEDIUM (M M) | — | 0 | T4 | The public `evaluate_solution_chance` has no data gate. Without a puvspr `prob` column it scores q = 1 everywhere and, under the new default, returns ≈W per feature (9.9995, against 0.0 at `b836860`). That contradicts its docstring, CHANGELOG:22-23 and spec §3.1. Internal callers are gated. | `probability.py:305, 316, 324-363` | Task 12 |
| 12 | MEDIUM (M M) | — | 4 | T4 | No Task 4 test pins that `Solution.objective` goes through the rerouted `_compute_probmode3_zscore_penalty`. Leaving the old E/V body in place passes all 38 Task 4 tests (objective 6.028 vs 3.028). Only one Task 7 assertion catches it. | `test_score_feature.py:194-196`; `utils.py:557-575` | Task 11 |
| 13 | MEDIUM (L M) | — | 5 | T5 | `ChanceCells` costs 72 B per cell: int64 copies of `held_csr`'s int32 arrays (the docstring and plan line 1865 say "share"), int64 `pos` and `c_rows`, and a full `abs_ev`. Above ~22 % occupancy it uses more memory than the dense pair it replaces (21.7 vs 8.0 MB at 10 000 × 50, 60 % occupancy). | `cache.py:73, 944-945, 971, 976` | Task 13 (docstring); layout defer-v048 |
| 14 | MEDIUM (M M) | — | 5 | platform | `c_ev = ev[:, order]` is F-ordered, so the recompute's `c_ev[:, a:b] @ m` takes MKL's threaded dgemv once a column passes ~800 cells (0.5–1 ms against ~10 µs). Mode 3 then runs at up to 7.8× mode 0 at 10 000 × 50, and 11× on the plan's dense layout, against a bound of ≤ 6×. CHANGELOG:30's 4.5–4.7× is a dev-machine figure. | `cache.py:977`; `chance_state.py:238` | Task 10 |
| 15 | MEDIUM (M M) | — | 6 | T6 | Deleting SA's temperature-sampling `delta_penalty` fails no test (380 tests), yet it changes mode 3's `initial_temp` (14.57 → 3.98, seed 1). Plan line 57's claim that "removing any one wiring site fails 2–4" is false for this site. | `simulated_annealing.py:230-233` | Task 11 |
| 16 | MEDIUM (M M) | — | 6 | T6 | Deleting the addition pass's `chance_state.apply_flip` fails no test (307 tests). `_addition_pass` then over-adds (4.0 vs 2.0), and the next removal pass hides it. | `iterative_improvement.py:335-336` | Task 11 |
| 17 | MEDIUM (M M) | — | 6 | T6 | `test_full_objective_disabled_ptarget_no_penalty` is vacuous now that the cache's full objective carries no chance term: it passes at ptarget 0.95 too. Its class docstring is now false. | `test_cache_probmode3.py:139-141, 167-183` | Task 11 |
| 18 | MEDIUM (M M) | — | 7 | T7 | `PROBZEROVAR = True` is accepted in memory (`float(True) == 1`). It is saved as `True`, reloaded as a string, and silently becomes "marxan" (objective 2.0 → 51.998). No test covers bools. | `solvers/utils.py:141-151`; `io/writers.py:122-131` | Task 12 |
| 19 | MEDIUM (L M) | — | 0 | architect | The per-term `ChanceState` repeats the whole per-flip sequence for each term. A second term roughly doubles the chance work: 7.5–10.8× mode 0, against spec §4's mode-5 bound of ≤ 6×. One-term mode 3 is within the bound. | `chance_state.py:144-175, 181-226` | defer-v048 |
| 20 | MEDIUM→LOW (L L) | — | 5 | architect, re-design | One `ptarget` vector serves every chance term (in `ChanceState`, the cache and the evaluator), but release B's 1D term needs `ptarget1d`. No effect while there is one term. | `chance_state.py:98-100, 222-225`; `cache.py:198`; `probability.py:348-352` | defer-v048 |
| 21 | MEDIUM→LOW (L L) | — | 7 | architect, T4, T7, re-design | The `ProblemCache.zero_var_rule` field default is still "met" after Task 7. Latent: `from_problem` always passes the rule. | `cache.py:190-191` | Task 12 |
| 22 | MEDIUM→LOW (L L) | — | 7 | T7 | `test_probmode_3_with_no_probability_data_returns_zero` (on the spec §3.6 list) still gives the zero-variance sentinel as its reason. It actually passes because the term is off. | `test_probability_dispatch.py:85-96` | Task 13 |

## Disputed

One line per disagreement; the ruling follows the arrow.

- **#1:** T4 rated it MEDIUM and pre-existing, since `135d025`'s dense `prob_matrix` used the same rule. Seven reviewers and 3/3 votes rated it HIGH and Task 5's: the plan rebuilt the builder, kept the rule by ruling, pinned it, and made it the tracked SA/II objective, and the Z = 4 default widens the gap. → HIGH, Task 5.
- **#2:** T5 rated it MEDIUM. Numerics and 3/3 votes rated it HIGH: it is a Task 5 regression, a blank field is plausible input, and release A does not reject it. → HIGH.
- **Fix route for #2 / #5:** three fixes were proposed:
  - rejecting non-finite cells at build (T5, numerics, science);
  - `zero |= ~(V > 0)` in the scoring core (T4);
  - a selected-only recompute (commit `bfd6c5c`).

  → Adopt all three. The fallback alone leaves `ChanceState` charging Z = 4 after a NaN PU leaves (running E/V stay NaN, and no trigger tests true on NaN). The selected-only sum alone leaves a NaN in a selected PU silent. Only rejection closes NaN.
- **#3:** T5 and 2 votes rated it HIGH: the test does not pin what it is named for. 1 vote rated it MEDIUM: the mutant still fails `test_commit_equals_preview_and_fresh_state` through `_preview is None`. → HIGH.
- **#4:** T6 and 2 votes rated it HIGH: the test spec §4 names is vacuous. 1 vote rated it MEDIUM: deleting the wiring still fails `test_passes_stop_at_an_evaluator_local_optimum`, so #15 / #16, not #4, are what falsify plan line 57. → HIGH.
- **#5:** T4 rated it HIGH and parity-critical: Marxan's V > 0 test gives Z = 4. Science and 3/3 votes rated it MEDIUM: the input is invalid, puvspr range checks are B2, the plan's predecessor also returned 0, and Marxan has no defined value for a blank field. → MEDIUM, parity on invalid input only.
- **#6:** T2 rated it HIGH, since it contradicts the task's own "agree" claim. 3/3 votes rated it MEDIUM: only invalid input triggers it, no wrong number is returned, the catch-all predates the plan, and the pre-plan output was silently wrong. → MEDIUM.
- **#7:** T5 and 1 vote rated it HIGH: the snap docstring claims what it does not pin. 2 votes rated it MEDIUM: the code is correct, and plan line 2338 never claimed per-trigger coverage. → MEDIUM. Loosening the snap rtol to spec §4's 1e-12 (platform-2) would weaken its E half further, so the two changes need one layout design.
- **#8:** T1 and 1 vote rated it MEDIUM: Task 1 rewrote this contract, which matters for parity, and no test guards it. 1 vote rated it LOW: the code is correct, and the gap predates the plan. → MEDIUM (tie).
- **#10:** T3 and 1 vote rated it MEDIUM: an output file loses a column without documentation. 1 vote rated it LOW: the change is spec-conformant, follows from the documented `Solution` change, and the dropped column was all zeros. → MEDIUM (tie).
- **#13:** T5 and 1 vote rated it MEDIUM: the "share" claim is false, and the arrays take 2.7× the dense pair at 60 %. 1 vote rated it LOW: memory is O(nnz), the worst case is 22 MB at the reference scale, and real puvspr tables are sparse. → MEDIUM (tie); fix the docstring now, change the layout with #19.
- **#19:** the architect and 1 vote rated it MEDIUM: A's layout makes B's mode-5 bound unreachable. 1 vote rated it LOW: spec §3.5 prescribes per-term arrays, so no release-A requirement is violated. → MEDIUM (tie), defer-v048.
- **#20:** the architect rated it MEDIUM. Re-design and 2/2 votes rated it LOW: release A has one term, `_TERMS = ("2d",)` rejects any other, and B's mode-5 test exists to catch this. → LOW, defer-v048.
- **#21:** T7 rated it MEDIUM. The architect, T4, re-design and 2/2 votes rated it LOW: it is latent, because `from_problem` always passes the rule. → LOW.
- **#22:** T7 rated it MEDIUM, because §3.6 requires a written reason. 2/2 votes rated it LOW: the assertion is right, and spec line 375 expects the test to stay green. → LOW.
- **task7-7 (LOW nit):** T7 reports that `validate()` loads the whole solver package, PuLP included (6.29 s cold). The architect calls the import lazy and pure, not a defect. → LOW nit kept (move the parser into `models`); no layering violation.

## Refuted or ruled out

No cluster was refuted. The claims below were checked and dropped; do not re-raise them without new evidence.

- **"The 1e-9 delta tolerance fails on badly conditioned features"** (coefficient of variation ≈ 1e-6, Z ≈ 0): `ChanceState` stays within the spread of its own full recomputes (`from_selection` is 5.2e-9 from the exact value, dense 2.1e-9), and the error is ≤ 6e-11 at CV ≥ 1e-4. Not a trigger defect (numerics).
- **"SA's tracked chance term drifts and needs a resync":** it telescopes to `penalty_total()` of the running state, so reported − tracked is the current state's error, not a cumulative one (numerics).
- **"A refill after the snap is missed" (ruling line 55):** the refill is caught once BE > 64·eps·|E|, which takes about 60 zero-variance touches, and the refilled E is within 1 ulp, as after a fresh recompute. The ruling is conditional, not wrong (numerics).
- **Surviving Task 5 mutants** (no `V ≤ 0` clause; apply always recomputing; preview matched on `idx` only): BV covers the first, the second is equivalent, and the third needs caller misuse (T5).
- **"Dropping `np.errstate` from `_chance_z` leaks warnings":** there are none on valid data (T5 and the Windows/MKL run). The `sqrt` warning on invalid q goes away with #5's fallback.
- **"Task 2's whole-problem pickling is slower or unsafe under spawn":**
  - It is faster per task than the old dict: 13 vs 50 ms at 20k PUs, 173 vs 216 ms at 100k.
  - The worker is module-level, `copy_with` keeps `type(self)`, and the test classes unpickle under spawn (T2, platform).
- **MKL thread oversubscription under `run_sweep_parallel` (#14):** not measured, so given no weight.
- **a² overflow (a ≥ 1.34e154) as a second poisoning source (#2):** same line and mechanism, but an implausible input and not reproduced. The blank-`prob` path carries #2.
- **"Task 1 switches Marxan 3's 2D term on in the binary path":** the header is now written, but Marxan 3 reads only `ptarget2d` (default −1), so `ComputeProbability2D` returns 0 (T1).
- **"`heuristic.py:199` is an unrouted `PROBMODE == 3` site":** it branches on modes 1 and 2 only (T3).
- **"`ndtr` instead of `probZUT` breaks parity":** the gap is ≤ 1.05e-9 for |z| < 5 and ≤ 2.87e-7 beyond, as documented (science).
- **"The normal approximation misstates few-cell features"** (one cell of 10·T at q = 0.7 scores P ≈ 0.90): that is Marxan's model, kept by user decision, and the docs claim nothing else (science).
- **Pre-plan behaviour in every mode:** `prop` resolution (max, against Marxan's override), and `write_puvspr` not enforcing Marxan 3's positional columns, `prob`-last header or PU order. These are not v0.47 changes (science, T1).
- **Wasted chance work, and no no-op-flip guard:** features with ptarget ≤ 0 are carried through every delta, and an all-−1 PROBMODE 3 still builds a `ChanceState`, but the 6× bound holds. Callers follow the same no-op-flip convention as `ClumpState` / `SepState` (re-design).
- **Sub-claims dropped inside confirmed rows:**
  - #17's "spec §3.6 lists this test": it lists two other tests, though §3.6's reasoning still applies.
  - #11's "the CHANGELOG's 'term off … as before' is false": it is true of the objective and misleading only for direct callers.
- **Weak guards accepted as guards** (T2, T3, platform):
  - `test_parallel`'s "caller's problem untouched" check, which runs in the parent process only.
  - `run_panel`'s requested gate, which is covered only by source-text tests (there is no Shiny harness).
  - Task 3's evaluator-on-requested-terms mutant, which cannot be detected at Task 3 and is pinned by Tasks 6 and 7.

## LOW nits (unverified)

These were not adversarially verified. "(checked)" means the claim was confirmed by a grep of `fbce740` for this synthesis.

- **T7 (architect-5):** spec §4's "Z uses the raw target (MISSLEVEL ≠ 1 does not change Z)" has no test, and no chance test sets MISSLEVEL (checked). Add a `hand_example(MISSLEVEL=0.5)` case. → Task 11
- **T1 (task1-2):** the `write_puvspr` docstring (`io/writers.py:65-69`) and CHANGELOG:45-46 give the v0.46 reason ("a missing column reads as 1"). Since Task 3, a missing column turns the term off (CHANGELOG:22-23). → Task 13
- **T2 (task2-3):** CHANGELOG:47-49 and the Task 2 commit message omit that the old rebuild also dropped `grid` and GeoDataFrame geometry/CRS, so separation lost its coordinates. → Task 13
- **T3 (task3-2):** the requested-term `mip_chance_strategy` gate in `MIPSolver.solve` (`mip_solver.py:869-871`) is unpinned for mode 3 without data. → Task 11
- **T3 (task3-3):** `_BaseModel.probmode` / `_BuiltModel.probmode` have no reader; the only read is the copy at `mip_solver.py:775` (checked). Ruling line 51 says the fields are "still read". → Task 12
- **T4 (task4-5):** two behaviours are unpinned (`test_score_feature.py:62-64, 71-76`): that "met" reports p = 1.0, and that zero variance is classified by the count rather than by V. → Task 11
- **T4 (task4-6):** the interim dense delta (`b836860`, `88a2861`) mis-scores a variance cancellation; Task 6 deletes it. → no action (Tasks 4–6 land together)
- **T5 (task5-6):** Interfaces line 1864 credits `_chance_penalty` to Task 4, but `b836860` has only `_chance_z` / `_chance_from_z` (checked). Also, `test_sparse_risk_recompute_floor` discards `worst` and `drift`, so the extra 3000 delta-checked flips claimed in ruling line 56 do not exist. → Task 11
- **T6 (task6-6):** `"pu_feat_matrix" not in cache.__dict__` (`test_cache_probmode3.py:108`) was already true before Task 6, so it does not pin "no densify". → Task 11
- **T7 (task7-5):** the docstring of `test_swap_pass_optimises_the_chance_term` (`test_sa_chance.py:160`) says zero variance is "met" in this release; that holds only because the test sets `PROBZEROVAR 1`. → Task 13
- **T7 (task7-6):** `hand_example` says f1 has E = 7.4, but its rows give 6.9 (checked), and no feature reaches the Heaviside-0 branch (`test_probzerovar.py:92`). Set f1's ptarget to 0.7. → Task 11
- **T7 (task7-7):** `validate()` with `PROBZEROVAR` imports the whole solver package through the lazy import at `models/problem.py:307` (checked). → Task 12
- **science (scientific-3):** the manual's "prob = 1 − p" workaround (`USER_MANUAL.md:431-433`) claims the same variance and zero-variance rule. For p < ~1.1e-16, q rounds to 1 and Z = 4 applies. → Task 13
- **numerics (numerics-3):** past ~1000 selected cells, the 4·eps·|value| reset is starting slack, not a bound. Say so in the module docstring and spec §3.5 line 309. Do not reset to n·eps·|value|: that breaks the 6× model. → Task 13
- **platform (platform-2):** the snap test's rtol = 1e-15 is tighter than spec §4's 1e-12 and only ~2.3× above the observed noise; "exact sum" should read "fresh sum". → Task 11, with #7

## Disposition

**Ground rules.**
- Tasks 1–7 stay as they are. Their code blocks are the prototype's byte for byte, which is what the replay check certifies, and none of the 37 items requires changing them, so execution of Tasks 1–7 continues unchanged.
- Accepted code and test fixes land as follow-up commits after Task 7 (Task 8+). Docs fixes land the same way.
- Each follow-up is built like Tasks 1–7: a prototype commit, extracted blocks, a replay check and measured RED/GREEN counts.
- Release-B items are deferred to v0.48.0.
- The plan gets a dated "Review outcome" addendum after its rulings; no task block is edited.

**Release gate and order.**
- #1 and #2 change solver results and block the v0.47.0 tag.
- The rest of Tasks 8–13 are tests and docs for code that ships in v0.47.0, and should land in the same release.
- Order: 8, 9, 10, 11, 12, 13. Task 8 goes first because it is the parity fix. Task 11 follows Task 10 because Task 10 changes the recompute's summation, which the snap-test design depends on.

- **Task 8: duplicate rows (#1, parity).** Commit `f835dd6`.
  - The fix: accumulate each row's `e = a·q` and `v = a·a·q·(1 − q)` per cell, in `_term_moments`' operand order, and keep `pos = v > 0`. The CSR/CSC layout and `ChanceState` are unchanged.
  - The commit replaces the pinning test with two cell tests (row by row, and a certain last row), adds a duplicate case to the Marxan 3 port and an SA tracked == reported case, and adds a duplicate-bearing `duplicated` layout to `LAYOUTS`, so `test_from_selection_equals_the_evaluator` and the delta tests cover duplicates. `mixed()`, `hand_example` and `_random_problem` are unchanged.
  - Cell-level equality is the pin that matters: reported == tracked alone cannot see a returned plan that misses the reported optimum.
  - Reverse ruling line 54 in the addendum.
- **Task 9: sweep and problem rebuild (#6, #9).**
  - #6 is covered by commit `9f2b849`: `_solve_single` returns `(index, None)` for a point with no solutions and the parent catches nothing, so a worker's error propagates as it does from `run_sweep`. `test_parallel_sweep_raises_what_run_sweep_raises` pins it.
  - Add #9: `add_connectivity_features` returns `problem.copy_with(features=…, pu_vs_features=…, parameters=dict(problem.parameters))`, the comment goes, and a test checks that the type, `probability`, `connectivity` and `grid` survive.
- **Task 10: non-finite and invalid cells, and the recompute (#2, #5, #14).** Commit `bfd6c5c` sums only selected cells in `_column_sums` (`bincount`, in column order). That removes NaN·0 and the BLAS call, and a recompute now equals a fresh state bit for bit. Still to add:
  - (a) A `ValueError` naming the PU ids for a non-finite puvspr `prob` or amount, raised both where the 2D cells are built and in `_term_moments`, as `_probability_map` does for prob.dat. Implement both sites: with only one, a NaN still reaches the other path silently.
  - (b) `zero |= ~(V > 0)` in `score_features` and `_chance_penalty`. This is Marxan's own test; it changes nothing on valid data and removes the `sqrt` warning. Update `_chance_z`'s docstring.
  - (c) Tests: a NaN on an unselected PU, with one more flip after the snap; q = 1.2 beside a valid uncertain cell, for both the evaluator and `ChanceState`; a NaN raising `ValueError`.
  - (d) A bench case above ~1000 cells per feature (the plan's ~30-cells-per-PU dense layout), measured on this MKL machine. Mark CHANGELOG:30's ratios as dev-machine figures.
  - Reporting an out-of-range puvspr `prob` in `validate()` stays in B2.
- **Task 11: test gaps.** Test-only, covering #3, #4, #7, #8, #12, #15, #16 and #17, plus the LOW nits architect-5, task3-2, task4-5, task5-6, task6-6, task7-6 and platform-2.
  - #3: an explicit two-PU, one-feature problem whose PUs share a cell with e, v > 0. Assert behaviour (penalty and E against a fresh `from_selection`), not `_preview`.
  - #4: add a fifth PU of cost 100 with no features, so the start (104) is worse than the unwired pass's result (39.49). For ITIMPTYPE 2, also assert `_removal_pass` directly.
  - #7, together with platform-2: the E-cancel layout (a = 1e9, q = 1 leaving) and the V-cancel layout (a = 1e5, q = 0.5 beside certain cells), both at 1e-9; plus a snap layout whose running E has drifted, checked against an explicit sum.
  - #8: write a three-column frame; assert the header is `species,pu,amount` and that there is no `prob` after `save_project` / `load_project`.
  - #12: assert `sol.objective == approx(sol.cost + sol.penalty + sol.prob_penalty)` under both rules.
  - #15: assert that mode 3's `initial_temp` differs from mode 0's for one seed.
  - #16: `_addition_pass` from {PU 1} on the reviewer's 4-PU problem returns [T, T, F, F] at 2.0.
  - #17: the `ChanceState` and `build_solution` penalties are 0 at ptarget −1 and > 0 at 0.95; fix the class docstring.
- **Task 12: small API fixes** (#11, #18, #21; LOW nits task3-3 and task7-7).
  - task7-7 first: move `_probzerovar_value` into `models/problem.py`, and have `solvers/utils.py` import it from there.
  - #18: return None for `bool` / `np.bool_`, so that `validate()` flags `True` before a save and memory and disk agree. Add `True` to `test_validate_reports_bad_probzerovar`. Writing bools as 1 / 0 in `write_input_dat` would affect every flag; that is a separate decision, not taken here.
  - #11: `evaluate_solution_chance` returns zero shortfalls and a zero penalty when the term's data column is missing, mirroring the existing no-`ptarget` early return. Add a direct-call test.
  - #21: `zero_var_rule: str = DEFAULT_ZERO_VAR_RULE` (no import cycle).
  - task3-3: drop the unread `probmode` fields.
- **Task 13: docs and the plan addendum** (#10, #13, #22; LOW nits task1-2, task2-3, task7-5, scientific-3 and numerics-3).
  - CHANGELOG:
    - #10: for mode 3 without a puvspr `prob`, mvbest omits `Prob_Gap` and the target-met table omits its probability columns.
    - task1-2: the corrected reason in the `write_puvspr` entry.
    - task2-3: the old rebuild also dropped `grid` and GeoDataFrame geometry.
  - Docstrings:
    - `ChanceCells`: "share" → "same structure as" (#13).
    - The dispatch test (#22) and `test_sa_chance.py:160` (task7-5).
    - `write_puvspr` (task1-2).
    - The reset-bound wording in `chance_state.py` and spec §3.5 line 309 (numerics-3).
  - Manual: the p < ~1e-16 caveat on the 1D workaround (scientific-3).
  - Plan addendum, by plan line:
    - Ruling 51: the `probmode` fields are unread.
    - Ruling 52: holds only for q ∈ [0, 1] (Task 10).
    - Ruling 54: reversed (Task 8).
    - Ruling 56: sparse-risk delta coverage is 2500 flips until Task 11.
    - Ruling 57: two wiring sites are unpinned until Task 11.
    - Ruling 59 and CHANGELOG:30: measured on the dev machine's BLAS (Task 10).
    - Line 472: Task 3 has three observable effects.
    - Lines 1864–1865: `_chance_penalty` is Task 5's, and `ChanceCells` copies the arrays rather than sharing them.
- **defer-v048 (release B).**
  - #19: decide before B3 between two options:
    - one stacked layout for all terms sharing the CSR (`ev` 2T × nnz, `EV` / `B` 2T × n_feat, `N` T × n_feat), giving one gather, one scatter and one vectorised score per flip;
    - an amended mode-5 bound in B's plan.
  - #20: per-term `ptarget` / `ptarget_den` on `_TermState`, plus an evaluator column map (`{"2d": "ptarget"}`, to which B adds `"1d": "ptarget1d"`). This goes with #19's layout.
  - #13's dtype and sharing changes (int32 indices, int8 `pos`, dropping `abs_ev`), bench-gated, since #19 reshapes those arrays.
  - #10's MIP notice text, together with B's "term without data" diagnostic (Shiny is release B).
  - `validate()` reporting of a non-finite or out-of-range puvspr `prob` (B2). The solve-time `ValueError` for non-finite cells is release A (Task 10).
  - Already in release B, so not re-filed: reading `ptarget2d` and switching 2D on from the `prob` header (B1 / B4); `PROBZEROVAR` in the Shiny panel.
- **Early follow-ups.** Commits `f835dd6`, `9f2b849` and `bfd6c5c` were written while this review ran, and this review does not cover them. Each was built TDD with measured RED/GREEN (Task 8: 6 failed → 184 passed in the eight chance test files; Task 9: 1 failed → 8 passed; Task 10: 2 failed → 6 passed, its `long_column` case failing bit for bit on MKL before the change). The remaining follow-ups were built and mutation-verified separately, one commit per finding group; the plan's "Review outcome" addendum lists them all.
