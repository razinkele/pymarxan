# PROBMODE 1/2 consistency (v0.45.0) — review synthesis

**Date:** 2026-10-17 · **Reviewed:** spec `2026-10-17-probmode-consistency-design.md` (three
review-loop rounds and a fresh-eyes final review before planning), plan
`2026-10-17-probmode-consistency-plan.md` (`bd1a60a`), prototype `2a92d0b` · Raw findings and
probes: session scratchpad `probmode/review/` (spec loop) and `probmode/planreview/` (plan loop).

## Spec review loop

| Round | Critical | Important | Minor | Headline |
|---|---|---|---|---|
| 1 | 1 | 7 | 13 | The Stage 3 golden's 8 PROBMODE 2 records move (the spec said none did) → `STAGE3_PROBMODE2_FIXED`; a separate `held_csr` so clumping/separation stay raw; representation and gap analysis in scope |
| 2 | 0 | 3 | 6 | `held_csr` must keep the CSR structure; the mode-2 oracle needs a high SPF; gap's denominator |
| 3 | 0 | 0 | 2 | Converged |
| Final (fresh eyes) | 0 | 2 | 6 | Red-first evaluator fixtures; pinned SA parameters (avoids ~9 min of test time); every remaining name and shape fixed |

## Plan review, round 1 (4 lenses)

| Lens | Critical | Important | Minor |
|---|---|---|---|
| Architect | 0 | 2 | 8 |
| Grounding (executed: RED counts reproduced, replay byte-identical, 29/30 mutants killed, perf flat, +11.5 s test time) | 0 | 2 | 7 |
| Science | 0 | 5 | 6 |
| Independent re-design | 0 | 2 | 5 |

| # | Lens | Finding | Ruling |
|---|---|---|---|
| 1 | architect, re-design | Probability semantics coded in five places (utils, cache, `_totals`, MIP, greedy) | **P1**: shared helpers in `solvers/utils.py`, used everywhere |
| 2 | architect | Representation passes an int 0/1 selection straight to `compute_held`; numpy treats it as row positions (215.0 instead of 37 on `simple`) | **P2**: `compute_held` applies `_coerce_selected` |
| 3 | re-design, grounding | NaN probabilities (blank `prob.dat` cells) pass `validate()` and behave differently in every path (SA NaN objectives, evaluator p = 0, MIP raises) | **P3**: `validate()` reports them; the helpers raise |
| 4 | re-design | PROBMODE 2 with clumping/separation: SA optimises raw amounts, reports use expected (tracked 73 vs reported 5073) | **P4**: one `UserWarning` per solve; full fix backlogged |
| 5 | science | Docs and UI disagree on what probability means (loss vs persistence) | **P5**: probability of loss everywhere |
| 6 | science | Mode 2's consequence unstated (expected = target ⇒ ~50 % chance) | **P5** |
| 7 | science | PROBMODE 3 called "Marxan's Probability 2D" despite two differences (zero-variance Z = 4; `ptarget` vs `ptarget2d`) | **P5**: "based on", differences stated; backlog |
| 8 | science | Marxan v4's `probZUT` has an integer-division bug (compiled and run) that inverts its probability penalty | **P5**: the Probability 1D stage references Marxan 3 |
| 9 | science | The 1D penalty summary is too loose to seed the next stage | **P5**: precise definition in the roadmap |
| 10 | grounding | The base-cost COSTTHRESH rule is untested; a mutant survives the whole suite | **P7**: COSTTHRESH in the fixtures |
| 11 | grounding, architect, re-design | No iterative-improvement test; a slow test unmarked | **P8** |
| 12 | grounding | Bench bound comment understates the noise floor | **P9** |
| — | various | Stale comments/docstrings, `_totals` docstring, golden-override assertion, duplicate-PU fixture in a MIP test | **P6** |

Not adopted: renaming `expected_feature_totals` (spec §4a is binding; docstring fixed instead);
building `held_csr` from shared structure (modest saving); presolve P6 expected reachability and
the full clumping/separation fix (documented limitations, backlog).

None of the findings changes the MIP's results; the only moving golden records remain the 8
Stage 3 PROBMODE 2 cases. The 35.0 anchor holds.

## Plan review, round 2 (grounding, executed)

Clean: 0 Critical, 0 Important, 6 Minor. Every round-1 Important from all four lenses addressed;
RED counts and the byte-identical replay re-verified; 43 mutants, 36 killed (2 equivalent, 4
greedy-only, 1 harmless); 450 MIP-model and greedy-plan records identical between base and
prototype after the shared-helper refactor. Minors applied: greedy ignores a NaN table when the
premium is off (R2-1); greedy's shared semantics pinned by tests that kill the first-row-wins and
default-mode-0 mutants (R2-2); warning frequency stated as once per cache build (R2-3); spec
duplicate-PU rationale corrected (R2-4); Shiny representation help text (R2-5). Loop converged.
