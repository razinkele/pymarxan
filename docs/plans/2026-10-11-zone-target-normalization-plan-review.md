# Zone-target normalization (#7) — implementation-plan review synthesis

Reviewed: `2026-10-11-zone-target-normalization-implementation.md` (rev 1, `d50c8ba`).
Run: Workflow `wf_951d7480-579` — dry-run executor (applied every task verbatim in an isolated
worktree), spec-coverage/interfaces, test-discrimination (mutation probes), blast-radius
(incl. the ROWER sibling repo); CRITICAL/HIGH findings adversarially verified. 6 agents, 0 errors.

**Verdict:** the plan executes. Every task's RED expectation matched (Task 3 under-listed its
failures), every task goes GREEN, ruff + mypy clean after line wraps, full suite **2220 passed**,
lookup/objective ratio **0.04** (budget 0.15), `tests/data/simple` MIP **35.0**, MarZone anchor
`(1, 2, 2)` at 7.0. One HIGH (ROWER compatibility) changes a design detail; the rest are
test-quality and plan-hygiene fixes. Plan patched to rev 2; spec patched to rev 3.

## Findings and dispositions

| # | Sev | Lens | Finding | Disposition |
|---|---|---|---|---|
| 1 | **HIGH** (verified) | blast-radius | ROWER `evaluation.py:118` indexes `check_zone_targets(...)[(z, f)]` for every raw row and deliberately emits target-0 rows; rev 1 dropped `t <= 0` pairs from the met-flags → `KeyError` on re-pin. | **Accepted, design detail changed.** `zone_target_lookup(include_inert=True)` also returns known pairs with surviving `t <= 0`. `check_zone_targets` and `write_zone_summary` use it and report those pairs as met — exactly the `check_overall_targets` convention (every feature, inert = True). Shortfalls, penalty, cache and MIP use effective (`t > 0`) pairs only, so MarZone's `target > 0` gate still holds. Unknown ids stay absent everywhere. |
| 2 | HIGH→MEDIUM (verified) | dry-run | `_known_id` calls `int()` → `validate()` raises on a non-numeric id (`'x'`), and 2.5 silently becomes zone 2. | **Accepted.** `float()` + `is_integer()` parse; tests add `'x'` and `2.5`. |
| 3 | MEDIUM | dry-run, tests, blast-radius | "One warning per run" is false for RUNMODE 2–5 (2–3 zone-solver calls per run); test asserted only `>= 1`. | **Accepted (a):** documented as one warning per zone-solver call; test parametrised over RUNMODE 1/2/5 pinning `num_solutions × stages`. |
| 4 | MEDIUM | coverage, blast-radius | `ZoneTargetWarning` never exported from `pymarxan.zones` (spec + CHANGELOG say it is). | **Accepted.** Export in Task 1 + `ZONES_EXPORTS`. |
| 5 | MEDIUM | blast-radius | MIP failure warning is a bare `UserWarning` — loops that expect infeasibility can't filter it. | **Accepted.** `ZoneMIPNoSolutionWarning(UserWarning)` in `diagnostics.py`, exported; diagnostic call wrapped so a diagnostic crash can never turn `[]` into an exception. |
| 6 | MEDIUM | tests | No test kills the `max(0, …)` floor mutant for free PUs. | **Accepted.** `test_free_pu_may_stay_unassigned_when_every_zone_is_negative`. |
| 7 | MEDIUM | dry-run | ZoneSASolver warning tests ran 1M iterations (38 s). | **Accepted.** `_fast()` sets `NUMITNS = 1000` (file: 0.46 s). |
| 8 | MEDIUM | dry-run | Bare-`python` parity harness in a worktree silently tests main's code; `.venv/bin/mypy` absent in worktrees. | **Accepted.** `PYTHONPATH=src`; mypy by absolute path `/home/razinka/marxan/.venv/bin/mypy`. |
| 9 | LOW | tests | `dup_10_then_25` compares `[]` with `[]`; no feasible discriminating MIP last-wins case. | **Accepted.** New variant `dup_zone1_10_then_20` (zone-1 target): last-wins → MIP 11.0 at `(1, 1, 0)`; first-wins would give 7.0. |
| 10 | LOW | tests | Zone-target MISSLEVEL scaling in the diagnostic untested. | **Accepted.** `test_misslevel_scales_zone_targets`. |
| 11 | LOW | tests | Mask gate pinned only for default contributions; no positive check. | **Accepted.** Extra structural asserts. |
| 12 | LOW | tests | `_mip_warning` prefix filter also catches `ZoneTargetWarning`. | **Accepted** (superseded by #5: filter on category). |
| 13 | LOW | tests | Review Focus #1 only partly covered (no MIP / metadata / summary). | **Accepted.** Task 3 test covers MIP; Task 2 covers met-flags/summary under the #1 convention. |
| 14 | LOW | dry-run, coverage | Task 3 RED line under-lists (12 fail); stale line anchors after Task 3; E501 from literal docstrings; ruff I001 before `diagnostics.py` exists; `ZONES_EXPORTS` not alphabetical. | **Accepted.** Exact RED list; content anchors; docstrings pre-wrapped; I001 note; "insert after" wording. |
| 15 | LOW | coverage | `zone_target_issues` ignored a missing `target` column while the lookup returned `{}`. | **Accepted.** Same column guard. |
| 16 | LOW | coverage | Bench differs from spec Testing #9 without saying so; spec still names `mip_solver._feature_groups`, `:457`. | **Accepted.** Spec rev 3 aligned. |
| 17 | LOW | blast-radius | In-tree helpers (`test_zone_mip_metadata.py:22-23`, `test_writers.py:261`) derive keys from raw rows. | **Accepted.** Switched to `zone_target_lookup(include_inert=True)`. |
| 18 | LOW | coverage | Spec test 7(g) (bound == MIP coefficient sum) has no dedicated test. | **Accepted as stated:** shared `_feature_groups` import + `test_duplicate_puvspr_rows_summed` stand in. |
| 19 | LOW | blast-radius | ROWER duplicates the diagnostic's bound logic (`attainable_by_target`). | **Deferred** — ROWER follow-on after re-pin. |
| 20 | LOW | dry-run | 5 of 14 Task 2 tests pass pre-fix. | **Accepted as pins**, labelled in the plan. |

## Verified numbers (dry-run + test lens)

- RED: Tasks 1, 2, 4 collection `ImportError` as stated; Task 3 — 12 failures
  (`PulpError: overlapping constraint names: shortfall_2_1`, `assert 0 == 1`).
- Diagnostic expectations reproduced: 15/10, 15/12, 15/3, 1/−5, compete case `[]`, ghost 5/0.
- Negative-weight anchor: DataFrame 21.0, cache 26.0 before / 21.0 after.
