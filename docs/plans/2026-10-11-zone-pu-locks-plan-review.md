# Zone PU locks (#6) — implementation-plan review synthesis

Reviewed: `2026-10-11-zone-pu-locks-implementation.md` rev 1 (`6c039fc`). Workflow
`wf_d34d45b9-c87`: dry-run executor (applied all 7 tasks verbatim in an isolated worktree),
spec-coverage / interfaces, test-discrimination (mutation probes on a scratch build), blast
radius (incl. ROWER); HIGH+ adversarially verified. 9 agents, 0 errors.

**Verdict:** the plan executes. Full suite **2323 passed**, zones+solvers+analysis 1094 passed,
ruff + mypy clean (after line wraps), parity MIP 35.0 / round-trip True, all 10 golden cases
green through every task; `read_input_dat` keeps `PULOCKNAME` / `PUZONENAME`. No HIGH survived
verification (4 raised, all confirmed and downgraded to MEDIUM). Plan patched to rev 2.

| # | Sev | Lens | Finding | Disposition (plan rev 2) |
|---|---|---|---|---|
| 1 | HIGH→MEDIUM | dry-run, tests | v0.37 golden fixture (4 PUs, 2 swappable, converged SA) does not catch an SA RNG-stream change or swap-order drift. | Second **large** fixture (30 PUs + a tie PU, 3 zones, NUMITNS 400) recorded on v0.37; explicit **mutation check** (extra RNG draw; reversed swap order) must fail before continuing. |
| 2 | HIGH→MEDIUM | tests | No strict-subset restriction test: SA/heuristic/II could ignore puzone and pass. | `_strict_subset()` fixture (zone 3 free + dominant, PUs restricted to {1, 2}) used by MIP, SA, heuristic, II 1/2/3. |
| 3 | HIGH→MEDIUM | tests | SA `must_assign` case cannot fail (SA never proposes 0). | Kept as a labelled pin; strict-subset carries the proof. |
| 4 | HIGH→MEDIUM | dry-run | Bench confounded: restricted runs skip ~12 % of delta calls → ratio ≈ 0.90 whatever the overhead; no comparison with v0.37's draw. | Time per **evaluated** delta; second assertion `_propose` vs the v0.37 inline draw ≤ 1.10×. |
| 5 | MEDIUM | dry-run, coverage, tests | Task 4 red list wrong (Task 1 already routes locks into the MIP). | Corrected (3 failing, 3 pins + strict-subset pin); Task 1 note updated. |
| 6 | MEDIUM | dry-run | Task 4 restriction-note snippet mis-indented. | Whole `else:` block given; note placed before the full stop. |
| 7 | MEDIUM | coverage | Issue messages contradicted the rules for status-3 / pulocked PUs (single-zone, all-unknown). | Those classes exclude status-3 and pulocked PUs; test. |
| 8 | MEDIUM | tests | Task 6 red list overstated. | "Must fail" set listed; count recorded at commit. |
| 9 | LOW | various | E501/I001 in plan text; wrong ImportError symbol; docstring text; unused locals (`pu_id_to_idx`, `zone_ids_list`, `n_zone_options`, `first_pid`); stale line anchors; `_emit` stacklevel for direct callers; conditional `load_project` instruction. | All fixed (content anchors, explicit cleanup lists, `stacklevel` parameter, fact statement). |
| 10 | LOW | coverage | Missing precedence pairs (status 1 + puzone), runtime invariant, per-row validate tests, repeated-lock-unknown-zone message, improve() warn-once, non-contiguous ids through solvers, reader `1,,2` / `;`, loader PU-30 and named-empty-file tests. | All added; reader uses one delimiter per file (`,`, else tab, else whitespace). |
| 11 | LOW | blast-radius | Issue scan ran on every v0.37 problem. | Short-circuit when both tables are `None`. |
| 12 | MEDIUM (out of repo) | blast-radius | ROWER copies status rules and hashes problems without the new tables. | Follow-on note in the plan (on the v0.38 pin bump). |
| 13 | LOW | blast-radius | Pipelines warn once per stage; no `save_zone_project`; status edits (WDPA) don't beat locks. | CHANGELOG lines added. |
