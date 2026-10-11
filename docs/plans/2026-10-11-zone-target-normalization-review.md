# Zone-target normalization (#7) — 4-lens design review synthesis

Reviewed: `2026-10-11-zone-target-normalization-design.md` (rev 1, `d31ff72`).
Run: Workflow `wf_f46a2ff5-6c5` — architect, codebase-grounding (executed prototypes),
MarZone-parity/science (read `Marxan-source-code/marzone` C++, scite), independent re-design;
each CRITICAL/HIGH finding then adversarially verified by a skeptic agent. 8 agents, 0 errors.

**Verdict:** core decisions confirmed by all four lenses (MarZone skip-unknown + last-wins,
validate() reports, single-target upper-bound diagnostic, `[]` + warning, public alias). One
false premise reshapes the plumbing. No CRITICAL. Spec patched to rev 2.

## Verified facts (grounding lens ran them)

- Issue table reproduces exactly: dup → `PulpError` ("overlapping constraint names:
  shortfall_2_1"); unknown zone → MIP `[]`, df 17.0, cache 7.0; unknown feature (target 1) →
  MIP `[]`, df 8.0, cache 7.0.
- Duplicates already agree last-wins in the DataFrame path (dict overwrite in
  `compute_zone_shortfalls`) and the cache; duplicates are a **MIP + writer** problem only.
  Unknown ids are a MIP + objective (phantom penalty) + writer problem.
- A prototype normalizer wired into all consumers makes every variant agree; no existing test
  pins the buggy behaviour (only `ZONES_EXPORTS` needs the new names).
- Diagnostic prototype: `[]` on the anchor; amount source matches the MIP (`_feature_groups`,
  no PROBMODE involvement; MISSLEVEL handled).
- MarZone C++: `LoadZoneTarget` only appends; `BuildZoneTarget` skips unknown species/zone and
  overwrites (last-wins); `PopulateZoneContributions` same. targettype-1 resolution is per-row,
  so resolving before normalizing is order-independent. Watts et al. 2009 verified real, not
  retracted (eq. numbers not confirmable from indexed text).

## Findings

| # | Sev | Lens(es) | Finding | Disposition |
|---|---|---|---|---|
| 1 | **HIGH** (verified ×2) | architect, grounding, re-design | Heuristic & II never build `ZoneProblemCache` (only `zones/solver.py:68`); under rev 1 they never warn, and `RunModePipeline` would warn per run. | **Accepted.** Warning moves to solve entry of all four zone solvers (+ `ZoneIISolver.improve`) via one helper next to `raise_if_separation_active`; downstream never warns. |
| 2 | HIGH→**MEDIUM** (verified) | architect, grounding, re-design | Pandas normalizer inside `check_zone_targets`/`compute_zone_shortfalls` runs per candidate move in heuristic/II: `compute_zone_objective` +121% (606→1340 µs). Rev 1's "once per solve" claim is false. | **Accepted.** Replace the DataFrame normalizer with a cheap insertion-ordered dict accessor `zone_target_lookup()` (pattern of `contribution_lookup`) that replaces the existing row loops; comparative bench ≤ 1.05× added. |
| 3 | MEDIUM | architect, grounding, re-design (P) | `writers.write_zone_summary` reads raw rows; missing from consumer table; writes contradictory rows today. | **Accepted.** Added to consumers + writer test. |
| 4 | MEDIUM (P) | parity | MarZone applies zone targets only when > 0, **after** last-wins (`reserve.hpp:241,472,846`); a later 0 switches a pair off. Rev 1 diagnostic had no `t > 0` guard (false report for target 0 + forced negative contribution); under ZONETARGETCONTRIB=1 a surviving 0-row becomes `expr >= 0`. | **Accepted.** Lookup: last-wins first, then drop `t <= 0`. Diagnostic loops guard `t > 0` (matches `mip_solver.py:457`). Tests added. |
| 5 | MEDIUM | architect, parity | Diagnostic belongs next to the MIP's amount source; locked-in/out parsing exists 5×. Status-2 → first zone is a **pymarxan** convention, not MarZone; `zone_ids` is a set. | **Accepted.** New `zones/diagnostics.py`; shared `_allowed_zones` used by MIP + diagnostic; reuse `_feature_groups`; `min(zone_ids)`; docstring cites pymarxan convention. |
| 6 | MEDIUM | architect, re-design | validate() and the normalizer would duplicate detection; plain `UserWarning` not filterable. | **Accepted.** One `zone_target_issues()` helper feeds validate() + warning; `ZoneTargetWarning(UserWarning)`. |
| 7 | MEDIUM | architect, grounding | MIP helpers would normalize 3×; wrong names cited (`_zone_target_exprs`, `_zone_target_penalty`, `from_problem`). | **Accepted.** Real names: `_zone_achieved_exprs`, `_build_penalty_expr`, `_add_zone_target_constraints`, `ZoneProblemCache.from_zone_problem`. Each reads the cheap lookup once. |
| 8 | LOW | re-design | Fallback message merges "compete" with "time limit" though `model.status` separates them. | **Accepted.** Branch on status. |
| 9 | LOW | grounding, re-design | Test numbers: dup must discriminate (25→10 and 10→25); 4(b) needs PU1+PU2 locked out (one PU → "compete" case); unknown-feature target is 1. | **Accepted.** |
| 10 | LOW | grounding, architect | Formula vs prose disagree on locked-in PUs; zone bound drops forced negatives. | **Accepted.** One per-PU rule for both bounds. |
| 11 | LOW | architect | `metadata["zone_targets_met"]` loses keys for unknown-id (and now ≤0) rows. | **Accepted.** CHANGELOG + test. |
| 12 | LOW | parity | Last-wins is per field (area vs occurrence); exact on (zone, feature) only because types 2/3 are rejected. | **Accepted** (one sentence). |
| 13 | LOW | parity | `ZONETARGET2NAME` / `ZONECONTRIB2NAME` / `ZONECONTRIB3NAME` silently ignored by the loader. | **Deferred** — separate issue. |
| 14 | LOW | re-design | Elastic re-solve would give exact shortfalls on the "compete" path. | **Deferred** (optional future); warning points to ZoneSASolver. |
| 15 | LOW | architect | Name overlap with objectives-package private `_compute_zone_achieved`. | Docstrings updated; private rename not required. |

P = parity-relevant.

## Lens notes

- The re-design lens independently converged on the same core design, and on the dict-lookup
  shape (finding 2) — the strongest signal the rev-2 plumbing is right.
- The grounding lens again earned its keep by executing: it found finding 9's test fixtures
  would not have discriminated last-wins (10→20 is met either way) and that a single lockout
  doesn't make the anchor's overall target individually unreachable.
