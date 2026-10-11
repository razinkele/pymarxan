# Zone-target normalization, validation and MIP infeasibility diagnostic — design

Issue: razinkele/pymarxan#7 (follow-on from #2 / v0.36.0).
Status: **rev 2** — rev 1 (`d31ff72`) approved in chat 2026-10-11 (option "MarZone semantics");
patched after the 4-lens review (`2026-10-11-zone-target-normalization-review.md`, findings
1–12 accepted, 13–14 deferred).

## Problem

`ZonalProblem.zone_targets` rows that name an unknown zone or feature, or that repeat a
`(zone, feature)` pair, get different treatments today. Reproduced on the #2 MarZone anchor
(`tests/pymarxan/zones/marzone_anchor.py`) with assignment `(1, 2, 2)`:

| Case | `ZoneMIPSolver` | `compute_zone_objective` | `ZoneProblemCache` | `write_zone_summary` |
|---|---|---|---|---|
| Duplicate (zone, feature) row | raises `PulpError` ("overlapping constraint names: shortfall_2_1") | last-wins (dict overwrite) | last-wins (overwrite) | one row **per input row** (contradictory) |
| Unknown zone id | `[]` | 17.0 (phantom penalty) | 7.0 (silently skipped) | row for the phantom pair |
| Unknown feature id (target 1) | `[]` | 8.0 | 7.0 | row for the phantom pair |

SA minimises the cache objective but reports the DataFrame objective, so on these inputs
`Solution.objective` disagrees with what the search minimised. No solver calls `validate()`.

Separately, `ZoneMIPSolver` returns `[]` with no explanation when the model is infeasible. Since
v0.36.0 overall targets are hard constraints and partial contribution tables default unlisted
pairs to 0, an unreachable target in a hand-built problem is easy to create and hard to find.

Finally, `zones/writers.py` imports the private `objective._compute_zone_achieved`.

## MarZone reference semantics (verified 2026-10-11, review §"Verified facts")

`Marxan-source-code/marzone`:

- `LoadZoneTarget` only appends rows (no filtering, no dedup).
- `zones.hpp` `BuildZoneTarget`: unknown species id → `continue // invalid species - skip`;
  unknown zone id → `continue // ignore invalid zone ids`; otherwise
  `zoneTarget[spindex][zoneind].target = …` — plain overwrite, so a repeated pair is
  **last row wins**. Last-wins is per field (area vs occurrence); deduplicating on
  `(zone, feature)` is exact only because pymarxan rejects occurrence types 2/3. If those are
  ever supported, the key becomes `(zone, feature, area|occurrence)`.
- `reserve.hpp:241,472,846`: a zone target is applied only when `target > 0` — evaluated
  **after** last-wins, so a later `0` row switches an earlier positive target off.
- targettype 1 (proportion) is resolved per row (`target × SpecArea`), so pymarxan resolving it
  in the reader before normalization is order-independent.
- `PopulateZoneContributions`: same skip-unknown / last-wins. pymarxan's `contribution_lookup()`
  (dict comprehension) and `contribution_matrix()` (skips unknown ids) already match; only
  validation reporting is missing.
- `speciesid == -1` ("all species") stays rejected by the reader — out of scope.

## Design

### 1. One cheap accessor — `ZonalProblem.zone_target_lookup()`

New method in `zones/model.py`, next to `contribution_lookup()` and following its pattern:

```python
def zone_target_lookup(self) -> dict[tuple[int, int], float]:
    """Effective zone targets keyed ``(zone_id, feature_id)``, MarZone load semantics.

    Rows naming a zone not in ``zone_ids`` or a feature not in ``features["id"]`` are
    skipped; a repeated pair keeps its LAST row (zones.hpp BuildZoneTarget); pairs whose
    surviving target is <= 0 are then dropped (reserve.hpp applies zone targets only when
    > 0). Insertion order = order of first appearance of each surviving pair's key, so
    MIP constraint names stay deterministic. ``{}`` when ``zone_targets`` is ``None`` or
    lacks the ``zone``/``feature``/``target`` columns (``validate()`` reports those).
    No side effects (never warns).
    """
```

- Plain Python over `zip(zone, feature, target)` with prebuilt `set`s of zone/feature ids — the
  same cost class as the row loops it replaces, not a pandas filter (review finding 2: a pandas
  normalizer per call made `compute_zone_objective` +121%).
- Values are raw targets; MISSLEVEL is applied by consumers, as today.
- NaN ids: a row whose zone or feature is NaN is treated as unknown (skipped), not `int()`-cast.
- Not cached on the instance (mutable dataclass; `copy_with` must not carry stale state).

**Consumers switched to it** — every read of `problem.zone_targets` that drives an objective,
constraint, met-flag or report:

| Site | After |
|---|---|
| `mip_solver._zone_achieved_exprs`, `_build_penalty_expr`, `_add_zone_target_constraints` | iterate the lookup (one call per helper; O(rows)) |
| `objective.check_zone_targets`, `objective.compute_zone_shortfalls` | iterate the lookup |
| `cache.ZoneProblemCache.from_zone_problem` zone-target matrix (`cache.py:180-188`) | fill from the lookup |
| `writers.write_zone_summary` (`writers.py:174-189`) | one row per lookup pair |

Effect: duplicates no longer raise `PulpError`; unknown-id and `<= 0` pairs contribute nothing
anywhere; all four components agree.

### 2. Shared issue detection, `validate()` and the warning

```python
def zone_target_issues(self) -> tuple[list[tuple[int, object, object]], list[tuple[int, int]]]:
    """(unknown rows as (row position, zone, feature), duplicate (zone, feature) keys)."""
```

One private-in-spirit, public-on-the-class helper (`zones/model.py`), returning `([], [])` when
`zone_targets` is `None` or lacks required columns. Used by:

- **`validate()`** (`model.py:137`): one error for unknown ids, one for duplicate pairs ("…
  resolved last-wins by the solvers"), first five shown, same format as the existing
  unknown-contribution error. Plus one error for duplicate `(feature, zone)` rows in
  `zone_contributions` (unknown ids there are already reported). These are errors (consistent
  with the existing check); `validate()` still never raises.
- **The solve-entry warning:** new `warn_zone_target_issues(problem, solver_name)` in
  `zones/model.py` (or beside `raise_if_separation_active` in `solvers/separation.py` — the plan
  picks one; it must not import solver modules). Emits one `ZoneTargetWarning(UserWarning)`
  (new class, exported from `pymarxan.zones`) when `zone_target_issues()` is non-empty, e.g.
  `"ZoneSASolver: zone_targets has 2 row(s) with unknown zone/feature ids (skipped, as MarZone
  does) and 1 duplicate (zone, feature) pair(s) (last row wins)"`. Called **once at the entry of
  every zone solver** — `ZoneMIPSolver.solve`, `ZoneSASolver.solve`, `ZoneHeuristicSolver.solve`,
  `ZoneIISolver.solve` and `ZoneIISolver.improve` — right after the type and separation checks.
  Nothing downstream warns.

`RunModePipeline` calls a zone solver once per run with `num_solutions=1`, so it emits one
warning per run; Python's default filter collapses repeats from the same call site. Documented,
not suppressed.

### 3. Infeasibility diagnostic — `diagnose_zone_infeasibility`

New module `zones/diagnostics.py`, function exported from `pymarxan.zones`.

**Shared allowed-zone rule.** New private `_allowed_zones(problem) -> list[frozenset[int]]`
(per PU, in `planning_units` order) used by **both** `ZoneMIPSolver` (replacing its inline
locked-in/out constraints' status parsing) and the diagnostic, so #6 part 2 (lock into a named
zone) changes one place:
status 3 → `∅`; status 2 → `{min(zone_ids)}`; otherwise all zones. Missing `status` column →
all 0. This is **pymarxan's** status convention (ZoneMIPSolver / SA / heuristic / II all lock
status 2 into the first sorted zone), not a MarZone rule — the docstring says so.

**Bounds** — one per-PU rule for both kinds of target. For PU `i` and a per-zone value
`v_i(z)`, the PU's best contribution is:

- `allowed(i) = ∅` → 0;
- status 2 (forced into `z*`) → `v_i(z*)` as-is (may be negative — it is forced);
- otherwise → `max(0, max_{z ∈ allowed(i)} v_i(z))` (the PU may stay unassigned:
  MIP `Σ_z x[i,z] ≤ 1`).

Overall target `f`: `v_i(z) = amount[i,f] × contribution[z,f]`; `UB_f = Σ_i best_i`.
Zone target `(k, f)`: `v_i(z) = amount[i,f] × w[k,f]` if `z = k`, else `0`; `UB_kf = Σ_i best_i`.

Amounts come from `mip_solver._feature_groups` (the MIP constraints' own source: summed
duplicate `pu_vs_features` rows, unknown PUs dropped; no PROBMODE adjustment), so the bound
is the MIP's by construction. Targets: overall `features.target`, zone targets from
`zone_target_lookup()`; both scaled by MISSLEVEL; **only `t > 0` checked** (matches
`mip_solver.py:457`). A target is reported when `UB < t × MISSLEVEL` beyond the shared 1e-9
relative tolerance (`_meets`). Messages, e.g.
`"overall target for feature 3 is 120.0 but at most 85.5 is reachable"` /
`"zone target (zone 2, feature 1) is 40.0 but at most 12.0 is reachable"`.
Returns `[]` when every target is individually reachable.

**Surfacing.** At `ZoneMIPSolver.solve`'s `return []` (`mip_solver.py:205-206`), branch on
`model.status`:

- `Infeasible`: diagnostic non-empty → `UserWarning`
  `"ZoneMIPSolver: infeasible. Unreachable targets: …"` (first five, then count); empty →
  `"ZoneMIPSolver: infeasible, but every target is reachable on its own — targets compete for
  the same planning units. ZoneSASolver treats targets as penalties and will return a
  best-effort plan."`
- `NotSolved` / no incumbent (time limit): `"ZoneMIPSolver: no feasible solution found within
  the time limit (status …); raise MIP_TIME_LIMIT or use ZoneSASolver."`
- other statuses: generic message with the status name.

Return value stays `[]` (no API change). `validate()` does **not** call the diagnostic —
unreachable targets are legitimate for the penalty-based solvers. Zone targets are hard only in
pymarxan's MIP (MarZone treats them as penalties); the docstring says so.

Deferred (review #14): elastic re-solve to report exact shortfalls on the "compete" path.

### 4. Public `compute_zone_achieved`

Rename `objective._compute_zone_achieved` → `compute_zone_achieved`; keep
`_compute_zone_achieved = compute_zone_achieved` as a private alias; export from
`pymarxan.zones.__all__` (and add to the `ZONES_EXPORTS` test list); switch `writers.py` and
internal callers; update the docstrings that name the private function (`objective.py:272`,
`writers.py:11`).

## Testing

All under `tests/pymarxan/zones/`, TDD, on the MarZone anchor unless noted.

1. **Agreement:** variants with duplicate rows **25→10 and 10→25** (so last-wins is
   discriminated), unknown zone 9, unknown feature 99 (target 1), and all three combined, at
   assignment `(1, 2, 2)`: `compute_zone_objective` == cache full objective == objective of the
   hand-normalized problem; `ZoneMIPSolver` does not raise and matches the hand-normalized
   MIP result (10→25 is jointly infeasible in both — same `[]`). The DataFrame-vs-cache
   duplicate assertions are pins (already last-wins); the RED cases are MIP duplicates,
   objective unknown ids, and the writer.
2. **Lookup semantics:** last-wins; unknown ids skipped; positive-then-0 pair → absent;
   NaN id skipped; missing columns → `{}`; insertion order.
3. **Warning:** each of the four solvers (+ `improve`) emits exactly one `ZoneTargetWarning` on
   bad input and none on the clean anchor.
4. **`metadata["zone_targets_met"]`** keys equal the lookup's pairs.
5. **validate():** each new error appears for the bad input, not for the anchor.
6. **Writer:** `write_zone_summary` with a duplicate row → one row with the last-wins target.
7. **Diagnostic:** (a) anchor → `[]`; (b) PU1 and PU2 locked out → overall target named, and the
   MIP warning names it; (c) partial contribution table leaving a feature at 0 → named;
   (d) locked-in PU counts only in `min(zone_ids)`, negative forced contribution included;
   (d') same under `ZONETARGETCONTRIB=1`; (e) PU1 locked out only (or zone target 30) →
   diagnostic `[]` and the MIP warning uses the "compete" wording; (f) overall target 0 with a
   forced negative contribution → `[]`; (g) bound equals the MIP constraint's coefficient sum
   (`_feature_groups` × contributions over allowed zones).
8. **Shared allowed-zones:** MIP built via `_allowed_zones` gives identical results on the
   existing zone MIP suite.
9. **Perf (comparative, per memory):** `compute_zone_objective` on the anchor after vs before
   ≤ 1.05× (median of repeats, `bench` marker); zone SA per-flip bench non-regressing.
10. **Anchors:** `tests/data/simple` MIP = 35.0, MarZone anchor results unchanged, `make check`.

## Compatibility

- **Behaviour changes (CHANGELOG, breaking for hand-built problems):** duplicate zone-target
  rows no longer raise `PulpError` in `ZoneMIPSolver` (last-wins + `ZoneTargetWarning`);
  unknown-id rows no longer make the MIP return `[]` nor add a phantom penalty;
  `metadata["zone_targets_met"]` and `write_zone_summary` no longer contain unknown-id or
  `target <= 0` pairs; a later `target = 0` row now switches a pair off.
- Loader-produced problems with clean files are unaffected. Public API additions:
  `ZonalProblem.zone_target_lookup`, `ZonalProblem.zone_target_issues`, `ZoneTargetWarning`,
  `diagnose_zone_infeasibility`, `compute_zone_achieved`.
- Version: minor bump (0.37.0).

## Out of scope

- MarZone `speciesid = -1` wildcard (reader keeps rejecting it).
- `ZONETARGET2NAME` / `ZONECONTRIB2NAME` / `ZONECONTRIB3NAME` silently ignored by the loader
  (review #13) — file a separate issue.
- Joint-infeasibility explanation (IIS / elastic re-solve, review #14).
- #6 part 2 (lock into a named zone) — `_allowed_zones` is its single extension point.
