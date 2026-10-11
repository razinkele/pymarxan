# Zone PU locks and zone restrictions (pulock + puzone) — design

Issue: razinkele/pymarxan#6 part 2 ("no way to lock a PU into a named zone"). Status: rev 1,
design approved in chat 2026-10-11 (scope "pulock + puzone"; puzone PUs may NOT be unassigned).
Pending 4-lens review.

## Problem

Zone solvers support only pymarxan's status convention: status 2 forces a PU into the first
sorted zone, status 3 forces it out of every zone (`diagnostics._pu_zone_rules`, plus separate
parsers in `solver.py`, `heuristic.py`, `iterative_improvement.py`). There is no way to say
"this existing restoration site must be in the *restore* zone" (the ROWER need), nor to restrict
a PU to a subset of zones. MarZone provides both through two input files.

## MarZone reference semantics (`Marxan-source-code/marzone` @ `85082e3`)

- **`pulock.dat`** (`puid,zoneid`; `PULOCKNAME` in input.dat). `pu.hpp:708-750` `LoadPuLock`:
  header line skipped; rows for a PU not in pu.dat are ignored with a warning; stored in a map,
  so a repeated PU keeps its **last** row. `pu.hpp:483-496` `PersistPuLock`: a lock naming an
  unknown zone is skipped; otherwise `fPULock = 1`, `numZones = 1`.
- **`puzone.dat`** (`puid,zoneid`, one row per allowed zone; `PUZONENAME`). `pu.hpp:752-790`
  `LoadPuZone` (unknown PU ignored with a warning); `pu.hpp:498-526` `PersistPuZone`: unknown
  zones skipped; a PU already pulocked keeps its lock; a PU left with exactly one allowed zone
  is reported as an **error** ("Do not use this file to lock planning units!").
- **Precedence.** `reserve.hpp:100-103`: "pu lock takes precedence above all." A puzone PU
  always takes one of its listed zones (`pu.hpp:286-311` `RtnValidZoneForPu`; MarZone has no
  unassigned state). `pu.dat` status is only a starting zone in MarZone (`reserve.hpp:108-109`).

## Design

### 1. Data model and I/O

- `ZonalProblem` gains two optional fields, `pu_locks: pd.DataFrame | None = None` and
  `pu_zones: pd.DataFrame | None = None` (`field(default=None, kw_only=True)` per CLAUDE.md),
  both with columns `pu, zone`. `copy_with` carries them.
- `zones/readers.py`: `read_pu_lock(path)` and `read_pu_zone(path)` (CSV/whitespace like the
  other zone readers; integer `pu`, `zone`). `load_zone_project` reads `PULOCKNAME` (default
  `pulock.dat`) and `PUZONENAME` (default `puzone.dat`) when the file exists, as it does for
  `zonetarget.dat`.
- `zones/writers.py`: `write_pu_lock(df, path)`, `write_pu_zone(df, path)` (round-trip).
- Exported from `pymarxan.zones`.

### 2. One per-PU rule

`diagnostics.PUZoneRule` becomes `PUZoneRule(allowed: frozenset[int], unassigned_ok: bool)`
(replacing v0.37's `(allowed, forced)`; the type is not in `__all__`). A rule is *forced* when
`len(allowed) == 1 and not unassigned_ok`.

| Source (highest precedence first) | `allowed` | `unassigned_ok` |
|---|---|---|
| `pu_locks` row (known PU, known zone; last row wins) | `{zone}` | False |
| status 2 (locked in) | `{min(zone_ids)}` | False |
| status 3 (locked out) | `∅` | True |
| `pu_zones` rows (known zones only; ≥ 1 left) | listed zones | **False** |
| otherwise (free) | all zones | True |

Precedence follows MarZone for the two files (pulock above everything); pymarxan's status locks
rank above puzone because in pymarxan they are locks, not starting zones. A PU whose puzone rows
name only unknown zones is free.

`_pu_zone_rules(problem)` builds this list in `planning_units` order. It is the **only** place
that interprets status / pulock / puzone; every solver and the diagnostic consume it.

### 3. Validation and warnings

`ZonalProblem.validate()` adds errors (never raises):
- `pu_locks` / `pu_zones` missing `pu`/`zone` columns;
- rows naming an unknown PU or unknown zone;
- a PU repeated in `pu_locks` ("resolved last-wins");
- a PU with both a pulock and status 2 or 3, or with puzone rows and status 2 or 3
  ("pulock / status takes precedence");
- a PU whose `pu_zones` rows leave exactly one known zone (MarZone's error; solvers treat it as
  a lock — use `pu_locks` instead).

Each zone solver emits one `ZoneLockWarning(UserWarning)` per call (beside the existing
`ZoneTargetWarning`) when any of the above row-level issues would change what the solver does
(skipped rows, last-wins, precedence overrides). Shared detection helper, like
`zone_target_issues`.

### 4. Solvers

- **MIP** (`mip_solver.py`): for each PU, `x[i,z] = 0` for `z ∉ allowed`
  (`locked_out_{pid}_{zid}`); if not `unassigned_ok`: `Σ_{z} x[i,z] == 1` (`assign_{pid}`
  becomes an equality) — for a forced PU this is exactly today's `locked_in_{pid}`; keep that
  name for forced PUs so existing tests that inspect names still hold. Otherwise `≤ 1`.
- **SA** (`solver.py`): per-PU option array = `sorted(allowed)` (SA never proposes zone 0 for a
  free PU today — `zone_options` holds zone ids only — and this is unchanged). PUs with exactly
  one option and `not unassigned_ok`, or with `allowed = ∅`, are not swappable (assigned their
  forced zone / 0). The move step draws a swappable PU, then a zone from that PU's options;
  for free PUs (all zones) it keeps today's draw from `zone_options`, so the default path is
  unchanged. Initial assignment: random option; status-1 PUs start in their first allowed zone.
- **Heuristic / II**: candidate zones per PU = `sorted(allowed)`; II's removal pass (move to 0)
  skips PUs with `unassigned_ok = False`; locked/forced PUs are not swappable. Their private
  status parsers are replaced by `_pu_zone_rules`.
- **Diagnostic**: `_best(rule, values)` = `max(values over allowed)` combined with `0` when
  `unassigned_ok` (so: forced → its value as-is; free → `max(0, best)`; puzone →
  `max over listed zones`, possibly negative). Unchanged results for today's statuses.

### 5. Performance

SA per-flip cost on problems without locks/restrictions must not regress (comparative bench,
≤ 1.05×). Per-PU option arrays are built once per solve.

## Testing

1. **Rules table:** each source and every precedence pair; last-wins pulock; unknown PU/zone
   rows ignored; puzone with only unknown zones → free.
2. **I/O:** `read_pu_lock`/`read_pu_zone` ↔ writers round-trip; `load_zone_project` picks up
   `PULOCKNAME`/`PUZONENAME`; absent files → `None`.
3. **Anchors (MarZone anchor, hand-derived; the grounding review must confirm by running):**
   pulock PU1 → zone 2 moves the MIP optimum from `(1, 2, 2)` at 7.0 to `(2, 1, 2)` at 8.0
   (overall 4+10+4 = 18 ≥ 15, zone 2 = 20 ≥ 10; runner-up `(2, 2, 1)` at 9.0; `(2, 2, 2)` is
   infeasible, overall 12). puzone PU1 ∈ {2} gives the same optimum as that lock (single-zone
   puzone = lock); puzone PU1 ∈ {1, 2} leaves `(1, 2, 2)` at 7.0. puzone PU3 ∈ {1, 2}
   forbids leaving PU3 unassigned (checked on a variant whose unrestricted optimum leaves a PU
   at 0).
4. **All four solvers respect every rule** on every returned solution (locks, restrictions,
   status), on a seeded small problem.
5. **validate()** errors and the one-warning-per-call `ZoneLockWarning`.
6. **Regression:** existing zone suites unchanged; `tests/data/simple` MIP = 35.0; MarZone
   anchor optimum `(1, 2, 2)` at 7.0 without locks; zone SA per-flip bench.

## Compatibility

Additive: two optional fields, new readers/writers/warning; behaviour without pulock/puzone is
unchanged. `PUZoneRule`'s second field changes meaning (`forced` → `unassigned_ok`) — internal.
Version: minor bump (0.38.0).

## Out of scope

- MarZone's per-PU zone *costs* overrides and `pu.dat` status as a starting zone.
- The `-1` / multi-file variants (`ZONETARGET2NAME` etc.) — separate issue.
