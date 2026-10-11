# Zone PU locks (#6 part 2) — 4-lens design review synthesis

Reviewed: `2026-10-11-zone-pu-locks-design.md` rev 1 (`1eed927`). Workflow `wf_44186472-03d`:
architect, codebase-grounding (executed prototypes + anchor enumeration), MarZone parity (C++ @
`85082e3`: pu.hpp, reserve.hpp, marzone.cpp, solvers/*.hpp, test data), independent re-design;
CRITICAL/HIGH adversarially verified. 9 agents, 0 errors.

**Verdict:** core design confirmed by all lenses (two kw_only fields — verified to work on the
dataclass; one shared per-PU rule; MIP equality/≤ constraints; SA per-PU option draw keeps the
default RNG stream identical with no measurable overhead). Two HIGH findings change the spec.
Spec patched to rev 2.

## Verified facts

- Anchor enumeration: unrestricted (1,2,2)=7; PU1 locked to zone 2 → (2,1,2)=8, runner-up
  (2,2,1)=9, (2,2,2) infeasible. Prototype MIP agrees.
- Anchor + PU4 (no features, zone costs (3, 2)): unrestricted (1,2,2,0)=7; puzone PU4∈{1,2} →
  (1,2,2,2)=9. Current heuristic and II both return PU4 = 0 (catches the HIGH-1 gap).
- MarZone: pulock map last-wins; unknown PU ignored with a warning; unknown zone skipped
  silently; a single-zone puzone is **fatal** (`ShowErrorMessage` exits); MarZone does **not**
  read a pu.dat status column at all (`ReadPUData`, pu.hpp:641-700); files are `puid, zoneid`
  read positionally; loaded only when PULOCKNAME/PUZONENAME is set.
- No zone-MIP test inspects constraint names.

## Findings

| # | Sev | Lens | Finding | Disposition |
|---|---|---|---|---|
| 1 | **HIGH** (verified ×2) | architect, grounding, re-design | Heuristic starts PUs at 0 and only accepts improving moves; II seeds `zone_ids[0]`, swap pass hard-codes 0, `improve()` keeps incoming zones → a puzone PU can end at 0 or in a forbidden zone. | **Accepted.** Rule carries a start zone; every solver seeds must-assign PUs in an allowed zone; II candidates = allowed (+0 iff may-be-unassigned) in all passes; `improve()` repairs the incoming assignment first; shared `rule_violations()` asserted on every solver's output in tests. |
| 2 | **HIGH→MEDIUM** (verified) | re-design | Status 2 ranked above puzone forces `min(zone_ids)` even when puzone forbids it. | **Accepted.** Status 2 with puzone rows = allowed puzone set, must be assigned. |
| 3 | HIGH→MEDIUM (verified) | architect | `PUZoneRule` second bool flips meaning — positional tuples compare equal by accident. | **Accepted.** Field renamed `may_be_unassigned`; `forced` becomes a derived property; keyword construction; the two `TestRules` tests rewritten (listed). |
| 4 | MEDIUM | architect, re-design | Rule builder in `diagnostics` but `validate()` in `model` → two parsers + import cycle. | **Accepted.** `ZonalProblem.pu_lock_lookup()`, `pu_zone_lookup()`, `pu_zone_rules()`, `pu_lock_issues()` on the model; `diagnostics._pu_zone_rules` becomes a thin alias. |
| 5 | MEDIUM (P) | architect, grounding | Pulock last-wins must precede the unknown-zone filter (MarZone). | **Accepted** + test `(1,2),(1,99)` → PU1 falls through. |
| 6 | MEDIUM (P) | parity, re-design, architect | Readers must accept `puid, zoneid` (spaces), positional, headerless. | **Accepted.** Writers emit `puid,zoneid`; test on MarZone's `pulock_test1.dat` / `puzone_test1.dat` verbatim. |
| 7 | MEDIUM | parity, grounding | Spec misstates MarZone pu.dat status; single-zone puzone is fatal in MarZone; puzone not de-duplicated. | **Accepted** (reference section corrected; pymarxan lenient: lock + validate error + warning; distinct zones counted; duplicate puzone rows reported). |
| 8 | MEDIUM | architect | Status-1 start zone still parsed in SA → "only place that interprets status" false. | **Accepted.** Rule carries `start`. |
| 9 | MEDIUM | re-design | `validate()` flagged status 2 + pulock though they agree. | **Accepted.** Errors only for real conflicts. |
| 10 | MEDIUM | re-design | II swap candidates for free PUs must keep 0. | **Accepted** (folded into #1). |
| 11 | MEDIUM (P) | architect, parity | Default-name loading diverges from MarZone; explicit name with missing file silently skipped. | **Accepted:** default names kept (consistency with other zone files, documented); explicitly named missing file → `FileNotFoundError`. |
| 12 | LOW | architect, grounding | Constraint-name preservation justified by tests that don't exist. | **Accepted.** `assign_{pid}` with `==`/`<=`; disallowed zones via `upBound = 0`. |
| 13 | LOW | architect, re-design | SA temperature loop also needs per-PU options; bench baseline undefined. | **Accepted.** One proposal helper for both loops; seeded golden test (no locks → v0.37-identical); bench restricted vs unrestricted ≤ 1.05×. |
| 14 | LOW | architect | Warning classes. | **Accepted.** `ZoneInputWarning` base for `ZoneTargetWarning` + `ZoneLockWarning`; one dispatcher. |
| 15 | LOW | parity | MarZone II candidate filter has an index bug — must not be copied; CalcPenalties pre-counts pulocked PUs. | Noted in spec (not copied / out of scope). |
| 16 | LOW | grounding, re-design | Anchor tests that pass on a no-op. | **Accepted.** PU4 variant as the primary must-assign test. |
