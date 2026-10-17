# SA optimises connectivity (v0.44.0) — design

**Date:** 2026-10-17 · **Status:** approved; amended by the 4-lens review (`2026-10-17-sa-connectivity-review.md`, rulings R1–R6)
**Follows:** v0.43.0, which aligned `compute_connectivity_penalty` with the MIP's connectivity
term (`docs/plans/2026-10-17-stage4-proportion-decisions-design.md`, as-built amendment).

## 1. Problem

Simulated annealing (SA) and iterative improvement (II) score moves through
`ProblemCache.compute_full_objective` / `compute_delta_objective`
(`src/pymarxan/solvers/cache.py`). Neither has a connectivity term. So with
`CONNECTIVITY_WEIGHT > 0`:

- SA and II search as if connectivity did not exist;
- `Solution.objective` (from `build_solution`) includes connectivity, so SA's internal
  `current_obj` / `best_obj` and its history disagree with the objective it reports;
- only the MIP actually optimises connectivity.

`CONNECTIVITY_WEIGHT` is a pymarxan/prioritizr extension, not a Marxan C++ parameter, so
this is consistency with the MIP and the reported objective, not Marxan parity.

## 2. Decisions (user-approved)

| Decision | Choice |
|---|---|
| Scope | Connectivity only. The PROBMODE 1 risk premium gap (same cache, verified: `simple` + PROBMODE 1, cache 44.0 vs reported 69.2) goes to the backlog. |
| Approach | **A**: connectivity adjacency arrays inside `ProblemCache`, consumed by `compute_full_objective` and `compute_delta_objective`. Not a `ConnState` side object (stateless pairwise math; would need six call-site edits) and not folded into boundary (opposite semantics, different weight). |
| Semantics | Exactly `compute_connectivity_penalty` (v0.43.0), which equals the MIP term. |
| Greedy heuristics | Unchanged: HEURTYPE 0–7 rank by rarity / irreplaceability and never use the objective. |
| Zones | Unchanged: `ZoneSA` has its own cache and connectivity term. |

## 3. Semantics (binary selections; `w = CONNECTIVITY_WEIGHT`)

Connectivity is **off** (term 0, no arrays consulted) when `problem.connectivity is None` or
`w <= 0`. Rows whose `id1` or `id2` is not a planning unit are skipped.

**Symmetric** (`ASYMMETRIC_CONNECTIVITY` absent or 0):
`C(x) = −w · Σ_{pairs {i,j}} v_ij · [x_i ∧ x_j]`.
Pairs are de-duplicated by unordered key; the first row of `(i, j)` / `(j, i)` wins.
A self-pair `(i, i)` rewards `v` when `i` is selected (`min(x_i, x_i) = x_i`).

**Asymmetric** (`ASYMMETRIC_CONNECTIVITY = 1`):
`C(x) = w · Σ_{rows (i→j)} v_ij · [x_i ∧ ¬x_j]`; duplicate `(i, j)` rows add up.
A self-row `(i, i)` contributes 0 always and is dropped.

## 4. Cache layout

New `ProblemCache` fields, built once in `from_problem`, appended **with defaults** so any
direct construction keeps working:

| Field | Meaning |
|---|---|
| `conn_mode: int` | 0 off, 1 symmetric, 2 asymmetric |
| `conn_weight: float` | `w` (0.0 when off) |
| `conn_self: (n_pu,) float64` | symmetric self-pair rewards `v_ii` (zeros otherwise) |
| `conn_out_start/idx/val` | CSR. Symmetric: each de-duplicated pair stored in both directions. Asymmetric: out-edges `i→j`, duplicates summed |
| `conn_in_start/idx/val` | CSR in-edges `k→i` (asymmetric only; empty otherwise) |

Building is vectorised or a single pass over the connectivity rows (no per-flip work).

## 5. Objective and delta

`compute_full_objective` adds `C(selected)` computed from the arrays.

`compute_delta_objective(idx, …)` adds, with `s = +1` when adding and `−1` when removing,
and `sel` the pre-flip selection:

- symmetric: `−s · w · (conn_self[idx] + Σ_out v · sel[nbr])`
- asymmetric: `s · w · Σ_out v · (1 − sel[j])  −  s · w · Σ_in v · sel[k]`

Both are O(degree). When `conn_mode == 0` the only cost is one integer comparison.

## 6. Effects

- SA and II plans change on problems with `CONNECTIVITY_WEIGHT > 0`; no other plan changes.
  The 35.0 anchor and every golden without connectivity are unaffected (no golden has an SA
  connectivity case; verify at implementation).
- SA's `current_obj`, `best_obj` and history equal `build_solution(...).objective` with
  connectivity (absent other known gaps: PROBMODE 1, clumping/separation/space handled by
  their states as today).
- `RUNMODE` combinations that chain SA → II inherit the change.

## 7. Testing

- **Cache unit tests** (`tests/pymarxan/solvers/test_cache.py` or a new
  `test_cache_connectivity.py`): array contents for symmetric (dedupe, both directions,
  self-pair), asymmetric (duplicates summed, self-row dropped, in/out lists), off cases
  (`None`, `w = 0`, `w < 0`), unknown ids skipped.
- **Full = evaluator:** `compute_full_objective` minus the no-connectivity value equals
  `compute_connectivity_penalty` on ≥ 20 seeded random selections, symmetric and asymmetric,
  with duplicate and reversed rows and self-pairs in the fixture.
- **Delta exactness:** for every PU and both directions, `compute_delta_objective` equals the
  difference of `compute_full_objective` before and after the flip (abs 1e-9).
- **SA consistency:** a seeded SA run on a connectivity problem (no clumping, separation,
  space or PROBMODE 1): the reported `Solution.objective` equals
  `metadata["history"]["best_objective"][-1]` to 1e-9 (`metadata["best_objective"]` is
  rounded to 4 places), and the plan's connectivity term is non-zero (without that check the
  symmetric case passed on the old code: its seed-42 plan had connectivity 0). Both modes fail
  on the old code; it is the headline regression test.
- **SA behaviour:** on the `test_connectivity_mip` fixture (strong symmetric 1–2 link), seeded
  SA selects both PUs 1 and 2; asymmetric fixture: never PU 1 without PU 2.
- **II:** iterative improvement's removal/addition passes see the term (a fixture where
  removing a unit would break a strong connection keeps it).
- **Bench** (`tests/benchmarks/bench_sa.py`, `bench` marker, comparative only): SA per-flip
  time with connectivity off ≤ 1.05 × a baseline measured in the same test; with a
  degree-~4 connectivity graph, report only (no absolute budget).
- `make check` green; parity harness (`examples/validate_marxan_parity.py`) unchanged.

## 8. Docs

CHANGELOG `### Changed` (SA/II results change on connectivity problems; v0.44.0, minor);
`docs/USER_MANUAL.md` connectivity section ("SA and the heuristics report but do not
optimise" becomes "the MIP, SA and iterative improvement optimise it; the greedy
heuristics do not"); roadmap backlog: remove the SA item, add "SA/II ignore the PROBMODE 1
risk premium (cache 44.0 vs reported 69.2 on `simple`)"; `cache.py` module docstring
mentions the connectivity arrays.

## 9. Out of scope

PROBMODE 1 premium in the cache (backlog); greedy heuristics; zone solvers; fractional
selections (SA is binary-only).
