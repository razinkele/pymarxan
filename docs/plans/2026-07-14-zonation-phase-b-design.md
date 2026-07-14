# Zonation Phase B — `ZonationSolver` adapter design

**Date:** 2026-07-14
**Status:** Approved (brainstorm), pending implementation plan + review
**Scope:** Phase B only — a thin `Solver`-ABC adapter over the Phase A
`rank_removal` engine, plus registry registration. Phases C (distribution
smoothing) and D (Shiny panel, incl. solver-picker wiring) remain deferred.

## Motivation

Phase A shipped `pymarxan.zonation.rank_removal` (a plain function returning a
`ZonationResult`). Phase B lets Zonation plug into the existing solver framework
(`Solver.solve(problem, config) -> list[Solution]`, the registry, and any
pipeline that consumes solvers) by wrapping the engine in a `Solver` subclass
that thresholds the priority ranking into a binary reserve.

## Design decision (approved) — one deterministic reserve

Zonation is deterministic: a problem + config yields exactly one priority
ranking, and thresholding it gives one reserve. `solve()` therefore returns a
**length-1 `list[Solution]`** (the top `top_fraction` of the ranking), regardless
of `config.num_solutions` — documented, not an error. Rationale: the ranking is
the product; the full `ZonationResult` (rank map + curves) rides in
`Solution.metadata`, so a caller wanting reserves at other thresholds calls
`result.top_fraction(f)` in one line. Padding with identical copies would fake
variety to downstream analysis (selection frequency, portfolios); a
nested-threshold portfolio would overload `num_solutions` with a
Zonation-specific meaning it has nowhere else. The thin adapter is the honest
mapping.

## Module layout

```
src/pymarxan/solvers/zonation_solver.py   # ZonationSolver(Solver)
src/pymarxan/solvers/registry.py          # + register("zonation", ...)
tests/pymarxan/solvers/test_zonation_solver.py
```

## Component — `ZonationSolver(Solver)`

```python
ZonationSolver(
    *,
    rule: str = "caz",             # "caz" | "abf"  (validated at __init__)
    top_fraction: float = 0.3,     # share of the landscape selected as reserve
    warp: int = 1,
    weights: dict[int, float] | None = None,
    use_cost: bool = True,
)
```

**`__init__`** stores the params and validates `rule in ("caz", "abf")` (fail
fast, matching `MIPSolver`'s objective validation) and `0 < top_fraction <= 1`.

**`solve(problem, config=None) -> list[Solution]`:**
1. `result = rank_removal(problem, rule=self.rule, weights=self.weights,
   warp=self.warp, use_cost=self.use_cost)`.
2. `selected_ids = result.top_fraction(self.top_fraction)`.
3. Build a boolean `selected` array aligned to `planning_units` order:
   `np.array([int(pid) in selected_ids for pid in
   problem.planning_units["id"]], dtype=bool)`.
4. `blm = float(problem.parameters.get("BLM", 0.0))` (the codebase convention,
   `heuristic.py:282`).
5. `meta = {"solver": "zonation", "rule": self.rule, "top_fraction":
   self.top_fraction, "priority_rank": result.priority_rank,
   "performance_curves": result.performance_curves}`.
6. `return [build_solution(problem, selected, blm, metadata=meta)]` —
   `build_solution` (`solvers/utils.py:463`) fills `cost`, `boundary`,
   `objective`, `targets_met`, `penalty`, `shortfall` from the selection.
   `config` is accepted for ABC compatibility; `num_solutions` is not used
   (deterministic → one solution).

**`name()`** → `"Zonation (rank-removal)"`.
**`supports_zones()`** → `False`.
**`available()`** → `True`.

## Registry

In `get_default_registry()` (`solvers/registry.py`), add the local import and
`reg.register("zonation", ZonationSolver)` alongside the existing seven, so
`registry.get("zonation")` returns a fresh `ZonationSolver`.

## Testing strategy (TDD)

Reuse the Phase A `_problem` helper shape (dense matrix → `ConservationProblem`).
Reference: **P1** `q=[[10,0],[0,10],[5,5]]`, uniform cost, CAZ ranking gives
`priority_rank` PU3=1/3, PU1=2/3, PU2=1.0.

- **Thresholded reserve is hand-known.** `ZonationSolver(top_fraction=2/3)` on P1
  → `top_fraction(2/3)` = `ceil(2)` = the 2 highest-ranked = `{PU1, PU2}` →
  `selected == [True, True, False]`; `sol.cost == 2.0`.
- **Metadata carries the ranking.** `sol.metadata["priority_rank"]` equals the
  engine's rank map; `sol.metadata["performance_curves"]` is the curves
  DataFrame; `sol.metadata["solver"] == "zonation"`.
- **`build_solution` populated the Solution.** `targets_met` is a non-empty dict
  and (with achievable targets) all True; `cost`/`objective` are finite.
- **`top_fraction` controls reserve size.** A larger `top_fraction` selects at
  least as many PUs (monotone); `top_fraction=1.0` selects all.
- **Deterministic — one solution.** `solve(p, SolverConfig(num_solutions=5))`
  returns a list of length 1.
- **ABC surface.** `name()`, `supports_zones() is False`, `available() is True`.
- **Registry.** `get_default_registry().get("zonation")` is a `ZonationSolver`.
- **Validation.** `ZonationSolver(rule="bogus")` and
  `ZonationSolver(top_fraction=0)` raise `ValueError` at construction.

**Target:** ~9–11 tests, `make check` green (0 ruff / 0 mypy), coverage ≥ 75%.

## Out of scope (YAGNI, Phase B)

- Distribution smoothing (Phase C — reuse `connectivity.smoothing`).
- Shiny panel and solver-picker UI wiring (Phase D).
- A nested-threshold portfolio mode (rejected above; callers slice the ranking
  in `Solution.metadata` themselves).
- Zone support (`supports_zones()` is `False`).

## Parity note

This adds no Marxan-solver or objective math — `build_solution` is the shared,
already-tested Solution builder, and the reserve is chosen entirely by the
Phase A ranking. The 35.0 min-set ground-truth anchor is untouched; a quick
`marxan-parity-check` after `make check` confirms no regression.
