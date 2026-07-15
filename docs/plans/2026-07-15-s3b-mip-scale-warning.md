# S3b — MIP-at-scale warning — spec + plan

**Date:** 2026-07-15
**Status:** Approved (brainstorm), streamlined (spec + TDD; no spec-review loop / design review —
a non-parity warning guard).
**Scope:** `solvers/mip_solver.py::MIPSolver` only. Third of the S3 pieces (S3a/S3c shipped).

## Motivation

The MIP solver creates one binary `LpVariable` per planning unit (plus boundary variables when
BLM > 0). At raster scale (hundreds of thousands to millions of PUs), pulp is slow just to *build*
the model, and CBC/HiGHS then choke or OOM (they don't scale to ~1e6 binary variables). S3b adds
an **early, actionable warning** so a user who points MIP at a huge (e.g. raster-ingested) problem
gets an immediate heads-up pointing to the heuristic solvers, rather than a silent multi-minute
build + OOM.

## Design

- New `MIPSolver` constructor param **`warn_above_pu: int | None = 200_000`** (after `objective`;
  consistent with the existing `mip_*` params). Stored as `self.warn_above_pu`.
- A module-level helper `_warn_if_large_mip(n_pu, threshold)`: when `threshold is not None and
  n_pu > threshold`, `warnings.warn(...)`. (Module-level so the zone/river MIP solvers can adopt
  it later; wired into core `MIPSolver` only for S3b.)
- Called at the **top of `solve()`** (right after `config` setup, before the strategy gating and
  the model build), so the warning fires before any expensive work.
- **Message (actionable):** names `n_pu` + the threshold, explains CBC/HiGHS don't scale to ~1e6
  binary vars (long build + likely OOM; BLM adds boundary vars on top), suggests the `sa` /
  `greedy` / `zonation` solvers, and notes `warn_above_pu=None` silences it.
- **No raise, no auto-switch** — warn and proceed (the caller's explicit choice). The suggestion
  to use heuristics is advice in the message, not a forced route.

Threshold rationale: normal vector problems rarely exceed ~100k PUs → no spurious warnings; a
raster-scale problem does. Overridable (raise/lower) or silenceable (`None`).

`warnings` is imported into `mip_solver.py` (new import).

## Tests (TDD)

- **Sub-threshold → no warning:** the tiny fixture (6 PUs) with the default threshold solves with
  no warning (`warnings.catch_warnings(record=True)` → none matching).
- **Over-threshold → warns AND still solves:** `MIPSolver(warn_above_pu=2)` on the tiny fixture
  (6 PUs > 2) emits a `UserWarning` (message mentions a heuristic, e.g. matches `"sa"`/`"greedy"`/
  `"heuristic"`) **and** still returns a valid `Solution` (proceeds — warn, not raise).
- **`warn_above_pu=None` → silent** even over any size.
- **Anchor:** `make check` green; the 35.0 MIP parity is untouched (guard only warns).

## References

MIP solver: `solvers/mip_solver.py` (one binary var per PU + boundary vars for BLM). Scoping:
`2026-07-15-raster-grid-pus-scoping.md` (S3b). Related (future adopters): zone/river MIP.
