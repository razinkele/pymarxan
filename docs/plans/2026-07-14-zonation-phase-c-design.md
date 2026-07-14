# Zonation Phase C — distribution smoothing design

**Date:** 2026-07-14
**Status:** Approved (brainstorm), pending implementation plan + review
**Scope:** Phase C only — an optional distribution-smoothing pre-step for the
Zonation ranking, reusing the existing `connectivity.smoothing`. Phase D (Shiny
panel + solver-picker wiring) remains deferred.

## Motivation

Zonation's signature spatial step is *distribution smoothing*: spread each
feature's amount to nearby planning units via a dispersal kernel, so the ranking
values *being near* abundance (spatial aggregation / boundary quality), not only
holding it. pymarxan already ships this exact operation —
`connectivity.smoothing.smooth_distribution` is documented as "the planning-unit
(vector) analogue of Zonation's distribution smoothing." Phase C wires it into
the Zonation engine as an **optional pre-transform of the feature matrix `q`**,
so the whole Phase A ranking machinery runs unchanged on the smoothed layer.

## Key insight — smoothing is a one-time pre-transform

Smooth each feature column of `q` once (right after
`problem.build_pu_feature_matrix()`), then the entire ranking loop (CAZ/ABF,
cost/status/warp) and the performance curves run on the smoothed `q` with **no
other change**. The hard part (the mass-conserving kernel) is already built and
tested in `connectivity.smoothing`; Phase C is one small config dataclass + one
optional param threaded through two call sites.

## Module layout

```
src/pymarxan/zonation/smoothing.py   # SmoothingSpec dataclass
src/pymarxan/zonation/rank_removal.py   # + smoothing param (pre-transform)
src/pymarxan/solvers/zonation_solver.py # + smoothing passthrough
src/pymarxan/zonation/__init__.py       # export SmoothingSpec
tests/pymarxan/zonation/test_smoothing.py
```

## Component — `SmoothingSpec` (`smoothing.py`)

A small config dataclass bundling the smoothing parameters:

```python
@dataclass(eq=False)   # eq=False: numpy-array fields make the auto __eq__
class SmoothingSpec:   # raise "ambiguous truth value"; identity eq + hashable is fine
    alpha: float
    coords: np.ndarray | None = None      # (n_pu, d) PU coordinates
    distances: np.ndarray | None = None   # (n_pu, n_pu) precomputed distances
```

- **Validation** (`__post_init__`): `alpha > 0` (else `ValueError`; also
  `negative_exponential` guards this downstream); **exactly one** of `coords` /
  `distances` is provided (neither or both → `ValueError`).
- **`resolve_distances(n_pu: int) -> np.ndarray`:** if `distances` is set,
  validate its shape is `(n_pu, n_pu)` and return it; else validate `coords` is
  2-D with `coords.shape[0] == n_pu` (`cdist` requires 2-D; a 1-D array must
  raise a clear `ValueError`, not a cryptic scipy error) and return
  `distance_matrix_from_points(coords)` (`connectivity.smoothing`).
- **`apply(q: np.ndarray) -> np.ndarray`:** resolve distances from `q.shape[0]`,
  then smooth each feature column —
  `smooth_distribution(q[:, j], distances, self.alpha)` (default
  `normalize=True`, mass-conserving) — returning a new `(n_pu, n_feat)` matrix.
  Reuses `smooth_distribution` per column (DRY); the smoothing runs once as a
  pre-step, not per removal, so the per-column kernel rebuild is acceptable at
  the vector-PU scale Zonation targets.

`normalize=True` is fixed (mass-conserving is what "smoothing" means here;
exposing it is deferred as YAGNI).

## `rank_removal` integration

Add `smoothing: SmoothingSpec | None = None` to `rank_removal`. Immediately after
`q = problem.build_pu_feature_matrix()`:

```python
if smoothing is not None:
    q = smoothing.apply(q)
```

Everything downstream is unchanged: `Q`/`T`, the CAZ/ABF loop, cost/status/warp,
and the performance curves all use the smoothed `q`. **The performance curves
therefore report retention of the smoothed distribution** — faithful to
Zonation, which replaces the input layer with the smoothed one (a raw-retention
view is a possible future add, out of scope here).

## `ZonationSolver` passthrough

Add `smoothing: SmoothingSpec | None = None` to `ZonationSolver.__init__` (stored)
and pass it to `rank_removal` in `solve()`. No other change.

## Testing strategy (TDD, hand-verifiable)

- **A point mass spreads monotonically with distance.** On a 1-D line
  (`coords=[[0],[1],[2]]`), a feature peaked entirely on PU1 (`q[:,0]=[10,0,0]`),
  `SmoothingSpec(alpha, coords).apply(q)` gives `smoothed[:,0] = a,b,c` with
  `a > b > c > 0` (the kernel `[1, e^-α, e^-2α]` normalized), verifying the
  neighbor-lift behavior directly. (NB: a *uniform* layer does **not** stay
  uniform — the column-normalized kernel isn't doubly stochastic, so edge vs.
  interior cells redistribute; only the total is invariant. Do not test
  "uniform unchanged".)
- **Total conserved.** For each feature, `sum(smoothed) ≈ sum(raw)` (the
  `normalize=True` invariant, from `smooth_distribution`) — assert via
  `SmoothingSpec.apply`.
- **Smoothing changes the ranking.** On a peaked problem, `rank_removal` with
  vs. without a `SmoothingSpec` produces a different `removal_order` — and in the
  expected direction (the peak's near-neighbor out-ranks the far cell once it
  inherits smoothed value).
- **`distances` vs `coords` agree.** A `SmoothingSpec(alpha, distances=D)` with
  `D = distance_matrix_from_points(coords)` yields the identical ranking to
  `SmoothingSpec(alpha, coords=coords)`.
- **Validation:** `alpha <= 0` raises; neither `coords` nor `distances` raises;
  both raise; a `distances` of wrong shape (or `coords` with wrong row count)
  raises in `resolve_distances`.
- **`ZonationSolver` passthrough:** `ZonationSolver(smoothing=spec, top_fraction=f)`
  selects the same PUs as `rank_removal(problem, smoothing=spec)` then
  `top_fraction(f)` (with status enforcement).

**Target:** ~10–12 tests, `make check` green (0 ruff / 0 mypy), coverage ≥ 75%.

## Out of scope (YAGNI, Phase C)

- Exposing `normalize=False` (accumulating) smoothing.
- Raw-distribution performance curves alongside the smoothed ones.
- Non-Euclidean / graph distances (the boundary matrix as a distance source).
- Shiny panel + solver-picker wiring (Phase D).

## Parity note

Adds no Marxan-solver/objective math; the smoothing is a pre-transform reusing
the already-tested `connectivity.smoothing` kernel. The 35.0 min-set anchor is
untouched; a quick `marxan-parity-check` after `make check` confirms no
regression.

## Reference

Reuses `connectivity.smoothing` (the vector analogue of Zonation's distribution
smoothing). Zonation smoothing lineage: Moilanen et al. 2005 / Lehtomäki &
Moilanen 2013 (scite-verified in Phase A).
