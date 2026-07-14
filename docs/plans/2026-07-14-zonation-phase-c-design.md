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
  `distances` is provided (neither or both → `ValueError`); and — since it's
  knowable at construction — `coords`, when given, must be **2-D** (a 1-D array
  raises here, fail-fast, not deep in a solve).
- **`resolve_distances(n_pu: int) -> np.ndarray`:** if `distances` is set,
  validate its shape is `(n_pu, n_pu)` and return it; else validate
  `coords.shape[0] == n_pu` and return `distance_matrix_from_points(coords)`
  (`connectivity.smoothing`). Only the row *count* is checkable here.
- **`apply(q: np.ndarray) -> np.ndarray`:** one call —
  `smooth_distribution(q, resolve_distances(q.shape[0]), self.alpha)`. Because
  the kernel is amount-independent, `smooth_distribution`'s `Knorm @ amounts`
  generalizes to a 2-D `(n_pu, n_feat)` right-hand side, smoothing every feature
  column in a single kernel build (no per-column loop). `normalize=True` is fixed
  (mass-conserving is what "smoothing" means here; exposing it is YAGNI). This
  requires a one-line note on `smooth_distribution`'s docstring stating it also
  accepts a 2-D array (smooths each column).

**Row-order contract (important).** `coords`/`distances` rows **must** be aligned
to `problem.planning_units` *positional* order — the same order
`build_pu_feature_matrix` uses for `q`'s rows (and that `cost`/`status` follow).
`resolve_distances` can validate only the row *count*, not the order, so a
correctly-sized but mis-ordered `coords` (e.g. built sorted-by-id while the
DataFrame isn't) silently produces a **wrong ranking**. This requirement is
stated in the `SmoothingSpec` docstring. (A future `from_problem(problem, alpha)`
classmethod deriving centroids in `planning_units` order would make this safe by
construction — deferred, needs geometry + CRS handling.)

## `rank_removal` integration

Add `smoothing: SmoothingSpec | None = None` to `rank_removal`. Immediately after
`q = problem.build_pu_feature_matrix()`:

```python
if smoothing is not None:
    q = smoothing.apply(q)
```

Everything downstream is unchanged: `Q`/`T`, the CAZ/ABF loop, cost/status/warp,
and the performance curves all use the smoothed `q`. **The performance curves
therefore report retention of the *smoothed* distribution** — Zonation treats the
smoothed layer as the working distribution. Note the resulting intra-`Solution`
asymmetry a caller should be aware of (documented on `ZonationResult`): with
smoothing on, `performance_curves` are smoothed-layer retention while a
`Solution`'s `targets_met` (from `build_solution` on the raw problem) reflect the
*raw* distribution. A raw-retention curve column is a deferred future add (YAGNI).

Smoothing is **status-blind**: the kernel spreads amount into and out of
locked-out cells (a locked-out cell's habitat can lift its neighbors, and its
inherited smoothed mass is stripped first since it's removed first). This is
defensible — smoothing is a landscape operation, locks are a planning constraint
enforced later — and noted in the docstring.

## `ZonationSolver` passthrough

Add `smoothing: SmoothingSpec | None = None` to `ZonationSolver.__init__` (stored)
and pass it to `rank_removal` in `solve()`. Also record a **provenance marker** in
`Solution.metadata` — `"smoothed": self.smoothing is not None` and
`"smoothing_alpha": self.smoothing.alpha if self.smoothing else None` — so a
downstream consumer can tell a smoothed run from a raw one and recover `alpha`
(the `SmoothingSpec` itself isn't JSON-friendly, so just the boolean + alpha).

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
- **`n_pu = 1` edge:** a single-PU problem smooths to itself (kernel `[[1]]`),
  no crash.
- The order-flip test carries a comment that `[3,2,1]` relies on the test
  helper's uniform cost.

**Target:** ~10 tests (8 unit + 2 integration), `make check` green (0 ruff /
0 mypy), coverage ≥ 75%.

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
