# Zonation Phase C design review — synthesis

**Date:** 2026-07-14
**Reviewed:** `2026-07-14-zonation-phase-c-design.md` + `...-implementation.md`
**Method:** three perspectives (architect, codebase-grounding [ran the engine +
ruff/mypy], independent re-design). Scientific dimension folded into grounding —
no new algorithm.

## Verdict

**Approve after folding in the findings.** No CRITICAL. The grounding agent
empirically confirmed both reference oracles (`[2,3,1]`→`[3,2,1]`), mass
conservation, the uniform-redistributes reasoning, and ruff/mypy-clean. The
independent re-designer converged on the plan's core shape. The findings are one
HIGH doc gap plus cheap MEDIUM refinements.

## Findings absorbed

### HIGH — document the coords/distances → PU positional-order contract
`build_pu_feature_matrix` orders `q` rows by `planning_units` *positional* order
(`problem.py:90,108`); `status`/`cost` are read the same way. So `coords[i]` /
`distances[i,j]` **must** align to `planning_units.iloc[i]`. `resolve_distances`
can validate only the row *count*, not the order — a correctly-sized but
mis-ordered `coords` (e.g. built sorted-by-id while the DataFrame isn't) passes
and silently produces a wrong ranking. **Fix (doc-only, both agents' top item):**
state the positional-alignment requirement in the `SmoothingSpec`/`apply`
docstring and the design; note only the count is checked.

### MEDIUM — collapse `apply` to a single `smooth_distribution` call
`smooth_distribution` is `Knorm @ amounts` with an amount-independent kernel, so
it already handles a 2-D `q` in one call — building the kernel once instead of
`n_feat` times. **Fix:** `apply` becomes
`return smooth_distribution(np.asarray(q, float), self.resolve_distances(q.shape[0]), self.alpha)`,
and add a one-line note to `smooth_distribution`'s docstring that it accepts a
2-D `(n, m)` array (smooths each column). Simpler + faster; makes the 2-D
contract explicit rather than relying on undocumented behavior.

### MEDIUM — eager `coords.ndim == 2` check in `__post_init__`
A 1-D `coords` is currently only caught deep in `resolve_distances` (at solve
time). `ndim` is knowable at construction, so check it in `__post_init__` to fail
fast at the call site; keep the row-*count* check lazy (needs `n_pu`). Update the
1-D-coords test to expect a construction-time raise.

### MEDIUM — record a smoothing marker in `Solution.metadata`
`ZonationSolver.solve` metadata carries `rule`/`top_fraction`/`priority_rank`/
`performance_curves` but nothing saying smoothing was applied — a downstream
consumer can't tell a smoothed run from a raw one or recover `alpha`. **Fix:** add
`"smoothed": self.smoothing is not None` and `"smoothing_alpha":
self.smoothing.alpha if self.smoothing else None` (the `SmoothingSpec` itself
isn't JSON-friendly, but the boolean + alpha closes the provenance gap).

### MEDIUM — document the smoothed-curves semantics (defer dual-column)
With smoothing on, `performance_curves` report retention of the *smoothed* layer,
while a `Solution`'s `targets_met` (from `build_solution`) reflects the *raw*
distribution — an intra-`Solution` inconsistency a user could trip over. Adding a
raw retention column is ~5 lines but expands the `ZonationResult` schema. **Fix
(Phase C):** document the smoothed-vs-raw semantics on `ZonationResult` /
`rank_removal`, and **soften** the "faithful to Zonation" claim (asserted without
a source) to "treats the smoothed layer as the working distribution." Dual-column
curves deferred (YAGNI).

### LOW — absorbed
- Comment in the order-flip test that `[3,2,1]` relies on the `_problem` helper's
  uniform cost (a future edit to costs could invalidate it).
- Docstring line that smoothing is status-blind (the kernel spreads amount into
  and out of locked cells — defensible, Zonation-like, but worth noting).
- Add an `n_pu=1` test (kernel `[[1]]` → smoothed == raw, no crash).
- Reconcile the test count: design "~10-12" → the plan now lands 10 (8 unit + 2
  integration).

## Not absorbed (with reason)
- **`SmoothingSpec.from_problem(problem, alpha)`** deriving centroids from
  planning-unit geometry (independent MEDIUM) — genuinely ergonomic AND would fix
  the HIGH ordering footgun by construction, but pulls in geometry extraction +
  CRS/units handling (α means different things in degrees vs metres). Deferred as
  a follow-up; the alignment-contract doc covers the correctness concern for now.
- **Raw-or-both performance curves** — deferred (YAGNI); documented instead.
- **Per-column loop** — replaced by the one-shot call (above), not kept.
