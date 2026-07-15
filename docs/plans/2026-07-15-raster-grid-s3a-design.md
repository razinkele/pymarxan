# Raster-grid PUs — S3a: sparse solver cache — design

**Date:** 2026-07-15
**Status:** Approved (brainstorm), pending spec review → implementation plan
**Scope:** S3a — shrink the `ProblemCache` memory footprint so the SA / iterative-improvement
solvers run on the large problems S3c can now ingest. Two pieces in one spec: (1) gate the
probmode-3 matrices; (2) CSR feature matrix. S3b (MIP-at-scale guard) and the `build_boundary`
vectorization remain separate. Builds on S3c (v0.19.0). Scoping:
`2026-07-15-raster-grid-pus-scoping.md`.

## Motivation

`ProblemCache` holds **three** dense `(n_pu × n_feat)` float64 matrices — `pu_feat_matrix`,
`expected_matrix`, `var_matrix` (`solvers/cache.py`) — the last dense structures at raster
scale (the boundary is already CSR). At 1M cells × 100 features that is ~2.4 GB, and
`expected`/`var` are **dead weight** for every problem that isn't PROBMODE 3. S3a removes both:
gate the probmode-3 matrices, and store the feature amounts as a `scipy.sparse` CSR (scipy is
already a dependency) — dropping a ~5 %-dense raster's cache from ~2.4 GB to ~60 MB. The MIP
solver already reads `pu_vs_features` sparsely, so this is specifically about the SA /
iterative-improvement (`ProblemCache`) path.

## Scope (S3a)

**Piece 1 — gate `expected_matrix` / `var_matrix` on `probmode == 3`.** They are read only
under `if self.probmode == 3` (`compute_full_objective`, `compute_delta_objective`,
`_compute_zscore_penalty` early-returns otherwise). Build them only when `probmode == 3`;
leave the empty `np.zeros((0, 0))` default otherwise. Behaviour-preserving.

**Piece 2 — CSR feature matrix; `pu_feat_matrix` becomes a lazy densify-property.** Store the
amounts as a CSR built directly from `pu_vs_features` (no dense materialization); route the
hot-loop accesses through CSR methods so a plain problem never densifies; keep
`pu_feat_matrix` as a cached property for the niche dense consumers.

Out of scope: MIP-at-scale guard (S3b), `build_boundary` vectorization (its own
`models/grid.py` task), out-of-core, and any change to the clumping / separation / probmode-3
math (they keep the dense matrix, via the property).

## Piece 2 — details

### New model method `ConservationProblem.build_pu_feature_csr()`

Symmetric with `build_pu_feature_matrix()`: returns a `scipy.sparse.csr_matrix` of shape
`(n_pu, n_feat)`, rows in `planning_units` order, columns in `features["id"]` order. Built by
mapping `pu_vs_features` `pu`/`species` → row/col via `pu_id_to_index`/feature-index, dropping
rows whose `pu`/`species` are unknown (as `build_pu_feature_matrix` does), and constructing
`csr_matrix((amount, (rows, cols)), shape=(n_pu, n_feat))` — which **sums duplicate `(pu,
species)` coordinates** (verified), matching the dense builder's accumulate semantics. Followed
by `.sum_duplicates()` for canonical form.
**Anchor:** `build_pu_feature_csr().toarray() == build_pu_feature_matrix()` exactly.

### `ProblemCache` changes

- Replace the `pu_feat_matrix: np.ndarray` **field** with a `pu_feat_csr` field (the CSR,
  the source of truth), and add:
  ```python
  @cached_property
  def pu_feat_matrix(self) -> np.ndarray:
      return np.asarray(self.pu_feat_csr.toarray())
  ```
  `cached_property` works on this frozen dataclass (verified — it writes to the instance
  `__dict__`, which `frozen=True` does not block). So the public `cache.pu_feat_matrix[:, j]` /
  `[idx, j]` / `[selected, j]` contract used by clumping / separation / analysis is unchanged,
  but it densifies **once, on first access** — and a plain problem never accesses it.
- `from_problem` builds `pu_feat_csr = problem.build_pu_feature_csr()` instead of the dense
  matrix. `feat_uses_pu` is built from the CSR's CSC form (`csr.tocsc()`, per-column
  `indices[data > 0]`) — no densify, preserving the current `where(matrix[:, j] > 0)` result
  for all problems.
- **Piece-1 interaction:** `expected_matrix`/`var_matrix` (and their `prob_matrix`) are built
  only when `probmode == 3`, and there they densify (`self.pu_feat_matrix` / a local
  `csr.toarray()`) — probmode 3 is niche, not the scale target.
- `pu_to_sep_feats` (separation, `sep_active_mask`) still reads `pu_feat_matrix[i, sep_col_ids]`
  — only when `separation_active`, triggering the densify then. Niche.

### Hot-loop rerouting (the parity-critical part)

Three access patterns move from dense indexing to CSR methods so plain problems never densify:

1. **`compute_held(selected)`** → `np.asarray(self.pu_feat_csr[selected].sum(axis=0)).ravel()`
   — O(nnz in selected rows); returns the same dense `(n_feat,)` held vector.
2. **`apply_flip_to_held(held, idx, sign)`** (new method) → from the CSR row's
   `indptr/indices/data`: `held[cols] += sign * amts` (O(nnz_row)). Replaces the direct
   `held += sign * cache.pu_feat_matrix[idx]` at:
   - `simulated_annealing.py:261` → `cache.apply_flip_to_held(held, idx, sign)`;
   - `iterative_improvement.py:220` (`held -= [i]`) → `apply_flip_to_held(held, i, -1.0)`;
   - `iterative_improvement.py:302` (`held += [i]`) → `apply_flip_to_held(held, i, +1.0)`;
   - `iterative_improvement.py:355` (`new_held = held - [r] + [a]`) →
     `new_held = held.copy(); apply_flip_to_held(new_held, r, -1.0); apply_flip_to_held(new_held, a, +1.0)`.
3. **`compute_delta_objective`** penalty block → replace `pu_amounts = self.pu_feat_matrix[idx]`
   + full-`n_feat` arithmetic with per-nonzero-column arithmetic:
   ```python
   s, e = self.pu_feat_csr.indptr[idx], self.pu_feat_csr.indptr[idx + 1]
   cols = self.pu_feat_csr.indices[s:e]
   amts = self.pu_feat_csr.data[s:e]
   eff = self.feat_targets[cols] * self.misslevel
   old_sf = np.maximum(0.0, eff - held[cols])
   new_sf = np.maximum(0.0, eff - (held[cols] + sign * amts))
   penalty_delta = float(np.dot(self._det_spf[cols], new_sf - old_sf))
   ```
   **Bit-identical** to the dense version — features absent from the row have `pu_amounts[j]=0`,
   so their shortfall is unchanged and their contribution to the dot product is exactly 0 —
   and now **O(nnz_row)** instead of O(n_feat).

The probmode-3 delta branch (`expected_matrix[idx]`, `var_matrix[idx]`,
`expected_matrix[selected].sum(0)`) is unchanged (dense, probmode-3 only).

## Parity & risk

This rewrites hot loops in SA, iterative-improvement, and the cache — all Marxan-parity
surface. The anchor is direct: the `{2, 4, 6} = 35.0` simple project is **plain** (no
`target2`/`sepnum`/`prob`), so it runs entirely through the CSR path — MIP must still return
35.0 and SA/greedy land **at or above** it, and `examples/validate_marxan_parity.py` must pass
unchanged. Additional guards: dense-vs-CSR unit tests on `build_pu_feature_csr`, `compute_held`,
`apply_flip_to_held`, and `compute_delta_objective` (identical to the pre-refactor dense
values on a random problem incl. duplicate `(pu,species)` rows and unknown-id rows); and a
**`bench` run** to confirm the O(nnz_row) delta does not regress per-flip cost (it should be
≤ the current O(n_feat)).

## Testing strategy (TDD)

- **`build_pu_feature_csr` == dense:** on `tiny_problem` and a random problem (with duplicate
  and unknown-id `pu_vs_features` rows), `build_pu_feature_csr().toarray()` equals
  `build_pu_feature_matrix()`.
- **`compute_held` dense-vs-CSR:** for several random selections, the CSR `compute_held` equals
  the old dense `pu_feat_matrix[selected].sum(0)`.
- **`apply_flip_to_held`:** for random `idx`/`sign`, `apply_flip_to_held(held, idx, sign)` equals
  `held + sign * dense_matrix[idx]`.
- **`compute_delta_objective` bit-identical:** on the simple project and a random plain problem,
  the CSR delta equals the value the dense implementation produced for the same flip (guard the
  bit-exact penalty-delta).
- **`pu_feat_matrix` property:** densifies to the right dense matrix, is cached (same object on
  second access), and a plain-problem solve never triggers it (assert via a spy /
  `pu_feat_csr` present and the property's `__dict__` slot absent after `compute_held`+delta).
- **`expected`/`var` gating:** a probmode-0 problem's cache has empty `expected_matrix`/
  `var_matrix`; a probmode-3 problem builds them and its Z-score penalty is unchanged (existing
  probmode-3 tests still pass).
- **Marxan parity:** `examples/validate_marxan_parity.py` and the solver/objective/penalty test
  suite green; MIP == 35.0; SA/greedy ≥ 35.0.
- **`bench`:** the SA per-flip benchmark does not regress.

**Target:** ~12–16 new/changed tests, `make check` green (0 ruff / 0 mypy), coverage ≥ 75%,
`bench` non-regressing.

## References

Scoping: `2026-07-15-raster-grid-pus-scoping.md` (S3). Cache: `solvers/cache.py`
(`from_problem`, `compute_held`, `compute_delta_objective`, the `pu_feat_matrix` contract).
Call sites: `solvers/simulated_annealing.py:261`, `solvers/iterative_improvement.py:220/302/355`,
`solvers/clumping.py`, `solvers/separation.py`. `scipy.sparse.csr_matrix` (already a dep;
sums duplicate coords). Parity harness: `examples/validate_marxan_parity.py`,
`marxan-parity-check` skill.
