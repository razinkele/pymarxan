# Zonation at raster scale — design spec

**Date:** 2026-08-08
**Status:** draft (pending loop review + multi-agent design review)
**Scope decision trail:** brainstormed interactively; user chose (1) target ~1M cells ×
10–100 features on a workstation, (2) output identical to the existing engine with `warp`
as the scale knob, (3) distribution smoothing deferred to a follow-up phase, (4) approach
A — sparse + dirty-set rescoring.

## 1. Problem

`pymarxan.zonation.rank_removal` (v0.13.0) is the CAZ/ABF backward-removal engine.
It is correct and literature-verified but structurally vector-scale:

1. **Memory** — `rank_removal.py:53` calls `build_pu_feature_matrix()` (dense
   `(n_pu, n_feat)` float64): 160 MB at 1M×20, 800 MB at 1M×100.
2. **Compute** — every batch rescores **all** remaining candidates:
   O(remaining × n_feat) per batch, O(n²·f/warp) total.
3. **Curves** — `record_curve()` runs per batch and computes `c[remaining].sum()`,
   a hidden O(n²/warp).
4. **Result build** — tolerable (see §8).

Meanwhile the raster pipeline (S1–S4b, v0.17–v0.25) ingests and solves million-cell
grids, and Zonation's native domain *is* such grids. This phase closes that gap.

## 2. Goals / non-goals

**Goals**

- `rank_removal` handles ~1M cells × 10–100 features on a normal workstation in
  minutes (at raster-appropriate `warp`), bounded by sparse-matrix memory (~0.5 GB
  at 20M nnz), never by a dense n_pu×n_feat allocation.
- Output is **identical in semantics** to the current engine for every valid input and
  every `warp` (FP contract in §7; the one deliberate exception: negative
  amounts/weights, previously undefined behavior, now raise — §4.1). No API change:
  same signature, same `ZonationResult`, `ZonationSolver` and the Shiny tab untouched.
- The dense engine survives only as a test-local reference implementation for
  equivalence tests.

**Non-goals (explicitly deferred)**

- CELF lazy-heap exact-`warp=1`-at-scale path (declined in brainstorm; log in roadmap).
- Grid-convolution (2-D) distribution smoothing — own follow-up phase with its own
  science review. This phase only adds a clear failure for absurd smoothing sizes (§6).
- Out-of-core / float32 / 10M+ cells; array-backed `ZonationResult` (§8).
- Concave power-benefit ABF generalization (pre-existing deferral, unchanged).

## 3. Data structures

Built once at start:

- `csr` — `problem.build_pu_feature_csr()` (rows = PU order, cols = feature order,
  `sum_duplicates()` applied → unique sorted column indices; `toarray()` equals the
  dense builder — this is the equivalence bridge). The returned matrix is freshly
  built and ours to mutate: call `eliminate_zeros()` on it so stored-zero entries
  can't mark features dirty for nothing.
- `csc` — `csr.tocsc()`, for "which cells hold feature j" lookups.
- Persistent per-run state:
  - `delta: float64[n_pu]` — current scores (all cells, regardless of lock phase),
  - `remaining: bool[n_pu]`, plus an integer `n_remaining` counter,
  - `Q: float64[n_feat]` — remaining totals; `T = Q.copy()` at start (curves),
  - `dirty: bool[n_pu]` — cells whose score must be recomputed before next use
    (initially all True),
  - `cost_remaining: float64` — maintained incrementally.

When `smoothing is not None`: apply it on the dense matrix exactly as today
(`build_pu_feature_matrix()` → `smoothing.apply(q)` → `csr_matrix(...)`) and run the
same sparse engine on the result. One engine code path; smoothing stays vector-scale
by the guard in §6.

## 4. The per-batch algorithm

Phase structure is preserved verbatim: candidates are locked-out cells while any
remain, then normal cells, then locked-in — recomputed per batch from `remaining` and
`status` (O(n) mask ops, ~ms at 1M; batches never span phases, as today).

Each batch:

1. **Rescore dirty candidates only.** Rows `R = dirty & candidate-mask` — *not* all
   remaining cells: a dirty locked-in cell simply stays flagged until phase 3 makes it
   a candidate, and is rescored then (lazier and strictly cheaper; the boolean flag
   absorbs repeated invalidations). On the CSR slice of `R`:
   `factor = np.where(Q > 0, w / Q_safe, 0.0)`; `vals = sub.data * factor[sub.indices]`;
   reduce per row — CAZ `np.maximum.reduceat`, ABF `np.add.reduceat` — with the two
   reduceat pitfalls handled: empty rows forced to 0.0 (reduceat's empty-segment
   quirk returns a neighbouring element), and CAZ floored at 0.0 to mirror the dense
   engine's implicit zeros for absent features. The 0-floor is exact only under
   nonnegative amounts and weights, so `rank_removal` now validates both up front
   (`csr.data.min() >= 0`, `w.min() >= 0` → `ValueError` otherwise). Negative values
   are scientifically meaningless in CAZ/ABF and today produce undefined behavior;
   review should confirm this validation (a behavior change for pathological inputs
   only). Divide by `c[R]`. Clear `dirty[R]`.
   Extinct features (`Q_j ≤ 0`) contribute 0 through `factor`, matching the dense
   `r[:, Q <= 0] = 0.0`.
2. **Select the k smallest with today's exact tie-break.** Current engine:
   stable argsort over candidates in ascending PU-index order → ties broken by PU
   index, and the batch is *emitted* in (delta, index) order (rank depends on
   intra-batch position). Scaled selection: `np.argpartition` for the kth value `v`;
   take all candidates with `delta < v`; fill remaining slots from `delta == v` in
   ascending PU-index order; sort the selected k by `(delta, PU index)` for emission.
   Same set, same order, without the O(m log m) full sort. Edge cases: `k == cand.size`
   (skip partition), `n_feat == 0` (all deltas 0 → pure index order, as today).
3. **Update state in the dense engine's order.** For each removed cell in emission
   order: append to `removal_order`, clear `remaining`, subtract its CSR row from `Q`
   (`Q[row_cols] -= row_data` — sequential per cell exactly like the dense
   `Q -= q[idx]`, so `Q`'s FP trajectory matches the reference given equal inputs),
   subtract its cost from `cost_remaining`, decrement `n_remaining`.
4. **Mark dirty.** `changed = union of column indices of the removed rows`;
   `dirty[csc columns of changed] = True` (restricted to `remaining` implicitly at
   next rescore). Cells untouched by any changed feature keep their score — which is
   **bit-identical** to what a full rescore would produce, because every input to
   their score (their `q` row, `w`, their features' `Q_j`, their cost) is unchanged.
5. **Record the curve row** from `Q/T`, `n_remaining/n_pu`, `cost_remaining/cost_total`
   — same values as today up to FP regrouping (§7), no O(n) scans.

Loop until `remaining` is empty; result assembly unchanged.

## 5. Guardrail: warp advisory

`warp=1` at raster scale is O(n²) in *selection* alone (an argpartition over ~n
candidates per single removal), regardless of rescoring cleverness. Mirroring the S3b
MIP-scale precedent (warn-and-proceed, never auto-route, module-level helper):
warn once at entry when `n_pu > 50_000 and warp < n_pu // 10_000`, saying warp is
Zonation's raster knob and suggesting `warp ≈ n_pu // 1000`. No new parameter;
standard `warnings.warn` (filterable).

## 6. Guardrail: smoothing stays vector-scale

`SmoothingSpec` builds a dense n×n kernel (8 TB at 1M cells). Raise `ValueError` in
`rank_removal` when `smoothing is not None and n_pu > 50_000`, with a message naming
the deferred grid-convolution follow-up. Threshold rationale: at 50k the kernel is
already ≥ 20 GB (would MemoryError obscurely today), while plausible current usage
(≤ ~10–15k vector PUs) is untouched — no regression for any workload that works now.

## 7. FP / equivalence contract (the S3a lesson, applied honestly)

Three reduction sites regroup between dense and sparse:

| Site | Dense | Sparse | Consequence |
|---|---|---|---|
| initial `Q` | `q.sum(axis=0)` (pairwise) | per-column sums | exact on integer amounts (integer float64 addition is order-free); ≤ few ULP on floats |
| ABF row score | `r.sum(axis=1)` (pairwise) | `add.reduceat` (sequential) | ≤ few ULP **even on integer amounts** — the addends are `q_ij·(w_j/Q_j)`, already non-integer after division, and the two paths group the sum differently |
| CAZ row score | `r.max(axis=1)` | `maximum.reduceat` | exact given identical inputs (max is order-free; the elementwise products are computed identically) |

Resulting claim, per rule:

- **CAZ, integer amounts:** `Q` is bitwise-identical along the whole trajectory
  (integer sums/subtractions are exact), every elementwise product is identical, max
  is order-free → **removal order and ranks exactly equal** to the dense reference.
- **ABF, any amounts:** row-sum regrouping means scores agree only to a few ULPs,
  which can flip *exact float near-ties*. Equivalence tests therefore use fixtures
  whose pairwise score gaps are ≫ ULP (asserted in the fixture itself), where exact
  order equality again holds deterministically; a separate tolerance test covers
  scores directly.
- **Float amounts (either rule):** `Q` itself drifts by ULPs → same near-tie caveat.

Dirty-set vs full-rescore is bit-identical *within* the sparse engine by construction
(§4.4) — the only FP boundary is dense-vs-sparse. Docstring and CHANGELOG state this
contract; no "bit-identical" claims anywhere.

## 8. Result container — unchanged (measured decision)

`priority_rank: dict[int, float]` costs ~140 MB at 1M cells and `top_fraction` sorts
O(n log n) — heavy but workstation-tolerable, and it is public API used by
`ZonationSolver` and the Shiny tab. Array-backing is YAGNI until a user hits it;
noted in roadmap as a possible follow-up. `performance_curves` at raster warp is
small (n_pu/warp rows ≈ 1000 × (2 + n_feat) columns).

## 9. Testing

All under the `shiny` micromamba env (marxan-testing skill). New test file name must
be repo-unique (pytest basename gotcha): `test_rank_removal_scale.py`.

1. **Reference-equivalence (the core):** the current dense engine is copied verbatim
   into the test module as `_dense_rank_removal` (the src version is rewritten);
   property-style comparison of `removal_order`, `priority_rank`, and curves against
   the sparse engine on integer-amount problems covering: no locks / locked-in +
   locked-out mix / feature going extinct mid-run / `warp ∈ {1, 3, 7, n_pu}` /
   `use_cost` both / weights / `rule ∈ {caz, abf}` / `n_feat == 0` / a PU with no
   features / duplicate `(pu, species)` rows. Exact equality asserted per the §7
   contract (CAZ everywhere; ABF on gap-verified fixtures).
2. **Float tolerance:** float-amount problem; scores compared to reference with
   `np.testing.assert_allclose(rtol=1e-12)`; ranks compared allowing near-tie swaps
   (or seeded to avoid ties).
3. **Tie-break pinning:** constructed exact-tie batch; assert selection and emission
   order match the reference (guards the argpartition boundary logic).
4. **Dirty-set correctness in isolation:** moderate grid (~50×50, localized
   features) — assert equality with reference; plus an internal assertion-style test
   that a full-rescore run equals the dirty-set run on the same seed.
5. **Guardrails:** warp advisory fires (and not below thresholds); smoothing raise
   at n_pu > 50_000; existing smoothing behavior below threshold unchanged.
6. **No-dense guarantee (fast, CI-enforced):** small problem, no smoothing —
   monkeypatch `build_pu_feature_matrix` to fail if called; the sparse path must
   never densify. Kept out of the bench marker so CI actually enforces it.
7. **Scale smoke (bench-marked, excluded from CI):** ~300×300 synthetic grid,
   `warp = n//1000` — asserts a wall-clock budget only.
8. **Existing suite:** all current zonation/solver/Shiny tests must pass unchanged —
   they are themselves equivalence evidence. Parity anchor 35.0 untouched
   (no Marxan solver touched; `make check` proves it).

## 10. Performance envelope (claims to check in review)

- Init: O(nnz) scoring + CSR/CSC build; ~0.5 GB at 20M nnz (both structures).
- Per batch: O(dirty-nnz) rescore + O(cand) partition + O(k·row-nnz) updates + O(n)
  phase masks (~1–2 ms at 1M).
- Localized features (raster norm): dirty sets are neighbourhood-sized; total ≈
  O(nnz · overlap-factor). Worst case (landscape-spanning feature): every batch
  rescans that feature's holders — degenerates toward full-rescore-per-batch
  (≈ approach B: ~2 min at 1M×20, warp=1000), never worse than it.
- First batch rescore is O(nnz) (everything dirty once) — one-time ~100 ms.

## 11. Files touched

- `src/pymarxan/zonation/rank_removal.py` — engine internals; docstring rewrite
  (drop "O(n²) inherent / suits vector PUs", state the new envelope + FP contract);
  the two guards.
- `tests/pymarxan/zonation/test_rank_removal_scale.py` — new (§9); existing tests
  untouched.
- `CHANGELOG.md` — `[Unreleased]` Changed entry (next release: minor, v0.31.0).
- Roadmap memory + docs: mark Zonation-at-raster-scale done; log deferred CELF heap
  and grid-convolution smoothing.

## 12. Risks / open questions for review

- The argpartition boundary-tie logic is the subtlest code; mitigated by test §9.3
  and the reference oracle.
- `reduceat` empty-segment and CAZ implicit-zero handling are classic silent-wrong
  spots; both explicitly tested (PU-with-no-features case).
- Is `eliminate_zeros()` safe given `build_pu_feature_csr`'s documented contract
  (`toarray()` equality holds; only stored-zero *structure* changes)? Grounding
  review should confirm no other consumer relies on stored zeros of the *returned
  copy* (we operate on our own copy either way).
- Negative amounts/weights now raise (§4.1) — reviewers to confirm this is the right
  call versus silently accepting undefined behavior as the dense engine does today.
- `.github/copilot-instructions.md` solver-matrix wording may describe zonation as
  vector-scale; check and update alongside §11.
