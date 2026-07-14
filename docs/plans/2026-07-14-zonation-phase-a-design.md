# Zonation rank-removal solver — Phase A (core) design

**Date:** 2026-07-14
**Status:** Approved (brainstorm), pending implementation plan + multi-agent review
**Scope:** Phase A only — the core `pymarxan.zonation` rank-removal engine. Phases
B (Solver adapter), C (distribution smoothing), D (Shiny panel) are deferred to
later sessions.

## Motivation

Zonation (Moilanen et al. 2005) is the one live competitor doing something
pymarxan structurally cannot: instead of Marxan's target-based minimum-set
(pick the cheapest *subset* that meets every target, binary in/out), it produces
a **continuous priority ranking of every planning unit** by iterative backward
removal. Starting from the whole landscape, it repeatedly removes the
least-valuable cell; the *order of removal* is the priority ranking, yielding a
nested 0–1 priority map plus per-feature performance curves. This is the
"priority-rank-removal" paradigm flagged as the highest-value genuinely-new
capability in the 2026-06-12 ecosystem survey.

This is distinct from the existing `analysis.rank_importance` (Jung et al. 2021),
which ranks only the *selected* PUs of an existing Marxan solution by objective
increase — a different question on a different input.

## Scope (Phase A, approved)

Core engine only:
- `rank_removal(problem, *, rule, weights, warp, use_cost) -> ZonationResult`
- Two removal rules: **CAZ** (core-area) and **ABF** (additive benefit).
- Cost-aware, feature-weighted, PU-status-aware, warp factor for speed.
- `ZonationResult`: priority rank per PU, removal order, per-feature performance
  curves.

**Deferred (later phases, not this spec):** `ZonationSolver` Solver-ABC adapter
(Phase B), distribution smoothing reusing `connectivity.smoothing` (Phase C),
Shiny panel (Phase D).

## Module layout

```
src/pymarxan/zonation/
  __init__.py       # exports rank_removal, ZonationResult
  rank_removal.py   # rank_removal(...) + the CAZ/ABF removal loop
  result.py         # ZonationResult dataclass
tests/pymarxan/zonation/
  test_rank_removal.py
  test_result.py
```

Consistent with the other core subpackages (`rivers`, `temporal`, `phylo`) —
pure core, no UI dependency.

## Algorithm (`rank_removal.py`)

```python
rank_removal(
    problem: ConservationProblem,
    *,
    rule: str = "caz",              # "caz" | "abf"
    weights: dict[int, float] | None = None,   # feature_id -> w_j; default 1.0
    warp: int = 1,                  # cells removed per iteration (1 = exact)
    use_cost: bool = True,          # divide marginal loss by PU cost
) -> ZonationResult
```

**Setup.** `q = problem.build_pu_feature_matrix()` — dense `(n_pu, n_feat)`
float, columns in `problem.features["id"]` order. PU ids from
`problem.planning_units["id"]`. `w` = length-`n_feat` weight vector aligned to
that feature order; `weights` is keyed by feature id and **any feature id absent
from the dict defaults to 1.0** (lenient — a partial `weights` dict is valid).
`c` = PU costs (`use_cost=True`) or all-ones (`use_cost=False`). `status` from
`planning_units["status"]`. The loop removes **every** PU (down to an empty
landscape), so all `n_pu` cells receive a rank.

**State.**
- `remaining` — boolean mask over PUs, all `True` initially.
- `Q_j = q[remaining].sum(axis=0)` — **remaining** total of each feature;
  updated incrementally as `Q_j -= q_ij` on each removal (delta computation, no
  full recompute — matches the repo's performance model).
- `T_j = q.sum(axis=0)` — original totals, fixed, for the performance curves.

**Removal rules (verified against Moilanen 2007, primary source).** For a
candidate cell `i`, over features with `Q_j > 0`:
- **CAZ (core-area):** `δ_i = max_j ( w_j · q_ij / Q_j )` — the largest weighted
  proportion of any feature's *remaining* stock held in cell `i`. Protects every
  feature's core; favors rare/narrow-range features.
- **ABF (additive benefit):** `δ_i = Σ_j ( w_j · q_ij / Q_j )` — the sum instead
  of the max, over each feature's share of its *remaining* stock. Favors
  species-rich cells. This is the proportional / remaining-sum form of Zonation's
  additive-benefit family (`q_ij/Q_j = (q_ij/T_j)/R_j`, i.e. a `B'(R)=1/R`
  marginal benefit — **not** a strictly *linear* benefit, which would use the
  fixed original total `T_j` and be static). The remaining-sum `Q_j` is what
  makes ABF adaptive; the concave power-benefit generalization
  (`V_j = w_j·R_j^z` with a user-set exponent `z`, Moilanen 2007) is a documented
  future extension. (Only the CAZ rule is an exact transcription of Moilanen 2007
  Eq. 1a; ABF here is the proportional member of the family, not the paper's
  general power form.)
- **Cost:** divide by cost — `δ_i ← δ_i / c_i` (value-per-cost efficiency; since
  `c_i` is constant over features it is equivalent whether applied inside or
  outside the `max`/`Σ`). Under uniform cost this is pure biological ranking.
  (The division form — rather than subtraction — is the standard Zonation
  cost-efficiency formulation; flagged for the design-review scientific pass to
  confirm against the Zonation manual.)

The cell(s) with the **smallest** `δ_i` are removed (least value lost).

**PU status** (removal precedence, so locks are honored regardless of `δ`):
- **status 3 (locked-out)** removed **first** — lowest priority. While any
  locked-out cells remain, only they are candidates.
- **status 0/1 (normal / initial-include)** removed next, by `δ`.
- **status 2 (locked-in)** removed **last** — highest priority. Only candidates
  once every normal cell is gone.

Locked-out cells are *removed first*, **not masked out**: their amounts are part
of `T_j`/`Q_j` and are stripped during the initial locked-out removals (so a
feature whose only stock sits in a locked-out cell correctly shows early
retention loss). The CAZ rarity property therefore applies to sole-occurrence
features in **normal** cells; the rarity test uses a status-0 cell.

**Warp factor.** Each iteration, compute `δ` once for the current candidate set
and remove the `min(warp, n_candidates)` cells with the smallest `δ` (batch),
then update `Q_j` by the summed contribution of the removed batch. A batch never
spans a status tier — the candidate set is chosen by tier *first* (below), so a
batch that empties the locked-out tier stops there and the next iteration moves
to normal cells. `warp=1` recomputes every removal (exact Zonation). Larger
`warp` trades exactness for `O(n²/warp · n_feat)` speed. **This recompute is
inherent** — removing a cell shifts *every* `Q_j`, so the Marxan per-flip
`ProblemCache` delta model does not apply (only the `Q_j -= q_ij` update is
incremental). The engine therefore suits vector PUs (hundreds to low-thousands);
million-cell rasters are the separate raster-PU gap, not this loop. Within a
batch, cells are ordered (for removal-position/rank) by `δ` ascending then PU
index; ties in `δ` across the whole step are broken by PU index (reproducible).

**Priority rank.** `removal_order` lists PUs first-removed → last-removed. For
the `k`-th removed cell (`k` 0-indexed), `priority_rank = (k + 1) / n_pu` — so
the first-removed cell gets `1/n` (lowest) and the last-removed cell gets `1.0`
(highest). Rank ∈ (0, 1], higher = more important.

**Performance curves.** Before the first removal and after each batch, record a
sample: the proportion of the landscape still remaining **by cell count**
(`prop_landscape_remaining`) and **by cost/budget** (`prop_cost_remaining` =
Σ remaining cost / Σ total cost, using the ranking's cost vector), plus, per
feature, the retained proportion `Q_j / T_j`. Both x-axes are recorded because
when `use_cost=True` the removal order is `δ/cost`, so a by-count axis alone
would misrepresent "how much budget was spent" (a variable-cost user reading the
curve needs the cost axis). Because cells are stripped worst-first, retention
stays near 1.0 until important cells start being removed — the classic Zonation
"proportion of distribution remaining vs proportion of landscape removed" curve.
Note the curves include locked-out (unprotectable) stock: `T_j`/`Q_j` count
locked-out cells, which are stripped during the initial forced removals, so the
early curve segment reflects unavoidable loss.

**Edge cases.**
- A feature with `T_j = 0` (present in no PU) contributes 0 to `δ` and its
  retained proportion is reported as `1.0` (nothing to lose); it never drives
  the ranking.
- `use_cost=True` requires every cost `> 0` (else `ValueError` — division would
  be undefined); `use_cost=False` sidesteps this.
- `warp` clamped to `[1, n_pu]`; `warp ≥ n_pu` removes everything in one bucket
  (degenerate ranking — documented).
- `rule` not in `{"caz", "abf"}` → `ValueError`.

## `ZonationResult` (`result.py`)

A dataclass (with `to_dataframe()`, like `PDResult`):
- `priority_rank: dict[int, float]` — PU id → rank in (0, 1], 1.0 = highest.
- `removal_order: list[int]` — PU ids, first-removed (lowest priority) first.
- `performance_curves: pd.DataFrame` — wide form: `prop_landscape_remaining` and
  `prop_cost_remaining` columns plus one `feat_<id>` column per feature (retained
  proportion), one row per recorded step (start → each batch → empty).
- `rule: str` — `"caz"` or `"abf"`.
- `top_fraction(f: float) -> set[int]` — the PU ids in the top `f` share by rank
  (the `ceil(f · n_pu)` highest-ranked). `0 < f ≤ 1`; used by the future
  `ZonationSolver` (Phase B) to threshold a rank into a `Solution`.
- `to_dataframe() -> pd.DataFrame` — columns `pu_id`, `priority_rank`,
  `removal_position`.

## Testing strategy (TDD, hand-computed oracle)

- **Hand-worked CAZ order.** A tiny problem (3 PUs, 2 features) with occurrences
  chosen so the CAZ removal order is derivable by hand; assert `removal_order`,
  `priority_rank`, and the first performance-curve rows.
- **Hand-worked ABF order** on the same problem where ABF and CAZ diverge
  (a species-rich cell ABF keeps but CAZ would drop, or vice versa) — proves the
  two rules are actually different.
- **Rarity property (CAZ):** the sole cell holding a feature always contributes
  `w_j · q_ij / Q_j = w_j` to its CAZ `δ` (proportion 1.0 of that feature's
  remaining stock), so it is removed **after every cell that holds no feature's
  sole occurrence**. Test on a problem with exactly one sole-occurrence feature
  so that cell is unambiguously last (rank 1.0); do **not** assert "rank 1.0" on
  a problem with multiple rare features (several cells reach `δ=1` and only one
  is literally last) — there, assert the rare cell is in the top tier / removed
  after all common-only cells.
- **Cost drives ties:** uniform feature values + non-uniform cost → the
  cheapest-to-keep (lowest cost, given equal value) ordering; and `use_cost=False`
  changes the order.
- **Locks:** every locked-in PU is removed after every normal PU (top tier —
  ranks above all normal cells), and every locked-out PU is in the first-removed
  positions (ranks below all normal cells), regardless of biological value. Only
  with a *single* locked-in cell is its rank exactly 1.0 (same uniqueness caveat
  as the rarity test); with several, assert the top-tier ordering, not rank 1.0
  on each.
- **Performance curve monotonic + bounded:** every retained proportion ∈ [0, 1]
  and is non-increasing as the landscape is stripped; the last row (empty
  landscape) is 0 for every present feature; `prop_landscape_remaining` runs
  1.0 → 0.0.
- **Warp consistency:** `warp=1` and a small `warp` agree on the coarse ranking
  buckets (documented approximation, not exact equality in general). On the P1
  oracle they happen to produce the identical `removal_order` (the batch is
  order-preserving there); the test uses P1 and notes this is coincidental, not
  the general guarantee.
- **Numeric guards (the fiddly float edges):** a feature present in **zero** PUs
  (`T_j = 0`) is excluded from `δ` and reports retained proportion 1.0 (no
  crash); a feature that goes **extinct mid-run** (`Q_j` reaches 0 while cells
  remain) is dropped from `δ` by the `Q_j > 0` guard without a divide-by-zero.
- **Validation:** `rule="bogus"` raises; `use_cost=True` with a zero cost raises.

**Target:** ~15–20 tests, `make check` green (0 ruff / 0 mypy), coverage ≥ 75%.

## Out of scope (YAGNI, Phase A)

- General/target-based benefit functions beyond linear ABF (Moilanen 2007
  extension), condition/retention layers, hierarchical masks.
- Distribution smoothing / connectivity (Phase C — will reuse
  `connectivity.smoothing`).
- The `ZonationSolver` Solver-ABC adapter + registry entry (Phase B).
- Shiny panel (Phase D).

## References (verified via scite; formulas via the open Moilanen 2007 PDF)

- Moilanen, A., Franco, A. M. A., Early, R. I., Fox, R., Wintle, B., & Thomas,
  C. D. (2005). Prioritizing multiple-use landscapes for conservation: methods
  for large multi-species planning problems. *Proceedings of the Royal Society
  B*, 272(1575), 1885–1891. https://doi.org/10.1098/rspb.2005.3164
- Moilanen, A. (2007). Landscape Zonation, benefit functions and target-based
  planning: unifying reserve selection strategies. *Biological Conservation*,
  134(4), 571–579. https://doi.org/10.1016/j.biocon.2006.09.008 — CAZ/ABF
  marginal-loss formulas.
- Lehtomäki, J., & Moilanen, A. (2013). Methods and workflow for spatial
  conservation prioritization using Zonation. *Environmental Modelling &
  Software*, 47, 128–137. https://doi.org/10.1016/j.envsoft.2013.05.001
