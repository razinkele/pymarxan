# restoptr — MESH-maximizing greedy optimizer — design

**Date:** 2026-07-15
**Status:** Approved (brainstorm). Pending spec self-review → user review → spec-review loop →
writing-plans → multi-agent design review → TDD.
**Scope:** The third `pymarxan.restoration` piece — a **greedy MESH-maximizing optimizer**:
`greedy_mesh_restore(problem, budget)` chooses which restorable cells to restore, under a cost
budget, to maximize the effective mesh size, returning a restoration plan + a budget–MESH frontier.
Turns the data model into actual restoration *plans*. Named competitor: **restoptr**
(`set_max_mesh_objective` + `add_budget_constraint`).

## Motivation

`RestorationProblem` (v0.29.0) can *evaluate* a restoration plan (`restore_mesh`), but there is no
way to *find* a good one. restoptr's flagship workflow is "maximize a landscape index (MESH) subject
to a restoration budget." This piece adds that as a fast greedy heuristic — mirroring how pymarxan's
greedy/SA solvers relate to its exact MIP, and the measure→greedy arc used for raptr and zonation.

## The model

`src/pymarxan/restoration/optimize.py`:

```python
@dataclass(eq=False)   # numpy field breaks the auto __eq__ (repo convention)
class RestorationResult:
    restored: np.ndarray      # (n_pu,) bool — the chosen restoration plan (⊆ restorable)
    mesh: float               # final post-restoration MESH (effective mesh size)
    baseline_mesh: float      # pre-restoration MESH (existing habitat only)
    total_cost: float         # Σ cost[restored]
    n_restored: int
    mesh_curve: np.ndarray     # MESH after each step (index 0 = baseline) — budget–MESH frontier

def greedy_mesh_restore(
    problem: RestorationProblem,
    budget: float,
    *,
    criterion: str = "gain_per_cost",   # "gain_per_cost" | "gain"
    connectivity: str = "rook",
    cell_area: float | None = None,
) -> RestorationResult
```

**Algorithm (greedy, benefit–cost).** `restored = zeros(n_pu, bool)`; `spent = 0.0`;
`current = baseline = compute_mesh(grid, existing_habitat)`. `mesh_curve = [baseline]`. Repeat:
1. Candidate set = restorable cells not yet restored whose `cost[c] <= budget − spent` (affordable).
2. If empty → stop.
3. For each candidate `c`: `gain_c = compute_mesh(grid, existing | restored | {c}).mesh − current`
   (a full `scipy.ndimage.label` recompute per candidate); `score_c = gain_c / cost[c]` when
   `criterion == "gain_per_cost"` (else `gain_c`).
   For `gain_per_cost`, a **zero-cost** cell (`cost[c] == 0`) with `gain_c > 0` scores `+inf` (free
   MESH → restored first).
4. Pick `c*` = argmax score (ties → lowest cell index, deterministic). **If `gain_{c*} <= 0` → stop**
   (only no-op cells remain — e.g. an already-habitat `existing ∩ restorable` overlap cell adds no
   MESH). Else restore it: `restored[c*] = True`; `spent += cost[c*]`; `current += gain_{c*}`;
   append `current` to `mesh_curve`.

Each candidate's MESH is computed with `compute_mesh(grid, existing_habitat | restored | onehot(c))`
**directly** (not `problem.restore_mesh`), skipping the per-call subset validation — the data-model
review's "hot loop" guidance.

Return `RestorationResult(restored, mesh=current, baseline_mesh=baseline, total_cost=spent,
n_restored=int(restored.sum()), mesh_curve=np.asarray(mesh_curve))`.

**Key properties (honesty, for the docstring + review):**
- **MESH is monotone** in the habitat mask — adding a *non-habitat* cell strictly raises
  `ΣA_i²/A_total` (a new isolated cell adds `cell_area²/A_total`; a bridging cell more). So every
  genuine restoration strictly improves MESH and the greedy spends the budget down; the decision is
  *which* cells (best gain-per-cost), never *whether* to keep going. The `gain > 0` guard is a real
  (edge) stop: a restorable cell that is *already* habitat (an `existing ∩ restorable` overlap) adds
  zero MESH, so once only such no-op cells remain affordable the greedy stops rather than wasting
  budget.
- **Heuristic, no optimality guarantee.** `Σarea²` is **super**modular (a cell bridging two patches
  yields an outsized, non-diminishing gain), so greedy has no approximation bound and can miss
  multi-cell bridges. restoptr uses an exact constraint-programming (Choco) solver for guarantees;
  ours is the fast heuristic. Documented, not hidden.
- **Performance.** `O(iterations × candidates × label)` — MESH is not incrementally decomposable
  (a flip can merge patches), so each candidate is a full `compute_mesh`. Fine for moderate grids;
  a union-find incremental delta is a deferred optimization.
- **`criterion` default `gain_per_cost`** — budget-aware; reduces to raw `gain` when cost is uniform
  (the `RestorationProblem.cost` default), so `greedy_mesh_restore(problem, budget=k)` on a
  uniform-cost problem restores the `k` best cells by marginal MESH gain.

## Edge cases / validation

- **`budget < 0`** → `ValueError`.
- **`budget = 0`**, or **no restorable cells**, or **no affordable candidate** at step 1 →
  `restored = ∅`, `mesh = baseline_mesh`, `total_cost = 0`, `n_restored = 0`,
  `mesh_curve = [baseline]`.
- **`criterion` not in {"gain_per_cost", "gain"}** → `ValueError`.
- **`existing_habitat ∩ restorable` overlap** — a restorable cell already habitat contributes zero
  gain (already in the habitat mask); it is skipped implicitly (gain 0). `RestorationProblem.validate`
  already flags the overlap as a data error; the optimizer does not re-validate.
- **`gain_per_cost` with a zero cost** — a `cost[c] == 0` restorable cell scores `+inf` (free MESH,
  restored first). Implemented as `score = inf if cost == 0 else gain / cost` (guarded so no
  division by zero).

## Testing strategy (TDD)

- **Hand-computed greedy on a small grid:** a 1×5 strip, ends already habitat, middle 3 restorable,
  uniform cost, budget=1 → greedy restores the cell that most raises MESH (the one that bridges /
  extends the largest patch); assert `restored`, `mesh`, `n_restored=1`, `total_cost=1`, and
  `mesh_curve = [baseline, mesh]`.
- **Budget fills:** budget = k (uniform cost) restores exactly k cells (MESH monotone → always
  spends); budget ≥ total restorable cost → all restorable restored, `mesh == compute_mesh(all
  existing|restorable)`.
- **Monotone frontier:** `mesh_curve` is non-decreasing; `mesh_curve[0] == baseline_mesh`;
  `mesh_curve[-1] == mesh`; `len(mesh_curve) == n_restored + 1`.
- **gain_per_cost vs gain:** on a problem where a cheap-but-lower-gain cell beats an
  expensive-higher-gain cell per unit cost, the two criteria pick different first cells.
- **Cost budget honored:** `total_cost <= budget`; a non-uniform cost problem stops before
  exceeding budget; a zero-cost cell is restored first.
- **Edge:** budget 0 → empty plan at baseline; no restorable → empty plan; `budget < 0` and bad
  `criterion` raise `ValueError`; result plan is a valid subset of `restorable` and
  `problem.restore_mesh(result.restored).mesh == result.mesh` (round-trip consistency).

**Target:** ~12–16 tests, `make check` green, parity 35.0 untouched (pure new subpackage, no solver
change).

## Out of scope (deferred, next pieces)

- **SA refiner** (flip restorable cells, accept by MESH with cooling) — a better-optima follow-on.
- **restoptr `min_restore` objective** (minimize restored cost to hit a MESH *target*) — the dual
  problem; `mesh_curve` already exposes the frontier a caller can threshold.
- **components / compactness constraints** (restoptr's `add_components_constraint` /
  `add_compactness_constraint`); locked-in/out beyond the `restorable` mask.
- **Incremental MESH delta** (union-find) to drop the per-candidate full recompute.
- **IIC/PC-maximizing optimization** (once those indices exist).
- The exact constraint-programming (Choco) solver — pymarxan stays greedy/heuristic.

## References

- restoptr: Justeau-Allaire et al. (2023) *Restoration Ecology* doi:10.1111/rec.13910
  (`set_max_mesh_objective` + `add_budget_constraint`); Justeau-Allaire et al. (2021) *J. Appl.
  Ecol.* doi:10.1111/1365-2664.13803. MESH: Jaeger (2000) doi:10.1023/A:1008129329289.
- Reuses `pymarxan.restoration.{RestorationProblem, compute_mesh}` (v0.28/v0.29). Precedent for a
  non-`Solver` optimizer returning a result dataclass: `pymarxan.zonation.rank_removal` →
  `ZonationResult`.
