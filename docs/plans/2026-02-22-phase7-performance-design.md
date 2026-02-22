# Phase 7: Solver Performance Optimization

**Date:** 2026-02-22
**Status:** Design Document
**Goal:** 100-1000x speedup for SA and Zone SA solvers via precomputed arrays and incremental delta computation, targeting medium-scale problems (1K-10K planning units).

---

## 1. Problem Statement

All pymarxan solvers currently recompute the **full objective function** from scratch on every iteration. For the SA solver, which runs 100K-1M iterations, this means:

- **`compute_objective()`** iterates all boundary rows with `iterrows()` — O(B) per call
- **`compute_penalty()`** iterates all feature rows with `iterrows()` — O(F*P) per call
- **`compute_boundary()`** iterates all boundary rows with `iterrows()` — O(B) per call

For a 5K PU / 100 feature / 10K boundary edges problem at 1M iterations, this is ~1M * (10K + 100*5K) = ~500 billion Python-level operations. The same pattern exists in Zone SA.

### What's Slow

| Operation | Current complexity per iteration | Target complexity |
|---|---|---|
| Cost delta | O(1) — already fast | O(1) |
| Boundary delta | O(B) — iterates all edges | O(degree of flipped PU) |
| Feature penalty delta | O(F*P) — iterates all PU-feature pairs | O(features in flipped PU) |
| Zone objective delta | O(B + F*P + Z) — same pattern | O(degree + features_in_pu) |

## 2. Solution: ProblemCache + Delta Computation

### 2.1 Core Idea

Precompute all DataFrame data into NumPy arrays **once** before the solve loop, then compute **incremental deltas** when a single PU is flipped, using only the local neighborhood of that PU.

### 2.2 ProblemCache

New module: `src/pymarxan/solvers/cache.py`

```python
@dataclass
class ProblemCache:
    """Precomputed NumPy arrays for fast solver operations.

    Built once from a ConservationProblem; used by SA and Zone SA
    for O(degree + features_per_pu) delta computation per flip.
    """
    n_pu: int
    n_feat: int

    # Core arrays
    costs: np.ndarray              # (n_pu,) float64
    statuses: np.ndarray           # (n_pu,) int32
    pu_id_to_idx: dict[int, int]

    # PU-feature matrix (dense)
    pu_feat_matrix: np.ndarray     # (n_pu, n_feat) float64 — amount contributed
    feat_targets: np.ndarray       # (n_feat,) float64
    feat_spf: np.ndarray           # (n_feat,) float64
    feat_id_to_col: dict[int, int]

    # Boundary adjacency
    neighbors: list[list[tuple[int, float]]]  # neighbors[i] = [(j, bval), ...]
    self_boundary: np.ndarray      # (n_pu,) float64 — diagonal boundary

    # Parameters
    blm: float
    misslevel: float
    cost_thresh: float
    thresh_pen1: float
    thresh_pen2: float

    @classmethod
    def from_problem(cls, problem: ConservationProblem) -> ProblemCache:
        """Build cache from a problem (one-time O(B + F*P) cost)."""
        ...

    def compute_held(self, selected: np.ndarray) -> np.ndarray:
        """Compute (n_feat,) amount held by selection. O(n_pu * n_feat)."""
        return self.pu_feat_matrix[selected].sum(axis=0)

    def compute_full_objective(
        self, selected: np.ndarray, held: np.ndarray
    ) -> float:
        """Full objective from arrays (no DataFrame iteration). O(B + F)."""
        ...

    def compute_delta_objective(
        self,
        idx: int,
        selected: np.ndarray,
        held: np.ndarray,
        total_cost: float,
    ) -> float:
        """Delta in objective when flipping PU idx.

        O(degree(idx) + features_in(idx)) — the key speedup.
        """
        ...
```

### 2.3 Delta Computation Detail

When flipping PU `idx` from unselected → selected (or vice versa):

**Cost delta:**
```
sign = +1 if adding, -1 if removing
delta_cost = sign * costs[idx]
```

**Boundary delta:**
```
delta_boundary = 0
for (neighbor, bval) in neighbors[idx]:
    if selected[neighbor] == selected[idx]:  # will become different
        delta_boundary += bval
    else:  # will become same
        delta_boundary -= bval
# Self-boundary
if adding:
    delta_boundary += self_boundary[idx]
else:
    delta_boundary -= self_boundary[idx]
```

**Penalty delta:**
```
delta_penalty = 0
feat_contribs = pu_feat_matrix[idx]  # (n_feat,) row
for f where feat_contribs[f] > 0:
    old_shortfall = max(0, feat_targets[f] * misslevel - held[f])
    new_held = held[f] + sign * feat_contribs[f]
    new_shortfall = max(0, feat_targets[f] * misslevel - new_held)
    delta_penalty += feat_spf[f] * (new_shortfall - old_shortfall)
```

**Cost threshold delta:**
```
new_cost = total_cost + delta_cost
delta_thresh = thresh_penalty(new_cost) - thresh_penalty(total_cost)
```

**Total delta:**
```
delta = delta_cost + blm * delta_boundary + delta_penalty + delta_thresh
```

### 2.4 SA Solver Changes

`src/pymarxan/solvers/simulated_annealing.py`:

- Build `ProblemCache` once at start
- Track `held` array (n_feat,) and `total_cost` float incrementally
- Main loop calls `cache.compute_delta_objective()` instead of `compute_objective()`
- When move accepted: `held += sign * cache.pu_feat_matrix[idx]`, `total_cost += delta_cost`
- Temperature estimation also uses delta approach
- Public API (`solve()` signature) unchanged

### 2.5 Zone SA Solver Changes

`src/pymarxan/zones/cache.py` — `ZoneProblemCache`:

```python
@dataclass
class ZoneProblemCache:
    """Precomputed arrays for zone SA delta computation."""
    base: ProblemCache              # standard PU/feature/boundary data
    zone_costs: np.ndarray          # (n_pu, n_zones) float64
    zone_contributions: np.ndarray  # (n_feat, n_zones) float64
    zone_boundary_costs: dict[tuple[int, int], float]
    zone_targets: np.ndarray        # (n_zones, n_feat) float64

    @classmethod
    def from_problem(cls, problem: ZonalProblem) -> ZoneProblemCache: ...

    def compute_delta_zone_objective(
        self, idx, old_zone, new_zone, assignment, held_per_zone, ...
    ) -> float: ...
```

`src/pymarxan/zones/solver.py`:
- Build `ZoneProblemCache` once
- Track `held_per_zone` as (n_zones, n_feat) array
- Delta computation per zone swap

## 3. Benchmark Suite

New `tests/benchmarks/` directory:

### 3.1 Synthetic Problem Generator

`tests/benchmarks/conftest.py`:
```python
def make_problem(n_pu: int, n_feat: int, density: float = 0.3) -> ConservationProblem:
    """Generate a random conservation problem for benchmarking."""
    ...
```

### 3.2 SA Benchmarks

`tests/benchmarks/bench_sa.py`:
- 100 PU / 10 features / 10K iterations → must complete in <1s
- 1K PU / 50 features / 100K iterations → must complete in <10s
- 5K PU / 100 features / 100K iterations → must complete in <30s

### 3.3 Zone SA Benchmarks

`tests/benchmarks/bench_zone_sa.py`:
- 100 PU / 10 features / 3 zones / 10K iterations → <2s
- 1K PU / 50 features / 3 zones / 100K iterations → <15s

## 4. File Changes

### New Files
| File | Purpose |
|---|---|
| `src/pymarxan/solvers/cache.py` | ProblemCache + delta computation |
| `src/pymarxan/zones/cache.py` | ZoneProblemCache + zone delta computation |
| `tests/pymarxan/solvers/test_cache.py` | ProblemCache unit tests |
| `tests/pymarxan/zones/test_zone_cache.py` | ZoneProblemCache unit tests |
| `tests/benchmarks/conftest.py` | Synthetic problem generator |
| `tests/benchmarks/bench_sa.py` | SA performance benchmarks |
| `tests/benchmarks/bench_zone_sa.py` | Zone SA performance benchmarks |

### Modified Files
| File | Change |
|---|---|
| `src/pymarxan/solvers/simulated_annealing.py` | Rewrite to use ProblemCache |
| `src/pymarxan/zones/solver.py` | Rewrite to use ZoneProblemCache |

### Unchanged
- All other solvers (MIP, heuristic, iterative improvement, run mode pipeline)
- `utils.py` — still used by `build_solution()` for final solution assembly
- All public APIs — same signatures, same behavior, just faster

## 5. Testing Strategy

1. **All 317 existing tests must continue to pass** — this is a refactor, not a feature change
2. **New cache unit tests** verify delta computation matches full computation
3. **Benchmarks** with timing assertions verify actual speedup
4. **Property**: for any problem and any flip, `delta == full_after - full_before`

## 6. Non-Goals (Deferred)

- Heuristic solver optimization (adequate speed for greedy)
- Iterative improvement optimization (called infrequently)
- Numba/Cython acceleration
- Multi-threading / parallel runs
- Scipy sparse matrix support
