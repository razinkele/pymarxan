# Phase 7: SA & Zone SA Performance Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 100-1000x speedup for the SA and Zone SA solvers via precomputed NumPy arrays and incremental delta computation, keeping all 317 existing tests passing.

**Architecture:** Introduce `ProblemCache` (standard) and `ZoneProblemCache` (zones) — one-time precomputation converting DataFrames to dense NumPy arrays plus adjacency lists. SA and Zone SA solvers then compute O(degree + features_per_pu) deltas per flip instead of O(B + F*P) full recomputation. All public APIs unchanged.

**Tech Stack:** Python 3.11+, NumPy (dense arrays), existing test fixtures

---

### Task 1: ProblemCache — Precomputed Arrays

**Files:**
- Create: `src/pymarxan/solvers/cache.py`
- Create: `tests/pymarxan/solvers/test_cache.py`

**Context:** The `ProblemCache` converts DataFrames into NumPy arrays once. It provides `compute_full_objective()` (no iterrows) and `compute_delta_objective()` (incremental per flip). This is the foundation all later tasks depend on.

**Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_cache.py`:

```python
"""Tests for ProblemCache precomputed arrays and delta computation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.cache import ProblemCache

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


@pytest.fixture()
def problem():
    return load_project(DATA_DIR)


@pytest.fixture()
def cache(problem):
    return ProblemCache.from_problem(problem)


class TestProblemCacheConstruction:
    def test_dimensions(self, cache, problem):
        assert cache.n_pu == problem.n_planning_units
        assert cache.n_feat == problem.n_features

    def test_costs_array(self, cache, problem):
        expected = problem.planning_units["cost"].values.astype(float)
        np.testing.assert_array_equal(cache.costs, expected)

    def test_statuses_array(self, cache, problem):
        expected = problem.planning_units["status"].values.astype(int)
        np.testing.assert_array_equal(cache.statuses, expected)

    def test_pu_feat_matrix_shape(self, cache):
        assert cache.pu_feat_matrix.shape == (cache.n_pu, cache.n_feat)

    def test_pu_feat_matrix_values(self, cache, problem):
        """Matrix entries match pu_vs_features rows."""
        for _, row in problem.pu_vs_features.iterrows():
            pu_idx = cache.pu_id_to_idx[int(row["pu"])]
            feat_col = cache.feat_id_to_col[int(row["species"])]
            assert cache.pu_feat_matrix[pu_idx, feat_col] == pytest.approx(
                float(row["amount"])
            )

    def test_feat_targets(self, cache, problem):
        for _, row in problem.features.iterrows():
            col = cache.feat_id_to_col[int(row["id"])]
            assert cache.feat_targets[col] == pytest.approx(float(row["target"]))

    def test_feat_spf(self, cache, problem):
        for _, row in problem.features.iterrows():
            col = cache.feat_id_to_col[int(row["id"])]
            assert cache.feat_spf[col] == pytest.approx(float(row["spf"]))

    def test_neighbors_symmetric(self, cache):
        """If j is in neighbors[i], then i is in neighbors[j]."""
        for i, adj in enumerate(cache.neighbors):
            for j, bval in adj:
                found = any(ni == i for ni, _ in cache.neighbors[j])
                assert found, f"Asymmetric: {j} not in neighbors of {i}"

    def test_self_boundary_shape(self, cache):
        assert cache.self_boundary.shape == (cache.n_pu,)


class TestComputeHeld:
    def test_all_selected(self, cache):
        selected = np.ones(cache.n_pu, dtype=bool)
        held = cache.compute_held(selected)
        assert held.shape == (cache.n_feat,)
        # Total held must equal column sums of pu_feat_matrix
        np.testing.assert_array_almost_equal(
            held, cache.pu_feat_matrix.sum(axis=0)
        )

    def test_none_selected(self, cache):
        selected = np.zeros(cache.n_pu, dtype=bool)
        held = cache.compute_held(selected)
        assert np.all(held == 0.0)


class TestFullObjective:
    def test_matches_utils(self, cache, problem):
        """Full objective from cache matches compute_objective from utils."""
        from pymarxan.solvers.utils import compute_objective

        blm = 1.0
        pu_ids = problem.planning_units["id"].tolist()
        pu_index = {pid: i for i, pid in enumerate(pu_ids)}

        rng = np.random.default_rng(42)
        for _ in range(20):
            selected = rng.random(cache.n_pu) > 0.5
            held = cache.compute_held(selected)
            total_cost = float(cache.costs[selected].sum())

            cache_obj = cache.compute_full_objective(selected, held, blm)
            utils_obj = compute_objective(problem, selected, pu_index, blm)

            assert cache_obj == pytest.approx(utils_obj, abs=1e-8), (
                f"Mismatch for selection {selected}"
            )


class TestDeltaObjective:
    def test_delta_matches_difference(self, cache):
        """Delta must equal full_after - full_before for every flip."""
        blm = 1.5
        rng = np.random.default_rng(123)
        selected = rng.random(cache.n_pu) > 0.5
        held = cache.compute_held(selected)
        total_cost = float(cache.costs[selected].sum())

        for idx in range(cache.n_pu):
            obj_before = cache.compute_full_objective(selected, held, blm)

            delta = cache.compute_delta_objective(
                idx, selected, held, total_cost, blm
            )

            # Actually flip
            sign = -1.0 if selected[idx] else 1.0
            selected[idx] = not selected[idx]
            held += sign * cache.pu_feat_matrix[idx]
            total_cost += sign * cache.costs[idx]

            obj_after = cache.compute_full_objective(selected, held, blm)

            assert delta == pytest.approx(
                obj_after - obj_before, abs=1e-8
            ), f"Delta mismatch for PU {idx}"

            # Flip back to restore state for next PU
            selected[idx] = not selected[idx]
            held -= sign * cache.pu_feat_matrix[idx]
            total_cost -= sign * cache.costs[idx]

    def test_delta_with_cost_threshold(self, cache, problem):
        """Delta accounts for COSTTHRESH penalty."""
        problem.parameters["COSTTHRESH"] = 20.0
        problem.parameters["THRESHPEN1"] = 5.0
        problem.parameters["THRESHPEN2"] = 1.0
        cache2 = ProblemCache.from_problem(problem)

        blm = 0.0
        rng = np.random.default_rng(7)
        selected = rng.random(cache2.n_pu) > 0.5
        held = cache2.compute_held(selected)
        total_cost = float(cache2.costs[selected].sum())

        for idx in range(cache2.n_pu):
            obj_before = cache2.compute_full_objective(selected, held, blm)
            delta = cache2.compute_delta_objective(
                idx, selected, held, total_cost, blm
            )
            sign = -1.0 if selected[idx] else 1.0
            selected[idx] = not selected[idx]
            held += sign * cache2.pu_feat_matrix[idx]
            total_cost += sign * cache2.costs[idx]
            obj_after = cache2.compute_full_objective(selected, held, blm)

            assert delta == pytest.approx(obj_after - obj_before, abs=1e-8)

            selected[idx] = not selected[idx]
            held -= sign * cache2.pu_feat_matrix[idx]
            total_cost -= sign * cache2.costs[idx]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/solvers/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymarxan.solvers.cache'`

**Step 3: Write the implementation**

Create `src/pymarxan/solvers/cache.py`:

```python
"""Precomputed NumPy arrays for fast solver operations.

Converts ConservationProblem DataFrames into dense arrays and adjacency
lists once, then provides O(degree + features_per_pu) incremental delta
computation for single-PU flips — the key to fast simulated annealing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymarxan.models.problem import ConservationProblem


@dataclass(frozen=True)
class ProblemCache:
    """Precomputed arrays for fast objective evaluation.

    Built once via ``from_problem()``. All solver hot loops should use
    this instead of iterating DataFrames.
    """

    n_pu: int
    n_feat: int

    # Core arrays
    costs: np.ndarray              # (n_pu,) float64
    statuses: np.ndarray           # (n_pu,) int32
    pu_id_to_idx: dict[int, int]

    # PU-feature matrix (dense)
    pu_feat_matrix: np.ndarray     # (n_pu, n_feat) float64
    feat_targets: np.ndarray       # (n_feat,) float64
    feat_spf: np.ndarray           # (n_feat,) float64
    feat_id_to_col: dict[int, int]

    # Boundary adjacency
    neighbors: list[list[tuple[int, float]]]  # neighbors[i] = [(j, bval)]
    self_boundary: np.ndarray      # (n_pu,) float64

    # Parameters (cached scalars)
    misslevel: float
    cost_thresh: float
    thresh_pen1: float
    thresh_pen2: float

    @classmethod
    def from_problem(cls, problem: ConservationProblem) -> ProblemCache:
        """Build cache from a ConservationProblem."""
        pu_ids = problem.planning_units["id"].values
        n_pu = len(pu_ids)
        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}

        costs = problem.planning_units["cost"].values.astype(np.float64)
        statuses = problem.planning_units["status"].values.astype(np.int32)

        # Feature indexing
        feat_ids = problem.features["id"].values
        n_feat = len(feat_ids)
        feat_id_to_col = {int(fid): c for c, fid in enumerate(feat_ids)}

        feat_targets = np.zeros(n_feat, dtype=np.float64)
        feat_spf = np.zeros(n_feat, dtype=np.float64)
        for _, row in problem.features.iterrows():
            col = feat_id_to_col[int(row["id"])]
            feat_targets[col] = float(row["target"])
            feat_spf[col] = float(row.get("spf", 1.0))

        # PU-feature matrix (dense)
        pu_feat_matrix = np.zeros((n_pu, n_feat), dtype=np.float64)
        for _, row in problem.pu_vs_features.iterrows():
            pu_idx = pu_id_to_idx.get(int(row["pu"]))
            feat_col = feat_id_to_col.get(int(row["species"]))
            if pu_idx is not None and feat_col is not None:
                pu_feat_matrix[pu_idx, feat_col] += float(row["amount"])

        # Boundary adjacency lists
        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_pu)]
        self_boundary = np.zeros(n_pu, dtype=np.float64)

        if problem.boundary is not None:
            for _, row in problem.boundary.iterrows():
                id1 = int(row["id1"])
                id2 = int(row["id2"])
                bval = float(row["boundary"])
                if id1 == id2:
                    idx = pu_id_to_idx.get(id1)
                    if idx is not None:
                        self_boundary[idx] = bval
                else:
                    idx1 = pu_id_to_idx.get(id1)
                    idx2 = pu_id_to_idx.get(id2)
                    if idx1 is not None and idx2 is not None:
                        neighbors[idx1].append((idx2, bval))
                        neighbors[idx2].append((idx1, bval))

        params = problem.parameters
        return cls(
            n_pu=n_pu,
            n_feat=n_feat,
            costs=costs,
            statuses=statuses,
            pu_id_to_idx=pu_id_to_idx,
            pu_feat_matrix=pu_feat_matrix,
            feat_targets=feat_targets,
            feat_spf=feat_spf,
            feat_id_to_col=feat_id_to_col,
            neighbors=neighbors,
            self_boundary=self_boundary,
            misslevel=float(params.get("MISSLEVEL", 1.0)),
            cost_thresh=float(params.get("COSTTHRESH", 0.0)),
            thresh_pen1=float(params.get("THRESHPEN1", 0.0)),
            thresh_pen2=float(params.get("THRESHPEN2", 0.0)),
        )

    # ------------------------------------------------------------------
    # Bulk computation (used once at init and for final solution)
    # ------------------------------------------------------------------

    def compute_held(self, selected: np.ndarray) -> np.ndarray:
        """Amount held per feature. Shape (n_feat,)."""
        return self.pu_feat_matrix[selected].sum(axis=0)

    def compute_full_objective(
        self,
        selected: np.ndarray,
        held: np.ndarray,
        blm: float,
    ) -> float:
        """Compute full objective from arrays (no DataFrame iteration)."""
        # Cost
        total_cost = float(self.costs[selected].sum())

        # Boundary
        boundary = 0.0
        for i in range(self.n_pu):
            if selected[i]:
                boundary += self.self_boundary[i]
                for j, bval in self.neighbors[i]:
                    if not selected[j]:
                        boundary += bval
            # Note: each edge counted once from each side, but we only
            # add when i is selected and j is not. For undirected edges
            # stored symmetrically, each (i,j) pair where exactly one
            # is selected adds bval once from the selected side.

        # Penalty
        effective_targets = self.feat_targets * self.misslevel
        shortfall = np.maximum(effective_targets - held, 0.0)
        penalty = float((self.feat_spf * shortfall).sum())

        obj = total_cost + blm * boundary + penalty

        # Cost threshold
        if self.cost_thresh > 0 and total_cost > self.cost_thresh:
            obj += self.thresh_pen1 + self.thresh_pen2 * (
                total_cost - self.cost_thresh
            )

        return obj

    # ------------------------------------------------------------------
    # Incremental delta (the key speedup — called per SA iteration)
    # ------------------------------------------------------------------

    def compute_delta_objective(
        self,
        idx: int,
        selected: np.ndarray,
        held: np.ndarray,
        total_cost: float,
        blm: float,
    ) -> float:
        """Change in objective when flipping PU idx.

        O(degree(idx) + features_in(idx)) — not O(n_pu * n_feat).

        Parameters
        ----------
        idx : int
            Index of the PU to flip.
        selected : np.ndarray
            Current selection (not modified).
        held : np.ndarray
            Current held amounts per feature (not modified).
        total_cost : float
            Current total cost of selected PUs.
        blm : float
            Boundary length modifier.
        """
        is_selected = selected[idx]
        sign = -1.0 if is_selected else 1.0  # removing vs adding

        # --- Cost delta ---
        delta_cost = sign * self.costs[idx]

        # --- Boundary delta ---
        delta_boundary = 0.0
        if blm != 0.0:
            # Self-boundary
            delta_boundary += sign * self.self_boundary[idx]
            # Neighbor edges
            for j, bval in self.neighbors[idx]:
                if is_selected:
                    # Removing idx: edge (idx, j) currently contributes
                    # bval if j is NOT selected. After removal, it
                    # contributes bval if j IS selected.
                    if selected[j]:
                        delta_boundary += bval   # new mismatch
                    else:
                        delta_boundary -= bval   # resolved mismatch
                else:
                    # Adding idx: edge (idx, j) currently contributes
                    # bval if j IS selected. After adding, it contributes
                    # bval if j is NOT selected.
                    if selected[j]:
                        delta_boundary -= bval   # resolved mismatch
                    else:
                        delta_boundary += bval   # new mismatch

        # --- Penalty delta ---
        delta_penalty = 0.0
        feat_contribs = self.pu_feat_matrix[idx]
        effective_targets = self.feat_targets * self.misslevel
        for f in range(self.n_feat):
            contrib = feat_contribs[f]
            if contrib == 0.0:
                continue
            old_shortfall = max(effective_targets[f] - held[f], 0.0)
            new_held_f = held[f] + sign * contrib
            new_shortfall = max(effective_targets[f] - new_held_f, 0.0)
            delta_penalty += self.feat_spf[f] * (
                new_shortfall - old_shortfall
            )

        # --- Cost threshold delta ---
        delta_thresh = 0.0
        if self.cost_thresh > 0:
            new_cost = total_cost + delta_cost
            old_thresh = 0.0
            if total_cost > self.cost_thresh:
                old_thresh = self.thresh_pen1 + self.thresh_pen2 * (
                    total_cost - self.cost_thresh
                )
            new_thresh = 0.0
            if new_cost > self.cost_thresh:
                new_thresh = self.thresh_pen1 + self.thresh_pen2 * (
                    new_cost - self.cost_thresh
                )
            delta_thresh = new_thresh - old_thresh

        return delta_cost + blm * delta_boundary + delta_penalty + delta_thresh
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pymarxan/solvers/test_cache.py -v`
Expected: All PASSED (~14 tests)

**Step 5: Lint and commit**

```bash
ruff check src/pymarxan/solvers/cache.py tests/pymarxan/solvers/test_cache.py --fix
git add src/pymarxan/solvers/cache.py tests/pymarxan/solvers/test_cache.py
git commit -m "feat: add ProblemCache with precomputed arrays and delta computation"
```

---

### Task 2: Rewrite SA Solver to Use ProblemCache

**Files:**
- Modify: `src/pymarxan/solvers/simulated_annealing.py`
- Tests: `tests/pymarxan/solvers/test_simulated_annealing.py` (existing, must still pass)

**Context:** Replace all `compute_objective()` calls in the SA hot loop with `cache.compute_delta_objective()`. Track `held` and `total_cost` incrementally. Public API unchanged — same `solve()` signature, same `Solution` output.

**Step 1: Run existing SA tests to establish baseline**

Run: `python -m pytest tests/pymarxan/solvers/test_simulated_annealing.py -v`
Expected: All PASSED (11 tests)

**Step 2: Rewrite simulated_annealing.py**

Replace `src/pymarxan/solvers/simulated_annealing.py` with:

```python
"""Native Python simulated annealing solver for Marxan conservation planning.

Uses ProblemCache for O(degree + features_per_pu) incremental objective
updates instead of full O(B + F*P) recomputation per iteration.
"""
from __future__ import annotations

import math

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.utils import build_solution


class SimulatedAnnealingSolver(Solver):
    """Simulated annealing solver implemented natively in Python/NumPy."""

    def __init__(
        self,
        num_iterations: int = 1_000_000,
        num_temp_steps: int = 10_000,
        initial_prop: float = 0.5,
    ):
        self._num_iterations = num_iterations
        self._num_temp_steps = num_temp_steps
        self._initial_prop = initial_prop

    def name(self) -> str:
        return "Simulated Annealing (Python)"

    def supports_zones(self) -> bool:
        return False

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig()

        cache = ProblemCache.from_problem(problem)
        blm = float(problem.parameters.get("BLM", 0.0))
        num_iterations = int(
            problem.parameters.get("NUMITNS", self._num_iterations)
        )
        num_temp_steps = int(
            problem.parameters.get("NUMTEMP", self._num_temp_steps)
        )
        initial_prop = float(
            problem.parameters.get("PROP", self._initial_prop)
        )

        # Identify locked and swappable PUs
        locked_in = set(np.where(cache.statuses == 2)[0])
        locked_out = set(np.where(cache.statuses == 3)[0])
        swappable = np.array(
            [i for i in range(cache.n_pu)
             if i not in locked_in and i not in locked_out],
            dtype=np.intp,
        )
        n_swappable = len(swappable)

        if n_swappable == 0:
            selected = np.zeros(cache.n_pu, dtype=bool)
            for idx in locked_in:
                selected[idx] = True
            sol = build_solution(
                problem, selected, blm, metadata={"solver": self.name()}
            )
            return [sol] * config.num_solutions

        solutions = []
        for run_idx in range(config.num_solutions):
            rng = np.random.default_rng(
                (config.seed + run_idx) if config.seed is not None else None
            )

            # Initialize selection
            selected = np.zeros(cache.n_pu, dtype=bool)
            for idx in locked_in:
                selected[idx] = True
            for idx in swappable:
                if rng.random() < initial_prop:
                    selected[idx] = True

            # Initialize tracked state
            held = cache.compute_held(selected)
            total_cost = float(cache.costs[selected].sum())
            current_obj = cache.compute_full_objective(selected, held, blm)

            # Estimate initial temperature via delta sampling
            deltas = []
            n_samples = min(1000, num_iterations // 10)
            for _ in range(n_samples):
                idx = swappable[rng.integers(n_swappable)]
                delta = cache.compute_delta_objective(
                    idx, selected, held, total_cost, blm
                )
                if delta > 0:
                    deltas.append(delta)

            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                initial_temp = -avg_delta / math.log(0.5)
            else:
                initial_temp = 1.0

            iters_per_step = max(1, num_iterations // num_temp_steps)
            alpha = (
                (0.001 / initial_temp) ** (1.0 / max(1, num_temp_steps))
                if initial_temp > 0
                else 0.99
            )

            # Main SA loop
            temp = initial_temp
            best_selected = selected.copy()
            best_held = held.copy()
            best_cost = total_cost
            best_obj = current_obj
            step_count = 0

            for _ in range(num_iterations):
                idx = int(swappable[rng.integers(n_swappable)])

                delta = cache.compute_delta_objective(
                    idx, selected, held, total_cost, blm
                )

                # Acceptance criterion
                if delta <= 0 or (
                    temp > 0 and rng.random() < math.exp(-delta / temp)
                ):
                    # Accept move
                    sign = -1.0 if selected[idx] else 1.0
                    selected[idx] = not selected[idx]
                    held += sign * cache.pu_feat_matrix[idx]
                    total_cost += sign * cache.costs[idx]
                    current_obj += delta

                    if current_obj < best_obj:
                        best_selected = selected.copy()
                        best_held = held.copy()
                        best_cost = total_cost
                        best_obj = current_obj

                # Cool
                step_count += 1
                if step_count >= iters_per_step:
                    temp *= alpha
                    step_count = 0

            sol = build_solution(
                problem, best_selected, blm,
                metadata={
                    "solver": self.name(),
                    "run": run_idx + 1,
                    "initial_temp": round(initial_temp, 4),
                    "final_temp": round(temp, 6),
                    "best_objective": round(best_obj, 4),
                },
            )
            solutions.append(sol)

        return solutions
```

**Step 3: Run existing SA tests**

Run: `python -m pytest tests/pymarxan/solvers/test_simulated_annealing.py -v`
Expected: All 11 PASSED (same behavior, faster execution)

**Step 4: Run full test suite (fast subset)**

Run: `python -m pytest tests/pymarxan/solvers/ tests/pymarxan/io/ tests/pymarxan/models/ tests/test_integration.py tests/test_integration_phase2.py tests/test_integration_phase6.py -v --tb=short -k "not (test_cost_nonneg or test_seed_reprod or test_finds_feasible)"`
Expected: All PASSED (no regressions)

**Step 5: Lint and commit**

```bash
ruff check src/pymarxan/solvers/simulated_annealing.py --fix
git add src/pymarxan/solvers/simulated_annealing.py
git commit -m "perf: rewrite SA solver to use ProblemCache delta computation"
```

---

### Task 3: ZoneProblemCache — Precomputed Zone Arrays

**Files:**
- Create: `src/pymarxan/zones/cache.py`
- Create: `tests/pymarxan/zones/test_zone_cache.py`

**Context:** Extends the caching pattern for multi-zone problems. Zone SA needs per-zone cost arrays, contribution matrices, zone boundary cost lookups, and zone targets — all as NumPy arrays. The delta computation handles zone assignment changes (PU moving from zone A to zone B).

**Step 1: Write the failing tests**

Create `tests/pymarxan/zones/test_zone_cache.py`:

```python
"""Tests for ZoneProblemCache precomputed arrays and delta computation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.objective import compute_zone_objective
from pymarxan.zones.readers import load_zone_project

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


@pytest.fixture()
def problem():
    return load_zone_project(DATA_DIR)


@pytest.fixture()
def cache(problem):
    return ZoneProblemCache.from_problem(problem)


class TestZoneCacheConstruction:
    def test_n_pu(self, cache, problem):
        assert cache.n_pu == problem.n_planning_units

    def test_n_feat(self, cache, problem):
        assert cache.n_feat == problem.n_features

    def test_n_zones(self, cache, problem):
        assert cache.n_zones == problem.n_zones

    def test_zone_cost_matrix_shape(self, cache):
        # (n_pu, n_zones + 1) — column 0 for unassigned
        assert cache.zone_cost_matrix.shape[0] == cache.n_pu

    def test_contribution_matrix_shape(self, cache):
        assert cache.contribution_matrix.shape == (
            cache.n_zones + 1, cache.n_feat
        )


class TestZoneDeltaObjective:
    def test_delta_matches_difference(self, cache, problem):
        """Delta must equal full_after - full_before for every zone change."""
        blm = 1.0
        rng = np.random.default_rng(42)
        zone_options = [0] + sorted(problem.zone_ids)

        assignment = np.zeros(cache.n_pu, dtype=int)
        for i in range(cache.n_pu):
            assignment[i] = zone_options[rng.integers(len(zone_options))]

        held_per_zone = cache.compute_held_per_zone(assignment)
        obj_before = cache.compute_full_zone_objective(
            assignment, held_per_zone, blm
        )

        for idx in range(cache.n_pu):
            old_zone = int(assignment[idx])
            new_zone = zone_options[rng.integers(len(zone_options))]
            if new_zone == old_zone:
                continue

            delta = cache.compute_delta_zone_objective(
                idx, old_zone, new_zone, assignment, held_per_zone, blm
            )

            # Actually apply
            assignment[idx] = new_zone
            cache.update_held_per_zone(
                held_per_zone, idx, old_zone, new_zone
            )
            obj_after = cache.compute_full_zone_objective(
                assignment, held_per_zone, blm
            )

            assert delta == pytest.approx(
                obj_after - obj_before, abs=1e-8
            ), f"Delta mismatch at PU {idx}: old={old_zone}, new={new_zone}"

            obj_before = obj_after

    def test_full_objective_matches_original(self, cache, problem):
        """Full objective from cache matches zones.objective module."""
        blm = 1.0
        rng = np.random.default_rng(7)
        zone_options = [0] + sorted(problem.zone_ids)

        for _ in range(10):
            assignment = np.array(
                [zone_options[rng.integers(len(zone_options))]
                 for _ in range(cache.n_pu)],
                dtype=int,
            )
            held = cache.compute_held_per_zone(assignment)
            cache_obj = cache.compute_full_zone_objective(
                assignment, held, blm
            )
            orig_obj = compute_zone_objective(problem, assignment, blm)
            assert cache_obj == pytest.approx(orig_obj, abs=1e-8)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pymarxan/zones/test_zone_cache.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/pymarxan/zones/cache.py`:

```python
"""Precomputed NumPy arrays for fast zone SA operations.

Extends ProblemCache with per-zone cost matrices, contribution factors,
zone boundary cost lookups, and zone target arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymarxan.zones.model import ZonalProblem


@dataclass(frozen=True)
class ZoneProblemCache:
    """Precomputed arrays for fast zone SA delta computation."""

    n_pu: int
    n_feat: int
    n_zones: int  # actual zone count (not counting zone 0)

    # Zone ID mapping: zone_id → column index (0 = unassigned, 1..n = zones)
    zone_id_to_col: dict[int, int]

    # Core arrays (same as ProblemCache)
    costs: np.ndarray              # (n_pu,) unused for zones but kept
    pu_feat_matrix: np.ndarray     # (n_pu, n_feat) float64
    feat_targets: np.ndarray       # (n_feat,) float64 — standard targets
    feat_spf: np.ndarray           # (n_feat,) float64
    statuses: np.ndarray           # (n_pu,) int32
    pu_id_to_idx: dict[int, int]
    feat_id_to_col: dict[int, int]

    # Zone-specific arrays
    zone_cost_matrix: np.ndarray   # (n_pu, n_zones+1) float64
    contribution_matrix: np.ndarray  # (n_zones+1, n_feat) float64
    zone_target_matrix: np.ndarray   # (n_zones+1, n_feat) float64

    # Boundary
    neighbors: list[list[tuple[int, float]]]
    self_boundary: np.ndarray
    zone_boundary_costs: dict[tuple[int, int], float]

    @classmethod
    def from_problem(cls, problem: ZonalProblem) -> ZoneProblemCache:
        """Build zone cache from a ZonalProblem."""
        pu_ids = problem.planning_units["id"].values
        n_pu = len(pu_ids)
        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}

        feat_ids = problem.features["id"].values
        n_feat = len(feat_ids)
        feat_id_to_col = {int(fid): c for c, fid in enumerate(feat_ids)}

        zone_ids_sorted = sorted(problem.zone_ids)
        n_zones = len(zone_ids_sorted)
        # Column 0 = unassigned, columns 1..n = zone IDs
        zone_id_to_col = {0: 0}
        for i, zid in enumerate(zone_ids_sorted):
            zone_id_to_col[zid] = i + 1

        costs = problem.planning_units["cost"].values.astype(np.float64)
        statuses = problem.planning_units["status"].values.astype(np.int32)

        feat_spf = np.ones(n_feat, dtype=np.float64)
        feat_targets = np.zeros(n_feat, dtype=np.float64)
        for _, row in problem.features.iterrows():
            col = feat_id_to_col[int(row["id"])]
            feat_targets[col] = float(row["target"])
            feat_spf[col] = float(row.get("spf", 1.0))

        # PU-feature matrix
        pu_feat_matrix = np.zeros((n_pu, n_feat), dtype=np.float64)
        for _, row in problem.pu_vs_features.iterrows():
            pu_idx = pu_id_to_idx.get(int(row["pu"]))
            feat_col = feat_id_to_col.get(int(row["species"]))
            if pu_idx is not None and feat_col is not None:
                pu_feat_matrix[pu_idx, feat_col] += float(row["amount"])

        # Zone cost matrix: (n_pu, n_zones+1) — column 0 always 0
        zone_cost_matrix = np.zeros((n_pu, n_zones + 1), dtype=np.float64)
        for _, row in problem.zone_costs.iterrows():
            pu_idx = pu_id_to_idx.get(int(row["pu"]))
            z_col = zone_id_to_col.get(int(row["zone"]))
            if pu_idx is not None and z_col is not None:
                zone_cost_matrix[pu_idx, z_col] = float(row["cost"])

        # Contribution matrix: (n_zones+1, n_feat) — how much a zone
        # contributes to each feature. Default 1.0 for all real zones.
        contribution_matrix = np.zeros(
            (n_zones + 1, n_feat), dtype=np.float64
        )
        # Default: real zones contribute 1.0
        contribution_matrix[1:, :] = 1.0
        if problem.zone_contributions is not None:
            for _, row in problem.zone_contributions.iterrows():
                z_col = zone_id_to_col.get(int(row["zone"]))
                f_col = feat_id_to_col.get(int(row["feature"]))
                if z_col is not None and f_col is not None:
                    contribution_matrix[z_col, f_col] = float(
                        row["contribution"]
                    )

        # Zone target matrix: (n_zones+1, n_feat)
        zone_target_matrix = np.zeros(
            (n_zones + 1, n_feat), dtype=np.float64
        )
        if problem.zone_targets is not None:
            for _, row in problem.zone_targets.iterrows():
                z_col = zone_id_to_col.get(int(row["zone"]))
                f_col = feat_id_to_col.get(int(row["feature"]))
                if z_col is not None and f_col is not None:
                    zone_target_matrix[z_col, f_col] = float(row["target"])

        # Boundary adjacency
        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_pu)]
        self_boundary = np.zeros(n_pu, dtype=np.float64)
        if problem.boundary is not None:
            for _, row in problem.boundary.iterrows():
                id1 = int(row["id1"])
                id2 = int(row["id2"])
                bval = float(row["boundary"])
                if id1 == id2:
                    idx = pu_id_to_idx.get(id1)
                    if idx is not None:
                        self_boundary[idx] = bval
                else:
                    idx1 = pu_id_to_idx.get(id1)
                    idx2 = pu_id_to_idx.get(id2)
                    if idx1 is not None and idx2 is not None:
                        neighbors[idx1].append((idx2, bval))
                        neighbors[idx2].append((idx1, bval))

        # Zone boundary costs
        zbc: dict[tuple[int, int], float] = {}
        if problem.zone_boundary_costs is not None:
            for _, row in problem.zone_boundary_costs.iterrows():
                z1 = int(row["zone1"])
                z2 = int(row["zone2"])
                zbc[(z1, z2)] = float(row["cost"])

        return cls(
            n_pu=n_pu,
            n_feat=n_feat,
            n_zones=n_zones,
            zone_id_to_col=zone_id_to_col,
            costs=costs,
            pu_feat_matrix=pu_feat_matrix,
            feat_targets=feat_targets,
            feat_spf=feat_spf,
            statuses=statuses,
            pu_id_to_idx=pu_id_to_idx,
            feat_id_to_col=feat_id_to_col,
            zone_cost_matrix=zone_cost_matrix,
            contribution_matrix=contribution_matrix,
            zone_target_matrix=zone_target_matrix,
            neighbors=neighbors,
            self_boundary=self_boundary,
            zone_boundary_costs=zbc,
        )

    # ------------------------------------------------------------------
    # Bulk computation
    # ------------------------------------------------------------------

    def compute_held_per_zone(
        self, assignment: np.ndarray
    ) -> np.ndarray:
        """Held amount per (zone, feature). Shape (n_zones+1, n_feat)."""
        held = np.zeros((self.n_zones + 1, self.n_feat), dtype=np.float64)
        for i in range(self.n_pu):
            z_col = self.zone_id_to_col.get(int(assignment[i]), 0)
            held[z_col] += (
                self.pu_feat_matrix[i]
                * self.contribution_matrix[z_col]
            )
        return held

    def update_held_per_zone(
        self,
        held: np.ndarray,
        idx: int,
        old_zone: int,
        new_zone: int,
    ) -> None:
        """Update held in-place after moving PU idx from old to new zone."""
        old_col = self.zone_id_to_col.get(old_zone, 0)
        new_col = self.zone_id_to_col.get(new_zone, 0)
        held[old_col] -= (
            self.pu_feat_matrix[idx] * self.contribution_matrix[old_col]
        )
        held[new_col] += (
            self.pu_feat_matrix[idx] * self.contribution_matrix[new_col]
        )

    def compute_full_zone_objective(
        self,
        assignment: np.ndarray,
        held_per_zone: np.ndarray,
        blm: float,
    ) -> float:
        """Full zone objective from arrays."""
        # Zone cost
        total_cost = 0.0
        for i in range(self.n_pu):
            z_col = self.zone_id_to_col.get(int(assignment[i]), 0)
            total_cost += self.zone_cost_matrix[i, z_col]

        # Standard boundary (selected = zone > 0)
        selected = assignment > 0
        std_boundary = 0.0
        for i in range(self.n_pu):
            if selected[i]:
                std_boundary += self.self_boundary[i]
                for j, bval in self.neighbors[i]:
                    if not selected[j]:
                        std_boundary += bval

        # Zone boundary
        zone_boundary = 0.0
        for i in range(self.n_pu):
            for j, _ in self.neighbors[i]:
                if i >= j:
                    continue  # count each edge once
                zi = int(assignment[i])
                zj = int(assignment[j])
                if zi == 0 or zj == 0 or zi == zj:
                    continue
                zone_boundary += self.zone_boundary_costs.get((zi, zj), 0.0)

        # Zone penalty
        penalty = 0.0
        for z_col in range(1, self.n_zones + 1):
            for f in range(self.n_feat):
                target = self.zone_target_matrix[z_col, f]
                if target > 0:
                    shortfall = max(target - held_per_zone[z_col, f], 0.0)
                    penalty += self.feat_spf[f] * shortfall

        return total_cost + blm * std_boundary + zone_boundary + penalty

    # ------------------------------------------------------------------
    # Incremental delta
    # ------------------------------------------------------------------

    def compute_delta_zone_objective(
        self,
        idx: int,
        old_zone: int,
        new_zone: int,
        assignment: np.ndarray,
        held_per_zone: np.ndarray,
        blm: float,
    ) -> float:
        """Delta in zone objective when moving PU idx from old_zone to new_zone."""
        old_col = self.zone_id_to_col.get(old_zone, 0)
        new_col = self.zone_id_to_col.get(new_zone, 0)

        # Cost delta
        delta_cost = (
            self.zone_cost_matrix[idx, new_col]
            - self.zone_cost_matrix[idx, old_col]
        )

        # Standard boundary delta (selected = zone > 0)
        was_selected = old_zone > 0
        will_be_selected = new_zone > 0
        delta_std_boundary = 0.0

        if was_selected != will_be_selected and blm != 0.0:
            sign = -1.0 if was_selected else 1.0
            delta_std_boundary += sign * self.self_boundary[idx]
            for j, bval in self.neighbors[idx]:
                j_selected = int(assignment[j]) > 0
                if was_selected:
                    # Removing from selected
                    if j_selected:
                        delta_std_boundary += bval
                    else:
                        delta_std_boundary -= bval
                else:
                    # Adding to selected
                    if j_selected:
                        delta_std_boundary -= bval
                    else:
                        delta_std_boundary += bval

        # Zone boundary delta
        delta_zone_boundary = 0.0
        for j, _ in self.neighbors[idx]:
            zj = int(assignment[j])
            if zj == 0:
                continue
            # Remove old contribution
            if old_zone != 0 and old_zone != zj:
                delta_zone_boundary -= self.zone_boundary_costs.get(
                    (old_zone, zj), 0.0
                )
            # Add new contribution
            if new_zone != 0 and new_zone != zj:
                delta_zone_boundary += self.zone_boundary_costs.get(
                    (new_zone, zj), 0.0
                )

        # Zone penalty delta
        delta_penalty = 0.0
        feat_row = self.pu_feat_matrix[idx]
        for f in range(self.n_feat):
            if feat_row[f] == 0.0:
                continue
            # Old zone contribution
            if old_zone != 0:
                old_target = self.zone_target_matrix[old_col, f]
                if old_target > 0:
                    old_shortfall = max(
                        old_target - held_per_zone[old_col, f], 0.0
                    )
                    new_held = (
                        held_per_zone[old_col, f]
                        - feat_row[f] * self.contribution_matrix[old_col, f]
                    )
                    new_shortfall = max(old_target - new_held, 0.0)
                    delta_penalty += self.feat_spf[f] * (
                        new_shortfall - old_shortfall
                    )
            # New zone contribution
            if new_zone != 0:
                new_target = self.zone_target_matrix[new_col, f]
                if new_target > 0:
                    old_shortfall = max(
                        new_target - held_per_zone[new_col, f], 0.0
                    )
                    new_held = (
                        held_per_zone[new_col, f]
                        + feat_row[f] * self.contribution_matrix[new_col, f]
                    )
                    new_shortfall = max(new_target - new_held, 0.0)
                    delta_penalty += self.feat_spf[f] * (
                        new_shortfall - old_shortfall
                    )

        return (
            delta_cost
            + blm * delta_std_boundary
            + delta_zone_boundary
            + delta_penalty
        )
```

**Step 4: Run tests**

Run: `python -m pytest tests/pymarxan/zones/test_zone_cache.py -v`
Expected: All PASSED (~5 tests)

**Step 5: Lint and commit**

```bash
ruff check src/pymarxan/zones/cache.py tests/pymarxan/zones/test_zone_cache.py --fix
git add src/pymarxan/zones/cache.py tests/pymarxan/zones/test_zone_cache.py
git commit -m "feat: add ZoneProblemCache with zone delta computation"
```

---

### Task 4: Rewrite Zone SA Solver to Use ZoneProblemCache

**Files:**
- Modify: `src/pymarxan/zones/solver.py`
- Tests: `tests/pymarxan/zones/test_solver.py` (existing, must still pass)

**Context:** Same pattern as Task 2 but for zones. Replace all `compute_zone_objective()` calls with `cache.compute_delta_zone_objective()`. Track `held_per_zone` and zone costs incrementally.

**Step 1: Run existing zone SA tests to establish baseline**

Run: `python -m pytest tests/pymarxan/zones/test_solver.py -v`
Expected: All PASSED (7 tests, slow ~30s+)

**Step 2: Rewrite solver.py**

Replace `src/pymarxan/zones/solver.py` with:

```python
"""Simulated annealing solver for multi-zone conservation planning.

Uses ZoneProblemCache for O(degree + features_per_pu) incremental
objective updates instead of full recomputation per iteration.
"""
from __future__ import annotations

import math

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import (
    check_zone_targets,
    compute_standard_boundary,
    compute_zone_boundary,
    compute_zone_cost,
)


class ZoneSASolver(Solver):
    def __init__(
        self,
        num_iterations: int = 1_000_000,
        num_temp_steps: int = 10_000,
    ):
        self._num_iterations = num_iterations
        self._num_temp_steps = num_temp_steps

    def name(self) -> str:
        return "Zone SA (Python)"

    def supports_zones(self) -> bool:
        return True

    def solve(
        self,
        problem: ConservationProblem,
        config: SolverConfig | None = None,
    ) -> list[Solution]:
        if not isinstance(problem, ZonalProblem):
            raise TypeError(
                "ZoneSASolver requires a ZonalProblem instance"
            )
        if config is None:
            config = SolverConfig()

        cache = ZoneProblemCache.from_problem(problem)
        blm = float(problem.parameters.get("BLM", 0.0))
        num_iterations = int(
            problem.parameters.get("NUMITNS", self._num_iterations)
        )
        num_temp_steps = int(
            problem.parameters.get("NUMTEMP", self._num_temp_steps)
        )

        zone_ids_list = sorted(problem.zone_ids)
        zone_options = [0] + zone_ids_list

        locked: dict[int, int] = {}
        if "status" in problem.planning_units.columns:
            pu_ids = problem.planning_units["id"].tolist()
            for _, row in problem.planning_units.iterrows():
                s = int(row["status"])
                idx = pu_ids.index(int(row["id"]))
                if s == 2:
                    locked[idx] = zone_ids_list[0]
                elif s == 3:
                    locked[idx] = 0

        swappable = np.array(
            [i for i in range(cache.n_pu) if i not in locked],
            dtype=np.intp,
        )
        n_swappable = len(swappable)
        n_zone_options = len(zone_options)

        solutions = []
        for run_idx in range(config.num_solutions):
            rng = np.random.default_rng(
                (config.seed + run_idx) if config.seed is not None else None
            )

            assignment = np.zeros(cache.n_pu, dtype=int)
            for idx, zid in locked.items():
                assignment[idx] = zid
            for idx in swappable:
                assignment[idx] = zone_options[rng.integers(n_zone_options)]

            held_per_zone = cache.compute_held_per_zone(assignment)
            current_obj = cache.compute_full_zone_objective(
                assignment, held_per_zone, blm
            )

            # Estimate initial temperature
            deltas = []
            n_samples = min(1000, num_iterations // 10)
            for _ in range(n_samples):
                idx = int(swappable[rng.integers(n_swappable)])
                old_zone = int(assignment[idx])
                new_zone = zone_options[rng.integers(n_zone_options)]
                if new_zone == old_zone:
                    continue
                delta = cache.compute_delta_zone_objective(
                    idx, old_zone, new_zone,
                    assignment, held_per_zone, blm,
                )
                if delta > 0:
                    deltas.append(delta)

            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                initial_temp = -avg_delta / math.log(0.5)
            else:
                initial_temp = 1.0

            iters_per_step = max(1, num_iterations // num_temp_steps)
            alpha = (
                (0.001 / initial_temp) ** (1.0 / max(1, num_temp_steps))
                if initial_temp > 0
                else 0.99
            )

            temp = initial_temp
            best_assignment = assignment.copy()
            best_obj = current_obj
            step_count = 0

            for _ in range(num_iterations):
                idx = int(swappable[rng.integers(n_swappable)])
                old_zone = int(assignment[idx])
                new_zone = zone_options[rng.integers(n_zone_options)]
                if new_zone == old_zone:
                    continue

                delta = cache.compute_delta_zone_objective(
                    idx, old_zone, new_zone,
                    assignment, held_per_zone, blm,
                )

                if delta <= 0 or (
                    temp > 0 and rng.random() < math.exp(-delta / temp)
                ):
                    assignment[idx] = new_zone
                    cache.update_held_per_zone(
                        held_per_zone, idx, old_zone, new_zone
                    )
                    current_obj += delta

                    if current_obj < best_obj:
                        best_assignment = assignment.copy()
                        best_obj = current_obj

                step_count += 1
                if step_count >= iters_per_step:
                    temp *= alpha
                    step_count = 0

            # Build final solution using original functions for metadata
            selected = best_assignment > 0
            cost = compute_zone_cost(problem, best_assignment)
            std_boundary = compute_standard_boundary(
                problem, best_assignment
            )
            zone_boundary = compute_zone_boundary(
                problem, best_assignment
            )
            zone_targets = check_zone_targets(problem, best_assignment)

            sol = Solution(
                selected=selected,
                cost=cost,
                boundary=std_boundary,
                objective=best_obj,
                targets_met={},
                zone_assignment=best_assignment.copy(),
                metadata={
                    "solver": self.name(),
                    "run": run_idx + 1,
                    "zone_boundary_cost": round(zone_boundary, 4),
                    "zone_targets_met": {
                        f"z{z}_f{f}": v
                        for (z, f), v in zone_targets.items()
                    },
                },
            )
            solutions.append(sol)

        return solutions
```

**Step 3: Run existing zone SA tests**

Run: `python -m pytest tests/pymarxan/zones/test_solver.py -v`
Expected: All 7 PASSED (should be noticeably faster)

**Step 4: Run broader regression check**

Run: `python -m pytest tests/pymarxan/zones/ -v --tb=short`
Expected: All zone tests PASSED

**Step 5: Lint and commit**

```bash
ruff check src/pymarxan/zones/solver.py --fix
git add src/pymarxan/zones/solver.py
git commit -m "perf: rewrite Zone SA solver to use ZoneProblemCache delta computation"
```

---

### Task 5: Benchmark Suite

**Files:**
- Create: `tests/benchmarks/conftest.py`
- Create: `tests/benchmarks/bench_sa.py`
- Create: `tests/benchmarks/bench_zone_sa.py`
- Create: `tests/benchmarks/__init__.py`

**Context:** Synthetic problem generators and timed benchmark tests. These validate that the performance targets are actually met. They're in a separate directory and can be run independently from the unit test suite.

**Step 1: Create conftest with synthetic problem generators**

Create `tests/benchmarks/__init__.py` (empty).

Create `tests/benchmarks/conftest.py`:

```python
"""Synthetic problem generators for performance benchmarking."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.zones.model import ZonalProblem


def make_problem(
    n_pu: int = 1000,
    n_feat: int = 50,
    density: float = 0.3,
    seed: int = 42,
) -> ConservationProblem:
    """Generate a random conservation problem for benchmarking.

    Parameters
    ----------
    n_pu : int
        Number of planning units.
    n_feat : int
        Number of conservation features.
    density : float
        Fraction of PU-feature pairs with nonzero amounts.
    seed : int
        Random seed.
    """
    rng = np.random.default_rng(seed)

    pu = pd.DataFrame({
        "id": range(1, n_pu + 1),
        "cost": rng.uniform(1.0, 100.0, n_pu),
        "status": np.zeros(n_pu, dtype=int),
    })

    feat = pd.DataFrame({
        "id": range(1, n_feat + 1),
        "name": [f"f{i}" for i in range(1, n_feat + 1)],
        "target": rng.uniform(10.0, 50.0, n_feat),
        "spf": np.ones(n_feat),
    })

    # Sparse PU-feature relationships
    mask = rng.random((n_pu, n_feat)) < density
    rows, cols = np.where(mask)
    amounts = rng.uniform(0.1, 10.0, len(rows))
    puvspr = pd.DataFrame({
        "species": cols + 1,
        "pu": rows + 1,
        "amount": amounts,
    })

    # Grid-like boundary: each PU connected to next
    id1_list = []
    id2_list = []
    bnd_list = []
    side = int(np.ceil(np.sqrt(n_pu)))
    for i in range(1, n_pu + 1):
        # Right neighbor
        if i % side != 0 and i + 1 <= n_pu:
            id1_list.append(i)
            id2_list.append(i + 1)
            bnd_list.append(1.0)
        # Bottom neighbor
        if i + side <= n_pu:
            id1_list.append(i)
            id2_list.append(i + side)
            bnd_list.append(1.0)

    boundary = pd.DataFrame({
        "id1": id1_list, "id2": id2_list, "boundary": bnd_list,
    })

    return ConservationProblem(
        planning_units=pu,
        features=feat,
        pu_vs_features=puvspr,
        boundary=boundary,
        parameters={"BLM": 1.0},
    )


def make_zone_problem(
    n_pu: int = 500,
    n_feat: int = 20,
    n_zones: int = 3,
    density: float = 0.3,
    seed: int = 42,
) -> ZonalProblem:
    """Generate a random zone problem for benchmarking."""
    base = make_problem(n_pu, n_feat, density, seed)
    rng = np.random.default_rng(seed + 1000)

    zones = pd.DataFrame({
        "id": range(1, n_zones + 1),
        "name": [f"zone_{i}" for i in range(1, n_zones + 1)],
    })

    # Zone costs: each PU has a cost in each zone
    zc_rows = []
    for pid in range(1, n_pu + 1):
        for zid in range(1, n_zones + 1):
            zc_rows.append({
                "pu": pid, "zone": zid,
                "cost": rng.uniform(1.0, 50.0),
            })
    zone_costs = pd.DataFrame(zc_rows)

    # Zone targets
    zt_rows = []
    for zid in range(1, n_zones + 1):
        for fid in range(1, n_feat + 1):
            zt_rows.append({
                "zone": zid, "feature": fid,
                "target": rng.uniform(2.0, 10.0),
            })
    zone_targets = pd.DataFrame(zt_rows)

    # Zone boundary costs
    zbc_rows = []
    for z1 in range(1, n_zones + 1):
        for z2 in range(z1 + 1, n_zones + 1):
            zbc_rows.append({
                "zone1": z1, "zone2": z2, "cost": rng.uniform(0.5, 2.0),
            })
    zone_boundary_costs = pd.DataFrame(zbc_rows)

    return ZonalProblem(
        planning_units=base.planning_units,
        features=base.features,
        pu_vs_features=base.pu_vs_features,
        boundary=base.boundary,
        parameters={"BLM": 1.0, "NUMITNS": 10_000, "NUMTEMP": 100},
        zones=zones,
        zone_costs=zone_costs,
        zone_targets=zone_targets,
        zone_boundary_costs=zone_boundary_costs,
    )
```

**Step 2: Create SA benchmarks**

Create `tests/benchmarks/bench_sa.py`:

```python
"""Performance benchmarks for SA solver.

Run with: pytest tests/benchmarks/bench_sa.py -v
"""
from __future__ import annotations

import time

import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

from .conftest import make_problem


class TestSAPerformance:
    def test_small_100pu_10feat(self):
        """100 PU, 10 features, 10K iterations — must complete in <2s."""
        problem = make_problem(n_pu=100, n_feat=10)
        problem.parameters["NUMITNS"] = 10_000
        problem.parameters["NUMTEMP"] = 100
        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert solutions[0].cost > 0
        assert elapsed < 2.0, f"Took {elapsed:.2f}s, expected <2s"

    def test_medium_1k_pu_50feat(self):
        """1K PU, 50 features, 100K iterations — must complete in <15s."""
        problem = make_problem(n_pu=1000, n_feat=50)
        problem.parameters["NUMITNS"] = 100_000
        problem.parameters["NUMTEMP"] = 1000
        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert solutions[0].cost > 0
        assert elapsed < 15.0, f"Took {elapsed:.2f}s, expected <15s"

    def test_large_5k_pu_100feat(self):
        """5K PU, 100 features, 100K iterations — must complete in <60s."""
        problem = make_problem(n_pu=5000, n_feat=100)
        problem.parameters["NUMITNS"] = 100_000
        problem.parameters["NUMTEMP"] = 1000
        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert solutions[0].cost > 0
        assert elapsed < 60.0, f"Took {elapsed:.2f}s, expected <60s"
```

**Step 3: Create Zone SA benchmarks**

Create `tests/benchmarks/bench_zone_sa.py`:

```python
"""Performance benchmarks for Zone SA solver.

Run with: pytest tests/benchmarks/bench_zone_sa.py -v
"""
from __future__ import annotations

import time

import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.solver import ZoneSASolver

from .conftest import make_zone_problem


class TestZoneSAPerformance:
    def test_small_100pu_3zones(self):
        """100 PU, 10 features, 3 zones, 10K iterations — must complete in <3s."""
        problem = make_zone_problem(n_pu=100, n_feat=10, n_zones=3)
        problem.parameters["NUMITNS"] = 10_000
        problem.parameters["NUMTEMP"] = 100
        solver = ZoneSASolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert elapsed < 3.0, f"Took {elapsed:.2f}s, expected <3s"

    def test_medium_1k_pu_3zones(self):
        """1K PU, 50 features, 3 zones, 100K iterations — must complete in <30s."""
        problem = make_zone_problem(n_pu=1000, n_feat=50, n_zones=3)
        problem.parameters["NUMITNS"] = 100_000
        problem.parameters["NUMTEMP"] = 1000
        solver = ZoneSASolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert elapsed < 30.0, f"Took {elapsed:.2f}s, expected <30s"
```

**Step 4: Run benchmarks**

Run: `python -m pytest tests/benchmarks/ -v`
Expected: All PASSED within timing bounds

**Step 5: Commit**

```bash
git add tests/benchmarks/
git commit -m "feat: add SA and Zone SA performance benchmark suite"
```

---

### Task 6: Full Regression Test + Lint + Types

**Files:** All modified/created files from Tasks 1-5

**Step 1: Run ruff**

Run: `ruff check src/ tests/ --fix`
Expected: All checks passed

**Step 2: Run mypy**

Run: `mypy src/pymarxan/ --ignore-missing-imports`
Expected: Success: no issues found

**Step 3: Run full test suite (excluding benchmarks)**

Run: `python -m pytest tests/ --ignore=tests/benchmarks -v --tb=short`
Expected: All 317+ tests PASSED

**Step 4: Run benchmarks separately**

Run: `python -m pytest tests/benchmarks/ -v`
Expected: All PASSED

**Step 5: Commit any fixes**

```bash
git add -u
git commit -m "chore: fix lint and type issues from phase 7"
```

---

## Summary

| Task | Module | Tests | Key change |
|------|--------|-------|------------|
| 1 | `solvers/cache.py` | ~14 | ProblemCache + delta computation |
| 2 | `solvers/simulated_annealing.py` | 11 existing | Rewrite SA to use cache |
| 3 | `zones/cache.py` | ~5 | ZoneProblemCache + zone delta |
| 4 | `zones/solver.py` | 7 existing | Rewrite Zone SA to use cache |
| 5 | `tests/benchmarks/` | 5 | Performance benchmark suite |
| 6 | All | 317+ | Regression + lint + types |

**New test count:** ~24 new tests + 317 existing = ~341 total
**Performance targets:** SA 5K PU × 100K iters < 60s, Zone SA 1K PU × 100K iters < 30s

**Dependency graph:**
- Task 1 is independent (foundation)
- Task 2 depends on Task 1
- Task 3 depends on Task 1 (shares patterns)
- Task 4 depends on Task 3
- Task 5 depends on Tasks 2 + 4
- Task 6 depends on all

**Parallelisable:** Tasks 1 + 3 can run in parallel (no shared files)
