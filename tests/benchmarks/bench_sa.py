"""Simulated annealing performance benchmarks."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from tests.benchmarks.conftest import make_problem

pytestmark = pytest.mark.bench


class TestSAPerformance:
    """Performance benchmarks for the SA solver at various problem sizes."""

    def test_small_100pu_10feat(self) -> None:
        """100 PU, 10 features, 10K iterations -- must complete in <2s."""
        problem = make_problem(n_pu=100, n_feat=10, density=0.3, seed=42)
        problem.parameters["NUMITNS"] = 10_000
        problem.parameters["NUMTEMP"] = 100

        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert solutions[0].cost > 0
        assert elapsed < 2.0, f"Small SA took {elapsed:.2f}s, expected <2s"

    def test_medium_1k_pu_50feat(self) -> None:
        """1K PU, 50 features, 100K iterations -- must complete in <15s."""
        problem = make_problem(n_pu=1000, n_feat=50, density=0.3, seed=42)
        problem.parameters["NUMITNS"] = 100_000
        problem.parameters["NUMTEMP"] = 1000

        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert solutions[0].cost > 0
        assert elapsed < 15.0, f"Medium SA took {elapsed:.2f}s, expected <15s"

    def test_large_5k_pu_100feat(self) -> None:
        """5K PU, 100 features, 100K iterations -- must complete in <60s."""
        problem = make_problem(n_pu=5000, n_feat=100, density=0.3, seed=42)
        problem.parameters["NUMITNS"] = 100_000
        problem.parameters["NUMTEMP"] = 1000

        solver = SimulatedAnnealingSolver()
        config = SolverConfig(num_solutions=1, seed=42)

        start = time.perf_counter()
        solutions = solver.solve(problem, config)
        elapsed = time.perf_counter() - start

        assert len(solutions) == 1
        assert solutions[0].cost > 0
        assert elapsed < 60.0, f"Large SA took {elapsed:.2f}s, expected <60s"


def _delta_loop_time(problem: ConservationProblem, n_calls: int, reps: int = 5) -> float:
    """Best-of-``reps`` seconds for ``n_calls`` compute_delta_objective calls.

    Identical seeded selection and flip sequence for every problem of the same size, so
    timings are directly comparable within one test.
    """
    cache = ProblemCache.from_problem(problem)
    rng = np.random.default_rng(0)
    selected = rng.random(cache.n_pu) < 0.5
    held = cache.compute_held(selected)
    total_cost = float(np.sum(cache.costs[selected]))
    flips = rng.integers(cache.n_pu, size=n_calls)
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        for idx in flips:
            cache.compute_delta_objective(int(idx), selected, held, total_cost, 1.0)
        best = min(best, time.perf_counter() - start)
    return best


class TestSAConnectivityDeltaCost:
    """v0.44.0: connectivity in the SA delta. Comparative only (machine-relative)."""

    N_CALLS = 20_000

    def _problems(self) -> tuple[ConservationProblem, ...]:
        """No table; table + weight 0; symmetric weight 1; asymmetric weight 1."""
        base = make_problem(n_pu=1000, n_feat=50, density=0.3, seed=42)
        assert base.boundary is not None
        conn = base.boundary.rename(columns={"boundary": "value"})  # grid graph, degree ~4
        conn["value"] = 1.0
        # copy_with is shallow: give each variant its own parameters dict.
        off = base.copy_with(
            connectivity=conn, parameters={**base.parameters, "CONNECTIVITY_WEIGHT": 0.0},
        )
        on = base.copy_with(
            connectivity=conn, parameters={**base.parameters, "CONNECTIVITY_WEIGHT": 1.0},
        )
        asym = base.copy_with(
            connectivity=conn,
            parameters={
                **base.parameters, "CONNECTIVITY_WEIGHT": 1.0, "ASYMMETRIC_CONNECTIVITY": 1,
            },
        )
        return base, off, on, asym

    def test_connectivity_off_costs_nothing(self) -> None:
        base, off, on, asym = self._problems()
        assert ProblemCache.from_problem(off).conn_mode == 0
        assert ProblemCache.from_problem(on).conn_mode == 1
        assert ProblemCache.from_problem(asym).conn_mode == 2
        t_base = _delta_loop_time(base, self.N_CALLS)
        t_off = _delta_loop_time(off, self.N_CALLS)
        t_on = _delta_loop_time(on, self.N_CALLS)
        t_asym = _delta_loop_time(asym, self.N_CALLS)
        per_call = 1e6 / self.N_CALLS
        print(
            f"\ndelta per call: no table {t_base * per_call:.2f} us, "
            f"table+weight 0 {t_off * per_call:.2f} us (x{t_off / t_base:.3f}), "
            f"degree~4 weight 1 symmetric {t_on * per_call:.2f} us (x{t_on / t_base:.3f}), "
            f"asymmetric {t_asym * per_call:.2f} us (x{t_asym / t_base:.3f})"
        )
        # Off must be free: <= 5% over the no-table baseline, with a 2 us/call floor
        # absorbing timer noise on fast machines.
        assert t_off <= max(1.05 * t_base, t_base + 2e-6 * self.N_CALLS)


class TestSAProbmodeDeltaCost:
    """v0.45.0: PROBMODE 1 / 2 in the SA delta. Comparative only (machine-relative)."""

    N_CALLS = 20_000

    def _problems(self) -> dict[str, ConservationProblem]:
        """No table; table + PROBMODE 0; mode 1 γ = 0; mode 2; mode 1 γ = 1."""
        base = make_problem(n_pu=1000, n_feat=50, density=0.3, seed=42)
        rng = np.random.default_rng(7)
        prob = pd.DataFrame({
            "pu": base.planning_units["id"].to_numpy(),
            "probability": rng.uniform(0.0, 0.6, len(base.planning_units)),
        })

        def variant(**params: float) -> ConservationProblem:
            # copy_with is shallow: give each variant its own parameters dict.
            return base.copy_with(probability=prob, parameters={**base.parameters, **params})

        return {
            "base": base,
            "mode0": variant(PROBMODE=0),
            "mode1_off": variant(PROBMODE=1, PROBABILITYWEIGHTING=0.0),
            "mode2": variant(PROBMODE=2),
            "mode1_on": variant(PROBMODE=1, PROBABILITYWEIGHTING=1.0),
        }

    def test_probmode_costs_nothing_when_off_or_prescaled(self) -> None:
        problems = self._problems()
        assert not ProblemCache.from_problem(problems["mode1_off"]).prob_active
        assert ProblemCache.from_problem(problems["mode1_on"]).prob_active
        c2 = ProblemCache.from_problem(problems["mode2"])
        assert c2.held_csr is not c2.pu_feat_csr
        t = {name: _delta_loop_time(p, self.N_CALLS) for name, p in problems.items()}
        per_call = 1e6 / self.N_CALLS
        t_base = t["base"]
        print(
            "\ndelta per call: "
            + ", ".join(
                f"{name} {v * per_call:.2f} us (x{v / t_base:.3f})" for name, v in t.items()
            )
        )
        # Off / pre-scaled must be free. Effective bound: max(1.05x, +2 us/call) over the
        # no-table baseline — the 2 us/call timer-noise floor dominates at ~17 us/call
        # (about +12%), as in the v0.44.0 connectivity bench. mode1_on is print-only.
        budget = max(1.05 * t_base, t_base + 2e-6 * self.N_CALLS)
        assert t["mode0"] <= budget
        assert t["mode1_off"] <= budget
        assert t["mode2"] <= budget
