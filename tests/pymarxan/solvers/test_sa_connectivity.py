"""Simulated annealing and iterative improvement optimise connectivity (v0.44.0).

Before v0.44.0 the SA/II cache had no connectivity term: they searched as if
connectivity did not exist, and SA's tracked best objective disagreed with the
objective it reported.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.solvers.utils import compute_connectivity_penalty


def _grid_problem(
    connectivity: pd.DataFrame | None,
    asymmetric: bool = False,
) -> ConservationProblem:
    """4x4 grid, 3 features; every number is an integer so SA's accumulated
    objective is exact and can be compared to the reported one at 1e-9."""
    rng = np.random.default_rng(5)
    n = 16
    pu = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "cost": rng.integers(1, 10, size=n).astype(float),
        "status": np.zeros(n, dtype=int),
    })
    feat = pd.DataFrame({
        "id": [1, 2, 3], "name": ["a", "b", "c"],
        "target": [8.0, 6.0, 5.0], "spf": [20.0, 20.0, 20.0],
    })
    rows = [(f, p, float(rng.integers(1, 5)))
            for f in (1, 2, 3) for p in range(1, n + 1) if rng.random() < 0.4]
    puvspr = pd.DataFrame(rows, columns=["species", "pu", "amount"])
    b_rows = []
    for i in range(1, n + 1):
        if i % 4 != 0:
            b_rows.append((i, i + 1, 1.0))
        if i + 4 <= n:
            b_rows.append((i, i + 4, 1.0))
    boundary = pd.DataFrame(b_rows, columns=["id1", "id2", "boundary"])
    problem = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
        boundary=boundary, connectivity=connectivity,
    )
    problem.parameters.update({
        "BLM": 1.0, "NUMITNS": 20_000, "NUMTEMP": 100, "CONNECTIVITY_WEIGHT": 2.0,
    })
    if asymmetric:
        problem.parameters["ASYMMETRIC_CONNECTIVITY"] = 1
    return problem


def _grid_connectivity() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = [(int(rng.integers(1, 17)), int(rng.integers(1, 17)), float(rng.integers(1, 9)))
            for _ in range(40)]
    return pd.DataFrame(rows, columns=["id1", "id2", "value"])


def _solve_sa(problem: ConservationProblem, seed: int = 42):
    sols = SimulatedAnnealingSolver().solve(problem, SolverConfig(num_solutions=1, seed=seed))
    assert len(sols) == 1
    return sols[0]


class TestSAObjectiveConsistency:
    def test_control_without_connectivity(self) -> None:
        sol = _solve_sa(_grid_problem(None))
        assert sol.objective == pytest.approx(
            sol.metadata["history"]["best_objective"][-1], abs=1e-9,
        )

    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_reported_objective_equals_tracked_best(self, asymmetric: bool) -> None:
        problem = _grid_problem(_grid_connectivity(), asymmetric=asymmetric)
        sol = _solve_sa(problem)
        # The plan's connectivity term is non-zero, so the equality below exercises it.
        pu_index = {int(pid): i for i, pid in enumerate(problem.planning_units["id"])}
        assert compute_connectivity_penalty(problem, sol.selected, pu_index) != 0.0
        assert sol.objective == pytest.approx(
            sol.metadata["history"]["best_objective"][-1], abs=1e-9,
        )


def _three_pu(
    connectivity: pd.DataFrame | None,
    costs: tuple[float, float, float],
    params: dict[str, float],
) -> ConservationProblem:
    """The test_connectivity_mip fixture shape: 3 PUs, 1 feature, any one PU meets it."""
    problem = ConservationProblem(
        planning_units=pd.DataFrame({"id": [1, 2, 3], "cost": list(costs), "status": [0] * 3}),
        features=pd.DataFrame({"id": [1], "name": ["f1"], "target": [1.0], "spf": [1.0]}),
        pu_vs_features=pd.DataFrame({
            "species": [1, 1, 1], "pu": [1, 2, 3], "amount": [5.0, 5.0, 5.0],
        }),
        connectivity=connectivity,
    )
    problem.parameters.update({"NUMITNS": 2000, "NUMTEMP": 20, **params})
    return problem


def _ids(sol) -> set[int]:
    return {i + 1 for i in np.flatnonzero(sol.selected)}


class TestSABehaviour:
    # Seeds chosen so each case fails on a connectivity-blind SA (v0.43.0 cache).
    @pytest.mark.parametrize("seed", [1, 3, 6])
    def test_symmetric_strong_link_selects_both(self, seed: int) -> None:
        conn = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "value": [100.0, 0.1]})
        problem = _three_pu(conn, (1.0, 10.0, 1.0), {"CONNECTIVITY_WEIGHT": 1.0})
        assert {1, 2} <= _ids(_solve_sa(problem, seed))

    @pytest.mark.parametrize("seed", [1, 3, 6])
    def test_asymmetric_never_source_without_sink(self, seed: int) -> None:
        # PU 1 is the cheapest single unit, so a connectivity-blind SA picks it alone.
        conn = pd.DataFrame({"id1": [1], "id2": [2], "value": [50.0]})
        problem = _three_pu(
            conn, (1.0, 10.0, 2.0),
            {"CONNECTIVITY_WEIGHT": 1.0, "ASYMMETRIC_CONNECTIVITY": 1},
        )
        chosen = _ids(_solve_sa(problem, seed))
        assert 1 not in chosen or 2 in chosen, chosen


class TestIterativeImprovement:
    CONN = pd.DataFrame({"id1": [1], "id2": [2], "value": [100.0]})

    def _solve(self, weight: float) -> set[int]:
        problem = _three_pu(
            self.CONN, (1.0, 10.0, 1.0), {"CONNECTIVITY_WEIGHT": weight, "ITIMPTYPE": 1},
        )
        sols = IterativeImprovementSolver().solve(problem, SolverConfig(num_solutions=1))
        return _ids(sols[0])

    def test_removal_drops_expensive_unit_without_connectivity(self) -> None:
        assert self._solve(0.0) == {3}

    def test_removal_keeps_strongly_connected_pair(self) -> None:
        assert self._solve(1.0) == {1, 2}


# Cross-copy fixture: duplicate, reversed (different value) and self rows; no unknown ids,
# non-negative values (the MIP rejects unknown ids and cannot bound negative asymmetric values).
_CLEAN_SYM = pd.DataFrame({
    "id1": [1, 2, 2, 3, 4, 4, 3, 1],
    "id2": [2, 1, 3, 2, 4, 5, 5, 2],
    "value": [4.0, 9.0, 3.0, 8.0, 5.0, 2.0, 1.0, 6.0],
})
_CLEAN_ASYM = pd.DataFrame({
    "id1": [1, 1, 2, 3, 3, 4, 5, 6],
    "id2": [2, 2, 1, 3, 2, 5, 4, 3],
    "value": [4.0, 6.0, 3.0, 9.0, 2.0, 5.0, 1.0, 7.0],
})


def _six_pu(connectivity: pd.DataFrame, asymmetric: bool) -> ConservationProblem:
    """6 PUs, 2 features, BLM 1; SPF high enough that the penalty optimum meets targets."""
    problem = ConservationProblem(
        planning_units=pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "cost": [3.0, 5.0, 2.0, 7.0, 4.0, 6.0],
            "status": [0] * 6,
        }),
        features=pd.DataFrame({
            "id": [1, 2], "name": ["a", "b"], "target": [6.0, 4.0], "spf": [1000.0, 1000.0],
        }),
        pu_vs_features=pd.DataFrame({
            "species": [1, 1, 1, 2, 2, 2],
            "pu": [1, 3, 5, 2, 4, 6],
            "amount": [3.0, 2.0, 4.0, 2.0, 3.0, 1.0],
        }),
        boundary=pd.DataFrame({
            "id1": [1, 2, 3, 4, 5, 1],
            "id2": [2, 3, 4, 5, 6, 1],
            "boundary": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        }),
        connectivity=connectivity,
    )
    problem.parameters.update({
        "BLM": 1.0, "CONNECTIVITY_WEIGHT": 3.0, "NUMITNS": 20_000, "NUMTEMP": 100,
    })
    if asymmetric:
        problem.parameters["ASYMMETRIC_CONNECTIVITY"] = 1
    return problem


class TestCrossCopyAgreement:
    """Cache (brute force), MIP and SA agree on the optimum of the same connectivity problem."""

    @pytest.mark.parametrize(("conn", "asymmetric"), [(_CLEAN_SYM, False), (_CLEAN_ASYM, True)])
    def test_brute_force_mip_and_sa_agree(self, conn: pd.DataFrame, asymmetric: bool) -> None:
        problem = _six_pu(conn, asymmetric)
        cache = ProblemCache.from_problem(problem)
        brute = min(
            cache.compute_full_objective(sel, cache.compute_held(sel), 1.0)
            for sel in (np.array(bits, dtype=bool)
                        for bits in itertools.product([False, True], repeat=6))
        )
        mip = MIPSolver(mip_backend="cbc").solve(problem, SolverConfig(num_solutions=1))[0]
        assert mip.all_targets_met
        assert mip.metadata["mip_objective_value"] == pytest.approx(brute, abs=1e-6)
        sa = _solve_sa(problem)
        assert sa.objective == pytest.approx(brute, abs=1e-9)
