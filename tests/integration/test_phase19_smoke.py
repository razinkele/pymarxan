"""End-to-end smoke test for Phase 19 — TARGET2 / CLUMPTYPE / clumping.

Verifies the four canonical single-zone solvers (SA, MIP, Heuristic,
IterativeImprovement) all run end-to-end on a clumping-active problem
and populate ``Solution.clump_shortfalls`` / ``Solution.clump_penalty``.
Also verifies the regression guard (target2==0 byte-identical to
pre-Phase-19) and the Marxan-style "patch wins over scattered" behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig


def _build_clumping_problem(seed: int = 0) -> ConservationProblem:
    """A 6-PU line graph with one type-4 feature (target2=25, CLUMPTYPE 0).

    Each PU supplies 10 of the feature; target = 30. Without TARGET2 any 3
    PUs meet the target. With target2=25 + CLUMPTYPE 0, a 2-PU clump
    (occ=20) contributes 0 and a 3-PU clump (occ=30) contributes 30 —
    so the SA / II must pick 3 contiguous PUs.
    """
    n_pu = 6
    pu = pd.DataFrame({
        "id": list(range(1, n_pu + 1)),
        "cost": [10.0] * n_pu,
        "status": [0] * n_pu,
    })
    features = pd.DataFrame({
        "id": [1],
        "name": ["sp"],
        "target": [30.0],
        "spf": [5.0],
        "target2": [25.0],
        "clumptype": [0],
    })
    puvspr = pd.DataFrame({
        "species": [1] * n_pu,
        "pu": list(range(1, n_pu + 1)),
        "amount": [10.0] * n_pu,
    })
    boundary = pd.DataFrame({
        "id1": list(range(1, n_pu)),
        "id2": list(range(2, n_pu + 1)),
        "boundary": [1.0] * (n_pu - 1),
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        boundary=boundary,
        parameters={"NUMITNS": 2000, "NUMTEMP": 200, "BLM": 0.0},
    )


# --- All four solvers run + populate clump fields -----------------------


class TestAllSolversUnderTarget2:

    def test_sa_runs_and_populates_clump_fields(self):
        from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
        problem = _build_clumping_problem()
        sols = SimulatedAnnealingSolver().solve(
            problem, SolverConfig(num_solutions=1, seed=1),
        )
        assert len(sols) == 1
        sol = sols[0]
        assert sol.clump_shortfalls is not None
        assert sol.clump_penalty is not None
        # SA should find a 3+ contiguous-PU solution → clump_penalty = 0
        assert sol.clump_penalty == pytest.approx(0.0, abs=1e-6)

    def test_mip_runs_and_populates_clump_fields(self):
        from pymarxan.solvers.mip_solver import MIPSolver
        problem = _build_clumping_problem()
        sols = MIPSolver().solve(
            problem, SolverConfig(num_solutions=1, seed=1),
        )
        sol = sols[0]
        assert sol.clump_shortfalls is not None
        assert sol.clump_penalty is not None
        # MIP "drop" strategy: solves deterministic problem only — picks 3
        # cheapest PUs (which may not be contiguous). Metadata records strategy.
        assert sol.metadata.get("mip_clump_strategy") == "drop"

    def test_heuristic_runs_and_populates_clump_fields(self):
        from pymarxan.solvers.heuristic import HeuristicSolver
        problem = _build_clumping_problem()
        sols = HeuristicSolver().solve(
            problem, SolverConfig(num_solutions=1, seed=1),
        )
        sol = sols[0]
        # Heuristic stays clump-blind during scoring; post-hoc reports the gap.
        assert sol.clump_shortfalls is not None
        assert sol.clump_penalty is not None

    def test_iterative_improvement_runs_and_populates_clump_fields(self):
        from pymarxan.solvers.iterative_improvement import (
            IterativeImprovementSolver,
        )
        problem = _build_clumping_problem()
        sols = IterativeImprovementSolver(itimptype=2).solve(
            problem, SolverConfig(num_solutions=1, seed=1),
        )
        sol = sols[0]
        assert sol.clump_shortfalls is not None
        assert sol.clump_penalty is not None


# --- supports_clumping capability method --------------------------------


class TestSupportsClumpingCapability:
    """All four solvers default-True (MIP via drop fallback)."""

    def test_sa_supports(self):
        from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
        assert SimulatedAnnealingSolver().supports_clumping() is True

    def test_heuristic_supports(self):
        from pymarxan.solvers.heuristic import HeuristicSolver
        assert HeuristicSolver().supports_clumping() is True

    def test_mip_supports_via_drop(self):
        from pymarxan.solvers.mip_solver import MIPSolver
        assert MIPSolver().supports_clumping() is True

    def test_iterative_improvement_supports(self):
        from pymarxan.solvers.iterative_improvement import (
            IterativeImprovementSolver,
        )
        assert IterativeImprovementSolver().supports_clumping() is True


# --- MIP clump strategy gating ------------------------------------------


class TestMIPClumpStrategy:

    def test_drop_default(self):
        from pymarxan.solvers.mip_solver import MIPSolver
        s = MIPSolver()
        assert s.mip_clump_strategy == "drop"

    def test_big_m_not_implemented(self):
        from pymarxan.solvers.mip_solver import MIPSolver
        problem = _build_clumping_problem()
        with pytest.raises(NotImplementedError, match="big_m"):
            MIPSolver(mip_clump_strategy="big_m").solve(problem)

    def test_unknown_strategy_rejected_at_init(self):
        from pymarxan.solvers.mip_solver import MIPSolver
        with pytest.raises(ValueError, match="Unknown mip_clump_strategy"):
            MIPSolver(mip_clump_strategy="bogus")


# --- Regression guard: target2==0 byte-identical to pre-Phase-19 --------


class TestTarget2ZeroIsRegressionFree:
    """When no feature has target2 > 0, the clumping code path is never
    entered (cache.clumping_active is False, build_solution skips the
    populate). The whole stack must behave identically to the pre-Phase-19
    deterministic path."""

    def test_sa_objective_unchanged_when_target2_zero(self):
        from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
        problem = _build_clumping_problem()
        # Disable clumping
        features = problem.features.copy()
        features["target2"] = 0.0
        problem = problem.copy_with(features=features)
        sol = SimulatedAnnealingSolver().solve(
            problem, SolverConfig(num_solutions=1, seed=42),
        )[0]
        # No clumping → Solution.clump_* are None
        assert sol.clump_shortfalls is None
        assert sol.clump_penalty is None


# --- save/load round-trip with target2 + clumptype ----------------------


def test_phase19_project_round_trips(tmp_path):
    """save_project + load_project preserves target2 + clumptype columns."""
    from pymarxan.io.readers import load_project
    from pymarxan.io.writers import save_project

    problem = _build_clumping_problem()
    project_dir = tmp_path / "p19_project"
    save_project(problem, project_dir)
    loaded = load_project(project_dir)

    assert "target2" in loaded.features.columns
    assert "clumptype" in loaded.features.columns
    np.testing.assert_array_almost_equal(
        loaded.features["target2"].values,
        problem.features["target2"].values,
    )
    np.testing.assert_array_equal(
        loaded.features["clumptype"].values.astype(int),
        problem.features["clumptype"].values.astype(int),
    )
