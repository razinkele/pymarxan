"""End-to-end PROBMODE 3 sanity tests for every solver.

Phase 18 Batch 3. Each solver must:
1. Run without raising under PROBMODE 3.
2. Populate ``Solution.prob_shortfalls`` and ``Solution.prob_penalty``.
3. Honor MIP's ``mip_chance_strategy`` kwarg (drop / piecewise / socp).
4. Expose the ``supports_probmode3()`` capability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig


def _make_probmode3_problem(num_pus: int = 6, seed: int = 0) -> ConservationProblem:
    rng = np.random.default_rng(seed)
    pu = pd.DataFrame({
        "id": list(range(1, num_pus + 1)),
        "cost": list(rng.uniform(5.0, 20.0, num_pus)),
        "status": [0] * num_pus,
    })
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["f1", "f2"],
        "target": [12.0, 8.0],
        "spf": [1.0, 2.0],
        "ptarget": [0.95, 0.80],
    })
    rows = []
    for fid in (1, 2):
        for pu_id in range(1, num_pus + 1):
            rows.append({
                "species": fid,
                "pu": pu_id,
                "amount": float(rng.uniform(1.0, 5.0)),
                "prob": float(rng.uniform(0.0, 0.3)),
            })
    puvspr = pd.DataFrame(rows)
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 1.0},
    )


# --- Capability method --------------------------------------------------


class TestSupportsProbmode3Capability:
    """All current solvers report True (MIP via the 'drop' fallback)."""

    def test_solver_base_default(self):
        from pymarxan.solvers.heuristic import HeuristicSolver
        assert HeuristicSolver().supports_probmode3() is True

    def test_sa_supports(self):
        from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
        assert SimulatedAnnealingSolver().supports_probmode3() is True

    def test_mip_supports_via_drop_fallback(self):
        from pymarxan.solvers.mip_solver import MIPSolver
        assert MIPSolver().supports_probmode3() is True

    def test_iterative_improvement_supports(self):
        from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
        assert IterativeImprovementSolver().supports_probmode3() is True


# --- Build_solution populates Solution.prob_* fields --------------------


class TestBuildSolutionPopulatesProbFields:
    """build_solution should populate prob_shortfalls and prob_penalty
    when PROBMODE 3 is active, and leave them None otherwise."""

    def test_probmode_0_leaves_prob_fields_none(self):
        from pymarxan.solvers.utils import build_solution
        problem = _make_probmode3_problem(num_pus=4)
        problem.parameters["PROBMODE"] = 0
        selected = np.array([True, True, False, False])
        sol = build_solution(problem, selected, blm=0.0)
        assert sol.prob_shortfalls is None
        assert sol.prob_penalty is None

    def test_probmode_3_populates_prob_fields(self):
        from pymarxan.solvers.utils import build_solution
        problem = _make_probmode3_problem(num_pus=4)
        selected = np.array([True, True, False, False])
        sol = build_solution(problem, selected, blm=0.0)
        assert sol.prob_shortfalls is not None
        assert sol.prob_penalty is not None
        # Every feature has an entry
        assert set(sol.prob_shortfalls.keys()) == {1, 2}


# --- MIP strategy gating ------------------------------------------------


class TestMIPChanceStrategy:
    """MIP under PROBMODE 3 honors mip_chance_strategy:
    - 'drop' (default): runs, populates post-hoc Solution.prob_*
    - 'piecewise' / 'socp': NotImplementedError with Phase pointer
    """

    def test_drop_default_runs_and_reports_gap(self):
        from pymarxan.solvers.mip_solver import MIPSolver
        problem = _make_probmode3_problem(num_pus=4)
        sols = MIPSolver().solve(problem, SolverConfig(num_solutions=1))
        assert len(sols) == 1
        sol = sols[0]
        assert sol.prob_shortfalls is not None
        assert sol.prob_penalty is not None
        # Metadata records the chosen strategy
        assert sol.metadata.get("mip_chance_strategy") == "drop"

    def test_piecewise_strategy_not_implemented(self):
        from pymarxan.solvers.mip_solver import MIPSolver
        problem = _make_probmode3_problem(num_pus=4)
        with pytest.raises(NotImplementedError, match="piecewise"):
            MIPSolver(mip_chance_strategy="piecewise").solve(problem)

    def test_socp_strategy_not_implemented_phase21_pointer(self):
        from pymarxan.solvers.mip_solver import MIPSolver
        problem = _make_probmode3_problem(num_pus=4)
        with pytest.raises(NotImplementedError, match="Phase 21"):
            MIPSolver(mip_chance_strategy="socp").solve(problem)

    def test_unknown_strategy_rejected_at_init(self):
        from pymarxan.solvers.mip_solver import MIPSolver
        with pytest.raises(ValueError, match="Unknown mip_chance_strategy"):
            MIPSolver(mip_chance_strategy="bogus")

    def test_mip_under_probmode0_unaffected(self):
        """Strategy gating only fires under PROBMODE 3."""
        from pymarxan.solvers.mip_solver import MIPSolver
        problem = _make_probmode3_problem(num_pus=4)
        problem.parameters["PROBMODE"] = 0
        # All three strategies should now be no-ops
        for strat in ("drop", "piecewise", "socp"):
            sols = MIPSolver(mip_chance_strategy=strat).solve(
                problem, SolverConfig(num_solutions=1),
            )
            assert len(sols) == 1


# --- End-to-end smoke for SA / heuristic / iterative_improvement --------


class TestProbmode3EndToEnd:
    """Each native solver runs end-to-end under PROBMODE 3 and produces
    a solution with prob_shortfalls and prob_penalty populated."""

    def test_sa_runs_under_probmode3(self):
        from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
        problem = _make_probmode3_problem(num_pus=4, seed=11)
        problem.parameters["NUMITNS"] = 200
        sols = SimulatedAnnealingSolver().solve(
            problem, SolverConfig(num_solutions=1, seed=11),
        )
        assert len(sols) == 1
        assert sols[0].prob_shortfalls is not None
        assert sols[0].prob_penalty is not None

    def test_heuristic_runs_under_probmode3(self):
        from pymarxan.solvers.heuristic import HeuristicSolver
        problem = _make_probmode3_problem(num_pus=4, seed=12)
        sols = HeuristicSolver().solve(problem, SolverConfig(num_solutions=1))
        assert len(sols) == 1
        assert sols[0].prob_shortfalls is not None
        assert sols[0].prob_penalty is not None

    def test_iterative_improvement_runs_under_probmode3(self):
        from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
        problem = _make_probmode3_problem(num_pus=4, seed=13)
        problem.parameters["ITIMPTYPE"] = 1  # removal pass
        sols = IterativeImprovementSolver().solve(
            problem, SolverConfig(num_solutions=1, seed=13),
        )
        assert len(sols) == 1
        assert sols[0].prob_shortfalls is not None
        assert sols[0].prob_penalty is not None
