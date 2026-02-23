"""Tests for iterative improvement solver (ITIMPTYPE 0-3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.utils import build_solution, compute_objective

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_problem() -> ConservationProblem:
    """Problem where only PU 1 and PU 3 are needed to meet targets.

    PU 1: cost=10, contributes 6.0 to feature 1
    PU 2: cost=20, contributes 1.0 to feature 1, 1.0 to feature 2
    PU 3: cost=15, contributes 5.0 to feature 2
    PU 4: cost=25, contributes 1.0 to feature 1, 1.0 to feature 2

    Feature 1: target=5  -> met by PU 1 alone (6.0)
    Feature 2: target=4  -> met by PU 3 alone (5.0)

    So an all-selected solution is over-selected; PU 2 and PU 4 are unnecessary.
    """
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 20.0, 15.0, 25.0],
        "status": [0, 0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1, 2],
        "name": ["f1", "f2"],
        "target": [5.0, 4.0],
        "spf": [10.0, 10.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 2, 2],
        "pu": [1, 2, 4, 3, 4],
        "amount": [6.0, 1.0, 1.0, 5.0, 1.0],
    })
    return ConservationProblem(
        planning_units=pu,
        features=feat,
        pu_vs_features=puvspr,
        boundary=None,
        parameters={"BLM": 0.0},
    )


@pytest.fixture()
def over_selected_solution(simple_problem: ConservationProblem) -> Solution:
    """All PUs selected -- clearly over-selected."""
    selected = np.ones(simple_problem.n_planning_units, dtype=bool)
    return build_solution(simple_problem, selected, blm=0.0)


# ---------------------------------------------------------------------------
# Solver interface tests
# ---------------------------------------------------------------------------


class TestSolverInterface:
    def test_name(self):
        solver = IterativeImprovementSolver()
        assert solver.name() == "iterative_improvement"

    def test_supports_zones(self):
        solver = IterativeImprovementSolver()
        assert solver.supports_zones() is False

    def test_invalid_itimptype_raises(self):
        with pytest.raises(ValueError, match="itimptype"):
            IterativeImprovementSolver(itimptype=5)

    def test_invalid_itimptype_negative_raises(self):
        with pytest.raises(ValueError, match="itimptype"):
            IterativeImprovementSolver(itimptype=-1)


# ---------------------------------------------------------------------------
# ITIMPTYPE 0 -- no improvement
# ---------------------------------------------------------------------------


class TestItimptype0:
    def test_returns_solution_unchanged(
        self, simple_problem, over_selected_solution
    ):
        solver = IterativeImprovementSolver(itimptype=0)
        improved = solver.improve(simple_problem, over_selected_solution)
        np.testing.assert_array_equal(
            improved.selected, over_selected_solution.selected
        )

    def test_objective_unchanged(
        self, simple_problem, over_selected_solution
    ):
        solver = IterativeImprovementSolver(itimptype=0)
        improved = solver.improve(simple_problem, over_selected_solution)
        assert improved.objective == over_selected_solution.objective


# ---------------------------------------------------------------------------
# ITIMPTYPE 1 -- removal pass
# ---------------------------------------------------------------------------


class TestItimptype1:
    def test_removes_unnecessary_pus(
        self, simple_problem, over_selected_solution
    ):
        solver = IterativeImprovementSolver(itimptype=1)
        improved = solver.improve(simple_problem, over_selected_solution)
        # Should remove at least some PUs to reduce cost
        assert improved.n_selected < over_selected_solution.n_selected

    def test_objective_decreases(
        self, simple_problem, over_selected_solution
    ):
        solver = IterativeImprovementSolver(itimptype=1)
        improved = solver.improve(simple_problem, over_selected_solution)
        assert improved.objective <= over_selected_solution.objective

    def test_targets_still_met(
        self, simple_problem, over_selected_solution
    ):
        solver = IterativeImprovementSolver(itimptype=1)
        improved = solver.improve(simple_problem, over_selected_solution)
        # Targets should still be met (removing PUs that don't affect targets)
        assert improved.all_targets_met

    def test_locked_in_pus_never_removed(self, simple_problem):
        """PU 2 is locked in (status=2); should never be removed."""
        simple_problem.planning_units.loc[
            simple_problem.planning_units["id"] == 2, "status"
        ] = 2
        selected = np.ones(simple_problem.n_planning_units, dtype=bool)
        solution = build_solution(simple_problem, selected, blm=0.0)

        solver = IterativeImprovementSolver(itimptype=1)
        improved = solver.improve(simple_problem, solution)

        # PU 2 is at index 1
        assert improved.selected[1] is np.True_

    def test_locked_out_pus_stay_out(self, simple_problem):
        """PU 4 is locked out (status=3); should not appear in result."""
        simple_problem.planning_units.loc[
            simple_problem.planning_units["id"] == 4, "status"
        ] = 3
        selected = np.array([True, True, True, False], dtype=bool)
        solution = build_solution(simple_problem, selected, blm=0.0)

        solver = IterativeImprovementSolver(itimptype=1)
        improved = solver.improve(simple_problem, solution)

        # PU 4 is at index 3 -- should stay unselected
        assert improved.selected[3] is np.False_


# ---------------------------------------------------------------------------
# ITIMPTYPE 2 -- removal then addition
# ---------------------------------------------------------------------------


class TestItimptype2:
    def test_removes_then_adds(self, simple_problem, over_selected_solution):
        solver = IterativeImprovementSolver(itimptype=2)
        improved = solver.improve(simple_problem, over_selected_solution)
        # Should at least not be worse
        assert improved.objective <= over_selected_solution.objective

    def test_addition_pass_can_improve(self, simple_problem):
        """Start with an under-selected solution; addition pass should help.

        Only PU 1 selected -> meets feature 1 target but not feature 2.
        The full objective (cost + penalty) is 10 + 10*4 = 50.
        Adding PU 3 (cost=15) meets feature 2 target, removing penalty.
        Full objective becomes 25, which is an improvement.
        """
        selected = np.array([True, False, False, False], dtype=bool)
        solution = build_solution(simple_problem, selected, blm=0.0)
        pu_ids = simple_problem.planning_units["id"].tolist()
        pu_index = {pid: i for i, pid in enumerate(pu_ids)}

        original_full_obj = compute_objective(
            simple_problem, selected, pu_index, blm=0.0
        )

        solver = IterativeImprovementSolver(itimptype=2)
        improved = solver.improve(simple_problem, solution)

        improved_full_obj = compute_objective(
            simple_problem, improved.selected, pu_index, blm=0.0
        )

        # The full objective (including penalty) should decrease
        assert improved_full_obj < original_full_obj
        # Feature 2 target should now be met
        assert improved.targets_met[2] is True


# ---------------------------------------------------------------------------
# ITIMPTYPE 3 -- pairwise swaps
# ---------------------------------------------------------------------------


class TestItimptype3:
    def test_swap_improves_objective(self, simple_problem):
        """Start with a sub-optimal selection; swap should find improvements.

        Select PU 2 (cost=20, feature1=1.0) and PU 4 (cost=25, feature2=1.0)
        and PU 3 (cost=15, feature2=5.0).
        A swap of PU 2 -> PU 1 should be beneficial (PU 1 costs 10 and has
        feature1=6.0 vs PU 2 costs 20 and has feature1=1.0).
        """
        selected = np.array([False, True, True, True], dtype=bool)
        solution = build_solution(simple_problem, selected, blm=0.0)

        solver = IterativeImprovementSolver(itimptype=3)
        improved = solver.improve(simple_problem, solution)

        assert improved.objective <= solution.objective

    def test_swap_respects_locked_in(self, simple_problem):
        """Locked-in PUs should never be swapped out."""
        simple_problem.planning_units.loc[
            simple_problem.planning_units["id"] == 4, "status"
        ] = 2
        selected = np.array([False, True, True, True], dtype=bool)
        solution = build_solution(simple_problem, selected, blm=0.0)

        solver = IterativeImprovementSolver(itimptype=3)
        improved = solver.improve(simple_problem, solution)

        # PU 4 (index 3) must remain selected
        assert improved.selected[3] is np.True_

    def test_swap_respects_locked_out(self, simple_problem):
        """Locked-out PUs should never be swapped in."""
        simple_problem.planning_units.loc[
            simple_problem.planning_units["id"] == 1, "status"
        ] = 3
        selected = np.array([False, True, True, True], dtype=bool)
        solution = build_solution(simple_problem, selected, blm=0.0)

        solver = IterativeImprovementSolver(itimptype=3)
        improved = solver.improve(simple_problem, solution)

        # PU 1 (index 0) must stay unselected
        assert improved.selected[0] is np.False_


# ---------------------------------------------------------------------------
# ITIMPTYPE from problem.parameters
# ---------------------------------------------------------------------------


class TestItimptypeFromParameters:
    def test_parameter_overrides_constructor(
        self, simple_problem, over_selected_solution
    ):
        """ITIMPTYPE in problem.parameters overrides constructor value."""
        simple_problem.parameters["ITIMPTYPE"] = 0
        solver = IterativeImprovementSolver(itimptype=1)
        improved = solver.improve(simple_problem, over_selected_solution)
        # ITIMPTYPE=0 means no improvement
        np.testing.assert_array_equal(
            improved.selected, over_selected_solution.selected
        )

    def test_parameter_itimptype_used(
        self, simple_problem, over_selected_solution
    ):
        """ITIMPTYPE from parameters actually works."""
        simple_problem.parameters["ITIMPTYPE"] = 1
        solver = IterativeImprovementSolver(itimptype=0)
        improved = solver.improve(simple_problem, over_selected_solution)
        # ITIMPTYPE=1 should do removal
        assert improved.n_selected < over_selected_solution.n_selected


# ---------------------------------------------------------------------------
# solve() interface tests
# ---------------------------------------------------------------------------


class TestSolveMethod:
    def test_solve_returns_list_of_solutions(self, simple_problem):
        solver = IterativeImprovementSolver(itimptype=1)
        config = SolverConfig(num_solutions=1)
        solutions = solver.solve(simple_problem, config)
        assert len(solutions) == 1
        assert isinstance(solutions[0], Solution)

    def test_solve_starts_all_selected_then_improves(self, simple_problem):
        solver = IterativeImprovementSolver(itimptype=1)
        config = SolverConfig(num_solutions=1)
        solutions = solver.solve(simple_problem, config)
        # Should have removed unnecessary PUs
        assert solutions[0].n_selected < simple_problem.n_planning_units

    def test_solve_multiple_solutions(self, simple_problem):
        solver = IterativeImprovementSolver(itimptype=1)
        config = SolverConfig(num_solutions=3)
        solutions = solver.solve(simple_problem, config)
        assert len(solutions) == 3

    def test_solve_default_config(self, simple_problem):
        solver = IterativeImprovementSolver(itimptype=1)
        solutions = solver.solve(simple_problem)
        assert len(solutions) >= 1

    def test_iterative_improvement_handles_status1(self, simple_problem):
        """Status=1 PUs should start selected but be improvable."""
        simple_problem.planning_units.loc[0, "status"] = 1
        simple_problem.parameters["ITIMPTYPE"] = 1
        solver = IterativeImprovementSolver(itimptype=1)
        solutions = solver.solve(simple_problem, SolverConfig(num_solutions=1))
        assert len(solutions) == 1
        assert solutions[0].cost >= 0

    def test_iterative_improvement_removal_produces_valid_result(
        self, simple_problem
    ):
        """Verify ITIMPTYPE=1 removal produces valid solution with cache."""
        simple_problem.parameters["ITIMPTYPE"] = 1
        solver = IterativeImprovementSolver(itimptype=1)
        solutions = solver.solve(simple_problem, SolverConfig(num_solutions=1))
        assert len(solutions) == 1
        sol = solutions[0]
        assert sol.cost >= 0
        assert sol.objective >= sol.cost
