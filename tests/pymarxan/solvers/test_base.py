import numpy as np

from pymarxan.solvers.base import Solution, SolverConfig


class TestSolution:
    def test_create_solution(self):
        sol = Solution(
            selected=np.array([True, False, True]), cost=30.0,
            boundary=2.0, objective=32.0,
            targets_met={1: True, 2: False}, metadata={"solver": "test"},
        )
        assert sol.cost == 30.0
        assert sol.selected.sum() == 2
        assert sol.targets_met[1] is True
        assert sol.targets_met[2] is False

    def test_all_targets_met(self):
        sol = Solution(selected=np.array([True, True]), cost=25.0, boundary=1.0,
                       objective=26.0, targets_met={1: True, 2: True}, metadata={})
        assert sol.all_targets_met

    def test_not_all_targets_met(self):
        sol = Solution(selected=np.array([True]), cost=10.0, boundary=0.0,
                       objective=10.0, targets_met={1: True, 2: False}, metadata={})
        assert not sol.all_targets_met

    def test_n_selected(self):
        sol = Solution(selected=np.array([True, False, True, True, False]),
                       cost=0, boundary=0, objective=0, targets_met={}, metadata={})
        assert sol.n_selected == 3

class TestSolverConfig:
    def test_defaults(self):
        config = SolverConfig()
        assert config.num_solutions == 10
        assert config.seed is None
        assert config.verbose is False

    def test_custom(self):
        config = SolverConfig(num_solutions=50, seed=42, verbose=True)
        assert config.num_solutions == 50
        assert config.seed == 42
