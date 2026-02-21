import numpy as np

from pymarxan.analysis.selection_freq import (
    SelectionFrequency,
    compute_selection_frequency,
)
from pymarxan.solvers.base import Solution


def _make_solutions() -> list[Solution]:
    """Create 4 mock solutions with known selection patterns."""
    return [
        Solution(
            selected=np.array([True, True, False, False]),
            cost=25.0, boundary=1.0, objective=26.0,
            targets_met={1: True}, metadata={},
        ),
        Solution(
            selected=np.array([True, False, True, False]),
            cost=30.0, boundary=2.0, objective=32.0,
            targets_met={1: True}, metadata={},
        ),
        Solution(
            selected=np.array([True, True, True, False]),
            cost=45.0, boundary=1.0, objective=46.0,
            targets_met={1: True}, metadata={},
        ),
        Solution(
            selected=np.array([False, True, True, True]),
            cost=40.0, boundary=3.0, objective=43.0,
            targets_met={1: True}, metadata={},
        ),
    ]


class TestSelectionFrequency:
    def test_returns_correct_type(self):
        result = compute_selection_frequency(_make_solutions())
        assert isinstance(result, SelectionFrequency)

    def test_frequency_values(self):
        result = compute_selection_frequency(_make_solutions())
        np.testing.assert_array_almost_equal(
            result.frequencies, [0.75, 0.75, 0.75, 0.25]
        )

    def test_count_values(self):
        result = compute_selection_frequency(_make_solutions())
        np.testing.assert_array_equal(result.counts, [3, 3, 3, 1])

    def test_n_solutions(self):
        result = compute_selection_frequency(_make_solutions())
        assert result.n_solutions == 4

    def test_best_solution(self):
        result = compute_selection_frequency(_make_solutions())
        assert result.best_solution.cost == 25.0

    def test_empty_solutions(self):
        result = compute_selection_frequency([])
        assert result.n_solutions == 0
        assert len(result.frequencies) == 0
