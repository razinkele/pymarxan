"""Stage 4 (spec §5.1, §5.2): the fractional Solution model and the binary-only guard."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.solvers import FractionalSolutionError, require_binary
from pymarxan.solvers.base import Solution


def _sol(selected: np.ndarray) -> Solution:
    return Solution(selected=selected, cost=0.0, boundary=0.0, objective=0.0, targets_met={})


def test_binary_solution_is_not_fractional() -> None:
    s = _sol(np.array([True, False, True]))
    assert not s.is_fractional
    assert s.n_selected == 2
    assert isinstance(s.n_selected, int)
    assert s.selected_sum == 2.0
    assert isinstance(s.selected_sum, float)


def test_fractional_solution_properties() -> None:
    s = _sol(np.array([0.25, 0.0, 1.0, 0.5]))
    assert s.is_fractional
    assert s.n_selected == 3  # count of x > 0, not the sum
    assert s.selected_sum == pytest.approx(1.75)


def test_dtype_is_the_discriminator_not_the_values() -> None:
    s = _sol(np.array([1.0, 0.0, 1.0]))
    assert s.is_fractional
    assert s.n_selected == 2
    assert s.selected_sum == 2.0


def test_require_binary_passes_binary() -> None:
    require_binary(_sol(np.array([True, False])), "overlap_matrix")


def test_require_binary_rejects_fractional() -> None:
    with pytest.raises(FractionalSolutionError) as exc:
        require_binary(_sol(np.array([0.5, 1.0])), "overlap_matrix")
    msg = str(exc.value)
    assert msg == (
        "overlap_matrix needs a binary solution, but this solution holds proportion "
        "decisions. Convert explicitly with binarize_solution(problem, solution, "
        "threshold=...)."
    )
    assert isinstance(exc.value, ValueError)
