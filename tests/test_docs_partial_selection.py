"""Pins the USER_MANUAL "Partial Selection" example (Stage 4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pymarxan.io.readers import load_project
from pymarxan.solvers import MIPSolver, binarize_solution

ROOT = Path(__file__).parent.parent


def test_partial_selection_example() -> None:
    problem = load_project(ROOT / "tests" / "data" / "simple")
    sol = MIPSolver(decisions="proportion", upper=0.8).solve(problem)[0]
    assert sol.selected.dtype == np.float64
    assert sol.selected.max() <= 0.8 + 1e-9
    assert sol.all_targets_met is True
    plan = binarize_solution(problem, sol, threshold=0.5)
    assert plan.all_targets_met is False
    expected = [0.11319149, 0.8, 0.0, 0.49787234, 0.36170213, 0.8]
    assert np.allclose(sol.selected, expected, atol=1e-6)
