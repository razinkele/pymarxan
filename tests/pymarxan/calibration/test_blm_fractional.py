"""Stage 4 review focus: BLM calibration (and the Cohon frontier) with a proportion solver."""
from __future__ import annotations

import warnings
from pathlib import Path

from pymarxan.calibration.blm import calibrate_blm
from pymarxan.calibration.pareto import pareto_frontier
from pymarxan.io.readers import load_project
from pymarxan.solvers import MIPSolver

DATA = Path(__file__).parent.parent.parent / "data"


def test_calibrate_blm_with_proportion_solver() -> None:
    problem = load_project(DATA / "simple")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        result = calibrate_blm(
            problem, MIPSolver(decisions="proportion", mip_backend="cbc"),
            blm_values=[0.0, 0.5, 1.0, 2.0, 5.0, 20.0],
        )
    assert result.blm_values == [0.0, 0.5, 1.0, 2.0, 5.0, 20.0]
    assert all(s.is_fractional for s in result.solutions)
    assert all(s.all_targets_met for s in result.solutions)
    b = result.boundaries
    assert all(b2 <= b1 + 1e-6 for b1, b2 in zip(b, b[1:])), b
    assert pareto_frontier(result).solutions
