"""Tests that MIP solver reads MIP_TIME_LIMIT, MIP_GAP, and verbose from config."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver


def _tiny_problem(**params: str) -> ConservationProblem:
    """Create a minimal 3-PU, 1-feature problem with optional parameters."""
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["f1"], "target": [1.0], "spf": [1.0]})
    puvspr = pd.DataFrame({"species": [1, 1, 1], "pu": [1, 2, 3], "amount": [1.0, 1.0, 1.0]})
    return ConservationProblem(
        planning_units=pu,
        features=feat,
        pu_vs_features=puvspr,
        parameters=dict(params),
    )


@patch("pymarxan.solvers.mip_solver.pulp.PULP_CBC_CMD")
def test_default_params(mock_cmd: MagicMock) -> None:
    """Default call uses msg=0, timeLimit=300, gapRel=0.0."""
    mock_cmd.return_value = MagicMock()
    solver = MIPSolver()
    problem = _tiny_problem()
    try:
        solver.solve(problem, SolverConfig(num_solutions=1))
    except Exception:
        pass
    mock_cmd.assert_called_once_with(msg=0, timeLimit=300, gapRel=0.0)


@patch("pymarxan.solvers.mip_solver.pulp.PULP_CBC_CMD")
def test_custom_time_limit(mock_cmd: MagicMock) -> None:
    """MIP_TIME_LIMIT='60' in parameters -> timeLimit=60."""
    mock_cmd.return_value = MagicMock()
    solver = MIPSolver()
    problem = _tiny_problem(MIP_TIME_LIMIT="60")
    try:
        solver.solve(problem, SolverConfig(num_solutions=1))
    except Exception:
        pass
    mock_cmd.assert_called_once_with(msg=0, timeLimit=60, gapRel=0.0)


@patch("pymarxan.solvers.mip_solver.pulp.PULP_CBC_CMD")
def test_custom_gap(mock_cmd: MagicMock) -> None:
    """MIP_GAP='0.05' in parameters -> gapRel=0.05."""
    mock_cmd.return_value = MagicMock()
    solver = MIPSolver()
    problem = _tiny_problem(MIP_GAP="0.05")
    try:
        solver.solve(problem, SolverConfig(num_solutions=1))
    except Exception:
        pass
    mock_cmd.assert_called_once_with(msg=0, timeLimit=300, gapRel=0.05)


@patch("pymarxan.solvers.mip_solver.pulp.PULP_CBC_CMD")
def test_verbose_flag(mock_cmd: MagicMock) -> None:
    """SolverConfig(verbose=True) -> msg=1."""
    mock_cmd.return_value = MagicMock()
    solver = MIPSolver()
    problem = _tiny_problem()
    try:
        solver.solve(problem, SolverConfig(num_solutions=1, verbose=True))
    except Exception:
        pass
    mock_cmd.assert_called_once_with(msg=1, timeLimit=300, gapRel=0.0)
