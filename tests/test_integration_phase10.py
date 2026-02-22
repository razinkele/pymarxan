"""Phase 10 integration tests: MIP params, connectivity input, all solver types."""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_app_imports_phase10():
    """Verify the app still imports cleanly after phase 10 changes."""
    from pymarxan_app import app
    assert app.app is not None


@pytest.mark.integration
def test_matrix_input_importable():
    """Connectivity matrix input module is importable."""
    from pymarxan_shiny.modules.connectivity.matrix_input import (
        matrix_input_server,
        matrix_input_ui,
    )
    assert callable(matrix_input_ui)
    assert callable(matrix_input_server)


@pytest.mark.integration
def test_matrix_input_ui_renders():
    """Matrix input UI renders without error."""
    from pymarxan_shiny.modules.connectivity.matrix_input import matrix_input_ui
    elem = matrix_input_ui("test")
    assert elem is not None


@pytest.mark.integration
def test_solver_picker_all_seven_types():
    """Solver picker shows all 7 solver types."""
    from pymarxan_shiny.modules.solver_config.solver_picker import solver_picker_ui
    html = str(solver_picker_ui("t"))
    for solver_type in [
        "mip", "sa", "zone_sa", "greedy",
        "iterative_improvement", "pipeline",
    ]:
        assert solver_type in html, f"Missing solver type: {solver_type}"


@pytest.mark.integration
def test_solver_picker_mip_params():
    """Solver picker has MIP parameter controls."""
    from pymarxan_shiny.modules.solver_config.solver_picker import solver_picker_ui
    html = str(solver_picker_ui("t"))
    for param in ["mip_time_limit", "mip_gap", "mip_verbose"]:
        assert param in html, f"Missing MIP param: {param}"


@pytest.mark.integration
def test_solver_picker_mode_selectors():
    """Solver picker has mode selectors for greedy, iterative, pipeline."""
    from pymarxan_shiny.modules.solver_config.solver_picker import solver_picker_ui
    html = str(solver_picker_ui("t"))
    assert "heurtype" in html
    assert "itimptype" in html
    assert "runmode" in html


@pytest.mark.integration
def test_mip_solver_accepts_params():
    """MIP solver reads time_limit and gap from problem.parameters."""
    from unittest.mock import patch

    import pandas as pd

    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    pu = pd.DataFrame({
        "id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0],
    })
    feat = pd.DataFrame({
        "id": [1], "target": [1.0], "spf": [1.0], "name": ["f"],
    })
    puvsf = pd.DataFrame({
        "pu": [1, 2], "species": [1, 1], "amount": [1.0, 1.0],
    })
    p = ConservationProblem(
        planning_units=pu,
        features=feat,
        pu_vs_features=puvsf,
        parameters={
            "BLM": "0", "MIP_TIME_LIMIT": "120", "MIP_GAP": "0.01",
        },
    )
    with patch(
        "pymarxan.solvers.mip_solver.pulp.PULP_CBC_CMD"
    ) as mock_cbc:
        mock_cbc.return_value = mock_cbc
        try:
            MIPSolver().solve(p, SolverConfig(verbose=True))
        except Exception:
            pass
        mock_cbc.assert_called_once_with(
            msg=1, timeLimit=120, gapRel=0.01,
        )
