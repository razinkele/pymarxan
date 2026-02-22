"""Tests for convergence plot Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.results.convergence import (
    convergence_server,
    convergence_ui,
    extract_history,
)


def test_convergence_ui_returns_tag():
    ui_elem = convergence_ui("test_conv")
    assert ui_elem is not None


def test_convergence_server_callable():
    assert callable(convergence_server)


def test_extract_history_from_solutions():
    """extract_history returns list of histories from solutions."""
    import numpy as np

    from pymarxan.solvers.base import Solution

    sol1 = Solution(
        selected=np.array([True, False]),
        cost=10.0,
        boundary=5.0,
        objective=15.0,
        targets_met={1: True},
        metadata={
            "history": {
                "iteration": [0, 1000, 2000],
                "objective": [100.0, 80.0, 60.0],
                "best_objective": [100.0, 80.0, 60.0],
                "temperature": [10.0, 5.0, 1.0],
            },
            "run": 1,
        },
    )
    sol2 = Solution(
        selected=np.array([True, True]),
        cost=20.0,
        boundary=3.0,
        objective=23.0,
        targets_met={1: True},
        metadata={"run": 2},  # No history (e.g., MIP solver)
    )
    histories = extract_history([sol1, sol2])
    assert len(histories) == 1
    assert histories[0]["run"] == 1
    assert len(histories[0]["iteration"]) == 3


def test_extract_history_empty():
    histories = extract_history([])
    assert histories == []
