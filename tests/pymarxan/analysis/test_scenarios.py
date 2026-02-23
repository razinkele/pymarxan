"""Tests for scenario comparison module."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.analysis.scenarios import Scenario, ScenarioSet
from pymarxan.solvers.base import Solution


def _make_solution(cost: float, n_selected: int, n: int = 10) -> Solution:
    selected = np.zeros(n, dtype=bool)
    selected[:n_selected] = True
    return Solution(
        selected=selected,
        cost=cost,
        boundary=10.0,
        objective=cost + 10.0,
        targets_met={1: True, 2: cost < 50},
    )


def test_scenario_creation():
    sol = _make_solution(30.0, 5)
    s = Scenario(name="low-blm", solution=sol, parameters={"BLM": 0.1})
    assert s.name == "low-blm"
    assert s.solution.cost == 30.0


def test_scenario_set_add_and_list():
    ss = ScenarioSet()
    ss.add("a", _make_solution(30.0, 5), {"BLM": 0.1})
    ss.add("b", _make_solution(50.0, 3), {"BLM": 1.0})
    assert len(ss) == 2
    assert ss.names == ["a", "b"]


def test_scenario_set_compare_dataframe():
    ss = ScenarioSet()
    ss.add("a", _make_solution(30.0, 5), {"BLM": 0.1})
    ss.add("b", _make_solution(50.0, 3), {"BLM": 1.0})
    df = ss.compare()
    assert len(df) == 2
    assert "name" in df.columns
    assert "cost" in df.columns
    assert "n_selected" in df.columns
    assert "all_targets_met" in df.columns


def test_scenario_set_overlap():
    """Overlap is fraction of shared selected PUs."""
    ss = ScenarioSet()
    ss.add("a", _make_solution(30.0, 5), {})
    ss.add("b", _make_solution(30.0, 5), {})  # same selection
    matrix = ss.overlap_matrix()
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == pytest.approx(1.0)  # identical selections


def test_scenario_set_get():
    ss = ScenarioSet()
    ss.add("x", _make_solution(30.0, 5), {"BLM": 0.1})
    s = ss.get("x")
    assert s.name == "x"


def test_scenario_set_remove():
    ss = ScenarioSet()
    ss.add("a", _make_solution(30.0, 5), {})
    ss.remove("a")
    assert len(ss) == 0


def test_remove_nonexistent_raises():
    from pymarxan.analysis.scenarios import ScenarioSet
    ss = ScenarioSet()
    with pytest.raises(KeyError, match="nonexistent"):
        ss.remove("nonexistent")


def test_overlap_matrix_partial():
    """Jaccard index with partially overlapping selections."""
    ss = ScenarioSet()
    # A: PUs 0-4 selected (5 of 10)
    sol_a = _make_solution(30.0, 5)
    # B: PUs 0-2 selected, 3-9 not (3 of 10)
    selected_b = np.zeros(10, dtype=bool)
    selected_b[:3] = True
    sol_b = Solution(
        selected=selected_b, cost=20.0, boundary=5.0,
        objective=25.0, targets_met={1: True, 2: True},
    )
    ss.add("a", sol_a, {})
    ss.add("b", sol_b, {})
    matrix = ss.overlap_matrix()
    # Intersection: 3 PUs (0,1,2), Union: 5 PUs (0,1,2,3,4)
    expected_jaccard = 3.0 / 5.0
    assert matrix[0, 1] == pytest.approx(expected_jaccard)
    assert matrix[1, 0] == pytest.approx(expected_jaccard)
