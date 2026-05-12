"""Tests for gap analysis module."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.analysis.gap_analysis import GapResult, compute_gap_analysis
from pymarxan.models.problem import ConservationProblem


@pytest.fixture()
def problem_with_protection() -> ConservationProblem:
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 20.0, 15.0, 25.0],
        "status": [2, 0, 2, 0],
    })
    feat = pd.DataFrame({
        "id": [1, 2],
        "name": ["bird", "mammal"],
        "target": [5.0, 8.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 2, 2, 2],
        "pu": [1, 2, 3, 2, 3, 4],
        "amount": [3.0, 4.0, 2.0, 5.0, 3.0, 4.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
    )


def test_gap_analysis_returns_result(problem_with_protection):
    result = compute_gap_analysis(problem_with_protection)
    assert isinstance(result, GapResult)


def test_gap_analysis_protected_amounts(problem_with_protection):
    result = compute_gap_analysis(problem_with_protection)
    assert result.protected_amount[1] == pytest.approx(5.0)
    assert result.protected_amount[2] == pytest.approx(3.0)


def test_gap_analysis_gap_values(problem_with_protection):
    result = compute_gap_analysis(problem_with_protection)
    assert result.gap[1] == pytest.approx(0.0)
    assert result.gap[2] == pytest.approx(5.0)


def test_gap_analysis_target_met(problem_with_protection):
    result = compute_gap_analysis(problem_with_protection)
    assert result.target_met[1] is True
    assert result.target_met[2] is False


def test_gap_analysis_to_dataframe(problem_with_protection):
    result = compute_gap_analysis(problem_with_protection)
    df = result.to_dataframe()
    assert len(df) == 2
    assert "feature_id" in df.columns
    assert "target" in df.columns
    assert "protected_amount" in df.columns
    assert "gap" in df.columns
    assert "percent_protected" in df.columns


def test_gap_analysis_no_protection():
    pu = pd.DataFrame({"id": [1, 2], "cost": [10.0, 20.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["f1"], "target": [5.0], "spf": [1.0]})
    puvspr = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [3.0, 4.0]})
    p = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=puvspr)
    result = compute_gap_analysis(p)
    assert result.protected_amount[1] == pytest.approx(0.0)
    assert result.gap[1] == pytest.approx(5.0)


def test_gap_analysis_applies_misslevel():
    """MISSLEVEL must scale the effective target — gap analysis previously
    compared against raw target, contradicting solver and exporter behaviour."""
    pu = pd.DataFrame({
        "id": [1, 2], "cost": [10.0, 20.0], "status": [2, 0],
    })
    feat = pd.DataFrame({
        "id": [1], "name": ["f1"], "target": [10.0], "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1], "pu": [1, 2], "amount": [6.0, 4.0],
    })
    # status=2 on PU 1 -> protected_amount = 6.0
    p_strict = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
        parameters={"MISSLEVEL": 1.0},
    )
    res_strict = compute_gap_analysis(p_strict)
    # raw target 10, protected 6 -> shortfall = 4, not met
    assert res_strict.gap[1] == pytest.approx(4.0)
    assert res_strict.target_met[1] is False

    p_relaxed = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
        parameters={"MISSLEVEL": 0.5},
    )
    res_relaxed = compute_gap_analysis(p_relaxed)
    # effective target = 5, protected = 6 -> shortfall = 0, met
    assert res_relaxed.gap[1] == pytest.approx(0.0)
    assert res_relaxed.target_met[1] is True
