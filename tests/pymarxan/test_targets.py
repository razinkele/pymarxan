"""Tests for automatic target-setting rules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.targets import (
    apply_targets,
    group_targets,
    loglinear_targets,
    relative_targets,
)


def _problem() -> ConservationProblem:
    planning_units = pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]}
    )
    features = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["common", "rare"],
            "target": [0.0, 0.0],
            "spf": [1.0, 1.0],
        }
    )
    # common total = 100 ; rare total = 10
    pu_vs_features = pd.DataFrame(
        {
            "species": [1, 1, 2],
            "pu": [1, 2, 3],
            "amount": [60.0, 40.0, 10.0],
        }
    )
    return ConservationProblem(planning_units, features, pu_vs_features)


def test_relative_targets_are_fraction_of_total():
    problem = _problem()
    targets = relative_targets(problem, 0.30)
    assert targets == {1: pytest.approx(30.0), 2: pytest.approx(3.0)}


def test_loglinear_clamps_below_lower_and_above_upper():
    problem = _problem()
    # below lower_area (10 <= 10) -> upper_target fraction 1.0;
    # above upper_area (100 >= 100) -> lower_target fraction 0.1
    targets = loglinear_targets(
        problem,
        lower_area=10.0,
        lower_target=1.0,
        upper_area=100.0,
        upper_target=0.1,
    )
    # rare total 10 at/below lower_area -> 100% -> 10.0
    assert targets[2] == pytest.approx(10.0)
    # common total 100 at/above upper_area -> 10% -> 10.0
    assert targets[1] == pytest.approx(10.0)


def test_loglinear_interpolates_on_log_scale():
    # A feature whose total is the geometric midpoint of [10, 1000] is 100,
    # which sits halfway in log10 space, so its fraction is halfway too.
    planning_units = pd.DataFrame(
        {"id": [1], "cost": [1.0], "status": [0]}
    )
    features = pd.DataFrame(
        {"id": [1], "name": ["mid"], "target": [0.0], "spf": [1.0]}
    )
    pu_vs_features = pd.DataFrame(
        {"species": [1], "pu": [1], "amount": [100.0]}
    )
    problem = ConservationProblem(planning_units, features, pu_vs_features)
    targets = loglinear_targets(
        problem,
        lower_area=10.0,
        lower_target=1.0,
        upper_area=1000.0,
        upper_target=0.0,
    )
    # fraction = 1.0 + (0.0 - 1.0) * (log10(100)-log10(10))/(log10(1000)-log10(10))
    #          = 1.0 - 0.5 = 0.5 ; target = 0.5 * 100 = 50
    assert targets[1] == pytest.approx(50.0)


def test_group_targets_apply_group_fraction_to_members():
    problem = _problem()
    groups = {1: "abundant", 2: "scarce"}
    fractions = {"abundant": 0.10, "scarce": 0.50}
    targets = group_targets(problem, groups, fractions)
    # common total 100 * 0.10 = 10 ; rare total 10 * 0.50 = 5
    assert targets == {1: pytest.approx(10.0), 2: pytest.approx(5.0)}


def test_group_targets_unknown_group_raises():
    problem = _problem()
    with pytest.raises(ValueError, match="group"):
        group_targets(problem, {1: "x", 2: "y"}, {"x": 0.1})


def test_apply_targets_writes_targets_onto_features():
    problem = _problem()
    apply_targets(problem, {1: 30.0, 2: 3.0})
    by_id = dict(zip(problem.features["id"], problem.features["target"]))
    assert by_id == {1: pytest.approx(30.0), 2: pytest.approx(3.0)}


def test_apply_targets_leaves_unlisted_features_unchanged():
    problem = _problem()
    problem.features.loc[problem.features["id"] == 2, "target"] = 7.0
    apply_targets(problem, {1: 30.0})  # only feature 1 listed
    by_id = dict(zip(problem.features["id"], problem.features["target"]))
    assert by_id[1] == pytest.approx(30.0)
    assert by_id[2] == pytest.approx(7.0)
