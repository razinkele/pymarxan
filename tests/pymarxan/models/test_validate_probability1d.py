"""``validate()`` checks added for Probability 1D (v0.48.0, spec §3.1 / §3.3).

``validate()`` stays errors-only: an invalid ``PROBMODE`` (anything but 0–5), and ``prob``
values that are non-finite or outside [0, 1] on ``planning_units`` (always, even when no
mode reads them, spec §5) and on ``pu_vs_features``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem, _probmode_value


def _problem(pu_prob=None, pv_prob=None, amount=(1.0, 2.0), **params) -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]})
    if pu_prob is not None:
        pu["prob"] = pu_prob
    pv = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": list(amount)})
    if pv_prob is not None:
        pv["prob"] = pv_prob
    return ConservationProblem(
        planning_units=pu,
        features=pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]}),
        pu_vs_features=pv,
        parameters=dict(params),
    )


@pytest.mark.parametrize("mode", [0, 1, 2, 3, 4, 5, "4", 5.0])
def test_valid_probmodes_are_not_reported(mode):
    assert _problem(PROBMODE=mode).validate() == []


@pytest.mark.parametrize("mode", [6, -1, 3.5, "abc", True, None])
def test_invalid_probmode_is_reported(mode):
    errors = _problem(PROBMODE=mode).validate()
    assert any("PROBMODE" in e for e in errors), errors


def test_probmode_value_parser():
    assert _probmode_value(4) == 4
    assert _probmode_value("5") == 5
    assert _probmode_value(3.0) == 3
    assert _probmode_value(3.5) is None
    assert _probmode_value(True) is None
    assert _probmode_value(np.bool_(False)) is None
    assert _probmode_value(7) is None


@pytest.mark.parametrize("probmode", [0, 4])
def test_pu_prob_out_of_range_is_reported_whatever_the_mode(probmode):
    errors = _problem(pu_prob=[0.5, 1.5], PROBMODE=probmode).validate()
    assert any("planning_units prob" in e and "[0, 1]" in e for e in errors), errors


def test_pu_prob_non_finite_is_reported_with_the_ids():
    errors = _problem(pu_prob=[np.nan, 0.5]).validate()
    assert any("planning_units prob" in e and "[1]" in e for e in errors), errors


def test_valid_pu_prob_is_not_reported():
    assert _problem(pu_prob=[0.0, 1.0], PROBMODE=4).validate() == []


def test_puvspr_prob_out_of_range_is_reported():
    errors = _problem(pv_prob=[0.5, -0.1]).validate()
    assert any("pu_vs_features prob" in e and "[0, 1]" in e for e in errors), errors


@pytest.mark.parametrize("probmode", [3, 5])
def test_non_finite_puvspr_cells_reported_when_2d_requested(probmode):
    errors = _problem(pv_prob=[np.nan, 0.5], PROBMODE=probmode).validate()
    assert any("pu_vs_features has non-finite values" in e for e in errors), errors


def test_non_finite_amount_reported_when_1d_requested():
    errors = _problem(pu_prob=[0.1, 0.2], amount=(np.nan, 1.0), PROBMODE=4).validate()
    assert any("amount for planning unit IDs [1]" in e for e in errors), errors


def test_mode_4_does_not_read_puvspr_prob_cells():
    # The 1D term reads puvspr amounts only: a NaN puvspr prob is the 2D term's data.
    errors = _problem(pu_prob=[0.1, 0.2], pv_prob=[np.nan, 0.5], PROBMODE=4).validate()
    assert not any("non-finite" in e for e in errors), errors


def test_non_finite_amount_not_reported_without_a_1d_source():
    # Review L-2: mode 4 without pu.dat prob or prob.dat has the 1D term off; no solver
    # reads the amounts as chance cells, so validate() does not report them.
    errors = _problem(amount=(np.nan, 1.0), PROBMODE=4).validate()
    assert not any("non-finite" in e for e in errors), errors


def test_non_finite_amount_reported_with_a_prob_dat_1d_source():
    problem = _problem(amount=(np.nan, 1.0), PROBMODE=4).copy_with(
        probability=pd.DataFrame({"pu": [1, 2], "probability": [0.1, 0.2]}),
    )
    errors = problem.validate()
    assert any("amount for planning unit IDs [1]" in e for e in errors), errors


def test_puvspr_prob_range_error_names_the_pairs():
    # Review L-4: the offending (pu, species) pairs, the first few and the count.
    errors = _problem(pv_prob=[0.5, -0.1]).validate()
    (msg,) = [e for e in errors if "pu_vs_features prob" in e]
    assert "[0, 1]" in msg and "(2, 1)" in msg and "1 in all" in msg


def test_puvspr_prob_range_error_lists_the_first_five_pairs():
    pu = pd.DataFrame({"id": range(1, 8), "cost": 1.0, "status": 0})
    pv = pd.DataFrame({"species": [1] * 7, "pu": range(1, 8), "amount": 1.0,
                       "prob": [1.5] * 7})
    problem = ConservationProblem(
        planning_units=pu,
        features=pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]}),
        pu_vs_features=pv,
    )
    (msg,) = [e for e in problem.validate() if "pu_vs_features prob" in e]
    assert "(5, 1)" in msg and "(6, 1)" not in msg and "7 in all" in msg


def _with_ptargets(**cols) -> ConservationProblem:
    problem = _problem(pu_prob=[0.1, 0.2], pv_prob=[0.5, 0.5])
    features = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0],
                             "spf": [1.0, 1.0], **cols})
    return problem.copy_with(features=features)


@pytest.mark.parametrize("column", ["ptarget", "ptarget1d"])
def test_probability_target_above_one_is_reported_with_the_ids(column):
    # Review L-3: > 1 cannot be met; <= 0 (and blank) means disabled.
    errors = _with_ptargets(**{column: [0.5, 1.5]}).validate()
    (msg,) = [e for e in errors if f"features {column} " in e]
    assert "(0, 1]" in msg and "feature IDs [2]" in msg


@pytest.mark.parametrize("column", ["ptarget", "ptarget1d"])
def test_probability_target_infinite_is_reported(column):
    errors = _with_ptargets(**{column: [np.inf, 0.5]}).validate()
    assert any(f"features {column} " in e and "feature IDs [1]" in e for e in errors), errors


@pytest.mark.parametrize("column", ["ptarget", "ptarget1d"])
def test_probability_target_non_numeric_is_reported_with_the_ids(column):
    errors = _with_ptargets(**{column: ["abc", 0.5]}).validate()
    assert any(
        f"features {column} " in e and "non-numeric" in e and "feature IDs [1]" in e
        for e in errors
    ), errors


@pytest.mark.parametrize("column", ["ptarget", "ptarget1d"])
def test_disabled_and_valid_probability_targets_are_not_reported(column):
    for pair in ([-1.0, 0.0], [np.nan, -np.inf], [1e-9, 1.0], [-5.0, 0.95]):
        assert _with_ptargets(**{column: pair}).validate() == [], pair
