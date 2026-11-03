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
