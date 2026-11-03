"""The Probability 1D loss-probability source (v0.48.0, spec §3.3).

``_loss_probability_vector(problem, pu_id_to_idx, n_pu)`` gives each planning unit's
probability of **loss** ``p_i``, in planning-unit order: the pu.dat ``prob`` column when
it exists (Marxan 3 ``input.c`` ``readPlanningUnits``), else ``problem.probability``
(prob.dat) through ``_probability_vector`` (a PU missing from the table has p = 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import _has_1d_data, _loss_probability_vector


def _problem(pu_prob=None, table=None) -> ConservationProblem:
    pu = pd.DataFrame({"id": [7, 3, 9], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0]})
    if pu_prob is not None:
        pu["prob"] = pu_prob
    return ConservationProblem(
        planning_units=pu,
        features=pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]}),
        pu_vs_features=pd.DataFrame({"species": [1], "pu": [7], "amount": [1.0]}),
        probability=table,
    )


IDX = {7: 0, 3: 1, 9: 2}


def test_pu_dat_prob_column_is_the_source():
    p = _loss_probability_vector(_problem(pu_prob=[0.1, 0.2, 0.3]), IDX, 3)
    assert p.dtype == np.float64
    np.testing.assert_array_equal(p, [0.1, 0.2, 0.3])


def test_pu_dat_prob_is_cast_to_float():
    p = _loss_probability_vector(_problem(pu_prob=["0.5", "0", "1"]), IDX, 3)
    np.testing.assert_array_equal(p, [0.5, 0.0, 1.0])


def test_prob_dat_is_the_fallback_and_a_missing_pu_is_zero():
    table = pd.DataFrame({"pu": [9, 7], "probability": [0.4, 0.25]})
    p = _loss_probability_vector(_problem(table=table), IDX, 3)
    np.testing.assert_array_equal(p, [0.25, 0.0, 0.4])


def test_pu_dat_wins_over_prob_dat():
    table = pd.DataFrame({"pu": [7, 3, 9], "probability": [0.9, 0.9, 0.9]})
    p = _loss_probability_vector(_problem(pu_prob=[0.1, 0.2, 0.3], table=table), IDX, 3)
    np.testing.assert_array_equal(p, [0.1, 0.2, 0.3])


def test_no_source_gives_zeros():
    np.testing.assert_array_equal(_loss_probability_vector(_problem(), IDX, 3), [0, 0, 0])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_pu_dat_prob_raises_naming_the_pu(bad):
    with pytest.raises(ValueError, match=r"\[3\]"):
        _loss_probability_vector(_problem(pu_prob=[0.1, bad, 0.3]), IDX, 3)


def test_non_finite_prob_dat_raises():
    table = pd.DataFrame({"pu": [9], "probability": [np.nan]})
    with pytest.raises(ValueError, match="non-finite"):
        _loss_probability_vector(_problem(table=table), IDX, 3)


def test_has_1d_data():
    assert not _has_1d_data(_problem())
    assert _has_1d_data(_problem(pu_prob=[0.0, 0.0, 0.0]))
    assert _has_1d_data(_problem(table=pd.DataFrame({"pu": [7], "probability": [0.1]})))
