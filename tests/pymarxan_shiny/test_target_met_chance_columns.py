"""Target-met table: one probability column group per active chance term (v0.48.0).

2D keeps ``ptarget`` / ``P(met)`` / ``prob_gap``; 1D adds ``ptarget1d`` / ``P1D(met)`` /
``prob1d_gap``. A column group shows only when its term is active and the solution
carries its gaps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import build_solution
from pymarxan_shiny.modules.results.target_met import chance_columns


def _problem(mode: int) -> ConservationProblem:
    return ConservationProblem(
        planning_units=pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": 0,
                                     "prob": [0.2, 0.4]}),
        features=pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [3.0, 1.0],
                               "spf": 1.0, "ptarget1d": [0.9, -1.0], "ptarget": [0.8, 0.6]}),
        pu_vs_features=pd.DataFrame({"species": [1, 1, 2], "pu": [1, 2, 2],
                                     "amount": [2.0, 2.0, 1.0], "prob": [0.9, 0.5, 1.0]}),
        parameters={"PROBMODE": mode},
    )


@pytest.mark.parametrize("mode, want", [
    (0, set()), (3, {"ptarget", "P(met)", "prob_gap"}),
    (4, {"ptarget1d", "P1D(met)", "prob1d_gap"}),
    (5, {"ptarget", "P(met)", "prob_gap", "ptarget1d", "P1D(met)", "prob1d_gap"}),
])
def test_columns_per_active_term(mode, want):
    p = _problem(mode)
    s = build_solution(p, np.array([True, True]), 0.0)
    cols = chance_columns(p, s, 1)
    assert set(cols) == want


def test_values_and_disabled_feature():
    p = _problem(4)
    s = build_solution(p, np.array([True, True]), 0.0)
    gap = s.prob1d_shortfalls[1]
    cols = chance_columns(p, s, 1)
    assert cols["ptarget1d"] == 0.9
    assert cols["prob1d_gap"] == round(gap, 4)
    assert cols["P1D(met)"] == round(max(0.0, 0.9 - gap), 4)
    assert chance_columns(p, s, 2) == {"ptarget1d": "—", "P1D(met)": "—", "prob1d_gap": "—"}
