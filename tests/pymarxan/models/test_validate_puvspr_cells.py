"""validate() reports the puvspr cells the PROBMODE 3 chance term rejects.

With a puvspr ``prob`` column under PROBMODE 3, every solver (through the cache's chance
cells and the evaluator) raises on a non-finite ``amount`` or ``prob`` in any row. validate()
must name the same cells beforehand, as it does for prob.dat; a non-numeric token (a ``-``
in the file) is non-finite too, with pymarxan's message rather than numpy's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.probability import evaluate_solution_chance


def _problem(prob, amount=(2.0, 3.0, 1.0), probmode=3) -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]})
    features = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "target": [2.0, 1.0], "spf": [1.0, 1.0],
        "ptarget": [0.9, -1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2], "pu": [1, 2, 3], "amount": list(amount), "prob": list(prob),
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters={"PROBMODE": probmode, "PROBABILITYWEIGHTING": 10.0},
    )


@pytest.mark.parametrize(
    ("prob", "amount", "pu_ids"),
    [
        ([0.5, np.nan, 1.0], (2.0, 3.0, 1.0), "[2]"),
        ([0.5, 0.5, np.inf], (2.0, 3.0, 1.0), "[3]"),
        ([0.5, "-", 1.0], (2.0, 3.0, 1.0), "[2]"),
        ([0.5, 0.5, 1.0], (2.0, np.nan, 1.0), "[2]"),
    ],
    ids=["nan_prob", "inf_prob_disabled_feature", "token_prob", "nan_amount"],
)
def test_validate_names_the_cells_the_chance_term_rejects(prob, amount, pu_ids):
    p = _problem(prob, amount)
    errors = p.validate()
    assert any("pu_vs_features has non-finite values" in e and pu_ids in e for e in errors)
    with pytest.raises(ValueError, match="pu_vs_features has non-finite values"):
        evaluate_solution_chance(p, np.ones(3, dtype=bool))


def test_valid_cells_and_other_modes_are_not_reported():
    assert _problem([0.5, 0.5, 1.0]).validate() == []
    # Outside PROBMODE 3 no chance cell is built, so nothing raises and nothing is reported.
    assert _problem([0.5, np.nan, 1.0], probmode=0).validate() == []
