"""Shared Stage 4 test fixtures (a helper module like ``lp_canon``, not a test file)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem

X = np.array([0.6, 0.2, 1.0])
PU_INDEX = {1: 0, 2: 1, 3: 2}


def three_pu(**params: float) -> ConservationProblem:
    """3 PUs (cost 10, 20, 30); f1 target 19 (amounts 10, 40, 5), f2 target 10 (8, -, 4)."""
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [10.0, 20.0, 30.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "target": [19.0, 10.0], "spf": [1.0, 1.0],
    })
    pvf = pd.DataFrame({
        "species": [1, 1, 1, 2, 2], "pu": [1, 2, 3, 1, 3],
        "amount": [10.0, 40.0, 5.0, 8.0, 4.0],
    })
    bnd = pd.DataFrame({
        "id1": [1, 1, 2, 3], "id2": [1, 2, 3, 3], "boundary": [3.0, 2.0, 4.0, 1.0],
    })
    conn = pd.DataFrame({"id1": [1, 2, 1], "id2": [2, 1, 3], "value": [3.0, 1.0, 2.0]})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf, boundary=bnd,
        parameters={"BLM": 1.0, "CONNECTIVITY_WEIGHT": 2.0, **params}, connectivity=conn,
    )
