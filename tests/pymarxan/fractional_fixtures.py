"""Shared Stage 4 test fixtures (a helper module like ``lp_canon``, not a test file)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig

ONE = SolverConfig(num_solutions=1)

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


def with_conn(problem: ConservationProblem, values: list[float] | None = None,
              asymmetric: bool = False, weight: float = 2.0) -> ConservationProblem:
    conn = pd.DataFrame({
        "id1": [1, 2, 3, 2, 5], "id2": [2, 1, 4, 5, 6],
        "value": values or [4.0, 3.0, 2.5, 6.0, 1.5],
    })
    return problem.copy_with(
        connectivity=conn,
        parameters={**problem.parameters, "CONNECTIVITY_WEIGHT": weight,
                    "ASYMMETRIC_CONNECTIVITY": int(asymmetric)},
    )


Case = tuple[ConservationProblem, SolverConfig]


def rejected_cases(tiny: ConservationProblem) -> dict[str, Case]:
    """Problems / configs non-binary decisions reject (spec §3.5), keyed by the cause
    named in the error."""
    f = tiny.features
    return {
        "PROBMODE 3": (tiny.copy_with(parameters={**tiny.parameters, "PROBMODE": 3}), ONE),
        "clumping": (tiny.copy_with(features=f.assign(target2=[1.0, 0.0, 0.0])), ONE),
        "separation": (tiny.copy_with(features=f.assign(sepnum=[2, 0, 0],
                                                        sepdistance=[1.0, 0.0, 0.0])), ONE),
        "space": (tiny.copy_with(features=f.assign(space_target=[0.5, 0.0, 0.0])), ONE),
        "forbidden_selections": (tiny, SolverConfig(
            num_solutions=1, metadata={"forbidden_selections": [[0, 1, 0, 1, 0, 1]]})),
        "negative connectivity": (with_conn(tiny, [4.0, -3.0, 2.5, 6.0, 1.5]), ONE),
    }
