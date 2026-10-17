"""Feature totals shared by representation and gap analysis (v0.45 PROBMODE consistency)."""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import _probmode2_active, compute_held


def expected_feature_totals(problem: ConservationProblem) -> dict[int, float]:
    """Feature id -> total amount: the denominator of representation and gap percentages.

    Under PROBMODE 2 with a probability table: the expected total ``Σ_i a_ij·(1 − p_i)``
    over every planning unit — ``compute_held`` of the all-selected plan, so selecting every
    unit is 100 % and the totals agree with ``build_solution``. Rows whose PU is not a
    planning unit are excluded and every feature id is a key (0.0 when it has no rows).

    Outside PROBMODE 2 these are the **raw** totals, ``feature_amounts()`` exactly as before
    (every ``pu_vs_features`` row, features with rows only).
    """
    if _probmode2_active(problem):
        held = compute_held(problem, np.ones(len(problem.planning_units), dtype=bool))
        return {int(f): float(held.get(int(f), 0.0)) for f in problem.features["id"]}
    return {int(f): float(a) for f, a in problem.feature_amounts().items()}
