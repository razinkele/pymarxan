"""Shared fixture for the target-method tests."""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem


def build_problem(totals: dict[int, float]) -> ConservationProblem:
    """A problem whose feature ``fid`` has native total ``totals[fid]``.

    Each feature with a non-zero total sits on its own planning unit; a
    zero-total feature has no ``pu_vs_features`` rows at all.
    """
    fids = list(totals)
    nonzero = [f for f in fids if totals[f] != 0]
    planning_units = pd.DataFrame(
        {
            "id": list(range(1, len(nonzero) + 1)),
            "cost": [1.0] * len(nonzero),
            "status": [0] * len(nonzero),
        }
    )
    features = pd.DataFrame(
        {
            "id": fids,
            "name": [f"f{f}" for f in fids],
            "target": [0.0] * len(fids),
            "spf": [1.0] * len(fids),
        }
    )
    pu_vs_features = pd.DataFrame(
        {
            "species": nonzero,
            "pu": list(range(1, len(nonzero) + 1)),
            "amount": [float(totals[f]) for f in nonzero],
        }
    )
    return ConservationProblem(planning_units, features, pu_vs_features)


@pytest.fixture
def make_problem() -> Callable[[dict[int, float]], ConservationProblem]:
    return build_problem
