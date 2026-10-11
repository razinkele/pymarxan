"""Replacement cost's status-3 variant must not be overridden by a pulock (#6 R2-6)."""
from __future__ import annotations

import math
import warnings

import pandas as pd
import pytest

from pymarxan.analysis.replacement_cost import compute_replacement_cost
from pymarxan.zones import ZoneLockWarning, ZoneMIPSolver
from tests.pymarxan.zones.marzone_anchor import make_anchor_problem


def test_locked_pu_replacement_cost_not_zero():
    p = make_anchor_problem().copy_with(
        pu_locks=pd.DataFrame({"pu": [1], "zone": [1]}),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = compute_replacement_cost(p, solver=ZoneMIPSolver())  # type: ignore[arg-type]
    # Without PU1, PUs 2-3 cannot meet overall 15 and zone-2 10 together -> infeasible.
    assert scores[1] == math.inf


def _with_pu4():
    p = make_anchor_problem()
    pu = pd.concat(
        [p.planning_units, pd.DataFrame({"id": [4], "cost": [0.0], "status": [0]})],
        ignore_index=True,
    )
    zc = pd.concat(
        [p.zone_costs, pd.DataFrame({"pu": [4, 4], "zone": [1, 2], "cost": [3.0, 2.0]})],
        ignore_index=True,
    )
    return p.copy_with(planning_units=pu, zone_costs=zc)


def test_variant_keeps_other_pus_locks():
    p = _with_pu4().copy_with(pu_locks=pd.DataFrame({"pu": [1, 4], "zone": [2, 1]}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = compute_replacement_cost(p, solver=ZoneMIPSolver())  # type: ignore[arg-type]
    # Base (2, 1, 2, 1) = 8 + 3. PU 4's variant keeps PU 1's lock: (2, 1, 2, 0) = 8 -> -3.
    # Dropping every lock row instead gives (1, 2, 2, 0) = 7 -> -4.
    assert scores[4] == pytest.approx(-3.0)


def test_variant_drops_pu_zones_rows():
    p = _with_pu4().copy_with(pu_zones=pd.DataFrame({"pu": [4, 4], "zone": [1, 2]}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = compute_replacement_cost(p, solver=ZoneMIPSolver())  # type: ignore[arg-type]
    # Base (1, 2, 2, 2) = 9; variant (1, 2, 2, 0) = 7 -> -2 either way (status 3 beats
    # puzone). Keeping PU 4's puzone rows in its own variant is the "status 3 with puzone
    # rows" conflict, so the solver would warn; dropping them keeps the variant clean.
    assert scores[4] == pytest.approx(-2.0)
    assert not [w for w in caught if issubclass(w.category, ZoneLockWarning)]
