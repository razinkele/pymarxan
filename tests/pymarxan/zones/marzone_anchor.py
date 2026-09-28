"""Spec §5 anchor for MarZone overall + raw zone targets (hand-verified; review oracle).

Three PUs, two zones, one feature. Amount 10 in every PU. Contributions listed for every pair:
zone 1 -> 1.0, zone 2 -> 0.4. Zone costs: zone 1 = (5, 6, 7), zone 2 = (1, 1, 1). Overall
target 15. Zone target (zone 2, feature 1) = 10 raw. MISSLEVEL 1, BLM 0, no boundary.
Unique feasible optimum (1, 2, 2) at cost 7; runner-up 8. At spf = 1 the penalised minimum is
the infeasible (2, 2, 2) at 6.0 (threshold spf > 4/3); the heuristic tier therefore uses
spf = 10, where the penalised argmin equals the hard optimum.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence

import pandas as pd

from pymarxan.zones.model import ZonalProblem

AMOUNT = 10.0
CONTRIB: dict[int, float] = {1: 1.0, 2: 0.4}
ZONE_COSTS: dict[int, tuple[float, float, float]] = {1: (5.0, 6.0, 7.0), 2: (1.0, 1.0, 1.0)}
OVERALL_TARGET = 15.0
ZONE_TARGET = 10.0  # zone 2, feature 1, raw units
OPTIMUM = (1, 2, 2)
OPTIMUM_COST = 7.0
RUNNER_UP_COST = 8.0
HEURISTIC_SPF = 10.0


def make_anchor_problem(
    *,
    spf: float = 1.0,
    zone_target_contrib: int = 0,
    blm: float = 0.0,
    misslevel: float = 1.0,
) -> ZonalProblem:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [0.0, 0.0, 0.0], "status": [0, 0, 0]})
    features = pd.DataFrame({
        "id": [1], "name": ["habitat"], "target": [OVERALL_TARGET], "spf": [spf],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1], "pu": [1, 2, 3], "amount": [AMOUNT, AMOUNT, AMOUNT],
    })
    zones = pd.DataFrame({"id": [1, 2], "name": ["protect", "restore"]})
    zone_costs = pd.DataFrame({
        "pu": [1, 2, 3, 1, 2, 3],
        "zone": [1, 1, 1, 2, 2, 2],
        "cost": [*ZONE_COSTS[1], *ZONE_COSTS[2]],
    })
    zone_contributions = pd.DataFrame({
        "feature": [1, 1], "zone": [1, 2], "contribution": [CONTRIB[1], CONTRIB[2]],
    })
    zone_targets = pd.DataFrame({"zone": [2], "feature": [1], "target": [ZONE_TARGET]})
    return ZonalProblem(
        planning_units=pu,
        features=features,
        pu_vs_features=puvspr,
        boundary=None,
        parameters={
            "BLM": blm,
            "MISSLEVEL": misslevel,
            "ZONETARGETCONTRIB": zone_target_contrib,
        },
        zones=zones,
        zone_costs=zone_costs,
        zone_contributions=zone_contributions,
        zone_targets=zone_targets,
        zone_boundary_costs=None,
    )


def oracle(
    assignment: Sequence[int],
    *,
    spf: float = 1.0,
    zone_target_contrib: int = 0,
) -> dict[str, float | bool]:
    """Spec §3 formulas written without pymarxan."""
    cost = sum(ZONE_COSTS[z][i] for i, z in enumerate(assignment) if z)
    overall = sum(AMOUNT * CONTRIB[z] for z in assignment if z)
    w = CONTRIB[2] if zone_target_contrib == 1 else 1.0
    zone2 = sum(AMOUNT * w for z in assignment if z == 2)
    penalty = spf * (max(0.0, OVERALL_TARGET - overall) + max(0.0, ZONE_TARGET - zone2))
    return {
        "cost": cost,
        "overall": overall,
        "zone2": zone2,
        "feasible": overall >= OVERALL_TARGET and zone2 >= ZONE_TARGET,
        "objective": cost + penalty,
    }


def all_assignments() -> list[tuple[int, int, int]]:
    return list(itertools.product([0, 1, 2], repeat=3))
