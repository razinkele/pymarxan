"""Shared fixtures for the recovery tests: the ROWER 4-unit anchor in rev-2 semantics.

Units 1-4, cost [1, 1, 1, 2]; one habitat "wetland" (id 1) with existing amounts
[10, 6, 0, 0] and incremental potential [0, 0, 8, 12]; p_success [0.5, 0.5, 0.5, 0.8], so
expected potential is [0, 0, 4, 9.6] (max 13.6). protect_prop 0.5 -> existing target 8.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.recovery import RecoveryInputs
from pymarxan.zones import ZonalProblem, check_overall_targets
from pymarxan.zones.objective import compute_zone_cost


def anchor_base(status: tuple[int, ...] = (0, 0, 0, 0)) -> ConservationProblem:
    return ConservationProblem(
        planning_units=pd.DataFrame(
            {"id": [1, 2, 3, 4], "cost": [1.0, 1.0, 1.0, 2.0], "status": list(status)}
        ),
        features=pd.DataFrame({"id": [1], "name": ["wetland"], "target": [0.0], "spf": [1.0]}),
        pu_vs_features=pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [10.0, 6.0]}),
        parameters={"BLM": 0.0},
    )


def anchor_inputs(status: tuple[int, ...] = (0, 0, 0, 0), **kw: Any) -> RecoveryInputs:
    args: dict[str, Any] = {
        "potential": pd.DataFrame({"pu": [3, 4], "feature": [1, 1], "amount": [8.0, 12.0]}),
        "p_success": pd.Series({1: 0.5, 2: 0.5, 3: 0.5, 4: 0.8}),
        "protect_prop": 0.5,
        "restore_prop": 0.45,
        "restore_cost_factor": 2.5,
    }
    args.update(kw)
    return RecoveryInputs(base=anchor_base(status), **args)


def brute(problem: ZonalProblem) -> list[tuple[float, tuple[int, ...]]]:
    """Every rule-respecting assignment meeting all overall targets, cheapest first."""
    zones = [0, *sorted(int(z) for z in problem.zone_ids)]
    rules = problem.pu_zone_rules()
    out = []
    for combo in itertools.product(zones, repeat=len(rules)):
        a = np.array(combo, dtype=int)
        if any((z == 0 and not r.may_be_unassigned) or (z != 0 and z not in r.allowed)
               for z, r in zip(a, rules, strict=True)):
            continue
        if all(check_overall_targets(problem, a).values()):
            out.append((round(float(compute_zone_cost(problem, a)), 9), combo))
    return sorted(out)
