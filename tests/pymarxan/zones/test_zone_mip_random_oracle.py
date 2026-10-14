"""Seeded random-instance enumeration oracle for the zone budgeted objectives (S5)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZonalProblem, ZoneMIPSolver

N_PU, ZONES, N_F = 5, (1, 2, 3), 3
OBJECTIVES = ("max_features", "max_weighted_features", "min_shortfall", "min_largest_shortfall")


def _instance(seed: int, mandatory: int) -> tuple[ZonalProblem, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 10, size=(N_PU, len(ZONES))).astype(float)
    amount = rng.integers(0, 6, size=(N_PU, N_F)).astype(float)
    contrib = rng.choice([0.0, 0.5, 1.0], size=(len(ZONES), N_F))
    target = rng.integers(1, 12, size=N_F).astype(float)
    spf = rng.choice([1.0, 2.0, 5.0], size=N_F)
    ztarget = rng.integers(1, 8)  # one zone target: zone 2, feature 1
    # mandatory == 1: never draw below the forced floor (every PU pays its cheapest zone).
    low = cost.min(axis=1).sum() if mandatory else cost.min()
    budget = float(rng.uniform(low, cost.max(axis=1).sum()))
    pu = pd.DataFrame({"id": range(1, N_PU + 1), "cost": 0.0, "status": 0})
    pvf = pd.DataFrame([
        {"species": f + 1, "pu": i + 1, "amount": amount[i, f]}
        for i in range(N_PU) for f in range(N_F) if amount[i, f] > 0
    ])
    p = ZonalProblem(
        planning_units=pu,
        features=pd.DataFrame({"id": range(1, N_F + 1), "name": list("abc"),
                               "target": target, "spf": spf}),
        pu_vs_features=pvf, boundary=None,
        parameters={"BLM": 0.0, "COSTBUDGET": budget, "MANDATORYALLOCATION": mandatory},
        zones=pd.DataFrame({"id": list(ZONES), "name": ["z1", "z2", "z3"]}),
        zone_costs=pd.DataFrame([
            {"pu": i + 1, "zone": z, "cost": cost[i, k]}
            for i in range(N_PU) for k, z in enumerate(ZONES)
        ]),
        zone_contributions=pd.DataFrame([
            {"feature": f + 1, "zone": z, "contribution": contrib[k, f]}
            for k, z in enumerate(ZONES) for f in range(N_F)
        ]),
        zone_targets=pd.DataFrame({"zone": [2], "feature": [1], "target": [float(ztarget)]}),
        zone_boundary_costs=None,
    )
    data = {"cost": cost, "amount": amount, "contrib": contrib, "target": target,
            "spf": spf, "ztarget": np.array([ztarget], dtype=float), "budget": np.array([budget])}
    return p, data


def _best(objective: str, mandatory: int, d: dict[str, np.ndarray]) -> float:
    choices = ZONES if mandatory else (0, *ZONES)
    values: list[float] = []
    for a in itertools.product(choices, repeat=N_PU):
        spend = sum(d["cost"][i, z - 1] for i, z in enumerate(a) if z)
        if spend > d["budget"][0] + 1e-9:
            continue
        terms = []  # (achieved, required, weight)
        zone2 = sum(d["amount"][i, 0] for i, z in enumerate(a) if z == 2)
        terms.append((zone2, d["ztarget"][0], d["spf"][0]))
        for f in range(N_F):
            got = sum(d["amount"][i, f] * d["contrib"][z - 1, f] for i, z in enumerate(a) if z)
            terms.append((got, d["target"][f], d["spf"][f]))
        met = [got >= r - 1e-9 for got, r, _ in terms]
        rel = [max(0.0, r - got) / r for got, r, _ in terms]
        if objective == "max_features":
            values.append(float(sum(met)))
        elif objective == "max_weighted_features":
            values.append(sum(w for (_, _, w), m in zip(terms, met) if m))
        elif objective == "min_shortfall":
            values.append(sum(w * s for (_, _, w), s in zip(terms, rel)))
        else:
            values.append(max(rel))
    return max(values) if objective.startswith("max") else min(values)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("mandatory", [0, 1])
def test_zone_mip_matches_enumeration(seed: int, mandatory: int) -> None:
    p, d = _instance(seed, mandatory)
    for objective in OBJECTIVES:
        expected = _best(objective, mandatory, d)
        sols = ZoneMIPSolver(objective=objective).solve(p, SolverConfig(num_solutions=1))
        assert sols, (seed, mandatory, objective)
        assert sols[0].metadata["mip_objective_value"] == pytest.approx(expected, abs=1e-6), (
            seed, mandatory, objective,
        )
