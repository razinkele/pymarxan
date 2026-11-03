"""Zone SA can choose "no action" (zone 0) for units whose rule allows it (v0.49.0 R1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZonalProblem, ZoneMIPSolver, ZoneSASolver, compute_zone_objective
from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.model import fixed_zones
from pymarxan.zones.solver import _proposal_options
from tests.benchmarks.conftest import make_zone_problem


def _one_feature_problem(**params: object) -> ZonalProblem:
    """4 PUs, 2 zones, one feature (target 1) held only by PU 1 — review C1 probe."""
    return ZonalProblem(
        planning_units=pd.DataFrame({"id": [1, 2, 3, 4], "cost": [1.0] * 4, "status": [0] * 4}),
        features=pd.DataFrame({"id": [1], "name": ["h"], "target": [1.0], "spf": [10.0]}),
        pu_vs_features=pd.DataFrame({"species": [1], "pu": [1], "amount": [1.0]}),
        parameters={"BLM": 0.0, "NUMITNS": 20_000, "NUMTEMP": 100, **params},
        zones=pd.DataFrame({"id": [1, 2], "name": ["protect", "restore"]}),
        zone_costs=pd.DataFrame(
            [{"pu": p, "zone": z, "cost": float(z)} for p in (1, 2, 3, 4) for z in (1, 2)]
        ),
    )


def test_sa_leaves_unneeded_units_unassigned() -> None:
    sol = ZoneSASolver().solve(_one_feature_problem(), SolverConfig(num_solutions=1, seed=1))[0]
    assert sol.zone_assignment.tolist() == [1, 0, 0, 0]
    assert sol.cost == pytest.approx(1.0)


def test_sa_matches_mip_on_c1_probe() -> None:
    p = _one_feature_problem(MIP_GAP=0.0)
    mip = ZoneMIPSolver().solve(p, SolverConfig(num_solutions=1))[0]
    for seed in range(5):
        sa = ZoneSASolver().solve(p, SolverConfig(num_solutions=1, seed=seed))[0]
        assert sa.cost == pytest.approx(mip.cost)


def test_options_include_zero_only_when_unassigned_is_allowed() -> None:
    p = _one_feature_problem().copy_with(
        planning_units=pd.DataFrame(
            {"id": [1, 2, 3, 4], "cost": [1.0] * 4, "status": [0, 2, 3, 0]}
        ),
        pu_zones=pd.DataFrame({"pu": [4, 4], "zone": [1, 2]}),
    )
    rules = p.pu_zone_rules()
    opts = _proposal_options(rules, [1, 2])
    assert opts[0].tolist() == [0, 1, 2]  # free: no action is an option
    assert opts[3].tolist() == [1, 2]  # puzone with >= 2 zones forbids unassigned
    assert set(fixed_zones(rules)) == {1, 2}  # status 2 -> zone 1, status 3 -> 0


def test_mandatory_allocation_never_proposes_zero() -> None:
    p = _one_feature_problem(MANDATORYALLOCATION=1)
    assert all(0 not in o for o in _proposal_options(p.pu_zone_rules(), [1, 2]))
    sol = ZoneSASolver().solve(p, SolverConfig(num_solutions=1, seed=1))[0]
    assert (sol.zone_assignment > 0).all()


def test_start_draw_includes_no_action() -> None:
    p = _one_feature_problem(NUMITNS=0)
    starts = [
        ZoneSASolver().solve(p, SolverConfig(num_solutions=1, seed=s))[0].zone_assignment
        for s in range(10)
    ]
    assert any((a == 0).any() for a in starts)


def _rich_problem() -> ZonalProblem:
    """Every objective term active: BLM, boundary, zone boundary costs, zone and overall
    targets with differing contributions, connectivity."""
    p = make_zone_problem(n_pu=40, n_feat=5, n_zones=3, density=0.4, seed=7)
    contrib = pd.DataFrame(
        [{"feature": f, "zone": z, "contribution": c}
         for f in range(1, 6) for z, c in ((1, 1.0), (2, 0.5), (3, 0.2))]
    )
    ids = p.planning_units["id"].to_numpy()
    conn = pd.DataFrame({"id1": ids[:-1], "id2": ids[1:], "value": np.linspace(0.1, 2.0, 39)})
    return p.copy_with(
        zone_contributions=contrib,
        connectivity=conn,
        parameters={**p.parameters, "BLM": 0.7, "CONNECTIVITY_WEIGHT": 0.3},
    )


def test_tracked_objective_equals_full_objective_after_every_flip() -> None:
    p = _rich_problem()
    cache = ZoneProblemCache.from_zone_problem(p)
    rng = np.random.default_rng(3)
    assignment = rng.integers(0, 4, size=cache.n_pu)
    held = cache.compute_held(assignment)
    tracked = cache.compute_full_zone_objective(assignment, held=held, blm=0.7)
    to_zero = from_zero = 0
    for _ in range(400):
        idx = int(rng.integers(cache.n_pu))
        old, new = int(assignment[idx]), int(rng.integers(0, 4))
        if old == new:
            continue
        to_zero += new == 0
        from_zero += old == 0
        tracked += cache.compute_delta_zone_objective(
            idx, old, new, assignment, held=held, blm=0.7,
        )
        assignment[idx] = new
        cache.update_held(held, idx, old, new)
        assert tracked == pytest.approx(compute_zone_objective(p, assignment, 0.7), rel=1e-9)
    assert to_zero > 50 and from_zero > 50
