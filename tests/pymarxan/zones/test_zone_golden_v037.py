"""v0.37 output of the zone SA / heuristic / II on a status-0/1/2/3 fixture (#6 spec §5).

The GOLDEN values were recorded on the unmodified v0.37 code; #6 must not change them
(no pu_locks / pu_zones here). Regenerate only with a written reason:
    PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python \
        tests/pymarxan/zones/test_zone_golden_v037.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZoneHeuristicSolver, ZoneIISolver, ZoneSASolver, load_zone_project

ZONES_DIR = Path(__file__).parent.parent.parent / "data" / "zones"
SOLVER_CASES = [
    ("sa", None, None), ("sa", 10.0, None),
    ("heuristic", None, None), ("heuristic", 10.0, None),
    ("ii", None, 1), ("ii", None, 2), ("ii", None, 3),
    ("ii", 10.0, 1), ("ii", 10.0, 2), ("ii", 10.0, 3),
]
CASES = [(fx, *case) for fx in ("small", "large") for case in SOLVER_CASES]
Golden = list[tuple[tuple[int, ...], float]]


def _small():
    """tests/data/zones (4 PUs, 2 zones), statuses 1/2/3/0 — converged SA."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = load_zone_project(ZONES_DIR)
    return p.copy_with(
        planning_units=p.planning_units.assign(status=[1, 2, 3, 0]),
        parameters={**p.parameters, "NUMITNS": 3000, "NUMTEMP": 30},
    )


def _large():
    """30 PUs + two tie PUs, 3 zones, statuses cycling 0/1/2/3/0/0, UNconverged SA
    (NUMITNS 100) so the output depends on the RNG stream; PU 31 (no features, zone costs
    5 / 0 / 7) gives II's swap pass a tie between zone 0 and zone 2 from its zone-1 seed,
    so candidate order matters.
    PU 32 (no features, zone costs 5 / -1 / -1) ties zones 2 and 3, both better than 0,
    so the order of the nonzero candidate zones in the II swap pass and the heuristic
    greedy pass matters too."""
    from tests.benchmarks.conftest import make_zone_problem

    p = make_zone_problem(n_pu=30, n_feat=5, n_zones=3, density=0.4, seed=3)
    statuses = [[0, 1, 2, 3, 0, 0][i % 6] for i in range(30)]
    pu = p.planning_units.assign(status=statuses)
    tie = {c: [0, 0] for c in pu.columns}
    tie.update({"id": [31, 32], "status": [0, 0]})
    pu = pd.concat([pu, pd.DataFrame(tie)], ignore_index=True)
    zc = pd.concat(
        [p.zone_costs, pd.DataFrame({"pu": [31, 31, 31, 32, 32, 32], "zone": [1, 2, 3, 1, 2, 3],
                                     "cost": [5.0, 0.0, 7.0, 5.0, -1.0, -1.0]})],
        ignore_index=True,
    )
    return p.copy_with(
        planning_units=pu, zone_costs=zc,
        parameters={**p.parameters, "NUMITNS": 100, "NUMTEMP": 10, "BLM": 0.5},
    )


def _fixture(fx: str, spf: float | None, itimptype: int | None):
    p = _small() if fx == "small" else _large()
    if itimptype is not None:
        p = p.copy_with(parameters={**p.parameters, "ITIMPTYPE": itimptype})
    if spf is not None:
        p = p.copy_with(features=p.features.assign(spf=spf))
    return p


def _run(fx: str, kind: str, spf: float | None, itimptype: int | None) -> Golden:
    p = _fixture(fx, spf, itimptype)
    solver = {"sa": ZoneSASolver, "heuristic": ZoneHeuristicSolver, "ii": ZoneIISolver}[kind]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sols = solver.solve(p, SolverConfig(num_solutions=2, seed=7))
    return [
        (tuple(int(z) for z in s.zone_assignment), round(float(s.objective), 6)) for s in sols
    ]


# small at spf None: the in-rule start (0,1,0,0) is already optimal, so use spf 10 where it moves
IMPROVE_SPF = {"small": 10.0, "large": None}


def _improve_start(p):
    """In-rule, non-optimal start: every PU in zone 0 except status-2 PUs at min(zone_ids)."""
    from pymarxan.zones.objective import build_zone_solution

    zmin = min(int(z) for z in p.zone_ids)
    start = [zmin if int(s) == 2 else 0 for s in p.planning_units["status"]]
    return start, build_zone_solution(
        p, np.array(start, dtype=int), float(p.parameters.get("BLM", 0.0)),
        solver_name="golden-start",
    )


def _run_improve(fx: str) -> Golden:
    """ZoneIISolver.improve() (no pu_locks / pu_zones) from an in-rule non-optimal start."""
    p = _fixture(fx, IMPROVE_SPF[fx], 3)
    _, start = _improve_start(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ZoneIISolver().improve(p, start)
    return [(tuple(int(z) for z in out.zone_assignment), round(float(out.objective), 6))]


GOLDEN: dict[str, Golden] = {
    'small-sa-None-None':
        [((1, 1, 0, 2), 318.8), ((1, 1, 0, 2), 318.8)],
    'small-sa-10.0-None':
        [((1, 1, 0, 2), 335.0), ((1, 1, 0, 2), 335.0)],
    'small-heuristic-None-None':
        [((0, 1, 0, 0), 184.0), ((0, 1, 0, 0), 184.0)],
    'small-heuristic-10.0-None':
        [((1, 1, 0, 2), 335.0), ((1, 1, 0, 2), 335.0)],
    'small-ii-None-1':
        [((0, 1, 0, 0), 184.0), ((0, 1, 0, 0), 184.0)],
    'small-ii-None-2':
        [((0, 1, 0, 0), 184.0), ((0, 1, 0, 0), 184.0)],
    'small-ii-None-3':
        [((0, 1, 0, 0), 184.0), ((0, 1, 0, 0), 184.0)],
    'small-ii-10.0-1':
        [((1, 1, 0, 0), 384.0), ((1, 1, 0, 0), 384.0)],
    'small-ii-10.0-2':
        [((1, 1, 0, 2), 335.0), ((1, 1, 0, 2), 335.0)],
    'small-ii-10.0-3':
        [((1, 1, 0, 2), 335.0), ((1, 1, 0, 2), 335.0)],
    'large-sa-None-None':
        [((2, 1, 1, 0, 3, 2, 1, 1, 1, 0, 1, 3, 2, 2, 1, 0, 2, 2, 2, 1, 1, 0, 3, 3, 1, 1, 1, 0, 1,
           2, 2, 3),
          1048.983571),
         ((1, 1, 1, 0, 3, 1, 2, 1, 1, 0, 1, 3, 2, 2, 1, 0, 2, 2, 2, 1, 1, 0, 3, 2, 1, 1, 1, 0, 1,
           2, 2, 3),
          931.972687)],
    'large-sa-10.0-None':
        [((2, 2, 1, 0, 3, 2, 1, 1, 1, 0, 1, 3, 2, 2, 1, 0, 2, 2, 2, 1, 1, 0, 3, 3, 1, 1, 1, 0, 1,
           2, 2, 2),
          2460.923847),
         ((1, 1, 1, 0, 3, 3, 2, 1, 1, 0, 3, 3, 2, 2, 1, 0, 3, 2, 2, 3, 1, 0, 3, 2, 3, 1, 1, 0, 1,
           2, 2, 3),
          2425.420875)],
    'large-heuristic-None-None':
        [((0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 3, 0, 1, 1, 1, 0, 0,
           0, 0, 2),
          488.579449),
         ((0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 3, 0, 1, 1, 1, 0, 0,
           0, 0, 2),
          488.579449)],
    'large-heuristic-10.0-None':
        [((2, 0, 1, 0, 3, 1, 2, 2, 1, 0, 0, 3, 2, 0, 1, 0, 0, 2, 2, 1, 1, 0, 3, 0, 1, 1, 1, 0, 1,
           0, 0, 2),
          2085.303055),
         ((2, 0, 1, 0, 3, 1, 2, 2, 1, 0, 0, 3, 2, 0, 1, 0, 0, 2, 2, 1, 1, 0, 3, 0, 1, 1, 1, 0, 1,
           0, 0, 2),
          2085.303055)],
    'large-ii-None-1':
        [((0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
           0, 0, 0),
          540.23398),
         ((0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
           0, 0, 0),
          540.23398)],
    'large-ii-None-2':
        [((0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 3, 0, 1, 1, 1, 0, 0,
           0, 0, 2),
          488.579449),
         ((0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 3, 0, 1, 1, 1, 0, 0,
           0, 0, 2),
          488.579449)],
    'large-ii-None-3':
        [((0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 3, 0, 1, 1, 1, 0, 0,
           0, 0, 2),
          488.579449),
         ((0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 3, 0, 1, 1, 1, 0, 0,
           0, 0, 2),
          488.579449)],
    'large-ii-10.0-1':
        [((1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
           0, 0, 0),
          2646.569426),
         ((1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
           0, 0, 0),
          2646.569426)],
    'large-ii-10.0-2':
        [((1, 0, 1, 0, 3, 1, 2, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 2, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
           0, 0, 2),
          2248.660891),
         ((1, 0, 1, 0, 3, 1, 2, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 2, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
           0, 0, 2),
          2248.660891)],
    'large-ii-10.0-3':
        [((2, 0, 1, 0, 3, 1, 2, 2, 1, 0, 0, 3, 2, 0, 1, 0, 0, 2, 2, 1, 1, 0, 3, 0, 1, 1, 1, 0, 1,
           0, 0, 2),
          2085.303055),
         ((2, 0, 1, 0, 3, 1, 2, 2, 1, 0, 0, 3, 2, 0, 1, 0, 0, 2, 2, 1, 1, 0, 3, 0, 1, 1, 1, 0, 1,
           0, 0, 2),
          2085.303055)],
}


GOLDEN_IMPROVE: dict[str, Golden] = {  # recorded on v0.37 source (a66459a), see __main__
    'small':
        [((1, 1, 0, 2), 335.0)],
    'large':
        [((0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 3, 0, 1, 1, 1, 0, 0,
           0, 0, 2),
          488.579449)],
}


@pytest.mark.parametrize("fx", ["small", "large"])
def test_improve_matches_v037(fx):
    start, _ = _improve_start(_fixture(fx, IMPROVE_SPF[fx], 3))
    assert tuple(start) != GOLDEN_IMPROVE[fx][0][0]  # improve() actually moved PUs
    assert _run_improve(fx) == GOLDEN_IMPROVE[fx]


@pytest.mark.parametrize("fx,kind,spf,itimptype", CASES)
def test_matches_v037(fx, kind, spf, itimptype):
    assert _run(fx, kind, spf, itimptype) == GOLDEN[f"{fx}-{kind}-{spf}-{itimptype}"]


if __name__ == "__main__":
    import pprint

    print("GOLDEN: dict[str, Golden] = {")
    for fx, kind, spf, itimptype in CASES:
        body = pprint.pformat(_run(fx, kind, spf, itimptype), width=91, compact=True)
        print(f"    {f'{fx}-{kind}-{spf}-{itimptype}'!r}:\n        "
              + body.replace("\n", "\n        ") + ",")
    print("}")
    print("GOLDEN_IMPROVE: dict[str, Golden] = {")
    for fx in ("small", "large"):
        body = pprint.pformat(_run_improve(fx), width=91, compact=True)
        print(f"    {fx!r}:\n        " + body.replace("\n", "\n        ") + ",")
    print("}")
