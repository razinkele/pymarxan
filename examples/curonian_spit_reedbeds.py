"""Curonian Spit reed beds — planning a cut/uncut mosaic along a linear shore.

The :doc:`Baltic example <baltic_marine_planning.py>` is the classic Marxan
problem: protect enough of each feature, as cheaply and as *compactly* as
possible. Reed beds on the lagoon shore of the Curonian Spit need the
opposite. The reed belt is a narrow, linear habitat, and the planning
premise here is that its biodiversity stays high only when the belt is kept
*fragmented*: a mosaic of cut (young, open) and uncut (old, dense) stretches
with many edges between them, rather than a few long uniform runs.

Scenario
--------
A park manager must decide, for one cutting season, which 1 km stretches of
the reed belt between Smiltynė and the state border south of Nida to cut.
We:

1.  Split the lagoon shore into 52 one-kilometre segments (a chain: each
    segment touches only its neighbours).
2.  Give each segment a reed area, a cutting cost (higher far from the
    villages that give road access) and two habitat amounts: old-reed
    nesting habitat, which counts only where the reed is *left*, and
    cut-reed mosaic habitat, which counts only where it is *cut*.
3.  Model it as Marxan with Zones: zone ``cut`` and zone ``leave``, every
    segment in exactly one (``MANDATORYALLOCATION = 1``), with zone targets
    for each habitat.
4.  Express "fragmentation is good" as a **negative zone boundary cost**
    between ``cut`` and ``leave``: every cut/uncut edge earns a reward. A
    plain BLM cannot do this — it only penalises edges.
5.  Solve exactly (zone MIP) with and without the edge reward, and sweep
    the reward to show the cost of each extra edge.

All data are synthetic and analytic (positions of the villages are
approximate), so the script is deterministic, needs no network access and
doubles as a regression fixture — see ``tests/test_examples.py``.

Every boundary length is 1, so each cut/uncut edge earns exactly the reward;
a zone boundary cost is multiplied by the shared boundary length, as in MarZone.

Run it::

    python examples/curonian_spit_reedbeds.py
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pymarxan.solvers.base import Solution
from pymarxan.zones import ZonalProblem

CUT, LEAVE = 1, 2
N_SEGMENTS = 52  # 1 km segments, Smiltynė (km 0) to the border south of Nida
DEFAULT_EDGE_REWARD = 1.0  # k€ credited per cut/uncut edge

# Approximate lagoon-shore distance (km) from Smiltynė; road access for cutting.
SETTLEMENTS = {"Smiltynė": 0.0, "Juodkrantė": 19.0, "Pervalka": 32.0, "Preila": 35.0,
               "Nida": 46.0}

OLD_REED, MOSAIC = 1, 2
LEAVE_TARGET_FRACTION = 0.5  # of the old-reed habitat must stay uncut
CUT_TARGET_FRACTION = 0.25  # of the mosaic habitat must be cut this season


@dataclass
class MosaicSummary:
    """How fragmented one cut/leave plan is, and what it costs."""

    edge_reward: float
    cutting_cost: float  # k€, zone costs only (the reward is not money spent)
    n_cut: int
    edges: int  # adjacent segment pairs where one is cut and the other is not
    longest_uncut_km: int
    longest_cut_km: int
    zone_targets_met: bool
    pattern: str  # one character per km: "C" cut, "·" left


def _shore_fields(n_segments: int) -> dict[str, np.ndarray]:
    """Reed area (ha), distance to road access (km) and habitat amounts per segment."""
    km = np.arange(n_segments) + 0.5
    reed_ha = (
        2.0
        + 6.0 * np.exp(-(((km - 26.0) / 6.0) ** 2))  # broad belt in the mid-spit bays
        + 4.0 * np.exp(-(((km - 40.0) / 4.0) ** 2))  # bay between Preila and Nida
        + 0.75 * (1.0 + np.sin(0.9 * km))  # small-scale shoreline variation
    )
    access = np.array(list(SETTLEMENTS.values()))
    dist = np.min(np.abs(km[:, None] - access[None, :]), axis=1)
    # Nesting in old reed is disturbance-sensitive, so it is best away from villages.
    old_reed = reed_ha * (1.0 - 0.8 * np.exp(-((dist / 3.0) ** 2)))
    # Cut-reed mosaic habitat scales with the reed area that a cut would open up.
    mosaic = reed_ha * (0.5 + 0.5 * np.cos(0.35 * km) ** 2)
    cut_cost = reed_ha * (0.4 + 0.12 * dist)  # k€: crew time plus haulage to the road
    return {"km": km, "reed_ha": reed_ha, "dist": dist, "old_reed": old_reed,
            "mosaic": mosaic, "cut_cost": cut_cost}


def build_reedbed_problem(
    edge_reward: float = DEFAULT_EDGE_REWARD, *, n_segments: int = N_SEGMENTS
) -> ZonalProblem:
    """The cut/leave zoning problem for the first ``n_segments`` km of the shore.

    ``edge_reward`` (k€ per cut/uncut edge) becomes the zone boundary cost
    ``-edge_reward`` for both zone orders; ``0`` means no zone boundary costs.
    """
    f = _shore_fields(n_segments)
    ids = np.arange(1, n_segments + 1)
    pu = pd.DataFrame({"id": ids, "cost": 0.0, "status": 0})
    pvf = pd.concat([
        pd.DataFrame({"species": OLD_REED, "pu": ids, "amount": f["old_reed"]}),
        pd.DataFrame({"species": MOSAIC, "pu": ids, "amount": f["mosaic"]}),
    ], ignore_index=True)
    features = pd.DataFrame({
        "id": [OLD_REED, MOSAIC],
        "name": ["old_reed_nesting", "cut_reed_mosaic"],
        "target": [0.0, 0.0],  # no overall targets: each habitat has a zone target
        "spf": [10.0, 10.0],
    })
    zone_costs = pd.concat([
        pd.DataFrame({"pu": ids, "zone": CUT, "cost": f["cut_cost"]}),
        pd.DataFrame({"pu": ids, "zone": LEAVE, "cost": 0.0}),
    ], ignore_index=True)
    zone_contributions = pd.DataFrame({
        "feature": [OLD_REED, MOSAIC], "zone": [LEAVE, CUT], "contribution": [1.0, 1.0],
    })
    zone_targets = pd.DataFrame({
        "zone": [LEAVE, CUT],
        "feature": [OLD_REED, MOSAIC],
        "target": [LEAVE_TARGET_FRACTION * f["old_reed"].sum(),
                   CUT_TARGET_FRACTION * f["mosaic"].sum()],
    })
    zone_boundary_costs = None
    if edge_reward != 0.0:
        zone_boundary_costs = pd.DataFrame({
            "zone1": [CUT, LEAVE], "zone2": [LEAVE, CUT], "cost": [-edge_reward] * 2,
        })
    return ZonalProblem(
        planning_units=pu,
        features=features,
        pu_vs_features=pvf,
        boundary=pd.DataFrame({"id1": ids[:-1], "id2": ids[1:], "boundary": 1.0}),
        parameters={"BLM": 0.0, "MANDATORYALLOCATION": 1},
        zones=pd.DataFrame({"id": [CUT, LEAVE], "name": ["cut", "leave"]}),
        zone_costs=zone_costs,
        zone_contributions=zone_contributions,
        zone_targets=zone_targets,
        zone_boundary_costs=zone_boundary_costs,
    )


def solve_mosaic(problem: ZonalProblem) -> Solution:
    """The exact optimum (zone MIP): cutting cost minus the edge reward."""
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.zones.mip_solver import ZoneMIPSolver

    return ZoneMIPSolver().solve(problem, SolverConfig(num_solutions=1))[0]


def _longest_run(is_zone: np.ndarray) -> int:
    longest = run = 0
    for flag in is_zone:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest


def summarize_mosaic(problem: ZonalProblem, solution: Solution) -> MosaicSummary:
    """Fragmentation and cost of a solved cut/leave plan."""
    from pymarxan.zones.objective import compute_zone_cost

    zones = np.asarray(solution.zone_assignment, dtype=int)
    zbc = problem.zone_boundary_costs
    return MosaicSummary(
        edge_reward=0.0 if zbc is None else -float(zbc["cost"].iloc[0]),
        cutting_cost=compute_zone_cost(problem, zones),
        n_cut=int((zones == CUT).sum()),
        edges=int((zones[1:] != zones[:-1]).sum()),
        longest_uncut_km=_longest_run(zones == LEAVE),
        longest_cut_km=_longest_run(zones == CUT),
        zone_targets_met=all(solution.metadata["zone_targets_met"].values()),
        pattern="".join("C" if z == CUT else "·" for z in zones),
    )


def reward_sweep(rewards: Sequence[float]) -> list[MosaicSummary]:
    """Solve the full shore once per edge reward: the fragmentation analogue of BLM calibration."""
    out = []
    for reward in rewards:
        problem = build_reedbed_problem(edge_reward=reward)
        out.append(summarize_mosaic(problem, solve_mosaic(problem)))
    return out


def _ruler(n_segments: int) -> str:
    """A line under the pattern with the first letter of each settlement at its km."""
    marks = [" "] * n_segments
    for name, km in SETTLEMENTS.items():
        if km < n_segments:
            marks[int(km)] = name[0]
    return "".join(marks)


def main() -> None:
    print("Curonian Spit lagoon-shore reed beds — cut/uncut mosaic")
    print("=" * 56)
    f = _shore_fields(N_SEGMENTS)
    print(f"Shore: {N_SEGMENTS} × 1 km segments, {f['reed_ha'].sum():.0f} ha of reed")
    print(f"Targets: leave {LEAVE_TARGET_FRACTION:.0%} of old-reed nesting habitat, "
          f"cut {CUT_TARGET_FRACTION:.0%} of the mosaic habitat")
    print()

    rows = reward_sweep([0.0, 0.25, 0.5, DEFAULT_EDGE_REWARD, 2.0, 4.0])
    print(" reward  cost k€  cut km  edges  longest uncut  longest cut")
    for r in rows:
        print(f" {r.edge_reward:6.2f}  {r.cutting_cost:7.1f}  {r.n_cut:6d}  {r.edges:5d}"
              f"  {r.longest_uncut_km:10d} km  {r.longest_cut_km:8d} km")
    print()

    plain = next(r for r in rows if r.edge_reward == 0.0)
    mosaic = next(r for r in rows if r.edge_reward == DEFAULT_EDGE_REWARD)
    print("Smiltynė (north) → border (south); C = cut this season")
    print(f"  no reward    {plain.pattern}")
    print(f"  reward {DEFAULT_EDGE_REWARD:.1f}   {mosaic.pattern}")
    print(f"               {_ruler(N_SEGMENTS)}")
    print("               (S Smiltynė, J Juodkrantė, P Pervalka/Preila, N Nida)")
    print()
    extra = mosaic.cutting_cost - plain.cutting_cost
    print(f"The mosaic has {mosaic.edges} cut/uncut edges instead of {plain.edges} "
          f"for {extra:+.1f} k€ of extra cutting.")


if __name__ == "__main__":
    main()
