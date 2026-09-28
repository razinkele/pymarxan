"""Cross-check against a reimplementation of MarZone reserve.hpp:148-182 (master@85082e3).

Runs on tests/data/zones. MarZone accumulates two arrays per reserve: zoneSpec[(zone, species)]
+= raw amount (:164) and speciesAmounts[species] += amount × GetZoneContrib(species, zone)
(:170). Contributions come from zones.hpp:619 (0 for an unlisted pair when a contribution file
is supplied). Only the ``zonecontrib.dat`` dialect (species × zone, ``zones.hpp:621-627``) is
mirrored; ``zonecontrib2.dat`` (per zone, all species, ``:629-638``) and ``zonecontrib3.dat``
(per PU, ``:639-647``, ``GetZoneContrib`` at ``:169-177``) are not read. PUs in pymarxan's
zone 0 are skipped by construction (MarZone has no unassigned state; spec §8).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.zones.cache import ZoneProblemCache
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import _compute_zone_achieved, compute_overall_achieved
from pymarxan.zones.readers import load_zone_project

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"
ASSIGNMENTS = [(1, 2, 1, 2), (1, 1, 2, 0), (2, 2, 2, 2), (0, 0, 0, 0)]


def marzone_accumulate(
    problem: ZonalProblem, solution: np.ndarray,
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    contrib: dict[tuple[int, int], float] = {}
    if problem.zone_contributions is not None:
        for r in problem.zone_contributions.itertuples():
            contrib[(int(r.feature), int(r.zone))] = float(r.contribution)
    default = 1.0 if problem.zone_contributions is None else 0.0     # zones.hpp:619 / :651
    pu_ids = problem.planning_units["id"].tolist()
    zone_spec: dict[tuple[int, int], float] = {}
    species_amounts: dict[int, float] = {}
    for r in problem.pu_vs_features.itertuples():
        z = int(solution[pu_ids.index(int(r.pu))])
        if z == 0:
            continue
        isp, amt = int(r.species), float(r.amount)
        weighted = amt * contrib.get((isp, z), default)
        if amt:
            zone_spec[(z, isp)] = zone_spec.get((z, isp), 0.0) + amt                    # :164
        if weighted:
            species_amounts[isp] = species_amounts.get(isp, 0.0) + weighted            # :170
    return zone_spec, species_amounts


def _partial_table_problem() -> ZonalProblem:
    """Fixture minus the (feature 2, zone 2) row: exercises the zones.hpp:619 zero default.

    Built with ``copy_with`` (not the loader), so no partial-table warning fires.
    """
    p = load_zone_project(DATA_DIR)
    zc = p.zone_contributions
    keep = ~((zc["feature"] == 2) & (zc["zone"] == 2))
    return p.copy_with(zone_contributions=zc[keep].reset_index(drop=True))


@pytest.fixture(params=["with_table", "without_table", "partial_table"])
def problem(request):
    p = load_zone_project(DATA_DIR)
    if request.param == "without_table":
        p = p.copy_with(zone_contributions=None)
    elif request.param == "partial_table":
        p = _partial_table_problem()
    return p


def test_partial_table_arm_drops_the_unlisted_pair():
    """Guard: the partial arm really changes the overall amount (feature 2 in zone 2 -> 0)."""
    p = _partial_table_problem()
    assert p.contribution_gaps() == [(2, 2)]
    a = np.array((2, 2, 2, 2))
    assert compute_overall_achieved(p, a)[2] == 0.0
    assert compute_overall_achieved(load_zone_project(DATA_DIR), a)[2] > 0.0


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_overall_achieved_matches_marzone_species_amounts(problem, assignment):
    a = np.array(assignment)
    _, species_amounts = marzone_accumulate(problem, a)
    got = compute_overall_achieved(problem, a)
    for fid in problem.features["id"]:
        assert got[int(fid)] == pytest.approx(species_amounts.get(int(fid), 0.0))


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_zone_achieved_matches_marzone_zone_spec_raw(problem, assignment):
    a = np.array(assignment)
    zone_spec, _ = marzone_accumulate(problem, a)
    got = _compute_zone_achieved(problem, a)
    assert {k: pytest.approx(v) for k, v in zone_spec.items()} == got


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_cache_held_matches_both_marzone_arrays(problem, assignment):
    a = np.array(assignment)
    zone_spec, species_amounts = marzone_accumulate(problem, a)
    cache = ZoneProblemCache.from_zone_problem(problem)
    held = cache.compute_held(a)
    for (z, f), v in zone_spec.items():
        assert held.per_zone[cache.zone_id_to_col[z], cache.feat_id_to_col[f]] == pytest.approx(v)
    for f, v in species_amounts.items():
        assert held.overall[cache.feat_id_to_col[f]] == pytest.approx(v)
    assert held.per_zone.sum() == pytest.approx(sum(zone_spec.values()))
    assert held.overall.sum() == pytest.approx(sum(species_amounts.values()))
