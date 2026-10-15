"""Cross-method invariants, public API and end-to-end tests (spec §9.3, §9.7)."""
from __future__ import annotations

import pytest

import pymarxan.targets as targets_pkg
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.targets import (
    apply_targets,
    area_targets,
    duran_targets,
    interp_area_targets,
    jung_targets,
    polak_targets,
    pop_size_targets,
    rl_ecosystem_targets,
    rl_species_targets,
    rodrigues_targets,
    rule_targets,
    sreekar_targets,
    ward_targets,
    watson_targets,
    wilson_targets,
)

EXPECTED_ALIASES = {"PerFeatureFloat", "PerFeatureStr"}
EXPECTED_PUBLIC = {
    "absolute_targets",
    "apply_targets",
    "area_targets",
    "duran_targets",
    "group_targets",
    "interp_absolute_targets",
    "interp_area_targets",
    "jung_targets",
    "loglinear_targets",
    "max_targets",
    "min_targets",
    "polak_targets",
    "pop_size_targets",
    "relative_targets",
    "rl_ecosystem_targets",
    "rl_species_targets",
    "rodrigues_targets",
    "rule_targets",
    "sreekar_targets",
    "ward_targets",
    "watson_targets",
    "wilson_targets",
}


def test_public_api_all_matches_expected():
    # pdoc renders an empty page for a re-exporting package without __all__
    assert set(targets_pkg.__all__) == EXPECTED_PUBLIC | EXPECTED_ALIASES
    for name in EXPECTED_PUBLIC:
        assert callable(getattr(targets_pkg, name))
    for name in EXPECTED_ALIASES:  # type aliases used in public signatures
        assert getattr(targets_pkg, name) is not None


# Every area-based method, as (name, callable taking problem + km2_per_unit).
AREA_METHODS = {
    "interp_area": lambda p, k: interp_area_targets(
        p, km2_per_unit=k, rare_area_threshold=1_000.0, rare_relative_target=0.5,
        rare_area_target=400.0, rare_method="max", common_area_threshold=100_000.0,
        common_relative_target=0.1, common_area_target=20_000.0, common_method="min",
        interp="log10",
    ),
    "rodrigues": lambda p, k: rodrigues_targets(p, km2_per_unit=k),
    "sreekar": lambda p, k: sreekar_targets(p, km2_per_unit=k),
    "polak": lambda p, k: polak_targets(p, km2_per_unit=k),
    "watson": lambda p, k: watson_targets(p, km2_per_unit=k),
    "rl_species": lambda p, k: rl_species_targets(
        p, km2_per_unit=k, status="EN", criterion_a="A1", criterion_b="B1", prop_uplift=0.1
    ),
    "rl_ecosystem": lambda p, k: rl_ecosystem_targets(
        p, km2_per_unit=k, status="VU", criterion_a="A3", criterion_b="B2"
    ),
    "jung": lambda p, k: jung_targets(p, km2_per_unit=k),
    "ward": lambda p, k: ward_targets(p, km2_per_unit=k),
    "pop_size": lambda p, k: pop_size_targets(
        p, km2_per_unit=k, pop_size=5_000.0, density_per_km2=0.5
    ),
    "wilson": lambda p, k: wilson_targets(
        p, km2_per_unit=k, mean_growth_rate=0.1, var_growth_rate=0.04, density_per_km2=0.01
    ),
    "duran": lambda p, k: duran_targets(
        p, km2_per_unit=k, probability_target=0.9, historical_area_km2=50_000.0, cap_km2=1e6
    ),
    "area": lambda p, k: area_targets(p, km2_per_unit=k, area_km2=3_000.0),
    "rule": lambda p, k: rule_targets(
        p, km2_per_unit=k, baseline=0.3, rules={"x": 0.2},
        traits={f: ({"x"} if f % 2 else set()) for f in range(1, 10)},
    ),
}

TOTALS_KM2 = {1: 0.0, 2: 50.0, 3: 900.0, 4: 2_500.0, 5: 10_000.0, 6: 60_000.0,
              7: 300_000.0, 8: 5e6, 9: 3e7}


@pytest.mark.parametrize("name", sorted(AREA_METHODS))
def test_targets_within_total_and_cap(make_problem, name):
    p = make_problem(TOTALS_KM2)
    out = AREA_METHODS[name](p, 1.0)
    assert list(out) == list(TOTALS_KM2)
    for fid, t in out.items():
        assert 0.0 <= t <= TOTALS_KM2[fid] * (1 + 1e-12), (fid, t)
        assert t <= 1e6 * (1 + 1e-12), (fid, t)
    assert out[1] == 0.0  # zero total -> zero target


@pytest.mark.parametrize("name", sorted(AREA_METHODS))
def test_rescaled_native_unit_scales_targets(make_problem, name):
    # Same ranges expressed in a unit 100x smaller: amounts x100, factor /100.
    km2 = AREA_METHODS[name](make_problem(TOTALS_KM2), 1.0)
    small = make_problem({f: v * 100.0 for f, v in TOTALS_KM2.items()})
    native = AREA_METHODS[name](small, 0.01)
    for fid in TOTALS_KM2:
        assert native[fid] == pytest.approx(km2[fid] * 100.0, rel=1e-12, abs=1e-9)


def test_end_to_end_rodrigues_mip(tiny_problem):
    # tests/data/simple totals: species 1 = 60, 2 = 45, 3 = 33 (native units).
    # Treat one unit as 1,000 km²: ranges 60,000 / 45,000 / 33,000 km² (log10 band).
    problem = tiny_problem.copy_with(features=tiny_problem.features.copy())
    t = rodrigues_targets(problem, km2_per_unit=1_000.0)
    totals = problem.feature_amounts()
    for fid, value in t.items():
        assert 0.0 < value < totals[fid]
    apply_targets(problem, t)
    sol = MIPSolver().solve(problem, SolverConfig(num_solutions=1))[0]
    assert all(sol.targets_met.values())
