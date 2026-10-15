"""Tests for the population-based target methods (spec §6).

Pinned values are hand-computed from the formulas in each test comment;
km2_per_unit=1 unless stated.
"""
from __future__ import annotations

import math

import pytest

from pymarxan.targets import duran_targets, pop_size_targets, wilson_targets

REL = 1e-6


def test_pop_size_divides_by_density(make_problem):
    # n / d = 500 / 2 = 250 km² (prioritizr's docs say n*d; code divides)
    p = make_problem({1: 10_000.0})
    out = pop_size_targets(p, km2_per_unit=1.0, pop_size=500.0, density_per_km2=2.0)
    assert out[1] == pytest.approx(250.0, rel=REL)


def test_pop_size_clips_to_range(make_problem):
    # 10,000 / 0.5 = 20,000 km²: fits a 50,000 range, clipped on a 5,000 range
    p = make_problem({1: 50_000.0, 2: 5_000.0})
    out = pop_size_targets(p, km2_per_unit=1.0, pop_size=10_000.0, density_per_km2=0.5)
    assert out == {1: pytest.approx(20_000.0), 2: pytest.approx(5_000.0)}


def test_pop_size_per_feature_out_of_order_subset(make_problem):
    # id-keyed lookups: a positional bug would swap the two features' inputs
    p = make_problem({1: 1e5, 2: 1e5, 3: 1e5})
    out = pop_size_targets(
        p, km2_per_unit=1.0, pop_size={3: 300.0, 2: 200.0},
        density_per_km2={2: 1.0, 3: 3.0}, features=[3, 2],
    )
    assert out == {2: pytest.approx(200.0), 3: pytest.approx(100.0)}


def test_pop_size_native_units(make_problem):
    # native unit = ha: 1e6 ha = 10,000 km²; 250 km² -> 25,000 ha
    p = make_problem({1: 1e6})
    out = pop_size_targets(p, km2_per_unit=0.01, pop_size=500.0, density_per_km2=2.0)
    assert out[1] == pytest.approx(25_000.0, rel=REL)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"pop_size": 0.0}, "pop_size"),
        ({"density_per_km2": -1.0}, "density_per_km2"),
        ({"pop_size": {2: 1.0}}, r"pop_size has no value for feature\(s\) \[1\]"),
    ],
)
def test_pop_size_validation(make_problem, kw, match):
    args = {"pop_size": 500.0, "density_per_km2": 2.0, **kw}
    with pytest.raises(ValueError, match=match):
        pop_size_targets(make_problem({1: 10.0}), km2_per_unit=1.0, **args)


@pytest.mark.parametrize(
    "r, s2, k_pop",
    [
        (0.1, 0.04, 13.374806099),  # beta = 4, k = 32000 ** 0.25
        (0.3, 1e-4, 1.003173576),  # beta = 5999 (prioritizr's example)
    ],
)
def test_wilson_population(make_problem, r, s2, k_pop):
    # density 1/km² so the km² target equals k
    p = make_problem({1: 1e6})
    out = wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=r, var_growth_rate=s2,
                         density_per_km2=1.0)
    assert out[1] == pytest.approx(k_pop, rel=REL)


def test_wilson_overflow_maps_to_cap_or_range(make_problem):
    # r = 500.5, s2 = 1000 -> beta = 1e-3; 50 ** 1000 overflows -> k = inf
    p = make_problem({1: 2e7})
    capped = wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=500.5,
                            var_growth_rate=1_000.0, density_per_km2=1.0)
    uncapped = wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=500.5,
                              var_growth_rate=1_000.0, density_per_km2=1.0, cap_km2=None)
    assert capped[1] == 1e6
    assert uncapped[1] == 2e7


def test_wilson_divides_by_density(make_problem):
    # k / d = 13.374806099 / 0.5 (prioritizr's docs write d*k; its code divides)
    p = make_problem({1: 1e6, 2: 1e6})
    out = wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=0.1, var_growth_rate=0.04,
                         density_per_km2={2: 2.0, 1: 0.5})
    assert out == {1: pytest.approx(26.749612199, rel=REL),
                   2: pytest.approx(6.687403050, rel=REL)}


@pytest.mark.parametrize(
    "r, d",
    [
        # beta = 5e-6, 1e5*s2*beta**2/2 = 5e-8: the approximation gives k ~ 0 (no
        # protection) while Lande's exact model needs ~e^63 individuals
        (0.0200001, 1.0),
        # 1e5*s2*beta**2/2 = 1.0125: approximation k ~ 1.737 (173.7 km² target at
        # d = 0.01) vs exact ~1.7e22 individuals (target = cap)
        (0.02045, 0.01),
    ],
)
def test_wilson_inaccurate_approximation_raises(make_problem, r, d):
    p = make_problem({1: 1e6, 2: 1e6})
    with pytest.raises(ValueError, match=r"differs from Lande's exact.*feature\(s\) \[1\]"):
        wilson_targets(p, km2_per_unit=1.0, mean_growth_rate={1: r, 2: 0.1},
                       var_growth_rate=0.04, density_per_km2=d)


def test_wilson_lande_exact_solver():
    # expm1(x) - x = base; large base -> x ~ ln(base); small base -> x ~ sqrt(2 base)
    from pymarxan.targets._population import _lande_exact_x

    for base in (0.5, 1.0, 32_000.0, 1e300):
        x = _lande_exact_x(base)
        assert math.expm1(x) - x == pytest.approx(base, rel=1e-9)
    # tiny base: compare with the series inverse x = s - s²/6 + s³/36, s = sqrt(2 base)
    # (expm1(x) - x cancels catastrophically here, so it is not a usable check)
    for base in (1e-8, 1e-30, 1e-300):
        s = math.sqrt(2.0 * base)
        assert _lande_exact_x(base) == pytest.approx(s - s * s / 6 + s**3 / 36, rel=1e-9)


@pytest.mark.parametrize(
    "r, expected",
    [
        # beta = 0.7, base = 980: approx k = 18,757.7 vs exact 18,974.0 (1.14% apart)
        (0.034, None),
        # beta = 0.75, base = 1125: approx k = 11,700.47 vs exact 11,811.99 (0.94%)
        (0.035, 11_700.4715047),
    ],
)
def test_wilson_tolerance_is_one_percent(make_problem, r, expected):
    # approx k = (1e5*s2*beta**2/2)**(1/beta); exact x solves expm1(x) - x = base,
    # k = e**(x/beta). Returned values are prioritizr's approximation.
    p = make_problem({1: 2e7})
    kw = dict(km2_per_unit=1.0, mean_growth_rate=r, var_growth_rate=0.04, density_per_km2=1.0)
    if expected is None:
        with pytest.raises(ValueError, match="differs from Lande's exact"):
            wilson_targets(p, **kw)
    else:
        assert wilson_targets(p, **kw)[1] == pytest.approx(expected, rel=1e-6)


def test_wilson_compares_after_cap(make_problem):
    # beta = 0.05, base = 12.5: raw k differ ~200x (8.7e21 vs 1.7e24 individuals) but
    # both targets exceed the 1e6 km² cap, so they agree and no error is raised
    p = make_problem({1: 2e7})
    out = wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=0.0525, var_growth_rate=0.1,
                         density_per_km2=1.0)
    assert out[1] == 1e6


def test_wilson_small_variance_route_raises(make_problem):
    # r = s2 = 0.01: beta = 1, base = 50 -> approx k = 500 vs exact 507.2 (1.43% apart);
    # small variance, not r near s2/2, makes the approximation inaccurate here
    p = make_problem({1: 2e7})
    with pytest.raises(ValueError, match=r"r near σ²/2, or σ² small"):
        wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=0.01, var_growth_rate=0.01,
                       density_per_km2=1.0)


def test_wilson_base_underflow_names_feature(make_problem):
    # s2 = 1e-300 and beta ~ 4e-16 make 1e5*s2*beta**2/2 underflow to 0.0; the error
    # must still name the feature rather than be a bare "math domain error"
    p = make_problem({1: 1e4})
    with pytest.raises(ValueError, match=r"Lande's exact.*\[1\]"):
        wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=1e-300 / 2 * (1 + 4e-16),
                       var_growth_rate=1e-300, density_per_km2=1.0)


def test_wilson_nonpositive_beta_raises(make_problem):
    p = make_problem({1: 1e4, 2: 1e4})
    with pytest.raises(ValueError, match=r"beta > 0.*feature\(s\) \[2\]"):
        wilson_targets(p, km2_per_unit=1.0, mean_growth_rate={1: 0.1, 2: 0.02},
                       var_growth_rate=0.04, density_per_km2=1.0)


@pytest.mark.parametrize(
    "p_target, expected", [(0.9, 656.1), (0.95, 814.50625), (1.0, 1_000.0), (0.0, 0.0)]
)
def test_duran(make_problem, p_target, expected):
    # h * p**4 with h = 1,000 km²
    pr = make_problem({1: 5_000.0})
    out = duran_targets(pr, km2_per_unit=1.0, probability_target=p_target,
                        historical_area_km2=1_000.0)
    assert out[1] == pytest.approx(expected, rel=REL)


def test_duran_clips_to_current_range_and_has_no_default_cap(make_problem):
    # historical 3e6 km², p = 1 -> 3e6, current range 2e6 -> 2e6 (no 1e6 cap by default)
    pr = make_problem({1: 2e6})
    out = duran_targets(pr, km2_per_unit=1.0, probability_target=1.0,
                        historical_area_km2=3e6)
    assert out[1] == 2e6
    capped = duran_targets(pr, km2_per_unit=1.0, probability_target=1.0,
                           historical_area_km2=3e6, cap_km2=1e6)
    assert capped[1] == 1e6


def test_duran_validation(make_problem):
    pr = make_problem({1: 10.0})
    with pytest.raises(ValueError, match="probability_target"):
        duran_targets(pr, km2_per_unit=1.0, probability_target=1.2, historical_area_km2=5.0)
    with pytest.raises(ValueError, match="historical_area_km2"):
        duran_targets(pr, km2_per_unit=1.0, probability_target=0.5, historical_area_km2=0.0)
