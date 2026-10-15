"""Tests for the range-size interpolation target methods.

Pinned values are hand-computed from the published rule as implemented in
prioritizr 9.0.1 (spec §1 provenance); each test names the formula used.
With ``km2_per_unit=1`` the native total equals the range f in km².
"""
from __future__ import annotations

import warnings

import pytest

from pymarxan.targets import (
    interp_absolute_targets,
    interp_area_targets,
    loglinear_targets,
    polak_targets,
    rodrigues_targets,
    sreekar_targets,
    watson_targets,
)

REL = 1e-6


def _one(fn, make_problem, f, **kw):
    """Target for a single feature with range f (km2_per_unit=1)."""
    return fn(make_problem({1: f}), km2_per_unit=1.0, **kw)[1]


# --- interp_area_targets ---------------------------------------------------

_BOTH = dict(
    rare_area_threshold=1_000.0,
    rare_relative_target=0.5,
    rare_area_target=400.0,
    rare_method="max",
    common_area_threshold=100_000.0,
    common_relative_target=0.1,
    common_area_target=20_000.0,
    common_method="min",
    interp="log10",
)


@pytest.mark.parametrize(
    "f, expected",
    [
        (500.0, 400.0),  # f < a: max(400, 0.5*500=250)
        # a<=f<=e: f*(0.5 + (0.1-0.5)*(log10 2000 - 3)/(5 - 3)) = 2000*0.43979
        (2_000.0, 879.5880017),
        (500_000.0, 20_000.0),  # f > e: min(20000, 0.1*500000=50000)
    ],
)
def test_interp_area_with_area_targets(make_problem, f, expected):
    assert _one(interp_area_targets, make_problem, f, **_BOTH) == pytest.approx(expected, rel=REL)


def test_interp_area_linear_between_thresholds(make_problem):
    # linear: 5500 * (1 + (0.1-1)*(5500-1000)/(10000-1000)) = 5500*0.55
    t = _one(
        interp_area_targets, make_problem, 5_500.0,
        rare_area_threshold=1_000.0, rare_relative_target=1.0,
        common_area_threshold=10_000.0, common_relative_target=0.1, interp="linear",
    )
    assert t == pytest.approx(3_025.0, rel=REL)


def test_interp_area_converts_units(make_problem):
    # native unit = ha (0.01 km²): F = 1,000,000 ha -> f = 10,000 km²;
    # rodrigues-style rule gives 6,246.778 km² -> 624,677.8 ha
    p = make_problem({1: 1_000_000.0})
    t = interp_area_targets(
        p, km2_per_unit=0.01, rare_area_threshold=1_000.0, rare_relative_target=1.0,
        common_area_threshold=250_000.0, common_relative_target=0.1, interp="log10",
    )[1]
    assert t == pytest.approx(624_677.8498, rel=REL)


def test_interp_area_per_feature_factor_mapping(make_problem):
    p = make_problem({1: 500.0, 2: 500.0})
    out = interp_area_targets(
        p, km2_per_unit={1: 1.0, 2: 100.0}, rare_area_threshold=1_000.0,
        rare_relative_target=1.0, common_area_threshold=10_000.0,
        common_relative_target=0.1, interp="linear",
    )
    # f1 = 500 km² (rare, 100%) -> 500; f2 = 50,000 km² (common, 10%) -> 5,000 km² -> 50 native
    assert out == {1: pytest.approx(500.0), 2: pytest.approx(50.0)}


@pytest.mark.parametrize(
    "f, expected",
    [
        # f == a takes the interpolation branch: b*f = 500, NOT max(800, 500)
        (1_000.0, 500.0),
        (999.0, 800.0),  # f < a: max(800, 0.5*999)
        # f == e takes the interpolation branch: g*f = 10,000, NOT min(5,000, 10,000)
        (100_000.0, 10_000.0),
        (100_001.0, 5_000.0),  # f > e: min(5000, 0.1*100001)
    ],
)
def test_interp_area_threshold_boundaries(make_problem, f, expected):
    # prioritizr applies the rare/common area targets only for f < a and f > e
    t = _one(
        interp_area_targets, make_problem, f,
        rare_area_threshold=1_000.0, rare_relative_target=0.5, rare_area_target=800.0,
        rare_method="max", common_area_threshold=100_000.0, common_relative_target=0.1,
        common_area_target=5_000.0, common_method="min", interp="log10",
    )
    assert t == pytest.approx(expected, rel=REL)


def test_interp_area_zero_total_is_zero(make_problem):
    p = make_problem({1: 0.0, 2: 10.0})
    out = interp_area_targets(p, km2_per_unit=1.0, **_BOTH)
    assert out[1] == 0.0


def test_interp_area_features_subset(make_problem):
    p = make_problem({1: 500.0, 2: 500.0, 3: 500.0})
    out = interp_area_targets(p, km2_per_unit=1.0, features=[3, 1], **_BOTH)
    assert list(out) == [1, 3]


@pytest.mark.parametrize(
    "override, match",
    [
        ({"rare_area_threshold": 2e5}, "must not exceed"),
        ({"rare_area_threshold": 0.0}, "requires rare_area_threshold > 0"),
        ({"interp": "cubic"}, "interp must be one of"),
        ({"rare_relative_target": 1.5}, "rare_relative_target out of range"),
        ({"rare_method": None}, "must be given together"),
        ({"common_area_target": None}, "must be given together"),
        ({"rare_method": "mean"}, "rare_method must be 'min' or 'max'"),
        ({"cap_km2": -1.0}, "cap_km2 out of range"),
    ],
)
def test_interp_area_validation(make_problem, override, match):
    kw = {**_BOTH, **override}
    with pytest.raises(ValueError, match=match):
        interp_area_targets(make_problem({1: 10.0}), km2_per_unit=1.0, **kw)


def test_interp_area_requires_km2_per_unit(make_problem):
    with pytest.raises(TypeError):
        interp_area_targets(make_problem({1: 10.0}), **_BOTH)  # type: ignore[call-arg]


# --- interp_absolute_targets + loglinear equivalence ------------------------


def test_interp_absolute_native_units_and_cap(make_problem):
    p = make_problem({1: 50.0, 2: 5_000.0})
    out = interp_absolute_targets(
        p, rare_absolute_threshold=100.0, rare_relative_target=1.0,
        common_absolute_threshold=1_000.0, common_relative_target=0.2,
        interp="linear", cap=600.0,
    )
    # 50 < 100 -> 50; 5000 > 1000 -> 1000 -> capped at 600
    assert out == {1: pytest.approx(50.0), 2: pytest.approx(600.0)}


def test_interp_absolute_linear_allows_zero_threshold(make_problem):
    out = interp_absolute_targets(
        make_problem({1: 50.0}), rare_absolute_threshold=0.0, rare_relative_target=1.0,
        common_absolute_threshold=100.0, common_relative_target=0.0, interp="linear",
    )
    assert out[1] == pytest.approx(25.0)  # 50 * (1 - 50/100)


def test_interp_absolute_cap_validation(make_problem):
    with pytest.raises(ValueError, match="cap out of range"):
        interp_absolute_targets(
            make_problem({1: 50.0}), rare_absolute_threshold=10.0, rare_relative_target=1.0,
            common_absolute_threshold=100.0, common_relative_target=0.1, interp="log10",
            cap=-1.0,
        )


@pytest.mark.parametrize("total", [0.0, 5.0, 10.0, 31.6, 55.0, 100.0, 400.0])
def test_loglinear_equals_interp_absolute(make_problem, total):
    p = make_problem({1: total})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = loglinear_targets(
            p, lower_area=10.0, lower_target=1.0, upper_area=100.0, upper_target=0.1
        )
    new = interp_absolute_targets(
        p, rare_absolute_threshold=10.0, rare_relative_target=1.0,
        common_absolute_threshold=100.0, common_relative_target=0.1, interp="log10",
    )
    assert new[1] == pytest.approx(old[1], rel=1e-12, abs=0.0)


# --- presets ----------------------------------------------------------------


@pytest.mark.parametrize(
    "f, expected",
    [
        (500.0, 500.0),  # < 1,000: 100%
        # 1e4*(1 + (0.1-1)*(4-3)/(log10 250000 - 3))
        (10_000.0, 6_246.778498),
        (100_000.0, 24_935.569969),
        (5e6, 500_000.0),  # > 250,000: 10%
        (2e7, 1e6),  # 10% = 2e6 -> cap 1e6
    ],
)
def test_rodrigues(make_problem, f, expected):
    assert _one(rodrigues_targets, make_problem, f) == pytest.approx(expected, rel=REL)


def test_rodrigues_cap_onset_at_1e7(make_problem):
    assert _one(rodrigues_targets, make_problem, 1e7) == pytest.approx(1e6, rel=REL)
    assert _one(rodrigues_targets, make_problem, 9e6) == pytest.approx(9e5, rel=REL)
    assert _one(rodrigues_targets, make_problem, 2e7, cap_km2=None) == pytest.approx(2e6)


@pytest.mark.parametrize(
    "f, expected",
    [
        (1_000.0, 842.010584),  # 1e3*(1 + (0.3-1)*(3 - log10 200)/(log10 250000 - log10 200))
        (100_000.0, 38_994.707786),
        (250_000.0, 75_000.0),
        (5e6, 1e6),  # 30% = 1.5e6 -> cap
    ],
)
def test_sreekar(make_problem, f, expected):
    assert _one(sreekar_targets, make_problem, f) == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize(
    "f, expected",
    [
        (500.0, 500.0),
        (5_500.0, 3_025.0),  # 5500*(1 - 0.9*4500/9000)
        (9_000.0, 1_800.0),  # 9000*0.2 — larger than at 10,000 (non-monotone)
        (10_000.0, 1_000.0),
        (50_000.0, 5_000.0),
    ],
)
def test_polak(make_problem, f, expected):
    assert _one(polak_targets, make_problem, f) == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize(
    "f, expected",
    [
        (500.0, 500.0),  # min(500, 1000)
        (2_000.0, 1_000.0),
        (5_000.0, 1_000.0),
        # At the threshold both branches give 1,000; this only proves no crash
        # (prioritizr errors here: interpolation over two identical points).
        (10_000.0, 1_000.0),
        (50_000.0, 5_000.0),
        (2e7, 1e6),
    ],
)
def test_watson(make_problem, f, expected):
    assert _one(watson_targets, make_problem, f) == pytest.approx(expected, rel=REL)


def test_presets_reject_fixed_settings(make_problem):
    p = make_problem({1: 10.0})
    with pytest.raises(TypeError):
        rodrigues_targets(p, km2_per_unit=1.0, interp="linear")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        watson_targets(p, km2_per_unit=1.0, common_area_threshold=5.0)  # type: ignore[call-arg]


def test_preset_defaults_overridable(make_problem):
    # Rodrigues with a 30% common target: 5e6 km² -> 1.5e6 -> cap 1e6
    t = _one(rodrigues_targets, make_problem, 5e6, common_relative_target=0.3)
    assert t == pytest.approx(1e6)
