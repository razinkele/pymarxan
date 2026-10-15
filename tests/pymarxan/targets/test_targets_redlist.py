"""Tests for the IUCN Red List target methods.

Pinned values are hand-computed from t = min(m(B(1+p), f(1-A)(1+p)), cap, f)
with the IUCN thresholds (spec §5); km2_per_unit=1 so the total is f in km².
"""
from __future__ import annotations

import pytest

from pymarxan.targets import (
    jung_targets,
    rl_ecosystem_targets,
    rl_species_targets,
    ward_targets,
)

REL = 1e-9


def _one(fn, make_problem, f, **kw):
    return fn(make_problem({1: f}), km2_per_unit=1.0, **kw)[1]


@pytest.mark.parametrize(
    "f, expected",
    [(3_000.0, 3_000.0), (10_000.0, 5_000.0), (50_000.0, 15_000.0)],
)
def test_rl_species_en_a1_b1(make_problem, f, expected):
    # max(5000, 0.3 f), clipped to f
    t = _one(rl_species_targets, make_problem, f, status="EN", criterion_a="A1",
             criterion_b="B1")
    assert t == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize("f, expected", [(20.0, 4.8), (1_000.0, 12.0)])
def test_rl_species_cr_min_with_uplift(make_problem, f, expected):
    # min(10*1.2, f*0.2*1.2)
    t = _one(rl_species_targets, make_problem, f, status="CR", criterion_a="A2",
             criterion_b="B2", prop_uplift=0.2, method="min")
    assert t == pytest.approx(expected, rel=REL)


def test_rl_species_en_b2_is_500(make_problem):
    # prioritizr docs say 50 km²; code and IUCN say 500. max(500, 0.5*600=300)
    t = _one(rl_species_targets, make_problem, 600.0, status="EN", criterion_a="A2",
             criterion_b="B2")
    assert t == pytest.approx(500.0, rel=REL)


@pytest.mark.parametrize(
    "f, expected",
    [(4_000.0, 4_000.0), (8_000.0, 5_000.0), (40_000.0, 20_000.0)],
)
def test_rl_ecosystem_vu_a3_is_half(make_problem, f, expected):
    # max(5000, (1-0.5) f). prioritizr's VU-A3 = 0.3 would give 0.7*40,000 = 28,000.
    t = _one(rl_ecosystem_targets, make_problem, f, status="VU", criterion_a="A3",
             criterion_b="B2")
    assert t == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize("crit", ["A2b", "a2b", "A2B", " a2B "])
def test_rl_ecosystem_a2b_any_case(make_problem, crit):
    # prioritizr can never match "A2b". max(2000, 0.5*10000)
    t = _one(rl_ecosystem_targets, make_problem, 10_000.0, status="en", criterion_a=crit,
             criterion_b="b2")
    assert t == pytest.approx(5_000.0, rel=REL)


@pytest.mark.parametrize("status", ["EN", "en", " En ", "eN\t"])
def test_rl_species_status_case_and_whitespace(make_problem, status):
    # spreadsheet values: max(5000, 0.3*50000) = 15,000 at EN/A1/B1
    t = _one(rl_species_targets, make_problem, 50_000.0, status=status, criterion_a="A1",
             criterion_b="B1")
    assert t == pytest.approx(15_000.0, rel=REL)


def test_rl_species_per_feature_status(make_problem):
    p = make_problem({1: 50_000.0, 2: 50_000.0})
    out = rl_species_targets(p, km2_per_unit=1.0, status={2: "VU", 1: "cr"},
                             criterion_a="A2", criterion_b="B1")
    # CR: max(100, 0.2*50000) = 10,000 ; VU: max(20000, 0.7*50000) = 35,000
    assert out == {1: pytest.approx(10_000.0), 2: pytest.approx(35_000.0)}


def test_rl_species_subset_with_status_only_for_subset(make_problem):
    p = make_problem({1: 50_000.0, 2: 50_000.0, 3: 50_000.0})
    out = rl_species_targets(p, km2_per_unit=1.0, status={3: "EN"}, criterion_a="A1",
                             criterion_b="B1", features=[3])
    assert out == {3: pytest.approx(15_000.0)}


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"status": "LC"}, r"status must be one of .*feature\(s\) \[1\]"),
        ({"status": {2: "VU"}}, r"status has no value for feature\(s\) \[1\]"),
        ({"criterion_a": "A5"}, "criterion_a must be one of"),
        ({"criterion_b": "B3"}, "criterion_b must be one of"),
        ({"prop_uplift": -0.1}, "prop_uplift out of range"),
        ({"method": "mean"}, "method must be 'max' or 'min'"),
    ],
)
def test_rl_species_validation(make_problem, kw, match):
    args = {"status": "VU", "criterion_a": "A2", "criterion_b": "B2", **kw}
    with pytest.raises(ValueError, match=match):
        rl_species_targets(make_problem({1: 10.0}), km2_per_unit=1.0, **args)


def test_rl_ecosystem_rejects_species_only_criterion(make_problem):
    with pytest.raises(ValueError, match="criterion_a"):
        rl_ecosystem_targets(make_problem({1: 10.0}), km2_per_unit=1.0, status="VU",
                             criterion_a="A4", criterion_b="B1")


@pytest.mark.parametrize(
    "f, expected",
    [
        (2_000.0, 2_000.0),  # max(2200, 1540) = 2200 -> clipped to f
        (2_500.0, 2_200.0),  # plateau
        (2_857.142857142857, 2_200.0),  # 2200/0.77: plateau ends
        (5_000.0, 3_850.0),  # 0.77 f
        (50_000.0, 38_500.0),
        (2e7, 1e6),
    ],
)
def test_jung(make_problem, f, expected):
    assert _one(jung_targets, make_problem, f) == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize(
    "f, expected",
    [(50.0, 50.0), (100.0, 100.0), (500.0, 100.0), (50_000.0, 10_000.0), (2e7, 1e6)],
)
def test_ward(make_problem, f, expected):
    # max(100, 0.2 f)
    assert _one(ward_targets, make_problem, f) == pytest.approx(expected, rel=REL)


def test_presets_reject_fixed_settings(make_problem):
    p = make_problem({1: 10.0})
    with pytest.raises(TypeError):
        jung_targets(p, km2_per_unit=1.0, criterion_a="A1")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ward_targets(p, km2_per_unit=1.0, method="min")  # type: ignore[call-arg]


def test_zero_total_is_zero(make_problem):
    p = make_problem({1: 0.0, 2: 5.0})
    assert jung_targets(p, km2_per_unit=1.0)[1] == 0.0


def test_rl_species_targets_rejects_positional_series(make_problem):
    problem = make_problem({1: 1e3, 2: 1e3, 3: 1e3, 4: 1e3})
    problem.features["status"] = ["CR", "EN", "VU", "VU"]
    with pytest.raises(ValueError, match="default RangeIndex"):
        rl_species_targets(
            problem, km2_per_unit=1.0, status=problem.features["status"],
            criterion_a="A1", criterion_b="B1", features=[1, 2],
        )
    ok = rl_species_targets(
        problem,
        km2_per_unit=1.0,
        status=problem.features.set_index("id")["status"],
        criterion_a="A1",
        criterion_b="B1",
        features=[1, 2],
    )
    assert set(ok) == {1, 2}
