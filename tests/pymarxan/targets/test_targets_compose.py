"""Tests for absolute/area/rule targets and the min/max combinators (spec §7)."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.targets import (
    absolute_targets,
    area_targets,
    max_targets,
    min_targets,
    relative_targets,
    rule_targets,
)


def test_absolute_scalar_not_clipped_even_at_zero_total(make_problem):
    p = make_problem({1: 0.0, 2: 3.0})
    assert absolute_targets(p, amount=5.0) == {1: 5.0, 2: 5.0}


def test_absolute_mapping_and_subset(make_problem):
    p = make_problem({1: 10.0, 2: 10.0, 3: 10.0})
    assert absolute_targets(p, amount={1: 2.5, 2: 7, 3: 1.0}, features=[2, 1]) == {
        1: 2.5,
        2: 7.0,
    }


def test_absolute_rejects_negative_and_positional_use(make_problem):
    p = make_problem({1: 10.0})
    with pytest.raises(ValueError, match="amount"):
        absolute_targets(p, amount=-1.0)
    with pytest.raises(TypeError):
        absolute_targets(p, 5.0)  # type: ignore[misc]


def test_area_targets_clip_and_units(make_problem):
    p = make_problem({1: 600.0, 2: 5_000.0})
    assert area_targets(p, km2_per_unit=1.0, area_km2=1_000.0) == {
        1: pytest.approx(600.0),
        2: pytest.approx(1_000.0),
    }
    # native unit = ha: 500,000 ha = 5,000 km²; 1,000 km² -> 100,000 ha
    ha = make_problem({1: 500_000.0})
    assert area_targets(ha, km2_per_unit=0.01, area_km2=1_000.0)[1] == pytest.approx(100_000.0)


def test_area_targets_cap(make_problem):
    p = make_problem({1: 5e6})
    assert area_targets(p, km2_per_unit=1.0, area_km2=2e6)[1] == 1e6
    assert area_targets(p, km2_per_unit=1.0, area_km2=2e6, cap_km2=None)[1] == 2e6


_RULES = {"endemic": 0.2, "harvested": -0.1}


@pytest.mark.parametrize(
    "traits, baseline, expected",
    [
        ({"endemic"}, 0.3, 5_000.0),  # 0.3 + 0.2
        ({"harvested"}, 0.3, 2_000.0),  # 0.3 - 0.1
        ({"endemic", "harvested"}, 0.3, 4_000.0),
        (set(), 0.3, 3_000.0),
        ({"endemic"}, 0.9, 10_000.0),  # 1.1 clamped to 1.0
        ({"harvested"}, 0.05, 0.0),  # -0.05 clamped to 0.0
    ],
)
def test_rule_targets(make_problem, traits, baseline, expected):
    p = make_problem({1: 10_000.0})
    out = rule_targets(p, km2_per_unit=1.0, baseline=baseline, rules=_RULES,
                       traits={1: traits})
    assert out[1] == pytest.approx(expected)


def test_rule_targets_joined_by_id_not_position(make_problem):
    p = make_problem({1: 10_000.0, 2: 10_000.0})
    out = rule_targets(p, km2_per_unit=1.0, baseline=0.3, rules=_RULES,
                       traits={2: {"endemic"}, 1: set()})
    assert out == {1: pytest.approx(3_000.0), 2: pytest.approx(5_000.0)}


def test_rule_targets_traits_series(make_problem):
    p = make_problem({1: 10_000.0, 2: 10_000.0})
    traits = pd.Series({2: {"endemic"}, 1: {"harvested"}})
    out = rule_targets(p, km2_per_unit=1.0, baseline=0.3, rules=_RULES, traits=traits)
    assert out == {1: pytest.approx(2_000.0), 2: pytest.approx(5_000.0)}


def test_rule_targets_rejects_non_mapping_traits(make_problem):
    # a plain list would otherwise apply to every feature positionally
    with pytest.raises(ValueError, match="traits must map feature id"):
        rule_targets(make_problem({1: 10.0, 2: 10.0}), km2_per_unit=1.0, baseline=0.3,
                     rules={"endemic": 0.2}, traits=["endemic"])  # type: ignore[arg-type]


def test_rule_targets_rejects_bare_string_trait(make_problem):
    # "ex" would otherwise be read as traits {"e", "x"}
    p = make_problem({1: 10_000.0})
    with pytest.raises(ValueError, match=r"feature\(s\) \[1\] must be a collection"):
        rule_targets(p, km2_per_unit=1.0, baseline=0.3, rules={"e": 0.1, "x": 0.1},
                     traits={1: "ex"})


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"traits": {1: {"migratory"}}}, r"traits not in rules: \['migratory'\]"),
        ({"traits": {}}, r"traits has no value for feature\(s\) \[1\]"),
        ({"baseline": 1.2}, "baseline out of range"),
        ({"rules": {"endemic": 1.5}}, r"rules\['endemic'\] out of range"),
    ],
)
def test_rule_targets_validation(make_problem, kw, match):
    args = {"baseline": 0.3, "rules": _RULES, "traits": {1: {"endemic"}}, **kw}
    with pytest.raises(ValueError, match=match):
        rule_targets(make_problem({1: 10.0}), km2_per_unit=1.0, **args)


def test_max_and_min_targets():
    a = {1: 100.0, 2: 50.0}
    b = {1: 80.0, 2: 90.0}
    assert max_targets(a, b) == {1: 100.0, 2: 90.0}
    assert min_targets(a, b) == {1: 80.0, 2: 50.0}
    assert list(max_targets({2: 1.0, 1: 1.0}, {1: 0.0, 2: 0.0})) == [2, 1]


def test_max_targets_three_inputs():
    assert max_targets({1: 1.0}, {1: 3.0}, {1: 2.0}) == {1: 3.0}


def test_combinators_reject_mismatched_keys_and_too_few_inputs():
    with pytest.raises(ValueError, match=r"only in input 0: \[2\]"):
        max_targets({1: 1.0, 2: 1.0}, {1: 2.0})
    with pytest.raises(ValueError, match="at least 2"):
        min_targets({1: 1.0})


def test_max_of_area_and_relative(make_problem):
    # the docstring example: at least 1,000 km² or 30%, whichever is larger
    p = make_problem({1: 2_000.0, 2: 10_000.0})
    out = max_targets(
        area_targets(p, km2_per_unit=1.0, area_km2=1_000.0), relative_targets(p, 0.3)
    )
    assert out == {1: pytest.approx(1_000.0), 2: pytest.approx(3_000.0)}


def test_rule_targets_non_str_trait_is_value_error(make_problem):
    with pytest.raises(ValueError, match=r"\[1\]"):
        rule_targets(
            make_problem({1: 100.0}),
            km2_per_unit=1.0,
            baseline=0.2,
            rules={"1": 0.1},
            traits={1: {1}},
        )


@pytest.mark.parametrize("bad", [None, 5])
def test_rule_targets_non_iterable_traits_is_value_error(make_problem, bad):
    with pytest.raises(ValueError, match=r"\[1\]"):
        rule_targets(
            make_problem({1: 100.0}),
            km2_per_unit=1.0,
            baseline=0.2,
            rules={"a": 0.1},
            traits={1: bad},
        )


def test_combine_non_id_keys_name_the_function():
    with pytest.raises(ValueError, match="max_targets keys"):
        max_targets({"a": 1.0}, {"a": 2.0})
