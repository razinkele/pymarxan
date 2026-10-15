"""Tests for the shared target-method helpers (pymarxan.targets._units)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pymarxan.targets._units import (
    check_cap,
    check_scalar,
    feature_totals,
    finish_area,
    km2_factors,
    numeric_per_feature,
    per_feature,
    select_features,
)


def test_select_features_defaults_to_all_in_problem_order(make_problem):
    p = make_problem({3: 1.0, 1: 2.0, 2: 3.0})
    assert select_features(p, None) == [3, 1, 2]


def test_select_features_subset_keeps_problem_order(make_problem):
    p = make_problem({3: 1.0, 1: 2.0, 2: 3.0})
    assert select_features(p, [2, 3]) == [3, 2]


def test_select_features_accepts_numpy_ids_and_empty(make_problem):
    p = make_problem({1: 1.0, 2: 2.0})
    assert select_features(p, np.array([2], dtype=np.int64)) == [2]
    assert select_features(p, [2.0]) == [2]
    assert select_features(p, []) == []


@pytest.mark.parametrize("bad", [[1.7], ["1"], [True]])
def test_select_features_rejects_non_integral_ids(make_problem, bad):
    p = make_problem({1: 1.0, 2: 2.0})
    with pytest.raises(ValueError, match="features must be feature ids"):
        select_features(p, bad)


def test_select_features_rejects_boolean_mask(make_problem):
    # features=df["threatened"] is a natural mistake; True would read as id 1
    p = make_problem({1: 1.0, 2: 2.0})
    with pytest.raises(ValueError, match="boolean mask"):
        select_features(p, pd.Series([True, True]))


def test_select_features_unknown_id_raises(make_problem):
    p = make_problem({1: 1.0})
    with pytest.raises(ValueError, match=r"not in problem: \[9\]"):
        select_features(p, [1, 9])


def test_per_feature_scalar_and_mapping():
    assert per_feature("x", 2.0, [1, 2]) == {1: 2.0, 2: 2.0}
    assert per_feature("x", {2: "b", 1: "a", 7: "z"}, [1, 2]) == {1: "a", 2: "b"}


def test_per_feature_reads_series_by_index():
    series = pd.Series({1: 0.5, 2: 0.7})
    assert per_feature("x", series, [2, 1]) == {2: 0.7, 1: 0.5}
    # a DataFrame column indexed by feature id, in any row order
    df = pd.DataFrame({"id": [2, 1], "density": [0.7, 0.5]})
    assert per_feature("x", df.set_index("id")["density"], [1, 2]) == {1: 0.5, 2: 0.7}


def test_per_feature_series_duplicate_index_raises():
    with pytest.raises(ValueError, match=r"duplicate entries for feature\(s\) \[1\]"):
        per_feature("x", pd.Series([1.0, 100.0, 1.0], index=[1, 1, 2]), [1, 2])


def test_per_feature_accepts_numpy_and_integral_float_keys():
    assert per_feature("x", {np.int64(1): 5.0, 2.0: 6.0}, [1, 2]) == {1: 5.0, 2: 6.0}


@pytest.mark.parametrize("key", ["f1", 1.5, True])
def test_per_feature_rejects_non_id_keys(key):
    # name-keyed dicts and float keys must not be misread or truncated
    with pytest.raises(ValueError, match="x keys must be feature ids"):
        per_feature("x", {key: 1.0}, [1])


def test_per_feature_missing_key_names_feature():
    with pytest.raises(ValueError, match=r"x has no value for feature\(s\) \[2\]"):
        per_feature("x", {1: 1.0}, [1, 2])


def test_numeric_per_feature_rejects_nan_negative_and_text():
    with pytest.raises(ValueError, match=r"invalid for feature\(s\) \[1, 2, 3\]"):
        numeric_per_feature("d", {1: math.nan, 2: -1.0, 3: "abc", 4: 1.0}, [1, 2, 3, 4],
                            lower=0.0)


def test_numeric_per_feature_exclusive_lower_bound():
    with pytest.raises(ValueError, match="> 0.0"):
        numeric_per_feature("d", 0.0, [1], lower=0.0, lower_inclusive=False)
    assert numeric_per_feature("d", 0.0, [1], lower=0.0) == {1: 0.0}


def test_numeric_per_feature_upper_bound():
    with pytest.raises(ValueError, match="<= 1.0"):
        numeric_per_feature("p", 1.5, [1], lower=0.0, upper=1.0)


def test_check_scalar_and_cap():
    assert check_scalar("a", 3, lower=0.0) == 3.0
    with pytest.raises(ValueError, match="a out of range"):
        check_scalar("a", -1.0, lower=0.0)
    with pytest.raises(ValueError, match="must be a number"):
        check_scalar("a", "x")
    assert check_cap("cap_km2", None) is None
    with pytest.raises(ValueError, match="cap_km2"):
        check_cap("cap_km2", math.inf)


def test_feature_totals_zero_for_features_without_rows(make_problem):
    p = make_problem({1: 4.0, 2: 0.0})
    assert feature_totals(p, [1, 2]) == {1: 4.0, 2: 0.0}


def test_feature_totals_negative_raises(make_problem):
    p = make_problem({1: -4.0})
    with pytest.raises(ValueError, match=r"invalid for feature\(s\) \[1\]"):
        feature_totals(p, [1])


@pytest.mark.parametrize("amounts", [[100.0, math.nan], [100.0, -50.0], [math.inf, 1.0]])
def test_feature_totals_bad_row_raises_even_if_total_looks_fine(make_problem, amounts):
    # pandas sums skip NaN, and [100, -50] sums to a plausible 50: check rows
    p = make_problem({1: 1.0, 2: 1.0})
    p.pu_vs_features = pd.DataFrame(
        {"species": [1, 1, 2], "pu": [1, 2, 2], "amount": [*amounts, 3.0]}
    )
    with pytest.raises(ValueError, match=r"amounts must be finite and >= 0.*\[1\]"):
        feature_totals(p, [1, 2])
    assert feature_totals(p, [2]) == {2: 3.0}  # other features unaffected


def test_feature_totals_include_locked_out_planning_units(make_problem):
    # as in prioritizr, totals count every PU, locked-out (status 3) ones too
    p = make_problem({1: 4.0, 2: 6.0})
    p.planning_units.loc[p.planning_units["id"] == 1, "status"] = 3
    assert feature_totals(p, [1, 2]) == {1: 4.0, 2: 6.0}


def test_km2_factors_requires_positive_factor_for_every_feature():
    assert km2_factors(0.5, [1, 2]) == {1: 0.5, 2: 0.5}
    with pytest.raises(ValueError, match="km2_per_unit"):
        km2_factors(0.0, [1])
    with pytest.raises(ValueError, match=r"no value for feature\(s\) \[2\]"):
        km2_factors({1: 1.0}, [1, 2])


def test_finish_area_clips_caps_converts_and_zero():
    # t 50 km² of a 40 km² range -> 40 km² -> /k=0.01 -> 4,000 native units
    assert finish_area(50.0, 40.0, 0.01, None) == pytest.approx(4000.0)
    assert finish_area(5e6, 2e7, 1.0, 1e6) == 1e6
    assert finish_area(math.inf, 300.0, 1.0, None) == 300.0
    assert finish_area(10.0, 0.0, 1.0, 1e6) == 0.0
