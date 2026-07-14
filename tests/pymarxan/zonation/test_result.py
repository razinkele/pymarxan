"""Tests for the ZonationResult container."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.zonation.result import ZonationResult


def _result() -> ZonationResult:
    # removal_order [3, 1, 2] → ranks 3=1/3, 1=2/3, 2=1.0
    return ZonationResult(
        priority_rank={3: 1 / 3, 1: 2 / 3, 2: 1.0},
        removal_order=[3, 1, 2],
        performance_curves=pd.DataFrame(
            {"prop_landscape_remaining": [1.0, 0.5, 0.0], "feat_1": [1.0, 0.5, 0.0]}
        ),
        rule="caz",
    )


def test_top_fraction_returns_highest_ranked():
    res = _result()
    # top 1/3 of 3 PUs = 1 cell = the rank-1.0 PU (id 2)
    assert res.top_fraction(1 / 3) == {2}
    # top 2/3 = 2 cells = ids 2 and 1
    assert res.top_fraction(2 / 3) == {2, 1}
    # top 1.0 = all
    assert res.top_fraction(1.0) == {1, 2, 3}


def test_top_fraction_rejects_out_of_range():
    res = _result()
    with pytest.raises(ValueError):
        res.top_fraction(0.0)
    with pytest.raises(ValueError):
        res.top_fraction(1.5)


def test_to_dataframe_columns_and_positions():
    res = _result()
    df = res.to_dataframe().set_index("pu_id")
    assert list(res.to_dataframe().columns) == [
        "pu_id",
        "priority_rank",
        "removal_position",
    ]
    # removal_position is the 0-indexed slot in removal_order
    assert df.loc[3, "removal_position"] == 0
    assert df.loc[2, "removal_position"] == 2
    assert df.loc[2, "priority_rank"] == pytest.approx(1.0)
