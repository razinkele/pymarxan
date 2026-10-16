"""pymarxan.multiobjective pure parts (Stage 3 spec §3, §4.1-4.3, §5 nondominated, §6)."""
from __future__ import annotations

import math

import numpy as np
import pulp
import pytest

from pymarxan.multiobjective import (
    Hierarchical,
    MaxFeatures,
    MaxWeightedFeatures,
    MaxWtdSum,
    MinBoundary,
    MinCost,
    MinLargestShortfall,
    MinShortfall,
    WeightedSum,
    nondominated,
    rel_tol_matrix,
    weights_matrix,
)
from pymarxan.solvers._mip_objectives import ObjectiveContext, TargetTerm
from pymarxan.solvers.base import Solution

# --- specs ----------------------------------------------------------------------


@pytest.mark.parametrize(("spec", "name", "sense"), [
    (MinCost(), "min_cost", "min"),
    (MinBoundary(), "min_boundary", "min"),
    (MinShortfall(), "min_shortfall", "min"),
    (MinLargestShortfall(), "min_largest_shortfall", "min"),
    (MaxFeatures(), "max_features", "max"),
    (MaxWeightedFeatures(), "max_weighted_features", "max"),
    (MaxWtdSum(), "max_wtd_sum", "max"),
])
def test_spec_name_sense_and_key(spec: object, name: str, sense: str) -> None:
    assert (spec.name, spec.sense, spec.key) == (name, sense, name)  # type: ignore[attr-defined]


def test_label_is_the_key_and_must_be_an_identifier() -> None:
    assert MinShortfall(label="rep").key == "rep"
    assert MinShortfall("absolute").shortfall == "absolute"  # shortfall is positional
    for bad in ("two words", "a-b", "1st", ""):
        with pytest.raises(ValueError, match="label"):
            MinCost(label=bad)


def test_unknown_shortfall_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="shortfall"):
        MinShortfall(shortfall="percent")
    with pytest.raises(ValueError, match="shortfall"):
        MinLargestShortfall(shortfall="percent")


def test_max_wtd_sum_weights_finite_any_sign() -> None:
    assert MaxWtdSum(weights={1: -2, 2: 0.5}).weights == {1: -2.0, 2: 0.5}
    assert MaxWtdSum(weights={1: 1}) != MaxWtdSum(weights={1: 2})  # weights take part in ==
    for bad in (math.nan, math.inf, "x"):
        with pytest.raises(ValueError, match="weights"):
            MaxWtdSum(weights={1: bad})  # type: ignore[dict-item]


def test_spec_build_prefixes_owned_names_with_the_key() -> None:
    m = pulp.LpProblem("o", pulp.LpMinimize)
    x = pulp.LpVariable("x_1", cat="Binary")
    ctx = ObjectiveContext(m, pulp.lpSum([x]), [TargetTerm("f1", "feature", 2 * x, 2.0, 1.0)])
    MinLargestShortfall(label="worst").build(ctx)
    MinLargestShortfall("absolute").build(ctx)
    assert {"worst__maxs_f1", "min_largest_shortfall__maxs_f1", "short_f1"} <= set(
        m.constraints
    )


# --- approaches -------------------------------------------------------------------


def test_hierarchical_rows() -> None:
    assert Hierarchical(0.1).rows(3) == [[0.1, 0.1]]
    assert Hierarchical([0.0, 2.0]).rows(3) == [[0.0, 2.0]]  # values > 1 allowed
    assert Hierarchical([[0.0, 0.5], [0.5, 0.0]]).rows(3) == [[0.0, 0.5], [0.5, 0.0]]
    assert Hierarchical(np.array([[0.1], [0.2]])).rows(2) == [[0.1], [0.2]]
    assert Hierarchical(0.3).rows(1) == [[]]
    assert Hierarchical(0.0).abs_tol == 1e-9
    assert Hierarchical(0.1) == Hierarchical(0.1) and WeightedSum([1, 2]) == WeightedSum([1, 2])


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, [0.1, -1.0], [[0.1], [math.nan]]])
def test_hierarchical_rel_tol_values(bad: object) -> None:
    with pytest.raises(ValueError, match="rel_tol"):
        Hierarchical(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-1e-9, math.inf])
def test_hierarchical_abs_tol(bad: float) -> None:
    with pytest.raises(ValueError, match="abs_tol"):
        Hierarchical(0.0, abs_tol=bad)


def test_hierarchical_row_length() -> None:
    with pytest.raises(ValueError, match="rel_tol row 1"):
        Hierarchical([[0.1, 0.2], [0.1]]).rows(3)


def test_weighted_sum_rows_and_errors() -> None:
    assert WeightedSum([1, 3]).rows(2) == [[1.0, 3.0]]
    assert WeightedSum([[1, 0], [0, 1]]).rows(2) == [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="weights row 0 has 3"):
        WeightedSum([1, 1, 1]).rows(2)
    with pytest.raises(ValueError, match="weights row 1 sums to 0"):
        WeightedSum([[1, 0], [0, 0]])
    for bad in ([-1, 1], [math.nan, 1], [[1, math.inf]]):
        with pytest.raises(ValueError, match="weights"):
            WeightedSum(bad)  # type: ignore[arg-type]


# --- prioritizr matrices (spec §4.3) -------------------------------------------------


def test_weights_matrix_2_5_is_prioritizrs_15_rows() -> None:
    assert weights_matrix(2, 5) == [
        [1, 1], [1, 0], [0, 1], [.5, .25], [.75, .25], [1, .25], [.25, .5], [.75, .5],
        [1, .5], [.25, .75], [.5, .75], [1, .75], [.25, 1], [.5, 1], [.75, 1],
    ]


def test_weights_matrix_counts_and_options() -> None:
    assert len(weights_matrix(3, 3)) == 22
    assert weights_matrix(np.int64(2), np.int64(5)) == weights_matrix(2, 5)  # numpy ints
    assert len(weights_matrix(3, 5)) == 112
    no_ext = weights_matrix(2, 5, include_extremes=False)
    assert no_ext[0] == [1, 1] and [1, 0] not in no_ext and len(no_ext) == 13
    # Without zeros: 4 points (.25 … 1) per column; the extremes still hold zeros.
    no_zero = weights_matrix(2, 4, include_zeros=False)
    assert no_zero[:3] == [[1, 1], [1, 0], [0, 1]] and no_zero[3] == [.5, .25]


def test_rel_tol_matrix_pins() -> None:
    assert rel_tol_matrix(3, 3, 1) == [[0, 0], [.5, 0], [1, 0], [.5, .5], [1, .5], [1, 1]]
    assert len(rel_tol_matrix(3, 5, 1)) == 15
    assert len(rel_tol_matrix(3, 5, 1, order=False)) == 25
    assert rel_tol_matrix(2, 3, 0.2) == [[0.0], [0.1], [0.2]]
    assert rel_tol_matrix(2, 2, 1, include_zeros=False) == [[0.5], [1.0]]


@pytest.mark.parametrize("args", [(1, 5), (2, 1), (2.0, 5)])
def test_matrix_helpers_reject_small_sizes(args: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="n_"):
        weights_matrix(*args)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_"):
        rel_tol_matrix(*args, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_rel_tol_matrix_rejects_bad_max(bad: float) -> None:
    with pytest.raises(ValueError, match="max"):
        rel_tol_matrix(3, 3, bad)


# --- nondominated (spec §5, §7.2) ---------------------------------------------------


def _sol(values: dict[str, float], senses: dict[str, str] | None = None) -> Solution:
    senses = senses or {k: "min" for k in values}
    return Solution(
        selected=np.zeros(1, dtype=bool), cost=0.0, boundary=0.0, objective=0.0,
        targets_met={}, metadata={"objective_values": values, "objective_senses": senses},
    )


def test_nondominated_ties_keep_the_first() -> None:
    a, b, c = _sol({"c": 1, "s": 2}), _sol({"c": 1, "s": 2}), _sol({"c": 2, "s": 1})
    b.cost = 99.0  # b equals a on every objective but is a different Solution
    kept = nondominated([a, b, c])
    assert len(kept) == 2 and kept[0] is a and kept[1] is c
    kept = nondominated([b, a, c])
    assert len(kept) == 2 and kept[0] is b and kept[1] is c


def test_nondominated_compares_exactly() -> None:
    # 1e-9 apart on both keys, in opposite directions: neither dominates, and they are not
    # identical, so both are kept (no tolerance).
    a, b = _sol({"c": 1.0, "s": 2.0}), _sol({"c": 1.0 + 1e-9, "s": 2.0 - 1e-9})
    kept = nondominated([a, b])
    assert len(kept) == 2 and kept[0] is a and kept[1] is b


def test_nondominated_drops_weakly_dominated() -> None:
    a, b = _sol({"c": 1, "s": 2}), _sol({"c": 1, "s": 3})
    assert nondominated([b, a]) == [a]


def test_nondominated_mixed_senses() -> None:
    senses = {"cost": "min", "held": "max"}
    a = _sol({"cost": 1, "held": 5}, senses)
    b = _sol({"cost": 2, "held": 4}, senses)  # dominated: costlier and holds less
    c = _sol({"cost": 3, "held": 9}, senses)
    assert nondominated([a, b, c]) == [a, c]


def test_nondominated_empty_and_errors() -> None:
    assert nondominated([]) == []
    with pytest.raises(ValueError, match="different"):
        nondominated([_sol({"c": 1}), _sol({"s": 1})])
    with pytest.raises(ValueError, match="different"):
        nondominated([_sol({"c": 1}), _sol({"c": 1}, {"c": "max"})])
    bare = _sol({"c": 1})
    bare.metadata = {}
    with pytest.raises(ValueError, match="lacks"):
        nondominated([bare])
