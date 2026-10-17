"""Stage 4 (spec §5.2, §5.3): fraction-linear evaluators, the target tolerance and
``binarize_solution``. Every number below is a hand calculation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers import binarize_solution
from pymarxan.solvers.utils import (
    TARGET_REL_TOL,
    _compute_achieved,
    _is_fractional,
    build_solution,
    check_targets,
    compute_boundary,
    compute_connectivity_penalty,
    compute_feature_shortfalls,
    compute_objective_terms,
    compute_probability_penalty,
    target_tol,
)
from tests.pymarxan.fractional_fixtures import PU_INDEX, X, three_pu


def test_target_tol() -> None:
    assert TARGET_REL_TOL == 1e-7
    assert target_tol(0.0) == 0.0
    assert target_tol(-1e8) == pytest.approx(10.0)


def test_is_fractional() -> None:
    assert _is_fractional(X)
    assert _is_fractional(np.array([1.0, 0.0]))
    assert not _is_fractional(np.array([True, False]))


def test_hand_calculation() -> None:
    p = three_pu()
    held = _compute_achieved(p, X, PU_INDEX)
    assert held == pytest.approx({1: 6.0 + 8.0 + 5.0, 2: 4.8 + 4.0})  # 19.0, 8.8
    # 3·0.6 + 2·|0.6 − 0.2| + 4·|0.2 − 1| + 1·1
    assert compute_boundary(p, X, PU_INDEX) == pytest.approx(6.8)
    # symmetric, reward-only −v·min(x_i, x_j): (1,2,3): −3·0.2; (2,1,1) repeats the pair
    # (first row wins); (1,3,2): −2·0.6
    assert compute_connectivity_penalty(p, X, PU_INDEX) == pytest.approx(2.0 * -1.8)
    # asymmetric v·max(0, x_i − x_j): (1,2,3): 3·0.4; (2,1,1): 0; (1,3,2): 0
    asym = p.copy_with(parameters={**p.parameters, "ASYMMETRIC_CONNECTIVITY": 1})
    assert compute_connectivity_penalty(asym, X, PU_INDEX) == pytest.approx(2.0 * 1.2)
    assert check_targets(p, X, PU_INDEX) == {1: True, 2: False}
    assert compute_feature_shortfalls(p, X, PU_INDEX) == pytest.approx({1: 0.0, 2: 1.2})
    s = build_solution(p, X, 1.0)
    assert s.is_fractional
    assert s.selected.dtype == np.float64
    assert s.cost == pytest.approx(6.0 + 4.0 + 30.0)
    assert s.boundary == pytest.approx(6.8)
    assert s.penalty == pytest.approx(1.2)
    assert s.shortfall == pytest.approx(1.2)
    assert s.objective == pytest.approx(40.0 + 6.8 + 1.2 - 3.6)
    assert s.n_selected == 3
    assert s.selected_sum == pytest.approx(1.8)


def test_formulas_reduce_to_binary() -> None:
    """A 0/1 float array gives the binary numbers (only the dtype differs)."""
    p = three_pu()
    b = np.array([True, False, True])
    f = b.astype(float)
    sb, sf = build_solution(p, b, 1.0), build_solution(p, f, 1.0)
    for attr in ("cost", "boundary", "objective", "penalty", "shortfall"):
        assert getattr(sf, attr) == pytest.approx(getattr(sb, attr), abs=1e-12), attr
    assert sf.targets_met == sb.targets_met
    assert compute_connectivity_penalty(p, f, PU_INDEX) == pytest.approx(
        compute_connectivity_penalty(p, b, PU_INDEX)
    )


def test_probmode1_premium_is_linear_in_x() -> None:
    p = three_pu(PROBMODE=1, PROBABILITYWEIGHTING=2.0).copy_with(
        probability=pd.DataFrame({"pu": [1, 2, 3], "probability": [0.5, 0.1, 0.0]}),
    )
    # 2 · (0.5·10·0.6 + 0.1·20·0.2 + 0)
    assert compute_probability_penalty(p, X, PU_INDEX) == pytest.approx(2.0 * (3.0 + 0.4))


@pytest.mark.parametrize("scale", [1.0, 1e4, 1e8])
def test_target_tolerance_fractional(scale: float) -> None:
    p = three_pu()
    r = 19.0 * scale
    tol = target_tol(r)
    p2 = p.copy_with(features=p.features.assign(target=[r, 10.0]))
    near = {1: r - 0.5 * tol, 2: 10.0}
    assert check_targets(p2, X, PU_INDEX, _achieved=near) == {1: True, 2: True}
    assert compute_feature_shortfalls(p2, X, PU_INDEX, _achieved=near) == {1: 0.0, 2: 0.0}
    far = {1: r - 2.0 * tol, 2: 10.0}
    assert check_targets(p2, X, PU_INDEX, _achieved=far)[1] is False
    assert compute_feature_shortfalls(p2, X, PU_INDEX, _achieved=far)[1] == pytest.approx(
        2.0 * tol
    )


def test_binary_comparison_stays_exact() -> None:
    p = three_pu()
    b = np.array([True, False, True])
    achieved = {1: 19.0 - 1e-12, 2: 10.0}
    assert check_targets(p, b, PU_INDEX, _achieved=achieved)[1] is False
    assert compute_feature_shortfalls(p, b, PU_INDEX, _achieved=achieved)[1] > 0


def test_misslevel_applies_to_fractional() -> None:
    p = three_pu(MISSLEVEL=0.8)  # f2 R = 8.0 <= 8.8
    assert check_targets(p, X, PU_INDEX) == {1: True, 2: True}


@pytest.mark.parametrize(
    ("column", "values", "word"),
    [
        ("target2", {"target2": [5.0, 0.0]}, "target2"),
        ("sep", {"sepnum": [2, 0], "sepdistance": [1.0, 0.0]}, "separation"),
        ("space", {"space_target": [0.5, 0.0]}, "space_target"),
    ],
)
def test_build_solution_rejects_fractional_combinations(
    column: str, values: dict[str, list[float]], word: str,
) -> None:
    p = three_pu()
    p = p.copy_with(features=p.features.assign(**values))
    with pytest.raises(ValueError, match=word):
        build_solution(p, X, 1.0)


def test_probmode3_rejects_fractional() -> None:
    p = three_pu(PROBMODE=3)
    with pytest.raises(ValueError, match="PROBMODE 3"):
        compute_probability_penalty(p, X, PU_INDEX)
    with pytest.raises(ValueError, match="PROBMODE 3"):
        compute_objective_terms(p, X, PU_INDEX, 1.0)
    with pytest.raises(ValueError, match="PROBMODE 3"):
        build_solution(p, X, 1.0)
    compute_probability_penalty(p, np.array([True, False, True]), PU_INDEX)  # binary: fine


def test_binarize_solution() -> None:
    p = three_pu()
    frac = build_solution(p, X, 1.0, metadata={"solver": "MIP"})
    b = binarize_solution(p, frac)
    assert b.selected.dtype == bool
    assert b.selected.tolist() == [True, False, True]
    assert b.metadata == {"solver": "MIP", "binarized_threshold": 0.5}
    assert b.cost == 40.0
    assert binarize_solution(p, frac, threshold=0.2).selected.tolist() == [True, True, True]
    assert binarize_solution(p, frac, threshold=1.0).selected.tolist() == [False, False, True]


def test_binarize_binary_returns_same_object() -> None:
    p = three_pu()
    sol = build_solution(p, np.array([True, False, True]), 1.0)
    assert binarize_solution(p, sol) is sol


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5, float("nan")])
def test_binarize_rejects_bad_threshold(threshold: float) -> None:
    p = three_pu()
    frac = build_solution(p, X, 1.0)
    with pytest.raises(ValueError, match="threshold"):
        binarize_solution(p, frac, threshold=threshold)


@pytest.mark.parametrize("target", [1e-6, 1e-9])
def test_empty_fractional_plan_misses_tiny_targets(target: float) -> None:
    """Review R1: a purely relative tolerance; nothing held never meets a positive target."""
    p = three_pu()
    p = p.copy_with(features=p.features.assign(target=[target, target / 10]))
    s = build_solution(p, np.zeros(3), 1.0)
    assert s.targets_met == {1: False, 2: False}
    assert s.shortfall == pytest.approx(1.1 * target)
    assert s.shortfall > 0


def test_zero_target_met_by_any_plan() -> None:
    p = three_pu()
    p = p.copy_with(features=p.features.assign(target=[0.0, 0.0]))
    assert build_solution(p, np.zeros(3), 1.0).targets_met == {1: True, 2: True}


@pytest.mark.parametrize(
    ("selected", "match"),
    [
        (np.array([0.5, np.nan, 1.0]), "entry 1 is nan"),
        (np.array([0.5, 2.0, 1.0]), "entry 1 is 2.0"),
        (np.array([-0.5, 0.0, 1.0]), "entry 0 is -0.5"),
        (np.array([1, 2, 0]), "only 0 and 1"),
        (np.array([True, False]), "one entry per planning unit"),
        (np.array(["a", "b", "c"]), "bool, integer 0/1 or float"),
    ],
)
def test_selection_contract_rejects(selected: np.ndarray, match: str) -> None:
    """Review R2: build_solution (and binarize_solution) enforce the selection contract."""
    with pytest.raises(ValueError, match=match):
        build_solution(three_pu(), selected, 1.0)


def test_selection_contract_coerces() -> None:
    p = three_pu()
    s32 = build_solution(p, X.astype(np.float32), 1.0)
    assert s32.selected.dtype == np.float64
    assert s32.cost == pytest.approx(40.0)
    ints = build_solution(p, np.array([1, 0, 1]), 1.0)  # fancy-indexed before Stage 4
    assert ints.selected.dtype == bool
    assert ints.cost == 40.0
    assert ints.objective == build_solution(p, np.array([True, False, True]), 1.0).objective
    assert build_solution(p, np.array([1, 0, 1], dtype=np.uint8), 1.0).cost == 40.0


def test_binarize_can_flip_a_met_target() -> None:
    """Review R8: binarising re-applies the exact binary rule. 0.7 + 0.1 holds 0.8 within
    the fractional tolerance but is short of 0.8 in floating point, so it flips to unmet."""
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 100.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [0.8], "spf": [1.0]})
    pvf = pd.DataFrame({"species": [1, 1, 1], "pu": [1, 2, 3], "amount": [0.7, 0.1, 5.0]})
    p = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pvf)
    frac = build_solution(p, np.array([1.0, 1.0, 0.0]), 0.0)
    assert frac.targets_met == {1: True}
    plan = binarize_solution(p, frac)
    assert plan.selected.tolist() == [True, True, False]
    assert plan.targets_met == {1: False}
    assert plan.shortfall > 0



@pytest.mark.parametrize("threshold", [np.float32(0.5), np.int64(1), np.float64(0.2)])
def test_binarize_accepts_numpy_scalar_threshold(threshold: object) -> None:
    p = three_pu()
    frac = build_solution(p, X, 1.0)
    assert binarize_solution(p, frac, threshold=threshold).selected.dtype == bool  # type: ignore[arg-type]
