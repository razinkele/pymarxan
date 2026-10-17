"""PROBMODE 1 / 2 in the evaluator (``solvers/utils.py``): the MIP's semantics (v0.45 §3).

PROBMODE 2 judges held amounts as expected amounts ``Σ amount·(1 − p)`` (binary and
fractional); the PROBMODE 1 risk premium is off for ``PROBABILITYWEIGHTING <= 0``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.utils import (
    binarize_solution,
    build_solution,
    compute_held,
    compute_probability_penalty,
)

DATA = Path(__file__).parent.parent.parent / "data"


def _simple(probability: pd.DataFrame | None = None, **params: Any) -> ConservationProblem:
    p = load_project(DATA / "simple")
    return p.copy_with(probability=probability, parameters={**p.parameters, **params})


def _half_456(**params: Any) -> ConservationProblem:
    """simple, p = 0.5 on PUs 4–6, PROBMODE 2 (§6 'Evaluator PROBMODE 2')."""
    prob = pd.DataFrame({"pu": [4, 5, 6], "probability": [0.5, 0.5, 0.5]})
    return _simple(prob, PROBMODE=2, **params)


def _mip(problem: ConservationProblem, **kw: Any) -> Solution:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        sols = MIPSolver(mip_backend="cbc", **kw).solve(problem, SolverConfig(num_solutions=1))
    assert len(sols) == 1
    return sols[0]


def _expected_held(problem: ConservationProblem, selected: np.ndarray) -> dict[int, float]:
    """Hand oracle: Σ amount·x·(1 − p) per feature (missing PU → p = 0)."""
    p_of = {int(u): float(v) for u, v in zip(
        problem.probability["pu"], problem.probability["probability"],
    )} if problem.probability is not None else {}
    x_of = {int(u): float(x) for u, x in zip(problem.planning_units["id"], selected)}
    out = {int(f): 0.0 for f in problem.features["id"]}
    for pu, sp, amt in zip(
        problem.pu_vs_features["pu"], problem.pu_vs_features["species"],
        problem.pu_vs_features["amount"],
    ):
        out[int(sp)] += float(amt) * x_of[int(pu)] * (1.0 - p_of.get(int(pu), 0.0))
    return out


class TestEvaluatorProbmode2:
    def test_max_features_met_count_matches_mip(self) -> None:
        """(i) The reported met count is the MIP's own count (expected amounts)."""
        p = _half_456(COSTBUDGET=33.0)
        s = _mip(p, objective="max_features")
        assert s.metadata["mip_objective_value"] == pytest.approx(1.0)
        assert sum(s.targets_met.values()) == 1

    def test_binarized_plan_judged_on_expected_amounts(self) -> None:
        """(ii) Feature 2 holds 22 raw but 18.5 expected against a target of 20."""
        p = _half_456()
        frac = build_solution(p, np.array([0.75, 1, 0.5, 0, 0, 1.0]), 1.0)
        b = binarize_solution(p, frac, 0.75)
        assert b.targets_met == {1: True, 2: False, 3: False}

    def test_min_set_plan_meets_every_target(self) -> None:
        """(iii) Regression pin: the MIP enforces every target on expected amounts."""
        s = _mip(_half_456())
        assert all(s.targets_met.values())

    def test_proportion_plan_meets_every_target(self) -> None:
        """(iii) Regression pin: proportion decisions, judged fraction-linearly."""
        s = _mip(_half_456(), decisions="proportion")
        assert s.selected.dtype.kind == "f"
        assert all(s.targets_met.values())

    @pytest.mark.parametrize(
        "selected",
        [
            np.array([True, False, True, True, False, True]),
            np.ones(6, dtype=bool),
            np.array([0.75, 1.0, 0.5, 0.0, 0.25, 1.0]),
        ],
    )
    def test_compute_held_is_expected_amount(self, selected: np.ndarray) -> None:
        prob = pd.DataFrame({"pu": [1, 2, 4, 5, 6], "probability": [0.1, 0.4, 0.25, 0.05, 1.0]})
        p = _simple(prob, PROBMODE=2)
        got = compute_held(p, selected)
        want = _expected_held(p, selected)
        assert got.keys() == want.keys()
        for fid in want:
            assert got[fid] == pytest.approx(want[fid], rel=1e-12, abs=1e-12)

    def test_shortfall_and_penalty_use_expected_amounts(self) -> None:
        p = _half_456()
        sel = np.array([True, True, False, False, False, True])
        s = build_solution(p, sel, 0.0)
        held = _expected_held(p, sel)  # f1 = 10+15+5 = 30, f2 = 5+10+3.5, f3 = 6+4
        targets = dict(zip(p.features["id"].astype(int), p.features["target"].astype(float)))
        want = sum(max(0.0, targets[f] - held[f]) for f in targets)
        assert s.shortfall == pytest.approx(want)
        assert s.penalty == pytest.approx(want)  # spf 1 throughout

    def test_no_table_is_a_noop(self) -> None:
        sel = np.array([True, False, True, True, False, True])
        raw = compute_held(_simple(), sel)
        for mode in (1, 2):
            p = _simple(PROBMODE=mode)
            assert compute_held(p, sel) == raw
            assert build_solution(p, sel, 1.0).objective == pytest.approx(
                build_solution(_simple(), sel, 1.0).objective
            )

    @pytest.mark.parametrize("mode", [0, 1, 3])
    def test_other_modes_keep_raw_amounts(self, mode: int) -> None:
        prob = pd.DataFrame({"pu": [4, 5, 6], "probability": [0.5, 0.5, 0.5]})
        sel = np.array([True, False, True, True, False, True])
        assert compute_held(_simple(prob, PROBMODE=mode), sel) == compute_held(_simple(), sel)


class TestProbmode1Weighting:
    @pytest.mark.parametrize("gamma", [0.0, -1.0])
    def test_premium_off_for_non_positive_weighting(self, gamma: float) -> None:
        prob = pd.DataFrame({"pu": [1, 2, 3, 4, 5, 6], "probability": np.linspace(0.1, 0.6, 6)})
        p = _simple(prob, PROBMODE=1, PROBABILITYWEIGHTING=gamma)
        sel = np.array([False, True, False, True, False, True])
        pu_index = {int(u): i for i, u in enumerate(p.planning_units["id"])}
        assert compute_probability_penalty(p, sel, pu_index) == 0.0
        assert build_solution(p, sel, 1.0).objective == pytest.approx(
            build_solution(_simple(), sel, 1.0).objective
        )

    def test_premium_on_for_positive_weighting(self) -> None:
        prob = pd.DataFrame({"pu": [2, 4], "probability": [0.5, 0.25]})
        p = _simple(prob, PROBMODE=1, PROBABILITYWEIGHTING=2.0)
        sel = np.array([False, True, False, True, False, True])
        pu_index = {int(u): i for i, u in enumerate(p.planning_units["id"])}
        assert compute_probability_penalty(p, sel, pu_index) == pytest.approx(
            2.0 * (0.5 * 15.0 + 0.25 * 12.0)
        )


class TestValidateDuplicateProbabilityRows:
    def test_duplicate_pu_reported(self) -> None:
        prob = pd.DataFrame({"pu": [2, 4, 2, 6, 6], "probability": [0.1, 0.2, 0.3, 0.4, 0.5]})
        errors = _simple(prob).validate()
        assert "probability lists planning unit IDs more than once: [2, 6]" in errors

    def test_unique_table_has_no_duplicate_error(self) -> None:
        prob = pd.DataFrame({"pu": [2, 4, 6], "probability": [0.1, 0.2, 0.3]})
        assert not any("more than once" in e for e in _simple(prob).validate())


class TestComputeHeldSelectionContract:
    def test_int_zero_one_selection_reads_as_bool(self) -> None:
        p = _simple()
        sel = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        assert compute_held(p, sel) == {1: 37.0, 2: 20.0, 3: 18.0}
        assert compute_held(p, sel) == compute_held(p, sel.astype(bool))

    def test_other_int_values_raise(self) -> None:
        with pytest.raises(ValueError, match="0 and 1"):
            compute_held(_simple(), np.array([0, 2, 0, 1, 0, 1]))


class TestCostThresholdSeesBaseCost:
    def test_cost_and_threshold_exclude_premium(self) -> None:
        prob = pd.DataFrame({"pu": [2, 4, 6], "probability": [0.5, 0.25, 0.5]})
        params = {"COSTTHRESH": 30.0, "THRESHPEN1": 5.0, "THRESHPEN2": 1.0}
        p = _simple(prob, PROBMODE=1, PROBABILITYWEIGHTING=2.0, **params)
        sel = np.array([False, True, False, True, False, True])  # cost 35
        s = build_solution(p, sel, 0.0)
        assert s.cost == 35.0
        premium = 2.0 * (0.5 * 15.0 + 0.25 * 12.0 + 0.5 * 8.0)  # 29.0
        threshold = 5.0 + 1.0 * (35.0 - 30.0)  # on base cost, not 35 + premium
        assert s.objective == pytest.approx(35.0 + premium + threshold)  # targets met, BLM 0


class TestProbabilityHelpers:
    """The one copy of the probability semantics (shared by every solver path)."""

    def test_probmode_default(self) -> None:
        from pymarxan.solvers.utils import _probmode, _probmode2_active

        prob = pd.DataFrame({"pu": [1], "probability": [0.5]})
        assert _probmode(_simple()) == 0
        assert _probmode(_simple(prob)) == 1
        assert _probmode(_simple(prob, PROBMODE=2)) == 2
        assert _probmode2_active(_simple(prob, PROBMODE=2))
        assert not _probmode2_active(_simple(None, PROBMODE=2))

    def test_map_and_vector(self) -> None:
        from pymarxan.solvers.utils import _probability_map, _probability_vector

        prob = pd.DataFrame({"pu": [2, 99, 4, 2], "probability": [0.9, 0.3, 1.0, 0.2]})
        p = _simple(prob)
        assert _probability_map(p) == {2: 0.2, 99: 0.3, 4: 1.0}  # last row wins
        idx = {int(u): i for i, u in enumerate(p.planning_units["id"])}
        np.testing.assert_array_equal(
            _probability_vector(p, idx, 6), [0.0, 0.2, 0.0, 1.0, 0.0, 0.0],
        )


_NAN_PROB = pd.DataFrame({"pu": [1, 2, 5], "probability": [0.3, np.nan, 0.4]})


class TestNonFiniteProbabilities:
    def test_validate_reports_nan(self) -> None:
        errors = _simple(_NAN_PROB).validate()
        assert (
            "probability has non-finite values (NaN or infinite) for planning unit IDs: [2]"
            in errors
        )
        assert "probability values must be in range [0, 1]" not in errors

    @pytest.mark.parametrize("mode", [1, 2, None])
    def test_evaluator_raises(self, mode: int | None) -> None:
        params = {} if mode is None else {"PROBMODE": mode}
        p = _simple(_NAN_PROB, **params)
        with pytest.raises(ValueError, match=r"non-finite .* \[2\]"):
            build_solution(p, np.array([False, True, False, True, False, True]), 1.0)

    @pytest.mark.parametrize("mode", [1, 2])
    def test_mip_raises(self, mode: int) -> None:
        with pytest.raises(ValueError, match=r"non-finite .* \[2\]"):
            _mip(_simple(_NAN_PROB, PROBMODE=mode))

    @pytest.mark.parametrize("mode", [1, 2])
    def test_greedy_raises(self, mode: int) -> None:
        p = _simple(_NAN_PROB, PROBMODE=mode)
        with pytest.raises(ValueError, match=r"non-finite .* \[2\]"):
            HeuristicSolver().solve(p, SolverConfig(num_solutions=1, seed=1))

    def test_mode_0_ignores_the_table(self) -> None:
        p = _simple(_NAN_PROB, PROBMODE=0)
        build_solution(p, np.array([False, True, False, True, False, True]), 1.0)

    @pytest.mark.parametrize("mode", [1, None])
    @pytest.mark.parametrize("gamma", [0.0, -1.0])
    def test_premium_off_ignores_the_table(self, mode: int | None, gamma: float) -> None:
        """Mode 1 with weighting <= 0 reads no probabilities, in every path (greedy too)."""
        params: dict[str, Any] = {"PROBABILITYWEIGHTING": gamma}
        if mode is not None:
            params["PROBMODE"] = mode
        p = _simple(_NAN_PROB, **params)
        build_solution(p, np.array([False, True, False, True, False, True]), 1.0)
        _mip(p)
        HeuristicSolver().solve(p, SolverConfig(num_solutions=1, seed=1))


def _greedy_plan(problem: ConservationProblem, heurtype: int) -> list[int]:
    (s,) = HeuristicSolver(heurtype=heurtype).solve(
        problem, SolverConfig(num_solutions=1, seed=1),
    )
    return [int(v) for v in s.selected]


class TestGreedySharedSemantics:
    """Greedy's half of the shared-helper refactor (round-2 R2-2)."""

    _P = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    @pytest.mark.parametrize(("mode", "heurtype"), [(1, 1), (2, 0)])
    def test_duplicate_pu_last_row_wins(self, mode: int, heurtype: int) -> None:
        low = pd.DataFrame({"pu": [1, 2, 3, 4, 5, 6], "probability": self._P})
        high = low.assign(probability=[0.1, 0.9, 0.3, 0.4, 0.5, 0.6])
        dup = pd.concat(
            [pd.DataFrame({"pu": [2], "probability": [0.9]}), low], ignore_index=True,
        )
        params: dict[str, Any] = {"PROBMODE": mode, "PROBABILITYWEIGHTING": 5.0}
        plan = _greedy_plan(_simple(dup, **params), heurtype)
        assert plan == _greedy_plan(_simple(low, **params), heurtype)  # last row (0.2)
        assert plan != _greedy_plan(_simple(high, **params), heurtype)  # not the first (0.9)

    def test_table_without_probmode_is_mode_1(self) -> None:
        prob = pd.DataFrame({"pu": [1, 6], "probability": [0.9, 0.9]})
        default = _greedy_plan(_simple(prob, PROBABILITYWEIGHTING=5.0), 1)
        assert default == _greedy_plan(_simple(prob, PROBMODE=1, PROBABILITYWEIGHTING=5.0), 1)
        assert default != _greedy_plan(_simple(prob, PROBMODE=0, PROBABILITYWEIGHTING=5.0), 1)
