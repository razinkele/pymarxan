"""ProblemCache under PROBMODE 1 / 2 (v0.45 §4): the MIP's and the evaluator's semantics.

Mode 1 adds ``γ·p·c`` per selected PU (``prob_cost``, read only when ``prob_active``); mode 2
reads held amounts from ``held_csr``, a copy of ``pu_feat_csr`` scaled by ``(1 − p)``. A
probability table with no ``PROBMODE`` is mode 1.
"""
from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.utils import build_solution

DATA = Path(__file__).parent.parent.parent / "data"
LINSPACE = pd.DataFrame({"pu": [1, 2, 3, 4, 5, 6], "probability": np.linspace(0.1, 0.6, 6)})


def _simple(
    probability: pd.DataFrame | None = None, mode: int | None = None, **params: Any,
) -> ConservationProblem:
    p = load_project(DATA / "simple")
    merged = {**p.parameters, **params}
    if mode is not None:
        merged["PROBMODE"] = mode
    return p.copy_with(probability=probability, parameters=merged)


def _edge(mode: int | None, gamma: float) -> ConservationProblem:
    """§6 fixture: PU 1 and 3 missing from the table, PU 2 listed twice (0.2 wins), PU 4
    p = 1, PU 3 cost −4, PU 1 locked in, MISSLEVEL 0.8, and a cost threshold (COSTTHRESH
    30, THRESHPEN1 5, THRESHPEN2 1) that must see base cost, not cost + premium."""
    p = _simple()
    pu = p.planning_units.copy()
    pu.loc[pu["id"] == 3, "cost"] = -4.0
    pu.loc[pu["id"] == 1, "status"] = 2
    prob = pd.DataFrame({"pu": [2, 2, 4, 5, 6], "probability": [0.9, 0.2, 1.0, 0.3, 0.45]})
    params = {
        **p.parameters, "PROBABILITYWEIGHTING": gamma, "MISSLEVEL": 0.8,
        "COSTTHRESH": 30.0, "THRESHPEN1": 5.0, "THRESHPEN2": 1.0,
    }
    if mode is not None:
        params["PROBMODE"] = mode
    return p.copy_with(planning_units=pu, probability=prob, parameters=params)


FIXTURES = [
    pytest.param(mode, gamma, id=f"mode{mode}-gamma{gamma:g}")
    for mode in (1, 2, None)
    for gamma in (2.0, -1.0, 0.0)
]


class TestDefaultModeAndFlags:
    def test_table_without_probmode_is_mode_1(self) -> None:
        c = ProblemCache.from_problem(_simple(LINSPACE))
        assert c.probmode == 1
        assert c.prob_active

    def test_no_table_without_probmode_is_mode_0(self) -> None:
        c = ProblemCache.from_problem(_simple())
        assert c.probmode == 0
        assert not c.prob_active

    def test_mode_1_prob_cost(self) -> None:
        p = _simple(LINSPACE, 1, PROBABILITYWEIGHTING=2.0)
        c = ProblemCache.from_problem(p)
        assert c.prob_active
        np.testing.assert_allclose(
            c.prob_cost, 2.0 * np.linspace(0.1, 0.6, 6) * p.planning_units["cost"].to_numpy(),
        )
        np.testing.assert_array_equal(c.costs, p.planning_units["cost"].to_numpy())

    @pytest.mark.parametrize("gamma", [0.0, -1.0])
    def test_mode_1_off_for_non_positive_weighting(self, gamma: float) -> None:
        c = ProblemCache.from_problem(_simple(LINSPACE, 1, PROBABILITYWEIGHTING=gamma))
        assert not c.prob_active

    def test_mode_2_has_no_premium(self) -> None:
        c = ProblemCache.from_problem(_simple(LINSPACE, 2))
        assert not c.prob_active

    @pytest.mark.parametrize("mode", [1, 2])
    def test_no_table_is_a_noop(self, mode: int) -> None:
        c = ProblemCache.from_problem(_simple(None, mode))
        assert not c.prob_active
        assert len(c.prob_cost) in (0, c.n_pu)
        assert not np.any(c.prob_cost)
        assert c.held_csr is c.pu_feat_csr

    def test_probmode_3_unchanged(self) -> None:
        c = ProblemCache.from_problem(_simple(LINSPACE, 3))
        assert c.probmode == 3
        assert not c.prob_active
        assert c.held_csr is c.pu_feat_csr


class TestHeldCsr:
    @pytest.mark.parametrize("mode", [0, 1, 3, None])
    def test_non_mode_2_shares_the_raw_matrix(self, mode: int | None) -> None:
        c = ProblemCache.from_problem(_simple(LINSPACE, mode))
        assert c.held_csr is c.pu_feat_csr

    def test_mode_2_structure_and_scaling(self) -> None:
        p = _edge(2, 1.0)  # PU 4 has p = 1: its row keeps explicit zeros
        c = ProblemCache.from_problem(p)
        raw, held = c.pu_feat_csr, c.held_csr
        assert held is not raw
        np.testing.assert_array_equal(held.indptr, raw.indptr)
        np.testing.assert_array_equal(held.indices, raw.indices)
        p_vec = np.array([0.0, 0.2, 0.0, 1.0, 0.3, 0.45])
        np.testing.assert_allclose(
            held.data, raw.data * np.repeat(1.0 - p_vec, np.diff(raw.indptr)),
        )
        row4 = slice(raw.indptr[3], raw.indptr[4])
        assert raw.indptr[4] > raw.indptr[3]
        assert np.all(held.data[row4] == 0.0)
        # Raw matrix untouched (clumping / separation read it).
        np.testing.assert_array_equal(
            raw.toarray(), ProblemCache.from_problem(_simple()).pu_feat_csr.toarray(),
        )

    def test_hand_built_cache_falls_back_to_raw(self) -> None:
        c = ProblemCache.from_problem(_simple(LINSPACE, 2))
        rebuilt = dataclasses.replace(c, held_csr=None)
        assert rebuilt.held_csr is rebuilt.pu_feat_csr

    def test_mode_2_compute_held_is_expected_amount(self) -> None:
        prob = pd.DataFrame({"pu": [4, 5, 6], "probability": [0.5, 0.5, 0.5]})
        p = _simple(prob, 2)
        c = ProblemCache.from_problem(p)
        sel = np.array([True, False, True, True, False, True])
        # f1: 10 + 5 + 12·0.5 + 10·0.5; f2: 5 + 8 + 3·0.5 + 7·0.5; f3: 10 + 4·0.5 + 8·0.5
        np.testing.assert_allclose(c.compute_held(sel), [26.0, 18.0, 16.0])


class TestDeltaAndFullObjective:
    @pytest.mark.parametrize(("mode", "gamma"), FIXTURES)
    def test_delta_equals_full_difference(self, mode: int | None, gamma: float) -> None:
        c = ProblemCache.from_problem(_edge(mode, gamma))
        rng = np.random.default_rng(0)
        for _ in range(10):
            base = rng.random(c.n_pu) < 0.5
            for i in range(c.n_pu):
                for adding in (True, False):  # both directions
                    s = base.copy()
                    s[i] = not adding
                    h = c.compute_held(s)
                    f0 = c.compute_full_objective(s, h, 1.0)
                    d = c.compute_delta_objective(i, s, h, float(c.costs[s].sum()), 1.0)
                    s2 = s.copy()
                    s2[i] = adding
                    f1 = c.compute_full_objective(s2, c.compute_held(s2), 1.0)
                    assert d == pytest.approx(f1 - f0, rel=1e-9, abs=1e-9)

    @pytest.mark.parametrize(("mode", "gamma"), FIXTURES)
    def test_full_equals_evaluator(self, mode: int | None, gamma: float) -> None:
        p = _edge(mode, gamma)
        c = ProblemCache.from_problem(p)
        rng = np.random.default_rng(1)
        for _ in range(20):
            s = rng.random(c.n_pu) < 0.5
            full = c.compute_full_objective(s, c.compute_held(s), 1.0)
            assert full == pytest.approx(build_solution(p, s, 1.0).objective, rel=1e-9, abs=1e-9)

    @pytest.mark.parametrize("mode", [1, 2, None])
    def test_full_equals_evaluator_every_plan(self, mode: int | None) -> None:
        p = _simple(LINSPACE, mode, PROBABILITYWEIGHTING=2.0)
        c = ProblemCache.from_problem(p)
        for bits in range(2 ** c.n_pu):
            s = np.array([(bits >> k) & 1 for k in range(c.n_pu)], dtype=bool)
            full = c.compute_full_objective(s, c.compute_held(s), 1.0)
            assert full == pytest.approx(build_solution(p, s, 1.0).objective, rel=1e-9, abs=1e-9)


_NAN_PROB = pd.DataFrame({"pu": [1, 2, 5], "probability": [0.3, np.nan, 0.4]})


class TestNonFinite:
    @pytest.mark.parametrize("mode", [1, 2, None])
    def test_cache_build_raises(self, mode: int | None) -> None:
        with pytest.raises(ValueError, match=r"non-finite .* \[2\]"):
            ProblemCache.from_problem(_simple(_NAN_PROB, mode))

    def test_mode_0_ignores_the_table(self) -> None:
        assert ProblemCache.from_problem(_simple(_NAN_PROB, 0)).probmode == 0


def _clumping(p: ConservationProblem) -> ConservationProblem:
    f = p.features.copy()
    f["target2"] = [10.0, 0.0, 0.0]
    f["clumptype"] = [0, 0, 0]
    return p.copy_with(features=f)


def _separation(p: ConservationProblem) -> ConservationProblem:
    pu = p.planning_units.copy()
    pu["xloc"] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    pu["yloc"] = 0.0
    f = p.features.copy()
    f["sepdistance"] = [0.0, 0.0, 2.0]
    f["sepnum"] = [1, 1, 2]
    return p.copy_with(planning_units=pu, features=f)


_RAW_WARNING = "PROBMODE 2 expected amounts are not applied to clumping/separation"


class TestProbmode2ClumpSepWarning:
    @pytest.mark.parametrize("combine", [_clumping, _separation])
    def test_warns_under_mode_2(self, combine: Any) -> None:
        with pytest.warns(UserWarning, match=_RAW_WARNING):
            ProblemCache.from_problem(combine(_simple(LINSPACE, 2)))

    @pytest.mark.parametrize("combine", [_clumping, _separation])
    @pytest.mark.parametrize(("probability", "mode"), [(LINSPACE, 1), (LINSPACE, 0), (None, 2)])
    def test_silent_outside_mode_2(
        self, combine: Any, probability: pd.DataFrame | None, mode: int,
    ) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=_RAW_WARNING)
            ProblemCache.from_problem(combine(_simple(probability, mode)))

    def test_silent_without_clump_or_sep(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=_RAW_WARNING)
            ProblemCache.from_problem(_simple(LINSPACE, 2))
