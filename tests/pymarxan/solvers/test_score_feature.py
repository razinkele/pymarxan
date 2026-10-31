"""The chance-term core ``score_feature`` (spec §3.4, final review I-5).

One function scores a feature from ``(E, V, nonzero, target, ptarget, zero_var_rule)``.
Zero variance is ``nonzero == 0`` (the number of selected cells whose variance term is
> 0), never a float threshold on V. ``zero_var_rule`` is ``"marxan"`` (Marxan 3:
Z = 4, then the normal path) or ``"met"`` (penalty and shortfall 0).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.probability import (
    FeatureChance,
    compute_zscore_penalty,
    evaluate_solution_chance,
    score_feature,
    score_features,
)

SF4 = float(norm.sf(4.0))


class TestScoreFeature:
    def test_is_frozen_dataclass(self) -> None:
        fc = score_feature(8.0, 4.0, 2, 10.0, 0.95, "marxan")
        assert isinstance(fc, FeatureChance)
        assert [f.name for f in dataclasses.fields(fc)] == [
            "z", "p", "shortfall", "penalty",
        ]
        with pytest.raises(dataclasses.FrozenInstanceError):
            fc.z = 0.0  # type: ignore[misc]

    @pytest.mark.parametrize("rule", ["marxan", "met"])
    def test_normal_path(self, rule: str) -> None:
        """E = 8, V = 4, target 10 -> Z = 1; both rules agree away from zero variance."""
        fc = score_feature(8.0, 4.0, 3, 10.0, 0.95, rule)
        p = float(norm.sf(1.0))
        assert fc.z == pytest.approx(1.0)
        assert fc.p == pytest.approx(p, abs=1e-15)
        assert fc.shortfall == pytest.approx(0.95 - p, abs=1e-15)
        assert fc.penalty == pytest.approx((0.95 - p) / 0.95, abs=1e-15)

    def test_exceedance_has_no_penalty(self) -> None:
        fc = score_feature(14.0, 4.0, 1, 10.0, 0.9, "marxan")
        assert fc.z == pytest.approx(-2.0)
        assert fc.penalty == 0.0 and fc.shortfall == 0.0

    def test_zero_variance_marxan_rule(self) -> None:
        """Marxan 3: zero variance -> Z = 4, P = sf(4) ~ 3.17e-5, the step penalty follows."""
        fc = score_feature(20.0, 0.0, 0, 10.0, 0.95, "marxan")
        assert fc.z == 4.0
        assert fc.p == pytest.approx(SF4, abs=1e-18)
        assert fc.shortfall == pytest.approx(0.95 - SF4)
        assert fc.penalty == pytest.approx((0.95 - SF4) / 0.95)

    def test_zero_variance_met_rule(self) -> None:
        fc = score_feature(5.0, 0.0, 0, 10.0, 0.95, "met")
        assert fc.penalty == 0.0 and fc.shortfall == 0.0

    def test_zero_variance_marxan_rule_with_tiny_ptarget(self) -> None:
        """ptarget below sf(4): even the Z = 4 rule is met."""
        fc = score_feature(0.0, 0.0, 0, 10.0, 1e-5, "marxan")
        assert fc.penalty == 0.0 and fc.shortfall == 0.0

    def test_classification_uses_the_count_not_the_value(self) -> None:
        """A tiny positive variance with nonzero > 0 is NOT zero variance."""
        fc = score_feature(10.0, 1e-30, 1, 10.0, 0.95, "met")
        assert fc.z == 0.0
        assert fc.p == pytest.approx(0.5)
        assert fc.penalty == pytest.approx((0.95 - 0.5) / 0.95)

    @pytest.mark.parametrize("rule", ["marxan", "met"])
    @pytest.mark.parametrize("ptarget", [-1.0, 0.0])
    def test_disabled_ptarget(self, rule: str, ptarget: float) -> None:
        for args in ((5.0, 4.0, 1), (0.0, 0.0, 0)):
            fc = score_feature(*args, 10.0, ptarget, rule)
            assert fc.penalty == 0.0 and fc.shortfall == 0.0

    def test_unknown_rule_raises(self) -> None:
        with pytest.raises(ValueError, match="zero_var_rule"):
            score_feature(1.0, 1.0, 1, 1.0, 0.5, "trivially_met")


class TestVectorisedAgreesWithScalar:
    @pytest.mark.parametrize("rule", ["marxan", "met"])
    def test_batch(self, rule: str) -> None:
        rng = np.random.default_rng(3)
        n = 400
        E = rng.uniform(0, 20, n)
        V = rng.uniform(0, 9, n)
        N = rng.integers(0, 4, n)
        V[N == 0] = 0.0
        V[:5] = 1e-300  # tiny but counted
        N[:5] = 1
        t = rng.uniform(0, 20, n)
        pt = rng.choice([-1.0, 0.0, 0.2, 0.5, 0.95, 1.0, 1e-6], n)
        z, p, sf, pen = score_features(E, V, N, t, pt, rule)
        for k in range(n):
            fc = score_feature(float(E[k]), float(V[k]), int(N[k]), float(t[k]),
                               float(pt[k]), rule)
            assert (fc.z, fc.p, fc.shortfall, fc.penalty) == (
                float(z[k]), float(p[k]), float(sf[k]), float(pen[k]),
            )


class TestComputeZscorePenaltyZeroVariance:
    def test_real_z_of_four_is_scored(self) -> None:
        """A real Z = 4 (not a zero-variance feature) goes through the normal path."""
        p = compute_zscore_penalty({1: 4.0}, {1: 0.95}, weight=2.0)
        assert p == pytest.approx(2.0 * (0.95 - SF4) / 0.95)

    def test_zero_variance_feature_marxan_rule(self) -> None:
        p = compute_zscore_penalty(
            {1: 4.0, 2: 1.0}, {1: 0.95, 2: 0.95}, weight=1.0,
            zero_variance=frozenset({1}), zero_var_rule="marxan",
        )
        expected = (0.95 - SF4) / 0.95 + (0.95 - float(norm.sf(1.0))) / 0.95
        assert p == pytest.approx(expected)

    def test_zero_variance_feature_met_rule(self) -> None:
        p = compute_zscore_penalty(
            {1: 4.0, 2: 1.0}, {1: 0.95, 2: 0.95}, weight=1.0,
            zero_variance=frozenset({1}), zero_var_rule="met",
        )
        assert p == pytest.approx((0.95 - float(norm.sf(1.0))) / 0.95)

    def test_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            compute_zscore_penalty({1: 1.0}, {1: 0.5}, 1.0, frozenset())  # type: ignore[misc]


def _two_feature_problem() -> ConservationProblem:
    """Feature 1 uncertain (q = 0.7), feature 2 certain (q = 1: zero variance)."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "target": [5.0, 5.0], "spf": [1.0, 1.0],
        "ptarget": [0.9, 0.8],
    })
    pv = pd.DataFrame({
        "species": [1, 1, 2], "pu": [1, 2, 1], "amount": [4.0, 4.0, 10.0],
        "prob": [0.7, 0.7, 1.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 3.0},
    )


class TestEvaluateSolutionChance:
    def test_signature_has_term(self) -> None:
        sel = np.array([True, True])
        a = evaluate_solution_chance(_two_feature_problem(), sel)
        b = evaluate_solution_chance(_two_feature_problem(), sel, term="2d")
        assert a == b

    def test_unknown_term_raises(self) -> None:
        with pytest.raises(ValueError, match="term"):
            evaluate_solution_chance(_two_feature_problem(), np.array([True, True]), term="3d")

    def test_matches_score_feature(self) -> None:
        sel = np.array([True, True])
        shortfalls, penalty = evaluate_solution_chance(_two_feature_problem(), sel)
        e, v = 8.0 * 0.7, 2 * 16.0 * 0.7 * 0.3
        f1 = score_feature(e, v, 2, 5.0, 0.9, "met")
        assert shortfalls[1] == pytest.approx(f1.shortfall, abs=1e-15)
        assert shortfalls[2] == 0.0  # zero variance under today's rule
        assert penalty == pytest.approx(3.0 * f1.penalty, abs=1e-15)


class TestRoutedThroughScoreFeature:
    """The evaluator and the cache both score through the core and its rule."""

    def test_evaluator_honours_the_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pymarxan.solvers.utils as utils

        sel = np.array([True, True])
        _, met = evaluate_solution_chance(_two_feature_problem(), sel)
        monkeypatch.setattr(utils, "_zero_var_rule", lambda problem: "marxan")
        shortfalls, marxan = evaluate_solution_chance(_two_feature_problem(), sel)
        assert shortfalls[2] == pytest.approx(0.8 - SF4)
        assert marxan - met == pytest.approx(3.0 * (0.8 - SF4) / 0.8, abs=1e-12)
        # build_solution's objective goes through the same evaluator
        from pymarxan.solvers.utils import build_solution

        sol = build_solution(_two_feature_problem(), sel, 0.0)
        assert sol.prob_penalty == pytest.approx(marxan, abs=1e-15)

    def test_cache_honours_the_rule(self) -> None:
        from pymarxan.solvers.cache import ProblemCache

        problem = _two_feature_problem()
        cache = ProblemCache.from_problem(problem)
        marxan = dataclasses.replace(cache, zero_var_rule="marxan")
        sel = np.array([True, True])
        held = cache.compute_held(sel)
        diff = (
            marxan.compute_full_objective(sel, held, 0.0)
            - cache.compute_full_objective(sel, held, 0.0)
        )
        assert diff == pytest.approx(3.0 * (0.8 - SF4) / 0.8, abs=1e-12)
        _, pen = evaluate_solution_chance(problem, sel)
        assert cache.compute_full_objective(sel, held, 0.0) == pytest.approx(
            2.0 + pen, abs=1e-12,
        )
