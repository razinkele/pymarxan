"""Tests for the objective framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.objectives.base import Objective, ZonalObjective
from pymarxan.objectives.max_coverage import MaxCoverageObjective
from pymarxan.objectives.max_utility import MaxUtilityObjective
from pymarxan.objectives.min_shortfall import MinShortfallObjective
from pymarxan.objectives.minset import MinSetObjective


def _make_problem(params=None, **kw):
    from pymarxan.models.problem import ConservationProblem

    pu = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 20.0, 5.0, 15.0],
        "status": [0, 0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["sp_a", "sp_b"],
        "target": [8.0, 6.0],
        "spf": [1.0, 2.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 2, 2],
        "pu": [1, 2, 3, 2, 4],
        "amount": [5.0, 4.0, 3.0, 7.0, 6.0],
    })
    return ConservationProblem(
        planning_units=pu,
        features=features,
        pu_vs_features=puvspr,
        parameters=params or {"BLM": 0.0},
        **kw,
    )


def _effective_amounts(problem, pu_index):
    """Build effective amounts matrix from problem data."""
    n_pu = len(problem.planning_units)
    feat_ids = list(problem.features["id"])
    n_feat = len(feat_ids)
    matrix = np.zeros((n_pu, n_feat))
    for _, row in problem.pu_vs_features.iterrows():
        pid = int(row["pu"])
        fid = int(row["species"])
        idx = pu_index.get(pid)
        fidx = feat_ids.index(fid) if fid in feat_ids else None
        if idx is not None and fidx is not None:
            matrix[idx, fidx] = float(row["amount"])
    return matrix


class TestMinSetObjective:
    def test_name(self):
        obj = MinSetObjective()
        assert obj.name() == "MinSet"

    def test_uses_target_penalty(self):
        obj = MinSetObjective()
        assert obj.uses_target_penalty() is True

    def test_is_zonal_objective(self):
        obj = MinSetObjective()
        assert isinstance(obj, ZonalObjective)

    def test_compute_base_score_all_selected(self):
        problem = _make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        ea = _effective_amounts(problem, pu_index)
        selected = np.array([True, True, True, True])
        obj = MinSetObjective()
        score = obj.compute_base_score(problem, selected, ea, pu_index)
        assert score == pytest.approx(50.0)  # 10+20+5+15

    def test_compute_base_score_none_selected(self):
        problem = _make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        ea = _effective_amounts(problem, pu_index)
        selected = np.array([False, False, False, False])
        obj = MinSetObjective()
        score = obj.compute_base_score(problem, selected, ea, pu_index)
        assert score == pytest.approx(0.0)

    def test_compute_base_score_partial(self):
        problem = _make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        ea = _effective_amounts(problem, pu_index)
        selected = np.array([True, False, True, False])
        obj = MinSetObjective()
        score = obj.compute_base_score(problem, selected, ea, pu_index)
        assert score == pytest.approx(15.0)  # 10+5


class TestMaxCoverageObjective:
    def test_name(self):
        obj = MaxCoverageObjective()
        assert obj.name() == "MaxCoverage"

    def test_uses_target_penalty(self):
        obj = MaxCoverageObjective()
        assert obj.uses_target_penalty() is False

    def test_is_zonal_objective(self):
        obj = MaxCoverageObjective()
        assert isinstance(obj, ZonalObjective)

    def test_score_negated(self):
        """Score should be negative (lower-is-better for more coverage)."""
        problem = _make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        ea = _effective_amounts(problem, pu_index)
        selected = np.array([True, True, True, True])
        obj = MaxCoverageObjective()
        score = obj.compute_base_score(problem, selected, ea, pu_index)
        # sp_a: achieved=12, target=8, cov=min(12,8)=8
        # sp_b: achieved=13, target=6, cov=min(13,6)=6
        # total coverage=14, score=-14
        assert score == pytest.approx(-14.0)

    def test_no_selection_zero_coverage(self):
        problem = _make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        ea = _effective_amounts(problem, pu_index)
        selected = np.array([False, False, False, False])
        obj = MaxCoverageObjective()
        score = obj.compute_base_score(problem, selected, ea, pu_index)
        assert score == pytest.approx(0.0)  # -0 = 0

    def test_partial_coverage(self):
        problem = _make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        ea = _effective_amounts(problem, pu_index)
        # Select PU 1 only: sp_a gets 5 (target 8), sp_b gets 0
        selected = np.array([True, False, False, False])
        obj = MaxCoverageObjective()
        score = obj.compute_base_score(problem, selected, ea, pu_index)
        assert score == pytest.approx(-5.0)  # min(5,8)=5 + 0 = 5


class TestMaxUtilityObjective:
    def test_name(self):
        obj = MaxUtilityObjective()
        assert obj.name() == "MaxUtility"

    def test_uses_target_penalty(self):
        obj = MaxUtilityObjective()
        assert obj.uses_target_penalty() is False

    def test_not_zonal(self):
        obj = MaxUtilityObjective()
        assert not isinstance(obj, ZonalObjective)

    def test_score_negated(self):
        problem = _make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        ea = _effective_amounts(problem, pu_index)
        selected = np.array([True, True, True, True])
        obj = MaxUtilityObjective()
        score = obj.compute_base_score(problem, selected, ea, pu_index)
        # sp_a: achieved=12, target=8, prop=min(12/8,1)=1.0, weight=1.0
        # sp_b: achieved=13, target=6, prop=min(13/6,1)=1.0, weight=2.0
        # utility = 1*1 + 2*1 = 3, score = -3
        assert score == pytest.approx(-3.0)

    def test_partial_utility(self):
        problem = _make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        ea = _effective_amounts(problem, pu_index)
        # Select PU 1 only: sp_a gets 5/8=0.625, sp_b gets 0/6=0
        selected = np.array([True, False, False, False])
        obj = MaxUtilityObjective()
        score = obj.compute_base_score(problem, selected, ea, pu_index)
        assert score == pytest.approx(-(1.0 * 0.625 + 2.0 * 0.0))

    def test_zero_target_raises(self):
        problem = _make_problem()
        problem.features.loc[problem.features["id"] == 2, "target"] = 0.0
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        ea = _effective_amounts(problem, pu_index)
        selected = np.array([True, True, True, True])
        obj = MaxUtilityObjective()
        with pytest.raises(ValueError, match="target"):
            obj.compute_base_score(problem, selected, ea, pu_index)


class TestMinShortfallObjective:
    def test_name(self):
        obj = MinShortfallObjective()
        assert obj.name() == "MinShortfall"

    def test_uses_target_penalty(self):
        obj = MinShortfallObjective()
        assert obj.uses_target_penalty() is False

    def test_is_zonal_objective(self):
        obj = MinShortfallObjective()
        assert isinstance(obj, ZonalObjective)

    def test_all_targets_met(self):
        problem = _make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        ea = _effective_amounts(problem, pu_index)
        selected = np.array([True, True, True, True])
        obj = MinShortfallObjective()
        score = obj.compute_base_score(problem, selected, ea, pu_index)
        assert score == pytest.approx(0.0)

    def test_partial_shortfall(self):
        problem = _make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        ea = _effective_amounts(problem, pu_index)
        # Select PU 1 only: sp_a achieved=5, target=8, shortfall=3
        # sp_b achieved=0, target=6, shortfall=6
        selected = np.array([True, False, False, False])
        obj = MinShortfallObjective()
        score = obj.compute_base_score(problem, selected, ea, pu_index)
        assert score == pytest.approx(9.0)

    def test_none_selected_max_shortfall(self):
        problem = _make_problem()
        pu_index = {1: 0, 2: 1, 3: 2, 4: 3}
        ea = _effective_amounts(problem, pu_index)
        selected = np.array([False, False, False, False])
        obj = MinShortfallObjective()
        score = obj.compute_base_score(problem, selected, ea, pu_index)
        assert score == pytest.approx(14.0)  # 8 + 6


class TestObjectiveABC:
    def test_abc_not_instantiable(self):
        with pytest.raises(TypeError):
            Objective()  # type: ignore[abstract]

    def test_default_compute_delta_returns_none(self):
        obj = MinSetObjective()
        result = obj.compute_delta(0, np.array([True]), None, None)  # type: ignore[arg-type]
        assert result is None

    def test_default_init_state_returns_none(self):
        obj = MinSetObjective()
        result = obj.init_state(None, np.array([True]), None)  # type: ignore[arg-type]
        assert result is None
