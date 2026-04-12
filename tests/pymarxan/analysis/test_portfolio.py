"""Tests for portfolio analysis functions."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.analysis.portfolio import (
    best_solution,
    gap_filter,
    selection_frequency,
    solution_diversity,
    summary_statistics,
)
from pymarxan.solvers.base import Solution


def _sol(
    selected: list[bool],
    cost: float = 10.0,
    boundary: float = 1.0,
    objective: float = 11.0,
    shortfall: float = 0.0,
) -> Solution:
    return Solution(
        selected=np.array(selected),
        cost=cost,
        boundary=boundary,
        objective=objective,
        targets_met={1: True},
        shortfall=shortfall,
        metadata={},
    )


def _make_portfolio() -> list[Solution]:
    return [
        _sol([True, True, False, False], cost=20, boundary=1, objective=21),
        _sol([True, False, True, False], cost=25, boundary=2, objective=27),
        _sol([False, True, True, True], cost=35, boundary=3, objective=38),
        _sol([True, True, True, False], cost=30, boundary=2, objective=32),
    ]


# --- selection_frequency ---


class TestSelectionFrequency:
    def test_known_values(self):
        freq = selection_frequency(_make_portfolio())
        np.testing.assert_array_almost_equal(freq, [0.75, 0.75, 0.75, 0.25])

    def test_single_solution(self):
        freq = selection_frequency([_sol([True, False, True])])
        np.testing.assert_array_equal(freq, [1.0, 0.0, 1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            selection_frequency([])


# --- best_solution ---


class TestBestSolution:
    def test_picks_minimum_objective(self):
        portfolio = _make_portfolio()
        best = best_solution(portfolio)
        assert best.objective == 21.0

    def test_single_solution(self):
        s = _sol([True], objective=5.0)
        assert best_solution([s]) is s

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            best_solution([])


# --- gap_filter ---


class TestGapFilter:
    def test_default_gap(self):
        portfolio = _make_portfolio()
        # best=21, cutoff = 21 + 0.01*21 = 21.21
        kept = gap_filter(portfolio)
        assert len(kept) == 1
        assert kept[0].objective == 21.0

    def test_wide_gap_keeps_all(self):
        portfolio = _make_portfolio()
        kept = gap_filter(portfolio, gap_tolerance=1.0)
        assert len(kept) == len(portfolio)

    def test_negative_objectives(self):
        sols = [
            _sol([True], objective=-100.0),
            _sol([False], objective=-99.0),
            _sol([True], objective=-50.0),
        ]
        # best=-100, cutoff = -100 + 0.01*100 = -99.0
        kept = gap_filter(sols, gap_tolerance=0.01)
        assert len(kept) == 2
        objectives = sorted(s.objective for s in kept)
        assert objectives == [-100.0, -99.0]

    def test_small_objectives_use_floor_of_one(self):
        sols = [
            _sol([True], objective=0.0),
            _sol([False], objective=0.005),
            _sol([True], objective=0.02),
        ]
        # best=0, max(1, 0)=1, cutoff = 0 + 0.01*1 = 0.01
        kept = gap_filter(sols, gap_tolerance=0.01)
        assert len(kept) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            gap_filter([])

    def test_single_solution(self):
        s = _sol([True], objective=5.0)
        kept = gap_filter([s])
        assert len(kept) == 1


# --- solution_diversity ---


class TestSolutionDiversity:
    def test_identical_solutions(self):
        s = _sol([True, True, False])
        assert solution_diversity([s, s, s]) == 0.0

    def test_diverse_solutions(self):
        s1 = _sol([True, True, False, False])
        s2 = _sol([False, False, True, True])
        div = solution_diversity([s1, s2])
        assert div == pytest.approx(1.0)

    def test_partial_overlap(self):
        s1 = _sol([True, True, False])
        s2 = _sol([True, False, True])
        # intersection=1, union=3, dist=2/3
        div = solution_diversity([s1, s2])
        assert div == pytest.approx(2.0 / 3.0)

    def test_single_solution_zero(self):
        assert solution_diversity([_sol([True])]) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            solution_diversity([])


# --- summary_statistics ---


class TestSummaryStatistics:
    def test_keys_present(self):
        stats = summary_statistics(_make_portfolio())
        expected_keys = {
            "n_solutions",
            "cost_mean", "cost_std", "cost_min", "cost_max",
            "boundary_mean", "boundary_std", "boundary_min", "boundary_max",
            "objective_mean", "objective_std", "objective_min", "objective_max",
            "shortfall_mean", "shortfall_std", "shortfall_min", "shortfall_max",
            "mean_n_selected",
        }
        assert set(stats.keys()) == expected_keys

    def test_n_solutions(self):
        stats = summary_statistics(_make_portfolio())
        assert stats["n_solutions"] == 4.0

    def test_cost_stats(self):
        stats = summary_statistics(_make_portfolio())
        assert stats["cost_min"] == 20.0
        assert stats["cost_max"] == 35.0
        assert stats["cost_mean"] == pytest.approx(27.5)

    def test_mean_n_selected(self):
        stats = summary_statistics(_make_portfolio())
        # 2 + 2 + 3 + 3 = 10, mean = 2.5
        assert stats["mean_n_selected"] == pytest.approx(2.5)

    def test_single_solution(self):
        stats = summary_statistics([_sol([True, False], cost=5, objective=6)])
        assert stats["n_solutions"] == 1.0
        assert stats["cost_std"] == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            summary_statistics([])
