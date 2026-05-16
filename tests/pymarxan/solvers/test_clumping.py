"""Tests for the Marxan-faithful clumping math (Phase 19 Batch 2).

The math layer lives in ``pymarxan.solvers.clumping`` and is pure-functional —
no problem objects, no solver state. Inputs are NumPy arrays / CSR adjacency
indices; outputs are scalars or arrays.

Formulation per ``clumping.cpp::PartialPen4`` (Marxan v4):

    if occ >= target2:    return occ                  # full credit
    CLUMPTYPE 0:          return 0                    # binary
    CLUMPTYPE 1:          return occ / 2              # "nicer step"
    CLUMPTYPE 2:          return occ² / target2       # graduated / quadratic
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.clumping import (
    compute_baseline_penalty,
    compute_clump_penalty_from_scratch,
    compute_feature_components,
    evaluate_solution_clumping,
    partial_pen4,
)

# --- Task 4a: partial_pen4 (Marxan PartialPen4 line-by-line) ------------


class TestPartialPen4:
    """Marxan-faithful per-clump contribution. The CLUMPTYPE 1/2 distinguishing
    tests are the bedrock — these were the formulas the multi-agent review
    caught wrong in the v1 design."""

    def test_at_or_above_target_full_credit_all_clumptypes(self):
        for ct in (0, 1, 2):
            assert partial_pen4(occ=50.0, target2=50.0, clumptype=ct) == 50.0
            assert partial_pen4(occ=75.0, target2=50.0, clumptype=ct) == 75.0

    def test_clumptype0_subtarget_returns_zero(self):
        assert partial_pen4(occ=30.0, target2=50.0, clumptype=0) == 0.0
        assert partial_pen4(occ=0.0, target2=50.0, clumptype=0) == 0.0

    def test_clumptype1_subtarget_returns_half_amount(self):
        """Marxan source: ``return amount / 2.0`` — NOT min(amount, target2)."""
        assert partial_pen4(occ=30.0, target2=50.0, clumptype=1) == 15.0
        assert partial_pen4(occ=10.0, target2=50.0, clumptype=1) == 5.0
        # Distinguishing from the v1 wrong "capped" formula:
        assert partial_pen4(occ=30.0, target2=50.0, clumptype=1) != 30.0
        assert partial_pen4(occ=30.0, target2=50.0, clumptype=1) != min(30.0, 50.0)

    def test_clumptype2_subtarget_returns_quadratic(self):
        """Marxan source: ``return amount / target2 * amount`` (= amount²/target2),
        NOT linear ``amount/target2`` as the User Manual implies."""
        assert partial_pen4(occ=30.0, target2=50.0, clumptype=2) == 30.0 ** 2 / 50.0  # 18.0
        assert partial_pen4(occ=10.0, target2=50.0, clumptype=2) == 10.0 ** 2 / 50.0  # 2.0
        # Distinguishing from CLUMPTYPE 0 (same as v1 wrong CLUMPTYPE 2):
        assert partial_pen4(occ=30.0, target2=50.0, clumptype=2) != 0.0
        # Distinguishing from CLUMPTYPE 1 (half):
        assert partial_pen4(occ=30.0, target2=50.0, clumptype=2) != 15.0

    def test_target2_zero_returns_zero_for_safety(self):
        """target2=0 means clumping disabled. partial_pen4 must not divide
        by zero for CLUMPTYPE 2."""
        assert partial_pen4(occ=10.0, target2=0.0, clumptype=2) == 0.0

    def test_unknown_clumptype_returns_zero(self):
        """Defensive: Marxan's switch falls through to 0 on bad clumptype."""
        assert partial_pen4(occ=30.0, target2=50.0, clumptype=99) == 0.0


# --- Task 4b: compute_feature_components --------------------------------


class TestComputeFeatureComponents:
    """Connected components on (selected ∩ has_feature) subgraph.

    Uses the existing CSR adjacency from ProblemCache. PUs that don't have
    the feature (amount_ij == 0) do NOT bridge components (Marxan's
    rtnClumpSpecAtPu convention).
    """

    def _line_graph(self, n_pu: int):
        """Simple 1-2-3-...-n line graph CSR adjacency."""
        edges = [(i, i + 1) for i in range(n_pu - 1)] + [(i + 1, i) for i in range(n_pu - 1)]
        edges.sort()
        adj_start = np.zeros(n_pu + 1, dtype=np.int32)
        adj_indices = []
        adj_weights = []
        cur = 0
        for i in range(n_pu):
            adj_start[i] = cur
            for src, dst in edges:
                if src == i:
                    adj_indices.append(dst)
                    adj_weights.append(1.0)
                    cur += 1
        adj_start[n_pu] = cur
        return (
            np.array(adj_indices, dtype=np.int32),
            np.array(adj_weights, dtype=np.float64),
            adj_start,
        )

    def test_empty_selection_no_components(self):
        adj_i, adj_w, adj_s = self._line_graph(5)
        feat_amounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        selected = np.zeros(5, dtype=bool)
        comps = compute_feature_components(selected, feat_amounts, adj_i, adj_s)
        assert comps == []

    def test_all_selected_single_component(self):
        adj_i, adj_w, adj_s = self._line_graph(5)
        feat_amounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        selected = np.ones(5, dtype=bool)
        comps = compute_feature_components(selected, feat_amounts, adj_i, adj_s)
        assert len(comps) == 1
        assert set(comps[0].tolist()) == {0, 1, 2, 3, 4}

    def test_split_by_gap_two_components(self):
        """selected = [T, T, F, T, T] on the line 1-2-3-4-5 → two components."""
        adj_i, adj_w, adj_s = self._line_graph(5)
        feat_amounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        selected = np.array([True, True, False, True, True])
        comps = compute_feature_components(selected, feat_amounts, adj_i, adj_s)
        assert len(comps) == 2
        sets = [set(c.tolist()) for c in comps]
        assert {0, 1} in sets
        assert {3, 4} in sets

    def test_feature_absence_does_not_bridge(self):
        """PU 3 selected but amount[3]=0: doesn't bridge {0,1,2} and {4}."""
        adj_i, adj_w, adj_s = self._line_graph(5)
        feat_amounts = np.array([1.0, 1.0, 1.0, 0.0, 1.0])  # PU 3 lacks feature
        selected = np.ones(5, dtype=bool)
        comps = compute_feature_components(selected, feat_amounts, adj_i, adj_s)
        assert len(comps) == 2
        sets = [set(c.tolist()) for c in comps]
        assert {0, 1, 2} in sets
        assert {4} in sets

    def test_isolated_selected_pu_is_own_component(self):
        """A selected PU with no selected neighbours is a singleton clump."""
        # 5-PU line. Only PU 2 selected.
        adj_i, adj_w, adj_s = self._line_graph(5)
        feat_amounts = np.ones(5)
        selected = np.array([False, False, True, False, False])
        comps = compute_feature_components(selected, feat_amounts, adj_i, adj_s)
        assert len(comps) == 1
        assert set(comps[0].tolist()) == {2}


# --- Task 5: compute_baseline_penalty + compute_clump_penalty_from_scratch ---


def _make_simple_clump_problem() -> ConservationProblem:
    """A 5-PU line graph with one type-4 feature; used across the next two test classes."""
    pu = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "cost": [10.0, 10.0, 10.0, 10.0, 10.0],
        "status": [0, 0, 0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1],
        "name": ["sp"],
        "target": [40.0],   # need 40 total
        "spf": [1.0],
        "target2": [25.0],  # need a clump of ≥25
        "clumptype": [0],   # binary
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 1, 1],
        "pu": [1, 2, 3, 4, 5],
        "amount": [10.0, 10.0, 10.0, 10.0, 10.0],
    })
    boundary = pd.DataFrame({
        "id1": [1, 2, 3, 4],
        "id2": [2, 3, 4, 5],
        "boundary": [1.0, 1.0, 1.0, 1.0],
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        boundary=boundary,
    )


class TestComputeBaselinePenalty:
    """The Marxan-faithful baseline penalty: cost to meet target via a greedy
    cheapest-PU-first heuristic."""

    def test_baseline_matches_greedy_cost(self):
        problem = _make_simple_clump_problem()
        # Greedy: PUs cost 10 each, each provides 10 of feature; target=40
        # → need 4 PUs → baseline_penalty = 40.0.
        baseline = compute_baseline_penalty(problem)
        assert baseline.shape == (1,)
        assert baseline[0] == pytest.approx(40.0)

    def test_baseline_zero_when_target_zero(self):
        problem = _make_simple_clump_problem()
        features = problem.features.copy()
        features["target"] = 0.0
        problem = problem.copy_with(features=features)
        baseline = compute_baseline_penalty(problem)
        assert baseline[0] == 0.0

    def test_baseline_picks_cheapest_first(self):
        """When PUs have different cost/amount ratios, baseline_penalty
        uses the cheapest-per-amount ordering."""
        pu = pd.DataFrame({
            "id": [1, 2], "cost": [100.0, 5.0], "status": [0, 0],
        })
        features = pd.DataFrame({
            "id": [1], "name": ["sp"], "target": [10.0], "spf": [1.0],
            "target2": [5.0], "clumptype": [0],
        })
        puvspr = pd.DataFrame({
            "species": [1, 1], "pu": [1, 2], "amount": [10.0, 10.0],
        })
        problem = ConservationProblem(
            planning_units=pu, features=features, pu_vs_features=puvspr,
        )
        baseline = compute_baseline_penalty(problem)
        # PU 2 is cheaper (5/10 vs 100/10); greedy picks it first; covers full target.
        assert baseline[0] == pytest.approx(5.0)


class TestComputeClumpPenaltyFromScratch:
    """Reference impl combining components + PartialPen4 + fractional shortfall."""

    def test_full_selection_meets_target2_and_target(self):
        """All 5 PUs selected → one clump of 50 ≥ target2=25 → full 50 ≥ target=40 → penalty 0."""
        problem = _make_simple_clump_problem()
        selected = np.ones(5, dtype=bool)
        held_eff, penalty = compute_clump_penalty_from_scratch(problem, selected)
        assert held_eff[0] == pytest.approx(50.0)
        assert penalty == 0.0

    def test_no_selection_full_shortfall(self):
        """Nothing selected → held_eff = 0 → fractional shortfall = 1.0
        → penalty = baseline_penalty · SPF · 1.0 = 40."""
        problem = _make_simple_clump_problem()
        selected = np.zeros(5, dtype=bool)
        held_eff, penalty = compute_clump_penalty_from_scratch(problem, selected)
        assert held_eff[0] == 0.0
        assert penalty == pytest.approx(40.0)

    def test_two_subtarget_clumps_clumptype_0(self):
        """Selected {1,2} and {4,5}: two clumps of 20 each. target2=25, CLUMPTYPE=0
        → both contribute 0 → held_eff = 0 → full shortfall."""
        problem = _make_simple_clump_problem()
        selected = np.array([True, True, False, True, True])
        held_eff, penalty = compute_clump_penalty_from_scratch(problem, selected)
        assert held_eff[0] == 0.0
        # Fractional shortfall = (40 - 0)/40 = 1.0; baseline=40, SPF=1 → penalty=40
        assert penalty == pytest.approx(40.0)

    def test_two_subtarget_clumps_clumptype_1(self):
        """Same clumps but CLUMPTYPE=1: each contributes occ/2 = 10 → held_eff=20."""
        problem = _make_simple_clump_problem()
        features = problem.features.copy()
        features["clumptype"] = 1
        problem = problem.copy_with(features=features)
        selected = np.array([True, True, False, True, True])
        held_eff, penalty = compute_clump_penalty_from_scratch(problem, selected)
        assert held_eff[0] == pytest.approx(20.0)
        # Fractional shortfall = (40-20)/40 = 0.5; baseline=40, SPF=1 → penalty=20
        assert penalty == pytest.approx(20.0)

    def test_two_subtarget_clumps_clumptype_2(self):
        """Same clumps but CLUMPTYPE=2: each contributes occ²/target2 = 400/25 = 16
        → held_eff = 32."""
        problem = _make_simple_clump_problem()
        features = problem.features.copy()
        features["clumptype"] = 2
        problem = problem.copy_with(features=features)
        selected = np.array([True, True, False, True, True])
        held_eff, penalty = compute_clump_penalty_from_scratch(problem, selected)
        assert held_eff[0] == pytest.approx(32.0)
        # Fractional shortfall = (40-32)/40 = 0.2; baseline=40 → penalty=8
        assert penalty == pytest.approx(8.0)

    def test_target2_zero_collapses_to_non_clumping_path(self):
        """target2=0 disables clumping for the feature: held_eff reduces to
        the raw selected-amount sum, and the *clumping* penalty contribution
        is 0 (the deterministic-shortfall path elsewhere handles it).
        """
        problem = _make_simple_clump_problem()
        features = problem.features.copy()
        features["target2"] = 0.0
        problem = problem.copy_with(features=features)
        selected = np.array([True, True, False, True, False])
        held_eff, penalty = compute_clump_penalty_from_scratch(problem, selected)
        # held_eff for feature 1 = 10+10+10 = 30 (raw sum, no clumping adjustment)
        assert held_eff[0] == pytest.approx(30.0)
        # Non-type-4 feature → no clumping penalty contribution
        # (the deterministic feature penalty path handles this elsewhere)
        assert penalty == 0.0


# --- Task 6: evaluate_solution_clumping ---------------------------------


class TestEvaluateSolutionClumping:
    """Post-hoc evaluator for the MIP "drop" strategy and Solution attrs."""

    def test_returns_shortfalls_dict_and_total_penalty(self):
        problem = _make_simple_clump_problem()
        selected = np.array([True, True, False, True, True])  # two sub-target clumps
        shortfalls, penalty = evaluate_solution_clumping(problem, selected)
        assert isinstance(shortfalls, dict)
        assert set(shortfalls.keys()) == {1}
        # CLUMPTYPE 0 default → held_eff=0 → fractional 1.0 → shortfall 40 (raw, not normalised)
        assert shortfalls[1] == pytest.approx(40.0)
        assert penalty == pytest.approx(40.0)

    def test_no_type_4_features_returns_empty(self):
        """A problem with no target2 > 0 features → empty dict, 0 penalty."""
        problem = _make_simple_clump_problem()
        features = problem.features.copy()
        features["target2"] = 0.0
        problem = problem.copy_with(features=features)
        selected = np.ones(5, dtype=bool)
        shortfalls, penalty = evaluate_solution_clumping(problem, selected)
        assert shortfalls == {}
        assert penalty == 0.0

    def test_full_selection_zero_shortfall(self):
        problem = _make_simple_clump_problem()
        selected = np.ones(5, dtype=bool)
        shortfalls, penalty = evaluate_solution_clumping(problem, selected)
        assert shortfalls == {1: 0.0}
        assert penalty == 0.0
