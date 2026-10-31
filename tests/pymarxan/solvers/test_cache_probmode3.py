"""ProblemCache PROB2D / PROBMODE 3 bookkeeping.

Phase 18 Batch 3 Task 7, reworked in v0.47.0 (spec §3.5). Verifies the cache:

- Precomputes the per-cell 2D chance arrays (``chance_cells["2d"]``) and
  feat_ptarget from the optional columns; no term (and no arrays) without
  a puvspr ``prob`` column.
- compute_full_objective / compute_delta_objective no longer include the
  chance term: it lives in the ``ChanceState`` sidecar, whose delta plus the
  cache's delta matches the difference of the full objectives (the bedrock
  SA contract).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.chance_state import ChanceState
from pymarxan.solvers.utils import build_solution


def _make_probmode3_problem(num_pus: int = 6, seed: int = 0) -> ConservationProblem:
    rng = np.random.default_rng(seed)
    pu = pd.DataFrame({
        "id": list(range(1, num_pus + 1)),
        "cost": list(rng.uniform(5.0, 20.0, num_pus)),
        "status": [0] * num_pus,
    })
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["f1", "f2"],
        "target": [12.0, 8.0],
        "spf": [1.0, 2.0],
        "ptarget": [0.95, 0.80],
    })
    rows = []
    for fid in (1, 2):
        for pu_id in range(1, num_pus + 1):
            rows.append({
                "species": fid,
                "pu": pu_id,
                "amount": float(rng.uniform(1.0, 5.0)),
                "prob": float(rng.uniform(0.0, 0.3)),
            })
    puvspr = pd.DataFrame(rows)
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 1.0},
    )


class TestProbmode3CacheFields:
    """The cache exposes the PROB2D state needed by all PROBMODE 3 paths."""

    def test_probmode_stored(self):
        problem = _make_probmode3_problem()
        cache = ProblemCache.from_problem(problem)
        assert cache.probmode == 3

    def test_prob_weight_stored(self):
        problem = _make_probmode3_problem()
        problem.parameters["PROBABILITYWEIGHTING"] = 2.5
        cache = ProblemCache.from_problem(problem)
        assert cache.prob_weight == 2.5

    def test_expected_cells_use_amount_times_prob(self):
        """Marxan PROB2D presence convention: expected = amount * prob
        (probability.c ComputeProbability2D), NOT amount * (1 - prob)
        (that's the PROBMODE 2 / Marxan 1D loss convention).

        Moved (v0.47.0) from the removed dense ``expected_matrix`` to the
        per-cell arrays ``chance_cells["2d"].e``.
        """
        problem = _make_probmode3_problem(num_pus=2, seed=42)
        cache = ProblemCache.from_problem(problem)
        cells = cache.chance_cells["2d"]
        for _, row in problem.pu_vs_features.iterrows():
            i = cache.pu_id_to_idx[int(row["pu"])]
            j = cache.feat_id_to_col[int(row["species"])]
            s, e = cells.indptr[i], cells.indptr[i + 1]
            k = s + int(np.flatnonzero(cells.indices[s:e] == j)[0])
            assert cells.e[k] == pytest.approx(float(row["amount"]) * float(row["prob"]))

    def test_variance_cells_are_bernoulli_form(self):
        """var = amount² · prob · (1 − prob); moved (v0.47.0) from ``var_matrix``."""
        problem = _make_probmode3_problem(num_pus=2, seed=42)
        cache = ProblemCache.from_problem(problem)
        cells = cache.chance_cells["2d"]
        for _, row in problem.pu_vs_features.iterrows():
            i = cache.pu_id_to_idx[int(row["pu"])]
            j = cache.feat_id_to_col[int(row["species"])]
            s, e = cells.indptr[i], cells.indptr[i + 1]
            k = s + int(np.flatnonzero(cells.indices[s:e] == j)[0])
            amount, prob = float(row["amount"]), float(row["prob"])
            assert cells.v[k] == pytest.approx(amount ** 2 * prob * (1.0 - prob))
            assert cells.pos[k] == int(cells.v[k] > 0)

    def test_dense_chance_matrices_are_gone(self):
        """The dense (n_pu, n_feat) expected / variance matrices were removed (v0.47.0)."""
        cache = ProblemCache.from_problem(_make_probmode3_problem())
        assert not hasattr(cache, "expected_matrix")
        assert not hasattr(cache, "var_matrix")
        assert "pu_feat_matrix" not in cache.__dict__  # mode 3 no longer densifies

    def test_feat_ptarget_stored(self):
        problem = _make_probmode3_problem()
        cache = ProblemCache.from_problem(problem)
        np.testing.assert_array_almost_equal(
            cache.feat_ptarget, np.array([0.95, 0.80]),
        )

    def test_missing_prob_column_turns_the_term_off(self):
        """No `prob` column → the 2D chance term is off ("term without data", spec §3.1).

        Rewritten (v0.47.0): this used to assert that the dense matrices collapsed to the
        deterministic case (variance 0). The term is now inactive instead: no per-cell
        arrays, no ChanceState, chance penalty 0 and no chance fields on the Solution.
        """
        problem = _make_probmode3_problem()
        problem = problem.copy_with(
            pu_vs_features=problem.pu_vs_features.drop(columns=["prob"]),
        )
        cache = ProblemCache.from_problem(problem)
        assert cache.chance_terms == frozenset()
        assert cache.chance_cells == {}
        sel = np.ones(cache.n_pu, dtype=bool)
        assert ChanceState.from_selection(cache, sel).penalty_total() == 0.0
        sol = build_solution(problem, sel, 0.0)
        assert sol.prob_penalty is None and sol.prob_shortfalls is None
        det = build_solution(problem.copy_with(parameters={"PROBMODE": 0}), sel, 0.0)
        assert sol.objective == det.objective


class TestProbmode3FullObjective:
    """compute_full_objective adds the PROBMODE 3 penalty on top of
    the deterministic objective."""

    def test_full_objective_plus_chance_state_is_the_evaluator_objective(self):
        """The cache's full objective excludes the chance term; adding
        ``ChanceState.penalty_total()`` gives the evaluator's (reported) objective.

        Rewritten (v0.47.0): the old test only asserted ``obj_3 >= obj_0``, which
        stays true (vacuously) once the term moves out of the cache (spec §3.6).
        """
        problem = _make_probmode3_problem(num_pus=4, seed=7)
        cache = ProblemCache.from_problem(problem)
        problem_det = copy.deepcopy(problem)
        problem_det.parameters["PROBMODE"] = 0
        cache_det = ProblemCache.from_problem(problem_det)
        rng = np.random.default_rng(7)
        for _ in range(8):
            selected = rng.random(cache.n_pu) > 0.5
            held = cache.compute_held(selected)
            obj_3 = cache.compute_full_objective(selected, held, blm=0.0)
            assert obj_3 == cache_det.compute_full_objective(selected, held, blm=0.0)
            chance = ChanceState.from_selection(cache, selected).penalty_total()
            reported = build_solution(problem, selected, 0.0)
            assert reported.prob_penalty is not None
            assert chance == pytest.approx(reported.prob_penalty, abs=1e-12)
            assert obj_3 + chance == pytest.approx(reported.objective, abs=1e-9)

    def test_full_objective_disabled_ptarget_no_penalty(self):
        """When every feature has ptarget=-1, PROBMODE 3 contributes 0."""
        problem = _make_probmode3_problem(num_pus=4, seed=7)
        features = problem.features.copy()
        features["ptarget"] = -1.0
        problem = problem.copy_with(features=features)
        cache = ProblemCache.from_problem(problem)
        selected = np.array([True, True, False, False])
        held = cache.compute_held(selected)

        problem_det = copy.deepcopy(problem)
        problem_det.parameters["PROBMODE"] = 0
        cache_det = ProblemCache.from_problem(problem_det)

        obj_3 = cache.compute_full_objective(selected, held, blm=0.0)
        obj_0 = cache_det.compute_full_objective(selected, held, blm=0.0)
        assert obj_3 == pytest.approx(obj_0, abs=1e-9)


class TestProbmode3Delta:
    """The cache's delta plus ``ChanceState.delta_penalty`` matches the full-recompute
    difference — the bedrock SA inner-loop contract (moved to ChanceState in v0.47.0)."""

    def test_delta_matches_full_under_probmode3(self):
        problem = _make_probmode3_problem(num_pus=8, seed=42)
        cache = ProblemCache.from_problem(problem)
        rng = np.random.default_rng(123)
        selected = rng.random(cache.n_pu) > 0.5
        if not selected.any():
            selected[0] = True
        held = cache.compute_held(selected)
        total_cost = float(np.sum(cache.costs[selected]))
        blm = 1.5
        chance = ChanceState.from_selection(cache, selected)

        for _ in range(50):
            flip_idx = int(rng.integers(cache.n_pu))
            adding = not selected[flip_idx]
            full_before = (
                cache.compute_full_objective(selected, held, blm)
                + ChanceState.from_selection(cache, selected).penalty_total()
            )
            delta = cache.compute_delta_objective(
                flip_idx, selected, held, total_cost, blm,
            ) + chance.delta_penalty(cache, flip_idx, adding)
            # Apply the flip
            sign = 1.0 if adding else -1.0
            cache.apply_flip_to_held(held, flip_idx, sign)
            total_cost += sign * cache.costs[flip_idx]
            selected = selected.copy()
            selected[flip_idx] = adding
            chance.apply_flip(cache, flip_idx, adding)
            full_after = (
                cache.compute_full_objective(selected, held, blm)
                + ChanceState.from_selection(cache, selected).penalty_total()
            )
            assert delta == pytest.approx(full_after - full_before, abs=1e-9)
