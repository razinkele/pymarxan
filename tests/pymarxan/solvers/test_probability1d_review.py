"""Probability 1D review pins (v0.48.0 core review M-2a, M-2b, L-6).

- M-2a: with both 1D sources present and disagreeing, the evaluator and ``ChanceState``
  use the pu.dat ``prob`` column, numerically (spec §3.3), not prob.dat.
- M-2b: ``merge_chance_cells`` offsets each term's row-order index ``r_indptr`` by the
  earlier terms' puvspr **rows**, not their cells: they differ with duplicate
  ``(pu, species)`` rows, and the merged state's exact-rule row sums under PROBMODE 5 +
  PROBZEROVAR 2 read the wrong rows otherwise.
- L-6: ``ptarget1d == 0`` does not select PROBMODE 4 (0 is disabled), and a NaN
  ``ptarget1d`` in a code-built problem is −1 for the evaluator and the cache alike.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem, _resolved_probmode
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.chance_state import ChanceState, _row_sum
from pymarxan.solvers.probability import _term_moments, evaluate_solution_chance
from pymarxan.solvers.utils import _chance_terms

# --- M-2a: pu.dat prob wins over prob.dat ------------------------------------------------

PU_PROB = [0.1, 0.5, 0.2]
DAT_PROB = [0.6, 0.0, 0.9]
AMOUNTS = [4.0, 3.0, 2.0]


def _sources_problem(*, pu_col: bool, table: bool) -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": 0})
    if pu_col:
        pu["prob"] = PU_PROB
    return ConservationProblem(
        planning_units=pu,
        features=pd.DataFrame({"id": [1], "name": ["a"], "target": [6.0], "spf": [1.0],
                               "ptarget1d": [0.9]}),
        pu_vs_features=pd.DataFrame({"species": [1, 1, 1], "pu": [1, 2, 3],
                                     "amount": AMOUNTS}),
        probability=(
            pd.DataFrame({"pu": [1, 2, 3], "probability": DAT_PROB}) if table else None
        ),
        parameters={"PROBMODE": 4},
    )


def test_pu_dat_prob_wins_over_prob_dat_in_evaluator_and_state():
    both = _sources_problem(pu_col=True, table=True)
    pu_only = _sources_problem(pu_col=True, table=False)
    dat_only = _sources_problem(pu_col=False, table=True)
    sel = np.ones(3, dtype=bool)
    ids = np.array([1])

    E, V, _ = _term_moments(both, sel, "1d", ids)
    # Hand-computed from pu.dat: E = Σ a(1 − p), V = Σ a²·p(1 − p).
    assert E[0] == pytest.approx(4 * 0.9 + 3 * 0.5 + 2 * 0.8)  # 6.7
    assert V[0] == pytest.approx(16 * 0.09 + 9 * 0.25 + 4 * 0.16)  # 4.33
    E_dat, _, _ = _term_moments(dat_only, sel, "1d", ids)
    assert E_dat[0] == pytest.approx(4 * 0.4 + 3 * 1.0 + 2 * 0.1)  # 4.8: really differs

    pen_both = evaluate_solution_chance(both, sel, "1d")[1]
    pen_pu = evaluate_solution_chance(pu_only, sel, "1d")[1]
    pen_dat = evaluate_solution_chance(dat_only, sel, "1d")[1]
    assert pen_both == pen_pu
    assert pen_both != pytest.approx(pen_dat)

    state = ChanceState.from_selection(ProblemCache.from_problem(both), sel)
    np.testing.assert_allclose(state.terms["1d"].E, E, rtol=1e-12)
    assert state.penalty_total() == pytest.approx(pen_pu, rel=1e-12)
    assert state.penalty_total() != pytest.approx(pen_dat)


# --- M-2b: merged row-order offsets -------------------------------------------------------


def _dup_rows_mode5() -> ConservationProblem:
    """PROBMODE 5, PROBZEROVAR 2, a duplicated (pu 1, species 1) row in both terms.

    PU 2 is the only cell with variance in either term (p = 0.5, q = 0.5), so flipping it
    out snaps both terms of feature 1 to zero variance and the exact rule decides from the
    row-order sum over the merged cells. Feature 2 has its own duplicate row, so term 0
    has two more rows (7) than cells (5): offsetting term 1's ``r_indptr`` by cells ends
    feature 1's range two rows early and drops its last rows (PU 3, the duplicate PU 1),
    so the 2D sum over {1, 3, 4} reads 2 instead of 4 and misses the target 3.5.
    """
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [1.0, 1.0, 1.0, 1.0], "status": 0,
                       "prob": [0.0, 0.5, 0.0, 0.0]})
    pv = pd.DataFrame({
        "species": [1, 1, 1, 1, 2, 2, 2],
        "pu": [2, 1, 3, 1, 4, 4, 1],
        "amount": [3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        "prob": [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })
    feat = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [3.5, 2.0],
                         "spf": [1.0, 1.0], "ptarget1d": [0.9, 0.9], "ptarget": [0.9, 0.9]})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 5, "PROBZEROVAR": 2, "PROBABILITYWEIGHTING": 1.0},
    )


def _evaluator_total(problem: ConservationProblem, sel: np.ndarray) -> float:
    return sum(evaluate_solution_chance(problem, sel, t)[1] for t in ("1d", "2d"))


def test_merged_row_order_offsets_count_rows_not_cells():
    cache = ProblemCache.from_problem(_dup_rows_mode5())
    cells = [cache.chance_cells[t] for t in ("1d", "2d")]
    merged = cache.chance_merged
    assert merged is not None
    assert merged.f_abs_e is None and merged.f_terms is None  # per-term data only
    n_feat = cache.n_feat
    n_rows0 = int(cells[0].r_indptr[-1])
    assert n_rows0 > int(cells[0].c_indptr[-1])  # duplicates: more rows than cells
    np.testing.assert_array_equal(
        merged.r_indptr[n_feat:], cells[1].r_indptr + n_rows0,
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        sel = rng.random(cache.n_pu) < 0.5
        for k, c in enumerate(cells):
            for j in range(n_feat):
                assert _row_sum(merged, k * n_feat + j, sel) == _row_sum(c, j, sel)


def test_mode5_exact_rule_duplicate_rows_tracked_equals_evaluator():
    problem = _dup_rows_mode5()
    cache = ProblemCache.from_problem(problem)
    assert cache.chance_merged is not None
    sel = np.ones(4, dtype=bool)
    state = ChanceState.from_selection(cache, sel)
    assert state.penalty_total() == pytest.approx(_evaluator_total(problem, sel), abs=1e-12)
    for idx in (1, 0, 1, 3, 2, 1, 0, 3, 2, 1):
        adding = not sel[idx]
        state.delta_penalty(cache, idx, adding)
        state.apply_flip(cache, idx, adding)
        sel[idx] = adding
        for t in ("1d", "2d"):
            shortfalls, _ = evaluate_solution_chance(problem, sel, t)
            expected = [shortfalls.get(f, 0.0) / 0.9 for f in (1, 2)]
            np.testing.assert_allclose(state.terms[t].pen, expected, rtol=1e-12, atol=1e-12)
        assert state.penalty_total() == pytest.approx(
            _evaluator_total(problem, sel), abs=1e-12,
        )


# --- L-6: boundaries ----------------------------------------------------------------------


def _pt1d_problem(ptarget1d, *, table: bool = False, **params) -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 2.0, 1.0], "status": 0,
                       "prob": [0.1, 0.4, 0.2]})
    return ConservationProblem(
        planning_units=pu,
        features=pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [3.0, 2.0],
                               "spf": [1.0, 1.0], "ptarget1d": ptarget1d}),
        pu_vs_features=pd.DataFrame({"species": [1, 1, 2, 2], "pu": [1, 2, 2, 3],
                                     "amount": [2.0, 3.0, 1.5, 2.5]}),
        probability=(
            pd.DataFrame({"pu": [1, 2, 3], "probability": [0.1, 0.4, 0.2]}) if table else None
        ),
        parameters=dict(params),
    )


@pytest.mark.parametrize(("table", "legacy"), [(False, 0), (True, 1)])
def test_ptarget1d_zero_does_not_select_mode_4(table, legacy):
    problem = _pt1d_problem([0.0, 0.0], table=table)
    assert _resolved_probmode(problem) == legacy
    assert _chance_terms(problem) == frozenset()


def test_nan_ptarget1d_is_disabled_for_evaluator_and_cache_alike():
    problem = _pt1d_problem([np.nan, 0.8], PROBMODE=4, PROBZEROVAR=2)
    disabled = _pt1d_problem([-1.0, 0.8], PROBMODE=4, PROBZEROVAR=2)
    cache = ProblemCache.from_problem(problem)
    np.testing.assert_array_equal(cache.feat_ptarget1d, [-1.0, 0.8])
    rng = np.random.default_rng(3)
    sel = rng.random(3) < 0.5
    state = ChanceState.from_selection(cache, sel)
    for _ in range(30):
        idx = int(rng.integers(0, 3))
        adding = not sel[idx]
        state.apply_flip(cache, idx, adding)
        sel[idx] = adding
        shortfalls, pen = evaluate_solution_chance(problem, sel, "1d")
        assert pen == evaluate_solution_chance(disabled, sel, "1d")[1]
        assert state.terms["1d"].pen[0] == 0.0
        assert state.terms["1d"].pen[1] == pytest.approx(shortfalls[2] / 0.8, abs=1e-12)
        assert state.penalty_total() == pytest.approx(pen, abs=1e-12)
