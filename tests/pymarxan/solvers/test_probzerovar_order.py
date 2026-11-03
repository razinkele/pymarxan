"""PROBZEROVAR 2 (exact rule) — B0 review fixes (v0.48.0).

M-1: ``ChanceState`` summed a zero-variance feature's cells in planning-unit index order
while the evaluator (``probability._term_moments``) sums puvspr rows in file order, so the
met / short decision disagreed with the evaluator whenever PU ids are not ascending,
a feature's rows are not in PU order, or a (pu, species) pair repeats — at
``from_selection`` already. The decision is now taken from a sum in the evaluator's order.

M-2: a target of 0 gets a noise-scale tolerance ``−n_j·eps·S_j`` (``S_j`` = Σ|e| over the
feature's puvspr rows, ``n_j`` their count), so an exact-zero certain amount is met in
every row order while a genuinely negative one is short.

M-3: ``compute_zscore_penalty(..., zero_var_rule="exact")`` raises for a zero-variance id
missing from ``achieved_mean`` / ``targets`` and applies the tolerance.

M-4: the knife-edge window's margins are pinned one by one.
"""
from __future__ import annotations

import itertools
import types

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers._tolerance import TARGET_REL_TOL
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.chance_state import EPS, ChanceState
from pymarxan.solvers.probability import (
    _exact_threshold,
    compute_zscore_penalty,
    evaluate_solution_chance,
    score_feature,
)


def _target_at(value: float) -> float:
    """A nonzero target whose exact-rule threshold is exactly ``value`` (knife edge)."""
    t = value / (1.0 - TARGET_REL_TOL)
    for _ in range(400):
        thr = float(_exact_threshold(np.array([t]))[0])
        if thr == value:
            return t
        t = float(np.nextafter(t, np.inf if thr < value else -np.inf))
    raise AssertionError("no target puts the threshold on value")


def _term(problem: ConservationProblem) -> str:
    return "1d" if problem.parameters["PROBMODE"] == 4 else "2d"


def _assert_matches_evaluator(problem, cache, cs, sel) -> None:
    """Per feature: ChanceState's penalty × ptarget == the evaluator's shortfall; exactly
    for a zero-variance feature (the exact rule's 0 / ptarget step)."""
    term = _term(problem)
    sf, _ = evaluate_solution_chance(problem, sel, term)
    t = cs.terms[term]
    for j, fid in enumerate(problem.features["id"]):
        pt = float(t.ptarget[j])
        if pt <= 0:
            continue
        got = float(t.pen[j]) * pt
        if t.N[j] == 0:
            assert got == sf[int(fid)], (fid, got, sf[int(fid)], float(t.E[j]))
        else:
            assert got == pytest.approx(sf[int(fid)], rel=1e-9, abs=1e-12)


def _descending_ids(seed: int, mode: int) -> ConservationProblem:
    """The review's pu_order probe: PU ids descending, puvspr ascending by PU, all cells
    certain, target on the evaluator's E of the full selection."""
    rng = np.random.default_rng(seed)
    n = 40
    pu = pd.DataFrame({"id": np.arange(n, 0, -1), "cost": 1.0, "status": 0})
    pv = pd.DataFrame({"species": 1, "pu": np.arange(1, n + 1),
                       "amount": rng.uniform(0.1, 3.0, n)})
    if mode == 4:
        pu["prob"] = 0.0
    else:
        pv["prob"] = 1.0
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [0.0], "spf": [1.0],
                         "ptarget": [0.9], "ptarget1d": [0.9]})
    p = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                            parameters={"PROBMODE": mode, "PROBABILITYWEIGHTING": 10.0,
                                        "PROBZEROVAR": 2})
    sel = np.ones(n, dtype=bool)
    from pymarxan.solvers.probability import _term_moments
    E = _term_moments(p, sel, _term(p), np.array([1]))[0]
    return p.copy_with(features=feat.assign(target=[_target_at(float(E[0]))]))


@pytest.mark.parametrize("mode", [3, 4])
@pytest.mark.parametrize("seed", range(20))
def test_from_selection_decides_as_the_evaluator_with_descending_pu_ids(seed, mode):
    problem = _descending_ids(seed, mode)
    cache = ProblemCache.from_problem(problem)
    sel = np.ones(cache.n_pu, dtype=bool)
    _assert_matches_evaluator(problem, cache, ChanceState.from_selection(cache, sel), sel)


def _fuzz_problem(seed: int, order: str, dup: bool) -> ConservationProblem:
    """The review's fuzz probe ("certain" kind): certain amounts, rows shuffled or
    duplicated, targets on the evaluator's E of a reference selection."""
    rng = np.random.default_rng(seed)
    n, f = 60, 3
    rows = [(j + 1, i + 1) for j in range(f) for i in range(n) if rng.random() < 0.7]
    pv = pd.DataFrame(rows, columns=["species", "pu"])
    pv["amount"] = rng.uniform(0.1, 3, len(pv))
    if dup:
        extra = pv.sample(frac=0.2, random_state=seed).copy()
        extra["amount"] = extra["amount"] * rng.uniform(-1, 1, len(extra))
        pv = pd.concat([pv, extra])
    if order == "shuffle":
        pv = pv.sample(frac=1.0, random_state=seed)
    pv = pv.reset_index(drop=True)
    pv["prob"] = 1.0
    pu = pd.DataFrame({"id": np.arange(1, n + 1), "cost": 1.0, "status": 0})
    feat = pd.DataFrame({"id": np.arange(1, f + 1), "name": ["a", "b", "c"], "target": 1.0,
                         "spf": 1.0, "ptarget": 0.8})
    p = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                            parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 3.0,
                                        "PROBZEROVAR": 2})
    from pymarxan.solvers.probability import _term_moments
    ref = rng.random(n) < 0.5
    E = _term_moments(p, ref, "2d", feat["id"].to_numpy())[0]
    return p.copy_with(features=feat.assign(target=[_target_at(float(e)) for e in E])), ref


@pytest.mark.parametrize("order, dup", [("shuffle", False), ("asc", True)])
@pytest.mark.parametrize("seed", range(3))
def test_flips_decide_as_the_evaluator_with_shuffled_or_duplicate_rows(seed, order, dup):
    problem, ref = _fuzz_problem(seed, order, dup)
    cache = ProblemCache.from_problem(problem)
    rng = np.random.default_rng(seed)
    cur = rng.random(cache.n_pu) < 0.5
    cs = ChanceState.from_selection(cache, cur)
    _assert_matches_evaluator(problem, cache, cs, cur)
    for _ in range(300):
        diff = np.flatnonzero(cur != ref)
        if rng.random() < 0.5 and len(diff):
            i = int(rng.choice(diff))  # walk back to the knife-edge selection
        else:
            i = int(rng.integers(cache.n_pu))
        add = not cur[i]
        cs.delta_penalty(cache, i, add)
        cs.apply_flip(cache, i, add)
        cur[i] = add
        _assert_matches_evaluator(problem, cache, cs, cur)


# ------------------------------------------------------------------- M-2: target 0


def _zero_target(amounts, rows, pu_ids, mode: int = 3) -> ConservationProblem:
    pu = pd.DataFrame({"id": pu_ids, "cost": 1.0, "status": 0})
    pv = pd.DataFrame({"species": 1, "pu": [1, 2, 3][: len(amounts)],
                       "amount": amounts}).iloc[list(rows)].reset_index(drop=True)
    if mode == 4:
        pu["prob"] = 0.0
    else:
        pv["prob"] = 1.0
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [0.0], "spf": [1.0],
                         "ptarget": [0.9], "ptarget1d": [0.9]})
    return ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                               parameters={"PROBMODE": mode, "PROBABILITYWEIGHTING": 1.0,
                                           "PROBZEROVAR": 2})


def _zero_target_penalties(problem):
    cache = ProblemCache.from_problem(problem)
    n = cache.n_pu
    sel = np.ones(n, dtype=bool)
    ev = evaluate_solution_chance(problem, sel, _term(problem))[1]
    init = ChanceState.from_selection(cache, sel).penalty_total()
    cs = ChanceState.from_selection(cache, np.zeros(n, dtype=bool))
    for i in range(n):
        cs.delta_penalty(cache, i, True)
        cs.apply_flip(cache, i, True)
    return ev, init, cs.penalty_total()


@pytest.mark.parametrize("mode", [3, 4])
@pytest.mark.parametrize("amounts", [[0.82, 0.01, -0.83], [0.3, -0.1, -0.2]])
def test_target_zero_exact_zero_sum_is_met_in_every_order(amounts, mode):
    for rows in itertools.permutations(range(3)):
        for pu_ids in ([1, 2, 3], [3, 2, 1], [2, 3, 1]):
            assert _zero_target_penalties(_zero_target(amounts, rows, pu_ids, mode)) == (
                0.0, 0.0, 0.0,
            ), (rows, pu_ids)


@pytest.mark.parametrize("mode", [3, 4])
def test_target_zero_negative_certain_amount_is_short(mode):
    for rows in itertools.permutations(range(2)):
        assert _zero_target_penalties(_zero_target([0.5, -0.6], rows, [1, 2], mode)) == (
            1.0, 1.0, 1.0,
        )


def test_exact_threshold_zero_target_tolerance():
    thr = _exact_threshold(np.array([0.0, 5.0, -2.0]), np.array([1e-15, 1e-15, 1e-15]))
    assert thr[0] == -1e-15
    assert thr[1] == 5.0 - TARGET_REL_TOL * 5.0  # nonzero targets unchanged
    assert thr[2] == -2.0 - TARGET_REL_TOL * 2.0
    assert _exact_threshold(np.array([0.0]))[0] == 0.0  # no noise scale given: none


def test_score_feature_zero_target_tolerance():
    assert score_feature(-1e-17, 0.0, 0, 0.0, 0.9, "exact", zero_tol=1e-16).penalty == 0.0
    assert score_feature(-1e-17, 0.0, 0, 0.0, 0.9, "exact").penalty == 1.0


# ---------------------------------------------------------------- M-3: dict API


def test_compute_zscore_penalty_exact_raises_on_a_missing_id():
    with pytest.raises(ValueError, match="7"):
        compute_zscore_penalty(
            {7: 4.0}, {7: 0.9}, zero_variance=frozenset({7}), zero_var_rule="exact",
            achieved_mean={}, targets={7: 1.0},
        )
    with pytest.raises(ValueError, match="7"):
        compute_zscore_penalty(
            {7: 4.0}, {7: 0.9}, zero_variance=frozenset({7}), zero_var_rule="exact",
            achieved_mean={7: 1.0}, targets={},
        )


def test_compute_zscore_penalty_exact_applies_the_tolerance():
    # E = T·(1 − 5e-8): within the 1e-7 relative tolerance → met (a raw E ≥ T is short).
    args = dict(zero_variance=frozenset({1}), zero_var_rule="exact")
    assert compute_zscore_penalty(
        {1: 4.0}, {1: 0.9}, achieved_mean={1: 10.0 * (1 - 5e-8)}, targets={1: 10.0}, **args,
    ) == 0.0
    assert compute_zscore_penalty(
        {1: 4.0}, {1: 0.9}, achieved_mean={1: -1e-17}, targets={1: 0.0},
        target_zero_tol={1: 1e-16}, **args,
    ) == 0.0


# ------------------------------------------------------------- M-4: window margins


def _fake_term(knife: float):
    return types.SimpleNamespace(knife=np.array([knife]))


def _hit(d: float, BE: float, knife: float) -> bool:
    cols = np.array([0])
    return bool(ChanceState._knife_hit(
        _fake_term(knife), cols, np.array([d]), np.array([0.0]), np.array([BE]),
    )[0])


def test_window_includes_the_8eps_margin():
    # |E − thr| = (1 + 4·eps)·BE: inside (1 + 8·eps)·BE, outside a window of BE alone.
    assert _hit(1.0 + 4 * EPS, 1.0, 0.0)
    assert not _hit(1.0 + 16 * EPS, 1.0, 0.0)


def test_window_includes_the_bound_term():
    assert _hit(0.5, 1.0, 1e-30)


def test_window_includes_the_evaluator_term():
    assert _hit(0.5e-10, 0.0, 1e-10)
    assert not _hit(2e-10, 0.0, 1e-10)


def test_window_is_strict():
    assert not _hit(1e-10, 0.0, 1e-10)


def test_knife_scale_per_feature():
    """knife_j = 3·max(64, n_j)·eps·S_j: a 3-row feature uses 64, a 100-row one 100."""
    rng = np.random.default_rng(0)
    pv = pd.DataFrame({
        "species": [1] * 3 + [2] * 100, "pu": list(range(1, 4)) + list(range(1, 101)),
        "amount": rng.uniform(0.5, 2.0, 103), "prob": 1.0,
    })
    pu = pd.DataFrame({"id": range(1, 101), "cost": 1.0, "status": 0})
    feat = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0],
                         "spf": 1.0, "ptarget": 0.9})
    p = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                            parameters={"PROBMODE": 3, "PROBZEROVAR": 2})
    cache = ProblemCache.from_problem(p)
    t = ChanceState.from_selection(cache, np.zeros(100, dtype=bool)).terms["2d"]
    S = np.array([pv.amount[:3].abs().sum(), pv.amount[3:].abs().sum()])
    np.testing.assert_allclose(t.knife, 3.0 * np.array([64, 100]) * EPS * S, rtol=1e-15)
