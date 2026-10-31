"""``ChanceState`` sidecar and the cache's per-cell chance arrays (spec §3.5).

The contract SA relies on: ``delta_penalty`` equals the difference of the from-scratch
chance penalty across a flip (tolerance 1e-9), never mutates, and ``apply_flip`` commits
exactly the previewed state. The adversarial layouts are the ones the spec review used
to break running sums: p forced to 0 / 1, features whose variance empties and refills,
amounts spread over 1e6 with the dominant-variance cell removed, p = 1e-17 cells beside
large-variance cells, and the sparse-risk layout (1 % risky PUs).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.chance_state import ChanceState
from pymarxan.solvers.probability import evaluate_solution_chance, score_features

TOL = 1e-9


# ----------------------------------------------------------------------------- fixtures


def _problem(pv: pd.DataFrame, n_pu: int, n_feat: int, rng, *, weight: float = 2.0,
             pu_ids=None, target_frac=(0.2, 0.6)) -> ConservationProblem:
    pu_ids = np.arange(1, n_pu + 1) if pu_ids is None else pu_ids
    pu = pd.DataFrame({"id": pu_ids, "cost": rng.uniform(1, 5, n_pu), "status": 0})
    tot = (pv.amount * pv.prob).groupby(pv.species).sum().reindex(
        range(1, n_feat + 1), fill_value=0.0,
    )
    feat = pd.DataFrame({
        "id": range(1, n_feat + 1), "name": [f"f{j}" for j in range(n_feat)],
        "target": tot.values * rng.uniform(*target_frac, n_feat), "spf": 1.0,
        # >= 2 features with different ptarget, one disabled
        "ptarget": np.r_[rng.uniform(0.3, 0.95, n_feat - 1), -1.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": weight},
    )


def mixed(seed: int, n_pu: int = 120, n_feat: int = 6) -> ConservationProblem:
    """Shuffled PU ids; q forced to 0 or 1 on ~40 % of cells."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(1, n_pu + 1) * 3)
    k = n_pu * 4
    pv = pd.DataFrame({
        "species": rng.integers(1, n_feat + 1, k), "pu": rng.choice(ids, k),
        "amount": rng.uniform(0.1, 3, k),
    }).drop_duplicates(["species", "pu"]).reset_index(drop=True)
    q = rng.uniform(0, 1, len(pv))
    r = rng.random(len(pv))
    q[r < 0.25] = 0.0
    q[(r >= 0.25) & (r < 0.4)] = 1.0
    pv["prob"] = q
    return _problem(pv, n_pu, n_feat, rng, pu_ids=ids)


def few_risky(seed: int, n_pu: int = 40, n_feat: int = 3) -> ConservationProblem:
    """Each feature has only 2-3 uncertain cells: its variance empties and refills."""
    rng = np.random.default_rng(seed)
    rows = [(f, i) for f in range(1, n_feat + 1) for i in range(1, n_pu + 1)
            if rng.random() < 0.5]
    pv = pd.DataFrame(rows, columns=["species", "pu"])
    pv["amount"] = rng.uniform(0.5, 4, len(pv))
    pv["prob"] = 1.0
    for f in range(1, n_feat + 1):
        idx = pv.index[pv.species == f][:3]
        pv.loc[idx, "prob"] = rng.uniform(0.2, 0.8, len(idx))
    return _problem(pv, n_pu, n_feat, rng)


def wide_amounts(seed: int, n_pu: int = 60, n_feat: int = 4) -> ConservationProblem:
    """Amounts spanning 1e0 .. 1e6, q in (0.01, 0.99)."""
    rng = np.random.default_rng(seed)
    rows = [(f, i) for f in range(1, n_feat + 1) for i in range(1, n_pu + 1)
            if rng.random() < 0.6]
    pv = pd.DataFrame(rows, columns=["species", "pu"])
    pv["amount"] = 10 ** rng.uniform(0, 6, len(pv))
    pv["prob"] = rng.uniform(0.01, 0.99, len(pv))
    return _problem(pv, n_pu, n_feat, rng)


def tiny_beside_big(seed: int, n_pu: int = 60, n_feat: int = 4) -> ConservationProblem:
    """q = 1e-17 cells sharing features with a = 1e4, q = 0.5 cells.

    Once the large cells leave, the running V is float residue of order 1e-8 while the
    true V is ~1e-17: without the error-bounded recompute the delta is off by up to the
    weight·ptarget scale (measured 4.0) or NaN.
    """
    rng = np.random.default_rng(seed)
    rows = [(f, i) for f in range(1, n_feat + 1) for i in range(1, n_pu + 1)
            if rng.random() < 0.6]
    pv = pd.DataFrame(rows, columns=["species", "pu"])
    pv["amount"] = rng.uniform(0.5, 3, len(pv))
    q = np.full(len(pv), 1e-17)
    big = rng.random(len(pv)) < 0.15
    pv.loc[big, "amount"] = 1e4
    q[big] = 0.5
    pv["prob"] = q
    return _problem(pv, n_pu, n_feat, rng)


def sparse_risk(seed: int, n_pu: int = 400, n_feat: int = 5) -> ConservationProblem:
    """1 % of PUs risky (presence q = 0.95, i.e. loss p = 0.05); every other cell q = 1."""
    rng = np.random.default_rng(seed)
    rows = [(f, i) for f in range(1, n_feat + 1) for i in range(1, n_pu + 1)
            if rng.random() < 0.6]
    pv = pd.DataFrame(rows, columns=["species", "pu"])
    pv["amount"] = rng.uniform(1, 10, len(pv))
    risky = rng.choice(np.arange(1, n_pu + 1), max(1, n_pu // 100), replace=False)
    pv["prob"] = np.where(pv.pu.isin(risky), 0.95, 1.0)
    return _problem(pv, n_pu, n_feat, rng, target_frac=(0.3, 0.5))


LAYOUTS = {
    "mixed": mixed, "few_risky": few_risky, "wide_amounts": wide_amounts,
    "tiny_beside_big": tiny_beside_big, "sparse_risk": sparse_risk,
}


def _cache(problem: ConservationProblem, rule: str) -> ProblemCache:
    return dataclasses.replace(ProblemCache.from_problem(problem), zero_var_rule=rule)


def _dense_cells(cache: ProblemCache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = cache.chance_cells["2d"]
    E = np.zeros((cache.n_pu, cache.n_feat))
    V = np.zeros_like(E)
    rows = np.repeat(np.arange(cache.n_pu), np.diff(c.indptr))
    E[rows, c.indices] = c.e
    V[rows, c.indices] = c.v
    return E, V, (V > 0).astype(float)


def _reference(cache: ProblemCache, dense, sel: np.ndarray) -> float:
    """From-scratch chance penalty, independent of ChanceState's arithmetic."""
    Ec, Vc, Pc = dense
    x = sel.astype(float)
    pen = score_features(
        x @ Ec, x @ Vc, np.rint(x @ Pc).astype(np.int64), cache.feat_targets,
        cache.feat_ptarget, cache.zero_var_rule,
    )[3]
    return cache.prob_weight * float(np.sum(pen))


# ----------------------------------------------------------------------- per-cell arrays


class TestChanceCells:
    def test_cells_are_aligned_with_held_csr(self) -> None:
        cache = ProblemCache.from_problem(mixed(0))
        c = cache.chance_cells["2d"]
        np.testing.assert_array_equal(c.indptr, cache.held_csr.indptr)
        np.testing.assert_array_equal(c.indices, cache.held_csr.indices)
        assert c.e.shape == c.v.shape == c.pos.shape == cache.held_csr.data.shape

    def test_cell_values_match_the_puvspr_rows(self) -> None:
        problem = mixed(1)
        cache = ProblemCache.from_problem(problem)
        E, V, P = _dense_cells(cache)
        for row in problem.pu_vs_features.itertuples():
            i = cache.pu_id_to_idx[int(row.pu)]
            j = cache.feat_id_to_col[int(row.species)]
            assert E[i, j] == pytest.approx(row.amount * row.prob, rel=1e-15)
            assert V[i, j] == pytest.approx(
                row.amount ** 2 * row.prob * (1.0 - row.prob), rel=1e-15,
            )
            assert P[i, j] == float(V[i, j] > 0)

    def test_csc_is_the_same_cells_feature_major(self) -> None:
        cache = ProblemCache.from_problem(mixed(2))
        c = cache.chance_cells["2d"]
        E, V, _ = _dense_cells(cache)
        for j in range(cache.n_feat):
            a, b = c.c_indptr[j], c.c_indptr[j + 1]
            rows = c.c_rows[a:b]
            assert np.all(np.diff(rows) > 0)
            np.testing.assert_array_equal(c.c_e[a:b], E[rows, j])
            np.testing.assert_array_equal(c.c_v[a:b], V[rows, j])
        assert c.c_indptr[-1] == len(c.e)

    def test_duplicate_rows_sum_amounts_and_last_prob_wins(self) -> None:
        pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
        feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0],
                             "ptarget": [0.5]})
        pv = pd.DataFrame({"species": [1, 1, 1, 9], "pu": [1, 1, 2, 1],
                           "amount": [2.0, 3.0, 4.0, 1.0], "prob": [0.1, 0.4, 0.5, 0.9]})
        problem = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                                      parameters={"PROBMODE": 3})
        cache = ProblemCache.from_problem(problem)
        c = cache.chance_cells["2d"]
        assert c.e.tolist() == [5.0 * 0.4, 4.0 * 0.5]
        assert c.v.tolist() == [25.0 * 0.4 * 0.6, 16.0 * 0.5 * 0.5]

    def test_built_only_for_active_terms(self) -> None:
        problem = mixed(3)
        assert set(ProblemCache.from_problem(problem).chance_cells) == {"2d"}
        off = problem.copy_with(parameters={"PROBMODE": 0})
        assert ProblemCache.from_problem(off).chance_cells == {}
        no_data = problem.copy_with(
            pu_vs_features=problem.pu_vs_features.drop(columns=["prob"]),
        )
        assert ProblemCache.from_problem(no_data).chance_cells == {}


# ------------------------------------------------------------------------- ChanceState


@pytest.mark.parametrize("rule", ["marxan", "met"])
def test_from_selection_equals_the_evaluator(rule: str, monkeypatch) -> None:
    import pymarxan.solvers.cache as cache_mod
    import pymarxan.solvers.utils as utils

    monkeypatch.setattr(utils, "_zero_var_rule", lambda problem: rule)
    monkeypatch.setattr(cache_mod, "_zero_var_rule", lambda problem: rule)
    rng = np.random.default_rng(5)
    for build in LAYOUTS.values():
        problem = build(11)
        cache = ProblemCache.from_problem(problem)
        assert cache.zero_var_rule == rule
        for _ in range(5):
            sel = rng.random(cache.n_pu) < rng.uniform(0, 1)
            cs = ChanceState.from_selection(cache, sel)
            assert cs.penalty_total() == pytest.approx(
                evaluate_solution_chance(problem, sel)[1], rel=1e-12, abs=1e-12,
            )


def _run(problem, rule: str, seed: int, flips: int, accept: float,
         sel=None) -> tuple[ChanceState, np.ndarray, float, float]:
    cache = _cache(problem, rule)
    dense = _dense_cells(cache)
    rng = np.random.default_rng(seed)
    sel = rng.random(cache.n_pu) < 0.5 if sel is None else sel.copy()
    cs = ChanceState.from_selection(cache, sel)
    full = _reference(cache, dense, sel)
    tracked = cs.penalty_total()
    assert tracked == pytest.approx(full, abs=TOL)
    worst = 0.0
    for _ in range(flips):
        idx = int(rng.integers(cache.n_pu))
        add = not sel[idx]
        d = cs.delta_penalty(cache, idx, add)
        assert np.isfinite(d)
        s2 = sel.copy()
        s2[idx] = add
        f2 = _reference(cache, dense, s2)
        worst = max(worst, abs(d - (f2 - full)))
        if rng.random() < accept:  # mixes accepted and rejected moves
            cs.apply_flip(cache, idx, add)
            sel, full = s2, f2
            tracked += d
    np.testing.assert_array_equal(cs.selected, sel)
    return cs, sel, worst, abs(tracked - full)


@pytest.mark.parametrize("rule", ["marxan", "met"])
@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_delta_equals_full_difference(layout: str, rule: str) -> None:
    flips = 5000 if layout != "sparse_risk" else 2500
    cs, _, worst, drift = _run(LAYOUTS[layout](7), rule, seed=17, flips=flips, accept=0.6)
    assert worst <= TOL
    assert drift <= 1e-8


def test_removing_the_dominant_variance_cells_largest_first() -> None:
    """A largest-first removal chain at amounts spanning 1e6 (round 3's drift case)."""
    n = 40
    rng = np.random.default_rng(4)
    a = 10 ** (6 * np.arange(n) / (n - 1)) * rng.uniform(0.9, 1.1, n)
    pu = pd.DataFrame({"id": np.arange(1, n + 1), "cost": 1.0, "status": 0})
    pv = pd.DataFrame({"species": 1, "pu": np.arange(1, n + 1), "amount": a, "prob": 0.5})
    keep = 3
    e_k = (a[:keep] * 0.5).sum()
    sd = np.sqrt((a[:keep] ** 2 * 0.25).sum())
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [e_k + 0.3 * sd], "spf": 1.0,
                         "ptarget": [0.9]})
    problem = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                                  parameters={"PROBMODE": 3})
    cache = ProblemCache.from_problem(problem)
    dense = _dense_cells(cache)
    sel = np.ones(n, dtype=bool)
    cs = ChanceState.from_selection(cache, sel)
    full = _reference(cache, dense, sel)
    for idx in range(n - 1, keep - 1, -1):
        d = cs.delta_penalty(cache, idx, False)
        s2 = sel.copy()
        s2[idx] = False
        f2 = _reference(cache, dense, s2)
        assert abs(d - (f2 - full)) <= TOL
        cs.apply_flip(cache, idx, False)
        sel, full = s2, f2
    assert cs.penalty_total() == pytest.approx(full, abs=TOL)


@pytest.mark.parametrize("rule", ["marxan", "met"])
def test_delta_never_mutates(rule: str) -> None:
    problem = tiny_beside_big(3)
    cache = _cache(problem, rule)
    rng = np.random.default_rng(1)
    cs = ChanceState.from_selection(cache, rng.random(cache.n_pu) < 0.5)
    t = cs.terms["2d"]
    snap = [x.copy() for x in (t.E, t.V, t.N, t.BE, t.BV, t.pen, cs.selected)]
    for _ in range(500):
        idx = int(rng.integers(cache.n_pu))
        cs.delta_penalty(cache, idx, not cs.selected[idx])
    for before, after in zip(snap, (t.E, t.V, t.N, t.BE, t.BV, t.pen, cs.selected)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("rule", ["marxan", "met"])
def test_commit_equals_preview_and_fresh_state(rule: str) -> None:
    problem = mixed(9)
    cache = _cache(problem, rule)
    rng = np.random.default_rng(2)
    cs = ChanceState.from_selection(cache, rng.random(cache.n_pu) < 0.5)
    t = cs.terms["2d"]
    for _ in range(300):
        idx = int(rng.integers(cache.n_pu))
        add = not cs.selected[idx]
        cs.delta_penalty(cache, idx, add)
        assert cs._preview is not None
        cols, EV2, _, N2, pen2 = cs._preview[3][0]
        E2, V2 = EV2
        cs.apply_flip(cache, idx, add)
        assert cs._preview is None
        # equals the previewed scratch state exactly
        np.testing.assert_array_equal(t.E[cols], E2)
        np.testing.assert_array_equal(t.V[cols], V2)
        np.testing.assert_array_equal(t.N[cols], N2)
        np.testing.assert_array_equal(t.pen[cols], pen2)
        # equals a fresh from_selection to 1e-12 relative
        fresh = ChanceState.from_selection(cache, cs.selected).terms["2d"]
        np.testing.assert_array_equal(t.N, fresh.N)
        np.testing.assert_allclose(t.E, fresh.E, rtol=1e-12, atol=1e-300)
        np.testing.assert_allclose(t.V, fresh.V, rtol=1e-12, atol=1e-300)
        np.testing.assert_allclose(t.pen, fresh.pen, rtol=0, atol=1e-12)


def test_stale_preview_is_never_committed() -> None:
    """delta(i), apply_flip(j), apply_flip(i) commits i against the post-j state."""
    problem = mixed(4)
    cache = ProblemCache.from_problem(problem)
    dense = _dense_cells(cache)
    sel = np.zeros(cache.n_pu, dtype=bool)
    sel[::2] = True
    cs = ChanceState.from_selection(cache, sel)
    i, j = 1, 3
    cs.delta_penalty(cache, i, True)
    cs.apply_flip(cache, j, True)
    cs.apply_flip(cache, i, True)
    sel[[i, j]] = True
    assert cs.penalty_total() == pytest.approx(_reference(cache, dense, sel), abs=TOL)
    np.testing.assert_allclose(
        cs.terms["2d"].E, ChanceState.from_selection(cache, sel).terms["2d"].E, rtol=1e-12,
    )


def test_a_pu_without_cells_has_zero_delta() -> None:
    problem = mixed(5)
    pu = problem.planning_units
    extra = pd.DataFrame({"id": [10_000], "cost": [1.0], "status": [0]})
    problem = problem.copy_with(planning_units=pd.concat([pu, extra], ignore_index=True))
    cache = ProblemCache.from_problem(problem)
    cs = ChanceState.from_selection(cache, np.zeros(cache.n_pu, dtype=bool))
    before = cs.penalty_total()
    assert cs.delta_penalty(cache, cache.n_pu - 1, True) == 0.0
    cs.apply_flip(cache, cache.n_pu - 1, True)
    assert cs.penalty_total() == before
    assert cs.selected[-1]


def test_sparse_risk_recompute_floor() -> None:
    """The 64·eps·|E| floor: mostly-certain features do not recompute on every touch."""
    cs, *_ = _run(sparse_risk(0, n_pu=2000, n_feat=5), "marxan", seed=3, flips=3000,
                  accept=0.6)
    assert cs.n_recompute < 300  # without the floor: ~one per touched feature per flip


def test_tiny_probability_has_variance() -> None:
    """q = 1e-17 gives a positive variance term: nonzero > 0, not zero variance."""
    pu = pd.DataFrame({"id": [1], "cost": [1.0], "status": [0]})
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0],
                         "ptarget": [0.5]})
    pv = pd.DataFrame({"species": [1], "pu": [1], "amount": [5.0], "prob": [1e-17]})
    problem = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                                  parameters={"PROBMODE": 3})
    cache = ProblemCache.from_problem(problem)
    cs = ChanceState.from_selection(cache, np.array([True]))
    assert cs.terms["2d"].N[0] == 1
    assert cs.terms["2d"].V[0] > 0


def test_no_active_term_is_inert() -> None:
    problem = mixed(6).copy_with(parameters={"PROBMODE": 0})
    cache = ProblemCache.from_problem(problem)
    cs = ChanceState.from_selection(cache, np.ones(cache.n_pu, dtype=bool))
    assert cs.penalty_total() == 0.0
    assert cs.delta_penalty(cache, 0, False) == 0.0
    cs.apply_flip(cache, 0, False)
    assert not cs.selected[0]


def test_variance_snaps_to_zero_when_nonzero_reaches_zero() -> None:
    """Removing a feature's last uncertain cell leaves V exactly 0 and E the exact sum."""
    problem = few_risky(2)
    cache = ProblemCache.from_problem(problem)
    c = cache.chance_cells["2d"]
    sel = np.ones(cache.n_pu, dtype=bool)
    cs = ChanceState.from_selection(cache, sel)
    t = cs.terms["2d"]
    risky_rows = np.unique(np.repeat(np.arange(cache.n_pu), np.diff(c.indptr))[c.pos > 0])
    for i in risky_rows:
        cs.delta_penalty(cache, int(i), False)
        cs.apply_flip(cache, int(i), False)
        sel[i] = False
    assert (t.N == 0).all()
    assert (t.V == 0.0).all()
    fresh = ChanceState.from_selection(cache, sel).terms["2d"]
    np.testing.assert_allclose(t.E, fresh.E, rtol=1e-15)


@pytest.mark.parametrize("rule", ["marxan", "met"])
def test_hot_path_penalty_is_the_core_arithmetic(rule: str) -> None:
    """``ChanceState``'s lean scorer equals ``score_features``' penalty bit for bit."""
    from pymarxan.solvers.probability import _chance_penalty, _ptarget_den

    rng = np.random.default_rng(8)
    n = 500
    N = rng.integers(0, 3, n)
    V = np.where(N > 0, rng.uniform(1e-300, 9, n), 0.0)
    E = rng.uniform(0, 20, n)
    t = rng.uniform(0, 20, n)
    pt = rng.choice([-1.0, 0.0, 1e-6, 0.3, 0.95, 1.0], n)
    ref = score_features(E, V, N, t, pt, rule)[3]
    got = _chance_penalty(E, V, N == 0, t, pt, _ptarget_den(pt), rule == "met")
    np.testing.assert_array_equal(got, ref)
