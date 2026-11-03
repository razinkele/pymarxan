"""``ChanceState`` with the Probability 1D term (PROBMODE 4) and both terms (PROBMODE 5).

The 2D contract (``test_chance_state.py``) carried to the 1D term: the cells hold
Marxan 3's row-by-row ``a·(1 − p)`` / ``a·a·p·(1 − p)`` with the PU's loss probability,
``delta_penalty`` equals the difference of the from-scratch penalty (1e-9) over ≥ 5000
mixed accepted / rejected flips, the commit equals the preview, and SA / II report what
they tracked. Mode 5 scores the 1D term against ``ptarget1d`` and the 2D term against
``ptarget`` (different on purpose), each from its own cells.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.chance_state import ChanceState, term_ptarget
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.probability import evaluate_solution_chance, score_features
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.solvers.utils import build_solution

TOL = 1e-9


def _problem(pv: pd.DataFrame, pu_prob: np.ndarray, rng, *, ids=None, mode: int = 4,
             target_frac=(0.2, 0.6), weight: float = 2.0) -> ConservationProblem:
    n_pu = len(pu_prob)
    ids = np.arange(1, n_pu + 1) if ids is None else ids
    n_feat = int(pv.species.max())
    pu = pd.DataFrame({"id": ids, "cost": rng.uniform(1, 5, n_pu), "status": 0,
                       "prob": pu_prob})
    p_of = dict(zip(ids, pu_prob))
    tot = (pv.amount * (1 - pv.pu.map(p_of))).groupby(pv.species).sum().reindex(
        range(1, n_feat + 1), fill_value=0.0,
    )
    feat = pd.DataFrame({
        "id": range(1, n_feat + 1), "name": [f"f{j}" for j in range(n_feat)],
        "target": tot.values * rng.uniform(*target_frac, n_feat), "spf": 1.0,
        "ptarget1d": np.r_[rng.uniform(0.3, 0.95, n_feat - 1), -1.0],
        # a different 2D target (used in mode 5 only)
        "ptarget": np.r_[-1.0, rng.uniform(0.3, 0.95, n_feat - 1)],
    })
    if mode == 5:
        pv = pv.assign(prob=rng.uniform(0.0, 1.0, len(pv)))
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": mode, "PROBABILITYWEIGHTING": weight},
    )


def _rows(rng, n_pu: int, n_feat: int, density: float, ids=None) -> pd.DataFrame:
    ids = np.arange(1, n_pu + 1) if ids is None else ids
    rows = [(f, int(i)) for f in range(1, n_feat + 1) for i in ids if rng.random() < density]
    return pd.DataFrame(rows, columns=["species", "pu"])


def mixed(seed: int, mode: int = 4, n_pu: int = 120, n_feat: int = 6) -> ConservationProblem:
    """Shuffled ids; p forced to 0 or 1 on ~40 % of PUs."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(1, n_pu + 1) * 3)
    pv = _rows(rng, n_pu, n_feat, 0.5, ids)
    pv["amount"] = rng.uniform(0.1, 3, len(pv))
    p = rng.uniform(0, 1, n_pu)
    r = rng.random(n_pu)
    p[r < 0.25] = 0.0
    p[(r >= 0.25) & (r < 0.4)] = 1.0
    return _problem(pv, p, rng, ids=ids, mode=mode)


def few_risky(seed: int, mode: int = 4, n_pu: int = 40, n_feat: int = 3) -> ConservationProblem:
    """Three risky PUs: features' variance empties and refills."""
    rng = np.random.default_rng(seed)
    pv = _rows(rng, n_pu, n_feat, 0.5)
    pv["amount"] = rng.uniform(0.5, 4, len(pv))
    p = np.zeros(n_pu)
    p[rng.choice(n_pu, 3, replace=False)] = rng.uniform(0.2, 0.8, 3)
    return _problem(pv, p, rng, mode=mode)


def wide_amounts(seed: int, mode: int = 4, n_pu: int = 60, n_feat: int = 4):
    """Amounts spanning 1e0 .. 1e6, p in (0.01, 0.99)."""
    rng = np.random.default_rng(seed)
    pv = _rows(rng, n_pu, n_feat, 0.6)
    pv["amount"] = 10 ** rng.uniform(0, 6, len(pv))
    return _problem(pv, rng.uniform(0.01, 0.99, n_pu), rng, mode=mode)


def tiny_beside_big(seed: int, mode: int = 4, n_pu: int = 60, n_feat: int = 4):
    """p = 1e-17 PUs sharing features with a = 1e4, p = 0.5 PUs (the recompute case)."""
    rng = np.random.default_rng(seed)
    pv = _rows(rng, n_pu, n_feat, 0.6)
    pv["amount"] = rng.uniform(0.5, 3, len(pv))
    big = rng.random(n_pu) < 0.15
    p = np.where(big, 0.5, 1e-17)
    pv.loc[pv.pu.isin(np.flatnonzero(big) + 1), "amount"] = 1e4
    return _problem(pv, p, rng, mode=mode)


def sparse_risk(seed: int, mode: int = 4, n_pu: int = 400, n_feat: int = 5):
    """1 % of PUs at p = 0.05, the rest p = 0 (the usual 1D layout)."""
    rng = np.random.default_rng(seed)
    pv = _rows(rng, n_pu, n_feat, 0.6)
    pv["amount"] = rng.uniform(1, 10, len(pv))
    p = np.zeros(n_pu)
    p[rng.choice(n_pu, max(1, n_pu // 100), replace=False)] = 0.05
    return _problem(pv, p, rng, mode=mode, target_frac=(0.3, 0.5))


def duplicated(seed: int, mode: int = 4, n_pu: int = 60, n_feat: int = 4):
    """Rows drawn with replacement: cells named by 2-3 puvspr rows (V adds a² per row)."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(1, n_pu + 1) * 5)
    k = n_pu * 3
    pv = pd.DataFrame({
        "species": rng.integers(1, n_feat + 1, k), "pu": rng.choice(ids, k),
        "amount": rng.uniform(0.1, 3, k),
    })
    assert pv.duplicated(["species", "pu"]).sum() > k // 10
    p = rng.uniform(0, 1, n_pu)
    p[rng.random(n_pu) < 0.2] = 0.0
    return _problem(pv, p, rng, ids=ids, mode=mode)


LAYOUTS = {
    "mixed": mixed, "few_risky": few_risky, "wide_amounts": wide_amounts,
    "tiny_beside_big": tiny_beside_big, "sparse_risk": sparse_risk, "duplicated": duplicated,
}


def _cache(problem: ConservationProblem, rule: str) -> ProblemCache:
    return dataclasses.replace(ProblemCache.from_problem(problem), zero_var_rule=rule)


def _dense(cache: ProblemCache):
    out = {}
    for name, c in cache.chance_cells.items():
        E = np.zeros((cache.n_pu, cache.n_feat))
        V = np.zeros_like(E)
        rows = np.repeat(np.arange(cache.n_pu), np.diff(c.indptr))
        E[rows, c.indices] = c.e
        V[rows, c.indices] = c.v
        out[name] = (E, V, (V > 0).astype(float))
    return out


def _reference(cache: ProblemCache, dense, sel: np.ndarray) -> float:
    """From-scratch penalty of every active term, independent of ChanceState."""
    x = sel.astype(float)
    total = 0.0
    for name, (Ec, Vc, Pc) in dense.items():
        total += float(np.sum(score_features(
            x @ Ec, x @ Vc, np.rint(x @ Pc).astype(np.int64), cache.feat_targets,
            term_ptarget(cache, name), cache.zero_var_rule,
        )[3]))
    return cache.prob_weight * total


class TestCells:
    def test_1d_cells_are_marxan3_row_sums(self):
        pu = pd.DataFrame({"id": [5, 2], "cost": [1.0, 1.0], "status": 0, "prob": [0.3, 0.1]})
        pv = pd.DataFrame({"species": [1, 1, 1], "pu": [5, 5, 2], "amount": [2.0, 3.0, 4.0]})
        feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [1.0], "spf": 1.0,
                             "ptarget1d": [0.5]})
        cache = ProblemCache.from_problem(ConservationProblem(
            planning_units=pu, features=feat, pu_vs_features=pv, parameters={"PROBMODE": 4},
        ))
        c = cache.chance_cells["1d"]
        assert cache.chance_terms == {"1d"}
        # PU 5 (index 0): rows a = 2, 3 at p = 0.3 → V = (4 + 9)·0.21, not 25·0.21
        np.testing.assert_allclose(c.e, [5 * 0.7, 4 * 0.9])
        np.testing.assert_allclose(c.v, [13 * 0.3 * 0.7, 16 * 0.1 * 0.9])
        np.testing.assert_array_equal(c.f_terms, [3])

    def test_mode_5_builds_both_terms(self):
        cache = ProblemCache.from_problem(mixed(1, mode=5))
        assert cache.chance_terms == {"1d", "2d"}
        assert set(cache.chance_cells) == {"1d", "2d"}
        np.testing.assert_array_equal(cache.chance_cells["1d"].indptr,
                                      cache.chance_cells["2d"].indptr)


@pytest.mark.parametrize("rule", ["marxan", "met", "exact"])
@pytest.mark.parametrize("mode", [4, 5])
def test_from_selection_equals_the_evaluator(rule, mode, monkeypatch):
    import pymarxan.solvers.cache as cache_mod
    import pymarxan.solvers.utils as utils

    monkeypatch.setattr(utils, "_zero_var_rule", lambda problem: rule)
    monkeypatch.setattr(cache_mod, "_zero_var_rule", lambda problem: rule)
    rng = np.random.default_rng(5)
    for build in LAYOUTS.values():
        problem = build(11, mode=mode)
        cache = ProblemCache.from_problem(problem)
        for _ in range(4):
            sel = rng.random(cache.n_pu) < rng.uniform(0, 1)
            ref = sum(evaluate_solution_chance(problem, sel, t)[1] for t in cache.chance_terms)
            cs = ChanceState.from_selection(cache, sel)
            assert cs.penalty_total() == pytest.approx(ref, rel=1e-12, abs=1e-12)


def _run(problem, rule: str, seed: int, flips: int, accept: float):
    cache = _cache(problem, rule)
    dense = _dense(cache)
    rng = np.random.default_rng(seed)
    sel = rng.random(cache.n_pu) < 0.5
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
        if rng.random() < accept:
            cs.apply_flip(cache, idx, add)
            sel, full = s2, f2
            tracked += d
    np.testing.assert_array_equal(cs.selected, sel)
    return worst, abs(tracked - full)


@pytest.mark.parametrize("rule", ["marxan", "met"])
@pytest.mark.parametrize("mode", [4, 5])
@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_delta_equals_full_difference(layout, mode, rule):
    flips = 5000 if layout != "sparse_risk" else 2500
    worst, drift = _run(LAYOUTS[layout](7, mode=mode), rule, seed=17, flips=flips, accept=0.6)
    assert worst <= TOL
    assert drift <= 1e-8


def test_delta_equals_full_difference_temperature_sampling():
    """Rejected-only previews (the SA temperature-sampling path) never mutate."""
    worst, drift = _run(mixed(3, mode=5), "marxan", seed=4, flips=2000, accept=0.0)
    assert worst <= TOL and drift == 0.0


@pytest.mark.parametrize("mode", [4, 5])
@pytest.mark.parametrize("layout", ["duplicated", "tiny_beside_big", "mixed"])
def test_commit_equals_preview_and_fresh_state(layout, mode):
    cache = _cache(LAYOUTS[layout](9, mode=mode), "marxan")
    rng = np.random.default_rng(2)
    cs = ChanceState.from_selection(cache, rng.random(cache.n_pu) < 0.5)
    for _ in range(200):
        idx = int(rng.integers(cache.n_pu))
        add = not cs.selected[idx]
        cs.delta_penalty(cache, idx, add)
        assert cs._preview is not None
        previews = cs._preview[3]
        cs.apply_flip(cache, idx, add)
        fresh = ChanceState.from_selection(cache, cs.selected)
        for (name, t), (cols, EV2, _, N2, pen2) in zip(cs.terms.items(), previews):
            np.testing.assert_array_equal(t.EV[:, cols], EV2)
            np.testing.assert_array_equal(t.N[cols], N2)
            np.testing.assert_array_equal(t.pen[cols], pen2)
            f = fresh.terms[name]
            np.testing.assert_array_equal(t.N, f.N)
            np.testing.assert_allclose(t.E, f.E, rtol=1e-12, atol=1e-300)
            np.testing.assert_allclose(t.V, f.V, rtol=1e-12, atol=1e-300)
            np.testing.assert_allclose(t.pen, f.pen, rtol=0, atol=1e-12)


def test_mode_5_terms_use_their_own_ptarget():
    problem = mixed(4, mode=5)
    cache = ProblemCache.from_problem(problem)
    cs = ChanceState.from_selection(cache, np.ones(cache.n_pu, dtype=bool))
    np.testing.assert_array_equal(cs.terms["1d"].ptarget, problem.features["ptarget1d"])
    np.testing.assert_array_equal(cs.terms["2d"].ptarget, problem.features["ptarget"])


# ------------------------------------------------------------ SA and II, end to end


def _small(seed: int, mode: int) -> ConservationProblem:
    problem = mixed(seed, mode=mode, n_pu=14, n_feat=3)
    rng = np.random.default_rng(seed)
    ids = problem.planning_units.id.to_numpy()
    b = [(int(a), int(c), 1.0) for a, c in zip(ids[:-1], ids[1:])]
    feats = problem.features.assign(spf=3.0)
    return problem.copy_with(
        features=feats,
        boundary=pd.DataFrame(b, columns=["id1", "id2", "boundary"]),
        planning_units=problem.planning_units.assign(cost=rng.uniform(1, 5, 14)),
        parameters={**problem.parameters, "PROBABILITYWEIGHTING": 20.0, "BLM": 0.3,
                    "NUMITNS": 20_000, "NUMTEMP": 100},
    )


def _obj(problem, sel) -> float:
    return build_solution(problem, sel, float(problem.parameters.get("BLM", 0.0))).objective


@pytest.mark.parametrize("mode", [4, 5])
@pytest.mark.parametrize("seed", [1, 2])
def test_sa_reported_objective_equals_tracked(mode, seed):
    problem = _small(seed, mode)
    (sol,) = SimulatedAnnealingSolver().solve(problem, SolverConfig(num_solutions=1, seed=seed))
    assert sol.prob1d_penalty is not None
    assert (sol.prob_penalty is not None) == (mode == 5)
    assert sol.objective == pytest.approx(
        sol.metadata["history"]["best_objective"][-1], abs=1e-9,
    )


@pytest.mark.parametrize("mode", [4, 5])
@pytest.mark.parametrize("itimptype", [1, 2, 3])
def test_ii_stops_at_an_evaluator_local_optimum(mode, itimptype):
    problem = _small(3, mode)
    (sol,) = IterativeImprovementSolver(itimptype=itimptype).solve(
        problem, SolverConfig(num_solutions=1, seed=3),
    )
    assert sol.objective == pytest.approx(_obj(problem, sol.selected), abs=1e-12)
    if itimptype in (1, 2):
        for i in range(len(sol.selected)):
            if sol.selected[i] or itimptype == 2:
                s2 = sol.selected.copy()
                s2[i] = not s2[i]
                assert _obj(problem, s2) >= sol.objective - 1e-9


def test_ii_removal_keeps_a_pu_the_1d_term_needs():
    """4 PUs × 1 feature, each alone meets the target (amount 10, target 10), but the 1D
    chance target needs all four (p = 0.4, ptarget1d 0.9); PU 5 is a costly spare."""
    pu = pd.DataFrame({"id": [1, 2, 3, 4, 5], "cost": [1.0, 1.0, 1.0, 1.0, 100.0],
                       "status": 0, "prob": [0.4, 0.4, 0.4, 0.4, 0.0]})
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [10.0], "spf": [100.0],
                         "ptarget1d": [0.9]})
    pv = pd.DataFrame({"species": [1] * 4, "pu": [1, 2, 3, 4], "amount": [10.0] * 4})
    problem = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 4, "PROBABILITYWEIGHTING": 50.0, "BLM": 0.0},
    )
    start = build_solution(problem, np.ones(5, dtype=bool), 0.0)
    sol = IterativeImprovementSolver(itimptype=1).improve(problem, start)
    assert sol.selected.tolist() == [True, True, True, True, False]
