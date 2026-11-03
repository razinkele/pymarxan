"""``PROBZEROVAR 2``: the exact zero-variance rule (v0.48.0, B0; spec §3.4 decision).

A zero-variance feature (no selected cell carrying variance) holds a certain amount ``E``.
Under ``PROBZEROVAR 2`` ("exact") it is met outright (z = −inf, P = 1, no shortfall or
penalty) when ``E ≥ target − TARGET_REL_TOL·|target|`` (1e-7, relative: the deterministic
targets' tolerance, since a target such as a prop = 1 total is summed in another order than
``E``), missed outright (z = +inf, P = 0, shortfall ``ptarget``, penalty 1) otherwise. A
target of 0 has no tolerance. ``PROBZEROVAR 0`` (Marxan's Z = 4, the default) and ``1``
("met") are unchanged, bit for bit (``TestRulesZeroAndOneAreUnchanged``: numbers recorded
from v0.47.1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pymarxan.io.readers import _resolve_prop_targets, load_project
from pymarxan.io.writers import save_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers._tolerance import TARGET_REL_TOL
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.chance_state import ChanceState
from pymarxan.solvers.probability import _term_moments, evaluate_solution_chance
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.solvers.utils import _zero_var_rule, build_solution

SF4 = float(norm.sf(4.0))


def hand_example(**params) -> tuple[ConservationProblem, np.ndarray]:
    """``test_probzerovar.hand_example``: PUs 1-3 selected, PU 4 not.

    f1: E = 6.9 > target 5 with variance; f2: only q = 1 cells selected (zero variance,
    E = 6 ≥ target 4: met under the exact rule); f3: unselected (zero variance, E = 0 <
    target 2: short); f4: E = 3.4 < target 6 with variance.
    """
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [1.0, 2.0, 3.0, 4.0], "status": [0] * 4})
    feat = pd.DataFrame({
        "id": [1, 2, 3, 4], "name": list("abcd"), "target": [5.0, 4.0, 2.0, 6.0],
        "spf": [1.0] * 4, "ptarget": [0.95, 0.6, 0.3, 0.8],
    })
    pv = pd.DataFrame({
        "species": [1, 1, 1, 2, 2, 3, 4, 4],
        "pu": [1, 2, 3, 1, 2, 4, 2, 3],
        "amount": [4.0, 3.0, 2.0, 5.0, 1.0, 9.0, 2.0, 3.0],
        "prob": [0.8, 0.7, 0.8, 1.0, 1.0, 0.4, 0.2, 1.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 7.0, **params},
    )
    return problem, np.array([True, True, True, False])


def _random_problem(**params) -> ConservationProblem:
    """30 PUs × 4 features, about half the cells certain (q = 1); rows species-major."""
    rng = np.random.default_rng(3)
    n, f = 30, 4
    pu = pd.DataFrame({"id": np.arange(1, n + 1), "cost": rng.uniform(1, 5, n).round(2),
                       "status": [0] * n})
    feat = pd.DataFrame({"id": np.arange(1, f + 1), "name": list("abcd"),
                         "target": [20.0, 15.0, 10.0, 25.0], "spf": [5.0] * f,
                         "ptarget": [0.9, 0.7, 0.8, 0.6]})
    rows = [(j + 1, i + 1) for j in range(f) for i in range(n) if rng.random() < 0.5]
    pv = pd.DataFrame({
        "species": [r[0] for r in rows], "pu": [r[1] for r in rows],
        "amount": rng.uniform(0.5, 4, len(rows)).round(2),
        "prob": np.where(rng.random(len(rows)) < 0.6, 1.0,
                         rng.uniform(0.3, 1, len(rows)).round(2)),
    })
    pv = pv.sort_values(["species", "pu"]).reset_index(drop=True)
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 10.0, "BLM": 0.0, **params},
    )


# ------------------------------------------- PROBZEROVAR 0 / 1: unchanged since v0.47.1


class TestRulesZeroAndOneAreUnchanged:
    """Numbers recorded from v0.47.1 (``75aeab1``), compared with ``==``: adding the exact
    rule must not move a bit of the Marxan (0) or "met" (1) results — evaluator, the
    ``ChanceState`` total and deltas, and seeded simulated annealing."""

    HAND = {
        0: (22.097994138995148, [0.149849145928745, 0.5999683287581669,
                                 0.29996832875816687, 0.7994229749576093],
            [4.3714867881710475, 2.5443807454512632, 1.8000441111011602,
             -6.999261004357227]),
        1: (8.099102632459308, [0.149849145928745, 0.0, 0.0, 0.7994229749576093],
            [4.3714867881710475, -4.455342131182697, 1.8000441111011602, 0.0]),
    }
    SA_HAND = {0: 26.098733134637918, 1: 12.64376050127661}
    SA_RANDOM = {0: [41.409708007619386, 40.10863846084933],
                 1: [39.42, 41.202451793803064]}

    @pytest.mark.parametrize("zv", [0, 1])
    def test_evaluator_and_chance_state(self, zv: int) -> None:
        total, shortfalls, deltas = self.HAND[zv]
        problem, sel = hand_example(PROBZEROVAR=zv)
        sf, tot = evaluate_solution_chance(problem, sel)
        assert tot == total
        assert [sf[f] for f in (1, 2, 3, 4)] == shortfalls
        cache = ProblemCache.from_problem(problem)
        cs = ChanceState.from_selection(cache, sel)
        assert cs.penalty_total() == total
        assert [cs.delta_penalty(cache, i, not sel[i]) for i in range(4)] == deltas

    @pytest.mark.parametrize("zv", [0, 1])
    def test_seeded_annealing(self, zv: int) -> None:
        problem, _ = hand_example(PROBZEROVAR=zv)
        sols = SimulatedAnnealingSolver(num_iterations=2000, num_temp_steps=20).solve(
            problem, SolverConfig(num_solutions=2, seed=42),
        )
        assert [s.objective for s in sols] == [self.SA_HAND[zv]] * 2
        sols = SimulatedAnnealingSolver(num_iterations=5000, num_temp_steps=50).solve(
            _random_problem(PROBZEROVAR=zv), SolverConfig(num_solutions=2, seed=7),
        )
        assert [s.objective for s in sols] == self.SA_RANDOM[zv]


# ---------------------------------------------------------------- the exact rule


class TestExactRuleEvaluator:
    def test_hand_example(self) -> None:
        """f2 (certain 6 ≥ 4) is met, f3 (certain 0 < 2) is missed outright; f1 / f4 (with
        variance) score as under every rule."""
        problem, sel = hand_example(PROBZEROVAR=2)
        marxan, _ = hand_example()
        sf, total = evaluate_solution_chance(problem, sel)
        sf0, _ = evaluate_solution_chance(marxan, sel)
        assert sf[1] == sf0[1] and sf[4] == sf0[4]
        assert sf[2] == 0.0 and sf[3] == 0.3
        f1 = sf0[1] / 0.95
        f4 = sf0[4] / 0.8
        assert total == pytest.approx(7.0 * (f1 + 1.0 + f4), abs=1e-12)
        sol = build_solution(problem, sel, 0.0)
        assert sol.prob_penalty == total
        assert sol.prob_shortfalls == sf

    @pytest.mark.parametrize("qs, sel, expected", [
        # target 3, amount 2 per cell; weighting 2, ptarget 0.9 / 0.5
        ([0.5, 0.5, 0.5, 0.5], [False, False], 2.0 * 2),    # nothing selected: E = 0 short
        ([1.0, 1.0, 1.0, 1.0], [True, True], 0.0),          # E = 4 ≥ 3: met
        ([1.0, 1.0, 1.0, 1.0], [True, False], 2.0 * 2),     # E = 2 < 3: short
        ([0.0, 0.0, 0.0, 0.0], [True, True], 2.0 * 2),      # q = 0: E = 0
        ([0.0, 1.0, 1.0, 0.0], [True, True], 2.0 * 2),      # E = 2 each: short
    ])
    def test_zero_variance_cases(self, qs, sel, expected) -> None:
        pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
        feat = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [3.0, 3.0],
                             "spf": [1.0, 1.0], "ptarget": [0.9, 0.5]})
        pv = pd.DataFrame({"species": [1, 1, 2, 2], "pu": [1, 2, 1, 2],
                           "amount": [2.0] * 4, "prob": qs})
        problem = ConservationProblem(
            planning_units=pu, features=feat, pu_vs_features=pv,
            parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 2.0, "PROBZEROVAR": 2},
        )
        s = np.array(sel)
        assert evaluate_solution_chance(problem, s)[1] == expected
        cache = ProblemCache.from_problem(problem)
        assert cache.zero_var_rule == "exact"
        assert ChanceState.from_selection(cache, s).penalty_total() == expected


def _prop_one_problem(seed: int, **params) -> ConservationProblem:
    """300 PUs, 10 features each in about half of them, every cell certain (q = 1) with an
    amount in U(0.01, 10), and prop = 1 resolved as ``load_project`` does
    (``_resolve_prop_targets``: a pandas groupby sum); species-major rows for odd seeds."""
    rng = np.random.default_rng(seed)
    n_pu, n_feat = 300, 10
    r, c = np.nonzero(rng.random((n_pu, n_feat)) < 0.5)
    pv = pd.DataFrame({"species": c + 1, "pu": r + 1,
                       "amount": rng.uniform(0.01, 10, len(r)), "prob": 1.0})
    if seed % 2:
        pv = pv.sort_values(["species", "pu"]).reset_index(drop=True)
    feat = _resolve_prop_targets(pd.DataFrame({
        "id": np.arange(1, n_feat + 1), "name": [f"f{j}" for j in range(n_feat)],
        "target": 0.0, "prop": 1.0, "spf": 1.0, "ptarget": 0.9,
    }), pv)
    pu = pd.DataFrame({"id": np.arange(1, n_pu + 1), "cost": 1.0, "status": 0})
    return ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                               parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 10.0,
                                           **params})


@pytest.mark.parametrize("seed", [0, 1])
def test_prop_one_targets_of_fully_held_certain_features_are_met(seed: int) -> None:
    """A prop = 1 target is the feature's total as a pandas groupby sum; the evaluator's
    E is a sequential sum, so the two can differ by an ulp or two. Here at least one
    feature's E is below its target in float. With every PU selected and every cell
    certain, the exact rule finds every feature met (the tolerance), as the deterministic
    target check does, in the evaluator, the reported Solution and ChanceState."""
    problem = _prop_one_problem(seed, PROBZEROVAR=2)
    sel = np.ones(len(problem.planning_units), dtype=bool)
    fids = problem.features["id"].to_numpy()
    E, _, N = _term_moments(problem, sel, "2d", fids)
    assert (N == 0).all()
    assert (E < problem.features["target"].to_numpy()).any()  # the layout needs the tolerance
    shortfalls, penalty = evaluate_solution_chance(problem, sel)
    assert shortfalls == dict.fromkeys(fids.tolist(), 0.0) and penalty == 0.0
    sol = build_solution(problem, sel, 0.0)
    assert sol.prob_penalty == 0.0 and all(sol.targets_met.values())
    cache = ProblemCache.from_problem(problem)
    assert cache.zero_var_rule == "exact"
    assert ChanceState.from_selection(cache, sel).penalty_total() == 0.0


def test_a_certain_amount_short_by_more_than_the_tolerance_is_missed() -> None:
    """The tolerance is relative (1e-7 of the target): leaving out one unit leaves every
    feature it holds short by that unit's amount (≥ 0.01, far more than 1e-7 of a total of
    about 750), so each is missed outright (shortfall ptarget, penalty the weighting)."""
    problem = _prop_one_problem(0, PROBZEROVAR=2)
    pv = problem.pu_vs_features
    sel = np.ones(len(problem.planning_units), dtype=bool)
    pu = int(pv.loc[pv["species"] == 1, "pu"].iloc[0])
    sel[pu - 1] = False
    held = set(pv.loc[pv["pu"] == pu, "species"].astype(int).tolist())
    shortfalls, penalty = evaluate_solution_chance(problem, sel)
    assert shortfalls == {f: (0.9 if f in held else 0.0)
                          for f in problem.features["id"].astype(int).tolist()}
    assert penalty == 10.0 * len(held)
    cache = ProblemCache.from_problem(problem)
    assert ChanceState.from_selection(cache, sel).penalty_total() == penalty


def _one_feature(amounts: list[float], target: float, **params) -> ConservationProblem:
    n = len(amounts)
    pu = pd.DataFrame({"id": np.arange(1, n + 1), "cost": 1.0, "status": 0})
    pv = pd.DataFrame({"species": 1, "pu": np.arange(1, n + 1), "amount": amounts,
                       "prob": 1.0})
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [target], "spf": [1.0],
                         "ptarget": [0.9]})
    return ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                               parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 10.0,
                                           "PROBZEROVAR": 2, **params})


def test_a_zero_target_with_a_negative_certain_amount_is_short() -> None:
    """Target 0 has no tolerance: a negative certain amount is short (P = 0), not met;
    nothing selected (E = 0) and a positive amount are met."""
    problem = _one_feature([-2.0, 3.0], 0.0)
    cache = ProblemCache.from_problem(problem)
    for sel, expected in (([True, False], 10.0), ([False, False], 0.0),
                          ([False, True], 0.0), ([True, True], 0.0)):
        s = np.array(sel)
        sf, tot = evaluate_solution_chance(problem, s)
        assert tot == expected and sf[1] == (0.9 if expected else 0.0)
        assert ChanceState.from_selection(cache, s).penalty_total() == expected


def test_a_negative_target_uses_its_absolute_tolerance() -> None:
    """threshold = target − 1e-7·|target|: for target −4 that is −4.0000004."""
    tol = TARGET_REL_TOL * 4.0
    for amount, expected in ((-4.0, 0.0), (-4.0 - 0.5 * tol, 0.0), (-4.0 - 2 * tol, 10.0)):
        problem = _one_feature([amount], -4.0)
        assert evaluate_solution_chance(problem, np.array([True]))[1] == expected


# ------------------------------------------------------------------ parameter handling


@pytest.mark.parametrize("value", [2, 2.0, "2"])
def test_rule_from_parameter(value) -> None:
    problem, _ = hand_example(PROBZEROVAR=value)
    assert _zero_var_rule(problem) == "exact"
    assert not any("PROBZEROVAR" in e for e in problem.validate())


@pytest.mark.parametrize("value", [3, -1, 1.5, "two"])
def test_validate_reports_other_values(value) -> None:
    problem, _ = hand_example(PROBZEROVAR=value)
    errors = [e for e in problem.validate() if "PROBZEROVAR" in e]
    assert errors and "0" in errors[0] and "1" in errors[0] and "2" in errors[0]
    assert _zero_var_rule(problem) == "marxan"


def test_round_trips_through_input_dat(tmp_path) -> None:
    problem, _ = hand_example(PROBZEROVAR=2)
    save_project(problem, tmp_path)
    loaded = load_project(tmp_path)
    assert loaded.parameters["PROBZEROVAR"] == 2
    assert _zero_var_rule(loaded) == "exact"


# ------------------------------------------- ChanceState: the knife-edge window (T2)
#
# Every fixture below has its puvspr rows in ascending PU order within each feature and
# no PU naming a feature twice: then ChanceState's fresh column sum (a bincount over the
# cells in ascending PU order) and the evaluator's (``_term_moments``: puvspr rows in file
# order) add the same floats in the same order, so a recompute reproduces the evaluator's
# E and its met / short decision bit for bit, and the penalties can be compared with ``==``.


def _target_with_threshold(value: float) -> float:
    """A target whose exact-rule met threshold (``_exact_threshold``) is exactly
    ``value``: a certain amount of ``value`` is met by 0 ulps, one ulp less is missed."""
    from pymarxan.solvers.probability import _exact_threshold

    t = value / (1.0 - TARGET_REL_TOL)
    for _ in range(200):
        thr = float(_exact_threshold(np.array([t]))[0])
        if thr == value:
            return t
        t = float(np.nextafter(t, np.inf if thr < value else -np.inf))
    raise AssertionError(f"no target has the threshold {value!r}")


def _flip(cs: ChanceState, cache: ProblemCache, cur: np.ndarray, i: int) -> None:
    add = not cur[i]
    cs.delta_penalty(cache, int(i), add)
    cs.apply_flip(cache, int(i), add)
    cur[i] = add


def _certain_feature_problem(n: int = 40, seed: int = 0) -> ConservationProblem:
    """One all-certain feature whose met threshold is exactly the evaluator's E with every
    PU selected: met by 0 ulps, one ulp less is missed."""
    rng = np.random.default_rng(seed)
    amounts = rng.uniform(0.1, 3.0, n).tolist()
    problem = _one_feature(amounts, 0.0, PROBABILITYWEIGHTING=5.0)
    E, _, _ = _term_moments(problem, np.ones(n, dtype=bool), "2d", np.array([1]))
    return problem.copy_with(
        features=problem.features.assign(target=[_target_with_threshold(float(E[0]))]),
    )


@pytest.mark.parametrize("seed", range(3))
def test_the_knife_edge_matches_the_evaluator(seed: int) -> None:
    """Remove / re-add random subsets of an all-certain feature whose met threshold equals
    its total: every time all PUs are back, ChanceState and the evaluator agree it is met,
    although the running E took another summation path (it can end an ulp below the
    threshold: only the knife-edge recompute makes them agree). Checked after every flip."""
    problem = _certain_feature_problem(seed=seed)
    n = len(problem.planning_units)
    assert evaluate_solution_chance(problem, np.ones(n, dtype=bool))[1] == 0.0
    cache = ProblemCache.from_problem(problem)
    assert cache.zero_var_rule == "exact"
    cur = np.zeros(n, dtype=bool)
    cs = ChanceState.from_selection(cache, cur)
    rng = np.random.default_rng(42 + seed)
    for idx in rng.permutation(n):
        _flip(cs, cache, cur, int(idx))
        assert cs.penalty_total() == evaluate_solution_chance(problem, cur)[1]
    for _ in range(60):
        out = rng.choice(n, int(rng.integers(1, n)), replace=False)
        for i in list(out) + list(rng.permutation(out)):
            _flip(cs, cache, cur, int(i))
            assert cs.penalty_total() == evaluate_solution_chance(problem, cur)[1]
        assert cur.all() and cs.penalty_total() == 0.0
    assert cs.n_recompute > 0


def test_the_window_covers_the_evaluators_own_rounding() -> None:
    """One all-certain feature: a 1.0 cell at PU 1, then 400 cells of 2^-53. The evaluator
    sums from PU 1 and loses every tiny cell (E = 1.0; the exact sum is 1 + 200 ulps); the
    state, built by adding the tiny cells first, holds 1 + 200 ulps. With the met
    threshold at 1 + 100 ulps, between the two, the evaluator says short while the
    running E says met: a window of BE alone (or BE + 64·eps·|E|) misses that; the
    evaluator-error term 3·max(64, n_j)·eps·S_j covers it, and the recompute reproduces
    the evaluator's E and decision."""
    n = 400
    ulp = 2.0 ** -52
    problem = _one_feature([1.0] + [2.0 ** -53] * n, _target_with_threshold(1.0 + 100 * ulp))
    cache = ProblemCache.from_problem(problem)
    cur = np.zeros(n + 1, dtype=bool)
    cs = ChanceState.from_selection(cache, cur)
    for i in list(range(1, n + 1)) + [0]:  # the tiny cells first, the 1.0 last
        _flip(cs, cache, cur, i)
        assert cs.penalty_total() == evaluate_solution_chance(problem, cur)[1]
    assert evaluate_solution_chance(problem, cur)[1] == 10.0  # short, by the evaluator's E
    assert cs.terms["2d"].E[0] == 1.0  # recomputed in the evaluator's order


def _grid_problem(case: str, seed: int = 0) -> tuple[ConservationProblem, np.ndarray]:
    """All-certain features with amounts and targets on one grid, and a start selection
    that holds each target exactly (in exact arithmetic): ``"integer"`` (2000 PUs, 5
    features, amounts 1-3), ``"two_decimals"`` (amounts 0.01-3.00), ``"integer_long"``
    (one feature in each of 130 000 PUs)."""
    rng = np.random.default_rng(seed)
    if case == "integer_long":
        n_pu, n_feat = 130_000, 1
        r, c = np.arange(n_pu), np.zeros(n_pu, dtype=int)
    else:
        n_pu, n_feat = 2000, 5
        r, c = np.nonzero(rng.random((n_pu, n_feat)) < 0.6)
    k = rng.integers(1, 301, len(r)) if case == "two_decimals" else rng.integers(1, 4, len(r))
    scale = 100 if case == "two_decimals" else 1
    pv = pd.DataFrame({"species": c + 1, "pu": r + 1, "amount": k / scale, "prob": 1.0})
    sel = rng.random(n_pu) < 0.5
    held_k = np.bincount(c[sel[r]], weights=k[sel[r]], minlength=n_feat)  # exact integers
    feat = pd.DataFrame({"id": np.arange(1, n_feat + 1),
                         "name": [f"f{j}" for j in range(n_feat)],
                         "target": held_k / scale, "spf": 1.0, "ptarget": 0.9})
    pu = pd.DataFrame({"id": np.arange(1, n_pu + 1), "cost": 1.0, "status": 0})
    problem = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                                  parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 5.0,
                                              "PROBZEROVAR": 2})
    return problem, sel


@pytest.mark.parametrize("case", [
    "integer", "two_decimals", pytest.param("integer_long", marks=pytest.mark.slow),
])
def test_grid_amounts_at_grid_targets_do_not_recompute(case: str) -> None:
    """Amounts and targets on one grid (counts, presence/absence, 2-decimal data): each
    time a chain of removals is re-added, every certain amount lands on its target again.
    The window is centred on the met threshold, 1e-7·|target| below the target, and is far
    narrower than that (under 1/500 of it on the 130 000-cell column), so the landings
    are not recomputed; ChanceState's penalty equals the evaluator's after every flip."""
    problem, sel = _grid_problem(case)
    cache = ProblemCache.from_problem(problem)
    assert cache.zero_var_rule == "exact"
    target = cache.feat_targets
    thr = target - TARGET_REL_TOL * np.abs(target)
    cs = ChanceState.from_selection(cache, sel)
    t = cs.terms["2d"]
    cur = sel.copy()
    rng = np.random.default_rng(7)
    for _ in range(30):
        out = rng.choice(np.flatnonzero(cur), int(rng.integers(1, 21)), replace=False)
        for i in list(out) + list(rng.permutation(out)):
            _flip(cs, cache, cur, int(i))
            if case != "integer_long":
                assert cs.penalty_total() == evaluate_solution_chance(problem, cur)[1]
        assert np.array_equal(cur, sel) and cs.penalty_total() == 0.0
        assert np.all(np.abs(t.E - target) <= 1e-12 * target)
        window = cs._knife_window(t, np.arange(cache.n_feat), t.BE)
        assert np.all(window < 1e-2 * np.abs(t.E - thr))  # well clear of the tolerance
    assert cs.n_recompute == 0
    assert evaluate_solution_chance(problem, cur)[1] == 0.0


def test_a_zero_target_without_expected_amount_does_not_recompute() -> None:
    """A target of 0 has no tolerance (threshold 0). A zero-variance feature whose cells
    all have e = 0 (prob 0) has E = BE = 0 and S_j = 0, so its window is 0 and the strict
    comparison never recomputes it, although every flip touches it (it is in every PU);
    the evaluator's sum of those zeros is 0 too (met)."""
    rng = np.random.default_rng(3)
    n_pu = 300
    r, c = np.nonzero(rng.random((n_pu, 2)) < 0.6)
    pv = pd.DataFrame({"species": c + 1, "pu": r + 1,
                       "amount": rng.uniform(0.5, 3.0, len(r)), "prob": 1.0})
    zero = pd.DataFrame({"species": 3, "pu": np.arange(1, n_pu + 1), "amount": 1.0,
                         "prob": 0.0})
    pv = pd.concat([pv, zero]).sort_values(["pu", "species"]).reset_index(drop=True)
    totals = pv.groupby("species")["amount"].sum().to_numpy()
    feat = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "z"],
                         "target": [0.5 * totals[0], 0.5 * totals[1], 0.0],
                         "spf": 1.0, "ptarget": 0.9})
    pu = pd.DataFrame({"id": np.arange(1, n_pu + 1), "cost": 1.0, "status": 0})
    problem = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                                  parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 10.0,
                                              "PROBZEROVAR": 2})
    assert problem.validate() == []
    cache = ProblemCache.from_problem(problem)
    cur = rng.random(n_pu) < 0.5
    cs = ChanceState.from_selection(cache, cur)
    t = cs.terms["2d"]
    seen = set()
    for _ in range(300):
        _flip(cs, cache, cur, int(rng.integers(n_pu)))
        assert t.E[2] == 0.0 and t.BE[2] == 0.0 and t.pen[2] == 0.0
        assert cs.penalty_total() == evaluate_solution_chance(problem, cur)[1]
        seen.add(cs.penalty_total())
    assert seen == {0.0, 10.0, 20.0}  # the walk meets and misses features a and b
    assert cs.n_recompute == 0


def test_a_zero_target_recomputes_the_residue_left_by_its_last_cell() -> None:
    """Removing every PU of an all-certain target-0 feature can leave the running E a
    rounding residue around 0 (here negative: short of the threshold 0, the full
    weighting) while the evaluator's sum over no cells is 0, met. That flip is inside the
    window and recomputed, so ChanceState agrees after every flip."""
    rng = np.random.default_rng(0)
    n = 30
    amounts = rng.uniform(0.1, 3.0, n)
    problem = _one_feature(amounts.tolist(), 0.0)
    order = rng.permutation(n)
    cache = ProblemCache.from_problem(problem)
    cur = np.ones(n, dtype=bool)
    cs = ChanceState.from_selection(cache, cur)
    t = cs.terms["2d"]
    for k, i in enumerate(order):
        if k == n - 1:  # the layout needs the recompute: a negative residue
            assert t.E[0] - amounts[i] < 0.0
        _flip(cs, cache, cur, int(i))
        assert cs.penalty_total() == evaluate_solution_chance(problem, cur)[1]
    assert not cur.any() and t.E[0] == 0.0 and cs.penalty_total() == 0.0
    assert cs.n_recompute >= 1


_CANCELLING_GROUPS = {"plus": [0], "ones": list(range(1, 11)), "minus": [11]}


@pytest.mark.parametrize("order", [
    "plus-ones-minus", "plus-minus-ones", "ones-plus-minus", "ones-minus-plus",
    "minus-plus-ones", "minus-ones-plus",
])
@pytest.mark.parametrize("probzerovar", [0, 2])
def test_a_negative_amount_keeps_the_knife_edge_decision_the_evaluators(
    probzerovar: int, order: str,
) -> None:
    """One all-certain feature: +1e16 at PU 1, ten 1.0 at PUs 2–11, −1e16 at PU 12;
    target 2. The exact sum over all twelve is 10 (met) but the evaluator's sum in PU order
    is 0 (each 1.0 is lost beside 1e16): short. The evaluator's rounding is up to
    (m − 1)·u·Σ|e|, far above any multiple of eps·|E| when large terms cancel, so the
    window's evaluator term uses S_j = Σ|e| over the feature's whole column. In every order
    of the three groups ChanceState's penalty is the evaluator's after every flip, under
    PROBZEROVAR 2 and under 0 (whose zero-variance penalty does not depend on E)."""
    problem = _one_feature([1e16] + [1.0] * 10 + [-1e16], 2.0, PROBZEROVAR=probzerovar)
    cache = ProblemCache.from_problem(problem)
    assert cache.zero_var_rule == ("exact" if probzerovar else "marxan")
    n = len(problem.planning_units)
    cur = np.zeros(n, dtype=bool)
    cs = ChanceState.from_selection(cache, cur)
    for group in order.split("-"):
        for i in _CANCELLING_GROUPS[group]:
            _flip(cs, cache, cur, i)
            assert cs.penalty_total() == evaluate_solution_chance(problem, cur)[1]
    assert cur.all()
    if probzerovar:  # the evaluator's sum, 0, is short of the threshold: the full weighting
        assert cs.penalty_total() == 10.0


def _mixed_problem(seed: int) -> ConservationProblem:
    """60 PUs × 6 features: features 1-3 all certain (targets near half their totals, one
    exactly at a knife edge), features 4-6 with some uncertain cells (q ∈ {0, 1} for
    most, so they go in and out of zero variance); PU-major rows."""
    rng = np.random.default_rng(seed)
    n_pu, n_feat = 60, 6
    r, c = np.nonzero(rng.random((n_pu, n_feat)) < 0.5)
    q = np.where(c < 3, 1.0, rng.choice([0.0, 1.0, 1.0, 0.6], len(r)))
    pv = pd.DataFrame({"species": c + 1, "pu": r + 1,
                       "amount": rng.uniform(0.1, 5.0, len(r)), "prob": q})
    totals = pv.assign(e=pv.amount * pv.prob).groupby("species")["e"].sum().to_numpy()
    target = 0.5 * totals
    feat = pd.DataFrame({"id": np.arange(1, n_feat + 1),
                         "name": [f"f{j}" for j in range(n_feat)], "target": target,
                         "spf": 1.0, "ptarget": [0.9, 0.5, 0.7, 0.9, 0.5, 0.8]})
    pu = pd.DataFrame({"id": np.arange(1, n_pu + 1), "cost": rng.uniform(1, 3, n_pu),
                       "status": 0})
    problem = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                                  parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 4.0,
                                              "PROBZEROVAR": 2, "BLM": 0.0})
    half = rng.random(n_pu) < 0.5
    E, _, _ = _term_moments(problem, half, "2d", np.array([1]))
    target[0] = _target_with_threshold(float(E[0]))  # f1 on the knife edge at ``half``
    return problem.copy_with(features=feat.assign(target=target))


@pytest.mark.parametrize("seed", range(4))
def test_tracked_penalty_and_deltas_equal_the_evaluator_after_every_flip(seed: int) -> None:
    """A random walk with previews that are committed (70 %) or rejected: the tracked
    total equals a fresh evaluation after every commit — exactly for the zero-variance
    features' met / short decisions, to 1e-12 for the features with variance (running
    sums) — and every delta equals the difference of fresh evaluations."""
    problem = _mixed_problem(seed)
    cache = ProblemCache.from_problem(problem)
    n = len(problem.planning_units)
    rng = np.random.default_rng(100 + seed)
    cur = rng.random(n) < 0.5
    cs = ChanceState.from_selection(cache, cur)
    before = evaluate_solution_chance(problem, cur)[1]
    for _ in range(400):
        i = int(rng.integers(n))
        add = not cur[i]
        after_sel = cur.copy()
        after_sel[i] = add
        after = evaluate_solution_chance(problem, after_sel)[1]
        d = cs.delta_penalty(cache, i, add)
        assert d == pytest.approx(after - before, abs=1e-12)
        if rng.random() < 0.7:
            cs.apply_flip(cache, i, add)
            cur = after_sel
            before = after
        assert cs.penalty_total() == pytest.approx(before, abs=1e-12)
        # the zero-variance decisions are exact: compare the per-feature penalties
        sf, _ = evaluate_solution_chance(problem, cur)
        zero = cs.terms["2d"].N == 0
        pt = cache.feat_ptarget
        ev_pen = np.array([sf[int(f)] for f in problem.features["id"]]) / pt
        assert np.array_equal(cs.terms["2d"].pen[zero], ev_pen[zero])


@pytest.mark.parametrize("seed", range(2))
def test_annealing_and_iterative_improvement_report_the_evaluators_objective(
    seed: int,
) -> None:
    """Under PROBZEROVAR 2, simulated annealing and iterative improvement (driven by
    ChanceState's deltas) end on solutions whose reported objective (``build_solution``,
    the evaluator) equals a fresh rebuild; each solver's own tracked objective agrees."""
    from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver

    problem = _mixed_problem(seed)
    sols = SimulatedAnnealingSolver(num_iterations=4000, num_temp_steps=40).solve(
        problem, SolverConfig(num_solutions=2, seed=seed),
    )
    for s in sols:
        fresh = build_solution(problem, s.selected, 0.0)
        assert s.objective == pytest.approx(fresh.objective, abs=1e-9)
        assert s.prob_penalty == fresh.prob_penalty
        improved = IterativeImprovementSolver(itimptype=1).improve(problem, s)
        again = build_solution(problem, improved.selected, 0.0)
        assert improved.objective == pytest.approx(again.objective, abs=1e-9)
        assert improved.objective <= s.objective + 1e-9


def test_a_dominant_cell_leaving_keeps_the_fresh_sums_error_in_the_window() -> None:
    """All amounts non-negative, yet the window's evaluator term must scale with S_j (the
    column's Σ|e|), not with the current |E|: the fresh sum over 5e6 + 2·10^4 × 0.1 is off
    by about 7e-6, and removing the 5e6 cell (an exact subtraction) leaves that error on
    E ≈ 2000, where 3·F_j·eps·|E| is only about 3e-8 and BE about 3e-9. The met threshold
    sits at the evaluator's own sum without PU 1, so only the S_j-scaled term makes that
    removal recompute; the evaluator says met, and so must ChanceState."""
    n = 20_000
    base = _one_feature([5e6] + [0.1] * n, 0.0)
    full = np.ones(n + 1, dtype=bool)
    without = full.copy()
    without[0] = False
    E_without = float(_term_moments(base, without, "2d", np.array([1]))[0][0])
    problem = base.copy_with(
        features=base.features.assign(target=[_target_with_threshold(E_without)]),
    )
    cache = ProblemCache.from_problem(problem)
    cs = ChanceState.from_selection(cache, full)
    stale = cs.terms["2d"].E[0] - 5e6
    assert abs(stale - E_without) > 1e-6  # the fresh sum's error, carried by the removal
    assert abs(stale - E_without) > 3 * max(64, n + 1) * np.finfo(float).eps * abs(stale)
    cur = full.copy()
    _flip(cs, cache, cur, 0)
    assert evaluate_solution_chance(problem, cur)[1] == 0.0
    assert cs.penalty_total() == 0.0 and cs.n_recompute == 1


# ------------------------------------------------------- documented examples (T3)


def _safe_vs_risky(safe_amount: float, **params) -> ConservationProblem:
    """The manual's example: target 10, ptarget 0.9, weighting 10, two PUs costing 1. PU 1
    risky (amount 20, q = 0.9: E = 18, V = 36, Z = −4/3, P ≈ 0.909 ≥ 0.9), PU 2 safe
    (``safe_amount`` at q = 1)."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [10.0], "spf": [1.0],
                         "ptarget": [0.9]})
    pv = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [20.0, safe_amount],
                       "prob": [0.9, 1.0]})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 10.0, "BLM": 0.0, **params},
    )


def test_the_manuals_safe_vs_risky_example() -> None:
    """USER_MANUAL "PROBMODE 3 and zero variance": holding the feature safely scores
    10.9996 under PROBZEROVAR 0, 1.0 under 1 and 2; the risky unit scores 1.0 under all
    three; too little in certain cells (8 < 10) pays nothing under 1 and the full
    weighting under 2."""
    risky, safe = np.array([True, False]), np.array([False, True])
    for zv, safe_score in ((0, 1 + 10 * (0.9 - SF4) / 0.9), (1, 1.0), (2, 1.0)):
        p = _safe_vs_risky(12.0, PROBZEROVAR=zv)
        assert build_solution(p, safe, 0.0).objective == pytest.approx(safe_score, abs=1e-12)
        assert build_solution(p, risky, 0.0).objective == pytest.approx(1.0, abs=1e-12)
    assert round(1 + 10 * (0.9 - SF4) / 0.9, 4) == 10.9996
    short = np.array([False, True])
    assert build_solution(_safe_vs_risky(8.0, PROBZEROVAR=1), short, 0.0).prob_penalty == 0.0
    assert build_solution(_safe_vs_risky(8.0, PROBZEROVAR=2), short, 0.0).prob_penalty == 10.0


def test_iterative_improvement_prefers_the_cheaper_safe_unit_under_the_exact_rule() -> None:
    """Under PROBZEROVAR 2 the safe unit's certain 12 ≥ 10 is met: with the safe unit
    cheaper, iterative improvement started on the risky unit swaps to it (under Marxan's
    rule it stays, the safe unit paying ≈ the full weighting)."""
    from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver

    def problem(**params):
        p = _safe_vs_risky(12.0, **params)
        return p.copy_with(planning_units=p.planning_units.assign(cost=[1.0, 0.5]))

    risky = np.array([True, False])
    for zv, expected in ((0, [True, False]), (2, [False, True])):
        p = problem(PROBZEROVAR=zv)
        sol = IterativeImprovementSolver(itimptype=3).improve(p, build_solution(p, risky, 0.0))
        assert sol.selected.tolist() == expected
